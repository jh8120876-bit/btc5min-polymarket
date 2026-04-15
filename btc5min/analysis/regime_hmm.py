"""
regime_hmm.py — Runtime HMM Regime Detector

Loads the pre-trained Gaussian HMM from `models/hmm_model.pkl` once at import
time and exposes a thread-safe `decode_current_regime()` that:

    1. Takes a *rolling window* of recent 5-min bars (default: last 200 bars).
    2. Builds the same 4-feature vector used during training
       (log_return, atr_pct, cvd_dir_vol, abs_log_return).
    3. Scales it with the trained StandardScaler.
    4. Runs Viterbi (model.decode) to obtain the most likely state sequence.
    5. Uses `model.predict_proba` on the *last observation* to get the
       posterior probability → "88% confidence in Latent State 2".
    6. Returns a lightweight dict safe to cache in engine state.

Design goals
------------
- ZERO hard dependency: if `hmmlearn` / `joblib` / `sklearn` / `numpy` or the
  pickle itself are missing, the module degrades to `available=False` and
  every public call returns `None`. The engine must never crash because of
  HMM absence.
- Must not stall the engine's `_main_loop` zero-drift tick. Decoding runs in
  ~1ms for 200 bars, but callers should still invoke it from the async
  prediction thread or a dedicated worker, NOT from inside `_tick()`.
- Result is cached with a short TTL (5s) so repeated calls from different
  callers (DeepSeek prompt, LSTM judge, RAG rotation) share the same output.

Public API
----------
    regime = RegimeDetector()          # singleton via `get_regime_detector()`
    regime.available                   # bool
    regime.refresh(candles_5m) -> dict # decode + update internal cache + returns

Cached result schema
--------------------
    {
        "label":         "TREND_BULL",
        "state_idx":     2,
        "confidence":    0.88,           # posterior probability of the picked state
        "state_probs":   [0.04, 0.08, 0.88],
        "transition_to": {"TREND_BULL": 0.91, "RANGE_CHOP": 0.06, ...},
        "features":      {"log_return": ..., "atr_pct": ..., ...},
        "decoded_at":    1712780400.12,
        "trained_at":    1711200000.0,
    }
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Optional

from ..config import log

# ── Optional imports (safe-fail) ─────────────────────────────
try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False
    log.warning("[HMM] numpy not available — RegimeDetector disabled")

try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False
    log.warning("[HMM] joblib not available — RegimeDetector disabled")

try:
    # Import is a presence probe only — the actual model instance is
    # reconstituted from joblib inside RegimeDetector._try_load.
    from hmmlearn.hmm import GaussianHMM as _GaussianHMMProbe  # noqa: F401
    _HMM_OK = True
except ImportError:
    _HMM_OK = False
    log.warning("[HMM] hmmlearn not available — RegimeDetector disabled")


# ── Paths ────────────────────────────────────────────────────
_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "hmm_model.pkl"

# Keep the rolling decode window short: Viterbi is O(N·K²) — cheap.
_DECODE_WINDOW_BARS = 200
# How long a cached result is considered fresh.
_CACHE_TTL_SEC = 5.0


# ── Feature engineering (MUST match train_hmm_regime.py) ────
def _atr_series(candles: list[dict], period: int = 14) -> list[float]:
    if len(candles) < 2:
        return [0.0] * len(candles)
    trs: list[float] = [0.0]
    for i in range(1, len(candles)):
        c, pc = candles[i], candles[i - 1]
        tr = max(
            c["high"] - c["low"],
            abs(c["high"] - pc["close"]),
            abs(c["low"] - pc["close"]),
        )
        trs.append(tr)
    atr = [0.0] * len(trs)
    if len(trs) >= period:
        seed = sum(trs[1:period + 1]) / period
        atr[period] = seed
        for i in range(period + 1, len(trs)):
            atr[i] = (atr[i - 1] * (period - 1) + trs[i]) / period
        for i in range(period):
            atr[i] = seed
    return atr


def _rolling_std_returns(candles: list[dict], window: int = 12) -> list[float]:
    rets: list[float] = [0.0]
    for i in range(1, len(candles)):
        pc, c = candles[i - 1]["close"], candles[i]["close"]
        rets.append(math.log(c / pc) if pc > 0 else 0.0)
    out = [0.0] * len(rets)
    for i in range(window, len(rets)):
        slc = rets[i - window + 1:i + 1]
        m = sum(slc) / len(slc)
        var = sum((x - m) ** 2 for x in slc) / len(slc)
        out[i] = math.sqrt(var)
    for i in range(window):
        out[i] = out[window] if len(out) > window else 0.0
    return out


def _build_features(candles: list[dict]):
    """Return (np.ndarray X, last_feature_dict). Same schema as trainer."""
    atr = _atr_series(candles)
    cvd_dvol = _rolling_std_returns(candles)
    rows: list[list[float]] = []
    for i in range(1, len(candles)):
        pc = candles[i - 1]["close"]
        c = candles[i]["close"]
        if pc <= 0 or c <= 0:
            continue
        log_ret = math.log(c / pc)
        atr_pct = (atr[i] / c) if c > 0 else 0.0
        rows.append([log_ret, atr_pct, cvd_dvol[i], abs(log_ret)])
    X = np.asarray(rows, dtype=np.float64)
    last = {
        "log_return": float(rows[-1][0]) if rows else 0.0,
        "atr_pct":    float(rows[-1][1]) if rows else 0.0,
        "cvd_dir_vol":float(rows[-1][2]) if rows else 0.0,
        "abs_log_return": float(rows[-1][3]) if rows else 0.0,
    }
    return X, last


# ── RegimeDetector singleton ─────────────────────────────────
class RegimeDetector:
    """Thread-safe wrapper around a persisted GaussianHMM bundle."""

    def __init__(self, model_path: Path = _MODEL_PATH):
        self._lock = threading.Lock()
        self._bundle: Optional[dict] = None
        self._cache: Optional[dict] = None
        self._cache_ts: float = 0.0
        self._model_path = model_path
        self._load_blocking_errors: list[str] = []
        self._try_load()

    # ── Lifecycle ──
    def _try_load(self) -> None:
        if not (_NUMPY_OK and _JOBLIB_OK and _HMM_OK):
            return
        if not self._model_path.exists():
            log.info(f"[HMM] Model not found at {self._model_path} — "
                     "RegimeDetector disabled (train with: "
                     "python -m ml.train_hmm_regime)")
            return
        try:
            payload = joblib.load(self._model_path)
            # trainer stores either HMMBundle.__dict__ or an HMMBundle
            if hasattr(payload, "__dict__") and not isinstance(payload, dict):
                payload = payload.__dict__
            required = {"model", "scaler", "labels", "feature_names", "n_states"}
            missing = required - set(payload.keys())
            if missing:
                raise ValueError(f"bundle missing keys: {missing}")
            self._bundle = payload
            log.info(
                f"[HMM] RegimeDetector loaded: "
                f"{payload['n_states']} states, "
                f"features={payload['feature_names']}, "
                f"labels={payload['labels']}"
            )
        except Exception as e:
            self._bundle = None
            self._load_blocking_errors.append(str(e))
            log.warning(f"[HMM] Failed to load {self._model_path}: {e}")

    # ── Properties ──
    @property
    def available(self) -> bool:
        return self._bundle is not None

    @property
    def labels(self) -> dict[int, str]:
        return dict(self._bundle["labels"]) if self._bundle else {}

    @property
    def trained_at(self) -> float:
        return float(self._bundle.get("trained_at", 0.0)) if self._bundle else 0.0

    # ── Main decode path ──
    def _decode_internal(self, candles_5m: list[dict]) -> Optional[dict]:
        """Pure decode: no caching. Returns None if HMM is unavailable or
        insufficient data is provided."""
        if not self.available:
            return None
        if not candles_5m or len(candles_5m) < 20:
            return None

        # Take the last _DECODE_WINDOW_BARS bars for a fast rolling decode.
        window = candles_5m[-_DECODE_WINDOW_BARS:]
        try:
            X_raw, last_features = _build_features(window)
            if X_raw.shape[0] < 10:
                return None
            scaler = self._bundle["scaler"]
            model = self._bundle["model"]
            X = scaler.transform(X_raw)

            # Viterbi decode over the whole rolling window
            logprob, state_seq = model.decode(X, algorithm="viterbi")
            # Posterior probabilities per observation → take the last one
            post = model.predict_proba(X)
            last_probs = post[-1].tolist()
            state_idx = int(np.argmax(last_probs))
            confidence = float(last_probs[state_idx])

            labels = self._bundle["labels"]
            # joblib can load int keys as str — normalize
            def _lbl(k: int) -> str:
                return labels.get(k, labels.get(str(k), f"STATE_{k}"))

            trans_row = model.transmat_[state_idx]
            transition_to = {
                _lbl(j): float(trans_row[j]) for j in range(model.n_components)
            }

            return {
                "label": _lbl(state_idx),
                "state_idx": state_idx,
                "confidence": round(confidence, 4),
                "state_probs": [round(p, 4) for p in last_probs],
                "transition_to": {k: round(v, 4) for k, v in transition_to.items()},
                "features": last_features,
                "decoded_at": time.time(),
                "trained_at": self.trained_at,
                "logprob_window": float(logprob),
                "n_bars_decoded": int(X.shape[0]),
            }
        except Exception as e:
            log.warning(f"[HMM] decode failed: {e}")
            return None

    def refresh(self, candles_5m: list[dict]) -> Optional[dict]:
        """Decode + store in internal cache."""
        out = self._decode_internal(candles_5m)
        if out is not None:
            with self._lock:
                self._cache = out
                self._cache_ts = out["decoded_at"]
        return out

    def current(self, max_age_sec: float = _CACHE_TTL_SEC) -> Optional[dict]:
        """Return the last cached result if it's still fresh."""
        with self._lock:
            if self._cache is None:
                return None
            if time.time() - self._cache_ts > max_age_sec:
                return None
            return dict(self._cache)


# ── Module-level singleton ───────────────────────────────────
_singleton: Optional[RegimeDetector] = None
_singleton_lock = threading.Lock()


def get_regime_detector() -> RegimeDetector:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = RegimeDetector()
    return _singleton


# ── RAG regime → strategy-family mapping ────────────────────
# Used by engine.py when retrieving from ChromaDB. The label is turned into a
# concept_family search hint so the RAG embedding retrieval is biased toward
# strategies aligned with the current Markov state. This *replaces* the
# 7-loss fatigue trigger as the primary rotation signal.
REGIME_TO_STRATEGY_FAMILY: dict[str, list[str]] = {
    "RANGE_CHOP":     ["mean_reversion", "range_fade", "bollinger_reversion",
                       "liquidity_sweep", "absorption"],
    "LOW_VOL_DRIFT":  ["mean_reversion", "carry", "micro_trend", "vwap_pullback"],
    "TREND_BULL":     ["momentum", "trend_following", "breakout",
                       "order_block_continuation"],
    "TREND_BEAR":     ["momentum_short", "trend_following", "breakdown",
                       "supply_zone"],
    "HIGH_VOL_SHOCK": ["news_fade", "volatility_contraction", "event_driven",
                       "liquidity_sweep"],
}


def regime_rag_hint(regime_label: str) -> str:
    """Return a search-string fragment to steer ChromaDB retrieval toward
    strategies compatible with the current regime. Empty string if unknown."""
    families = REGIME_TO_STRATEGY_FAMILY.get(regime_label, [])
    if not families:
        return ""
    return (
        f"Regimen de Markov detectado: {regime_label}. "
        f"Prefiere estrategias de familia: {', '.join(families)}."
    )
