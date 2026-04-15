"""
rl_wrapper.py — Runtime wrapper around a Stable-Baselines3 portfolio-sizing agent.

Responsibilities
----------------
1. Define the canonical 13-dimensional observation vector that both the
   trainer (`ml/train_rl_agent.py`) and the live risk manager consume.
   If you change the order, BOTH sides must be retrained/rebuilt.
2. Load `models/rl_risk_agent.zip` once, safely, with graceful degradation
   if Stable-Baselines3 / PyTorch / the .zip artifact are missing.
3. Translate a continuous action in [0, 1] into a concrete dollar bet,
   honoring a hard cap so the RL policy cannot blow up the account.
4. Stay zero-dependency on the live engine's tick loop: `predict()` is a
   ~1 ms PyTorch forward pass, safe to call from within the engine's
   async `_fetch_and_process_prediction` thread.

Action-space contract (matches the Gymnasium env in ml/train_rl_agent.py)
------------------------------------------------------------------------
    action: float in [0.0, 1.0]                (shape=(1,))
        0.0            → SKIP the trade (register as ghost).
        (0.0, 1.0]     → fraction of the dynamic budget cap to risk.

    effective_bet = action * budget_cap
    budget_cap    = min(balance * MAX_PCT_CAP, HARD_USD_CAP)

`MAX_PCT_CAP` (5%) and `HARD_USD_CAP` ($5) are the existing RiskManager
guard-rails — the RL agent can only move WITHIN them, never beyond.

Retrocompatibility
------------------
- If `stable_baselines3` is not installed → `available=False`, all callers
  must fall back to the classic Kelly path.
- If the `.zip` artifact is missing or corrupted → same degradation.
- If the observation passed at inference time is malformed → logs a warning
  and returns `None` so the caller can fall back.
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
    log.warning("[RL] numpy missing — RLPortfolioAgent disabled")

try:
    from stable_baselines3 import PPO, SAC
    _SB3_OK = True
except ImportError:
    PPO = SAC = None  # type: ignore
    _SB3_OK = False
    log.info("[RL] stable-baselines3 not installed — classic Kelly only")


# ── Canonical observation contract ───────────────────────────
# DO NOT REORDER without retraining the agent.
OBS_FEATURE_NAMES: list[str] = [
    "judge_prob",          # 0  — XGBoost/LSTM Judge probability of being correct
    "hmm_confidence",      # 1  — HMM Viterbi posterior of picked state
    "hmm_is_range",        # 2  — one-hot: RANGE_CHOP
    "hmm_is_low_vol",      # 3  — one-hot: LOW_VOL_DRIFT
    "hmm_is_trend",        # 4  — one-hot: TREND_BULL | TREND_BEAR | HIGH_VOL_SHOCK
    "balance_ratio",       # 5  — balance / initial_balance, clipped [0,2]
    "pnl_lag1",            # 6  — last realized pnl / initial_balance
    "pnl_lag2",            # 7
    "pnl_lag3",            # 8
    "pnl_lag4",            # 9
    "pnl_lag5",            # 10
    "atr_pct_norm",        # 11 — volatility (atr_pct) / 0.2 clipped [0,1]
    "odds_norm",           # 12 — odds_cents / 100 in [0,1]
]
OBS_DIM = len(OBS_FEATURE_NAMES)   # 13

# Hard safety rails (mirrored in Gymnasium env for training consistency)
MAX_PCT_CAP: float = 0.05          # RL can never risk more than 5% of balance
HARD_USD_CAP: float = 5.0          # absolute $5 cap per trade


# ── Regime label → one-hot bucket ────────────────────────────
_REGIME_BUCKETS = {
    "RANGE_CHOP":     (1, 0, 0),
    "LOW_VOL_DRIFT":  (0, 1, 0),
    "TREND_BULL":     (0, 0, 1),
    "TREND_BEAR":     (0, 0, 1),
    "HIGH_VOL_SHOCK": (0, 0, 1),
}


def _clip(x: float, lo: float, hi: float) -> float:
    if x != x:  # NaN guard
        return 0.0
    return max(lo, min(hi, x))


def build_observation(
    judge_prob: float,
    hmm: dict | None,
    balance: float,
    initial_balance: float,
    last_pnls: list[float],
    atr_pct: float,
    odds_cents: int,
) -> list[float]:
    """Build the 13-dim observation vector.

    All inputs are scalars / plain Python — no numpy needed here so the
    trainer and the live engine speak the same language. The caller
    (engine.py) passes what it already has in hand when the sniper
    consensus fires.

    Args:
        judge_prob:     probability from XGBoost/LSTM Judge (0..1). Pass the
                        AI confidence/100 if judge is unavailable.
        hmm:            snapshot dict from RegimeDetector (or None).
        balance:        current agent balance in USD.
        initial_balance: seed balance, used to scale pnl lags.
        last_pnls:      list of the agent's last 5 realized P&L values (USD).
                        Missing slots are padded with 0.0.
        atr_pct:        current ATR%.
        odds_cents:     Polymarket odds for the chosen side (cents 5..95).
    """
    # ── judge + regime ──
    j = _clip(float(judge_prob or 0.5), 0.0, 1.0)
    if hmm:
        hmm_conf = _clip(float(hmm.get("confidence", 0.0)), 0.0, 1.0)
        bucket = _REGIME_BUCKETS.get(hmm.get("label", ""), (0, 0, 0))
    else:
        hmm_conf = 0.0
        bucket = (0, 0, 0)

    # ── balance ratio ──
    ib = max(1.0, float(initial_balance or 100.0))
    bal_ratio = _clip(float(balance) / ib, 0.0, 2.0)

    # ── pnl lags (pad/truncate to 5) ──
    pnls = list(last_pnls or [])[-5:]
    while len(pnls) < 5:
        pnls.insert(0, 0.0)
    pnl_lags = [_clip(float(p) / ib, -1.0, 1.0) for p in pnls]

    # ── volatility normalization ──
    atr_norm = _clip(float(atr_pct or 0.0) / 0.20, 0.0, 1.0)

    # ── odds normalization ──
    odds_norm = _clip(float(odds_cents or 50) / 100.0, 0.05, 0.95)

    obs = [
        j,
        hmm_conf,
        float(bucket[0]),
        float(bucket[1]),
        float(bucket[2]),
        bal_ratio,
        pnl_lags[0], pnl_lags[1], pnl_lags[2], pnl_lags[3], pnl_lags[4],
        atr_norm,
        odds_norm,
    ]
    assert len(obs) == OBS_DIM, f"obs dim mismatch: {len(obs)} vs {OBS_DIM}"
    return obs


def action_to_bet(action: float, balance: float) -> tuple[float, str]:
    """Convert a scalar action in [0,1] into a concrete bet amount in USD.

    Returns (amount, label). `amount == 0.0` means SKIP (caller should ghost).
    """
    a = _clip(float(action), 0.0, 1.0)
    if a < 1e-3:
        return 0.0, "rl_skip"
    budget_cap = min(balance * MAX_PCT_CAP, HARD_USD_CAP)
    bet = a * budget_cap
    # Floor to $1 if the RL wants to size above epsilon — matches existing
    # MIN_AGENT_BET in risk.py so the Polymarket execution path stays healthy.
    if bet < 1.0:
        return 1.0, f"rl_floor (action={a:.3f})"
    return round(bet, 2), f"rl_size (action={a:.3f}, cap=${budget_cap:.2f})"


# ── RLPortfolioAgent ─────────────────────────────────────────
_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "rl_risk_agent.zip"


class RLPortfolioAgent:
    """Thread-safe wrapper that loads the trained SB3 policy and predicts bets."""

    def __init__(self, model_path: Path = _MODEL_PATH):
        self._lock = threading.Lock()
        self._model = None
        self._model_path = model_path
        self._algo_name = "?"
        self._loaded_at: float = 0.0
        self._try_load()

    def _try_load(self) -> None:
        if not (_NUMPY_OK and _SB3_OK):
            return
        if not self._model_path.exists():
            log.info(f"[RL] Agent not found at {self._model_path} — "
                     "classic Kelly will be used. Train with: "
                     "python -m ml.train_rl_agent")
            return
        # Try PPO first (default trainer output), then SAC fallback
        for algo_cls, algo_name in ((PPO, "PPO"), (SAC, "SAC")):
            if algo_cls is None:
                continue
            try:
                self._model = algo_cls.load(str(self._model_path), device="cpu")
                self._algo_name = algo_name
                self._loaded_at = time.time()
                log.info(f"[RL] Loaded {algo_name} policy from {self._model_path} "
                         f"(obs_dim={OBS_DIM})")
                return
            except Exception as e:
                log.debug(f"[RL] {algo_name}.load failed: {e}")
        log.warning(f"[RL] Could not load {self._model_path} with PPO or SAC — "
                    "falling back to Kelly")

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def algo(self) -> str:
        return self._algo_name

    def predict(self, obs: list[float]) -> Optional[float]:
        """Run the policy forward. Returns a scalar action in [0,1] or None
        if the agent is unavailable / observation is malformed."""
        if not self.available:
            return None
        if not _NUMPY_OK:
            return None
        if len(obs) != OBS_DIM:
            log.warning(f"[RL] obs dim mismatch: got {len(obs)}, expected {OBS_DIM}")
            return None
        try:
            arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            with self._lock:
                action, _ = self._model.predict(arr, deterministic=True)
            val = float(action.flatten()[0])
            # sanity clip
            if val != val or math.isinf(val):
                return None
            return _clip(val, 0.0, 1.0)
        except Exception as e:
            log.warning(f"[RL] predict failed (non-fatal): {e}")
            return None

    def reload(self) -> bool:
        """Re-read the .zip from disk (call after retraining)."""
        with self._lock:
            self._model = None
            self._loaded_at = 0.0
        self._try_load()
        return self.available


# ── Module-level singleton ───────────────────────────────────
_singleton: Optional[RLPortfolioAgent] = None
_singleton_lock = threading.Lock()


def get_rl_agent() -> RLPortfolioAgent:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = RLPortfolioAgent()
    return _singleton
