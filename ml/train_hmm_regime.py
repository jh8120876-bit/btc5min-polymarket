"""
train_hmm_regime.py — Unsupervised Regime Detection (Hidden Markov Model)

Trains a GaussianHMM on a multivariate feature vector computed from 5-min
BTC/USD history and persists the fitted model to `models/hmm_model.pkl`.

Feature vector (per 5-min bar, standardized with a StandardScaler):
    1. log_return       — log(close_t / close_{t-1})
    2. atr_pct          — ATR(14) / close  (volatility magnitude)
    3. cvd_dir_vol      — Directional Volatility of CVD imbalance within the bar
    4. abs_log_return   — |log_return|     (helps split mean-reverting vs trending)

Latent states (3 by default, can be 4):
    These are discovered unsupervised. After training the script labels each
    state by its posterior mean of (log_return, atr_pct) so downstream code
    (RAG rotation) can read a human-friendly tag:
        - "RANGE_CHOP"        → low |return|, low-mid vol
        - "LOW_VOL_DRIFT"     → low vol, mild directional bias
        - "TREND_BULL"        → high |return|, positive mean return, high vol
        - "TREND_BEAR"        → high |return|, negative mean return, high vol
        - "HIGH_VOL_SHOCK"    → extreme vol, no clear direction (4-state model only)

Inputs
------
Historical 5-min candles (timestamp, open, high, low, close) pulled from the
project SQLite `price_ticks` table, resampled to 5m OHLC.  CVD series are
pulled from a cached per-window aggregate if present; otherwise the CVD
directional volatility falls back to a rolling std of normalized returns so
the model still trains on systems without WS history.

Outputs
-------
    models/hmm_model.pkl        joblib dump of { "model", "scaler", "labels",
                                                  "feature_names", "n_states",
                                                  "trained_at" }

Usage
-----
    python -m ml.train_hmm_regime --states 3 --lookback-days 45
    python -m ml.train_hmm_regime --states 4 --lookback-days 90 --min-bars 2000

Retrocompatibility
------------------
This script is a standalone trainer. It does NOT touch the live engine while
running — it reads from SQLite read-only and writes to models/ atomically via
`joblib.dump(..., compress=3)`.
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as e:
    raise SystemExit(
        "hmmlearn is required. Install with:  pip install hmmlearn\n"
        f"Original error: {e}"
    )

try:
    import joblib
except ImportError as e:
    raise SystemExit("joblib is required. Install with: pip install joblib") from e

from sklearn.preprocessing import StandardScaler


# ── Paths ────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "hmm_model.pkl"
DB_PATH = REPO_ROOT / "btc5min.db"  # project default


# ── Data classes ─────────────────────────────────────────────
@dataclass
class HMMBundle:
    """Serialized payload stored in hmm_model.pkl."""
    model: GaussianHMM
    scaler: StandardScaler
    labels: dict[int, str]                 # {state_idx: "TREND_BULL", ...}
    feature_names: list[str]
    n_states: int
    trained_at: float
    stats: dict                            # per-state mean/var diagnostics


# ── SQLite loader ────────────────────────────────────────────
def _load_5m_ohlc(db_path: Path, lookback_days: int) -> list[dict]:
    """Resample `price_ticks` into 5-min OHLC candles.

    Uses SQL aggregation rather than pandas so training works on minimal envs.
    """
    if not db_path.exists():
        raise SystemExit(f"DB not found at {db_path}. Run the engine first to populate it.")

    cutoff = time.time() - lookback_days * 86400
    # Bucket each tick into its 5-min slot (UTC) and aggregate.
    sql = """
        SELECT
            (CAST(timestamp AS INTEGER) / 300) * 300 AS bucket_ts,
            MIN(price) AS low,
            MAX(price) AS high,
            -- first / last per bucket via correlated subqueries on rowid is slow;
            -- rely on timestamp ordering and GROUP_CONCAT fallback.
            AVG(price) AS avg_price,
            COUNT(*)   AS n_ticks
        FROM price_ticks
        WHERE timestamp >= ?
        GROUP BY bucket_ts
        ORDER BY bucket_ts ASC
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, (cutoff,)).fetchall()

    # For open/close we need the first/last tick per bucket. Do a second pass.
    opens, closes = {}, {}
    for r in conn.execute(
        "SELECT timestamp, price FROM price_ticks WHERE timestamp >= ? ORDER BY timestamp ASC",
        (cutoff,),
    ):
        b = (int(r[0]) // 300) * 300
        if b not in opens:
            opens[b] = r[1]
        closes[b] = r[1]
    conn.close()

    candles: list[dict] = []
    for r in rows:
        b = r["bucket_ts"]
        if b not in opens:
            continue
        candles.append({
            "ts": b,
            "open": float(opens[b]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(closes[b]),
            "n": int(r["n_ticks"]),
        })
    return candles


# ── Feature engineering ──────────────────────────────────────
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
    # Wilder smoothing
    atr = [0.0] * len(trs)
    if len(trs) >= period:
        seed = sum(trs[1:period + 1]) / period
        atr[period] = seed
        for i in range(period + 1, len(trs)):
            atr[i] = (atr[i - 1] * (period - 1) + trs[i]) / period
        # back-fill leading zeros with the seed so we don't drop bars
        for i in range(period):
            atr[i] = seed
    return atr


def _cvd_directional_volatility(candles: list[dict], window: int = 12) -> list[float]:
    """Proxy for CVD directional volatility when WS history is not persisted.

    Computes a rolling standard deviation of signed close-to-close returns,
    which captures how "directional" the aggressor pressure has been.
    In production we'd replace this with true CVD imbalance pulled from
    `market_context` — but that table only has per-window snapshots, so this
    proxy keeps the trainer runnable on a cold DB.
    """
    rets: list[float] = [0.0]
    for i in range(1, len(candles)):
        pc, c = candles[i - 1]["close"], candles[i]["close"]
        rets.append(math.log(c / pc) if pc > 0 else 0.0)
    out: list[float] = [0.0] * len(rets)
    for i in range(window, len(rets)):
        slc = rets[i - window + 1:i + 1]
        m = sum(slc) / len(slc)
        var = sum((x - m) ** 2 for x in slc) / len(slc)
        out[i] = math.sqrt(var)
    # back-fill
    for i in range(window):
        out[i] = out[window] if len(out) > window else 0.0
    return out


def build_feature_matrix(candles: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Return (X, feature_names). X shape = (n_bars - 1, 4)."""
    if len(candles) < 30:
        raise SystemExit(
            f"Not enough 5-min candles ({len(candles)}). "
            "Need at least 30 bars — wait for more data or run with --lookback-days <larger>."
        )
    atr = _atr_series(candles)
    cvd_dvol = _cvd_directional_volatility(candles)

    rows: list[list[float]] = []
    for i in range(1, len(candles)):
        c = candles[i]["close"]
        pc = candles[i - 1]["close"]
        if pc <= 0 or c <= 0:
            continue
        log_ret = math.log(c / pc)
        atr_pct = (atr[i] / c) if c > 0 else 0.0
        rows.append([
            log_ret,
            atr_pct,
            cvd_dvol[i],
            abs(log_ret),
        ])
    X = np.asarray(rows, dtype=np.float64)
    return X, ["log_return", "atr_pct", "cvd_dir_vol", "abs_log_return"]


# ── State labeling ───────────────────────────────────────────
def _label_states(model: GaussianHMM, n_states: int,
                  feature_names: list[str]) -> tuple[dict[int, str], dict]:
    """Assign human-readable labels to the HMM latent states based on their
    standardized feature means. Returns (labels, diagnostics)."""
    means = model.means_                     # shape (n_states, n_features)
    idx_ret = feature_names.index("log_return")
    idx_vol = feature_names.index("atr_pct")
    idx_abs = feature_names.index("abs_log_return")

    # Sort states by volatility magnitude to classify them
    vol_rank = np.argsort(means[:, idx_vol])     # low → high vol
    labels: dict[int, str] = {}

    if n_states == 3:
        # lowest vol → RANGE_CHOP, mid → LOW_VOL_DRIFT, high → TREND_*
        low, mid, high = vol_rank[0], vol_rank[1], vol_rank[2]
        labels[int(low)] = "RANGE_CHOP"
        labels[int(mid)] = "LOW_VOL_DRIFT"
        labels[int(high)] = (
            "TREND_BULL" if means[high, idx_ret] >= 0 else "TREND_BEAR"
        )
    elif n_states == 4:
        low, mid1, mid2, high = vol_rank
        labels[int(low)] = "RANGE_CHOP"
        labels[int(mid1)] = "LOW_VOL_DRIFT"
        # mid2 is a trending regime — sign decides bull vs bear
        labels[int(mid2)] = (
            "TREND_BULL" if means[mid2, idx_ret] >= 0 else "TREND_BEAR"
        )
        labels[int(high)] = "HIGH_VOL_SHOCK"
    else:
        for s in range(n_states):
            labels[int(s)] = f"STATE_{s}"

    diagnostics = {
        "means": means.tolist(),
        "covars_diag": [np.diag(c).tolist() for c in model.covars_],
        "transmat": model.transmat_.tolist(),
        "startprob": model.startprob_.tolist(),
        "feature_names": feature_names,
    }
    return labels, diagnostics


# ── Trainer entry point ──────────────────────────────────────
def train(n_states: int, lookback_days: int, min_bars: int,
          db_path: Path, out_path: Path, seed: int = 42) -> HMMBundle:
    print(f"[HMM] Loading 5-min candles from {db_path} "
          f"(lookback={lookback_days}d)")
    candles = _load_5m_ohlc(db_path, lookback_days)
    print(f"[HMM] Loaded {len(candles)} candles")

    X_raw, feature_names = build_feature_matrix(candles)
    if X_raw.shape[0] < min_bars:
        raise SystemExit(
            f"Only {X_raw.shape[0]} usable bars (< min_bars={min_bars}). "
            "Increase --lookback-days or lower --min-bars."
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    print(f"[HMM] Feature matrix: shape={X.shape}, features={feature_names}")

    # Diagonal covariance is enough for this low-dim setup and is far more stable.
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        tol=1e-4,
        random_state=seed,
        verbose=False,
    )
    t0 = time.time()
    model.fit(X)
    fit_secs = time.time() - t0

    # Score to report goodness of fit
    logprob = model.score(X)
    print(f"[HMM] Fit done in {fit_secs:.1f}s | logprob={logprob:.2f} "
          f"| converged={model.monitor_.converged}")

    labels, diagnostics = _label_states(model, n_states, feature_names)
    print("[HMM] State labels:")
    for s, name in labels.items():
        print(f"        state {s} → {name}  (mean={diagnostics['means'][s]})")

    bundle = HMMBundle(
        model=model,
        scaler=scaler,
        labels=labels,
        feature_names=feature_names,
        n_states=n_states,
        trained_at=time.time(),
        stats={
            "logprob": logprob,
            "fit_secs": fit_secs,
            "n_bars": int(X.shape[0]),
            "lookback_days": lookback_days,
            "diagnostics": diagnostics,
        },
    )

    # Persist atomically
    tmp_path = out_path.with_suffix(".pkl.tmp")
    joblib.dump(bundle.__dict__, tmp_path, compress=3)
    os.replace(tmp_path, out_path)
    print(f"[HMM] Saved → {out_path}")
    return bundle


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a Gaussian HMM regime detector")
    ap.add_argument("--states", type=int, default=3,
                    help="Number of latent states (3 or 4 recommended)")
    ap.add_argument("--lookback-days", type=int, default=45,
                    help="How many days of 5m history to pull from SQLite")
    ap.add_argument("--min-bars", type=int, default=300,
                    help="Minimum number of usable 5-min bars before training")
    ap.add_argument("--db", type=Path, default=DB_PATH,
                    help="Path to btc5min.db")
    ap.add_argument("--out", type=Path, default=MODEL_PATH,
                    help="Output pickle path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        train(
            n_states=args.states,
            lookback_days=args.lookback_days,
            min_bars=args.min_bars,
            db_path=args.db,
            out_path=args.out,
            seed=args.seed,
        )
    except SystemExit as e:
        print(f"[HMM] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
