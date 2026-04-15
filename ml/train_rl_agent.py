"""
train_rl_agent.py — Reinforcement Learning Portfolio Allocator for btc5min

Replaces the hard-coded Half-Kelly sizing in `btc5min/risk/risk.py` with a
learned policy that sees the Judge probability, the HMM Markov regime, the
agent's recent PnL tape, and the local volatility, then outputs a single
continuous action in [0, 1] representing the *fraction of the dynamic budget
cap* to risk on this window (0.0 = skip).

Reward function is Sortino-aware:

    step_reward =   pnl / initial_balance                       (scale)
                  - lambda_dd * max(0, dd_pct - 0.05) ** 2      (convex DD penalty)
                  + beta_skip * (action==0 and judge in [0.45,0.55])
                  + beta_conv * (action>0 and matched correct direction)

At episode end a terminal bonus is added:

    terminal_bonus = clip(sortino(episode_returns) * 0.10, -1, 1)

This preserves gradient signal per step while still steering the agent toward
policies that maximize downside-adjusted growth.

Observation space
-----------------
13-dim continuous Box. Order and semantics are defined ONCE in
`btc5min/risk/rl_wrapper.py:OBS_FEATURE_NAMES` and imported here so the
training-time and inference-time contracts can never drift.

Action space
------------
Box(low=0.0, high=1.0, shape=(1,), dtype=float32).
Mapped to dollars via `action_to_bet()` from the same wrapper module.

Data loader
-----------
Pulls resolved decision points from the project SQLite DB. Required columns:
    predictions: window_id, confidence, ai_model, ml_oracle_prob (optional)
    bets:        window_id, ai_model, odds_cents, won, profit, amount
    windows:     window_id, outcome
    market_context (optional): atr_pct

If the DB has fewer than `--min-samples` decision points it falls back to a
synthetic generator so you can at least smoke-test the full pipeline.

Usage
-----
    # First training (PPO, from live DB history):
    python -m ml.train_rl_agent --algo ppo --timesteps 200000 --episodes 10

    # SAC alternative (more sample-efficient for continuous actions):
    python -m ml.train_rl_agent --algo sac --timesteps 150000

    # Smoke test on synthetic data:
    python -m ml.train_rl_agent --algo ppo --timesteps 20000 --synthetic

Outputs
-------
    models/rl_risk_agent.zip          the trained policy (loaded by risk.py)
    models/rl_risk_agent_monitor.csv  SB3 rollout metrics
    models/rl_risk_agent_report.txt   final Sortino / Sharpe / WR summary
"""

from __future__ import annotations

import argparse
import math
import random
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise SystemExit(
        "gymnasium is required. Install with: pip install gymnasium"
    ) from e

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as e:
    raise SystemExit(
        "stable-baselines3 is required. Install with: "
        "pip install 'stable-baselines3[extra]>=2.3.0'"
    ) from e

# Import the SHARED observation contract. This guarantees train/infer parity.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from btc5min.risk.rl_wrapper import (  # noqa: E402
    OBS_DIM,
    OBS_FEATURE_NAMES,
    MAX_PCT_CAP,
    HARD_USD_CAP,
    build_observation,
    action_to_bet,
)

# ── Paths ────────────────────────────────────────────────────
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "rl_risk_agent.zip"
MONITOR_PATH = MODELS_DIR / "rl_risk_agent_monitor.csv"
REPORT_PATH = MODELS_DIR / "rl_risk_agent_report.txt"
DB_PATH = REPO_ROOT / "btc5min.db"


# ── Decision-point schema ────────────────────────────────────
@dataclass
class DecisionPoint:
    """A single historical trade opportunity replayed by the env.

    All fields are plain Python so the env can be fully deterministic and
    pickle-safe for vectorized training workers.
    """
    judge_prob: float          # P(correct) from Judge (or confidence/100)
    predicted_side: str        # "UP" or "DOWN"
    odds_cents: int            # Polymarket ask at bet time
    won: bool                  # realized outcome
    atr_pct: float             # volatility at decision time
    hmm_label: str             # HMM regime label (or "")
    hmm_confidence: float      # HMM posterior (0 if no HMM)


# ── Data loader ──────────────────────────────────────────────
def load_historical_decisions(db_path: Path,
                              min_samples: int = 200,
                              ) -> list[DecisionPoint]:
    """Pull resolved historical bets from SQLite and turn them into
    DecisionPoints the env can replay."""
    if not db_path.exists():
        print(f"[RL-DATA] DB not found at {db_path}")
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Discover optional columns so this works across schema versions
    pred_cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    bets_cols = {r[1] for r in conn.execute("PRAGMA table_info(bets)").fetchall()}
    mctx_cols = {r[1] for r in conn.execute("PRAGMA table_info(market_context)").fetchall()}

    has_oracle = "ml_oracle_prob" in pred_cols
    has_atr = "atr_pct" in mctx_cols or "atr_pct" in pred_cols
    has_regime = "regime_label" in pred_cols

    # Prefer bets that have odds + won; LEFT JOIN predictions for context
    sql = f"""
        SELECT
            b.window_id                                  AS window_id,
            b.ai_model                                   AS ai_model,
            b.odds_cents                                 AS odds_cents,
            b.won                                        AS won,
            b.side                                       AS side,
            p.confidence                                 AS confidence,
            {'p.ml_oracle_prob' if has_oracle else 'NULL'} AS ml_oracle_prob,
            {'p.atr_pct' if 'atr_pct' in pred_cols else 'NULL'} AS pred_atr,
            {'p.regime_label' if has_regime else 'NULL'} AS regime_label,
            {'p.regime_confidence' if 'regime_confidence' in pred_cols else 'NULL'} AS regime_conf,
            {'mc.atr_pct' if 'atr_pct' in mctx_cols else 'NULL'} AS mc_atr
        FROM bets b
        LEFT JOIN predictions p
               ON p.window_id = b.window_id AND p.ai_model = b.ai_model
        LEFT JOIN market_context mc
               ON mc.window_id = b.window_id
        WHERE b.won IS NOT NULL
          AND b.odds_cents IS NOT NULL
          AND b.is_ghost = 0
        ORDER BY b.window_id ASC
    """
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.OperationalError as e:
        print(f"[RL-DATA] Query failed: {e}. Schema may lack 'is_ghost'.")
        rows = []
    conn.close()

    decisions: list[DecisionPoint] = []
    for r in rows:
        # Judge prob preference: ml_oracle_prob > confidence/100 > 0.5
        if r["ml_oracle_prob"] is not None:
            jp = float(r["ml_oracle_prob"])
        elif r["confidence"] is not None:
            jp = float(r["confidence"]) / 100.0
        else:
            jp = 0.5
        jp = max(0.01, min(0.99, jp))

        atr = r["mc_atr"] if r["mc_atr"] is not None else r["pred_atr"]
        atr = float(atr or 0.03)

        decisions.append(DecisionPoint(
            judge_prob=jp,
            predicted_side=(r["side"] or "UP").upper(),
            odds_cents=int(r["odds_cents"]),
            won=bool(r["won"]),
            atr_pct=atr,
            hmm_label=(r["regime_label"] or ""),
            hmm_confidence=float(r["regime_conf"] or 0.0),
        ))

    print(f"[RL-DATA] Loaded {len(decisions)} decisions from {db_path}")
    return decisions


def synthetic_decisions(n: int = 5000, seed: int = 42) -> list[DecisionPoint]:
    """Fallback dataset so the trainer is runnable on a cold DB."""
    rng = random.Random(seed)
    labels = ["RANGE_CHOP", "LOW_VOL_DRIFT", "TREND_BULL", "TREND_BEAR", ""]
    out: list[DecisionPoint] = []
    for _ in range(n):
        # Regime-dependent edge: trend regimes have higher judge accuracy
        label = rng.choice(labels)
        if "TREND" in label:
            jp = rng.uniform(0.55, 0.80)
        elif label == "RANGE_CHOP":
            jp = rng.uniform(0.40, 0.60)
        else:
            jp = rng.uniform(0.48, 0.68)
        # True outcome sampled from jp (so the agent CAN learn)
        won = rng.random() < jp
        out.append(DecisionPoint(
            judge_prob=jp,
            predicted_side=rng.choice(["UP", "DOWN"]),
            odds_cents=rng.choice([45, 50, 52, 55, 58]),
            won=won,
            atr_pct=rng.uniform(0.01, 0.08),
            hmm_label=label,
            hmm_confidence=rng.uniform(0.55, 0.92) if label else 0.0,
        ))
    return out


# ── Gymnasium environment ────────────────────────────────────
class TradingRiskEnv(gym.Env):
    """Replays historical decision points and learns a sizing policy.

    Episode = a contiguous slice of N decisions (default 256). The agent
    starts with `initial_balance` and at each step picks an action in [0,1].
    Reward blends a step-wise PnL signal with a convex drawdown penalty and
    a terminal Sortino bonus.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 decisions: list[DecisionPoint],
                 episode_length: int = 256,
                 initial_balance: float = 100.0,
                 lambda_dd: float = 2.0,
                 beta_skip: float = 0.005,
                 beta_conv: float = 0.001,
                 terminal_sortino_weight: float = 0.10,
                 seed: int = 0):
        super().__init__()
        if len(decisions) < 50:
            raise ValueError(f"Need at least 50 decisions, got {len(decisions)}")
        self._decisions = decisions
        self._episode_length = min(episode_length, len(decisions) - 1)
        self._initial_balance = float(initial_balance)
        self._lambda_dd = lambda_dd
        self._beta_skip = beta_skip
        self._beta_conv = beta_conv
        self._terminal_sortino_weight = terminal_sortino_weight
        self._rng = np.random.default_rng(seed)

        # Observation: 13-dim Box
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32,
        )
        # Action: single scalar [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        # Episode state (reset fills these)
        self._ptr: int = 0
        self._start: int = 0
        self._balance: float = initial_balance
        self._peak_balance: float = initial_balance
        self._last_pnls: list[float] = [0.0] * 5
        self._step_returns: list[float] = []
        self._skip_count: int = 0
        self._bet_count: int = 0

    # ── Gym API ──
    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        max_start = max(0, len(self._decisions) - self._episode_length - 1)
        self._start = int(self._rng.integers(0, max_start + 1))
        self._ptr = self._start
        self._balance = self._initial_balance
        self._peak_balance = self._initial_balance
        self._last_pnls = [0.0] * 5
        self._step_returns = []
        self._skip_count = 0
        self._bet_count = 0
        return self._current_obs(), {}

    def step(self, action):
        a = float(np.clip(action, 0.0, 1.0).flatten()[0])
        dp = self._decisions[self._ptr]

        # Translate action → bet amount (uses the shared wrapper logic)
        bet_amount, _label = action_to_bet(a, self._balance)
        is_skip = bet_amount == 0.0

        # ── Realize PnL ──
        pnl = 0.0
        if not is_skip and bet_amount > 0:
            self._bet_count += 1
            odds = max(5, min(95, dp.odds_cents)) / 100.0
            if dp.won:
                # Polymarket payout: token price -> $1, fee 2% on winnings
                gross = (1.0 / odds) * bet_amount
                profit = (gross - bet_amount) * 0.98
                pnl = profit
            else:
                pnl = -bet_amount
        else:
            self._skip_count += 1

        prev_balance = self._balance
        self._balance = max(0.0, self._balance + pnl)
        self._peak_balance = max(self._peak_balance, self._balance)
        self._last_pnls = (self._last_pnls + [pnl])[-5:]
        self._step_returns.append(pnl / self._initial_balance)

        # ── Reward shaping ──
        step_reward = pnl / self._initial_balance

        # Convex drawdown penalty beyond 5%
        dd_pct = (self._peak_balance - self._balance) / self._peak_balance if self._peak_balance > 0 else 0.0
        if dd_pct > 0.05:
            step_reward -= self._lambda_dd * (dd_pct - 0.05) ** 2

        # Reward skipping marginal signals (judge in [0.45, 0.55])
        if is_skip and 0.45 <= dp.judge_prob <= 0.55:
            step_reward += self._beta_skip

        # Tiny conviction bonus when betting on high-edge signals
        if not is_skip and dp.judge_prob >= 0.60:
            step_reward += self._beta_conv * (dp.judge_prob - 0.60) * 10

        # ── Advance ptr / termination ──
        self._ptr += 1
        steps_done = self._ptr - self._start
        terminated = (
            self._balance < 2.0                              # ruin
            or steps_done >= self._episode_length            # episode length
            or self._ptr >= len(self._decisions) - 1         # data exhausted
        )
        truncated = False

        # Terminal Sortino bonus
        if terminated:
            step_reward += self._terminal_sortino_weight * self._episode_sortino()

        obs = self._current_obs()
        info = {
            "balance": self._balance,
            "peak": self._peak_balance,
            "dd_pct": dd_pct,
            "bets": self._bet_count,
            "skips": self._skip_count,
            "pnl": pnl,
        }
        return obs, float(step_reward), terminated, truncated, info

    # ── Helpers ──
    def _current_obs(self) -> np.ndarray:
        idx = min(self._ptr, len(self._decisions) - 1)
        dp = self._decisions[idx]
        hmm_dict = None
        if dp.hmm_label:
            hmm_dict = {"label": dp.hmm_label, "confidence": dp.hmm_confidence}
        obs_list = build_observation(
            judge_prob=dp.judge_prob,
            hmm=hmm_dict,
            balance=self._balance,
            initial_balance=self._initial_balance,
            last_pnls=self._last_pnls,
            atr_pct=dp.atr_pct,
            odds_cents=dp.odds_cents,
        )
        return np.asarray(obs_list, dtype=np.float32)

    def _episode_sortino(self) -> float:
        """Unannualized Sortino over the episode's step returns."""
        if len(self._step_returns) < 5:
            return 0.0
        r = np.asarray(self._step_returns, dtype=np.float64)
        mean_r = float(r.mean())
        downside = r[r < 0]
        if downside.size == 0:
            return float(np.clip(mean_r * 20, -1, 1))
        downside_dev = float(np.sqrt(np.mean(downside ** 2)))
        if downside_dev <= 1e-9:
            return 0.0
        sortino = mean_r / downside_dev
        return float(np.clip(sortino, -1.0, 1.0))


# ── Evaluation ───────────────────────────────────────────────
def evaluate_policy(model, env: TradingRiskEnv, episodes: int = 20) -> dict:
    """Run the trained policy on held-out episodes and report key stats."""
    final_balances = []
    all_returns = []
    win_bets = 0
    total_bets = 0
    skips = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        final_balances.append(info["balance"])
        total_bets += info["bets"]
        skips += info["skips"]
        all_returns.extend(env._step_returns)

    fb = np.asarray(final_balances)
    ret = np.asarray(all_returns)
    neg = ret[ret < 0]
    downside_dev = float(np.sqrt(np.mean(neg ** 2))) if neg.size else 0.0
    sortino = float(ret.mean() / downside_dev) if downside_dev > 1e-9 else 0.0
    sharpe = float(ret.mean() / ret.std()) if ret.std() > 1e-9 else 0.0

    return {
        "episodes": episodes,
        "mean_final_balance": float(fb.mean()),
        "median_final_balance": float(np.median(fb)),
        "worst_final_balance": float(fb.min()),
        "best_final_balance": float(fb.max()),
        "sortino": sortino,
        "sharpe": sharpe,
        "total_bets": total_bets,
        "total_skips": skips,
        "skip_ratio": skips / max(1, total_bets + skips),
    }


# ── Main ─────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train a Stable-Baselines3 RL portfolio allocator"
    )
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--episode-length", type=int, default=256)
    ap.add_argument("--initial-balance", type=float, default=100.0)
    ap.add_argument("--min-samples", type=int, default=200,
                    help="Minimum DB decisions before falling back to synthetic")
    ap.add_argument("--synthetic", action="store_true",
                    help="Force synthetic data (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--eval-episodes", type=int, default=20)
    args = ap.parse_args()

    # ── Load data ──
    decisions: list[DecisionPoint] = []
    if not args.synthetic:
        decisions = load_historical_decisions(args.db, args.min_samples)
    if len(decisions) < args.min_samples:
        print(f"[RL] Using synthetic dataset "
              f"(live DB had {len(decisions)} < {args.min_samples})")
        decisions = synthetic_decisions(n=5000, seed=args.seed)
    print(f"[RL] Decision dataset: {len(decisions)} points, "
          f"features={OBS_FEATURE_NAMES}")

    # ── Build env ──
    def _make_env():
        env = TradingRiskEnv(
            decisions,
            episode_length=args.episode_length,
            initial_balance=args.initial_balance,
            seed=args.seed,
        )
        return Monitor(env, filename=str(MONITOR_PATH))

    # Smoke-check once
    check_env(_make_env(), warn=True, skip_render_check=True)
    vec_env = DummyVecEnv([_make_env])

    # ── Train ──
    t0 = time.time()
    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            seed=args.seed,
            device="cpu",
        )
    else:
        model = SAC(
            "MlpPolicy", vec_env,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=100_000,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            seed=args.seed,
            device="cpu",
        )
    print(f"[RL] Training {args.algo.upper()} for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    train_secs = time.time() - t0
    print(f"[RL] Training done in {train_secs:.1f}s")

    # ── Save ──
    model.save(str(args.out))
    print(f"[RL] Saved policy → {args.out}")

    # ── Evaluate on fresh env ──
    eval_env = TradingRiskEnv(
        decisions, episode_length=args.episode_length,
        initial_balance=args.initial_balance, seed=args.seed + 1,
    )
    stats = evaluate_policy(model, eval_env, episodes=args.eval_episodes)
    print("\n[RL] Evaluation:")
    for k, v in stats.items():
        print(f"    {k:24s} = {v}")

    # ── Write report ──
    report_lines = [
        f"btc5min RL Risk Agent — Training Report",
        f"trained_at         : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"algo               : {args.algo.upper()}",
        f"timesteps          : {args.timesteps}",
        f"episode_length     : {args.episode_length}",
        f"initial_balance    : {args.initial_balance}",
        f"decisions_used     : {len(decisions)}",
        f"obs_features       : {OBS_FEATURE_NAMES}",
        f"action_space       : Box([0,1], shape=(1,))",
        f"safety_caps        : max_pct={MAX_PCT_CAP}, hard_usd={HARD_USD_CAP}",
        f"train_secs         : {train_secs:.1f}",
        "",
        "Evaluation:",
    ]
    for k, v in stats.items():
        report_lines.append(f"    {k:24s} = {v}")
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[RL] Report → {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
