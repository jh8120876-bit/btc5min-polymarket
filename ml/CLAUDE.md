# ML Pipeline — Training & Evaluation Guide

> **Read this before touching any ML script.** This folder contains all training, evaluation, and backtesting scripts for the btc5min prediction system. All scripts run from the **project root** using `python -m ml.<script>`.

## Architecture Overview

```
ml/
├── CLAUDE.md               # This file — ML pipeline guide
├── generate_dataset.py     # Step 1: Export training CSV from SQLite
├── train_xgboost_judge.py  # Step 2a: Train XGBoost Judge (binary classifier)
├── train_rf.py             # Step 2b: Train RandomForest Oracle (alternative)
├── train_regressor.py      # Step 2c: Train XGBoost Regressor (bet sizing)
├── train_walkforward.py    # Step 3: Validate with walk-forward (temporal)
├── backtest_sniper.py      # Step 4: Backtest sniper consensus strategy
├── train_lstm_judge.py     # PyTorch BiLSTM+Attention temporal judge
├── train_hmm_regime.py     # GaussianHMM unsupervised regime detector
└── train_rl_agent.py       # SB3 PPO/SAC position sizing RL agent
```

**Outputs go to:**
- `data/` — training CSVs, validation results, backtest results
- `models/` — trained model artifacts (.pkl, .joblib)

---

## Quick Start — Full Retrain Pipeline

Run from **project root** (`btc5min/`):

```bash
# 1. Export fresh training data from the live database
python -m ml.generate_dataset

# 2. Train the XGBoost Judge (primary gate for bet filtering)
python -m ml.train_xgboost_judge

# 3. Validate temporal generalization (no data leakage)
python -m ml.train_walkforward

# 4. (Optional) Train regression model for dynamic sizing
python -m ml.train_regressor

# 5. (Optional) Backtest sniper consensus on historical data
python -m ml.backtest_sniper
```

After training, **restart `app.py`** — the engine auto-loads models from `models/` at startup.

---

## Step-by-Step Details

### Step 1: Generate Dataset (`generate_dataset.py`)

Exports training CSV by joining `predictions`, `windows`, `bets`, `market_context`, and `liquidations` tables.

```bash
python -m ml.generate_dataset                      # all data, default agent
python -m ml.generate_dataset --last-days 14       # last 14 days only
python -m ml.generate_dataset --model deepseek     # filter by agent
python -m ml.generate_dataset --model all          # all agents
python -m ml.generate_dataset --include-ghosts     # include ghost trades
```

**Output:** `data/training_data_<model>.csv`

**Key columns:**
- Features: `confidence`, `rsi`, `bb_pct`, `volatility`, `momentum`, `atr_pct`, `hour_utc`, `price_change_pct`, `trend_alignment`, `funding_rate`, `order_book_imbalance`, `liq_buy_vol_5m`, `liq_sell_vol_5m`, `reversal_alert`, `smc_bos_last`, `smc_choch_last`, `smc_in_kill_zone`, `smc_ob_distance_pct`, `consensus_agreement`, `bid_ask_spread`, `sniper_time_to_bet_sec`, `volatility_regime`, `fear_greed`, `rsi_velocity`, `rsi_acceleration`, `momentum_velocity`, `cvd_velocity`
- Targets: `bet_won` (binary), `actual_change_pct` (regression)
- Metadata: `ai_model`, `strategy_name`, `open_time`

**Minimum data:** 200+ rows with `bet_won` for reliable training. 500+ recommended.

### Step 2a: Train XGBoost Judge (`train_xgboost_judge.py`)

The **primary ML gate** — filters AI predictions before betting. If P(win) < 55%, the bet is blocked (saved as ghost trade for data collection).

```bash
python -m ml.train_xgboost_judge
python -m ml.train_xgboost_judge --csv data/training_data_deepseek.csv
python -m ml.train_xgboost_judge --output models/ml_judge_model.pkl
```

**Output:** `models/ml_judge_model.pkl`

**What it reports:**
- Confusion matrix, classification report (precision/recall/F1)
- Feature importance ranking
- Target distribution (win/loss balance)

**Key metric:** Precision on wins > 55%. We optimize for avoiding false positives (betting on bad signals is worse than missing good ones).

**38 features** (must stay in sync with `engine.py:_JUDGE_FEATURES`):
`confidence, rsi, bb_pct, volatility, momentum, atr_pct, hour_utc, price_change_pct, trend_alignment, funding_rate, order_book_imbalance, liq_buy_vol_5m, liq_sell_vol_5m, reversal_alert, smc_bos_last, smc_choch_last, smc_in_kill_zone, smc_ob_distance_pct, consensus_agreement, bid_ask_spread, sniper_time_to_bet_sec, volatility_regime, fear_greed, cvd_imbalance_pct, cvd_aggressor_ratio, is_liquidity_swept_1m, mss_detected, post_sweep_reversal, gex_net_bias, gex_dist_call_wall_pct, gex_dist_put_wall_pct, polymarket_odds, expected_value, distance_to_strike, rsi_velocity, rsi_acceleration, momentum_velocity, cvd_velocity`

**Analytical column** (NOT in Judge, export-only): `api_tokens_used` — tracks API cost per prediction for ROI analysis.

**Categorical encoding:** `volatility_regime` (LOW=0, NORMAL=1, HIGH=2) is encoded automatically in both training and inference.

### Step 2b: Train RandomForest Oracle (`train_rf.py`)

Alternative/complementary classifier. Reads directly from SQLite (not CSV). Uses TimeSeriesSplit for temporal validation.

```bash
python -m ml.train_rf
python -m ml.train_rf --min-rows 100 --splits 5
```

**Output:** `models/oracle_rf.joblib`

### Step 2c: Train Regressor (`train_regressor.py`)

Predicts `actual_change_pct` (magnitude of price movement) for Kelly-optimal position sizing.

```bash
python -m ml.train_regressor
```

**Output:** `models/ml_regressor_model.pkl`

**Key metrics:** MAE, RMSE, R², direction accuracy, magnitude correlation.

### Step 3: Walk-Forward Validation (`train_walkforward.py`)

**Critical step** — validates that the model generalizes temporally. A random-split accuracy of 65% might be only 52% in walk-forward (temporal leakage).

```bash
python -m ml.train_walkforward
python -m ml.train_walkforward --train-days 7 --test-days 1
```

**Output:** `data/walkforward_results.csv`

**Interpretation:**
- Avg accuracy >= 55%: Model generalizes well
- Avg accuracy 50-55%: Marginal, needs more features or data
- Avg accuracy < 50%: Random-split accuracy is inflated — DO NOT deploy

### Step 4: Backtest Sniper (`backtest_sniper.py`)

Simulates the sniper consensus strategy on historical predictions.

```bash
python -m ml.backtest_sniper
python -m ml.backtest_sniper --min-confidence 60 --consensus-threshold 2
```

**Output:** `data/backtest_sniper_results.csv`

### Train LSTM Judge (`train_lstm_judge.py`)

PyTorch Bidirectional LSTM with attention for intra-candle microstructure prediction.

```bash
python -m ml.train_lstm_judge
```

**What it does:**
- `build_sequences_from_db()`: extracts raw `price_ticks` per window into fixed-length temporal sequences
- `_build_temporal_sequence()`: 5 features per timestep (price, volume, delta, CVD cumulative, volatility)
- BiLSTM + self-attention + dense head → binary classification
- Single temporal split (last 20%) for validation

**Output:** `models/lstm_judge.pt`

**Requirements:** PyTorch. Falls back to stub class if not installed.

**Key methods for inference:**
- `load_lstm_judge()` → loads model + normalization params
- `predict_with_lstm(sequence)` → P(win) probability

### Train HMM Regime Detector (`train_hmm_regime.py`)

Unsupervised GaussianHMM for market regime classification.

```bash
python -m ml.train_hmm_regime
```

**What it does:**
- `_load_5m_ohlc()`: loads Binance 5m klines from DB
- `build_feature_matrix()`: log returns, volatility, volume ratio, CVD directional vol proxy
- Trains 3-state HMM (trending, mean-reverting, volatile)
- `_label_states()`: assigns semantic labels based on state statistics

**Output:** `models/hmm_model.pkl` (HMMBundle dict via joblib)

**Consumed by:** `analysis/regime_hmm.py` (runtime Viterbi decode)

### Train RL Sizing Agent (`train_rl_agent.py`)

Stable-Baselines3 PPO/SAC for learned position sizing as alternative to Kelly.

```bash
python -m ml.train_rl_agent
```

**What it does:**
- `load_historical_decisions()`: loads past bet decisions from DB as training episodes
- `synthetic_decisions()`: generates synthetic data if insufficient history
- `TradingRiskEnv`: Gym environment with 13-dim observation space, continuous [0,1] action
- `evaluate_policy()`: compares RL agent vs Kelly baseline on held-out data

**Output:** `models/rl_risk_agent.zip`

**Consumed by:** `risk/rl_wrapper.py` (runtime inference)

---

## How the ML System Feeds Back Into the Engine

```
generate_dataset.py ──► CSV ──► train_xgboost_judge.py ──► models/ml_judge_model.pkl
                                                                    │
                                                                    ▼
                                                          engine.py loads at startup
                                                          _evaluate_oracle() gates bets
                                                                    │
                                                                    ▼
                                                          P(win) >= 55% → BET
                                                          P(win) < 55%  → FORCE EXPLORE ($1)
                                                                    │
                                                                    ▼
                                                          Explore trades collect more data
                                                          ──► next retrain cycle
```

**Feedback loop:** Ghost trades (blocked by the Judge) are still resolved and saved with outcomes. This means each retrain cycle has more data, including examples the previous model rejected — preventing the model from only seeing its own filtered output.

---

## When to Retrain

| Trigger | Action |
|---------|--------|
| 200+ new bets since last train | Full retrain pipeline |
| Walk-forward accuracy drops <52% | Retrain with more recent data |
| New features added to `_JUDGE_FEATURES` | Regenerate dataset + retrain |
| Win rate drops below 48% for 2+ days | Emergency retrain |
| After modifying `technical.py` or `smc_features.py` | Regenerate + retrain |

---

## Adding New Features to the Judge

1. Add column migration in `btc5min/database.py` `_run_migrations()`
2. Add column to `save_prediction()` INSERT in `database.py`
3. Populate the feature in `engine.py` `ml_features` dict
4. Add to `_JUDGE_FEATURES` list in `engine.py`
5. Add to `FEATURES` list in `ml/train_xgboost_judge.py`
6. Add to SQL query in `ml/generate_dataset.py`
7. If categorical: add encoding in both `train_xgboost_judge.py` and `engine.py:_evaluate_oracle()`
8. Regenerate dataset and retrain

---

## Common Issues

- **"CSV not found"**: Run `python -m ml.generate_dataset` first
- **"Model not loaded"**: Check `models/ml_judge_model.pkl` exists; restart `app.py`
- **Low accuracy**: Check walk-forward results — random-split may be misleading
- **Feature mismatch**: `_JUDGE_FEATURES` in `engine.py` must match `FEATURES` in training scripts exactly
- **NaN features**: Missing market data → features default to 0/median. Normal for first few windows after startup
