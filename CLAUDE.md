# CLAUDE.md

> **MANDATORY**: Read this file at session start. It is the single source of truth for architecture. Run `python -m scripts.generate_ai_context` for live DB state if needed.

## Running

```bash
pip install -r requirements.txt
python app.py          # Dashboard at http://localhost:5000
```

Primary AI provider is configured in `dynamic_rules.json` (agents section, `is_primary: true`).
API keys via `.env` or UI config panel. Optional: `LOCAL_LLM_URL` (default `http://localhost:1234/v1`).

---

## Project Structure

```
btc5min/
├── app.py                     # Entry point
├── btc5min/                   # Core runtime package
│   ├── config.py              # Constants, trading params (no hardcoded API keys)
│   ├── config_manager.py      # Hot-reload dynamic_rules.json
│   ├── models.py              # Dataclasses: PriceData, AISignal, WindowState, Bet, BetResult
│   ├── database.py            # SQLite3 WAL + background write-worker queue
│   ├── engine.py              # PredictionEngine — main coordinator
│   ├── routes.py              # Flask app factory + all API endpoints
│   ├── data_feeds/            # See data_feeds/CLAUDE.md
│   │   ├── price_feed.py      # Chainlink WS (primary) + Binance (fallback)
│   │   ├── binance_data.py    # Volume, funding, OI, liquidations, aggTrade CVD via REST+WS
│   │   ├── polymarket.py      # Gamma API + CLOB prices + WS live stream
│   │   └── options_data.py    # Deribit GEX polling (Call/Put Wall, Net GEX bias)
│   ├── ai/                    # See ai/CLAUDE.md
│   │   ├── ai_engine.py       # Provider-agnostic predictions, trust score, bias correction
│   │   ├── local_llm.py       # Stateless LM Studio/Ollama probe (utility only)
│   │   ├── swarm.py           # Multi-agent parallel predictions (incl. api_type=local)
│   │   ├── rag_db.py          # ChromaDB RAG — strategy retrieval
│   │   └── ingestar_rag.py    # Bulk-load strategies into ChromaDB
│   ├── analysis/              # See analysis/CLAUDE.md
│   │   ├── technical.py       # 22 indicators: RSI, EMA, MACD, BB, ATR, S/R
│   │   ├── smc_features.py    # Smart Money Concepts (FVG, OB, sweeps)
│   │   ├── feature_engineering.py # ML features: acceleration, vol z-score
│   │   ├── vendor_smc.py      # Vendored SMC library
│   │   └── (smc_features.py also provides compute_midwindow_mss for real-time sweeps)
│   ├── sentiment/             # See sentiment/CLAUDE.md
│   │   └── sentiment.py       # Fear & Greed + RSS + circuit breaker
│   ├── execution/             # See execution/CLAUDE.md
│   │   ├── clob_executor.py   # ClobExecutor — py_clob_client.ClobClient wrapper (native Polymarket L2)
│   │   ├── web3_auth.py       # ClobClient singleton, signature_type=1 (POLY_PROXY), create_or_derive_api_creds
│   │   ├── ladder.py          # Maker limit-order escalator (3 rungs, configurable BPS)
│   │   ├── ptb_filter.py      # Price-To-Beat strike filter (ATR multiples + $20 hard block)
│   │   ├── survival.py        # SurvivalMonitor — Profit-Lock (≥98%) & Flip-Stop (≤30%)
│   │   └── relayer.py         # DEPRECATED no-op stub (py-clob-client has no auto_redeem)
│   ├── observability/         # See observability/CLAUDE.md
│   │   └── prometheus_exporter.py  # 12 metrics on :9108 (lazy init, no-op when disabled)
│   ├── risk/                  # See risk/CLAUDE.md
│   │   └── risk.py            # Kelly Criterion, daily limits, trust sizing
│   └── strategies/            # See strategies/CLAUDE.md
│       ├── loader.py          # File-based prompt & strategy loader
│       ├── prompts/system/    # System prompts (.txt per agent)
│       ├── prompts/fatigue/   # Fatigue rotation prompts (.txt ordered)
│       └── rag/               # RAG strategies (.json per strategy, 19+)
├── ml/                        # ML training & evaluation (see ml/CLAUDE.md)
│   ├── generate_dataset.py    # Export training CSV from SQLite
│   ├── train_xgboost_judge.py # XGBoost Judge — primary bet filter
│   ├── train_rf.py            # RandomForest Oracle (alternative)
│   ├── train_regressor.py     # Regression model for bet sizing
│   ├── train_walkforward.py   # Walk-forward temporal validation
│   ├── backtest_sniper.py     # Sniper consensus backtester
│   ├── train_lstm_judge.py    # PyTorch LSTM temporal judge (intra-candle sequences)
│   ├── train_hmm_regime.py    # GaussianHMM unsupervised regime detector
│   └── train_rl_agent.py      # SB3 PPO/SAC position sizing RL agent
├── scripts/                   # DB maintenance & utilities
│   ├── clean_db.py            # Deduplicate predictions/bets tables
│   ├── clean_neutral.py       # Remove non-binary predictions
│   ├── generate_ai_context.py # Generate AI_CONTEXT_NOW.md snapshot
│   ├── coliseum_evaluate.py   # Weekly leaderboard: winrate/sharpe/PF per agent×strategy
│   └── coliseum_evolve.py     # Auto-promote winning prompts, replace losers (≥55% gate)
├── models/                    # Trained model artifacts (.pkl, .joblib)
├── data/                      # Training CSVs, validation results
├── templates/                 # HTML (dashboard, history, brain, config)
├── dynamic_rules.json         # Hot-reloadable: prompts, risk, agents, fatigue
└── memory/                    # ChromaDB vector store for RAG
```

**Each subsystem has its own `CLAUDE.md`** — read the relevant one before modifying that subsystem.

---

## Data & Execution Pipeline

### Zero-Drift Tick Loop (`engine.py:301-334`)
- 2s tick interval with **absolute clock sync**: `now % 300` computes drift to next 5-min boundary
- When boundary is imminent (<2.5s away), sleeps exactly to `boundary + 0.01s` instead of fixed 2s
- Result: windows open precisely at :00, :05, :10, :15... UTC with <50ms jitter

### Price Oracle Hierarchy
1. **Chainlink BTC/USD** (primary) — WebSocket via Polymarket RTDS (`wss://ws-live-data.polymarket.com`)
2. **Binance spot** (fallback) — kicks in after 30s Chainlink staleness
3. Resolution oracle: Chainlink (matches Polymarket's UMA resolution source). Binance fallback if Chainlink unavailable at resolve time.

### Polymarket CLOB Integration (`polymarket.py`)
- **REST**: Gamma API slug lookup `btc-updown-5m-{window_start_ts}` -> event -> clobTokenIds
- **CLOB executable price**: `/price?side=BUY` (best_ask) overrides Gamma indicative prices. Fallback: `/book` endpoint
- **WebSocket**: `wss://ws-subscriptions-clob.polymarket.com/ws/market` — subscribes to UP/DOWN token IDs per window, receives `price_change`, `last_trade_price`, `book` events
- Odds: prefer live WS prices > internal formula (price diff + AI confidence)
- Spread simulation: when no CLOB data, adds 2c penalty to internal odds

### Order Flow / CVD (`binance_data.py`)
- **aggTrade WebSocket**: `wss://stream.binance.com:9443/ws/btcusdt@aggTrade`
- Tick-level Market Buy vs Sell classification via `m` (buyer_is_maker) flag
- `get_cvd_summary(minutes)` returns imbalance_pct, net delta, aggressor ratio
- `get_cvd_intracandle_series()` returns time-bucketed CVD for LSTM temporal features
- CVD accumulator resets per window via `reset_cvd_window()`

### Gamma Exposure / GEX (`options_data.py`)
- Deribit public API polling every 5min for BTC option book summaries
- Computes Call Wall (max call OI strike), Put Wall (max put OI strike)
- Net GEX Bias: normalized call_gamma - put_gamma [-100..+100]
- GEX regimes: LONG_GAMMA (mean-reverting), SHORT_GAMMA (trending), NEUTRAL

### MSS & Liquidity Sweeps (`smc_features.py:compute_midwindow_mss`)
- Real-time mid-window detection of liquidity sweeps (swing high/low pierced)
- Market Structure Shift (MSS) via recent CHoCH detection
- Post-sweep reversal confirmation flag for sniper high-probability entries
- Daily level piercing (session high/low breach)
- Refreshed every sniper evaluation cycle (~60s)

### Epistemic Reflection (`ai_engine.py:trigger_reflection`)
- After every LOST window, async DeepSeek call diagnoses root cause
- Extracts causal label (Absorcion Pasiva, Divergencia CVD, etc.)
- Injected as `recent_reflection_lesson` into next prediction prompt
- One-shot consumption: cleared after use, persisted as learned_lesson

### P&L Model (`polymarket.py:calc_polymarket_pnl`)
- Buy outcome tokens at market price: tokens = amount / price
- Win: each token pays $1.00, net profit minus 2% fee on winnings only
- Loss: tokens worth $0, full loss, no fee

### Execution Modes (`config.py:TRADING_MODE`)
- Backbone: **official `py-clob-client`** — native Polymarket L2 CLOB (Polygon) integration, direct contract-level signing, no market pre-import cap
- `offline_sim` (default): Legacy SQLite simulation, no real orders
- `paper_simmer`: legacy label, now aliased to offline_sim (Simmer backend deprecated)
- `live_mainnet`: real USDC on Polymarket CLOB. Requires `WALLET_PRIVATE_KEY` + `POLYMARKET_PROXY_FUNDER`
- **Delegated proxy wallet flow**: `ClobClient(host, key, chain_id=137, signature_type=1, funder=PROXY_FUNDER)` — required for Google/Email login accounts
- Orders: `create_market_order`+`OrderType.FOK` / `create_order`+`OrderType.GTC` → `post_order` returns native Polymarket hash (0x...)
- Token routing: engine passes `poly_quote.up_token_id` / `poly_quote.down_token_id` to the executor — NEVER `condition_id`
- Fail-safe: missing SDK / keys / funder / cred derivation errors all degrade to offline_sim + WARN

### PTB Filter (`execution/ptb_filter.py`)
- Extracts strike price from Polymarket market (Gamma variant/regex)
- Blocks bets where strike-spot gap > 3× ATR or > $20 absolute
- Penalizes (0.5× sizing) when gap > 1.5× ATR
- Hot-reloadable thresholds via `dynamic_rules.json:ptb_filter`

### Maker Ladder (`execution/ladder.py`)
- Replaces taker orders with 3 limit rungs at [200, 500, 800] BPS spread
- Weights [0.3, 0.3, 0.4], TTL 60s, post-only GTC
- Degrades to single rung if budget < min_per_rung × rungs

### Survival Shields (`execution/survival.py`)
- Daemon thread polling every 500ms via PolymarketLiveStream WS prices
- **Profit-Lock**: token prob ≥ 98% → sell to lock profit (min $0.50, min 10s remaining)
- **Flip-Stop**: token prob ≤ 30% (was favorite ≥ 60%) → bail to recover partial

### Redeem Relayer (`execution/relayer.py`) — DEPRECATED STUB
- `py-clob-client` has no `auto_redeem()` equivalent (redemption is a raw on-chain CTF call)
- Relayer is now a no-op stub, logs a one-time warning on start()
- Manual workaround: polymarket.com → Portfolio → Claim winnings (same delegated wallet)

---

## Multi-Agent Swarm Ecosystem

### Agent Architecture
- **Primary**: configurable via UI/dynamic_rules.json (`is_primary: true`). Provider-agnostic (DeepSeek/Anthropic/OpenAI/Local). Rate-limited via `Semaphore(2)` (skipped for `api_type=local`)
- **Swarm agents** (`swarm.py`): configurable in `dynamic_rules.json` `agents` section. Support: DeepSeek, Anthropic, OpenAI, Local (LM Studio/Ollama). Run in parallel threads with `SWARM_JOIN_TIMEOUT=30s`
- **Local models** are just another `api_type` — no second door, no stateful toggle, no parallel prediction path. `local_llm.py` is only a stateless model-listing helper for the UI dropdowns + Live Observer
- Each agent has independent portfolio (`agent_portfolios` table), independent bets (shadow bets), independent fatigue state

### Dual Fatigue/Rotation Motor

**DeepSeek (RAG Strategy Rotation)**:
- Uses ChromaDB RAG (`rag_db.py`) with `paraphrase-multilingual-MiniLM-L12-v2` embeddings
- Retrieves best institutional strategy (SMC, Wyckoff, Elliott, etc.) by cosine similarity to market context
- **Blacklist on fatigue**: 7 consecutive losses OR 2h age -> strategy blacklisted for 2h cooldown
- Blacklisted strategies excluded from retrieval. 19 strategies rotate via exclusion, not substitution
- RAG block injected into DeepSeek prompt with conditional adoption rules

**Swarm Agents (System Prompt Rotation)**:
- Fatigue triggers: 7 consecutive losses OR 2h age -> agent enters "fatigued" state
- **Prompt override**: base system prompt replaced by rotating fatigue prompts:
  - `#0` Reversion Extrema (RSI/BB extremes)
  - `#1` Contrarian Total (inverse momentum/funding)
  - `#2` Momentum Puro (follow trend, ignore reversion)
- `rotation_idx` cycles through prompts. Win resets fatigue. Cooldown auto-resets after 1h
- Labels shown in UI as personality badges

### Sniper Consensus — Score-based (`engine.py:_run_sniper_evaluation`)
- First eval auto-registered at T+0 from `_fetch_and_process_prediction`; next eval after `first_eval_delay_sec` (25s default)
- Re-evaluates every `eval_interval_sec` (15s default) via `ai.reevaluate_for_bet()` — pure TA recalc, no API cost
- **Score-based consensus** (`_check_sniper_consensus`): seeds `_consensus_direction` from first tracked eval (score=1); +1 per confirming, −1 per contradicting; score ≤ `consensus_score_flip` (−2) → FLIP direction, reset score to 1; FIRE when `score ≥ consensus_score_fire` (2) AND `elapsed ≥ min_consensus_age_sec` (20s)
- **Confidence decay**: time-based `confidence_decay_per_min` (2%/min) vs elapsed — decoupled from eval count. Floor at `decay_floor_conf` (51%). Applied to runtime sizing only; DB `save_prediction` still stores the raw confidence at T+0
- **Trigger variants** (stamped in `_last_trigger_type`): `EARLY-FLASH` (EV > `ev_early_fire_threshold` 1.30 at eval #1), `FLASH-FIRE` (EV > `ev_flash_fire_threshold` 1.18), `CONSENSUS` (score-based), `RESCUE` (`_try_last_minute_rescue` at cutoff: best eval in consensus dir with `conf ≥ last_minute_rescue_conf` 60% and `ev ≥ last_minute_rescue_ev` 1.05), `OVERRIDE` (Meta-Swarm unanimous)
- Time-stop order at `eval_cutoff_sec` (235s): Last-Minute Rescue → Meta-Swarm Override → SKIP
- FLIP detection on the full `_window_evaluations` series still syncs DB via `update_prediction_direction()` for ML data integrity
- Config hot-reloadable via `dynamic_rules.json:sniper`. Schema registered in `config_manager._FIELD_SCHEMA`

### Second Opinion — Tactical LLM Consults (`ai_engine.py:quick_second_opinion`)
- 4 checkpoints layered on top of pure-TA sniper loop. Always uses primary agent via `_call_primary_api`. Shared cache `_second_opinion` (TTL 60s, window-scoped). Config: `dynamic_rules.json:second_opinion`
- **Variant D** (background @ T+120s): daemon thread, non-blocking. Updates `signal.confidence` via **Option Y clamp**: `new_conf = min(llm_conf, pre_decay_cap)` where `pre_decay_cap = original_confidence − elapsed_decay`. Time-decay floor always respected
- **Variant B** (flip async): fires when `_run_sniper_evaluation` detects prediction flip. Async — next eval consumes fresh opinion
- **Variant C** (Flash Fire sanity check, sync + veto): before committing `FLASH-FIRE`/`EARLY-FLASH`, releases `_lock` for ≤3s LLM call. If `veto=true`, cancels the flash and falls back to score-consensus path
- **Variant A** (rescue sync + veto): inside `_try_last_minute_rescue`, same lock-release pattern. If veto → returns False → caller falls through to Meta-Swarm Override. If LLM agrees with direction, clamps `best["confidence"] = min(llm_conf, best_conf)`
- **Semaphore**: reuses `swarm._api_semaphore` (2 slots) with 2s acquire timeout — if busy, 2nd-opinion degrades gracefully (no blocking, no exception)
- **Window-rolling safety**: each op is tagged with `window_id`; callers drop stale results if the window rolled during the call
- **Prompt key**: `prompts.second_opinion_system` in `dynamic_rules.json`. Response shape: `{"direction": "UP|DOWN", "confidence": 10-95, "veto": bool, "reason": "..."}`
- **Telemetry**: exposed in `/api/state` under `sniper.second_opinion` (source, direction, confidence, veto, age_sec, latency_ms, in_flight). Dashboard renders as `2nd[X] UP 68%` or `2nd[C] ⛔ VETO` suffix in the sniper alert line
- **Toggles**: `second_opinion.enabled` (master) and `variant_{a,b,c,d}_enabled` for granular control. Disabled = identical to pre-refactor sniper behavior

### Meta-Swarm Consensus Override (`engine.py:_check_meta_swarm_override`)
- If primary vetoes (time-stop, EV ghosting), checks enabled secondaries for unanimous override
- Requires ALL enabled secondary agents armed, same direction, confidence >= `min_confidence` (70%)
- Constructs synthetic AISignal with avg confidence, fires `_place_auto_bet`
- Only counts agents with `enabled: true` in `dynamic_rules.json` — inactive agents ignored
- Config: `dynamic_rules.json:meta_swarm` (enabled, min_confidence, min_agents)
- Logs with `[Meta-Swarm]` prefix

### Swarm Rides the Primary Scope (`engine.py:_broadcast_swarm_fire`)
- `_run_swarm_agents` does NOT bet at T+0 — it only arms `_swarm_signals[aid]` with `signal_obj`+`strategy_label`+`initial_balance`+`agent_cfg_display`+`armed_at_ts`+`fired=False` ("bullet in chamber"). Agents with `api_type: "local"` go through this same path — there is no parallel local-LLM door
- `_place_auto_bet` tail-calls `_broadcast_swarm_fire(poly_quote, is_early, primary_ghosted, force_explore)` — every armed agent (including local ones) fires with SAME CLOB snapshot + `is_early` as primary
- Per-side odds computed from shared `poly_quote` (agents may predict opposite sides)
- 3 broadcast modes: `NORMAL` (per-agent Kelly w/ shared odds), `GHOST_HARVEST` (primary ghosted → swarm ghost shadows for ML), `EXPLORE` (all fire flat $1 w/ `|explore` suffix)
- Resolve routing: `_resolve_swarm_bets` resolves every non-primary agent bet (local agents included), handling ghost shadows with `if not is_ghost` balance guard

### Forced Exploration (`_place_auto_bet(..., force_explore=False)`)
- Judge ML rejection (`self._ml_skipped`) → recycled to `force_explore=True` instead of pure ghost (see sniper `_run_sniper_evaluation`)
- Trust gate low → flips `force_explore=True` in-place (no more trust-ghost)
- `force_explore=True`: bypasses Kelly entirely, $1.00 flat, `smc_liquidity|explore` strategy label, entire swarm also fires $1 `|explore`
- Circuit Breaker = HARD stop (Black Swan not recyclable); swarm still gets ghost broadcast for data harvest

### ML Oracle Gate (`engine.py:939-971`)
- Optional RandomForest model loaded from `models/oracle_rf.joblib`
- If P(correct) < threshold → rejection recycled to `force_explore=True` (was: pure ghost)
- Features: all TA + ML features + RAG strategy + SMC features

---

## Risk & Survival Systems

### Kelly Criterion (`risk.py`)
- Formula: `(b * p - q) / b` where `b = (100/odds) - 1`
- Trust score (10-95, persists in SQLite) modifies sizing: 0x to 1.5x multiplier
- LOW_QUALITY signals capped at 0.1% bet. Daily loss limit 10%

### Auto-Bailout / Banco Central (`engine.py:1380-1406`)
- After every window resolve, scans all `agent_portfolios`
- If any agent balance < $2.00 -> reset to $100.00
- Also resets fatigue state (losses=0, fatigued=false) — clean restart

### Circuit Breakers
- **Primary AI Circuit Breaker**: HTTP 402/500/503 or timeout -> `_primary_halted=True`, predictions paused. State exposed as `primary_halted`/`primary_halt_reason` in `/api/state`. Resume via `/api/resume_primary`
- **News Circuit Breaker** (`sentiment.py`): Black Swan events block real bets (paper mode: warning only)
- **Liquidation chaos**: Binance forceOrder WS. Spike >3x = reversal alert, >5x = skip bet
- **Trust gate**: trust score too low -> ghost trade instead of real bet

### Ghost Trades
- Circuit Breaker (Black Swan, non-paper mode) and swarm-agent-own-Kelly-blocks still save `is_ghost=1`
- Judge ML + Trust + primary Kelly rejections now recycle to `force_explore=$1` instead (see Forced Exploration)
- `_resolve_swarm_bets` now resolves ghost shadows too (outcome stamped for ML; balance untouched via guard)

---

## Data Schema (SQLite3 WAL)

### Core Tables
- `price_ticks` — (timestamp, price, source). Write-worker queue batches inserts
- `windows` — (window_id, open/close time/price, outcome, target_magnitude)
- `predictions` — per-agent per-window: direction, confidence, 22 TA indicators, ML features, RAG strategy, SMC features, ML oracle prob
- `bets` — per-agent per-window: side, amount, odds_cents, P&L, is_shadow, is_ghost, ai_model, strategy_name, odds_source, order_id, fill_price, fill_size, exec_status, redeemed_at, exit_reason, exit_price
- `agent_portfolios` — per-agent: balance, wins/losses, total_profit, peak_balance
- `ai_memory` — key-value store for trust score, lessons, config persistence
- `market_context` — per-window: volume, funding, OI, spread
- `liquidations` — Binance forceOrder events

### Key Relationships
- `predictions.window_id` + `predictions.ai_model` -> unique prediction per agent per window
- `bets.window_id` + `bets.ai_model` -> unique bet per agent per window
- `bets.is_shadow=1` for swarm bets (all non-primary agents, including local), `is_ghost=1` for risk-blocked intentions
- `predictions.rag_strategy` stores RAG strategy name (DeepSeek) or fatigue rotation label (agents)

---

## UI: History Master Table (`history.html`)

### Unified View
- Shows ALL bets across ALL agents in one table
- Filters: TODOS | REALES | GHOST | SHADOW
- Stats row: total trades, PnL, win rate, ghost count, ML readiness

### Badge System
- **Agent badge** (`badge-agent`): color-coded by `api_type` (DeepSeek=blue, Anthropic=cyan, OpenAI=yellow, Local=green)
- **Strategy badge** (below agent name):
  - Primary (`badge-primary`): shows RAG strategy name or "SMC Base"
  - Swarm (`badge-persona`): shows fatigue rotation label or "Base"
- **Type badges**: `badge-real`, `badge-ghost`, `badge-shadow`
- **Result badges**: `badge-win` (W), `badge-loss` (L)
- **ML badge** (`badge-ml`): shows Oracle P(correct) probability

### Other Pages
- `dashboard.html` — live prices, signals, active bets, "AHORA MISMO" banner
- `brain.html` — AI brain state viewer
- `config.html` — agent config, LM Studio settings

---

## Security & Performance (Megarefactor B)

### API Authentication (`routes.py`)
- All POST endpoints protected by `@_require_api_auth` decorator
- Key via `BTC5MIN_API_KEY` env var. If unset, auth disabled (backward compat)
- Thread-safe `dynamic_rules.json` R/W via `_rules_file_lock` (threading.Lock)

### HFT 2-Phase Prediction (`ai_engine.py` + `swarm.py`)
- **Phase 1**: `call_fast_prediction()` — max_tokens=20, timeout=8s, temperature=0.05 → instant UP/DOWN + confidence
- **Phase 2**: `call_background_reasoning()` — background thread, full reasoning → `db.update_prediction_reasoning()` backfill
- Fallback: if Phase 1 fails, degrades to full traditional API call

### DB Performance (`database.py`)
- 12 composite indexes on predictions/bets/windows for query acceleration
- `update_prediction_reasoning()` for HFT Phase 2 backfill

### Engine Lock Reduction (`engine.py`)
- TA analysis + HMM regime decode moved outside `self._lock` ("Phase 1.5")
- Deferred boolean flags (`_need_ta_update`, `_need_regime_decode`) set inside lock, executed after release

### SMC Cache (`smc_features.py`)
- 60-second fingerprint-based cache: `(len(ohlcv), round(last_close, 2))`
- Prevents O(N²) recomputation on every 2s tick

### Toxic Spread Detection (`engine.py`)
- Spread > `ladder.toxic_spread_cents` (default 5c) → forces Maker Ladder even on explore, sizing penalized × `toxic_spread_sizing_penalty` (0.5)

### ML Temporal Integrity (`ml/train_xgboost_judge.py`, `ml/train_regressor.py`)
- `TimeSeriesSplit` (5 folds) replaces `train_test_split` — eliminates data leakage
- Per-fold metrics reported, final model retrained on ALL data

### Frontend (`templates/*.html`)
- Poll intervals: dashboard 3s (was 1.5s), agents 30s
- All `setInterval` tracked + cleanup on `beforeunload` / `visibilitychange`

### Black Swan Regex (`sentiment.py`)
- Word-boundary `\b` regex replaces substring matching (prevents "ban" matching "urban")

---

## Adding New Features

- **New indicator**: `technical.py` -> `analyze()` return dict -> include in `ai_engine.py:_build_prompt()` and `_fallback()`
- **New API endpoint**: `routes.py` inside `create_app()`
- **New data model**: `models.py` -> expose in `engine.py:get_state()`
- **New agent**: add to `dynamic_rules.json` `agents` section with `api_type`, `api_key_env`, `system_prompt_key`
- **New RAG strategy**: add to ChromaDB via `ingestar_rag.py`
- **New ML Judge feature**: see `ml/CLAUDE.md` "Adding New Features to the Judge" section
- **ML retrain**: `python -m ml.generate_dataset && python -m ml.train_xgboost_judge && python -m ml.train_walkforward`
- **LSTM train**: `python -m ml.train_lstm_judge` (requires PyTorch, uses price_ticks intra-candle sequences)
- **Config changes**: `config.py` (static) or `dynamic_rules.json` (hot-reloadable via `config_manager.py`)

---

## Core Directive: Memory Protocol

1. **READ THIS FILE** at session start. It is the single source of truth for architecture.
2. **OBLIGATORIO**: If you modify architecture, DB schema, agents, data flow, or add/remove modules, you **MUST** update this document before finishing the task.
3. **Limit strict**: Keep this file under 1500 tokens of dense information. Use bullet points, not prose. Zero source code blocks longer than 3 lines.
4. **Verify before assuming**: This document may lag behind code changes. When in doubt, read the actual source file.
