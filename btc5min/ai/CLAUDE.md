# AI — Prediction Engines & Multi-Agent System

> All AI prediction logic: DeepSeek primary engine, Local LLM, swarm agents, RAG strategy retrieval.

## Architecture

```
ai/
├── ai_engine.py      # Primary predictions (provider-agnostic), trust score, bias
├── local_llm.py      # Stateless LM Studio / Ollama probe + model listing
├── swarm.py          # Multi-agent parallel predictions (DeepSeek/Anthropic/OpenAI/Local)
├── rag_db.py         # ChromaDB RAG — institutional strategy retrieval
└── ingestar_rag.py   # Bulk-load institutional strategies into ChromaDB
```

## ai_engine.py — AIEngine (Primary)

The main prediction brain. Provider-agnostic — reads primary agent config from `dynamic_rules.json`.

**Key responsibilities:**
- `_call_primary_api()`: generic API dispatch via `swarm._API_CALLERS` (DeepSeek/Anthropic/OpenAI/Local)
- Build market analysis prompt from TA indicators + market context
- Parse AI response into `AISignal` (direction, confidence, reasoning)
- `get_full_context_prompt()`: shared RAG+lessons+HMM prompt for swarm agents
- Manage trust score (10-95, persists in SQLite)
- Bias correction: track directional bias, adjust signals
- `volatility_regime` classification: LOW/NORMAL/HIGH based on ATR
- `reevaluate_for_bet()`: recalculate TA without API call (sniper consensus)

**Rate limiting**: `Semaphore(2)` shared with swarm agents (skipped for local)

**Dependencies**: `..config`, `..models`, `..database`, `..config_manager`, `.swarm`

## local_llm.py — LocalLLM (stateless utility)

Pure helper for probing an OpenAI-compatible local server (LM Studio / Ollama / any
compatible runtime) at `LOCAL_LLM_URL`. Exposes:

- `is_available(url=None) -> bool` (30s TTL cache)
- `list_models(url=None) -> list[dict]` — powers the UI dropdowns
- `detect_runtime(url=None) -> str` (`ollama` / `lmstudio` / `compatible`)

**No state, no predictions, no portfolio.** Local models now run as first-class
agents via `dynamic_rules.json` with `api_type: "local"` — dispatched through
`swarm._API_CALLERS["local"]` exactly like DeepSeek/Anthropic/OpenAI. The engine
no longer runs a parallel "second door" for local LLM, and there is no enable/disable
flag in the DB. The only surviving direct consumer is the 15-second Live Observer,
which picks any local agent found in dynamic_rules (primary or swarm) to generate
free commentary via a short `/chat/completions` call.

## swarm.py — Multi-Agent Swarm

Parallel execution of multiple AI agents.

**Architecture:**
- Agents configured in `dynamic_rules.json` → `agents` section
- Support: DeepSeek, Anthropic, OpenAI, Local (`api_type: "local"`) API types
- `ThreadPoolExecutor` runs all agents in parallel
- `SWARM_JOIN_TIMEOUT=30s` hard ceiling
- Each agent: independent portfolio, independent shadow bets, independent fatigue

**Key function**: `run_swarm(prompt, market_data) -> dict[agent_id, AISignal]`

**Shared rate limiter**: `_api_semaphore` set by engine.py at startup

## rag_db.py — StrategyMemory (RAG)

ChromaDB vector store for institutional trading strategies.

**Embedding model**: `paraphrase-multilingual-MiniLM-L12-v2`
**Collection**: `trading_strategies_v2`

**Key methods:**
- `query(market_context) -> best_strategy` — cosine similarity retrieval
- `upsert(name, text, metadata)` — idempotent by strategy name
- `blacklist(name, duration)` — exclude fatigued strategies from retrieval

**Blacklist on fatigue**: 7 consecutive losses OR 2h age → strategy blacklisted for 2h cooldown

## ingestar_rag.py — Strategy Bulk Loader

Loads 19 institutional strategies (SMC, Wyckoff, Elliott, ICT, Volume Profile, etc.) into ChromaDB.

**Usage**: `python -m btc5min.ai.ingestar_rag`

Idempotent — uses upsert, safe to run multiple times.

## Dual Fatigue Motor

| Agent Type | Fatigue Trigger | Fatigue Action | Reset |
|---|---|---|---|
| Primary (DeepSeek/Anthropic/…) | 7 losses OR 2h | Blacklist RAG strategy for 2h | Win resets. Cooldown auto-resets 1h |
| Swarm agents (incl. api_type=local) | 7 losses OR 2h | Rotate system prompt (3 personas) | Win resets. Cooldown auto-resets 1h |

## Dependencies

- **External**: `requests`, `chromadb` (optional), `sentence-transformers` (optional)
- **Internal**: `..config`, `..models`, `..database`, `..config_manager`
- **No dependency** on data_feeds, analysis, risk, or sentiment

## Second Opinion — ReAct Variant (`ai_engine.py:quick_second_opinion_react`)

Advanced second opinion using a ReAct reasoning loop with tool access:
- `tool_broker` callback allows the LLM to request specific tools mid-reasoning
- Tools available: `get_live_ta`, `get_cvd_snapshot`, `get_gex_snapshot`, `get_order_book`
- Multi-turn: LLM reasons → requests tool → receives data → continues reasoning
- Used for higher-fidelity sanity checks when compute budget allows
- Falls back to standard `quick_second_opinion` on error or timeout

## HFT 2-Phase Prediction (`swarm.py`)

- **Phase 1**: `call_fast_prediction()` — max_tokens=20, timeout=8s, temperature=0.05 → instant UP/DOWN + confidence
- **Phase 2**: `call_background_reasoning()` — background thread, full reasoning → `db.update_prediction_reasoning()` backfill
- Fallback: if Phase 1 fails, degrades to full traditional API call

## Adding a New Agent

1. Add entry in `dynamic_rules.json` → `agents` section
2. Set `api_type` (deepseek/anthropic/openai), `api_key_env`, `system_prompt_key`
3. Agent auto-discovers and runs in next swarm cycle
4. Portfolio auto-created in `agent_portfolios` table on first bet
