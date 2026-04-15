# Strategies — System Prompts & RAG Strategy Library

> File-based store for AI system prompts and institutional trading strategies.
> Add new content by dropping files — no code changes needed.

## Architecture

```
strategies/
├── __init__.py               # Re-exports from loader
├── loader.py                 # StrategyLoader: reads .txt/.json files
├── prompts/
│   ├── system/               # System prompts (1 per agent type)
│   │   ├── deepseek_system.txt
│   │   ├── claude_opus_mean_reversion_system.txt
│   │   ├── chatgpt_momentum_system.txt
│   │   ├── local_llm_smc_system.txt
│   │   ├── local_llm_mean_reversion_system.txt
│   │   └── local_llm_post_analysis_system.txt
│   └── fatigue/              # Fatigue rotation prompts (ordered by filename)
│       ├── 00_reversion_extrema.txt
│       ├── 01_contrarian_total.txt
│       └── 02_momentum_puro.txt
└── rag/                      # RAG strategies (1 JSON per strategy)
    ├── ict_liquidity_sweep_and_choch.json
    ├── 5m_smc_order_block_and_fvg_mitigation.json
    ├── wyckoff_accumulation_spring_5m.json
    └── ... (19 files)
```

## How to Add a New System Prompt

1. Create a `.txt` file in `prompts/system/` with the prompt key as filename:
   ```
   prompts/system/my_new_agent_system.txt
   ```
2. Write the full system prompt in plain text (Spanish or English)
3. Reference the key in `dynamic_rules.json` agent config:
   ```json
   "my_agent": {
     "system_prompt_key": "my_new_agent_system",
     ...
   }
   ```
4. The prompt is loaded automatically — no code changes needed

**Priority**: `dynamic_rules.json` overrides file-based prompts. If the prompt key exists in both, JSON wins. This allows hot-reload overrides.

## How to Add a New Fatigue Prompt

1. Create a `.txt` file in `prompts/fatigue/` with a numbered prefix:
   ```
   prompts/fatigue/03_volume_divergence.txt
   ```
2. The prefix (`03_`) determines rotation order (sorted alphabetically)
3. Write the fatigue prompt. Must include:
   - Alert header explaining the strategy failed
   - Clear rules for the alternative approach
   - JSON response format: `{"prediction":"UP","confidence":51-95,"reasoning":"..."}`

**Note**: `dynamic_rules.json → fatigue_prompts.rotation[]` takes priority over files.

## How to Add a New RAG Strategy

1. Create a `.json` file in `rag/` with a descriptive slug:
   ```
   rag/harmonic_pattern_gartley_5m.json
   ```

2. Use this exact format:
   ```json
   {
     "strategy_name": "Harmonic Pattern Gartley (5m)",
     "concept_family": "Harmonic | Fibonacci",
     "optimal_context": {
       "sentiment": "neutral",
       "volatility": "medium",
       "market_phase": "reversal"
     },
     "entry_rules": [
       "Rule 1: identify the pattern...",
       "Rule 2: confirm with volume...",
       "Rule 3: entry trigger..."
     ],
     "invalidation_condition": "When the pattern breaks..."
   }
   ```

3. Run re-ingestion to load into ChromaDB:
   ```bash
   python -m btc5min.ai.ingestar_rag
   ```

### Required Fields

| Field | Type | Description |
|---|---|---|
| `strategy_name` | string | Human-readable name (must be unique) |
| `concept_family` | string | Category: SMC, Wyckoff, Elliott Wave, etc. |
| `entry_rules` | list[str] | Step-by-step entry conditions |

### Optional Fields

| Field | Type | Description |
|---|---|---|
| `optimal_context.sentiment` | string | fear / neutral / greed / trending |
| `optimal_context.volatility` | string | low / medium / medium-high / high |
| `optimal_context.market_phase` | string | ranging / trending / reversal / ranging_to_trend |
| `invalidation_condition` | string | When the trade setup becomes invalid |

### How RAG Selection Works

1. Engine builds context: `"Fear&Greed=70(Neutral) | [news headlines]"`
2. ChromaDB computes cosine similarity between context and all strategy documents
3. Best-matching non-blacklisted strategy is selected
4. Strategy `entry_rules` injected into AI prompt as evaluation criteria
5. AI must check if current market meets the rules → adopt or discard

### Strategy Families

| Family | Count | Description |
|---|---|---|
| SMC / ICT | 10 | Liquidity sweeps, order blocks, FVG, breaker blocks |
| Wyckoff | 2 | Accumulation spring, distribution UTAD |
| Elliott Wave | 2 | Wave 3 entry, ABC correction |
| Volume Profile | 2 | Value area rejection, LVN breakout |
| Mean Reversion | 1 | Bollinger Band reversion |
| Order Flow | 1 | Delta divergence |
| Liquidation | 1 | Cascade reversal |

## Dependencies

- **External**: None (stdlib: `json`, `pathlib`)
- **Internal**: None — this is a pure data loader
- **Consumed by**: `config_manager.py` (prompts), `ai/ingestar_rag.py` (strategies), `engine.py` (fatigue)

## Loader API

```python
from btc5min.strategies import (
    load_system_prompt,       # key -> str | None
    load_all_system_prompts,  # -> dict[str, str]
    load_fatigue_prompts,     # -> list[str] (ordered)
    load_rag_strategies,      # -> list[dict]
    list_rag_files,           # -> list[str] (filenames)
    list_prompt_files,        # -> dict[str, list[str]]
)
```
