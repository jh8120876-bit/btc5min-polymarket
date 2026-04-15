# Risk — Position Sizing & Survival Systems

> Kelly Criterion bet sizing, trust-modified positions, daily loss limits.

## Architecture

```
risk/
├── risk.py       # RiskManager class — all sizing logic
└── rl_wrapper.py # SB3 PPO/SAC portfolio sizing agent (optional)
```

## RiskManager

Stateless utility class. All methods are `@staticmethod` reading from `dynamic_rules.json` via `config_manager`.

### Core Formula: Kelly Criterion

```
kelly = (b * p - q) / b
where:
  b = (100 / odds_cents) - 1   (decimal odds from Polymarket price)
  p = estimated win probability
  q = 1 - p
```

### Sizing Modifiers

| Factor | Effect |
|---|---|
| Trust score (10-95) | Multiplier: 0x (trust<20) to 1.5x (trust>80) |
| LOW_QUALITY signal | Capped at 0.1% of balance |
| Daily loss limit | 10% of portfolio — no bets after limit hit |
| Agent balance | Each agent has independent $100 portfolio |

### Key Method

`calculate_bet_size(signal, balance, trust_score, odds_cents) -> float`

Returns dollar amount to bet. Returns 0 if any safety check fails.

## Safety Systems (in engine.py, not here)

These use RiskManager but are implemented in the engine orchestrator:
- **Auto-Bailout**: Agent balance < $2 → reset to $100 + clear fatigue
- **Ghost trades**: Circuit Breaker (Black Swan) → bet blocked, saved as `is_ghost=1`
- **Trust gate**: Trust too low → recycled to `force_explore` ($1 flat bet), NOT ghost
- **Judge ML rejection**: Recycled to `force_explore` ($1 flat bet), NOT ghost

## rl_wrapper.py — RLPortfolioAgent (optional)

Stable-Baselines3 PPO/SAC wrapper for learned position sizing as alternative to Kelly.
- Loads model from `models/rl_risk_agent.zip`
- `build_observation()`: 13-dim vector (confidence, RSI, momentum, volatility, etc.)
- `action_to_bet()`: maps [0,1] continuous action to dollar amount with safety clamps
- Thread-safe singleton via `get_rl_agent()`
- Falls back to classic Kelly if model not found

## Dependencies

- **External**: None (stdlib only)
- **Internal**: `..models` (AISignal), `..config_manager` (rules), `..config`, `..database`
- **No dependency** on AI, analysis, data feeds, or sentiment

## Configuration (dynamic_rules.json → "risk" section)

All risk parameters are hot-reloadable via `config_manager`:
- `max_daily_loss_pct` — Daily loss limit (default 10%)
- `min_bet_usd` — Minimum bet size
- `max_bet_pct` — Maximum % of balance per bet
- `kelly_fraction` — Fractional Kelly (default 0.25 = quarter-Kelly)
