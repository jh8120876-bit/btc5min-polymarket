# Sentiment — Market Sentiment & Circuit Breaker

> Fear & Greed Index, RSS news headlines, and Black Swan circuit breaker.

## Architecture

```
sentiment/
└── sentiment.py   # All sentiment logic in one module
```

## Data Sources (All Free, No API Keys)

1. **Alternative.me Fear & Greed Index** — polled every 1 hour
2. **CoinDesk / CoinTelegraph RSS feeds** — polled every 3-5 minutes

## Key Functions

### Market Data
- `get_fear_greed() -> dict` — Latest Fear & Greed value (0-100) and classification
- `get_headlines() -> list[dict]` — Recent crypto news headlines with sentiment
- `get_market_context() -> str` — Formatted string for AI prompt injection

### Circuit Breaker (Black Swan Protection)
- `activate_circuit_breaker(reason, duration_minutes=15)` — Block real bets
- `is_circuit_breaker_active() -> bool` — Check if breaker is engaged
- `cancel_circuit_breaker() -> bool` — Manual override to resume trading
- `get_circuit_breaker_status() -> dict` — Full status with reason and expiry

## Circuit Breaker Logic

Headlines are scanned for Black Swan keywords (crash, hack, ban, exploit, etc.).
When triggered:
- **Paper mode**: Warning only, bets continue
- **Live mode**: Real bets blocked for `duration_minutes` (default 15)
- Can be manually cancelled via `/api/resume_deepseek` endpoint

## Background Fetcher

Sentiment data is fetched in background threads to avoid blocking the main engine loop.
All network errors are silently handled — the sentiment module never crashes the engine.

## Dependencies

- **External**: `requests` (HTTP), `xml.etree.ElementTree` (RSS parsing)
- **Internal**: `..config` (logging only)
- **No dependency** on database, AI, analysis, risk, or data feed modules

## Integration Points

- `engine.py`: injects `get_market_context()` into AI prompts, checks circuit breaker before betting
- `routes.py`: exposes `cancel_circuit_breaker()` via API
- `ai_engine.py`: uses Fear & Greed value in prompt building
- ML features: `fear_greed` value stored in predictions table for Judge training
