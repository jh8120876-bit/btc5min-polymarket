# Analysis — Technical Indicators & Smart Money Concepts

> All market analysis: 22 TA indicators, SMC structural features, and ML feature engineering.

## Architecture

```
analysis/
├── technical.py           # 22 indicators: RSI, EMA, MACD, BB, ATR, S/R, pressure
├── smc_features.py        # Smart Money Concepts: FVG, BOS/CHoCH, OB, Kill Zones
├── feature_engineering.py # ML features: acceleration, vol z-score, multi-timeframe
└── vendor_smc.py          # Vendored smartmoneyconcepts library (pandas-based)
```

## technical.py — TechnicalAnalysis

Stateless utility class. No external dependencies (only `math`).

**22 indicators** returned by `analyze(prices)`:
- RSI (14-period), EMA 9/21, MACD (12/26/9), Bollinger Bands (20,2)
- ATR (14-period), ATR percentage, Volatility
- Support/Resistance levels, S/R distance
- Momentum, Price change %, Trend alignment
- Buy/Sell pressure, Volume ratio
- Bollinger %B, BB width

**Usage**: `TechnicalAnalysis.analyze(prices: list[float]) -> dict`

All indicators are pure functions over price history. No state, no side effects.

## smc_features.py — Smart Money Concepts Engine

Computes institutional-level structural features from Binance 5m OHLCV klines.

**Features computed**:
- FVG (Fair Value Gaps) — count and nearest distance
- BOS (Break of Structure) / CHoCH (Change of Character) — timing
- Order Blocks — distance to nearest, strength
- Liquidity pools — sweep detection
- Kill Zones — London/NY session overlap flags
- SMC-based Support/Resistance

**Safe-fail**: Returns empty defaults if pandas/numpy unavailable (`_SMC_AVAILABLE` flag).

**Dependencies**: `vendor_smc` (vendored library), `pandas`, `numpy` (all optional)

## feature_engineering.py — FeatureEngineer

Transforms raw TA + market data into ML-ready features.

**Key transformations**:
- Price acceleration (2nd derivative of momentum)
- Volatility z-score (normalized by rolling mean)
- Multi-timeframe features (from kline data)
- Hour-of-day encoding (cyclic sin/cos or raw `hour_utc`)

**Usage**: `FeatureEngineer.build_features(ta_dict, market_ctx, klines) -> dict`

No external dependencies (only `math`, `datetime`).

## vendor_smc.py — Vendored Library

Standalone pandas/numpy library for Smart Money Concepts calculations.
**Do not modify** — this is vendored from an external source.
If upgrading, replace the entire file.

## Dependencies

- **External**: `pandas`, `numpy` (optional, for SMC only)
- **Internal**: `..config` (logging only, in smc_features.py)
- **No dependency** on database, AI, data feeds, or risk modules

## Adding a New Indicator

1. Add calculation to `technical.py:analyze()` return dict
2. Include in `ai_engine.py:_build_prompt()` to feed to AI
3. If ML feature: add to `feature_engineering.py` and `ml/train_xgboost_judge.py` FEATURES list
4. Update this CLAUDE.md
