# Data Feeds — Price Oracles & Market Data

> Real-time price feeds, Binance market data, and Polymarket CLOB integration.

## Architecture

```
data_feeds/
├── price_feed.py      # Chainlink WS (primary) + Binance spot (fallback)
├── binance_data.py    # Volume, funding, OI, liquidations, aggTrade CVD via REST+WS
├── polymarket.py      # Gamma API + CLOB prices + WS live stream
└── options_data.py    # Deribit GEX polling (Call/Put Wall, Net GEX bias)
```

## Price Oracle Hierarchy

1. **Chainlink BTC/USD** (primary) — WebSocket via Polymarket RTDS (`wss://ws-live-data.polymarket.com`)
2. **Binance spot** (fallback) — activates after 30s Chainlink staleness
3. Resolution: Chainlink (matches Polymarket UMA source). Binance fallback if unavailable at resolve time.

## price_feed.py — ChainlinkPriceFeed

- Connects to `POLYMARKET_WS_URL` for Chainlink BTC/USD stream
- Falls back to `BINANCE_WS_URL` (`wss://stream.binance.com`) if Chainlink stale >30s
- Maintains `deque` of recent prices for tick history
- Thread-safe: WS runs in daemon thread, `get_price()` returns latest
- Key class: `ChainlinkPriceFeed` — instantiated once in `engine.py`

## binance_data.py — BinanceMarketData

- **REST endpoints**: volume (24h), funding rate, open interest, klines (OHLCV)
- **WebSocket**: `wss://fstream.binance.com` forceOrder stream for liquidation events
- `ThreadPoolExecutor` fetches 4 REST endpoints in parallel
- Liquidations stored in `deque(maxlen=500)`, aggregated by 5m windows
- Exponential backoff on API failures (graceful degradation)
- Key method: `get_market_context()` returns dict with volume, funding, OI, spread, liquidations

### aggTrade WebSocket / CVD (Order Flow)

- **Stream**: `wss://stream.binance.com:9443/ws/btcusdt@aggTrade` — tick-level trades
- Classifies Market Buy vs Sell via `m` (buyer_is_maker) flag
- `deque(maxlen=10000)` accumulates ~5min of ticks at peak activity
- Rolling per-window CVD accumulators (`_cvd_buy_vol`, `_cvd_sell_vol`) reset each window

**Key methods:**
- `reset_cvd_window()` — reset CVD accumulators at start of new 5-min window
- `get_cvd_summary(minutes=5)` — returns `{cvd_net, imbalance_pct, aggressor_ratio, buy_vol, sell_vol, trade_count}`
- `get_cvd_intracandle_series(seconds=300, bucket_sec=10)` — time-bucketed CVD series `[{bucket_ts, buy_vol, sell_vol, net, cumulative_net}]` for LSTM temporal features

## options_data.py — DeribitGEXProvider

Polls Deribit public API every 5min for BTC option book summaries. No API key required.

**What it computes:**
- **Call Wall**: strike with max call open interest (magnetic resistance)
- **Put Wall**: strike with max put open interest (magnetic support)
- **Net GEX Bias**: normalized `(call_gamma - put_gamma)` scaled to [-100..+100]
- **GEX Regime**: `LONG_GAMMA` (>20, mean-reverting), `SHORT_GAMMA` (<-20, trending), `NEUTRAL_GAMMA`

**Architecture:**
- `_fetch_gex()` — fetches `get_book_summary_by_currency` for BTC options, filters to 7-day expiry
- Gamma contribution: `OI * exp(-5 * moneyness²)` (peaks at ATM, decays with distance)
- Background daemon thread with exponential backoff on errors

**Key methods:**
- `start()` — launch background polling thread
- `get_gex()` → dict with `call_wall`, `put_wall`, `net_gex_bias`, `gex_regime`, etc.
- `get_status()` → dict with `has_data`, `last_fetch_age`, summary fields

## polymarket.py — PolymarketClient

- **Gamma API**: slug lookup `btc-updown-5m-{timestamp}` → event → clobTokenIds
- **CLOB price**: `/price?side=BUY` (best_ask) overrides Gamma indicative prices
- **WebSocket**: `wss://ws-subscriptions-clob.polymarket.com/ws/market` — subscribes to UP/DOWN token IDs, receives `price_change`, `last_trade_price`, `book` events
- **P&L model** (`calc_polymarket_pnl`): Buy tokens at market price, win pays $1.00/token minus 2% fee on winnings, loss = full amount
- Spread simulation: when no CLOB data, adds 2c penalty to internal odds

## Dependencies

- **External**: `websocket-client`, `requests`
- **Internal**: `..config` (URLs, logging), `..models` (PriceData dataclass)
- **No dependency** on database, AI, or analysis modules

## Key Config Values (config.py)

- `POLYMARKET_WS_URL` — Chainlink RTDS WebSocket
- `BINANCE_WS_URL` — Binance spot stream
- `BINANCE_FUTURES_WS` — Binance futures liquidation stream
- `_AGGTRADE_WS_URL` — Binance spot aggTrade stream (CVD)
- `_GEX_REFRESH_INTERVAL` — Deribit GEX polling interval (300s)

## Adding a New Data Source

1. Create `new_source.py` in this folder
2. Follow the pattern: class with `start()` (background thread) and `get_*()` (thread-safe getter)
3. Add re-export in `__init__.py`
4. Instantiate in `engine.py` constructor
5. Update this CLAUDE.md
