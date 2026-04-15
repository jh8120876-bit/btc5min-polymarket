# Execution — Native Polymarket CLOB Layer (py-clob-client)

> CLOB order execution against the Polymarket L2 CLOB (Polygon) via the
> **official `py-clob-client` Python client**. Direct contract-level
> integration — no Simmer routing, no market pre-import cap, no
> per-day market limit. Supports 5-minute flash markets.

## Why we migrated off simmer-sdk

- Simmer required `client.import_market()` for every market touched,
  capped at ~10 imports/day. Completely incompatible with our Sniper
  HFT hitting 5-minute flash BTC up/down markets generated on the fly.
- Going direct to the Polymarket CLOB means every outcome token ID is
  addressable, no upstream gate, and the signature flow natively
  supports delegated proxy wallets (Google/Email login).

## Architecture

```
execution/
├── clob_executor.py   # ClobExecutor — thin wrapper over py_clob_client.ClobClient
├── web3_auth.py       # ClobClient singleton + create_or_derive_api_creds
├── ladder.py          # LadderBuilder — Maker limit-order escalator (spec only)
├── ptb_filter.py      # PTB strike extraction & mathematical filter  [UNTOUCHED]
├── survival.py        # SurvivalMonitor — Profit-Lock & Flip-Stop    [UNTOUCHED]
└── relayer.py         # DEPRECATED no-op stub (no auto_redeem in py-clob-client)
```

## Execution Modes (config.py → TRADING_MODE)

| Mode | Backend | Money | Behavior |
|---|---|---|---|
| `offline_sim` | None | Simulated | Legacy SQLite + calc_polymarket_pnl (default) |
| `paper_simmer` | None (legacy) | Simulated | Aliased to offline_sim — Simmer is deprecated |
| `live_mainnet` | py-clob-client | Real USDC.e | Production — requires WALLET_PRIVATE_KEY + POLYMARKET_PROXY_FUNDER |

## ClobExecutor (clob_executor.py)

- Single contact point for all CLOB operations
- `from_config()` factory — returns None in offline_sim or if SDK unavailable
- Public methods: `submit_market`, `submit_limit`, `submit_maker_ladder`,
  `cancel_order`, `cancel_all_for_token`, `cancel_all`, `get_order_status`, `get_balance_usdc`
- Market orders: `MarketOrderArgs` + `OrderType.FOK` via `create_market_order` → `post_order`
- Limit orders: `OrderArgs` + `OrderType.GTC` via `create_order` → `post_order`
  (size is in outcome-token shares, converted internally from USD)
- Side constants imported from `py_clob_client.order_builder.constants` (BUY/SELL)
- Cancellation: native `client.cancel(order_id=...)`, `cancel_market_orders(asset_id=...)`,
  `cancel_all()` with graceful per-order fallback on batch failure
- Logs the **native Polymarket order hash** (0x...) returned by `post_order`
- `get_balance_usdc()` returns -1 (py-clob-client has no native L2 balance reader;
  callers treat -1 as "unknown")
- Never exposes raw `py_clob_client` to the rest of the codebase

## Auth Bootstrap (web3_auth.py)

- `get_clob_client()` returns a process-wide `ClobClient` singleton
- Initialized with `signature_type=1` (POLY_PROXY) because the user operates
  a delegated Google/Email Polymarket wallet
- Requires **BOTH**:
  - `WALLET_PRIVATE_KEY` — the L2 EOA signing key (exported from Polymarket UI)
  - `POLYMARKET_PROXY_FUNDER` — the public 0x... Safe/Proxy holding USDC.e
- On first call runs `client.set_api_creds(client.create_or_derive_api_creds())`
  (idempotent server-side — API creds deterministically derived from signing key)
- Private key never logged — only signing address (eth_account) and funder address
- Back-compat shim `get_simmer_client()` transparently returns the new ClobClient
- `derive_or_load_creds()` kept for old call sites; returns `{"backend": "py_clob_client"}`

## Redeem Relayer (relayer.py) — DEPRECATED STUB

- `py-clob-client` has no `auto_redeem()` equivalent; redemption requires a raw
  on-chain CTF `redeemPositions()` call plus `web3` + Polygon RPC infra
- The relayer daemon is now a no-op stub that logs a one-time warning on start()
- Manual workaround: polymarket.com → Portfolio → Claim winnings (uses the
  same delegated Google/Email wallet flow the bot is tied to)

## Fail-Safe

- If `py-clob-client` not installed → `from_config()` returns None → offline_sim + WARN
- If `WALLET_PRIVATE_KEY` or `POLYMARKET_PROXY_FUNDER` missing/invalid → same
- If `create_or_derive_api_creds` fails → same (log.error, singleton stays None)
- If `post_order()` raises mid-tick → marks `_connected=False`, engine tick loop NEVER crashes

## Dependencies

- **External**: `py-clob-client>=0.17`, `eth-account` (signing address only)
- **Internal**: `..config` (mode, keys, funder), `..config_manager` (ladder rules)
- **Removed**: `simmer-sdk`, raw Simmer handshake markers

## Engine integration notes

- `engine.py` passes `poly_quote.up_token_id` / `poly_quote.down_token_id`
  to the executor — NEVER `condition_id`. The CLOB trades outcome-token
  UUIDs; condition_id is only used for market lookup / redemption.

## Adding New Execution Features

1. Add to this package
2. Wire through ClobExecutor (single contact point) — never import py_clob_client elsewhere
3. Include a short `reasoning=...` log prefix so operators can grep order hashes
4. Add config to `dynamic_rules.json` (hot-reloadable)
5. Update this CLAUDE.md
