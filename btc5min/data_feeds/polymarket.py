"""
Polymarket Integration — Gamma API client for BTC 5-min markets.

Fetches real market data (order book prices, condition IDs) from Polymarket's
Gamma API for realistic paper trading P&L simulation.

Polymarket BTC 5-min markets:
- Oracle: UMA Optimistic Oracle
- Resolution source: Chainlink BTC/USD data stream
- Market structure: Binary outcome tokens (UP/DOWN) traded on CLOB
- Fees: ~2% on net winnings (not on losing trades)
"""

import json
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import requests
import websocket

from ..config import log

# ── Polymarket API endpoints ─────────────────────────────────
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Fee structure: Polymarket official formula = C × feeRate × p × (1-p)
# feeRateBps from CLOB API: GET /fee-rate?token_id={id}
# Crypto markets typically use 200 bps (2% base rate)
POLYMARKET_FEE_BPS = 200
POLYMARKET_FEE_RATE = 0.02  # Legacy compat (not used in new formula)


@dataclass
class PolymarketQuote:
    """Snapshot of a BTC 5-min market from Polymarket."""
    condition_id: str = ""
    question: str = ""
    up_token_id: str = ""
    down_token_id: str = ""
    up_price: float = 0.50    # Price to buy UP token (0.01 - 0.99)
    down_price: float = 0.50  # Price to buy DOWN token
    up_best_bid: float = 0.0
    up_best_ask: float = 0.0
    down_best_bid: float = 0.0
    down_best_ask: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    end_date: str = ""
    fetched_at: float = 0.0
    is_stale: bool = True
    # PTB (Price-To-Beat / Strike Price) extracted from market description
    strike_price: float = 0.0       # e.g. 103500.0 (USD)
    strike_source: str = "unknown"  # gamma_variant | regex_question | unknown


class PolymarketLiveStream:
    """WebSocket stream for real-time CLOB price updates.

    Connects to wss://ws-subscriptions-clob.polymarket.com/ws/market
    and subscribes to the UP/DOWN token IDs for the current 5-min window.
    Receives price_change, last_trade_price, and book events.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._up_token: str = ""
        self._down_token: str = ""
        self._up_price: float = 0.0
        self._down_price: float = 0.0
        self._last_update: float = 0.0
        self._connected: bool = False
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._current_window_ts: int = 0  # tracks which window we're subscribed to

    def start(self):
        """Start the WebSocket connection thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._thread.start()

    def subscribe_window(self, up_token: str, down_token: str, window_ts: int):
        """Subscribe to new token IDs when a window changes.

        Sends unsubscribe for old tokens, then subscribe for new ones.
        """
        with self._lock:
            if window_ts == self._current_window_ts:
                return  # already subscribed to this window
            old_up = self._up_token
            old_down = self._down_token
            self._up_token = up_token
            self._down_token = down_token
            self._up_price = 0.0
            self._down_price = 0.0
            self._current_window_ts = window_ts

        # Send subscription messages
        ws = self._ws
        if not ws or not self._connected:
            return

        # Unsubscribe old
        old_ids = [t for t in [old_up, old_down] if t]
        if old_ids:
            try:
                ws.send(json.dumps({
                    "assets_ids": old_ids,
                    "operation": "unsubscribe",
                }))
            except Exception:
                pass

        # Subscribe new
        new_ids = [t for t in [up_token, down_token] if t]
        if new_ids:
            try:
                ws.send(json.dumps({
                    "assets_ids": new_ids,
                    "type": "market",
                    "custom_feature_enabled": True,
                }))
                log.info(f"[POLY_WS] Subscribed to {len(new_ids)} tokens "
                         f"for window ts={window_ts}")
            except Exception as e:
                log.warning(f"[POLY_WS] Subscribe failed: {e}")

    def get_live_prices(self) -> tuple[float, float, float]:
        """Return (up_price, down_price, last_update_ts).

        Returns (0, 0, 0) if no live data available.
        """
        with self._lock:
            return self._up_price, self._down_price, self._last_update

    def _ws_loop(self):
        """Reconnect loop for WebSocket."""
        while True:
            try:
                log.info(f"[POLY_WS] Connecting to {CLOB_WS_URL}")
                self._ws = websocket.WebSocketApp(
                    CLOB_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, e: log.debug(f"[POLY_WS] Error: {e}"),
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=10, ping_timeout=5)
            except Exception as e:
                log.debug(f"[POLY_WS] Connection error: {e}")
            finally:
                try:
                    if self._ws is not None:
                        self._ws.close()
                except Exception:
                    pass
                self._ws = None
            self._connected = False
            time.sleep(3)

    def _on_open(self, ws):
        self._connected = True
        log.info("[POLY_WS] Connected")
        # Re-subscribe if we already have tokens (reconnect scenario)
        with self._lock:
            tokens = [t for t in [self._up_token, self._down_token] if t]
        if tokens:
            try:
                ws.send(json.dumps({
                    "assets_ids": tokens,
                    "type": "market",
                    "custom_feature_enabled": True,
                }))
                log.info(f"[POLY_WS] Re-subscribed to {len(tokens)} tokens")
            except Exception:
                pass
    def _on_close(self, ws, code, msg):
        self._connected = False
        log.debug(f"[POLY_WS] Closed: {code} {msg}")

    def _on_message(self, ws, message):
        """Handle CLOB market events: price_change, last_trade_price, book."""
        if message == "PONG":
            return
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return

        # Handle list of events (batch) or single event
        events = data if isinstance(data, list) else [data]
        for evt in events:
            etype = evt.get("event_type", "")
            asset_id = evt.get("asset_id", "")

            if etype == "price_change":
                self._handle_price_change(evt)
            elif etype == "last_trade_price":
                self._handle_trade(asset_id, evt)
            elif etype == "book":
                self._handle_book(asset_id, evt)

    def _handle_price_change(self, evt: dict):
        """Process price_change event with best_bid/best_ask."""
        changes = evt.get("price_changes", [])
        with self._lock:
            for ch in changes:
                aid = ch.get("asset_id", "")
                # Use best_ask as execution price (what you'd pay to BUY)
                ask = ch.get("best_ask")
                price = float(ask) if ask else 0.0
                if price <= 0:
                    # Fallback to the price field
                    price = float(ch.get("price", 0))
                if price <= 0:
                    continue
                if aid == self._up_token:
                    self._up_price = price
                    self._last_update = time.time()
                elif aid == self._down_token:
                    self._down_price = price
                    self._last_update = time.time()

    def _handle_trade(self, asset_id: str, evt: dict):
        """Process last_trade_price event."""
        price = float(evt.get("price", 0))
        if price <= 0:
            return
        with self._lock:
            if asset_id == self._up_token:
                self._up_price = price
                self._last_update = time.time()
            elif asset_id == self._down_token:
                self._down_price = price
                self._last_update = time.time()

    def _handle_book(self, asset_id: str, evt: dict):
        """Process full book snapshot — extract best ask."""
        asks = evt.get("asks", [])
        if not asks:
            return
        best_ask = float(asks[0].get("price", 0))
        if best_ask <= 0:
            return
        with self._lock:
            if asset_id == self._up_token:
                self._up_price = best_ask
                self._last_update = time.time()
            elif asset_id == self._down_token:
                self._down_price = best_ask
                self._last_update = time.time()


class PolymarketClient:
    """
    Client for Polymarket Gamma API — fetches BTC 5-min market prices.

    Combines REST (Gamma + CLOB) for initial fetch with WebSocket for
    real-time price streaming during the window.

    Usage:
        client = PolymarketClient()
        quote = client.get_current_btc5min_quote()
        if not quote.is_stale:
            print(f"UP: ${quote.up_price:.2f}, DOWN: ${quote.down_price:.2f}")
    """

    def __init__(self, cache_ttl: float = 8.0):
        self._cache_ttl = cache_ttl
        self._last_quote: Optional[PolymarketQuote] = None
        self._last_fetch: float = 0.0
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "btc5min-paper-trader/1.0",
        })
        self.live = PolymarketLiveStream()
        self.live.start()

    def get_current_btc5min_quote(self) -> PolymarketQuote:
        """
        Get the current BTC 5-min market quote from Polymarket.

        Priority: WebSocket live prices > cached REST quote > fresh REST fetch.
        """
        now = time.time()

        # Check if WebSocket has fresh live prices (< 5s old)
        ws_up, ws_down, ws_ts = self.live.get_live_prices()
        ws_age = now - ws_ts if ws_ts > 0 else 999
        if ws_up > 0 and ws_down > 0 and ws_age < 5.0:
            with self._lock:
                if self._last_quote and not self._last_quote.is_stale:
                    # Update the cached quote with live WS prices
                    self._last_quote.up_price = ws_up
                    self._last_quote.down_price = ws_down
                    self._last_quote.fetched_at = ws_ts
                    return self._last_quote

        # Fall back to REST with cache
        with self._lock:
            if self._last_quote and (now - self._last_fetch) < self._cache_ttl:
                return self._last_quote

        try:
            quote = self._fetch_btc5min_market()
            with self._lock:
                self._last_quote = quote
                self._last_fetch = now
            # Start WS subscription for this market's tokens
            if quote.up_token_id and quote.down_token_id:
                window_ts = (int(now) // 300) * 300
                self.live.subscribe_window(
                    quote.up_token_id, quote.down_token_id, window_ts
                )
            return quote
        except Exception as e:
            log.warning(f"[POLYMARKET] Fetch failed: {e}")
            with self._lock:
                if self._last_quote:
                    self._last_quote.is_stale = True
                    return self._last_quote
            return PolymarketQuote()  # empty fallback

    def _fetch_btc5min_market(self) -> PolymarketQuote:
        """
        Fetch the BTC 5-min market that matches our current window.

        Polymarket BTC 5-min markets use predictable slugs:
            btc-updown-5m-{unix_timestamp_of_window_start}
        where the timestamp is the UTC start of the 5-minute window,
        aligned to 300-second boundaries.

        Strategy:
        1. Compute the slug for our current window (direct lookup — no search)
        2. Fetch the event by slug from Gamma API
        3. Extract clobTokenIds from the market
        4. Get executable CLOB prices (/price?side=BUY)
        """
        # Step 1: Compute slug for current 5-min window
        now_ts = int(time.time())
        window_start = (now_ts // 300) * 300  # align to 5-min UTC boundary
        slug = f"btc-updown-5m-{window_start}"

        # Step 2: Direct lookup by slug (single HTTP call, no filtering)
        resp = self._session.get(
            f"{GAMMA_API_BASE}/events",
            params={"slug": slug, "limit": 1},
            timeout=8,
        )
        resp.raise_for_status()
        events = resp.json()

        if not events:
            log.debug(f"[POLYMARKET] No market for slug {slug}")
            return PolymarketQuote(is_stale=True)

        event = events[0]
        markets = event.get("markets", [])
        if not markets:
            log.debug(f"[POLYMARKET] Event {slug} has no markets")
            return PolymarketQuote(is_stale=True)

        market = markets[0]
        condition_id = market.get("conditionId", "")

        quote = PolymarketQuote(
            condition_id=condition_id,
            question=market.get("question", ""),
            volume_24h=float(market.get("volume24hr") or 0),
            liquidity=float(market.get("liquidity") or 0),
            end_date=market.get("endDate", ""),
            fetched_at=time.time(),
            is_stale=False,
        )

        # Step 2b: Extract PTB (strike price)
        strike, strike_src = self._extract_strike(market)
        quote.strike_price = strike
        quote.strike_source = strike_src
        if strike > 0:
            log.info(f"[POLYMARKET] PTB extracted: ${strike:,.2f} (source={strike_src})")

        # Step 3: Extract token IDs from clobTokenIds
        # Both clobTokenIds and outcomes come as JSON strings from Gamma API
        clob_ids_raw = market.get("clobTokenIds", "")
        outcomes_raw = market.get("outcomes", "")
        try:
            clob_ids = (json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str)
                        else clob_ids_raw or [])
        except (json.JSONDecodeError, TypeError):
            clob_ids = []
        try:
            outcomes = (json.loads(outcomes_raw) if isinstance(outcomes_raw, str)
                        else outcomes_raw or [])
        except (json.JSONDecodeError, TypeError):
            outcomes = []

        # Map token IDs by outcome name
        for i, token_id in enumerate(clob_ids):
            outcome = (outcomes[i] if i < len(outcomes) else "").upper()
            if outcome == "UP":
                quote.up_token_id = token_id
            elif outcome == "DOWN":
                quote.down_token_id = token_id

        # Also extract Gamma indicative prices as baseline
        prices_raw = market.get("outcomePrices")
        try:
            gamma_prices = (json.loads(prices_raw) if isinstance(prices_raw, str)
                           else prices_raw or [])
        except (json.JSONDecodeError, TypeError):
            gamma_prices = []

        if len(gamma_prices) >= 2:
            quote.up_price = float(gamma_prices[0])
            quote.down_price = float(gamma_prices[1])

        # Step 4: Fetch executable CLOB prices (overrides Gamma indicative)
        self._enrich_with_clob_prices(quote)

        log.info(f"[POLYMARKET] {slug} | {quote.question} | "
                 f"UP: ${quote.up_price:.3f} DOWN: ${quote.down_price:.3f}")

        return quote

    @staticmethod
    def _extract_strike(market: dict) -> tuple[float, str]:
        """Extract the PTB (strike price) from a Polymarket BTC 5-min market.

        Strategy (ordered by reliability):
        1. Gamma API 'variant' field — look for numeric target directly.
        2. Regex on 'question' — "Will BTC be above $XX,XXX.XX at ..."
        3. Regex on 'description' — broader pattern matching.
        Returns (strike_usd, source_label). (0.0, "unknown") if not found.
        """
        # ── Strategy 1: variant field ──
        # Some Gamma markets use variant="fifteen" or include the target number
        # in structured data. Check for numeric values in variant/customData.
        for key in ("variant", "customData"):
            val = market.get(key)
            if val and isinstance(val, str):
                # Look for a dollar-like number embedded in the variant
                m = re.search(r"[\$]?([\d,]+\.?\d*)", val)
                if m:
                    candidate = float(m.group(1).replace(",", ""))
                    # Sanity: BTC price should be > $1000 and < $10M
                    if 1_000 < candidate < 10_000_000:
                        return candidate, "gamma_variant"

        # ── Strategy 2: regex on question ──
        question = market.get("question", "")
        description = market.get("description", "")

        # Ordered by specificity — first match wins
        _STRIKE_PATTERNS = [
            r"(?:above|below|at or above|reach|exceed|over|under)\s*\$\s*([\d,]+\.?\d*)",
            r"(?:above|below|at)\s*([\d,]+\.?\d*)",
            r"\$\s*([\d,]+\.?\d*)",
        ]
        for text, source in [(question, "regex_question"),
                             (description, "regex_description")]:
            if not text:
                continue
            for pat in _STRIKE_PATTERNS:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    candidate = float(m.group(1).replace(",", ""))
                    if 1_000 < candidate < 10_000_000:
                        return candidate, source

        return 0.0, "unknown"

    def _enrich_with_clob_prices(self, quote: PolymarketQuote):
        """Fetch executable BUY prices from CLOB /price endpoint.

        Uses /price?side=BUY (best_ask) as the real execution price.
        Falls back to /book for bid/ask if /price fails.
        Overwrites Gamma's indicative prices with CLOB executable prices.
        """
        if not quote.up_token_id:
            return

        for label, token_id in [
            ("up", quote.up_token_id),
            ("down", quote.down_token_id),
        ]:
            if not token_id:
                continue
            try:
                # Primary: /price endpoint (fastest, single value)
                resp = self._session.get(
                    f"{CLOB_API_BASE}/price",
                    params={"token_id": token_id, "side": "BUY"},
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    exec_price = float(data.get("price", 0))
                    if exec_price > 0:
                        if label == "up":
                            quote.up_best_ask = exec_price
                            quote.up_price = exec_price  # Override Gamma
                        else:
                            quote.down_best_ask = exec_price
                            quote.down_price = exec_price
                        log.debug(f"[CLOB] {label.upper()} BUY price: "
                                  f"${exec_price:.3f}")
                        continue  # success — skip book fallback

                # Fallback: /book endpoint (heavier, full order book)
                resp = self._session.get(
                    f"{CLOB_API_BASE}/book",
                    params={"token_id": token_id},
                    timeout=5,
                )
                if resp.status_code == 200:
                    book = resp.json()
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    best_bid = float(bids[0]["price"]) if bids else 0.0
                    best_ask = float(asks[0]["price"]) if asks else 0.0
                    if label == "up":
                        quote.up_best_bid = best_bid
                        quote.up_best_ask = best_ask
                        if best_ask > 0:
                            quote.up_price = best_ask
                    else:
                        quote.down_best_bid = best_bid
                        quote.down_best_ask = best_ask
                        if best_ask > 0:
                            quote.down_price = best_ask
            except Exception as e:
                log.debug(f"[CLOB] {label} price fetch failed: {e}")


def calc_polymarket_pnl(
    amount_usd: float,
    side: str,
    price_cents: float,
    outcome_correct: bool,
    fee_rate_bps: int = POLYMARKET_FEE_BPS,
) -> dict:
    """
    Calculate realistic Polymarket P&L for paper trading.

    Uses the OFFICIAL Polymarket fee formula (docs.polymarket.com/trading/fees):
        fee = C × feeRate × p × (1 - p)
    where C = tokens bought, feeRate = feeRateBps / 10000, p = token price.

    The fee is charged at order MATCHING time (entry), meaning it applies
    to ALL trades — wins AND losses. This makes the simulation conservative:
    if you're profitable in paper mode, you'll be profitable in real trading.

    Mechanics:
    - Buy $5 of UP tokens at $0.55 each → get 9.09 tokens
    - Fee = 9.09 × 0.02 × 0.55 × 0.45 = $0.045
    - If UP wins: payout = 9.09 × $1.00 = $9.09, net = $9.09 - $0.045 - $5.00 = $4.045
    - If UP loses: tokens = $0, loss = -$5.00 - $0.045 = -$5.045

    Args:
        amount_usd: Amount wagered in USD
        side: "UP" or "DOWN"
        price_cents: Token price in cents (e.g., 55 for $0.55)
        outcome_correct: Whether the bet won
        fee_rate_bps: Fee rate in basis points (default from CLOB API, typically 200 for crypto)

    Returns:
        dict with payout, profit, fee, tokens_bought, effective_odds
    """
    price = max(0.01, min(0.99, price_cents / 100.0))
    tokens_bought = amount_usd / price
    fee_rate = fee_rate_bps / 10000.0

    # Polymarket official fee formula: fee = C × feeRate × p × (1 - p)
    # Charged at order matching time (both wins and losses)
    fee = tokens_bought * fee_rate * price * (1 - price)

    if outcome_correct:
        gross_payout = tokens_bought * 1.00  # each token pays $1
        gross_profit = gross_payout - amount_usd
        net_payout = gross_payout - fee
        net_profit = net_payout - amount_usd
    else:
        gross_payout = 0.0
        gross_profit = -amount_usd
        net_payout = 0.0
        net_profit = -(amount_usd + fee)  # lose bet + fee paid at entry

    return {
        "tokens_bought": round(tokens_bought, 4),
        "token_price": price,
        "gross_payout": round(gross_payout, 2),
        "gross_profit": round(gross_profit, 2),
        "fee": round(fee, 4),
        "net_payout": round(net_payout, 2),
        "net_profit": round(net_profit, 2),
        "effective_odds": round(1.0 / price, 4),  # e.g., 1.818x at $0.55
        "won": outcome_correct,
    }

