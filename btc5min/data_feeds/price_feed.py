import json
import time
import threading
from typing import Optional
from collections import deque

import websocket

from ..config import POLYMARKET_WS_URL, BINANCE_WS_URL, log
from ..models import PriceData


class ChainlinkPriceFeed:
    """
    Connects to Polymarket RTDS WebSocket for real-time
    Chainlink BTC/USD prices (same source Polymarket uses).
    Falls back to Binance BTC/USDT if Chainlink is stale.

    Also runs a secondary Binance trade stream for 1s-resolution
    tick data used by feature engineering (stored in hf_history).
    """

    def __init__(self):
        self.current_price: float = 0.0
        self.current_source: str = "connecting..."
        self.price_history: deque = deque(maxlen=600)
        # High-frequency price history from Binance trades (~1s resolution)
        self.hf_history: deque = deque(maxlen=3600)
        self._last_hf_append: float = 0
        self.last_update: float = 0.0
        self.connected: bool = False
        self.ws: Optional[websocket.WebSocketApp] = None
        self._lock = threading.Lock()
        self._reconnect_delay = 2

    def start(self):
        threading.Thread(target=self._connect_loop, daemon=True).start()
        threading.Thread(target=self._binance_hf_loop, daemon=True).start()

    def _connect_loop(self):
        while True:
            try:
                log.info(f"Connecting to Polymarket RTDS: {POLYMARKET_WS_URL}")
                self.ws = websocket.WebSocketApp(
                    POLYMARKET_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.ws.run_forever(ping_interval=5, ping_timeout=3)
            except Exception as e:
                log.error(f"WebSocket error: {e}")
            finally:
                # Close previous WS to release the socket before looping.
                try:
                    if self.ws is not None:
                        self.ws.close()
                except Exception:
                    pass
                self.ws = None
            self.connected = False
            log.info(f"Reconnecting in {self._reconnect_delay}s...")
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(30, self._reconnect_delay * 1.5)

    def _on_open(self, ws):
        self.connected = True
        self._reconnect_delay = 2
        log.info("Connected to Polymarket RTDS WebSocket")
        chainlink_sub = {
            "action": "subscribe",
            "subscriptions": [
                {"topic": "crypto_prices_chainlink", "type": "*",
                 "filters": '{"symbol":"btc/usd"}'}
            ],
        }
        ws.send(json.dumps(chainlink_sub))
        log.info("Subscribed to Chainlink BTC/USD")
        binance_sub = {
            "action": "subscribe",
            "subscriptions": [
                {"topic": "crypto_prices", "type": "update", "filters": "btcusdt"}
            ],
        }
        ws.send(json.dumps(binance_sub))
        log.info("Subscribed to Binance BTC/USDT as backup")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, ValueError):
            return  # Non-JSON frame (handshake, ack) — ignore silently
        try:
            topic = data.get("topic", "")
            msg_type = data.get("type", "")
            payload = data.get("payload", {})
            if not payload:
                return
            price = 0.0
            source = ""
            if topic == "crypto_prices_chainlink":
                symbol = payload.get("symbol", "")
                if symbol == "btc/usd" and msg_type == "update":
                    price = float(payload.get("value", 0))
                    source = "chainlink"
            elif topic == "crypto_prices":
                symbol = payload.get("symbol", "")
                if symbol == "btcusdt":
                    price = float(payload.get("value", 0))
                    source = "binance"
            if price > 0:
                with self._lock:
                    updated_current = False
                    if source == "chainlink" or \
                       (source == "binance" and self.current_source != "chainlink"):
                        self.current_price = price
                        self.current_source = source
                        updated_current = True
                    elif source == "binance" and (time.time() - self.last_update > 30):
                        self.current_price = price
                        self.current_source = "binance (fallback)"
                        updated_current = True
                    ts = payload.get("timestamp", time.time() * 1000) / 1000
                    self.price_history.append(
                        PriceData(price=price, timestamp=ts, source=source)
                    )
                    # Only update last_update when current_price actually changed.
                    # Otherwise get_price() reports stale Chainlink prices as fresh
                    # because Binance messages kept bumping last_update without
                    # changing current_price.
                    if updated_current:
                        self.last_update = time.time()
        except Exception as e:
            log.error(f"Message parse error: {e}")

    def _on_error(self, ws, error):
        log.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self.connected = False
        log.warning(f"WebSocket closed: {close_status_code} {close_msg}")

    def get_price(self) -> Optional[PriceData]:
        with self._lock:
            if self.current_price > 0:
                return PriceData(
                    price=self.current_price,
                    timestamp=self.last_update,
                    source=self.current_source,
                )
            return None

    def get_recent_prices(self, n: int = 60) -> list[float]:
        with self._lock:
            prices = list(self.price_history)
            return [p.price for p in prices[-n:]]

    def get_binance_price(self) -> Optional[PriceData]:
        """Get the latest Binance spot price (oracle source for Polymarket resolution).

        Uses the HF trade stream (~1s resolution) as primary source.
        Falls back to the main WS Binance price if HF is empty.
        Returns None if no Binance data is available.
        """
        with self._lock:
            # Primary: last price from Binance HF trade stream
            if self.hf_history:
                return PriceData(
                    price=self.hf_history[-1],
                    timestamp=self._last_hf_append,
                    source="binance_spot",
                )
            # Fallback: Binance price from Polymarket WS relay
            if "binance" in self.current_source.lower() and self.current_price > 0:
                return PriceData(
                    price=self.current_price,
                    timestamp=self.last_update,
                    source="binance_ws",
                )
            return None

    def get_hf_prices(self, n: int = 3600) -> list[float]:
        """Get high-frequency prices from Binance trade stream (~1s intervals)."""
        with self._lock:
            return list(self.hf_history)[-n:]

    def get_status(self) -> dict:
        with self._lock:
            age = time.time() - self.last_update if self.last_update else 999
            return {
                "connected": self.connected,
                "source": self.current_source,
                "price": self.current_price,
                "last_update_age": round(age, 1),
                "history_size": len(self.price_history),
                "hf_history_size": len(self.hf_history),
            }

    # ── Binance high-frequency trade stream ──────────────────

    def _binance_hf_loop(self):
        """Connect to Binance trade stream for ~1s tick resolution."""
        while True:
            ws = None
            try:
                log.info(f"Connecting to Binance HF stream: {BINANCE_WS_URL}")
                ws = websocket.WebSocketApp(
                    BINANCE_WS_URL,
                    on_message=self._on_hf_message,
                    on_error=lambda ws, e: log.debug(f"Binance HF error: {e}"),
                    on_close=lambda ws, c, m: log.debug("Binance HF closed"),
                    on_open=lambda ws: log.info("Binance HF stream connected"),
                )
                ws.run_forever(ping_interval=10, ping_timeout=5)
            except Exception as e:
                log.debug(f"Binance HF connection error: {e}")
            finally:
                try:
                    if ws is not None:
                        ws.close()
                except Exception:
                    pass
            time.sleep(3)

    def _on_hf_message(self, ws, message):
        """Handle Binance individual trade messages. Throttle to ~1s."""
        try:
            data = json.loads(message)
            price = float(data.get("p", 0))
            if price <= 0:
                return
            now = time.time()
            with self._lock:
                # Append at most once per second to keep deque manageable
                if now - self._last_hf_append >= 1.0:
                    self.hf_history.append(price)
                    self._last_hf_append = now
        except Exception:
            pass
