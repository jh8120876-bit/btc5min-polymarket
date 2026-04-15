"""
Binance market context provider.
Fetches volume, funding rate, open interest, klines via REST API.
Streams liquidation events via Futures WebSocket.
All with exponential backoff — falls back gracefully if Binance is down.
"""

import json
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
import websocket

from ..config import log

# Binance public REST endpoints (no API key needed)
_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"
_FUNDING_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
_OI_URL = "https://fapi.binance.com/fapi/v1/openInterest"
_BOOK_TICKER_URL = "https://api.binance.com/api/v3/ticker/bookTicker"
_KLINES_URL = "https://api.binance.com/api/v3/klines"
_DEPTH_URL = "https://api.binance.com/api/v3/depth"

# Binance Futures WebSocket for liquidations (forceOrder)
_LIQUIDATION_WS_URL = "wss://fstream.binance.com/ws/btcusdt@forceOrder"

# Binance Spot WebSocket for aggTrade (Order Flow / CVD)
_AGGTRADE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"


class BinanceMarketData:
    """Fetches BTC market context from Binance at configurable intervals."""

    def __init__(self, refresh_interval: int = 60):
        self._cache: dict = {}
        self._klines_cache: dict = {}  # {"15m": [...], "1h": [...]}
        self._klines_5m_ohlcv: list[dict] = []  # Full OHLCV for SMC analysis
        self._last_fetch: float = 0
        self._prev_oi: Optional[float] = None
        self._refresh_interval = refresh_interval
        self._lock = threading.Lock()
        self._running = False

        # Liquidation tracking
        self._liq_events: deque = deque(maxlen=500)
        self._liq_lock = threading.Lock()
        self._liq_connected = False

        # ── Order Flow / CVD (aggTrade stream) ──
        self._agg_trades: deque = deque(maxlen=10000)  # ~5min of ticks at peak
        self._agg_lock = threading.Lock()
        self._agg_connected = False
        # Rolling CVD accumulator per 5-min window (keyed by window_start_ts)
        self._cvd_window_start: float = 0
        self._cvd_buy_vol: float = 0
        self._cvd_sell_vol: float = 0

    def start(self):
        """Start background refresh + liquidation stream + aggTrade stream threads."""
        self._running = True
        threading.Thread(target=self._refresh_loop, daemon=True).start()
        threading.Thread(target=self._liquidation_ws_loop, daemon=True).start()
        threading.Thread(target=self._aggtrade_ws_loop, daemon=True).start()
        log.info("BinanceMarketData started (REST + liquidation WS + aggTrade WS)")

    def stop(self):
        self._running = False

    # ── REST Polling Loop ─────────────────────────────────────

    def _refresh_loop(self):
        backoff = 2
        while self._running:
            try:
                self._fetch_all()
                backoff = 2  # reset on success
            except Exception as e:
                log.error(f"Binance data fetch error: {e}")
                backoff = min(120, backoff * 1.5)
            time.sleep(max(self._refresh_interval, backoff))

    def _fetch_all(self):
        """Fetch all market context data from Binance in parallel (~1s vs ~5s)."""
        ctx = {}
        klines = {}
        ohlcv_result = []

        def _fetch_ticker():
            resp = requests.get(_TICKER_URL, params={"symbol": "BTCUSDT"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return {"volume_24h": float(data.get("quoteVolume", 0))}
            return {}

        def _fetch_funding():
            resp = requests.get(_FUNDING_URL, params={"symbol": "BTCUSDT"}, timeout=5)
            if resp.status_code == 200:
                return {"funding_rate": float(resp.json().get("lastFundingRate", 0))}
            return {}

        def _fetch_oi():
            resp = requests.get(_OI_URL, params={"symbol": "BTCUSDT"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                oi = float(data.get("openInterest", 0))
                result = {"open_interest": oi}
                with self._lock:
                    prev = self._prev_oi
                    self._prev_oi = oi
                if prev and prev > 0:
                    result["oi_change_pct"] = round(
                        (oi - prev) / prev * 100, 4)
                else:
                    result["oi_change_pct"] = 0
                return result
            return {}

        def _fetch_book_ticker():
            resp = requests.get(_BOOK_TICKER_URL, params={"symbol": "BTCUSDT"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                bid = float(data.get("bidPrice", 0))
                ask = float(data.get("askPrice", 0))
                if bid > 0:
                    return {"bid_ask_spread": round((ask - bid) / bid * 100, 6)}
            return {}

        def _fetch_depth():
            resp = requests.get(_DEPTH_URL, params={"symbol": "BTCUSDT", "limit": 10}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                bid_vols = sum(float(b[1]) for b in data.get("bids", []))
                ask_vols = sum(float(a[1]) for a in data.get("asks", []))
                total = bid_vols + ask_vols
                if total > 0:
                    return {"order_book_imbalance": round(bid_vols / total, 4)}
                return {"order_book_imbalance": 0.5}
            return {}

        def _fetch_klines(interval, limit=4):
            resp = requests.get(
                _KLINES_URL,
                params={"symbol": "BTCUSDT", "interval": interval, "limit": limit},
                timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                closes = [float(k[4]) for k in data[:-1]]
                return interval, closes[-3:] if len(closes) >= 3 else closes
            return interval, []

        def _fetch_ohlcv_5m():
            resp = requests.get(
                _KLINES_URL,
                params={"symbol": "BTCUSDT", "interval": "5m", "limit": 60},
                timeout=5)
            if resp.status_code == 200:
                raw = resp.json()
                return [{
                    "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]),
                } for k in raw[:-1]]
            return []

        # Run all 8 fetches in parallel
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_fetch_ticker): "ticker",
                pool.submit(_fetch_funding): "funding",
                pool.submit(_fetch_oi): "oi",
                pool.submit(_fetch_book_ticker): "book",
                pool.submit(_fetch_depth): "depth",
                pool.submit(_fetch_klines, "15m"): "klines_15m",
                pool.submit(_fetch_klines, "1h"): "klines_1h",
                pool.submit(_fetch_ohlcv_5m): "ohlcv",
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    result = future.result()
                    if label in ("klines_15m", "klines_1h"):
                        interval, closes = result
                        if closes:
                            klines[interval] = closes
                    elif label == "ohlcv":
                        ohlcv_result = result
                    else:
                        ctx.update(result)
                except Exception as e:
                    log.debug(f"Binance {label} error: {e}")

        # Compute volume_1h from last 12 closed 5m candles (= 1 hour)
        if ohlcv_result and len(ohlcv_result) >= 12:
            last_12 = ohlcv_result[-12:]
            avg_price = sum(c["close"] for c in last_12) / 12
            ctx["volume_1h"] = sum(c["volume"] for c in last_12) * avg_price
        elif "volume_24h" in ctx:
            ctx["volume_1h"] = ctx["volume_24h"] / 24  # fallback

        with self._lock:
            self._cache = ctx
            self._last_fetch = time.time()
            if klines:
                self._klines_cache = klines
            if ohlcv_result:
                self._klines_5m_ohlcv = ohlcv_result

    # ── Klines Access ──────────────────────────────────────────

    def get_klines(self, interval: str = "15m") -> list[float]:
        """Get last 3 closed candle closes for an interval."""
        with self._lock:
            return list(self._klines_cache.get(interval, []))

    def get_klines_5m_ohlcv(self) -> list[dict]:
        """Get last ~59 closed 5m candles as OHLCV dicts for SMC analysis."""
        with self._lock:
            return list(self._klines_5m_ohlcv)

    # ── Liquidation WebSocket ──────────────────────────────────

    def _liquidation_ws_loop(self):
        """Connect to Binance Futures forceOrder stream with exponential backoff."""
        backoff = 2
        while self._running:
            ws = None
            try:
                log.info("Connecting to Binance liquidation stream...")
                ws = websocket.WebSocketApp(
                    _LIQUIDATION_WS_URL,
                    on_open=self._on_liq_open,
                    on_message=self._on_liq_message,
                    on_error=self._on_liq_error,
                    on_close=self._on_liq_close,
                )
                ws.run_forever(ping_interval=15, ping_timeout=10)
            except Exception as e:
                log.debug(f"Liquidation WS connection error: {e}")
            finally:
                try:
                    if ws is not None:
                        ws.close()
                except Exception:
                    pass
            self._liq_connected = False
            log.info(f"Liquidation WS reconnecting in {backoff:.0f}s...")
            time.sleep(backoff)
            backoff = min(120, backoff * 1.5)

    def _on_liq_open(self, ws):
        self._liq_connected = True
        log.info("Binance liquidation stream connected")

    def _on_liq_message(self, ws, message):
        """
        Handle forceOrder messages.
        Format: {"e":"forceOrder","o":{"s":"BTCUSDT","S":"BUY"|"SELL","q":"0.01","p":"69000",...}}
        S=BUY means shorts got liquidated (bullish), S=SELL means longs got liquidated (bearish).
        """
        try:
            data = json.loads(message)
            order = data.get("o", {})
            symbol = order.get("s", "")
            if symbol != "BTCUSDT":
                return
            side = order.get("S", "")  # BUY or SELL
            qty = float(order.get("q", 0))
            price = float(order.get("p", 0))
            usd_value = qty * price

            event = {
                "timestamp": time.time(),
                "side": side,
                "qty": qty,
                "price": price,
                "usd_value": usd_value,
            }
            with self._liq_lock:
                self._liq_events.append(event)

        except Exception as e:
            log.debug(f"Liquidation parse error: {e}")

    def _on_liq_error(self, ws, error):
        log.debug(f"Liquidation WS error: {error}")

    def _on_liq_close(self, ws, close_status_code, close_msg):
        self._liq_connected = False
        log.debug(f"Liquidation WS closed: {close_status_code}")

    def get_liquidation_summary(self, minutes: int = 5) -> dict:
        """
        Get aggregated liquidation data for the last N minutes.
        Returns buy/sell volumes and a spike detection.
        """
        cutoff = time.time() - (minutes * 60)
        with self._liq_lock:
            recent = [e for e in self._liq_events if e["timestamp"] > cutoff]

        buy_vol = sum(e["usd_value"] for e in recent if e["side"] == "BUY")
        sell_vol = sum(e["usd_value"] for e in recent if e["side"] == "SELL")
        buy_count = sum(1 for e in recent if e["side"] == "BUY")
        sell_count = sum(1 for e in recent if e["side"] == "SELL")
        total_vol = buy_vol + sell_vol

        # Spike detection: compare to longer baseline (30 min)
        cutoff_30m = time.time() - 1800
        with self._liq_lock:
            baseline = [e for e in self._liq_events if e["timestamp"] > cutoff_30m]
        baseline_vol = sum(e["usd_value"] for e in baseline) if baseline else 0
        # Avg 5-min volume over 30 min baseline
        avg_5m_vol = baseline_vol / 6 if baseline_vol > 0 else 0

        spike = False
        spike_ratio = 0.0
        if avg_5m_vol > 0:
            spike_ratio = total_vol / avg_5m_vol
            spike = spike_ratio > 3.0  # 3x normal = spike

        # Dominant side in liquidations
        net_side = ("BUY" if buy_vol > sell_vol * 1.5
                    else "SELL" if sell_vol > buy_vol * 1.5
                    else "NEUTRAL")

        return {
            "buy_vol": round(buy_vol, 2),
            "sell_vol": round(sell_vol, 2),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_vol": round(total_vol, 2),
            "net_side": net_side,
            "spike": spike,
            "spike_ratio": round(spike_ratio, 2),
            "connected": self._liq_connected,
        }

    def get_liq_cluster_proximity(self, current_price: float,
                                   atr: float,
                                   window_sec: int = 7200,
                                   bucket_usd: float = 50.0,
                                   top_k: int = 3) -> dict:
        """Liquidation cluster proximity score (Vibe-Trading inspired).

        Aggregates liquidation USD value into price buckets over the last
        ``window_sec`` (default 2h), identifies the densest clusters, and
        returns ATR-scaled distance to the nearest dense cluster + directional
        bias (cluster above/below spot) + magnitude percentile.

        Interpretation for sniper:
        - ``distance_atr < 0.5`` + ``magnitude_pct > 0.9`` + side aligned
          with prediction → stop-hunt cascade imminent → boost confidence.
        - Cluster above spot = short liquidations pending = bullish sweep.
        - Cluster below spot = long liquidations pending = bearish sweep.
        """
        if current_price <= 0 or atr <= 0:
            return {
                "nearest_cluster_price": None, "distance_usd": None,
                "distance_atr": None, "cluster_magnitude_usd": 0.0,
                "magnitude_pct": 0.0, "cluster_side": "NEUTRAL",
                "top_clusters": [], "n_events": 0,
            }

        cutoff = time.time() - window_sec
        with self._liq_lock:
            events = [e for e in self._liq_events if e["timestamp"] > cutoff]

        if not events:
            return {
                "nearest_cluster_price": None, "distance_usd": None,
                "distance_atr": None, "cluster_magnitude_usd": 0.0,
                "magnitude_pct": 0.0, "cluster_side": "NEUTRAL",
                "top_clusters": [], "n_events": 0,
            }

        # Bucketize liquidations by price ($50-wide bins by default).
        buckets: dict[float, float] = {}
        for e in events:
            p = float(e.get("price", 0.0))
            if p <= 0:
                continue
            key = round(p / bucket_usd) * bucket_usd
            buckets[key] = buckets.get(key, 0.0) + float(e.get("usd_value", 0.0))

        if not buckets:
            return {
                "nearest_cluster_price": None, "distance_usd": None,
                "distance_atr": None, "cluster_magnitude_usd": 0.0,
                "magnitude_pct": 0.0, "cluster_side": "NEUTRAL",
                "top_clusters": [], "n_events": len(events),
            }

        # Top-K densest clusters.
        ranked = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[:top_k]

        # Nearest cluster among top-K (spatial, not just magnitude).
        nearest = min(top, key=lambda kv: abs(kv[0] - current_price))
        cluster_price, cluster_mag = nearest
        distance_usd = cluster_price - current_price  # signed
        distance_atr = abs(distance_usd) / atr

        # Magnitude percentile vs all buckets in window.
        all_mags = sorted(buckets.values())
        rank = sum(1 for m in all_mags if m <= cluster_mag)
        magnitude_pct = rank / len(all_mags)

        if distance_usd > atr * 0.1:
            cluster_side = "ABOVE"  # shorts trapped → bullish sweep target
        elif distance_usd < -atr * 0.1:
            cluster_side = "BELOW"  # longs trapped → bearish sweep target
        else:
            cluster_side = "AT_SPOT"

        return {
            "nearest_cluster_price": round(cluster_price, 2),
            "distance_usd": round(distance_usd, 2),
            "distance_atr": round(distance_atr, 3),
            "cluster_magnitude_usd": round(cluster_mag, 2),
            "magnitude_pct": round(magnitude_pct, 3),
            "cluster_side": cluster_side,
            "top_clusters": [
                {"price": round(p, 2), "usd": round(m, 2)} for p, m in top
            ],
            "n_events": len(events),
            "window_sec": window_sec,
        }

    # ── aggTrade WebSocket (Order Flow / CVD) ────────────────

    def _aggtrade_ws_loop(self):
        """Connect to Binance aggTrade stream for tick-level order flow."""
        backoff = 2
        while self._running:
            ws = None
            try:
                log.info("Connecting to Binance aggTrade stream...")
                ws = websocket.WebSocketApp(
                    _AGGTRADE_WS_URL,
                    on_open=self._on_agg_open,
                    on_message=self._on_agg_message,
                    on_error=self._on_agg_error,
                    on_close=self._on_agg_close,
                )
                ws.run_forever(ping_interval=15, ping_timeout=10)
            except Exception as e:
                log.debug(f"aggTrade WS connection error: {e}")
            finally:
                try:
                    if ws is not None:
                        ws.close()
                except Exception:
                    pass
            self._agg_connected = False
            log.info(f"aggTrade WS reconnecting in {backoff:.0f}s...")
            time.sleep(backoff)
            backoff = min(120, backoff * 1.5)

    def _on_agg_open(self, ws):
        self._agg_connected = True
        log.info("Binance aggTrade stream connected")

    def _on_agg_message(self, ws, message):
        """
        Handle aggTrade messages for CVD computation.
        Format: {"e":"aggTrade","s":"BTCUSDT","p":"69000.5","q":"0.01","m":true,...}
        m=true means the buyer is the maker → this is a SELL (market sell hit the bid).
        m=false means the seller is the maker → this is a BUY (market buy lifted the ask).
        """
        try:
            data = json.loads(message)
            qty = float(data.get("q", 0))
            price = float(data.get("p", 0))
            is_buyer_maker = data.get("m", False)
            usd_value = qty * price
            ts = data.get("T", 0) / 1000.0  # Trade timestamp (ms -> s)

            # Classify: m=true → market SELL, m=false → market BUY
            side = "SELL" if is_buyer_maker else "BUY"

            trade = {
                "timestamp": ts if ts > 0 else time.time(),
                "side": side,
                "qty": qty,
                "price": price,
                "usd_value": usd_value,
            }

            with self._agg_lock:
                self._agg_trades.append(trade)
                # Accumulate into current CVD window
                if side == "BUY":
                    self._cvd_buy_vol += usd_value
                else:
                    self._cvd_sell_vol += usd_value

        except Exception as e:
            log.debug(f"aggTrade parse error: {e}")

    def _on_agg_error(self, ws, error):
        log.debug(f"aggTrade WS error: {error}")

    def _on_agg_close(self, ws, close_status_code, close_msg):
        self._agg_connected = False
        log.debug(f"aggTrade WS closed: {close_status_code}")

    def reset_cvd_window(self):
        """Reset CVD accumulators at the start of a new 5-min window."""
        with self._agg_lock:
            self._cvd_window_start = time.time()
            self._cvd_buy_vol = 0
            self._cvd_sell_vol = 0

    def get_cvd_summary(self, minutes: int = 5) -> dict:
        """
        Get Cumulative Volume Delta for the last N minutes.

        Returns:
            cvd_net: BUY_vol - SELL_vol (positive = aggressive buying dominates)
            cvd_imbalance_pct: (BUY - SELL) / (BUY + SELL) * 100
            cvd_buy_vol: total aggressive buy volume in USD
            cvd_sell_vol: total aggressive sell volume in USD
            cvd_total_vol: total volume
            cvd_trade_count: number of aggTrades
            cvd_buy_count / cvd_sell_count: trade counts by side
            connected: bool
        """
        cutoff = time.time() - (minutes * 60)
        with self._agg_lock:
            recent = [t for t in self._agg_trades if t["timestamp"] > cutoff]

        buy_vol = sum(t["usd_value"] for t in recent if t["side"] == "BUY")
        sell_vol = sum(t["usd_value"] for t in recent if t["side"] == "SELL")
        buy_count = sum(1 for t in recent if t["side"] == "BUY")
        sell_count = sum(1 for t in recent if t["side"] == "SELL")
        total_vol = buy_vol + sell_vol
        net = buy_vol - sell_vol

        imbalance_pct = 0.0
        if total_vol > 0:
            imbalance_pct = round(net / total_vol * 100, 2)

        return {
            "cvd_net": round(net, 2),
            "cvd_imbalance_pct": imbalance_pct,
            "cvd_buy_vol": round(buy_vol, 2),
            "cvd_sell_vol": round(sell_vol, 2),
            "cvd_total_vol": round(total_vol, 2),
            "cvd_trade_count": len(recent),
            "cvd_buy_count": buy_count,
            "cvd_sell_count": sell_count,
            "connected": self._agg_connected,
        }

    def get_cvd_window_accumulator(self) -> dict:
        """Get the current window's accumulated CVD (since last reset_cvd_window)."""
        with self._agg_lock:
            buy = self._cvd_buy_vol
            sell = self._cvd_sell_vol
        total = buy + sell
        net = buy - sell
        imbalance = round(net / total * 100, 2) if total > 0 else 0.0
        return {
            "cvd_window_buy": round(buy, 2),
            "cvd_window_sell": round(sell, 2),
            "cvd_window_net": round(net, 2),
            "cvd_window_imbalance_pct": imbalance,
        }

    def get_cvd_intracandle_series(self, seconds: int = 300,
                                    bucket_sec: int = 10) -> list[dict]:
        """
        Get time-bucketed CVD series within current candle.
        Returns list of {bucket_ts, buy_vol, sell_vol, net, cumulative_net}
        for use in LSTM temporal features.
        """
        cutoff = time.time() - seconds
        with self._agg_lock:
            recent = [t for t in self._agg_trades if t["timestamp"] > cutoff]

        if not recent:
            return []

        # Bucket by time
        buckets: dict[int, dict] = {}
        for t in recent:
            bucket_key = int(t["timestamp"] // bucket_sec) * bucket_sec
            if bucket_key not in buckets:
                buckets[bucket_key] = {"buy": 0.0, "sell": 0.0}
            if t["side"] == "BUY":
                buckets[bucket_key]["buy"] += t["usd_value"]
            else:
                buckets[bucket_key]["sell"] += t["usd_value"]

        series = []
        cum_net = 0.0
        for ts_key in sorted(buckets.keys()):
            b = buckets[ts_key]
            net = b["buy"] - b["sell"]
            cum_net += net
            series.append({
                "bucket_ts": ts_key,
                "buy_vol": round(b["buy"], 2),
                "sell_vol": round(b["sell"], 2),
                "net": round(net, 2),
                "cumulative_net": round(cum_net, 2),
            })
        return series

    # ── General Access ────────────────────────────────────────

    def get_context(self) -> dict:
        """Get latest market context. Returns empty dict if no data."""
        with self._lock:
            return dict(self._cache)

    def get_status(self) -> dict:
        with self._lock:
            age = time.time() - self._last_fetch if self._last_fetch else 999
            return {
                "has_data": bool(self._cache),
                "last_fetch_age": round(age, 1),
                "funding_rate": self._cache.get("funding_rate"),
                "open_interest": self._cache.get("open_interest"),
                "klines_15m": bool(self._klines_cache.get("15m")),
                "klines_1h": bool(self._klines_cache.get("1h")),
                "liq_connected": self._liq_connected,
                "agg_connected": self._agg_connected,
            }
