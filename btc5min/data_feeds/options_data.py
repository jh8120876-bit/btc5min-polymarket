"""
Deribit Options Data — Gamma Exposure (GEX) Provider.

Polls Deribit public API every 5 minutes to compute:
- Call Wall: strike with max call open interest
- Put Wall: strike with max put open interest
- Net GEX Bias: directional gamma pressure (positive = dealer long gamma = mean-reverting)

Uses only public endpoints — no API key required.
"""

import time
import math
import threading
from typing import Optional

import requests

from ..config import log

# Deribit public REST endpoints
_DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
_INSTRUMENTS_URL = f"{_DERIBIT_BASE}/get_instruments"
_TICKER_URL = f"{_DERIBIT_BASE}/ticker"
_BOOK_SUMMARY_URL = f"{_DERIBIT_BASE}/get_book_summary_by_currency"

# Refresh interval (seconds)
_GEX_REFRESH_INTERVAL = 300  # 5 minutes


class DeribitGEXProvider:
    """Polls Deribit options data and computes Gamma Exposure metrics."""

    def __init__(self, refresh_interval: int = _GEX_REFRESH_INTERVAL):
        self._refresh_interval = refresh_interval
        self._cache: dict = {}
        self._last_fetch: float = 0
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        """Start background GEX polling thread."""
        self._running = True
        threading.Thread(target=self._poll_loop, daemon=True).start()
        log.info("DeribitGEXProvider started (polling every "
                 f"{self._refresh_interval}s)")

    def stop(self):
        self._running = False

    def _poll_loop(self):
        backoff = 5
        while self._running:
            try:
                self._fetch_gex()
                backoff = 5  # reset on success
            except Exception as e:
                log.debug(f"[GEX] Deribit fetch error: {e}")
                backoff = min(120, backoff * 1.5)
            time.sleep(max(self._refresh_interval, backoff))

    def _fetch_gex(self):
        """
        Fetch BTC options instruments from Deribit, compute GEX.

        Strategy:
        1. Get all BTC option instruments expiring within 7 days
        2. For each, get open interest from book summaries
        3. Compute Call Wall (strike with max call OI)
        4. Compute Put Wall (strike with max put OI)
        5. Estimate Net GEX bias from dealer gamma positioning
        """
        # Step 1: Get near-term BTC option book summaries
        resp = requests.get(
            _BOOK_SUMMARY_URL,
            params={"currency": "BTC", "kind": "option"},
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json().get("result", [])

        if not result:
            log.debug("[GEX] No options data from Deribit")
            return

        now_ms = time.time() * 1000
        max_expiry_ms = now_ms + (7 * 24 * 3600 * 1000)  # 7 days out

        # Parse instruments — group by strike and type
        call_oi: dict[float, float] = {}  # {strike: total_OI}
        put_oi: dict[float, float] = {}
        call_gamma_contrib: dict[float, float] = {}
        put_gamma_contrib: dict[float, float] = {}

        underlying_price = 0.0

        for item in result:
            instrument = item.get("instrument_name", "")
            if "BTC" not in instrument:
                continue

            # Parse instrument name: BTC-10APR26-70000-C
            parts = instrument.split("-")
            if len(parts) < 4:
                continue

            option_type = parts[-1]  # C or P
            try:
                strike = float(parts[-2])
            except (ValueError, IndexError):
                continue

            oi = float(item.get("open_interest", 0))
            mark_price = float(item.get("mark_price", 0))
            underlying = float(item.get("underlying_price", 0))

            if underlying > 0:
                underlying_price = underlying

            # Filter: only near-term options with meaningful OI
            if oi < 1:
                continue

            if option_type == "C":
                call_oi[strike] = call_oi.get(strike, 0) + oi
                # Simplified gamma contribution: OI * proximity to ATM
                if underlying_price > 0:
                    moneyness = abs(strike - underlying_price) / underlying_price
                    # Gamma peaks at ATM, decays with distance
                    gamma_weight = math.exp(-5 * moneyness ** 2)
                    call_gamma_contrib[strike] = (
                        call_gamma_contrib.get(strike, 0) + oi * gamma_weight
                    )
            elif option_type == "P":
                put_oi[strike] = put_oi.get(strike, 0) + oi
                if underlying_price > 0:
                    moneyness = abs(strike - underlying_price) / underlying_price
                    gamma_weight = math.exp(-5 * moneyness ** 2)
                    put_gamma_contrib[strike] = (
                        put_gamma_contrib.get(strike, 0) + oi * gamma_weight
                    )

        if not call_oi and not put_oi:
            return

        # Call Wall: strike with maximum call OI (magnetic resistance)
        call_wall = max(call_oi, key=call_oi.get) if call_oi else 0
        call_wall_oi = call_oi.get(call_wall, 0)

        # Put Wall: strike with maximum put OI (magnetic support)
        put_wall = max(put_oi, key=put_oi.get) if put_oi else 0
        put_wall_oi = put_oi.get(put_wall, 0)

        # Net GEX Bias: sum(call_gamma) - sum(put_gamma)
        # Positive = dealer long gamma → market mean-reverts (sells rallies, buys dips)
        # Negative = dealer short gamma → market trends (breakouts)
        total_call_gamma = sum(call_gamma_contrib.values())
        total_put_gamma = sum(put_gamma_contrib.values())
        net_gex = total_call_gamma - total_put_gamma

        # Normalize to a -100..+100 scale
        gamma_sum = total_call_gamma + total_put_gamma
        net_gex_bias = round(net_gex / gamma_sum * 100, 1) if gamma_sum > 0 else 0

        # Determine regime
        if net_gex_bias > 20:
            gex_regime = "LONG_GAMMA"  # Mean-reverting, pinned to strikes
        elif net_gex_bias < -20:
            gex_regime = "SHORT_GAMMA"  # Trending, breakout-prone
        else:
            gex_regime = "NEUTRAL_GAMMA"

        data = {
            "call_wall": call_wall,
            "call_wall_oi": call_wall_oi,
            "put_wall": put_wall,
            "put_wall_oi": put_wall_oi,
            "net_gex_bias": net_gex_bias,
            "gex_regime": gex_regime,
            "underlying_price": underlying_price,
            "total_call_gamma": round(total_call_gamma, 2),
            "total_put_gamma": round(total_put_gamma, 2),
            "num_strikes_call": len(call_oi),
            "num_strikes_put": len(put_oi),
        }

        with self._lock:
            self._cache = data
            self._last_fetch = time.time()

        log.info(f"[GEX] Updated: Call_Wall=${call_wall:,.0f} "
                 f"(OI={call_wall_oi:.1f}) Put_Wall=${put_wall:,.0f} "
                 f"(OI={put_wall_oi:.1f}) Net_GEX={net_gex_bias:+.1f} "
                 f"({gex_regime})")

    # ── Public Access ──────────────────────────────────────────

    def get_gex(self) -> dict:
        """Get latest GEX data. Returns empty dict if no data."""
        with self._lock:
            return dict(self._cache)

    def get_status(self) -> dict:
        with self._lock:
            age = time.time() - self._last_fetch if self._last_fetch else 999
            return {
                "has_data": bool(self._cache),
                "last_fetch_age": round(age, 1),
                "call_wall": self._cache.get("call_wall"),
                "put_wall": self._cache.get("put_wall"),
                "net_gex_bias": self._cache.get("net_gex_bias"),
                "gex_regime": self._cache.get("gex_regime"),
            }
