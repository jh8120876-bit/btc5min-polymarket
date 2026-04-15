"""
Feature engineering for ML-ready prediction data.
Computes derived features from raw price ticks, TA indicators,
multi-timeframe klines, and market microstructure.
"""

import math
from datetime import datetime, timezone


class FeatureEngineer:
    """Computes ML features from price history and TA snapshot."""

    @staticmethod
    def compute(prices: list[float], ta: dict,
                price_history_1h: list[float] | None = None,
                market_ctx: dict | None = None,
                klines_15m: list[float] | None = None,
                klines_1h: list[float] | None = None,
                cvd_data: dict | None = None,
                gex_data: dict | None = None) -> dict:
        """
        Compute all ML features for a prediction window.

        Args:
            prices: recent price list (at least 60 values preferred)
            ta: current TA dict from TechnicalAnalysis.analyze()
            price_history_1h: HF price ticks from last hour (for range position)
            market_ctx: Binance market context dict (optional)
            klines_15m: last 3 closed 15m candle closes (optional)
            klines_1h: last 3 closed 1h candle closes (optional)
        """
        now = datetime.now(timezone.utc)
        features = {
            "hour_utc": now.hour,
            "day_of_week": now.weekday(),
        }

        # ── Log returns at different horizons ──
        if len(prices) >= 2:
            features["price_change_pct"] = (
                (prices[-1] - prices[-2]) / prices[-2] * 100
            ) if prices[-2] > 0 else 0

        features["return_1m"] = _log_return(prices, 6)
        features["return_5m"] = _log_return(prices, 30)
        features["return_15m"] = _log_return(prices, 90)

        # ── Price acceleration (second derivative) ──
        features["price_acceleration"] = _price_acceleration(prices)

        # ── Relative range position (last hour) ──
        hr_prices = price_history_1h if price_history_1h else prices
        features["range_position"] = _range_position(prices[-1], hr_prices)

        # ── Volatility Z-Score ──
        features["volatility_zscore"] = _volatility_zscore(prices)

        # ── Multi-timeframe trend slopes ──
        features["trend_slope_15m"] = _trend_slope(klines_15m)
        features["trend_slope_1h"] = _trend_slope(klines_1h)

        # Trend alignment: both 15m and 1h agree on direction
        slope_15m = features["trend_slope_15m"]
        slope_1h = features["trend_slope_1h"]
        if slope_15m is not None and slope_1h is not None:
            both_up = slope_15m > 0 and slope_1h > 0
            both_down = slope_15m < 0 and slope_1h < 0
            features["trend_alignment"] = 1 if (both_up or both_down) else 0
        else:
            features["trend_alignment"] = None

        # ── Market context pass-through ──
        if market_ctx:
            features["volume_24h"] = market_ctx.get("volume_24h")
            features["funding_rate"] = market_ctx.get("funding_rate")
            features["open_interest"] = market_ctx.get("open_interest")
            features["order_book_imbalance"] = market_ctx.get(
                "order_book_imbalance"
            )

        # ── CVD / Order Flow features ──
        if cvd_data:
            features["cvd_imbalance_pct"] = cvd_data.get("cvd_imbalance_pct")
            features["cvd_net"] = cvd_data.get("cvd_net")
            features["cvd_total_vol"] = cvd_data.get("cvd_total_vol")
            features["cvd_trade_count"] = cvd_data.get("cvd_trade_count")
            # Derived: aggressive delta ratio (buy_count / total)
            bc = cvd_data.get("cvd_buy_count", 0)
            sc = cvd_data.get("cvd_sell_count", 0)
            total_c = bc + sc
            features["cvd_aggressor_ratio"] = (
                round(bc / total_c, 4) if total_c > 0 else 0.5
            )

        # ── GEX / Gamma Exposure features ──
        if gex_data:
            features["gex_call_wall"] = gex_data.get("call_wall")
            features["gex_put_wall"] = gex_data.get("put_wall")
            features["gex_net_bias"] = gex_data.get("net_gex_bias")
            # Price distance to magnetic levels
            if prices and gex_data.get("call_wall") and gex_data.get("put_wall"):
                cp = prices[-1]
                cw = gex_data["call_wall"]
                pw = gex_data["put_wall"]
                if cp > 0 and cw > 0:
                    features["gex_dist_call_wall_pct"] = round(
                        (cw - cp) / cp * 100, 4
                    )
                if cp > 0 and pw > 0:
                    features["gex_dist_put_wall_pct"] = round(
                        (pw - cp) / cp * 100, 4
                    )

        return features

    @staticmethod
    def atr_quality_check(ta: dict, current_price: float,
                          rolling_atr_baseline: float | None) -> tuple[str, str]:
        """
        Check if the expected 5-min move is within ATR noise.

        Uses a rolling 24h baseline of average 5-min price changes.
        If the current ATR is below the rolling baseline, the signal
        is noise — mark as LOW_QUALITY.

        Returns:
            (quality: str, reason: str)
            quality: "LOW_QUALITY" or "NORMAL"
        """
        atr_pct = ta.get("atr_pct", 0)

        if rolling_atr_baseline is None:
            # No baseline yet — use static fallback
            if atr_pct < 0.005:
                return ("LOW_QUALITY",
                        f"ATR {atr_pct:.4f}% < static threshold 0.005% — "
                        f"movement within noise")
            return ("NORMAL", "")

        # Dynamic threshold: if current ATR < 50% of rolling average,
        # the market is unusually quiet → low quality signal
        threshold = rolling_atr_baseline * 0.5
        if atr_pct < threshold:
            return ("LOW_QUALITY",
                    f"ATR {atr_pct:.4f}% < dynamic threshold "
                    f"{threshold:.4f}% (50% of 24h avg {rolling_atr_baseline:.4f}%) "
                    f"— movement within noise")

        # High quality: ATR significantly above baseline
        if atr_pct > rolling_atr_baseline * 1.5:
            return ("HIGH_QUALITY",
                    f"ATR {atr_pct:.4f}% > 1.5x baseline "
                    f"{rolling_atr_baseline:.4f}% — strong move expected")

        return ("NORMAL", "")

    @staticmethod
    def get_trend_direction(klines_15m: list[float] | None,
                            klines_1h: list[float] | None) -> str | None:
        """
        Get the aligned trend direction from multi-timeframe analysis.
        Returns "UP", "DOWN", or None if not aligned.
        """
        slope_15m = _trend_slope(klines_15m)
        slope_1h = _trend_slope(klines_1h)
        if slope_15m is None or slope_1h is None:
            return None
        if slope_15m > 0 and slope_1h > 0:
            return "UP"
        if slope_15m < 0 and slope_1h < 0:
            return "DOWN"
        return None


# ── Private helpers ──────────────────────────────────────────

def _log_return(prices: list[float], lookback: int) -> float | None:
    """Compute log return over `lookback` ticks."""
    if len(prices) < lookback + 1:
        return None
    p_now = prices[-1]
    p_past = prices[-(lookback + 1)]
    if p_past <= 0 or p_now <= 0:
        return None
    return math.log(p_now / p_past) * 100


def _price_acceleration(prices: list[float]) -> float | None:
    """Second derivative of price (normalized)."""
    if len(prices) < 6:
        return None
    n = min(10, len(prices) // 2)
    mid = len(prices) - n
    v1 = (prices[mid] - prices[mid - n]) / n if mid >= n else 0
    v2 = (prices[-1] - prices[-n - 1]) / n
    acceleration = v2 - v1
    if prices[-1] > 0:
        return round(acceleration / prices[-1] * 1e6, 4)
    return None


def _range_position(current: float, prices: list[float]) -> float | None:
    """Where is price within the hour's range? 0=low, 100=high."""
    if not prices or current <= 0:
        return None
    high = max(prices)
    low = min(prices)
    rng = high - low
    if rng <= 0:
        return 50.0
    return round((current - low) / rng * 100, 2)


def _volatility_zscore(prices: list[float], window: int = 30) -> float | None:
    """Z-score of current volatility vs recent volatility."""
    if len(prices) < window + 5:
        return None
    returns = []
    for i in range(5, len(prices)):
        if prices[i - 5] > 0:
            r = (prices[i] - prices[i - 5]) / prices[i - 5]
            returns.append(r)
    if len(returns) < 10:
        return None
    current_return = returns[-1]
    mean_r = sum(returns[:-1]) / len(returns[:-1])
    variance = sum((r - mean_r) ** 2 for r in returns[:-1]) / len(returns[:-1])
    std_r = variance ** 0.5
    if std_r < 1e-10:
        return 0.0
    return round((current_return - mean_r) / std_r, 3)


def _trend_slope(closes: list[float] | None) -> float | None:
    """
    Compute trend slope from candle closes using linear regression slope.
    Positive = uptrend, negative = downtrend.
    Normalized as percentage change per candle.
    """
    if not closes or len(closes) < 2:
        return None
    n = len(closes)
    # Simple linear regression slope
    x_mean = (n - 1) / 2
    y_mean = sum(closes) / n
    numerator = sum((i - x_mean) * (closes[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0 or y_mean == 0:
        return 0.0
    slope = numerator / denominator
    # Normalize as % change per candle
    return round(slope / y_mean * 100, 6)
