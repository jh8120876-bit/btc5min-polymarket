import math


class TechnicalAnalysis:
    """Extended technical analysis: RSI, EMA, MACD, Bollinger, ATR, S/R."""

    @staticmethod
    def _ema(prices: list[float], period: int) -> list[float]:
        if len(prices) < period:
            return []
        mult = 2 / (period + 1)
        ema_vals = [sum(prices[:period]) / period]
        for p in prices[period:]:
            ema_vals.append(p * mult + ema_vals[-1] * (1 - mult))
        return ema_vals

    @staticmethod
    def analyze(prices: list[float], ohlcv_5m: list[dict] | None = None) -> dict:
        defaults = {
            "rsi": 50, "trend": "NEUTRAL", "momentum": 0, "volatility": 0,
            "sma_cross": "NEUTRAL", "sma5": 0, "sma10": 0,
            "ema12": 0, "ema26": 0, "ema_cross": "NEUTRAL",
            "macd": 0, "macd_signal": 0, "macd_histogram": 0,
            "bb_upper": 0, "bb_lower": 0, "bb_middle": 0, "bb_pct": 50,
            "atr": 0, "atr_pct": 0,
            "support": 0, "resistance": 0,
            "price_vs_support_pct": 0, "price_vs_resistance_pct": 0,
            "tick_buy_pressure": 50,
        }
        if len(prices) < 5:
            return defaults

        cur = prices[-1]

        # ── RSI (Wilder's smoothing — matches TradingView) ──
        period = 14
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        if len(deltas) < period:
            # Not enough data: SMA fallback
            avg_g = sum(max(0, d) for d in deltas) / len(deltas) if deltas else 0
            avg_l = sum(max(0, -d) for d in deltas) / len(deltas) if deltas else 0
        else:
            # Seed with SMA of first `period` deltas
            avg_g = sum(max(0, d) for d in deltas[:period]) / period
            avg_l = sum(max(0, -d) for d in deltas[:period]) / period
            # Wilder's recursive smoothing for remaining deltas
            for d in deltas[period:]:
                avg_g = (avg_g * (period - 1) + max(0, d)) / period
                avg_l = (avg_l * (period - 1) + max(0, -d)) / period
        if avg_l == 0:
            rsi = 100.0 if avg_g > 0 else 50.0
        elif avg_g == 0:
            rsi = 0.0
        else:
            rsi = 100 - (100 / (1 + avg_g / avg_l))

        # ── SMA ──
        sma5 = sum(prices[-5:]) / 5
        n10 = min(10, len(prices))
        sma10 = sum(prices[-n10:]) / n10
        sma_cross = "BULLISH" if sma5 > sma10 else "BEARISH" if sma5 < sma10 else "NEUTRAL"

        # ── Momentum ──
        n_mom = min(5, len(prices))
        base_price = prices[-n_mom]
        momentum = ((cur - base_price) / base_price) * 100 if base_price > 0 else 0

        # ── Volatility ──
        mean_p = sum(prices) / len(prices)
        variance = sum((p - mean_p) ** 2 for p in prices) / len(prices)
        volatility = (math.sqrt(variance) / mean_p * 100) if mean_p > 0 else 0

        # ── Trend ──
        up_count = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i - 1])
        ratio = up_count / (len(prices) - 1)
        trend = "BULLISH" if ratio > 0.6 else "BEARISH" if ratio < 0.4 else "NEUTRAL"

        # ── EMA 12/26 ──
        ema12_vals = TechnicalAnalysis._ema(prices, 12)
        ema26_vals = TechnicalAnalysis._ema(prices, 26)
        ema12 = ema12_vals[-1] if ema12_vals else sma5
        ema26 = ema26_vals[-1] if ema26_vals else sma10
        ema_cross = "BULLISH" if ema12 > ema26 else "BEARISH" if ema12 < ema26 else "NEUTRAL"

        # ── MACD ──
        macd_val = ema12 - ema26
        macd_signal = 0.0
        macd_histogram = 0.0
        if len(ema12_vals) >= 9 and len(ema26_vals) >= 1:
            macd_series = []
            offset = len(ema12_vals) - len(ema26_vals)
            for i in range(len(ema26_vals)):
                macd_series.append(ema12_vals[i + offset] - ema26_vals[i])
            signal_vals = TechnicalAnalysis._ema(macd_series, 9)
            if signal_vals:
                macd_signal = signal_vals[-1]
                macd_histogram = macd_val - macd_signal

        # ── Bollinger Bands (20-period) ──
        bb_upper = bb_lower = bb_middle = cur
        bb_pct = 50.0
        n_bb = min(20, len(prices))
        if n_bb >= 5:
            bb_slice = prices[-n_bb:]
            bb_middle = sum(bb_slice) / n_bb
            bb_std = math.sqrt(sum((p - bb_middle) ** 2 for p in bb_slice) / n_bb)
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
            bb_range = bb_upper - bb_lower
            bb_pct = ((cur - bb_lower) / bb_range * 100) if bb_range > 0 else 50

        # ── ATR (True Range from OHLCV when available, close-proxy fallback) ──
        if ohlcv_5m and len(ohlcv_5m) >= 2:
            true_ranges = []
            for i in range(1, len(ohlcv_5m)):
                c = ohlcv_5m[i]
                prev_close = ohlcv_5m[i - 1]["close"]
                tr = max(
                    c["high"] - c["low"],
                    abs(c["high"] - prev_close),
                    abs(c["low"] - prev_close),
                )
                true_ranges.append(tr)
        else:
            true_ranges = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        n_atr = min(14, len(true_ranges))
        atr = sum(true_ranges[-n_atr:]) / n_atr if true_ranges else 0
        atr_pct = (atr / cur * 100) if cur > 0 else 0

        # ── Support / Resistance ──
        n_sr = min(60, len(prices))
        sr_slice = prices[-n_sr:]
        support = min(sr_slice)
        resistance = max(sr_slice)
        price_vs_support_pct = ((cur - support) / support * 100) if support > 0 else 0
        price_vs_resistance_pct = ((resistance - cur) / cur * 100) if cur > 0 else 0

        # ── Tick Buy Pressure (last 30) ──
        n_bp = min(30, len(prices) - 1)
        if n_bp > 0:
            buys = sum(1 for i in range(len(prices) - n_bp, len(prices))
                       if prices[i] > prices[i - 1])
            tick_buy_pressure = round(buys / n_bp * 100)
        else:
            tick_buy_pressure = 50

        return {
            "rsi": round(rsi, 1), "trend": trend,
            "momentum": round(momentum, 4), "volatility": round(volatility, 4),
            "sma_cross": sma_cross, "sma5": round(sma5, 2), "sma10": round(sma10, 2),
            "ema12": round(ema12, 2), "ema26": round(ema26, 2), "ema_cross": ema_cross,
            "macd": round(macd_val, 2), "macd_signal": round(macd_signal, 2),
            "macd_histogram": round(macd_histogram, 2),
            "bb_upper": round(bb_upper, 2), "bb_lower": round(bb_lower, 2),
            "bb_middle": round(bb_middle, 2), "bb_pct": round(bb_pct, 1),
            "atr": round(atr, 2), "atr_pct": round(atr_pct, 4),
            "support": round(support, 2), "resistance": round(resistance, 2),
            "price_vs_support_pct": round(price_vs_support_pct, 4),
            "price_vs_resistance_pct": round(price_vs_resistance_pct, 4),
            "tick_buy_pressure": tick_buy_pressure,
        }
