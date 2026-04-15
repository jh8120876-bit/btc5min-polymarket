"""
SMC Feature Engine — Smart Money Concepts indicators via smartmoneyconcepts library.

Computes FVG, BOS/CHoCH, Order Blocks, Liquidity pools, Retracements,
Kill Zones, and SMC-based S/R from Binance 5m OHLCV klines.

All functions are safe-fail: return empty/default if data is insufficient
or the library is unavailable.
"""

import threading
import time
from datetime import datetime, timezone
from typing import Optional

from ..config import log

try:
    import pandas as pd
    import numpy as np
    from .vendor_smc import smc
    _SMC_AVAILABLE = True
except ImportError as e:
    _SMC_AVAILABLE = False
    log.warning(f"[SMC] pandas/numpy not available — SMC features disabled ({e})")


# ICT Kill Zone schedule (UTC hours)
_KILL_ZONES = {
    "asian":        (0, 4),    # 00:00-04:00 UTC
    "london_open":  (6, 9),    # 06:00-09:00 UTC
    "ny_open":      (11, 14),  # 11:00-14:00 UTC (NY kill zone)
    "london_close": (14, 16),  # 14:00-16:00 UTC
}

# ── SMC Candle Cache (megarefactor B3) ──
# Avoids recomputing O(N²) SMC features every 2s tick.
# Cache key = fingerprint of last candle close price + count.
# TTL = 60s (one new 5m candle at most).
_smc_cache: dict = {}
_smc_cache_ts: float = 0
_SMC_CACHE_TTL = 60.0  # seconds
_smc_cache_lock = threading.Lock()


def _build_df(ohlcv: list[dict]) -> Optional["pd.DataFrame"]:
    """Convert list of OHLCV dicts to pandas DataFrame."""
    if not _SMC_AVAILABLE or len(ohlcv) < 10:
        return None
    df = pd.DataFrame(ohlcv)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            return None
    return df


def compute_smc_features(ohlcv: list[dict]) -> dict:
    """Compute all SMC features from 5m OHLCV klines.

    Uses a 60s candle-level cache to avoid recomputing O(N²) operations
    every 2-second tick. Cache invalidates when candle count or last close changes.

    Args:
        ohlcv: List of dicts with keys: open, high, low, close, volume
               Typically ~59 closed 5m candles from Binance.

    Returns:
        Dict of SMC features (all prefixed with smc_). Empty dict if unavailable.
    """
    global _smc_cache, _smc_cache_ts

    # ── Cache check: fingerprint = (candle_count, last_close_price) ──
    if ohlcv:
        fingerprint = (len(ohlcv), round(ohlcv[-1].get("close", 0), 2))
        now = time.time()
        with _smc_cache_lock:
            if (_smc_cache
                    and (now - _smc_cache_ts) < _SMC_CACHE_TTL
                    and _smc_cache.get("_fingerprint") == fingerprint):
                return _smc_cache

    features = {}
    df = _build_df(ohlcv)
    if df is None:
        return features

    try:
        _t0 = time.time()

        # ── 1. Swing Highs/Lows (base for most other indicators) ──
        swing_length = 5  # 5 candles each side for 5m timeframe
        swings = smc.swing_highs_lows(df, swing_length=swing_length)

        # ── 2. Fair Value Gaps ──
        fvg_data = smc.fvg(df, join_consecutive=True)
        _extract_fvg_features(features, fvg_data, df)

        # ── 3. BOS / CHoCH (Market Structure) ──
        bos_choch_data = smc.bos_choch(df, swings, close_break=True)
        _extract_structure_features(features, bos_choch_data)

        # ── 4. Order Blocks ──
        ob_data = smc.ob(df, swings, close_mitigation=False)
        _extract_ob_features(features, ob_data, df)

        # ── 5. Liquidity Pools ──
        liq_data = smc.liquidity(df, swings, range_percent=0.01)
        _extract_liquidity_features(features, liq_data, df)

        # ── 6. Retracements ──
        ret_data = smc.retracements(df, swings)
        _extract_retracement_features(features, ret_data)

        # ── Opción C: SMC-based Support/Resistance (additional, not replacing) ──
        _extract_smc_sr(features, swings, ob_data, df)

        # ── Opción D: Kill Zone detection ──
        _extract_kill_zone(features)

        elapsed_ms = round((time.time() - _t0) * 1000, 1)
        features["smc_compute_ms"] = elapsed_ms
        log.debug(f"[SMC] Features computed in {elapsed_ms}ms "
                  f"({len(ohlcv)} candles, {len(features)} features)")

    except Exception as e:
        log.warning(f"[SMC] Feature computation error (safe-fail): {e}")

    # ── Store in cache ──
    if ohlcv and features:
        features["_fingerprint"] = (len(ohlcv), round(ohlcv[-1].get("close", 0), 2))
        with _smc_cache_lock:
            _smc_cache = features
            _smc_cache_ts = time.time()

    return features


def build_smc_prompt_block(features: dict, current_price: float) -> str:
    """Build a text block summarizing SMC state for the DeepSeek prompt.

    Returns empty string if no meaningful SMC data.
    """
    if not features or not features.get("smc_fvg_active"):
        if not features:
            return ""

    lines = ["[SMC ANALYSIS (computed)]"]

    # FVG
    fvg = features.get("smc_fvg_active", 0)
    if fvg != 0:
        direction = "BULLISH" if fvg == 1 else "BEARISH"
        top = features.get("smc_fvg_top", 0)
        bottom = features.get("smc_fvg_bottom", 0)
        mitigated = features.get("smc_fvg_mitigated", 0)
        lines.append(f"FVG: {direction} ${bottom:,.0f}-${top:,.0f}"
                     f"{' (MITIGATED)' if mitigated else ' (OPEN)'}")

    # Market Structure
    bos = features.get("smc_bos_last", 0)
    choch = features.get("smc_choch_last", 0)
    if choch != 0:
        lines.append(f"CHoCH: {'BULLISH' if choch == 1 else 'BEARISH'} "
                     f"(cambio de carácter detectado)")
    elif bos != 0:
        lines.append(f"BOS: {'BULLISH' if bos == 1 else 'BEARISH'} "
                     f"(ruptura de estructura)")

    # Order Blocks
    ob = features.get("smc_ob_nearest", 0)
    if ob != 0:
        ob_top = features.get("smc_ob_top", 0)
        ob_bottom = features.get("smc_ob_bottom", 0)
        ob_dir = "BULLISH" if ob == 1 else "BEARISH"
        dist_pct = features.get("smc_ob_distance_pct", 0)
        strength = features.get("smc_ob_strength", 0)
        lines.append(f"OB: {ob_dir} ${ob_bottom:,.0f}-${ob_top:,.0f} "
                     f"(dist: {dist_pct:+.3f}%, strength: {strength:.0f}%)")

    # Liquidity
    liq = features.get("smc_liq_nearest", 0)
    if liq != 0:
        liq_level = features.get("smc_liq_level", 0)
        liq_dir = "above (sell-side)" if liq == 1 else "below (buy-side)"
        swept = features.get("smc_liq_swept", 0)
        lines.append(f"Liquidity: {liq_dir} @ ${liq_level:,.0f}"
                     f"{' (SWEPT)' if swept else ''}")

    # Retracement
    ret_pct = features.get("smc_retracement_pct", 0)
    if ret_pct > 0:
        ret_dir = features.get("smc_retracement_dir", 0)
        deep = features.get("smc_retracement_deepest", 0)
        lines.append(f"Retracement: {ret_pct:.1f}% "
                     f"({'from high' if ret_dir == 1 else 'from low'}, "
                     f"deepest: {deep:.1f}%)")

    # SMC S/R (Opción C — additional)
    smc_sup = features.get("smc_support", 0)
    smc_res = features.get("smc_resistance", 0)
    if smc_sup and smc_res:
        lines.append(f"SMC_S/R: Soporte=${smc_sup:,.0f} "
                     f"Resistencia=${smc_res:,.0f}")

    # Kill Zone
    kz = features.get("smc_kill_zone")
    if kz:
        lines.append(f"Kill_Zone: {kz} (alta actividad institucional)")

    if len(lines) <= 1:
        return ""

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Internal extraction helpers
# ═══════════════════════════════════════════════════════════════════════

def _extract_fvg_features(features: dict, fvg_data: "pd.DataFrame",
                          df: "pd.DataFrame"):
    """Extract most recent active FVG."""
    fvg_col = fvg_data["FVG"].values
    valid = np.where(~np.isnan(fvg_col))[0]

    if len(valid) == 0:
        features["smc_fvg_active"] = 0
        features["smc_fvg_top"] = 0.0
        features["smc_fvg_bottom"] = 0.0
        features["smc_fvg_mitigated"] = 0
        features["smc_fvg_count_bull"] = 0
        features["smc_fvg_count_bear"] = 0
        return

    # Most recent FVG
    last_idx = valid[-1]
    features["smc_fvg_active"] = int(fvg_col[last_idx])
    features["smc_fvg_top"] = float(fvg_data["Top"].iloc[last_idx])
    features["smc_fvg_bottom"] = float(fvg_data["Bottom"].iloc[last_idx])
    mit_idx = fvg_data["MitigatedIndex"].iloc[last_idx]
    features["smc_fvg_mitigated"] = 1 if (not np.isnan(mit_idx) and mit_idx > 0) else 0

    # Count active FVGs in last 20 candles
    recent = fvg_col[-20:]
    features["smc_fvg_count_bull"] = int(np.nansum(recent == 1))
    features["smc_fvg_count_bear"] = int(np.nansum(recent == -1))


def _extract_structure_features(features: dict, bos_choch: "pd.DataFrame"):
    """Extract most recent BOS and CHoCH signals."""
    bos_col = bos_choch["BOS"].values
    choch_col = bos_choch["CHOCH"].values

    # Last BOS
    bos_valid = np.where(~np.isnan(bos_col))[0]
    features["smc_bos_last"] = int(bos_col[bos_valid[-1]]) if len(bos_valid) > 0 else 0
    features["smc_bos_level"] = float(bos_choch["Level"].iloc[bos_valid[-1]]) if len(bos_valid) > 0 else 0.0

    # Last CHoCH
    choch_valid = np.where(~np.isnan(choch_col))[0]
    features["smc_choch_last"] = int(choch_col[choch_valid[-1]]) if len(choch_valid) > 0 else 0
    features["smc_choch_level"] = float(bos_choch["Level"].iloc[choch_valid[-1]]) if len(choch_valid) > 0 else 0.0

    # Recency: how many candles since last structure event
    last_struct = max(
        bos_valid[-1] if len(bos_valid) > 0 else 0,
        choch_valid[-1] if len(choch_valid) > 0 else 0,
    )
    features["smc_structure_age"] = len(bos_col) - 1 - last_struct if last_struct > 0 else -1


def _extract_ob_features(features: dict, ob_data: "pd.DataFrame",
                         df: "pd.DataFrame"):
    """Extract nearest unmitigated Order Block relative to current price."""
    ob_col = ob_data["OB"].values
    valid = np.where(~np.isnan(ob_col))[0]

    if len(valid) == 0:
        features["smc_ob_nearest"] = 0
        features["smc_ob_top"] = 0.0
        features["smc_ob_bottom"] = 0.0
        features["smc_ob_distance_pct"] = 0.0
        features["smc_ob_strength"] = 0.0
        features["smc_ob_count"] = 0
        return

    current_price = df["close"].iloc[-1]

    # Find nearest unmitigated OB
    best_idx = None
    best_dist = float("inf")
    for idx in reversed(valid):
        mit = ob_data["MitigatedIndex"].iloc[idx]
        if not np.isnan(mit) and mit > 0:
            continue  # Already mitigated
        ob_mid = (ob_data["Top"].iloc[idx] + ob_data["Bottom"].iloc[idx]) / 2
        dist = abs(current_price - ob_mid)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    if best_idx is not None:
        features["smc_ob_nearest"] = int(ob_col[best_idx])
        features["smc_ob_top"] = float(ob_data["Top"].iloc[best_idx])
        features["smc_ob_bottom"] = float(ob_data["Bottom"].iloc[best_idx])
        ob_mid = (features["smc_ob_top"] + features["smc_ob_bottom"]) / 2
        features["smc_ob_distance_pct"] = round(
            (current_price - ob_mid) / current_price * 100, 6
        )
        features["smc_ob_strength"] = float(ob_data["Percentage"].iloc[best_idx])
    else:
        features["smc_ob_nearest"] = 0
        features["smc_ob_top"] = 0.0
        features["smc_ob_bottom"] = 0.0
        features["smc_ob_distance_pct"] = 0.0
        features["smc_ob_strength"] = 0.0

    # Count active (unmitigated) OBs
    count = 0
    for idx in valid:
        mit = ob_data["MitigatedIndex"].iloc[idx]
        if np.isnan(mit) or mit == 0:
            count += 1
    features["smc_ob_count"] = count


def _extract_liquidity_features(features: dict, liq_data: "pd.DataFrame",
                                df: "pd.DataFrame"):
    """Extract nearest liquidity pool."""
    liq_col = liq_data["Liquidity"].values
    valid = np.where(~np.isnan(liq_col))[0]

    if len(valid) == 0:
        features["smc_liq_nearest"] = 0
        features["smc_liq_level"] = 0.0
        features["smc_liq_swept"] = 0
        features["smc_liq_distance_pct"] = 0.0
        return

    current_price = df["close"].iloc[-1]

    # Find nearest unswept liquidity
    best_idx = None
    best_dist = float("inf")
    for idx in reversed(valid):
        swept = liq_data["Swept"].iloc[idx]
        if not np.isnan(swept) and swept > 0:
            continue
        level = liq_data["Level"].iloc[idx]
        dist = abs(current_price - level)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    if best_idx is not None:
        features["smc_liq_nearest"] = int(liq_col[best_idx])
        features["smc_liq_level"] = float(liq_data["Level"].iloc[best_idx])
        features["smc_liq_swept"] = 0
        features["smc_liq_distance_pct"] = round(
            (current_price - features["smc_liq_level"]) / current_price * 100, 6
        )
    else:
        # All swept — report the most recent one
        last = valid[-1]
        features["smc_liq_nearest"] = int(liq_col[last])
        features["smc_liq_level"] = float(liq_data["Level"].iloc[last])
        features["smc_liq_swept"] = 1
        features["smc_liq_distance_pct"] = round(
            (current_price - features["smc_liq_level"]) / current_price * 100, 6
        )


def _extract_retracement_features(features: dict, ret_data: "pd.DataFrame"):
    """Extract current retracement state."""
    last = len(ret_data) - 1
    direction = ret_data["Direction"].iloc[last]
    current_ret = ret_data["CurrentRetracement%"].iloc[last]
    deepest_ret = ret_data["DeepestRetracement%"].iloc[last]

    features["smc_retracement_dir"] = int(direction)
    features["smc_retracement_pct"] = float(current_ret)
    features["smc_retracement_deepest"] = float(deepest_ret)

    # OTE zone: 62-79% retracement (optimal trade entry)
    features["smc_in_ote_zone"] = 1 if 62 <= current_ret <= 79 else 0


def _extract_smc_sr(features: dict, swings: "pd.DataFrame",
                    ob_data: "pd.DataFrame", df: "pd.DataFrame"):
    """Opción C — SMC-based S/R using swing levels + order blocks.

    Additional to existing TA S/R. Not replacing anything.
    Uses the most recent swing high as resistance and swing low as support,
    refined by nearest order block zones.
    """
    current_price = df["close"].iloc[-1]
    hl = swings["HighLow"].values
    levels = swings["Level"].values

    # Find nearest swing high above price (resistance)
    resistance = None
    for i in range(len(hl) - 1, -1, -1):
        if hl[i] == 1 and levels[i] > current_price:
            resistance = float(levels[i])
            break

    # Find nearest swing low below price (support)
    support = None
    for i in range(len(hl) - 1, -1, -1):
        if hl[i] == -1 and levels[i] < current_price:
            support = float(levels[i])
            break

    # Refine with OB zones if they're closer
    ob_col = ob_data["OB"].values
    valid_ob = np.where(~np.isnan(ob_col))[0]
    for idx in reversed(valid_ob):
        mit = ob_data["MitigatedIndex"].iloc[idx]
        if not np.isnan(mit) and mit > 0:
            continue
        ob_top = float(ob_data["Top"].iloc[idx])
        ob_bottom = float(ob_data["Bottom"].iloc[idx])
        if ob_col[idx] == 1 and ob_top < current_price:
            # Bullish OB below price → potential support
            if support is None or ob_top > support:
                support = ob_top
        elif ob_col[idx] == -1 and ob_bottom > current_price:
            # Bearish OB above price → potential resistance
            if resistance is None or ob_bottom < resistance:
                resistance = ob_bottom

    features["smc_support"] = support or 0.0
    features["smc_resistance"] = resistance or 0.0
    if support and support > 0:
        features["smc_price_vs_support_pct"] = round(
            (current_price - support) / current_price * 100, 4
        )
    else:
        features["smc_price_vs_support_pct"] = 0.0
    if resistance and resistance > 0:
        features["smc_price_vs_resistance_pct"] = round(
            (current_price - resistance) / current_price * 100, 4
        )
    else:
        features["smc_price_vs_resistance_pct"] = 0.0


def _extract_kill_zone(features: dict):
    """Opción D — ICT Kill Zone detection based on current UTC hour."""
    now = datetime.now(timezone.utc)
    hour = now.hour
    active_kz = None
    for kz_name, (start, end) in _KILL_ZONES.items():
        if start <= hour < end:
            active_kz = kz_name
            break

    features["smc_kill_zone"] = active_kz or ""
    features["smc_in_kill_zone"] = 1 if active_kz else 0


# ═══════════════════════════════════════════════════════════════════════
# Mid-Window: Market Structure Shift (MSS) & Real-Time Liquidity Sweeps
# ═══════════════════════════════════════════════════════════════════════

def compute_midwindow_mss(ohlcv: list[dict],
                          window_open_price: float,
                          current_price: float,
                          daily_high: float = 0,
                          daily_low: float = 0) -> dict:
    """
    Compute real-time Market Structure Shift (MSS) and liquidity sweep
    flags using intra-window price action.

    Called mid-candle (every ~10-30s) with the latest 5m OHLCV history
    plus live price data.

    Args:
        ohlcv: Last ~59 closed 5m candles (Binance OHLCV).
        window_open_price: Price at the start of current 5-min window.
        current_price: Latest tick price.
        daily_high: Today's session high (for daily-level sweep detection).
        daily_low: Today's session low.

    Returns:
        Dict with MSS/sweep flags for immediate sniper use.
    """
    result = {
        "is_liquidity_swept_1m": 0,
        "sweep_direction": "",        # "BULL_SWEEP" or "BEAR_SWEEP"
        "sweep_magnitude_pct": 0.0,
        "mss_detected": 0,
        "mss_direction": "",          # "BULLISH_MSS" or "BEARISH_MSS"
        "daily_level_pierced": 0,
        "daily_level_type": "",       # "HIGH" or "LOW"
        "post_sweep_reversal": 0,     # Price reversed after sweep
    }

    if not _SMC_AVAILABLE or not ohlcv or len(ohlcv) < 10:
        return result

    try:
        df = _build_df(ohlcv)
        if df is None:
            return result

        # ── 1. Identify recent swing highs/lows as liquidity levels ──
        swing_length = 3  # Tighter swings for 5m scalping
        swings = smc.swing_highs_lows(df, swing_length=swing_length)
        hl = swings["HighLow"].values
        levels = swings["Level"].values

        # Recent swing highs (sell-side liquidity) and lows (buy-side liquidity)
        recent_highs = []
        recent_lows = []
        for i in range(max(0, len(hl) - 20), len(hl)):
            if np.isnan(hl[i]):
                continue
            if hl[i] == 1:  # Swing high
                recent_highs.append(float(levels[i]))
            elif hl[i] == -1:  # Swing low
                recent_lows.append(float(levels[i]))

        # ── 2. Check if current price swept a liquidity level ──
        # Sweep = price pierced ABOVE a swing high (took buy stops) or
        #         pierced BELOW a swing low (took sell stops)
        sweep_threshold_pct = 0.015  # Must pierce by at least 0.015%

        for sh in recent_highs:
            if sh <= 0:
                continue
            pierce_pct = (current_price - sh) / sh * 100
            if pierce_pct > sweep_threshold_pct:
                result["is_liquidity_swept_1m"] = 1
                result["sweep_direction"] = "BEAR_SWEEP"  # Swept highs = bearish intent
                result["sweep_magnitude_pct"] = round(pierce_pct, 4)
                break

        if not result["is_liquidity_swept_1m"]:
            for sl in recent_lows:
                if sl <= 0:
                    continue
                pierce_pct = (sl - current_price) / sl * 100
                if pierce_pct > sweep_threshold_pct:
                    result["is_liquidity_swept_1m"] = 1
                    result["sweep_direction"] = "BULL_SWEEP"  # Swept lows = bullish intent
                    result["sweep_magnitude_pct"] = round(pierce_pct, 4)
                    break

        # ── 3. Market Structure Shift detection ──
        # MSS = after a sweep, price breaks the most recent opposite swing level,
        # confirming a directional change.
        bos_choch = smc.bos_choch(df, swings, close_break=True)
        choch_col = bos_choch["CHOCH"].values
        choch_valid = np.where(~np.isnan(choch_col))[0]

        if len(choch_valid) > 0:
            last_choch_idx = choch_valid[-1]
            recency = len(choch_col) - 1 - last_choch_idx
            # CHoCH in last 5 candles (25 min) is "recent" for MSS context
            if recency <= 5:
                choch_dir = int(choch_col[last_choch_idx])
                result["mss_detected"] = 1
                result["mss_direction"] = (
                    "BULLISH_MSS" if choch_dir == 1 else "BEARISH_MSS"
                )

        # ── 4. Daily level piercing (session high/low) ──
        if daily_high > 0 and current_price > daily_high:
            result["daily_level_pierced"] = 1
            result["daily_level_type"] = "HIGH"
        elif daily_low > 0 and current_price < daily_low:
            result["daily_level_pierced"] = 1
            result["daily_level_type"] = "LOW"

        # ── 5. Post-sweep reversal confirmation ──
        # If price swept a level but has already reversed back through it,
        # this confirms the sweep was a trap (strong entry signal).
        if result["is_liquidity_swept_1m"] and window_open_price > 0:
            move_from_open = (current_price - window_open_price) / window_open_price * 100
            if result["sweep_direction"] == "BEAR_SWEEP" and move_from_open < -0.01:
                # Swept highs, now falling → confirmed bear sweep → entry SHORT
                result["post_sweep_reversal"] = 1
            elif result["sweep_direction"] == "BULL_SWEEP" and move_from_open > 0.01:
                # Swept lows, now rising → confirmed bull sweep → entry LONG
                result["post_sweep_reversal"] = 1

    except Exception as e:
        log.warning(f"[SMC] Mid-window MSS computation error (safe-fail): {e}")

    return result


def build_mss_prompt_block(mss: dict) -> str:
    """Build a text block summarizing MSS/sweep state for the AI prompt."""
    if not mss or (not mss.get("is_liquidity_swept_1m")
                   and not mss.get("mss_detected")):
        return ""

    lines = ["[SMC MID-WINDOW — Liquidity Sweep & MSS]"]

    if mss.get("is_liquidity_swept_1m"):
        sd = mss["sweep_direction"]
        mag = mss["sweep_magnitude_pct"]
        implied = "SHORT" if "BEAR" in sd else "LONG"
        lines.append(
            f"** LIQUIDITY SWEPT: {sd} ({mag:+.4f}%) — "
            f"liquidez institucional tomada, sesgo {implied}"
        )
        if mss.get("post_sweep_reversal"):
            lines.append(
                ">> POST-SWEEP REVERSAL CONFIRMADO: "
                "precio revirtió tras la barrida — entrada de alta probabilidad"
            )

    if mss.get("mss_detected"):
        lines.append(
            f">> MSS (Market Structure Shift): {mss['mss_direction']} — "
            f"cambio de carácter confirmado por CHoCH reciente"
        )

    if mss.get("daily_level_pierced"):
        lines.append(
            f">> NIVEL DIARIO PERFORADO: {mss['daily_level_type']} de sesión"
        )

    return "\n".join(lines)
