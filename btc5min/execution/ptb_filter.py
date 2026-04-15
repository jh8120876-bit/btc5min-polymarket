"""
PTB (Price-To-Beat) Filter — Mathematical strike-vs-spot gatekeeper.

Evaluates whether the AI's proposed side (UP/DOWN) is mathematically
plausible given the strike price extracted from Polymarket and the
current Binance spot price, within the remaining time-to-live of the
window.

Verdicts:
  allow    — PTB is reachable; proceed with normal sizing.
  penalize — PTB is stretched; reduce sizing by penalize_sizing_factor.
  block    — PTB is unreachable; recycle to force_explore=$1.

All thresholds are hot-reloadable via dynamic_rules.json → "ptb_filter".
"""

from dataclasses import dataclass

from ..config import log
from ..config_manager import rules


@dataclass
class PtbVerdict:
    """Result of PTB evaluation."""
    action: str          # "allow" | "penalize" | "block"
    reason: str
    edge_usd: float      # strike - spot (signed)
    sizing_factor: float  # 1.0 for allow, <1.0 for penalize, 0.0 for block


def _cfg(key: str, fallback):
    return rules.get("ptb_filter", key, fallback)


def evaluate_ptb(
    side: str,
    strike: float,
    spot: float,
    ttl_seconds: float,
    atr_pct: float,
) -> PtbVerdict:
    """Evaluate whether the proposed side is compatible with the PTB.

    Parameters
    ----------
    side : str
        "UP" or "DOWN"
    strike : float
        Target price from Polymarket market (USD).
    spot : float
        Current Binance BTC/USD spot price.
    ttl_seconds : float
        Seconds remaining in the current 5-min window.
    atr_pct : float
        ATR as a fraction of price (e.g., 0.003 for 0.3%).

    Returns
    -------
    PtbVerdict with action, reason, edge_usd, sizing_factor.
    """
    if not _cfg("enabled", True):
        return PtbVerdict("allow", "ptb_filter disabled", 0.0, 1.0)

    if strike <= 0 or spot <= 0:
        return PtbVerdict("allow", "no strike or spot data", 0.0, 1.0)

    edge_usd = strike - spot

    # ── Already on the winning side? ──
    if side == "UP" and spot >= strike:
        return PtbVerdict("allow", f"spot ${spot:,.0f} already above strike ${strike:,.0f}", edge_usd, 1.0)
    if side == "DOWN" and spot <= strike:
        return PtbVerdict("allow", f"spot ${spot:,.0f} already below strike ${strike:,.0f}", edge_usd, 1.0)

    # ── Calculate required move ──
    ttl = max(ttl_seconds, 1.0)
    gap_usd = abs(edge_usd)
    pct_gap = gap_usd / spot

    # ATR-based move rate per second (normalized to 5m window = 300s)
    typical_move_per_sec = (atr_pct * spot) / 300.0
    needed_move_per_sec = gap_usd / ttl

    block_mult = _cfg("block_multiplier_atr", 3.0)
    pen_mult = _cfg("penalize_multiplier_atr", 1.5)
    hard_block_usd = _cfg("hard_block_usd", 20.0)
    pen_factor = _cfg("penalize_sizing_factor", 0.5)

    # ── Hard block: absolute USD gap ──
    if gap_usd >= hard_block_usd:
        reason = (f"PTB hard block: ${gap_usd:,.0f} gap "
                  f"(spot=${spot:,.0f} strike=${strike:,.0f}, side={side}) "
                  f">= ${hard_block_usd} threshold")
        log.warning(f"[PTB] {reason}")
        return PtbVerdict("block", reason, edge_usd, 0.0)

    # ── ATR-relative block ──
    if typical_move_per_sec > 0 and needed_move_per_sec > block_mult * typical_move_per_sec:
        reason = (f"PTB ATR block: needs {needed_move_per_sec:.2f}$/s "
                  f"but typical is {typical_move_per_sec:.2f}$/s "
                  f"({needed_move_per_sec/typical_move_per_sec:.1f}x > {block_mult}x) "
                  f"| gap=${gap_usd:,.0f} ttl={ttl:.0f}s")
        log.warning(f"[PTB] {reason}")
        return PtbVerdict("block", reason, edge_usd, 0.0)

    # ── ATR-relative penalize ──
    if typical_move_per_sec > 0 and needed_move_per_sec > pen_mult * typical_move_per_sec:
        reason = (f"PTB penalize: needs {needed_move_per_sec:.2f}$/s "
                  f"({needed_move_per_sec/typical_move_per_sec:.1f}x > {pen_mult}x) "
                  f"| sizing reduced to {pen_factor:.0%}")
        log.info(f"[PTB] {reason}")
        return PtbVerdict("penalize", reason, edge_usd, pen_factor)

    # ── Allow ──
    reason = (f"PTB allow: gap=${gap_usd:,.0f} "
              f"({pct_gap*100:.3f}%) reachable in {ttl:.0f}s")
    log.debug(f"[PTB] {reason}")
    return PtbVerdict("allow", reason, edge_usd, 1.0)
