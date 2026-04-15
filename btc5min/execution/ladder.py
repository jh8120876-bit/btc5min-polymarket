"""
LadderBuilder — Maker limit-order escalator for Polymarket CLOB.

Instead of buying tokens as a Taker (eating the spread), splits the
Kelly-approved budget into multiple limit BUY orders at progressively
cheaper prices. If the market bounces, we fill at the lowest price
without paying spread.

Configuration is hot-reloadable via dynamic_rules.json → "ladder" section.

Example: $10 budget, spread_bps=[200, 500, 800], weights=[0.3, 0.3, 0.4]
  → Rung 1: $3.00 @ (best_ask - 2.0c)
  → Rung 2: $3.00 @ (best_ask - 5.0c)
  → Rung 3: $4.00 @ (best_ask - 8.0c)
"""

from dataclasses import dataclass

from ..config import log
from ..config_manager import rules
from .clob_executor import LadderRung


def _cfg(key, fallback):
    return rules.get("ladder", key, fallback)


def is_ladder_enabled() -> bool:
    return bool(_cfg("enabled", True))


@dataclass
class LadderSpec:
    """Computed ladder specification ready for the executor."""
    rungs: list[LadderRung]
    total_usd: float
    degraded: bool = False  # True if budget too small for full ladder


def build_ladder(
    total_usd: float,
    best_ask_cents: float,
    best_bid_cents: float = 0.0,
) -> LadderSpec:
    """Build a Maker Ladder from the current order book snapshot.

    Behaviour:
        - When **both** bid and ask are known we compute a mid price
          ``(bid + ask) / 2`` and hang rungs progressively BELOW the mid
          (we always BUY the outcome token). This is what protects the
          bot from the "no match" slippage crash we hit in L2 when the
          best_ask was empty — we simply never send a Taker price.
        - When only the ask is known, we fall back to the legacy
          best_ask-minus-bps path so stale snapshots still produce a
          valid ladder.

    Args:
        total_usd:        Kelly-approved budget to spend.
        best_ask_cents:   Best ask price in cents.
        best_bid_cents:   Best bid price in cents (0 if unknown).

    Returns:
        LadderSpec with list of LadderRung and metadata.
    """
    num_rungs = int(_cfg("rungs", 3))
    spread_bps_list = list(_cfg("spread_bps", [200, 500, 800]))
    weights = list(_cfg("weights", [0.3, 0.3, 0.4]))
    ttl = int(_cfg("rung_ttl_seconds", 60))
    slippage_buffer = float(_cfg("slippage_buffer_cents", 1))
    min_per_rung = float(_cfg("min_usd_per_rung", 1.0))

    # Validate weights sum ~1.0
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 0.05:
        log.warning(f"[LADDER] weights sum={weight_sum:.2f} != 1.0 — normalizing")
        weights = [w / weight_sum for w in weights]

    # ── Anchor: mid price if both sides are present, else best_ask ──
    if best_bid_cents > 0 and best_ask_cents > 0 and best_ask_cents >= best_bid_cents:
        mid_cents = (best_bid_cents + best_ask_cents) / 2.0
        anchor = mid_cents
        anchor_tag = f"mid={mid_cents:.2f}c (bid={best_bid_cents:.1f}/ask={best_ask_cents:.1f})"
    else:
        anchor = max(1.0, best_ask_cents)
        anchor_tag = f"best_ask={best_ask_cents:.1f}c (no-bid fallback)"

    # Degrade to single rung if budget too small
    if total_usd < min_per_rung * num_rungs:
        price = max(1.0, anchor - slippage_buffer)
        rung = LadderRung(
            price_cents=round(price, 1),
            usd_amount=round(total_usd, 2),
            ttl_seconds=ttl,
        )
        log.info(f"[LADDER] Budget ${total_usd:.2f} too small for {num_rungs} "
                 f"rungs — degraded to single rung @ {price:.1f}c | {anchor_tag}")
        return LadderSpec(rungs=[rung], total_usd=total_usd, degraded=True)

    # Build multi-rung ladder — rungs cuelgan POR DEBAJO del ancla (we BUY).
    rungs = []
    for i in range(min(num_rungs, len(spread_bps_list), len(weights))):
        bps = spread_bps_list[i]
        weight = weights[i]

        # Price = anchor minus spread offset (in cents; bps/100 = cents).
        # Distributes the budget in small Limit Orders just below the mid
        # so we never pay the Taker "no match" slippage penalty.
        price = anchor - (bps / 100.0)
        price = round(max(1.0, min(99.0, price)), 1)  # clamp to Polymarket range

        usd = round(total_usd * weight, 2)
        usd = max(min_per_rung, usd)

        rungs.append(LadderRung(
            price_cents=price,
            usd_amount=usd,
            ttl_seconds=ttl,
        ))

    actual_total = sum(r.usd_amount for r in rungs)
    log.info(f"[LADDER] Built {len(rungs)} rungs | "
             f"total=${actual_total:.2f} | "
             f"prices=[{', '.join(f'{r.price_cents:.1f}c' for r in rungs)}] | "
             f"anchor={anchor_tag}")

    return LadderSpec(rungs=rungs, total_usd=actual_total)
