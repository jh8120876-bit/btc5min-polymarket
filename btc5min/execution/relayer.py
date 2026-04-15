"""Redeem Relayer — DEPRECATED no-op stub.

py-clob-client has no auto_redeem() equivalent. Redemption requires
raw on-chain CTF redeemPositions() call. Manual workaround:
polymarket.com -> Portfolio -> Claim winnings.
"""

import threading

from ..config import TRADING_MODE, log


class RedeemRelayer:
    """No-op stub kept for engine.py import compatibility."""

    def __init__(self, executor=None):
        self._warned = False

    def start(self):
        if TRADING_MODE in ("offline_sim", "paper_simmer"):
            return
        if self._warned:
            return
        self._warned = True
        log.warning(
            "[REDEEM] Relayer is a no-op stub — py-clob-client has no "
            "auto_redeem(). Claim winnings manually via polymarket.com."
        )
