"""
ClobExecutor — Native Polymarket CLOB execution via py-clob-client.

This is the ONLY module that imports `py_clob_client`. The rest of the
codebase interacts with the CLOB exclusively through this wrapper.

Why direct py-clob-client (and not simmer-sdk)?
- Simmer requires `import_market()` per market (≤10/day) — incompatible
  with our 5-minute "flash" BTC up/down markets that are generated on
  the fly.
- py-clob-client talks straight to the Polymarket CTF on Polygon, so
  there is no upstream gate. Every outcome token ID is addressable.
- Native support for delegated proxy wallets (Google/Email login) via
  `signature_type=1` (POLY_PROXY) — see `web3_auth.py`.

Supports:
- Market orders (FOK) — for sniper fire and survival exits.
- Limit orders (GTC) — for the Maker Ladder rungs.
- Order cancellation — single, per-token, or global.
- Order status queries.

Fail-safe: if `py-clob-client` is not installed, WALLET_PRIVATE_KEY /
POLYMARKET_PROXY_FUNDER is missing, or the credential derivation fails,
`from_config()` returns None and the engine degrades to offline_sim.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from ..config import TRADING_MODE, log
from .web3_auth import get_clob_client


# ── Data classes (public interface — unchanged for callers) ────

@dataclass
class OrderResult:
    """Result of a single order submission."""
    success: bool = False
    order_id: str = ""         # native Polymarket hash (0x...)
    fill_price: float = 0.0    # cents
    fill_size: float = 0.0     # tokens
    status: str = "failed"     # live | matched | pending | partial | failed
    error: str = ""


@dataclass
class LadderRung:
    """A single rung of a Maker Ladder."""
    price_cents: float     # limit price in cents (e.g., 48.0)
    usd_amount: float      # USD to spend on this rung
    ttl_seconds: int = 60  # time-to-live for the order


@dataclass
class LadderResult:
    """Result of a Maker Ladder submission."""
    rungs_submitted: int = 0
    rungs_filled: int = 0
    total_filled_usd: float = 0.0
    avg_fill_price: float = 0.0
    order_ids: list = field(default_factory=list)
    errors: list = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────

def _is_sell(side: str) -> bool:
    """True when `side` means we want to exit/sell a held position."""
    s = (side or "").strip().lower()
    return s in ("sell", "close", "exit")


def _side_constant(side: str):
    """Map an internal side label to the py-clob-client BUY/SELL constant.

    - "UP" / "DOWN" / "BUY" / "OPEN" → BUY
      (we are BUYING the outcome token — the UP/DOWN selector is baked
      into the token_id itself, not the order side).
    - "SELL" / "CLOSE" / "EXIT"     → SELL
    """
    from py_clob_client.order_builder.constants import BUY, SELL  # type: ignore
    return SELL if _is_sell(side) else BUY


def _parse_post_response(resp, fallback_reason: str = "") -> OrderResult:
    """Normalize a py-clob-client post_order() response into an OrderResult.

    The CLOB returns (spec per Polymarket docs):
        {
          "errorMsg": "",
          "orderID": "0xabcdef...",        ← native order hash
          "takingAmount": "...",           ← string decimals
          "makingAmount": "...",
          "status": "matched" | "live" | "delayed" | "unmatched",
          "success": true,
          "transactionsHashes": ["0x..."]  ← present on immediate match
        }
    """
    if resp is None:
        return OrderResult(error=f"empty response from CLOB ({fallback_reason})")

    # py-clob-client usually returns a dict; cover the object path too.
    def _g(key, default=None):
        if isinstance(resp, dict):
            return resp.get(key, default)
        return getattr(resp, key, default)

    err = _g("errorMsg", "") or _g("error", "") or ""
    raw_status = (_g("status", "") or "").lower()
    success = bool(_g("success", False)) and not err

    order_id = str(_g("orderID", "") or _g("order_id", "") or "")

    # Fill tracking: py-clob-client returns size/price on the signed
    # order object, not on post_order. We stamp what we can: if status
    # is "matched" the full size filled at the limit price; otherwise
    # the caller leaves fill_price=0 and relies on later get_order()
    # polling (or Survival WS prices) to reconcile.
    status = raw_status or ("live" if success else "failed")
    if status == "matched":
        normalized_status = "filled"
    elif status in ("delayed", "live", "open"):
        normalized_status = "pending"
    elif status == "unmatched":
        normalized_status = "failed"
    else:
        normalized_status = status if status else ("pending" if success else "failed")

    return OrderResult(
        success=success and bool(order_id),
        order_id=order_id,
        fill_price=0.0,  # populated later by order polling or Survival WS
        fill_size=0.0,
        status=normalized_status,
        error=str(err or ""),
    )


# ── Main wrapper ───────────────────────────────────────────────

class ClobExecutor:
    """Thin wrapper over `py_clob_client.client.ClobClient`.

    Thread-safe. All public methods are safe to call from the engine
    tick loop, sniper evaluator, or survival monitor threads.
    """

    def __init__(self, client, mode: str):
        """Use `ClobExecutor.from_config()` instead of direct init."""
        self._client = client
        self._mode = mode
        self._lock = threading.Lock()
        self._connected = True
        # Track active limit orders for cancellation bookkeeping.
        self._active_orders: dict[str, dict] = {}  # order_id -> metadata

    @classmethod
    def from_config(cls) -> Optional["ClobExecutor"]:
        """Factory: build a ClobExecutor from environment config.

        Returns None (and the engine falls back to offline_sim) if:
        - TRADING_MODE is offline_sim (or legacy paper_simmer)
        - py-clob-client is not installed
        - WALLET_PRIVATE_KEY / POLYMARKET_PROXY_FUNDER are missing
        - credential derivation fails
        """
        if TRADING_MODE in ("offline_sim", "paper_simmer"):
            log.info(f"[EXEC] TRADING_MODE={TRADING_MODE} — ClobExecutor disabled")
            return None

        client = get_clob_client()
        if client is None:
            log.warning("[EXEC] py-clob-client unavailable — offline_sim fallback")
            return None

        log.info(f"[EXEC] ClobExecutor ready (py-clob-client, mode={TRADING_MODE})")
        return cls(client, TRADING_MODE)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> str:
        return self._mode

    # ── Market order (FOK taker) ─────────────────────────────

    def submit_market(self, token_id: str, side: str, usd_amount: float,
                      reasoning: str = "") -> OrderResult:
        """Submit a market order (Fill-Or-Kill against resting liquidity).

        Args:
            token_id: Polymarket CTF outcome-token ID (UP or DOWN token).
            side: "BUY" / "UP" / "DOWN" → BUY the outcome;
                  "SELL" / "EXIT"       → SELL a held position.
            usd_amount: BUY → USD notional to spend; SELL → token count.
            reasoning: free-text reason; included in the bot log prefix.
                (Polymarket CLOB has no native audit-trail field — we
                log it here so ops can grep for the order hash.)
        """
        with self._lock:
            return self._submit_market_internal(token_id, side, usd_amount, reasoning)

    def _submit_market_internal(self, token_id: str, side: str,
                                usd_amount: float, reasoning: str) -> OrderResult:
        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType  # type: ignore

            side_const = _side_constant(side)

            # Provide explicit slippage tolerance to the matching engine.
            # Without this, py-clob-client calculates the exact top-of-book price,
            # which Polymarket's backend often rejects for FAK/FOK. 
            worst_price = 0.99 if side_const == "BUY" else 0.01
            
            # MarketOrderArgs semantics in py-clob-client:
            #   amount on BUY  = USD notional (dollars)
            #   amount on SELL = outcome-token shares
            market_args = MarketOrderArgs(
                token_id=str(token_id),
                amount=float(usd_amount),
                side=side_const,
                price=worst_price,
            )
            signed_order = self._client.create_market_order(market_args)
            resp = self._client.post_order(signed_order, OrderType.FAK)
            result = _parse_post_response(resp, fallback_reason="market FAK")

            if result.success:
                log.info(
                    f"[EXEC] Market {side} → CLOB order hash: "
                    f"{result.order_id} | ${usd_amount:.2f} | status={result.status} "
                    f"| reason='{reasoning[:60]}'"
                )
            else:
                log.warning(
                    f"[EXEC] Market order failed ({side} {token_id[:12]}...): "
                    f"{result.error[:200]}"
                )
            return result
        except Exception as e:
            log.error(f"[EXEC] Market order exception: {e}")
            return OrderResult(error=str(e))

    # ── Limit order (GTC maker, for ladder rungs) ────────────

    def submit_limit(self, token_id: str, side: str, price: float, usd_amount: float,
                     reasoning: str = "") -> OrderResult:
        """Submit a GTC limit order.

        Args:
            token_id: CTF outcome-token ID.
            side: "UP" / "DOWN" / "BUY" → BUY; "SELL" → SELL.
            price: Limit price in DECIMAL (e.g., 0.48 for 48c).
            usd_amount: USD to deploy at this price. Converted internally
                to token size = usd_amount / price.
            reasoning: free-text reason logged alongside the order hash.
        """
        with self._lock:
            return self._submit_limit_internal(token_id, side, price, usd_amount, reasoning)

    def _submit_limit_internal(self, token_id: str, side: str, price: float,
                               usd_amount: float, reasoning: str) -> OrderResult:
        try:
            if price <= 0 or usd_amount <= 0:
                return OrderResult(error="Invalid price/amount for limit order")

            from py_clob_client.clob_types import OrderArgs, OrderType  # type: ignore

            side_const = _side_constant(side)

            # OrderArgs.size is in outcome-token shares, not USD.
            size_tokens = round(float(usd_amount) / float(price), 4)
            if size_tokens <= 0:
                return OrderResult(error="Computed token size is zero")

            order_args = OrderArgs(
                price=float(price),
                size=float(size_tokens),
                side=side_const,
                token_id=str(token_id),
            )
            signed_order = self._client.create_order(order_args)
            resp = self._client.post_order(signed_order, OrderType.GTC)
            result = _parse_post_response(resp, fallback_reason="limit GTC")

            if result.success:
                self._active_orders[result.order_id] = {
                    "token_id": token_id,
                    "price": price,
                    "usd": usd_amount,
                    "size": size_tokens,
                    "created_at": time.time(),
                }
                log.info(
                    f"[EXEC] Limit {side} posted → CLOB order hash: "
                    f"{result.order_id} | ${usd_amount:.2f} ({size_tokens:.2f} tk) "
                    f"@ {price:.4f} | reason='{reasoning[:60]}'"
                )
                if result.status not in ("filled", "partial"):
                    result.status = "pending"
            else:
                log.warning(f"[EXEC] Limit order failed: {result.error[:200]}")
            return result
        except Exception as e:
            log.error(f"[EXEC] Limit order exception: {e}")
            return OrderResult(error=str(e))

    # ── Maker Ladder ─────────────────────────────────────────

    def submit_maker_ladder(self, token_id: str, side: str,
                            total_usd: float,
                            rungs: list[LadderRung],
                            reasoning: str = "") -> LadderResult:
        """Submit a Maker Ladder: N limit orders at the configured prices.

        Crash-safe: a per-rung exception (network hiccup, CLOB 5xx,
        invalid price rounding) is caught and appended to ``errors``
        without tearing down the rest of the ladder or the engine tick
        loop. Callers can inspect ``rungs_submitted`` to see if anything
        landed; the engine treats a 0-rung ladder as a tolerable skip
        rather than a fatal error.
        """
        result = LadderResult()
        n = len(rungs)
        base_reason = reasoning or f"ladder {side} ${total_usd:.2f}"
        for i, rung in enumerate(rungs, start=1):
            try:
                price_decimal = rung.price_cents / 100.0
                rung_reason = (f"{base_reason} - Rung {i}/{n} @ "
                               f"{rung.price_cents:.1f}c")
                r = self.submit_limit(token_id, side, price_decimal,
                                      rung.usd_amount,
                                      reasoning=rung_reason)
                if r.success:
                    result.rungs_submitted += 1
                    result.order_ids.append(r.order_id)
                else:
                    result.errors.append(r.error)
            except Exception as rung_err:
                # Defensive: a single rung failure MUST NOT abort the
                # rest of the ladder (the whole point of maker rungs is
                # to degrade gracefully when liquidity is thin).
                err_str = f"rung {i}/{n} @ {rung.price_cents:.1f}c: {rung_err}"
                log.warning(f"[EXEC] Ladder rung {i}/{n} raised "
                            f"(continuing): {rung_err}")
                result.errors.append(err_str)

        status_tag = ("OK" if result.rungs_submitted == n
                      else "PARTIAL" if result.rungs_submitted > 0
                      else "ALL-FAILED")
        log.info(f"[EXEC] Ladder submitted [{status_tag}]: "
                 f"{result.rungs_submitted}/{n} rungs for {side} | "
                 f"total=${total_usd:.2f}"
                 f"{f' | errors={result.errors[:3]}' if result.errors else ''}")
        return result

    # ── Cancellation ─────────────────────────────────────────

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID via ClobClient.cancel()."""
        with self._lock:
            try:
                # py-clob-client exposes cancel(order_id=...) which takes
                # a single ID; cancel_orders(order_ids=[...]) for batch.
                self._client.cancel(order_id=order_id)
                self._active_orders.pop(order_id, None)
                log.debug(f"[EXEC] Cancelled order {order_id}")
                return True
            except Exception as e:
                log.warning(f"[EXEC] Cancel failed for {order_id}: {e}")
                return False

    def cancel_all(self) -> int:
        """Cancel ALL active orders. Used on shutdown / window resolve."""
        with self._lock:
            all_ids = list(self._active_orders.keys())

        cancelled = 0
        try:
            # Native batch: drop every live order for this API key.
            self._client.cancel_all()
            cancelled = len(all_ids)
            with self._lock:
                self._active_orders.clear()
        except Exception as e:
            log.warning(f"[EXEC] cancel_all native call failed, falling back: {e}")
            for oid in all_ids:
                if self.cancel_order(oid):
                    cancelled += 1

        if cancelled:
            log.info(f"[EXEC] Cancelled ALL {cancelled} active orders")
        return cancelled

    # ── Queries ──────────────────────────────────────────────

    def get_balance_usdc(self) -> float:
        """USDC balance query — Live Dynamic Sizing entry point.

        Tries every shape py-clob-client has historically exposed for
        reading the funder's on-CLOB USDC.e balance:

            - ``client.get_balance_allowance(...)`` (newer releases,
              returns a dict with ``balance`` in atomic USDC units)
            - ``client.get_balance()``                (older shim)

        Any exception / unsupported SDK path degrades gracefully to -1.0
        so callers (Kelly sizing, Live Dynamic Sizing toggle) can treat
        the result as "unknown — fall back to paper balance".

        Returns:
            Balance in USDC (float). Returns -1.0 on any failure.
        """
        try:
            # ── Path A: modern py-clob-client ──────────────────
            # get_balance_allowance needs BalanceAllowanceParams; pass
            # a COLLATERAL probe so we read the wallet's USDC.e, not a
            # per-token CTF position.
            if hasattr(self._client, "get_balance_allowance"):
                try:
                    from py_clob_client.clob_types import (  # type: ignore
                        BalanceAllowanceParams, AssetType,
                    )
                    params = BalanceAllowanceParams(
                        asset_type=AssetType.COLLATERAL,
                    )
                    resp = self._client.get_balance_allowance(params)
                    if resp is None:
                        raise RuntimeError("empty response")
                    raw = None
                    if isinstance(resp, dict):
                        raw = resp.get("balance", resp.get("Balance"))
                    else:
                        raw = getattr(resp, "balance", None)
                    if raw is None:
                        raise RuntimeError(
                            f"unexpected shape: {type(resp).__name__}")
                    # USDC has 6 decimals — scale from atomic units.
                    bal = float(raw) / 1_000_000.0
                    log.info(f"[Live Sizing] CLOB USDC balance probe = "
                             f"${bal:.2f} (via get_balance_allowance)")
                    return bal
                except Exception as inner:
                    log.debug(f"[Live Sizing] get_balance_allowance path "
                              f"failed, trying legacy: {inner}")

            # ── Path B: legacy get_balance() shim ──────────────
            if hasattr(self._client, "get_balance"):
                raw = self._client.get_balance()
                if raw is None:
                    raise RuntimeError("get_balance returned None")
                # Some versions return a float already, others atomic.
                try:
                    bal = float(raw)
                    # Heuristic: values > 1e4 almost certainly atomic.
                    if bal > 10_000:
                        bal = bal / 1_000_000.0
                except (TypeError, ValueError):
                    raise RuntimeError(f"unparseable: {raw!r}")
                log.info(f"[Live Sizing] CLOB USDC balance probe = "
                         f"${bal:.2f} (via legacy get_balance)")
                return bal

            log.warning("[Live Sizing] py-clob-client exposes neither "
                        "get_balance_allowance nor get_balance — "
                        "returning -1 (unknown)")
            return -1.0

        except Exception as e:
            log.warning(f"[Live Sizing] Balance probe failed cleanly: "
                        f"{e} — returning -1 (unknown)")
            return -1.0
