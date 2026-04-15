"""
SurvivalMonitor — Intra-candle Profit-Lock & Flip-Stop parachutes.

Monitors live Polymarket CLOB prices for open positions and triggers
emergency exits:

- **Profit-Lock**: Token probability >= 98% → sell immediately to lock
  profits before potential last-second rug-pulls.
- **Flip-Stop**: Token probability drops to <= 30% → bail out to recover
  20-30% instead of losing 100%.

Runs as a daemon thread, evaluating every ~500ms. Uses the same
PolymarketLiveStream WebSocket prices that the reader already maintains.

All thresholds are hot-reloadable via dynamic_rules.json.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from ..config import log
from ..config_manager import rules
from ..observability.prometheus_exporter import metrics as prom_metrics


@dataclass
class TrackedPosition:
    """A position being monitored by the SurvivalMonitor."""
    token_id: str
    side: str               # "UP" or "DOWN"
    entry_price: float       # price paid (decimal, e.g., 0.55)
    amount_usd: float        # USD spent
    size_tokens: float       # tokens bought (amount / entry_price)
    window_id: int
    window_close_ts: float   # UTC timestamp when window expires
    agent_id: str
    # Filled on exit
    exit_reason: str = ""    # profit_lock | flip_stop
    exit_price: float = 0.0
    exited: bool = False


class SurvivalMonitor:
    """Daemon thread monitoring open positions for emergency exits.

    Thread-safe. The engine calls:
      - track() after a bet is placed
      - untrack_all() when a window resolves
      - stop() on shutdown

    The monitor reads prices from the PolymarketLiveStream and uses
    the ClobExecutor to sell when thresholds are hit.
    """

    def __init__(self, executor, live_stream):
        """
        Args:
            executor: ClobExecutor instance (or None if offline_sim).
            live_stream: PolymarketLiveStream for real-time prices.
        """
        self._executor = executor
        self._live = live_stream
        self._lock = threading.Lock()
        self._positions: dict[int, TrackedPosition] = {}  # window_id -> pos
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the monitoring thread."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("[SURVIVAL] Monitor started")

    def stop(self):
        self._running = False

    def track(self, pos: TrackedPosition):
        """Register a position for monitoring."""
        with self._lock:
            self._positions[pos.window_id] = pos
        log.info(f"[SURVIVAL] Tracking {pos.side} position: "
                 f"wid={pos.window_id} entry={pos.entry_price:.4f} "
                 f"${pos.amount_usd:.2f} ({pos.size_tokens:.2f} tokens)")

    def untrack_all(self, window_id: int = 0):
        """Remove tracking for a resolved window (or all if window_id=0)."""
        with self._lock:
            if window_id:
                self._positions.pop(window_id, None)
            else:
                self._positions.clear()

    def get_exit_info(self, window_id: int) -> Optional[TrackedPosition]:
        """Check if a position was exited early. Returns the position or None."""
        with self._lock:
            pos = self._positions.get(window_id)
            if pos and pos.exited:
                return pos
        return None

    def _loop(self):
        """Main monitoring loop — runs every 500ms."""
        while self._running:
            try:
                self._evaluate_all()
            except Exception as e:
                log.error(f"[SURVIVAL] Evaluation error: {e}")
            time.sleep(0.5)

    def _evaluate_all(self):
        """Evaluate all tracked positions against survival thresholds."""
        now = time.time()

        with self._lock:
            positions = list(self._positions.values())

        for pos in positions:
            if pos.exited:
                continue

            # Get live prices from WebSocket
            up_price, down_price, ws_ts = self._live.get_live_prices()
            ws_age = now - ws_ts if ws_ts > 0 else 999

            # Skip if WS data is stale (>5s)
            if ws_age > 5.0:
                continue

            current_price = up_price if pos.side == "UP" else down_price
            if current_price <= 0:
                continue

            seconds_left = pos.window_close_ts - now
            if seconds_left < 2:
                continue  # too close to natural resolve — let it expire

            # ── Profit-Lock ──
            pl_enabled = rules.get("profit_lock", "enabled", True)
            pl_trigger = rules.get("profit_lock", "trigger_prob", 0.98)
            pl_min_remaining = rules.get("profit_lock", "min_remaining_seconds", 10)
            pl_min_profit = rules.get("profit_lock", "min_profit_usd", 0.5)

            if pl_enabled and current_price >= pl_trigger:
                if seconds_left >= pl_min_remaining:
                    profit = (current_price - pos.entry_price) * pos.size_tokens
                    if profit >= pl_min_profit:
                        self._execute_exit(pos, "profit_lock", current_price)
                        continue

            # ── Flip-Stop ──
            fs_enabled = rules.get("flip_stop", "enabled", True)
            fs_trigger = rules.get("flip_stop", "trigger_prob", 0.30)
            fs_require_fav = rules.get("flip_stop", "require_was_favorite", True)
            fs_fav_threshold = rules.get("flip_stop", "favorite_threshold", 0.60)

            if fs_enabled and current_price <= fs_trigger:
                # Only trigger if the position was originally a "favorite"
                if fs_require_fav and pos.entry_price < fs_fav_threshold:
                    continue  # was never a favorite — skip
                self._execute_exit(pos, "flip_stop", current_price)

    def _execute_exit(self, pos: TrackedPosition, reason: str, price: float):
        """Execute an emergency exit (sell the position).

        The `exited` flag flips inside the lock to guarantee a single caller
        owns the exit path — prevents double-exit from the 500ms poll re-entering
        while the CLOB sell is in flight.
        """
        with self._lock:
            if pos.exited:
                return
            pos.exited = True
            pos.exit_reason = reason
            pos.exit_price = price

        # Record to Prometheus
        if reason == "profit_lock":
            profit = (price - pos.entry_price) * pos.size_tokens
            prom_metrics.record_profit_lock()
            log.info(f"[PROFIT-LOCK] {pos.side} sold at {price:.4f} | "
                     f"profit=${profit:+.2f} | wid={pos.window_id}")
        else:
            recovery = price * pos.size_tokens
            loss = pos.amount_usd - recovery
            prom_metrics.record_flip_stop()
            log.warning(f"[FLIP-STOP] {pos.side} bailed at {price:.4f} | "
                        f"recovered=${recovery:.2f} (loss=${loss:.2f}) | "
                        f"wid={pos.window_id}")

        # Submit sell order to CLOB (if executor available)
        if self._executor and self._executor.connected:
            try:
                sell_result = self._executor.submit_market(
                    pos.token_id, "SELL", pos.size_tokens,
                )
                if sell_result.success:
                    log.info(f"[SURVIVAL] SELL filled: {sell_result.order_id}")
                else:
                    log.warning(f"[SURVIVAL] SELL failed: {sell_result.error} "
                                f"— exit recorded but may not have executed")
            except Exception as e:
                log.error(f"[SURVIVAL] SELL exception: {e}")

        # Update the position in DB (exit_reason, exit_price)
        try:
            from .. import database as db
            conn = db._get_conn()
            conn.execute(
                """UPDATE bets SET exit_reason=?, exit_price=?
                   WHERE window_id=? AND ai_model=?""",
                (reason, round(price * 100, 1), pos.window_id, pos.agent_id),
            )
            conn.commit()
        except Exception as e:
            log.error(f"[SURVIVAL] DB update failed: {e}")
