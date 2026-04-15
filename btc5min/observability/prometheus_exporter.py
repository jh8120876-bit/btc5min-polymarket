"""
Prometheus Exporter — Institutional-grade metrics for btc5min.

Exposes metrics on an HTTP endpoint (default :9108/metrics) for scraping
by Prometheus/Grafana. Guarded by PROMETHEUS_ENABLED flag — zero cost
when disabled.

Usage from engine.py:
    from btc5min.observability.prometheus_exporter import metrics
    metrics.record_window(agent_id, outcome, ...)
    metrics.record_bet(agent_id, side, amount, ...)
"""

from ..config import PROMETHEUS_ENABLED, PROMETHEUS_PORT, log

# ── Lazy init: only import prometheus_client if enabled ──
_initialized = False
_server_started = False

# Metric handles (set during _init_metrics)
_windows_total = None
_winrate_pct = None
_pnl_usd_total = None
_balance_usd = None
_swarm_confidence = None
_slippage_cents = None
_judge_prob = None
_ptb_edge_usd = None
_ladder_fill_rate = None
_profit_lock_total = None
_flip_stop_total = None
_circuit_breaker_active = None
_sniper_fire_latency_ms = None
_pending_retry_total = None


def _init_metrics():
    """Initialize Prometheus metrics (called once on first use)."""
    global _initialized
    global _windows_total, _winrate_pct, _pnl_usd_total, _balance_usd
    global _swarm_confidence, _slippage_cents, _judge_prob
    global _ptb_edge_usd, _ladder_fill_rate
    global _profit_lock_total, _flip_stop_total, _circuit_breaker_active
    global _sniper_fire_latency_ms, _pending_retry_total

    if _initialized:
        return
    _initialized = True

    try:
        from prometheus_client import Counter, Gauge, Histogram

        _windows_total = Counter(
            "btc5min_windows_total",
            "Windows resolved per agent",
            ["agent_id", "outcome"],
        )
        _winrate_pct = Gauge(
            "btc5min_winrate_pct",
            "Rolling winrate percentage",
            ["agent_id", "scope"],
        )
        _pnl_usd_total = Gauge(
            "btc5min_pnl_usd_total",
            "Cumulative P&L in USD",
            ["agent_id"],
        )
        _balance_usd = Gauge(
            "btc5min_balance_usd",
            "Live agent balance in USD",
            ["agent_id"],
        )
        _swarm_confidence = Histogram(
            "btc5min_swarm_confidence",
            "Distribution of prediction confidence",
            ["direction"],
            buckets=[50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        )
        _slippage_cents = Histogram(
            "btc5min_slippage_cents",
            "Fill slippage in cents (fill_price - snapshot_price)",
            ["side"],
            buckets=[0, 0.5, 1, 2, 3, 5, 10],
        )
        _judge_prob = Histogram(
            "btc5min_judge_prob",
            "XGBoost Judge P(correct) distribution",
            ["decision"],
            buckets=[0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
        )
        _ptb_edge_usd = Histogram(
            "btc5min_ptb_edge_usd",
            "PTB strike-spot distance in USD",
            ["verdict"],
            buckets=[0, 2, 5, 10, 20, 50, 100],
        )
        _ladder_fill_rate = Gauge(
            "btc5min_ladder_fill_rate_pct",
            "Percentage of ladder USD filled vs requested",
        )
        _profit_lock_total = Counter(
            "btc5min_profit_lock_triggers_total",
            "Number of Profit-Lock early exits",
        )
        _flip_stop_total = Counter(
            "btc5min_flip_stop_triggers_total",
            "Number of Flip-Stop bail-outs",
        )
        _circuit_breaker_active = Gauge(
            "btc5min_circuit_breaker_active",
            "Whether a circuit breaker is active (0/1)",
            ["source"],
        )
        _sniper_fire_latency_ms = Histogram(
            "btc5min_sniper_fire_latency_ms",
            "Elapsed ms from sniper trigger decision to bet commit",
            ["trigger"],
            buckets=[10, 25, 50, 100, 200, 400, 800, 1500, 3000],
        )

        _pending_retry_total = Counter(
            "btc5min_pending_retry_total",
            "Watchful Hold PENDING_RETRY lifecycle events",
            ["event", "veto_kind"],
        )

        log.info(f"[PROMETHEUS] Metrics initialized ({PROMETHEUS_PORT})")

    except ImportError:
        log.warning("[PROMETHEUS] prometheus_client not installed — metrics disabled")


def _start_server():
    """Start the Prometheus HTTP server (once)."""
    global _server_started
    if _server_started:
        return
    _server_started = True

    try:
        from prometheus_client import start_http_server
        start_http_server(PROMETHEUS_PORT)
        log.info(f"[PROMETHEUS] HTTP server started on :{PROMETHEUS_PORT}")
    except ImportError:
        pass
    except OSError as e:
        log.warning(f"[PROMETHEUS] Could not start server on :{PROMETHEUS_PORT}: {e}")


class MetricsRecorder:
    """Facade for recording metrics. No-ops when Prometheus is disabled."""

    def __init__(self):
        self._enabled = PROMETHEUS_ENABLED

    def start(self):
        """Initialize metrics and start HTTP server if enabled."""
        if not self._enabled:
            return
        _init_metrics()
        _start_server()

    def record_window(self, agent_id: str, outcome: str):
        """Record a resolved window. outcome = 'win' | 'loss' | 'skip'."""
        if not self._enabled or _windows_total is None:
            return
        _windows_total.labels(agent_id=agent_id, outcome=outcome).inc()

    def record_pnl(self, agent_id: str, total_pnl: float):
        if not self._enabled or _pnl_usd_total is None:
            return
        _pnl_usd_total.labels(agent_id=agent_id).set(total_pnl)

    def record_balance(self, agent_id: str, balance: float):
        if not self._enabled or _balance_usd is None:
            return
        _balance_usd.labels(agent_id=agent_id).set(balance)

    def record_confidence(self, direction: str, confidence: float):
        if not self._enabled or _swarm_confidence is None:
            return
        _swarm_confidence.labels(direction=direction).observe(confidence)

    def record_judge_prob(self, decision: str, prob: float):
        if not self._enabled or _judge_prob is None:
            return
        _judge_prob.labels(decision=decision).observe(prob)

    def record_ptb_edge(self, verdict: str, edge_usd: float):
        if not self._enabled or _ptb_edge_usd is None:
            return
        _ptb_edge_usd.labels(verdict=verdict).observe(abs(edge_usd))

    def record_ladder_fill_rate(self, pct: float):
        if not self._enabled or _ladder_fill_rate is None:
            return
        _ladder_fill_rate.set(pct)

    def record_profit_lock(self):
        if not self._enabled or _profit_lock_total is None:
            return
        _profit_lock_total.inc()

    def record_flip_stop(self):
        if not self._enabled or _flip_stop_total is None:
            return
        _flip_stop_total.inc()

    def record_circuit_breaker(self, source: str, active: bool):
        if not self._enabled or _circuit_breaker_active is None:
            return
        _circuit_breaker_active.labels(source=source).set(1 if active else 0)

    def record_sniper_fire_latency(self, trigger: str, latency_ms: float):
        if not self._enabled or _sniper_fire_latency_ms is None:
            return
        _sniper_fire_latency_ms.labels(trigger=trigger).observe(latency_ms)

    def record_pending_retry(self, event: str, veto_kind: str):
        """Watchful Hold lifecycle. event = 'triggered' | 'upgraded' | 'timeout'."""
        if not self._enabled or _pending_retry_total is None:
            return
        _pending_retry_total.labels(event=event, veto_kind=veto_kind).inc()

    def inc_counter(self, name: str, labels: dict | None = None):
        """Dispatcher used by code that names counters as strings."""
        labels = labels or {}
        if name == "pending_retry_triggered_total":
            self.record_pending_retry("triggered", labels.get("veto", "unknown"))
        elif name == "pending_retry_upgraded_total":
            self.record_pending_retry("upgraded", labels.get("veto", "unknown"))
        elif name == "pending_retry_timeout_total":
            self.record_pending_retry("timeout", labels.get("veto", "unknown"))


# ── Module-level singleton ──
metrics = MetricsRecorder()
