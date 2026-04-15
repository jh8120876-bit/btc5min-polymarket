# Observability — Prometheus Metrics & Grafana

> Institutional-grade metrics export for real-time monitoring.

## Architecture

```
observability/
└── prometheus_exporter.py   # Gauge/Counter/Histogram via prometheus_client
```

## Prometheus Exporter

- HTTP server on `PROMETHEUS_PORT` (default 9108)
- Guarded by `PROMETHEUS_ENABLED` flag — zero cost when disabled
- Metrics recorded from engine.py hooks (post-resolve, post-bet)
- 12 initial metrics: winrate, P&L, balance, confidence, slippage, judge prob, PTB edge, fill rate, profit-lock/flip-stop counters, circuit breaker state

## Dependencies

- **External**: `prometheus-client`
- **Internal**: `..config` (flags, port)
- **No dependency** on execution, AI, or data feeds

## Adding New Metrics

1. Define in `prometheus_exporter.py` (Gauge/Counter/Histogram)
2. Call `metrics.record_*()` from the source module
3. Update this CLAUDE.md
