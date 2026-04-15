import hmac
import json
import os
import threading
import time
from functools import wraps
from pathlib import Path

from flask import Flask, render_template, jsonify, request

from .config import log
from .models import Bet
from .engine import PredictionEngine, get_primary_agent_id
from . import database as db
from . import config as cfg
from .config_manager import rules
from .ai import swarm
from . import sentiment

_engine = None

# ── File write lock for dynamic_rules.json (kills race conditions) ──
_rules_file_lock = threading.Lock()


def get_engine() -> PredictionEngine:
    """Lazy singleton — instantiated once on first access, not on import."""
    global _engine
    if _engine is None:
        _engine = PredictionEngine()
    return _engine


# Whitelist of environment variable names that API keys can be injected into
_ALLOWED_API_KEY_ENVS = frozenset({
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
})

# ── API Key Authentication ──
# Set BTC5MIN_API_KEY env var to protect all mutating endpoints.
# If unset, auth is disabled (backward-compatible).
_API_AUTH_KEY = os.environ.get("BTC5MIN_API_KEY", "")


def _require_api_auth(f):
    """Decorator: reject requests without valid X-API-Key header (if configured)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if _API_AUTH_KEY:
            provided = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(provided, _API_AUTH_KEY):
                return jsonify({"ok": False, "error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


def _write_rules_file(current: dict) -> None:
    """Thread-safe atomic write of dynamic_rules.json."""
    rules_path = Path(__file__).resolve().parent.parent / "dynamic_rules.json"
    with _rules_file_lock:
        current.setdefault("_meta", {})["last_updated"] = time.strftime("%Y-%m-%d")
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
    rules.force_reload()


def _read_rules_file() -> dict:
    """Thread-safe read of dynamic_rules.json."""
    rules_path = Path(__file__).resolve().parent.parent / "dynamic_rules.json"
    with _rules_file_lock:
        with open(rules_path, "r", encoding="utf-8") as f:
            return json.load(f)


def create_app() -> Flask:
    engine = get_engine()
    app = Flask(
        __name__,
        template_folder="../templates",
    )

    # ── Pages ──────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("dashboard.html", active_page="dashboard")

    @app.route("/history")
    def history():
        return render_template("history.html", active_page="history")

    @app.route("/brain")
    def brain():
        return render_template("brain.html", active_page="brain")

    @app.route("/config")
    def config_page():
        return render_template("config.html", active_page="config")

    # ── API ────────────────────────────────────────────────

    @app.route("/api/state")
    def api_state():
        return jsonify(engine.get_state())

    @app.route("/api/logs")
    def api_logs():
        """Tail the rotating log file.

        Query params:
          - file: "main" (default) | "errors"
          - lines: int, default 500, max 5000
          - level: optional substring filter (e.g. "ERROR", "SNIPER")
        """
        import os as _os
        log_dir = _os.getenv(
            "LOG_DIR",
            _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "logs"),
        )
        which = (request.args.get("file") or "main").lower()
        fname = "errors.log" if which == "errors" else "btc5min.log"
        path = _os.path.join(log_dir, fname)
        if not _os.path.isfile(path):
            return jsonify({"error": "log file not found"}), 404
        try:
            n = max(1, min(5000, int(request.args.get("lines", 500))))
        except ValueError:
            n = 500
        level_filter = (request.args.get("level") or "").strip()
        try:
            # Efficient tail: seek from end in chunks
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                block = 8192
                data = b""
                while size > 0 and data.count(b"\n") <= n:
                    read = min(block, size)
                    size -= read
                    f.seek(size)
                    data = f.read(read) + data
                text = data.decode("utf-8", errors="replace")
            tail = text.splitlines()[-n:]
            if level_filter:
                tail = [ln for ln in tail if level_filter in ln]
            return jsonify({
                "file": fname,
                "path": path,
                "lines": len(tail),
                "bytes": _os.path.getsize(path),
                "content": tail,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/manual_bet", methods=["POST"])
    @_require_api_auth
    def api_manual_bet():
        """Place a manual simulated bet (clamped to $1-$5)."""
        data = request.get_json() or {}
        side = data.get("side", "").upper()
        amount = float(data.get("amount", 0))
        if side not in ("UP", "DOWN"):
            return jsonify({"ok": False, "error": "side must be UP or DOWN"}), 400
        with engine._lock:
            if not engine.window.is_active:
                return jsonify({"ok": False,
                                "error": "No hay ventana activa"}), 400
            if engine.current_bet:
                return jsonify({"ok": False,
                                "error": "Ya hay una apuesta activa"}), 400
            agent_port = db.get_agent_portfolio(get_primary_agent_id())
            _default_bal = float(
                rules.get("defaults", "initial_balance",
                          getattr(cfg, "DEFAULT_INITIAL_BALANCE", 25.0))
            )
            agent_balance = agent_port["balance"] if agent_port else _default_bal
            amount = max(1.0, min(amount, 5.0, agent_balance))
            if amount <= 0:
                return jsonify({"ok": False,
                                "error": "Balance insuficiente"}), 400
            odds = (engine.window.up_odds if side == "UP"
                    else engine.window.down_odds)
            engine.current_bet = Bet(
                side=side, amount=amount, price_cents=odds,
                open_price=engine.window.open_price,
                window_id=engine.window.window_id, timestamp=time.time(),
                is_auto=False,
            )
            payout = round(amount * (100 / max(1, odds)), 2)
            log.info(f"MANUAL BET: {side} ${amount:.2f} @ {odds}c "
                     f"(in-memory only, quarantined from DB)")
            return jsonify({
                "ok": True, "side": side, "amount": amount,
                "odds": odds, "potential_payout": payout,
            })

    # ── Primary AI Circuit Breaker ────────────────────────────

    @app.route("/api/resume_primary", methods=["POST"])
    @_require_api_auth
    def api_resume_primary():
        """Resume primary AI after a circuit breaker halt."""
        result = engine.resume_primary()
        return jsonify({"ok": True, **result})

    # Backward-compatible alias (now also requires auth)
    @app.route("/api/resume_deepseek", methods=["POST"])
    @_require_api_auth
    def api_resume_deepseek_compat():
        return api_resume_primary()

    @app.route("/api/cancel_circuit_breaker", methods=["POST"])
    @_require_api_auth
    def api_cancel_circuit_breaker():
        """Manually cancel the Black Swan circuit breaker."""
        was_active = sentiment.cancel_circuit_breaker()
        return jsonify({"ok": True, "was_active": was_active})

    # ── AI Config endpoint ─────────────────────────────────

    @app.route("/api/config", methods=["GET", "POST"])
    @_require_api_auth
    def api_config():
        if request.method == "GET":
            trust = engine.ai.get_trust_info()
            return jsonify({
                "min_confidence": cfg.MIN_CONFIDENCE,
                "early_bet_threshold": engine.ai.early_bet_threshold,
                "max_bet_pct": cfg.MAX_BET_PCT,
                "high_risk_min_conf": getattr(cfg, '_HIGH_RISK_MIN_CONF', 55),
                "trust_score": trust["trust_score"],
                "bet_multiplier": trust["bet_multiplier"],
                "sessions": trust["sessions"],
                "lessons_count": trust["lessons_count"],
                "has_api_key": bool(engine.ai._get_primary_config().get("api_key")),
            })
        # POST — save config
        data = request.get_json() or {}
        errors = []
        if "min_confidence" in data:
            v = int(data["min_confidence"])
            if 20 <= v <= 80:
                cfg.MIN_CONFIDENCE = v
            else:
                errors.append("min_confidence debe ser 20-80")
        if "early_bet_threshold" in data:
            v = int(data["early_bet_threshold"])
            if 50 <= v <= 95:
                engine.ai.early_bet_threshold = v
                db.save_ai_memory("early_bet_threshold", v)
            else:
                errors.append("early_bet_threshold debe ser 50-95")
        if "max_bet_pct" in data:
            v = float(data["max_bet_pct"])
            if 0.01 <= v <= 0.20:
                cfg.MAX_BET_PCT = v
            else:
                errors.append("max_bet_pct debe ser 0.01-0.20")
        if "high_risk_min_conf" in data:
            v = int(data["high_risk_min_conf"])
            if 30 <= v <= 90:
                cfg._HIGH_RISK_MIN_CONF = v
            else:
                errors.append("high_risk_min_conf debe ser 30-90")
        if errors:
            return jsonify({"ok": False, "error": "; ".join(errors)}), 400
        return jsonify({"ok": True})

    # ── ML Judge toggle ─────────────────────────────────────

    @app.route("/api/toggle_ml_judge", methods=["POST"])
    @_require_api_auth
    def api_toggle_ml_judge():
        """Enable/disable the XGBoost ML Judge filter at runtime."""
        data = request.get_json() or {}
        enabled = bool(data.get("enabled", not cfg.ENABLE_ML_JUDGE))
        cfg.ENABLE_ML_JUDGE = enabled
        log.info(f"ML Judge {'ENABLED' if enabled else 'DISABLED'} via API")
        return jsonify({"ok": True, "enabled": cfg.ENABLE_ML_JUDGE})

    # ── ML Judge Feature Importance ──────────────────────

    @app.route("/api/judge_features")
    def api_judge_features():
        """Return feature importance from loaded XGBoost Judge model."""
        from . import engine as _eng
        model = _eng._judge_model
        if model is None:
            return jsonify({"ok": False, "error": "Judge model not loaded"})
        features = list(_eng._JUDGE_FEATURES)
        importances = model.feature_importances_.tolist()
        # Sort by importance descending
        pairs = sorted(zip(features, importances),
                       key=lambda x: x[1], reverse=True)
        return jsonify({
            "ok": True,
            "features": [{"name": n, "importance": round(v, 4)} for n, v in pairs],
        })

    # ── Local LLM model listing (powers UI dropdowns only) ─

    @app.route("/api/local_llm/models")
    def api_llm_models():
        """List models from a local OpenAI-compatible server.

        Used exclusively by the UI dropdowns when configuring a local
        agent (Primary or Swarm). Local agents themselves are wired via
        dynamic_rules.json — there is no enable/disable flag any more.
        """
        available = engine.local_llm.is_available()
        models = engine.local_llm.list_models() if available else []
        return jsonify({
            "available": available,
            "models": models,
        })

    @app.route("/api/stats_history")
    def api_stats_history():
        """Aggregated stats for the history analytics panel."""
        try:
            limit = request.args.get("limit", 200, type=int)
            data = db.get_stats_history(limit=limit)
            return jsonify(data)
        except Exception as e:
            log.error(f"stats_history error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/ai_history")
    def api_ai_history():
        """Get detailed AI prediction + bet history from DB."""
        try:
            windows = db.get_recent_windows(limit=15)
            bet_stats = db.get_bet_stats()
            accuracy = db.get_historical_accuracy(limit=50)
            return jsonify({
                "windows": windows,
                "bet_stats": bet_stats,
                "accuracy": accuracy,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── Dynamic Rules Hot-Reload ──────────────────────────────

    @app.route("/api/reload_rules", methods=["POST"])
    @_require_api_auth
    def api_reload_rules():
        """Force-reload dynamic_rules.json into memory."""
        try:
            rules.force_reload()
            current = rules.get_all()
            version = current.get("_meta", {}).get("version", "?")
            log.info(f"Dynamic rules reloaded via API (v{version})")
            return jsonify({
                "ok": True,
                "version": version,
                "sections": list(current.keys()),
            })
        except Exception as e:
            log.error(f"Rules reload error: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/rules", methods=["GET"])
    @_require_api_auth
    def api_get_rules():
        """Return current active rules (read-only view, auth required)."""
        current = rules.get_all()
        return jsonify(current)

    @app.route("/api/update_risk_preset", methods=["POST"])
    @_require_api_auth
    def api_update_risk_preset():
        """Update risk parameters based on UI preset slider."""
        data = request.get_json() or {}
        preset = data.get("preset", "calibrated").lower()
        
        # Read atomic state
        current = _read_rules_file()
        
        if preset == "cautious":
            # Very tight filters, prioritizes capital preservation
            current.setdefault("ev_gate", {})["min_ev"] = 1.05
            current["ev_gate"]["coinflip_margin_cents"] = 8
            current.setdefault("market_filters", {})["cvd_divergence_penalty"] = 15
            current["market_filters"]["dead_hour_penalty"] = 0.15
        elif preset == "aggressive":
            # Loose filters, accepts almost any trade the LLM likes
            current.setdefault("ev_gate", {})["min_ev"] = 1.00
            current["ev_gate"]["coinflip_margin_cents"] = 2
            current.setdefault("market_filters", {})["cvd_divergence_penalty"] = 4
            current["market_filters"]["dead_hour_penalty"] = 0.05
        else: # calibrated
            current.setdefault("ev_gate", {})["min_ev"] = 1.02
            current["ev_gate"]["coinflip_margin_cents"] = 5
            current.setdefault("market_filters", {})["cvd_divergence_penalty"] = 8
            current["market_filters"]["dead_hour_penalty"] = 0.10

        # Inject a comment so the user knows UI is controlling this
        current["ev_gate"]["_ui_preset"] = preset
        current["market_filters"]["_ui_preset"] = preset
        
        # Write state and force reload
        _write_rules_file(current)
        
        return jsonify({"ok": True, "preset": preset})

    @app.route("/api/window_trace/<int:window_id>", methods=["GET"])
    def api_window_trace(window_id: int):
        """Reconstruct a per-window timeline: window row, every agent's
        prediction + reasoning + strategy, every bet (real/shadow/ghost)
        + outcome + order_id, market_context snapshot, and liquidations
        during the window. Vibe-Trading inspired replay/debug view.
        """
        conn = db._get_conn()
        wrow = conn.execute(
            "SELECT * FROM windows WHERE window_id = ?", (window_id,)
        ).fetchone()
        if not wrow:
            return jsonify({"ok": False, "error": "window_not_found"}), 404
        window = dict(wrow)

        preds = [dict(r) for r in conn.execute(
            "SELECT * FROM predictions WHERE window_id = ? ORDER BY id",
            (window_id,),
        ).fetchall()]

        bets = [dict(r) for r in conn.execute(
            "SELECT * FROM bets WHERE window_id = ? ORDER BY id",
            (window_id,),
        ).fetchall()]

        mctx_row = conn.execute(
            "SELECT * FROM market_context WHERE window_id = ?", (window_id,)
        ).fetchone()
        market_context = dict(mctx_row) if mctx_row else {}

        open_t = window.get("open_time") or 0
        close_t = window.get("close_time") or (open_t + 300)
        liqs = [dict(r) for r in conn.execute(
            "SELECT timestamp, side, qty, price, usd_value "
            "FROM liquidations WHERE timestamp BETWEEN ? AND ? "
            "ORDER BY timestamp",
            (open_t, close_t),
        ).fetchall()]

        # Per-agent summary for fast UI scan.
        agents: dict[str, dict] = {}
        for p in preds:
            aid = p.get("ai_model") or "?"
            agents.setdefault(aid, {"agent": aid})
            agents[aid]["prediction"] = p.get("prediction")
            agents[aid]["confidence"] = p.get("confidence")
            agents[aid]["strategy"] = p.get("rag_strategy") or p.get("strategy_name")
            agents[aid]["correct"] = p.get("correct")
            agents[aid]["reasoning"] = (p.get("reasoning") or "")[:500]
        for b in bets:
            aid = b.get("ai_model") or "?"
            agents.setdefault(aid, {"agent": aid})
            agents[aid].setdefault("bets", []).append({
                "side": b.get("side"),
                "amount": b.get("amount"),
                "odds_cents": b.get("odds_cents"),
                "profit": b.get("profit"),
                "is_shadow": b.get("is_shadow"),
                "is_ghost": b.get("is_ghost"),
                "exec_status": b.get("exec_status"),
                "order_id": b.get("order_id"),
                "exit_reason": b.get("exit_reason"),
            })

        return jsonify({
            "ok": True,
            "window": window,
            "market_context": market_context,
            "predictions": preds,
            "bets": bets,
            "liquidations": liqs,
            "agents": list(agents.values()),
            "counts": {
                "predictions": len(preds),
                "bets": len(bets),
                "real_bets": sum(1 for b in bets if not b.get("is_ghost") and not b.get("is_shadow")),
                "shadow_bets": sum(1 for b in bets if b.get("is_shadow") and not b.get("is_ghost")),
                "ghost_bets": sum(1 for b in bets if b.get("is_ghost")),
                "liquidations": len(liqs),
            },
        })

    @app.route("/api/confidence_rules", methods=["GET", "POST"])
    @_require_api_auth
    def api_confidence_rules():
        """Get/set confidence-related knobs from dynamic_rules.json (sniper section).

        POST body: {min_confidence, decay_floor_conf, confidence_decay_per_min,
                    decay_mode, ev_flash_fire_threshold}
        All fields optional; only provided keys are written. Persisted to disk
        and hot-reloaded — survives restart.
        """
        _ALLOWED = {
            "min_confidence": (float, 30.0, 95.0),
            "decay_floor_conf": (float, 30.0, 90.0),
            "confidence_decay_per_min": (float, 0.0, 20.0),
            "ev_flash_fire_threshold": (float, 1.00, 2.00),
            "decay_mode": (str, None, None),
        }
        if request.method == "GET":
            sn = rules.get("sniper", None, {}) or {}
            return jsonify({k: sn.get(k) for k in _ALLOWED})

        data = request.get_json() or {}
        current = _read_rules_file()
        sniper = current.setdefault("sniper", {})
        updated = {}
        for k, (typ, lo, hi) in _ALLOWED.items():
            if k not in data:
                continue
            v = data[k]
            try:
                if typ is str:
                    v = str(v).lower()
                    if v not in ("linear", "quadratic"):
                        continue
                else:
                    v = typ(v)
                    if lo is not None and not (lo <= v <= hi):
                        continue
            except Exception:
                continue
            sniper[k] = v
            updated[k] = v
        _write_rules_file(current)
        log.info(f"[confidence_rules] updated: {updated}")
        return jsonify({"ok": True, "updated": updated})

    @app.route("/api/brain_config", methods=["GET", "POST"])
    @_require_api_auth
    def api_brain_config():
        """Unified GET/POST for all AI brain configuration sections.

        GET  → returns all configurable knobs grouped by section.
        POST → accepts partial updates to any section. Hot-reload + persist.
        """
        # ── Section schemas: key → (type, min, max, section_in_rules) ──
        _SCHEMA = {
            # — Confianza General —
            "risk.min_confidence":            (int,   30,  95, "risk", "min_confidence"),
            "risk.high_risk_min_conf":        (int,   30,  90, "risk", "high_risk_min_conf"),
            "risk.kelly_fraction":            (float, 0.1, 1.0, "risk", "kelly_fraction"),
            "risk.max_bet_pct":               (float, 0.01, 0.20, "risk", "max_bet_pct"),
            # — Circuit Breakers —
            "risk.max_drawdown_halt_pct":     (int,   5,  100, "risk", "max_drawdown_halt_pct"),
            "risk.max_drawdown_reduce_pct":   (int,   1,   50, "risk", "max_drawdown_reduce_pct"),
            "risk.daily_loss_limit_pct":      (float, 0.01, 1.0, "risk", "daily_loss_limit_pct"),
            "risk.losing_streak_threshold":   (int,   1,   20, "risk", "losing_streak_threshold"),
            "risk.losing_streak_reduction":   (float, 0.1, 1.0, "risk", "losing_streak_reduction"),
            "risk.low_quality_cap_pct":       (float, 0.0001, 0.05, "risk", "low_quality_cap_pct"),
            "risk.liq_chaos_threshold":       (float, 2.0, 20.0, "risk", "liq_chaos_threshold"),
            # — EV Gate —
            "ev_gate.enabled":                (bool,  None, None, "ev_gate", "enabled"),
            "ev_gate.min_ev":                 (float, 0.90, 2.00, "ev_gate", "min_ev"),
            "ev_gate.min_payout":             (float, 1.00, 3.00, "ev_gate", "min_payout"),
            "ev_gate.coinflip_margin_cents":  (int,   0,   15, "ev_gate", "coinflip_margin_cents"),
            # — Market Filters —
            "market_filters.enabled":                (bool,  None, None, "market_filters", "enabled"),
            "market_filters.funding_rate_threshold": (float, 0.0001, 0.01, "market_filters", "funding_rate_threshold"),
            "market_filters.funding_rate_penalty":   (float, 0.01, 0.50, "market_filters", "funding_rate_penalty"),
            "market_filters.cvd_divergence_penalty": (int,   0,   30, "market_filters", "cvd_divergence_penalty"),
            "market_filters.liquidation_penalty":    (int,   0,   30, "market_filters", "liquidation_penalty"),
            "market_filters.dead_hour_start":        (int,   0,   23, "market_filters", "dead_hour_start"),
            "market_filters.dead_hour_end":          (int,   0,   23, "market_filters", "dead_hour_end"),
            "market_filters.dead_hour_penalty":      (float, 0.0, 0.50, "market_filters", "dead_hour_penalty"),
            # — Trust Score —
            "trust.default":          (int,   10,  90, "trust", "default"),
            "trust.win_bonus":        (int,   1,   20, "trust", "win_bonus"),
            "trust.loss_penalty":     (int,   1,   20, "trust", "loss_penalty"),
            "trust.high_conf_bonus":  (int,   0,   10, "trust", "high_conf_bonus"),
            "trust.min":              (int,   0,   50, "trust", "min"),
            "trust.max":              (int,   50, 100, "trust", "max"),
            # — Second Opinion —
            "second_opinion.enabled":           (bool,  None, None, "second_opinion", "enabled"),
            "second_opinion.timeout_sec":       (float, 1.0, 15.0, "second_opinion", "timeout_sec"),
            "second_opinion.variant_a_enabled": (bool,  None, None, "second_opinion", "variant_a_enabled"),
            "second_opinion.variant_b_enabled": (bool,  None, None, "second_opinion", "variant_b_enabled"),
            "second_opinion.variant_c_enabled": (bool,  None, None, "second_opinion", "variant_c_enabled"),
            "second_opinion.variant_d_enabled": (bool,  None, None, "second_opinion", "variant_d_enabled"),
            "second_opinion.variant_d_launch_sec": (int, 30, 240, "second_opinion", "variant_d_launch_sec"),
            # — Sniper Advanced —
            "sniper.eval_interval_sec":        (int,   3,   30, "sniper", "eval_interval_sec"),
            "sniper.eval_cutoff_sec":          (int,  60,  280, "sniper", "eval_cutoff_sec"),
            "sniper.first_eval_delay_sec":     (int,   5,   60, "sniper", "first_eval_delay_sec"),
            "sniper.consensus_score_fire":     (int,   1,    5, "sniper", "consensus_score_fire"),
            "sniper.consensus_score_flip":     (int,  -5,   -1, "sniper", "consensus_score_flip"),
            "sniper.last_minute_rescue_conf":  (int,  40,   90, "sniper", "last_minute_rescue_conf"),
            "sniper.last_minute_rescue_ev":    (float, 1.00, 2.00, "sniper", "last_minute_rescue_ev"),
        }

        if request.method == "GET":
            current = _read_rules_file()
            result = {}
            for full_key, (typ, lo, hi, section, key) in _SCHEMA.items():
                sec = current.get(section, {})
                val = sec.get(key)
                if val is not None:
                    result[full_key] = val
                elif typ is bool:
                    result[full_key] = True
                elif typ is int:
                    result[full_key] = lo if lo is not None else 0
                else:
                    result[full_key] = lo if lo is not None else 0.0
            
            # Inject early_bet_threshold from engine AI memory instead of static rules file
            if hasattr(app.config.get("engine"), "ai"):
                result["risk.early_bet_threshold"] = app.config["engine"].ai.early_bet_threshold

            return jsonify(result)

        # POST — partial update
        data = request.get_json() or {}
        current = _read_rules_file()
        updated = {}
        for full_key, raw_val in data.items():
            if full_key not in _SCHEMA:
                continue
            typ, lo, hi, section, key = _SCHEMA[full_key]
            try:
                if typ is bool:
                    v = bool(raw_val)
                else:
                    v = typ(raw_val)
                    if lo is not None and hi is not None:
                        if not (lo <= v <= hi):
                            continue
            except (ValueError, TypeError):
                continue
            current.setdefault(section, {})[key] = v
            updated[full_key] = v

        # Handle early_bet_threshold specially (goes to AI memory, not static rules)
        if "risk.early_bet_threshold" in data:
            try:
                v = int(data["risk.early_bet_threshold"])
                if 50 <= v <= 95 and hasattr(app.config.get("engine"), "ai"):
                    app.config["engine"].ai.early_bet_threshold = v
                    from btc5min import database as db
                    db.save_ai_memory("early_bet_threshold", v)
                    updated["risk.early_bet_threshold"] = v
            except Exception:
                pass

        if updated:
            _write_rules_file(current)
            rules.reload()
            log.info(f"[brain_config] updated {len(updated)} params: "
                     f"{list(updated.keys())}")
        return jsonify({"ok": True, "updated": updated})


    @app.route("/api/watchful_hold", methods=["GET", "POST"])
    @_require_api_auth
    def api_watchful_hold():
        """Get/set Watchful Hold settings (hot-reloadable, persisted)."""
        _ALLOWED = {
            "enabled": bool,
            "retry_interval_sec": int,
            "commit_ghost_before_close_sec": int,
            "upgrade_requires_consecutive_ok": int,
            "max_retries": int,
        }
        if request.method == "GET":
            wh = rules.get_section("watchful_hold") or {}
            return jsonify({
                "enabled": bool(wh.get("enabled", False)),
                "retry_interval_sec": int(wh.get("retry_interval_sec", 15)),
                "commit_ghost_before_close_sec": int(
                    wh.get("commit_ghost_before_close_sec", 15)),
                "upgrade_requires_consecutive_ok": int(
                    wh.get("upgrade_requires_consecutive_ok", 2)),
                "max_retries": int(wh.get("max_retries", 10)),
                "retry_soft_vetos": wh.get("retry_soft_vetos",
                    ["ev_low", "low_payout", "coinflip", "judge_borderline"]),
            })
        data = request.get_json() or {}
        current = _read_rules_file()
        wh = current.setdefault("watchful_hold", {})
        updated = {}
        for k, typ in _ALLOWED.items():
            if k not in data:
                continue
            try:
                v = typ(data[k]) if typ is not bool else bool(data[k])
            except Exception:
                continue
            wh[k] = v
            updated[k] = v
        _write_rules_file(current)
        log.info(f"[watchful_hold] updated: {updated}")
        return jsonify({"ok": True, "updated": updated})


    # ── Primary Agent Configuration ────────────────────────────

    @app.route("/api/primary_agent", methods=["GET", "POST"])
    @_require_api_auth
    def api_primary_agent():
        """Get or set the primary agent configuration."""
        if request.method == "GET":
            primary = swarm.get_primary_agent()
            if not primary:
                return jsonify({"error": "No primary agent configured"}), 404
            key_env = primary.get("api_key_env", "")
            return jsonify({
                "agent_id": primary.get("agent_id"),
                "display_name": primary.get("display_name", ""),
                "api_type": primary.get("api_type", ""),
                "model": primary.get("model", ""),
                "api_key_env": key_env,
                "has_api_key": bool(os.environ.get(key_env, "")) if key_env else False,
                "api_base_url": primary.get("api_base_url"),
                "system_prompt_key": primary.get("system_prompt_key", ""),
            })

        # POST — set a new primary agent
        data = request.get_json() or {}
        api_type = data.get("api_type", "")
        model = data.get("model", "")
        api_key_env = data.get("api_key_env", "")
        base_url = data.get("api_base_url") or None
        display_name = data.get("display_name", f"{api_type}/{model}")

        if not api_type or not model:
            return jsonify({"ok": False,
                            "error": "api_type and model are required"}), 400

        # Exclusivity check: if local is chosen as primary,
        # disable any local swarm agents
        if api_type == "local":
            for agent in swarm.get_secondary_agents():
                if agent.get("api_type") == "local":
                    return jsonify({
                        "ok": False,
                        "error": "Local LLM ya esta en uso como agente Swarm. "
                                 "Desactívalo del Swarm antes de asignarlo como Primario."
                    }), 409

        # Inject API key into environment if provided
        if data.get("api_key") and api_key_env:
            if api_key_env in _ALLOWED_API_KEY_ENVS:
                os.environ[api_key_env] = str(data["api_key"])[:256]

        # Update dynamic_rules.json (thread-safe)
        try:
            current = _read_rules_file()

            agents_section = current.get("agents", {})

            # Unmark current primary
            for aid, acfg in agents_section.items():
                if isinstance(acfg, dict) and acfg.get("is_primary"):
                    acfg["is_primary"] = False

            # Create or update the new primary agent entry
            agent_id = data.get("agent_id") or f"{api_type}_{model}".replace("-", "_")[:30]
            _global_default = float(
                rules.get("defaults", "initial_balance",
                          getattr(cfg, "DEFAULT_INITIAL_BALANCE", 25.0))
            )
            if agent_id not in agents_section:
                agents_section[agent_id] = {
                    "enabled": True,
                    "initial_balance": _global_default,
                }
            agents_section[agent_id].update({
                "display_name": display_name,
                "model": model,
                "api_type": api_type,
                "api_key_env": api_key_env,
                "api_base_url": base_url,
                "is_primary": True,
            })

            current["agents"] = agents_section
            _write_rules_file(current)

            # Ensure portfolio exists
            db.upsert_agent_portfolio(
                agent_id, display_name,
                agents_section[agent_id].get("strategy", ""),
                agents_section[agent_id].get("initial_balance", _global_default),
            )

            log.info(f"Primary agent changed to: {agent_id} ({api_type}/{model})")
            return jsonify({"ok": True, "agent_id": agent_id,
                            "api_type": api_type, "model": model})
        except Exception as e:
            log.error(f"Error setting primary agent: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    # ── Multi-Agent Swarm Endpoints ───────────────────────────

    @app.route("/api/agents")
    def api_agents():
        """Return all agent configs + their portfolio state."""
        agents = swarm.get_all_agents()
        portfolios = {p["agent_id"]: p
                      for p in db.get_all_agent_portfolios()}
        # Merge portfolio data into agent configs
        for agent in agents:
            aid = agent["agent_id"]
            port = portfolios.get(aid)
            if port:
                agent["portfolio"] = {
                    "balance": round(port["balance"], 2),
                    "initial_balance": port["initial_balance"],
                    "total_bets": port["total_bets"],
                    "wins": port["wins"],
                    "losses": port["losses"],
                    "total_profit": round(port["total_profit"], 2),
                    "peak_balance": round(port["peak_balance"], 2),
                    "win_rate": round(port["wins"] / port["total_bets"] * 100, 1)
                               if port["total_bets"] > 0 else 0,
                    "score": port.get("score", 0),
                    "consecutive_losses": port.get("consecutive_losses", 0),
                    "consecutive_wins": port.get("consecutive_wins", 0),
                    "is_fatigued": bool(port.get("is_fatigued", 0)),
                }
            else:
                _def = float(rules.get("defaults", "initial_balance",
                                       getattr(cfg, "DEFAULT_INITIAL_BALANCE", 25.0)))
                _bal = agent.get("initial_balance", _def)
                agent["portfolio"] = {
                    "balance": _bal,
                    "initial_balance": _bal,
                    "total_bets": 0, "wins": 0, "losses": 0,
                    "total_profit": 0, "peak_balance": _bal, "win_rate": 0,
                    "score": 0, "consecutive_losses": 0,
                    "consecutive_wins": 0, "is_fatigued": False,
                }
        # Attach last signal from engine swarm state
        swarm_state = engine.get_swarm_state()
        for agent in agents:
            aid = agent["agent_id"]
            agent["last_signal"] = swarm_state.get(aid)
        return jsonify(agents)

    @app.route("/api/save_config", methods=["POST"])
    @_require_api_auth
    def api_save_config():
        """Save agent toggles, API keys, and paper mode to dynamic_rules.json.

        Expects JSON: {
            "paper_mode": bool,
            "agents": {
                "deepseek_smc": {"enabled": true, "api_key": "sk-..."},
                ...
            }
        }
        """
        data = request.get_json() or {}

        # Paper mode toggle — with portfolio snapshot/restore
        if "paper_mode" in data:
            was_paper = getattr(cfg, "PAPER_TRADING_MODE", True)
            now_paper = bool(data["paper_mode"])
            cfg.PAPER_TRADING_MODE = now_paper
            log.info(f"Paper mode: {'ON' if now_paper else 'OFF'}")

            # ── Paper → Live: snapshot paper balances, sync peak to real ──
            if was_paper and not now_paper:
                try:
                    portfolios = db.get_all_agent_portfolios()
                    snapshot = {
                        p["agent_id"]: {
                            "balance": p["balance"],
                            "peak_balance": p["peak_balance"],
                            "initial_balance": p["initial_balance"],
                            "total_profit": p["total_profit"],
                        }
                        for p in portfolios
                    }
                    db.save_ai_memory("_paper_portfolio_snapshot", snapshot)
                    log.info(f"[Mode Switch] Paper snapshot saved for "
                             f"{len(snapshot)} agent(s)")

                    # Sync primary agent's peak_balance to real CLOB balance
                    # so drawdown calculation starts fresh for live trading.
                    # NOTE: targeted UPDATE — do NOT use reset_agent_portfolio
                    # which would wipe stats (wins, losses, total_bets).
                    primary_id = get_primary_agent_id()
                    if (engine.executor is not None
                            and getattr(engine.executor, "connected", False)):
                        try:
                            real_bal = engine.executor.get_balance_usdc()
                            if real_bal is not None and real_bal > 0:
                                conn = db._get_conn()
                                conn.execute(
                                    """UPDATE agent_portfolios SET
                                       balance=?, peak_balance=?,
                                       updated_at=datetime('now')
                                       WHERE agent_id=?""",
                                    (float(real_bal), float(real_bal),
                                     primary_id),
                                )
                                conn.commit()
                                log.info(
                                    f"[Mode Switch] Primary '{primary_id}' "
                                    f"peak_balance synced to CLOB: "
                                    f"${real_bal:.2f} (drawdown reset)")
                        except Exception as sync_err:
                            log.warning(f"[Mode Switch] CLOB sync failed: "
                                        f"{sync_err}")
                except Exception as snap_err:
                    log.warning(f"[Mode Switch] Snapshot save failed: "
                                f"{snap_err}")

            # ── Live → Paper: restore saved paper balances ──
            elif not was_paper and now_paper:
                try:
                    snapshot = db.load_ai_memory("_paper_portfolio_snapshot")
                    if snapshot and isinstance(snapshot, dict):
                        restored = 0
                        conn = db._get_conn()
                        for agent_id, state in snapshot.items():
                            conn.execute(
                                """UPDATE agent_portfolios SET
                                   balance=?, peak_balance=?,
                                   initial_balance=?,
                                   updated_at=datetime('now')
                                   WHERE agent_id=?""",
                                (state["balance"], state["peak_balance"],
                                 state["initial_balance"], agent_id),
                            )
                            restored += 1
                        conn.commit()
                        log.info(f"[Mode Switch] Paper balances restored "
                                 f"for {restored} agent(s)")
                    else:
                        log.info("[Mode Switch] No paper snapshot found "
                                 "— keeping current balances")
                except Exception as restore_err:
                    log.warning(f"[Mode Switch] Restore failed: "
                                f"{restore_err}")

        # Agent updates
        agents_update = data.get("agents", {})
        if agents_update:
            try:
                current = _read_rules_file()
            except Exception as e:
                return jsonify({"ok": False, "error": f"Error reading rules: {e}"}), 500

            agents_section = current.get("agents", {})

            for agent_id, updates in agents_update.items():
                if agent_id not in agents_section:
                    continue
                agent_cfg = agents_section[agent_id]

                # Toggle enabled/disabled
                if "enabled" in updates:
                    agent_cfg["enabled"] = bool(updates["enabled"])

                # Inject API key into environment (CRITICAL: immediate effect)
                # Only allow whitelisted env var names to prevent env injection
                if "api_key" in updates and updates["api_key"]:
                    key_env = agent_cfg.get("api_key_env", "")
                    if key_env and key_env in _ALLOWED_API_KEY_ENVS:
                        os.environ[key_env] = str(updates["api_key"])[:256]
                        log.info(f"API key injected for {agent_id} -> {key_env}")
                    elif key_env:
                        log.warning(f"Blocked env injection attempt: {key_env} "
                                    f"not in whitelist")

            # Write back to disk (thread-safe)
            current["agents"] = agents_section
            try:
                _write_rules_file(current)
                # Ensure agent portfolios exist in DB.
                # Newly-enabled agents inherit the Global Default Balance
                # (persisted by /api/agents/reset) — so any IA added via
                # the Config panel "nace" with the current global preset.
                _global_default = float(rules.get(
                    "defaults", "initial_balance",
                    getattr(cfg, "DEFAULT_INITIAL_BALANCE", 25.0),
                ))
                for agent_id, agent_cfg in agents_section.items():
                    if agent_cfg.get("enabled"):
                        if "initial_balance" not in agent_cfg:
                            agent_cfg["initial_balance"] = _global_default
                        db.upsert_agent_portfolio(
                            agent_id,
                            agent_cfg.get("display_name", agent_id),
                            agent_cfg.get("strategy", ""),
                            agent_cfg.get("initial_balance", _global_default),
                        )
                log.info(f"Agent config saved: "
                         f"{sum(1 for a in agents_section.values() if a.get('enabled'))} "
                         f"agents active")
            except Exception as e:
                return jsonify({"ok": False, "error": f"Error writing rules: {e}"}), 500

        return jsonify({
            "ok": True,
            "paper_mode": cfg.PAPER_TRADING_MODE,
            "active_agents": len(swarm.get_active_agents()),
        })

    def _current_default_balance() -> float:
        """Shared helper: read live default from dynamic_rules → fallback cfg."""
        try:
            return float(rules.get(
                "defaults", "initial_balance",
                getattr(cfg, "DEFAULT_INITIAL_BALANCE", 25.0),
            ))
        except Exception:
            return float(getattr(cfg, "DEFAULT_INITIAL_BALANCE", 25.0))

    def _persist_default_balance(new_default: float) -> None:
        """Persist the new Global Default Balance into dynamic_rules.json."""
        try:
            current = _read_rules_file()
            defaults = current.get("defaults", {}) or {}
            defaults["initial_balance"] = float(new_default)
            current["defaults"] = defaults
            _write_rules_file(current)
            log.info(f"[Reset] Global default balance persisted = "
                     f"${new_default:.2f} (dynamic_rules.defaults.initial_balance)")
        except Exception as e:
            log.warning(f"[Reset] Could not persist global default: {e}")

    @app.route("/api/agents/reset", methods=["POST"])
    @_require_api_auth
    def api_agent_reset():
        """MAESTRO: Reset Balance Global IAs.

        Single master entry point. If the payload includes an
        ``agent_id`` it degrades to the legacy per-agent reset
        (kept for back-compat with older UI bundles); otherwise it
        resets EVERY portfolio in the DB to the requested amount,
        persists the new amount as the global default so any IA
        created/enabled afterwards inherits it, and returns a
        count summary.
        """
        data = request.get_json() or {}
        # Default comes from the live Global Default (currently $25),
        # NOT the legacy hard-coded 100.
        new_balance = float(data.get("balance", _current_default_balance()))
        new_balance = max(1.0, min(new_balance, 10000.0))

        agent_id = (data.get("agent_id") or "").strip()

        # ── Legacy per-agent path ──
        if agent_id:
            db.reset_agent_portfolio(agent_id, new_balance)
            log.info(f"[Reset] Agent {agent_id} portfolio reset to ${new_balance:.2f}")
            return jsonify({"ok": True, "agent_id": agent_id,
                            "new_balance": new_balance,
                            "scope": "single"})

        # ── Global path (the único botón maestro) ──
        _persist_default_balance(new_balance)
        portfolios = db.get_all_agent_portfolios()
        reset_count = 0
        for p in portfolios:
            db.reset_agent_portfolio(p["agent_id"], new_balance)
            reset_count += 1
        log.info(f"[Reset] GLOBAL RESET — {reset_count} agente(s) a "
                 f"${new_balance:.2f} | nuevo default global persistido")
        return jsonify({"ok": True, "scope": "global",
                        "reset_count": reset_count,
                        "new_balance": new_balance,
                        "persisted_default": True})

    @app.route("/api/agents/reset_all", methods=["POST"])
    def api_agent_reset_all():
        """Back-compat alias — delegates to the master /api/agents/reset."""
        return api_agent_reset()

    @app.route("/api/sync_paper_balance", methods=["POST"])
    @_require_api_auth
    def api_sync_paper_balance():
        """Query the real Polymarket USDC.e balance and reset the primary
        agent's simulated balance to match it.  One-shot sync action."""
        if engine.executor is None or not engine.executor.connected:
            return jsonify({"ok": False,
                            "error": "Executor no conectado a Polymarket"}), 400
        try:
            real_bal = engine.executor.get_balance_usdc()
            if real_bal is None or real_bal <= 0:
                return jsonify({"ok": False,
                                "error": f"Balance real inválido: {real_bal}"}), 400
            primary_id = get_primary_agent_id()
            db.reset_agent_portfolio(primary_id, float(real_bal))
            log.info(f"[Sync Paper] Primary '{primary_id}' paper balance "
                     f"synced to real USDC.e: ${real_bal:.2f}")
            return jsonify({"ok": True,
                            "synced_balance": round(real_bal, 2),
                            "agent_id": primary_id})
        except Exception as e:
            log.warning(f"[Sync Paper] Error: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    # ── Live Dynamic Sizing toggle ────────────────────────────

    @app.route("/api/toggle_live_dynamic_sizing", methods=["POST"])
    @_require_api_auth
    def api_toggle_live_dynamic_sizing():
        """Enable/disable the Live Dynamic Sizing feature at runtime.

        When ON, Kelly ingests the real USDC.e balance from the
        py-clob-client connection (via ClobExecutor.get_balance_usdc).
        When OFF, Kelly uses the simulated paper balance sourced from
        ``agent_portfolios`` (default $25).
        """
        data = request.get_json() or {}
        enabled = bool(data.get("enabled",
                                not getattr(cfg, "LIVE_DYNAMIC_SIZING", False)))
        cfg.LIVE_DYNAMIC_SIZING = enabled
        log.info(f"[Live Sizing] Toggle {'ON' if enabled else 'OFF'} via API")
        return jsonify({
            "ok": True,
            "enabled": cfg.LIVE_DYNAMIC_SIZING,
            "default_initial_balance": _current_default_balance(),
        })

    @app.route("/api/toggle_maker_ladder", methods=["POST"])
    @_require_api_auth
    def api_toggle_maker_ladder():
        """Enable/disable Maker Ladder execution from the UI."""
        data = request.get_json() or {}
        try:
            current = _read_rules_file()
            ladder_enabled = bool(data.get("enabled", True))
            if "ladder" not in current:
                current["ladder"] = {}
            current["ladder"]["enabled"] = ladder_enabled
            _write_rules_file(current)
            log.info(f"[LADDER] Maker Ladder toggle set to {ladder_enabled}")
            return jsonify({"ok": True, "enabled": ladder_enabled})
        except Exception as e:
            log.error(f"[LADDER] Error toggling ladder: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/live_dynamic_sizing", methods=["GET"])
    def api_live_dynamic_sizing_status():
        """Return current Live Sizing toggle state + live probe result."""
        live_bal = None
        if (getattr(cfg, "LIVE_DYNAMIC_SIZING", False)
                and engine.executor is not None
                and engine.executor.connected):
            try:
                live_bal = engine.executor.get_balance_usdc()
            except Exception as e:
                log.debug(f"[Live Sizing] status probe failed: {e}")
        return jsonify({
            "enabled": bool(getattr(cfg, "LIVE_DYNAMIC_SIZING", False)),
            "default_initial_balance": _current_default_balance(),
            "executor_connected": bool(
                engine.executor and engine.executor.connected
            ),
            "live_balance": live_bal,
            "ladder_enabled": rules.get("ladder", "enabled", True)
        })

    # ── Fatigue & Scoring Endpoints ─────────────────────────

    @app.route("/api/fatigue_config", methods=["GET"])
    def api_fatigue_config_get():
        """Return current fatigue config + per-agent fatigue/score state."""
        fatigue_section = rules.get_section("fatigue_prompts") or {}
        portfolios = db.get_all_agent_portfolios()
        agents_fatigue = {}
        for p in portfolios:
            agents_fatigue[p["agent_id"]] = {
                "display_name": p["display_name"],
                "score": p.get("score", 0),
                "consecutive_losses": p.get("consecutive_losses", 0),
                "consecutive_wins": p.get("consecutive_wins", 0),
                "is_fatigued": bool(p.get("is_fatigued", 0)),
                "rotation_idx": p.get("rotation_idx", 0),
            }
        return jsonify({
            "max_losses": fatigue_section.get("max_losses", 3),
            "max_age_sec": fatigue_section.get("max_age_sec", 7200),
            "cooldown_sec": fatigue_section.get("cooldown_sec", 3600),
            "rotation_prompts": fatigue_section.get("rotation", []),
            "agents_fatigue": agents_fatigue,
        })

    @app.route("/api/fatigue_config", methods=["POST"])
    @_require_api_auth
    def api_fatigue_config_save():
        """Save fatigue config (max_losses, cooldown, rotation prompts).

        Expects JSON: {
            "max_losses": int,
            "cooldown_sec": int,
            "rotation_prompts": ["prompt1", "prompt2", ...]
        }
        """
        data = request.get_json() or {}
        try:
            current = _read_rules_file()
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        fp = current.get("fatigue_prompts", {})
        if "max_losses" in data:
            fp["max_losses"] = max(1, min(int(data["max_losses"]), 20))
        if "cooldown_sec" in data:
            fp["cooldown_sec"] = max(60, min(int(data["cooldown_sec"]), 86400))
        if "rotation_prompts" in data:
            prompts = data["rotation_prompts"]
            if isinstance(prompts, list) and len(prompts) > 0:
                fp["rotation"] = [str(p).strip() for p in prompts if str(p).strip()]
        current["fatigue_prompts"] = fp

        try:
            _write_rules_file(current)
            log.info(f"Fatigue config saved: max_losses={fp.get('max_losses')}, "
                     f"cooldown={fp.get('cooldown_sec')}s, "
                     f"{len(fp.get('rotation', []))} prompts")
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        return jsonify({"ok": True})

    @app.route("/api/agent/<agent_id>/reset_fatigue", methods=["POST"])
    @_require_api_auth
    def api_agent_reset_fatigue(agent_id):
        """Reset fatigue state for a single agent."""
        engine.reset_agent_fatigue(agent_id)
        return jsonify({"ok": True, "agent_id": agent_id})

    # ── RAG Strategy Endpoints ────────────────────────────────

    @app.route("/api/rag/strategies")
    def api_rag_strategies():
        """List all stored RAG strategies."""
        strategies = engine.rag.list_strategies()
        return jsonify({
            "count": len(strategies),
            "strategies": strategies,
        })

    @app.route("/api/rag/evaluate")
    def api_rag_evaluate():
        """Evaluate RAG effectiveness: accuracy WITH vs WITHOUT rag_strategy.

        Compares prediction accuracy for windows that used a RAG strategy
        vs those that didn't, providing a statistical breakdown.
        """
        conn = db._get_conn()
        # All resolved predictions (with outcome)
        rows = conn.execute("""
            SELECT p.rag_strategy, p.correct, p.confidence,
                   p.signal_quality, w.outcome
            FROM predictions p
            JOIN windows w ON p.window_id = w.window_id
            WHERE p.correct IS NOT NULL
              AND p.ai_model = 'deepseek'
            ORDER BY p.id
        """).fetchall()

        with_rag = {"total": 0, "correct": 0, "conf_sum": 0, "strategies": {}}
        without_rag = {"total": 0, "correct": 0, "conf_sum": 0}

        for r in rows:
            strat = r["rag_strategy"]
            correct = r["correct"]
            conf = r["confidence"] or 0

            if strat and strat.strip():
                with_rag["total"] += 1
                with_rag["correct"] += correct
                with_rag["conf_sum"] += conf
                # Per-strategy breakdown
                if strat not in with_rag["strategies"]:
                    with_rag["strategies"][strat] = {"total": 0, "correct": 0}
                with_rag["strategies"][strat]["total"] += 1
                with_rag["strategies"][strat]["correct"] += correct
            else:
                without_rag["total"] += 1
                without_rag["correct"] += correct
                without_rag["conf_sum"] += conf

        def _stats(bucket):
            t = bucket["total"]
            c = bucket["correct"]
            return {
                "total": t,
                "correct": c,
                "accuracy": round(c / t * 100, 1) if t > 0 else None,
                "avg_confidence": round(bucket["conf_sum"] / t, 1) if t > 0 else None,
            }

        # Per-strategy accuracy ranking
        strategy_ranking = []
        for name, s in with_rag["strategies"].items():
            t, c = s["total"], s["correct"]
            strategy_ranking.append({
                "strategy": name,
                "total": t,
                "correct": c,
                "accuracy": round(c / t * 100, 1) if t > 0 else None,
            })
        strategy_ranking.sort(key=lambda x: (x["accuracy"] or 0), reverse=True)

        return jsonify({
            "with_rag": _stats(with_rag),
            "without_rag": _stats(without_rag),
            "strategy_ranking": strategy_ranking,
            "total_predictions": len(rows),
            "strategies_in_db": engine.rag.count(),
        })

    return app
