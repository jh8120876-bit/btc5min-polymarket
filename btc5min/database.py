import json
import math
import queue
import sqlite3
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import log

DB_PATH = str(Path(__file__).resolve().parent.parent / "btc5min.db")
_local = threading.local()

# ── Write Worker Queue ────────────────────────────────────────
# A single dedicated thread processes high-frequency writes (price ticks,
# market context) so the engine tick loop and AI threads never block on DB I/O.
_write_queue: queue.Queue = queue.Queue(maxsize=5000)
_write_worker_started = False
_write_worker_lock = threading.Lock()


def _start_write_worker():
    """Start the background write-worker thread (once)."""
    global _write_worker_started
    with _write_worker_lock:
        if _write_worker_started:
            return
        _write_worker_started = True
        t = threading.Thread(target=_write_worker_loop, daemon=True)
        t.start()
        log.info("DB write-worker thread started")


def _write_worker_loop():
    """Drain the write queue, batching price ticks with executemany.

    Auto-reconnects on unrecoverable SQLite errors and always closes the
    connection on thread exit to avoid FD leaks.
    """
    conn: Optional[sqlite3.Connection] = None

    def _open():
        c = sqlite3.connect(DB_PATH)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=5000")
        return c

    try:
        conn = _open()
        consecutive_errors = 0
        while True:
            batch = []
            try:
                item = _write_queue.get(timeout=5)
                batch.append(item)
                for _ in range(49):
                    try:
                        batch.append(_write_queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            tick_rows = []
            other_ops = []
            for op in batch:
                if op[0] == "tick":
                    tick_rows.append(op[1])
                else:
                    other_ops.append(op)

            try:
                if tick_rows:
                    conn.executemany(
                        "INSERT INTO price_ticks (timestamp, price, source) "
                        "VALUES (?, ?, ?)",
                        tick_rows,
                    )
                for op in other_ops:
                    if op[0] == "sql":
                        conn.execute(op[1], op[2])
                conn.commit()
                consecutive_errors = 0
            except sqlite3.Error as e:
                consecutive_errors += 1
                log.error(f"Write-worker SQLite error: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
                # Reconnect after 3 consecutive failures
                if consecutive_errors >= 3:
                    log.warning("Write-worker: reconnecting DB")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    try:
                        conn = _open()
                        consecutive_errors = 0
                    except Exception as re:
                        log.error(f"Write-worker reconnect failed: {re}")
                        time.sleep(2)
            except Exception as e:
                log.error(f"Write-worker batch error: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local connection (sqlite3 is not thread-safe)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=5000")
    return _local.conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS price_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            price REAL NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_id INTEGER NOT NULL,
            open_time REAL NOT NULL,
            close_time REAL,
            open_price REAL NOT NULL,
            close_price REAL,
            outcome TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_id INTEGER NOT NULL,
            prediction TEXT NOT NULL,
            confidence INTEGER NOT NULL,
            original_confidence INTEGER,
            final_confidence INTEGER,
            reasoning TEXT,
            risk_score TEXT,
            outcome TEXT,
            correct INTEGER,
            -- TA snapshot at prediction time
            rsi REAL,
            macd_histogram REAL,
            bb_pct REAL,
            ema_cross TEXT,
            momentum REAL,
            volatility REAL,
            tick_buy_pressure INTEGER,
            atr_pct REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_id INTEGER NOT NULL,
            side TEXT NOT NULL,
            amount REAL NOT NULL,
            odds_cents INTEGER NOT NULL,
            open_price REAL NOT NULL,
            close_price REAL,
            outcome TEXT,
            payout REAL,
            profit REAL,
            won INTEGER,
            is_manual INTEGER DEFAULT 0,
            is_early INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS ai_memory (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS market_context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_id INTEGER NOT NULL,
            volume_24h REAL,
            volume_1h REAL,
            funding_rate REAL,
            open_interest REAL,
            oi_change_pct REAL,
            bid_ask_spread REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS liquidations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            usd_value REAL NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_window ON predictions(window_id);
        CREATE INDEX IF NOT EXISTS idx_bets_window ON bets(window_id);
        CREATE INDEX IF NOT EXISTS idx_price_ticks_ts ON price_ticks(timestamp);
        CREATE INDEX IF NOT EXISTS idx_market_context_window ON market_context(window_id);
        CREATE INDEX IF NOT EXISTS idx_liquidations_ts ON liquidations(timestamp);

        -- Performance indexes (megarefactor B3)
        CREATE INDEX IF NOT EXISTS idx_predictions_ai_model ON predictions(ai_model);
        CREATE INDEX IF NOT EXISTS idx_predictions_strategy ON predictions(strategy_name);
        CREATE INDEX IF NOT EXISTS idx_predictions_quality ON predictions(signal_quality);
        CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
        CREATE INDEX IF NOT EXISTS idx_predictions_window_model ON predictions(window_id, ai_model);
        CREATE INDEX IF NOT EXISTS idx_bets_ai_model ON bets(ai_model);
        CREATE INDEX IF NOT EXISTS idx_bets_strategy ON bets(strategy_name);
        CREATE INDEX IF NOT EXISTS idx_bets_created ON bets(created_at);
        CREATE INDEX IF NOT EXISTS idx_bets_window_model ON bets(window_id, ai_model);
        CREATE INDEX IF NOT EXISTS idx_bets_ghost ON bets(is_ghost);
        CREATE INDEX IF NOT EXISTS idx_bets_shadow ON bets(is_shadow);
        CREATE INDEX IF NOT EXISTS idx_windows_outcome ON windows(outcome);

        -- Multi-Agent Swarm: independent portfolios per AI agent
        CREATE TABLE IF NOT EXISTS agent_portfolios (
            agent_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL DEFAULT '',
            strategy TEXT NOT NULL DEFAULT '',
            balance REAL NOT NULL DEFAULT 100.0,
            initial_balance REAL NOT NULL DEFAULT 100.0,
            total_bets INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            total_profit REAL DEFAULT 0.0,
            peak_balance REAL DEFAULT 100.0,
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()

    # Migrations for existing databases
    _run_migrations(conn)
    
    # Limpieza de inicio: Eliminar ventanas y apuestas "vivas" de una sesión anterior
    # que se cerró de forma abrupta para que no contaminen estadísticas ni la vista en vivo.
    conn.executescript("""
        DELETE FROM bets WHERE outcome IS NULL;
        DELETE FROM predictions WHERE outcome IS NULL;
        DELETE FROM windows WHERE outcome IS NULL;
    """)
    conn.commit()

    log.info(f"Database initialized: {DB_PATH}")


def _run_migrations(conn: sqlite3.Connection):
    """Run ALTER TABLE migrations for columns added after initial release."""
    migrations = [
        ("bets", "is_early", "INTEGER DEFAULT 0"),
        # ML feature columns on predictions
        ("predictions", "hour_utc", "INTEGER"),
        ("predictions", "day_of_week", "INTEGER"),
        ("predictions", "price_change_pct", "REAL"),
        ("predictions", "return_1m", "REAL"),
        ("predictions", "return_5m", "REAL"),
        ("predictions", "return_15m", "REAL"),
        # Feature engineering columns
        ("predictions", "price_acceleration", "REAL"),
        ("predictions", "range_position", "REAL"),
        ("predictions", "volatility_zscore", "REAL"),
        # Market context snapshot
        ("predictions", "volume_24h", "REAL"),
        ("predictions", "funding_rate", "REAL"),
        ("predictions", "open_interest", "REAL"),
        # Calibration
        ("predictions", "calibrated_confidence", "INTEGER"),
        # Multi-timeframe & contextual features
        ("predictions", "trend_slope_15m", "REAL"),
        ("predictions", "trend_slope_1h", "REAL"),
        ("predictions", "trend_alignment", "INTEGER"),
        ("predictions", "signal_quality", "TEXT DEFAULT 'NORMAL'"),
        ("predictions", "quality_reason", "TEXT"),
        ("predictions", "liq_buy_vol_5m", "REAL DEFAULT 0"),
        ("predictions", "liq_sell_vol_5m", "REAL DEFAULT 0"),
        ("predictions", "reversal_alert", "INTEGER DEFAULT 0"),
        # Regression target (log-return magnitude)
        ("windows", "target_magnitude", "REAL"),
        # Double-brain: local LLM ensemble features
        ("predictions", "local_ai_prediction", "TEXT"),
        ("predictions", "local_ai_confidence", "INTEGER"),
        ("predictions", "local_ai_reasoning", "TEXT"),
        # Shadow mode: ai_model discriminator for parallel predictions/bets
        ("predictions", "ai_model", "TEXT DEFAULT 'deepseek'"),
        ("bets", "ai_model", "TEXT DEFAULT 'deepseek'"),
        ("bets", "is_shadow", "INTEGER DEFAULT 0"),
        ("bets", "is_ghost", "INTEGER DEFAULT 0"),
        # Multi-Agent Swarm: strategy discriminator
        ("predictions", "strategy_name", "TEXT DEFAULT 'smc_liquidity'"),
        ("bets", "strategy_name", "TEXT DEFAULT 'smc_liquidity'"),
        # ML enrichment: latency, price snapshot, consensus
        ("predictions", "ai_latency_ms", "REAL"),
        ("predictions", "price_at_bet", "REAL"),
        ("predictions", "consensus_agreement", "REAL"),
        # RAG institutional strategy used for this prediction
        ("predictions", "rag_strategy", "TEXT"),
        # ML Oracle filter: probability and skip flag
        ("predictions", "ml_oracle_prob", "REAL"),
        ("predictions", "ml_skipped", "INTEGER DEFAULT 0"),
        # SMC Features (smartmoneyconcepts library)
        ("predictions", "smc_fvg_active", "INTEGER DEFAULT 0"),
        ("predictions", "smc_bos_last", "INTEGER DEFAULT 0"),
        ("predictions", "smc_choch_last", "INTEGER DEFAULT 0"),
        ("predictions", "smc_ob_nearest", "INTEGER DEFAULT 0"),
        ("predictions", "smc_ob_distance_pct", "REAL"),
        ("predictions", "smc_ob_strength", "REAL"),
        ("predictions", "smc_liq_nearest", "INTEGER DEFAULT 0"),
        ("predictions", "smc_liq_distance_pct", "REAL"),
        ("predictions", "smc_retracement_pct", "REAL"),
        ("predictions", "smc_in_ote_zone", "INTEGER DEFAULT 0"),
        ("predictions", "smc_support", "REAL"),
        ("predictions", "smc_resistance", "REAL"),
        ("predictions", "smc_kill_zone", "TEXT"),
        ("predictions", "smc_in_kill_zone", "INTEGER DEFAULT 0"),
        # Order Book Imbalance
        ("market_context", "order_book_imbalance", "REAL"),
        ("predictions", "order_book_imbalance", "REAL"),
        # Francotirador layer alignment audit
        ("predictions", "layer_alignment", "TEXT"),
        # Odds source tracking: "clob_live" or "internal_fallback"
        ("bets", "odds_source", "TEXT DEFAULT 'internal_fallback'"),
        # Sniper metadata for ML training
        ("predictions", "sniper_eval_count", "INTEGER"),
        ("predictions", "sniper_time_to_bet_sec", "REAL"),
        ("predictions", "sniper_flipped", "INTEGER DEFAULT 0"),
        # ML Judge B-features: volatility regime + fear & greed
        ("predictions", "volatility_regime", "TEXT"),
        ("predictions", "fear_greed", "INTEGER"),
        # Agent scoring & fatigue persistence
        ("agent_portfolios", "score", "INTEGER DEFAULT 0"),
        ("agent_portfolios", "consecutive_losses", "INTEGER DEFAULT 0"),
        ("agent_portfolios", "consecutive_wins", "INTEGER DEFAULT 0"),
        ("agent_portfolios", "is_fatigued", "INTEGER DEFAULT 0"),
        ("agent_portfolios", "rotation_idx", "INTEGER DEFAULT 0"),
        ("agent_portfolios", "fatigue_since", "REAL DEFAULT 0"),
        # ── Web3 Execution columns (Fase F0) ──
        # CLOB order tracking
        ("bets", "order_id", "TEXT"),               # CLOB order ID from py-clob-client
        ("bets", "fill_price", "REAL"),              # actual fill price (cents)
        ("bets", "fill_size", "REAL"),               # tokens actually filled
        ("bets", "exec_status", "TEXT"),             # pending/partial/filled/unfilled/cancelled
        # Gasless redeem tracking
        ("bets", "redeemed_at", "INTEGER"),          # UTC timestamp of gasless redeem
        # Survival monitor (Profit-Lock / Flip-Stop)
        ("bets", "exit_reason", "TEXT"),             # resolve/profit_lock/flip_stop/cancel
        ("bets", "exit_price", "REAL"),              # price at which position was exited
        # ── Polymarket EV context (Task 5) ──
        ("predictions", "polymarket_odds", "REAL"),          # buy-side price in cents for predicted side
        ("predictions", "expected_value", "REAL"),           # EV = (conf/100) * (100/price_cents)
        ("predictions", "distance_to_strike", "REAL"),       # current_price - strike_price (USD)
        # ── Temporal velocity features (history vectors) ──
        ("predictions", "rsi_velocity", "REAL"),             # rsi_t0 - rsi_t-1
        ("predictions", "rsi_acceleration", "REAL"),         # velocity_t0 - velocity_t-1
        ("predictions", "momentum_velocity", "REAL"),        # momentum_t0 - momentum_t-1
        ("predictions", "cvd_velocity", "REAL"),             # tick_buy_pressure_t0 - t-1
        # ── API token cost tracking (analytical, not Judge input) ──
        ("predictions", "api_tokens_used", "INTEGER"),
        # Nuevas features avanzadas
        ("predictions", "cvd_imbalance_pct", "REAL"),
        ("predictions", "cvd_aggressor_ratio", "REAL"),
        ("predictions", "is_liquidity_swept_1m", "INTEGER DEFAULT 0"),
        ("predictions", "mss_detected", "INTEGER DEFAULT 0"),
        ("predictions", "post_sweep_reversal", "INTEGER DEFAULT 0"),
        ("predictions", "gex_net_bias", "REAL"),
        ("predictions", "gex_dist_call_wall_pct", "REAL"),
        ("predictions", "gex_dist_put_wall_pct", "REAL"),
        # Liquidation cluster proximity (Vibe-Trading inspired)
        ("predictions", "liq_cluster_distance_atr", "REAL"),
        ("predictions", "liq_cluster_magnitude_pct", "REAL"),
        ("predictions", "liq_cluster_side", "TEXT"),  # ABOVE / BELOW / AT_SPOT / NEUTRAL
    ]
    for table, column, col_type in migrations:
        try:
            conn.execute(f"SELECT {column} FROM {table} LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            conn.commit()


def save_price_tick(price: float, timestamp: float, source: str):
    """Queue a price tick for async batch insertion via write-worker."""
    _start_write_worker()
    try:
        _write_queue.put_nowait(("tick", (timestamp, price, source)))
    except queue.Full:
        log.warning("Write queue full — dropping price tick")


def save_window(window_id: int, open_time: float, open_price: float):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO windows (window_id, open_time, open_price) VALUES (?, ?, ?)",
        (window_id, open_time, open_price),
    )
    conn.commit()


def close_window(window_id: int, close_time: float, close_price: float, outcome: str):
    conn = _get_conn()
    # Compute log-return target magnitude: ln(close/open)
    row = conn.execute(
        "SELECT open_price FROM windows WHERE window_id=?", (window_id,)
    ).fetchone()
    target_mag = None
    if row and row["open_price"] and row["open_price"] > 0 and close_price > 0:
        target_mag = round(math.log(close_price / row["open_price"]) * 100, 6)
    conn.execute(
        """UPDATE windows SET close_time=?, close_price=?, outcome=?,
           target_magnitude=? WHERE window_id=?""",
        (close_time, close_price, outcome, target_mag, window_id),
    )
    conn.commit()


def save_prediction(window_id: int, prediction: str, confidence: int,
                    original_confidence: int, reasoning: str,
                    risk_score: str, ta: dict,
                    ml_features: dict | None = None,
                    ai_model: str = "deepseek",
                    strategy_name: str = "smc_liquidity",
                    rag_strategy: str | None = None,
                    ml_oracle_prob: float | None = None,
                    ml_skipped: bool = False):
    conn = _get_conn()
    # ── Duplicate guard: skip if prediction already exists for this window+model ──
    existing = conn.execute(
        "SELECT id FROM predictions WHERE window_id=? AND ai_model=? LIMIT 1",
        (window_id, ai_model),
    ).fetchone()
    if existing:
        log.info(f"save_prediction: window_id={window_id} ai_model={ai_model} "
                 f"already exists (row id={existing['id']}), skipping duplicate INSERT")
        return
    ml = ml_features or {}
    conn.execute(
        """INSERT INTO predictions
           (window_id, prediction, confidence, original_confidence, reasoning,
            risk_score, rsi, macd_histogram, bb_pct, ema_cross, momentum,
            volatility, tick_buy_pressure, atr_pct,
            hour_utc, day_of_week, price_change_pct,
            return_1m, return_5m, return_15m,
            price_acceleration, range_position, volatility_zscore,
            volume_24h, funding_rate, open_interest,
            calibrated_confidence,
            trend_slope_15m, trend_slope_1h, trend_alignment,
            signal_quality, quality_reason,
            liq_buy_vol_5m, liq_sell_vol_5m, reversal_alert,
            local_ai_prediction, local_ai_confidence, local_ai_reasoning,
            ai_model, strategy_name,
            ai_latency_ms, price_at_bet, consensus_agreement,
            rag_strategy,
            ml_oracle_prob, ml_skipped,
            smc_fvg_active, smc_bos_last, smc_choch_last,
            smc_ob_nearest, smc_ob_distance_pct, smc_ob_strength,
            smc_liq_nearest, smc_liq_distance_pct,
            smc_retracement_pct, smc_in_ote_zone,
            smc_support, smc_resistance,
            smc_kill_zone, smc_in_kill_zone,
            order_book_imbalance,
            layer_alignment,
            volatility_regime, fear_greed,
            polymarket_odds, expected_value, distance_to_strike,
            rsi_velocity, rsi_acceleration, momentum_velocity,
            cvd_velocity, api_tokens_used,
            cvd_imbalance_pct, cvd_aggressor_ratio,
            is_liquidity_swept_1m, mss_detected, post_sweep_reversal,
            gex_net_bias, gex_dist_call_wall_pct, gex_dist_put_wall_pct,
            liq_cluster_distance_atr, liq_cluster_magnitude_pct, liq_cluster_side)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                   ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                   ?, ?, ?, ?, ?, ?, ?, ?, ?,
                   ?, ?, ?,
                   ?, ?,
                   ?, ?, ?,
                   ?,
                   ?, ?,
                   ?, ?, ?,
                   ?, ?, ?,
                   ?, ?,
                   ?, ?,
                   ?, ?,
                   ?, ?,
                   ?, ?,
                   ?,
                   ?,
                   ?, ?,
                   ?, ?, ?,
                   ?, ?, ?,
                   ?, ?,
                   ?, ?, ?, ?, ?, ?,
                   ?, ?, ?)""",
        (window_id, prediction, confidence, original_confidence, reasoning,
         risk_score,
         ta.get("rsi", 0), ta.get("macd_histogram", 0), ta.get("bb_pct", 0),
         ta.get("ema_cross", ""), ta.get("momentum", 0), ta.get("volatility", 0),
         ta.get("tick_buy_pressure", 0), ta.get("atr_pct", 0),
         ml.get("hour_utc"), ml.get("day_of_week"), ml.get("price_change_pct"),
         ml.get("return_1m"), ml.get("return_5m"), ml.get("return_15m"),
         ml.get("price_acceleration"), ml.get("range_position"),
         ml.get("volatility_zscore"),
         ml.get("volume_24h"), ml.get("funding_rate"), ml.get("open_interest"),
         ml.get("calibrated_confidence"),
         ml.get("trend_slope_15m"), ml.get("trend_slope_1h"),
         ml.get("trend_alignment"),
         ml.get("signal_quality", "NORMAL"), ml.get("quality_reason"),
         ml.get("liq_buy_vol_5m", 0), ml.get("liq_sell_vol_5m", 0),
         ml.get("reversal_alert", 0),
         ml.get("local_ai_prediction"), ml.get("local_ai_confidence"),
         ml.get("local_ai_reasoning"),
         ai_model, strategy_name,
         ml.get("ai_latency_ms"), ml.get("price_at_bet"),
         ml.get("consensus_agreement"),
         rag_strategy,
         ml_oracle_prob, int(ml_skipped),
         ml.get("smc_fvg_active", 0), ml.get("smc_bos_last", 0), ml.get("smc_choch_last", 0),
         ml.get("smc_ob_nearest", 0), ml.get("smc_ob_distance_pct"), ml.get("smc_ob_strength"),
         ml.get("smc_liq_nearest", 0), ml.get("smc_liq_distance_pct"),
         ml.get("smc_retracement_pct"), ml.get("smc_in_ote_zone", 0),
         ml.get("smc_support"), ml.get("smc_resistance"),
         ml.get("smc_kill_zone"), ml.get("smc_in_kill_zone", 0),
         ml.get("order_book_imbalance"),
         ml.get("layer_alignment"),
         ml.get("volatility_regime"),
         ml.get("fear_greed"),
         ml.get("polymarket_odds"),
         ml.get("expected_value"),
         ml.get("distance_to_strike"),
         ml.get("rsi_velocity"),
         ml.get("rsi_acceleration"),
         ml.get("momentum_velocity"),
         ml.get("cvd_velocity"),
         ml.get("api_tokens_used"),
         ml.get("cvd_imbalance_pct"), ml.get("cvd_aggressor_ratio"),
         ml.get("is_liquidity_swept_1m", 0), ml.get("mss_detected", 0), ml.get("post_sweep_reversal", 0),
         ml.get("gex_net_bias"), ml.get("gex_dist_call_wall_pct"), ml.get("gex_dist_put_wall_pct"),
         ml.get("liq_cluster_distance_atr"), ml.get("liq_cluster_magnitude_pct"),
         ml.get("liq_cluster_side"),
         )
    )
    conn.commit()

def save_market_context(window_id: int, ctx: dict):
    """Save Binance market context for a window."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO market_context
           (window_id, volume_24h, volume_1h, funding_rate,
            open_interest, oi_change_pct, bid_ask_spread,
            order_book_imbalance)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (window_id, ctx.get("volume_24h"), ctx.get("volume_1h"),
         ctx.get("funding_rate"), ctx.get("open_interest"),
         ctx.get("oi_change_pct"), ctx.get("bid_ask_spread"),
         ctx.get("order_book_imbalance")),
    )
    conn.commit()


def get_confidence_calibration(limit: int = 50) -> dict:
    """Get actual accuracy per confidence bucket for calibration."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT confidence, correct FROM predictions
           WHERE outcome IS NOT NULL AND ai_model IN ('deepseek', 'deepseek_smc')
           ORDER BY id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    if not rows:
        return {}
    buckets = {}
    for r in rows:
        bucket = (r["confidence"] // 10) * 10  # 40,50,60,70,80...
        if bucket not in buckets:
            buckets[bucket] = {"total": 0, "correct": 0}
        buckets[bucket]["total"] += 1
        buckets[bucket]["correct"] += r["correct"] or 0
    result = {}
    for bucket, data in sorted(buckets.items()):
        if data["total"] >= 3:
            result[bucket] = round(data["correct"] / data["total"] * 100, 1)
    return result


def update_prediction_direction(window_id: int, prediction: str, confidence: int,
                                ai_model: str = "deepseek", flip_reasoning: str = ""):
    """Update prediction direction and confidence when flipped by consensus/reevaluation."""
    conn = _get_conn()
    conn.execute(
        """UPDATE predictions SET prediction=?, confidence=?, reasoning=?
           WHERE window_id=? AND ai_model=?""",
        (prediction, confidence, flip_reasoning, window_id, ai_model),
    )
    conn.commit()


def update_prediction_outcome(window_id: int, outcome: str, final_confidence: int,
                              ai_model: str | None = None):
    conn = _get_conn()
    if ai_model:
        conn.execute(
            """UPDATE predictions SET outcome=?, correct=(prediction=?),
               final_confidence=? WHERE window_id=? AND ai_model=?""",
            (outcome, outcome, final_confidence, window_id, ai_model),
        )
    else:
        # Update all models for this window (backward compatible)
        conn.execute(
            """UPDATE predictions SET outcome=?, correct=(prediction=?),
               final_confidence=? WHERE window_id=?""",
            (outcome, outcome, final_confidence, window_id),
        )
    conn.commit()


def update_prediction_reasoning(window_id: int, ai_model: str,
                                reasoning: str, news_impact: str = ""):
    """Phase 2 HFT: backfill reasoning + news_impact after fast prediction."""
    conn = _get_conn()
    conn.execute(
        """UPDATE predictions SET reasoning=?, risk_score=COALESCE(risk_score, 'MEDIUM')
           WHERE window_id=? AND ai_model=?""",
        (reasoning, window_id, ai_model),
    )
    conn.commit()


def save_bet(window_id: int, side: str, amount: float, odds_cents: float,
             open_price: float, is_manual: bool = False, is_early: bool = False,
             ai_model: str = "deepseek", is_shadow: bool = False,
             is_ghost: bool = False, strategy_name: str = "smc_liquidity",
             odds_source: str = "internal_fallback"):
    conn = _get_conn()
    # ── Duplicate guard: skip if bet already exists for this window+model ──
    existing = conn.execute(
        "SELECT id FROM bets WHERE window_id=? AND ai_model=? LIMIT 1",
        (window_id, ai_model),
    ).fetchone()
    if existing:
        log.info(f"save_bet: window_id={window_id} ai_model={ai_model} already has a bet "
                 f"(row id={existing['id']}), skipping duplicate INSERT")
        return
    conn.execute(
        """INSERT INTO bets (window_id, side, amount, odds_cents, open_price,
           is_manual, is_early, ai_model, is_shadow, is_ghost, strategy_name,
           odds_source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (window_id, side, amount, odds_cents, open_price,
         int(is_manual), int(is_early), ai_model, int(is_shadow),
         int(is_ghost), strategy_name, odds_source),
    )
    conn.commit()


def update_bet_exec(window_id: int, ai_model: str,
                    order_id: str = "", fill_price: float = 0.0,
                    fill_size: float = 0.0, exec_status: str = ""):
    """Update a bet row with CLOB execution results (order_id, fill, status)."""
    conn = _get_conn()
    conn.execute(
        """UPDATE bets
           SET order_id=?, fill_price=?, fill_size=?, exec_status=?
           WHERE window_id=? AND ai_model=?""",
        (order_id, fill_price, fill_size, exec_status, window_id, ai_model),
    )
    conn.commit()


def close_bet(window_id: int, close_price: float, outcome: str,
              payout: float, profit: float, won: bool,
              ai_model: str = "deepseek"):
    conn = _get_conn()
    conn.execute(
        """UPDATE bets SET close_price=?, outcome=?, payout=?, profit=?, won=?
           WHERE window_id=? AND ai_model=?""",
        (close_price, outcome, payout, profit, int(won), window_id, ai_model),
    )
    conn.commit()


# ── Queries for AI context ──────────────────────────────────

def get_historical_accuracy(limit: int = 50) -> dict:
    """Get prediction accuracy stats from DB for AI prompt context."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT prediction, outcome, correct, confidence, original_confidence,
                  rsi, macd_histogram, bb_pct, momentum, volatility
           FROM predictions WHERE outcome IS NOT NULL AND ai_model IN ('deepseek', 'deepseek_smc')
           ORDER BY id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    if not rows:
        return {"total": 0}

    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    up_preds = [r for r in rows if r["prediction"] == "UP"]
    dn_preds = [r for r in rows if r["prediction"] == "DOWN"]
    up_correct = sum(1 for r in up_preds if r["correct"]) if up_preds else 0
    dn_correct = sum(1 for r in dn_preds if r["correct"]) if dn_preds else 0

    # Find which TA conditions correlate with correct predictions
    patterns = _analyze_patterns(rows)

    return {
        "total": total,
        "correct": correct,
        "pct": round(correct / total * 100, 1),
        "up_total": len(up_preds),
        "up_correct": up_correct,
        "up_pct": round(up_correct / len(up_preds) * 100, 1) if up_preds else 0,
        "dn_total": len(dn_preds),
        "dn_correct": dn_correct,
        "dn_pct": round(dn_correct / len(dn_preds) * 100, 1) if dn_preds else 0,
        "patterns": patterns,
        "last10": "".join(
            "W" if r["correct"] else "L" for r in rows[:10]
        ),
    }


def _analyze_patterns(rows: list) -> str:
    """Find TA conditions that correlate with correct/incorrect predictions."""
    if len(rows) < 10:
        return "Datos insuficientes para analisis de patrones"

    insights = []

    # RSI ranges
    high_rsi = [r for r in rows if r["rsi"] and r["rsi"] > 65]
    low_rsi = [r for r in rows if r["rsi"] and r["rsi"] < 35]
    if len(high_rsi) >= 3:
        pct = sum(1 for r in high_rsi if r["correct"]) / len(high_rsi) * 100
        insights.append(f"RSI>65: {pct:.0f}% accuracy ({len(high_rsi)} cases)")
    if len(low_rsi) >= 3:
        pct = sum(1 for r in low_rsi if r["correct"]) / len(low_rsi) * 100
        insights.append(f"RSI<35: {pct:.0f}% accuracy ({len(low_rsi)} cases)")

    # High vs low volatility
    high_vol = [r for r in rows if r["volatility"] and r["volatility"] > 0.05]
    low_vol = [r for r in rows if r["volatility"] and r["volatility"] <= 0.05]
    if len(high_vol) >= 3:
        pct = sum(1 for r in high_vol if r["correct"]) / len(high_vol) * 100
        insights.append(f"Alta volatilidad: {pct:.0f}% accuracy ({len(high_vol)} cases)")
    if len(low_vol) >= 3:
        pct = sum(1 for r in low_vol if r["correct"]) / len(low_vol) * 100
        insights.append(f"Baja volatilidad: {pct:.0f}% accuracy ({len(low_vol)} cases)")

    # MACD histogram positive vs negative
    macd_pos = [r for r in rows if r["macd_histogram"] and r["macd_histogram"] > 0]
    macd_neg = [r for r in rows if r["macd_histogram"] and r["macd_histogram"] < 0]
    if len(macd_pos) >= 3:
        pct = sum(1 for r in macd_pos if r["correct"]) / len(macd_pos) * 100
        insights.append(f"MACD+: {pct:.0f}% accuracy ({len(macd_pos)} cases)")
    if len(macd_neg) >= 3:
        pct = sum(1 for r in macd_neg if r["correct"]) / len(macd_neg) * 100
        insights.append(f"MACD-: {pct:.0f}% accuracy ({len(macd_neg)} cases)")

    # Confidence calibration
    high_conf = [r for r in rows if r["confidence"] and r["confidence"] >= 70]
    mid_conf = [r for r in rows if r["confidence"] and 55 <= r["confidence"] < 70]
    if len(high_conf) >= 3:
        pct = sum(1 for r in high_conf if r["correct"]) / len(high_conf) * 100
        insights.append(f"Confianza>=70: {pct:.0f}% real accuracy ({len(high_conf)} cases)")
    if len(mid_conf) >= 3:
        pct = sum(1 for r in mid_conf if r["correct"]) / len(mid_conf) * 100
        insights.append(f"Confianza 55-69: {pct:.0f}% real accuracy ({len(mid_conf)} cases)")

    return "; ".join(insights) if insights else "Sin patrones claros aun"


def get_last_window_state() -> dict:
    """Get the last window_id and open_time from DB for restart recovery."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT w.window_id, w.open_time
           FROM windows w ORDER BY w.window_id DESC LIMIT 1"""
    ).fetchone()
    if not row:
        return {"window_id": 0, "open_time": 0.0}
    return {
        "window_id": row["window_id"],
        "open_time": row["open_time"],
    }


def get_recent_windows(limit: int = 20) -> list[dict]:
    """Get recent window results for AI context."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT w.window_id, w.open_price, w.close_price, w.outcome,
                  p.prediction, p.confidence, p.correct,
                  p.rsi, p.macd_histogram, p.bb_pct, p.momentum,
                  p.rag_strategy
           FROM windows w
           LEFT JOIN predictions p ON w.window_id = p.window_id
                AND p.ai_model IN ('deepseek', 'deepseek_smc')
           WHERE w.outcome IS NOT NULL
           ORDER BY w.id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_bet_stats() -> dict:
    """Get betting performance stats."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT COUNT(*) as total,
                  SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as wins,
                  SUM(profit) as total_profit,
                  SUM(amount) as total_wagered
           FROM bets WHERE outcome IS NOT NULL AND is_shadow=0
                          AND is_ghost=0"""
    ).fetchone()
    if not row or row["total"] == 0:
        return {"total": 0}
    return {
        "total": row["total"],
        "wins": row["wins"],
        "win_rate": round(row["wins"] / row["total"] * 100, 1),
        "total_profit": round(row["total_profit"] or 0, 2),
        "total_wagered": round(row["total_wagered"] or 0, 2),
        "roi": round((row["total_profit"] or 0) / (row["total_wagered"] or 1) * 100, 1),
    }


def get_early_vs_delayed_stats() -> dict:
    """Compare performance of early bets vs delayed bets."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT is_early, COUNT(*) as total,
                  SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as wins
           FROM bets WHERE outcome IS NOT NULL AND is_manual=0
           GROUP BY is_early"""
    ).fetchall()
    result = {"early": {"total": 0, "wins": 0}, "delayed": {"total": 0, "wins": 0}}
    for r in rows:
        key = "early" if r["is_early"] else "delayed"
        result[key] = {"total": r["total"], "wins": r["wins"]}
    return result


def get_rolling_atr_baseline(hours: int = 24) -> float:
    """Get the average ATR over the specified hours to detect volatility compression."""
    conn = _get_conn()
    row = conn.execute(
        f"SELECT AVG(atr_pct) as avg_atr FROM predictions WHERE atr_pct > 0 AND created_at >= datetime('now', '-{hours} hours')"
    ).fetchone()
    return float(row["avg_atr"]) if row and row["avg_atr"] is not None else 0.0


def get_price_history_from_db(minutes: int = 30) -> list[float]:
    """Get stored price ticks from DB for TA on cold start."""
    conn = _get_conn()
    cutoff = time.time() - (minutes * 60)
    rows = conn.execute(
        "SELECT price FROM price_ticks WHERE timestamp > ? ORDER BY timestamp",
        (cutoff,),
    ).fetchall()
    return [r["price"] for r in rows]


# ── AI Memory (persistent state between sessions) ─────────

def save_ai_memory(key: str, value):
    """Save a key-value pair to AI memory."""
    conn = _get_conn()
    val_str = json.dumps(value) if not isinstance(value, str) else value
    conn.execute(
        "INSERT OR REPLACE INTO ai_memory (key, value, updated_at) "
        "VALUES (?, ?, datetime('now'))",
        (key, val_str),
    )
    conn.commit()


def load_ai_memory(key: str, default=None):
    """Load a value from AI memory."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT value FROM ai_memory WHERE key=?", (key,)
    ).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return row["value"]


def load_all_ai_memory() -> dict:
    """Load all AI memory as a dict."""
    conn = _get_conn()
    rows = conn.execute("SELECT key, value FROM ai_memory").fetchall()
    result = {}
    for r in rows:
        try:
            result[r["key"]] = json.loads(r["value"])
        except (json.JSONDecodeError, TypeError):
            result[r["key"]] = r["value"]
    return result


# ── Liquidations ──────────────────────────────────────────

def cleanup_old_liquidations(hours: int = 24):
    """Remove liquidation records older than N hours to keep DB small."""
    conn = _get_conn()
    cutoff = time.time() - (hours * 3600)
    conn.execute("DELETE FROM liquidations WHERE timestamp < ?", (cutoff,))
    conn.commit()


def cleanup_old_ticks(hours: int = 72):
    """Remove price ticks older than N hours to prevent unbounded DB growth."""
    conn = _get_conn()
    cutoff = time.time() - (hours * 3600)
    result = conn.execute(
        "DELETE FROM price_ticks WHERE timestamp < ?", (cutoff,)
    )
    conn.commit()
    deleted = result.rowcount
    if deleted > 0:
        log.info(f"Cleaned up {deleted} price ticks older than {hours}h")
    return deleted


def get_stats_history(limit: int = 200) -> dict:
    """Get aggregated stats for the history analytics panel.

    Returns PnL evolution, rolling win rate, real vs ghost distribution,
    and summary stats for stat cards.
    """
    conn = _get_conn()

    # ── All bets (resolved + pending) ordered chronologically ──
    all_bets = conn.execute(
        """SELECT b.id, b.window_id, b.side, b.amount, b.odds_cents,
                  b.profit, b.won, b.is_ghost, b.is_shadow, b.ai_model,
                  b.created_at, b.strategy_name, b.odds_source,
                  w.open_price, w.close_price, w.outcome,
                  p.rag_strategy,
                  p.confidence AS ai_confidence,
                  p.prediction AS ai_prediction,
                  p.ml_oracle_prob, p.ml_skipped,
                  p.signal_quality
           FROM bets b
           LEFT JOIN windows w ON b.window_id = w.window_id
           LEFT JOIN predictions p ON b.window_id = p.window_id
                                   AND (b.ai_model = p.ai_model
                                        OR (b.ai_model = 'deepseek_smc'
                                            AND p.ai_model = 'deepseek'))
           ORDER BY b.id DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()

    # Reverse to chronological order for PnL series calculation
    all_bets = list(reversed(all_bets))

    # ── Separate resolved vs pending, then real/ghost/shadow ──
    # Exclude CANCELLED bets from stats (they are dead windows from restarts)
    # |explore bets are real $1 recycles from risk-vetoes — reclassified as
    # ghost-equivalent in the UI so the history filter groups them with ghosts.
    def _is_explore(row) -> bool:
        sn = (row["strategy_name"] or "") if row["strategy_name"] is not None else ""
        return "explore" in sn.lower()

    resolved_bets = [dict(r) for r in all_bets
                     if r["outcome"] is not None and r["outcome"] != "CANCELLED"]
    pending_bets = [dict(r) for r in all_bets if r["outcome"] is None]
    # Effective ghost = DB is_ghost OR strategy_name contains "explore"
    real_bets = [b for b in resolved_bets
                 if not b["is_ghost"] and not _is_explore(b) and not b["is_shadow"]]
    ghost_bets = [b for b in resolved_bets
                  if (b["is_ghost"] or _is_explore(b)) and not b["is_shadow"]]
    shadow_bets = [b for b in resolved_bets if b["is_shadow"]]

    # ── PnL evolution (cumulative, real bets only) ──
    pnl_series = []
    cumulative = 0.0
    for b in real_bets:
        cumulative += (b["profit"] or 0)
        pnl_series.append({
            "window_id": b["window_id"],
            "profit": round(b["profit"] or 0, 2),
            "cumulative": round(cumulative, 2),
            "created_at": b["created_at"],
        })

    # ── Rolling win rate (last 20 resolved predictions, real only) ──
    win_rate_series = []
    for i in range(len(real_bets)):
        window = real_bets[max(0, i - 19):i + 1]
        wins = sum(1 for x in window if x["won"])
        rate = round(wins / len(window) * 100, 1)
        win_rate_series.append({
            "window_id": real_bets[i]["window_id"],
            "win_rate": rate,
            "sample": len(window),
        })

    # ── Summary stats ──
    total_real = len(real_bets)
    total_ghost = len(ghost_bets)
    total_shadow = len(shadow_bets)
    total_pending = len(pending_bets)
    total_all = total_real + total_ghost + total_shadow
    real_wins = sum(1 for b in real_bets if b["won"])
    ghost_wins = sum(1 for b in ghost_bets if b["won"])
    shadow_wins = sum(1 for b in shadow_bets if b["won"])
    real_win_rate = round(real_wins / total_real * 100, 1) if total_real else 0
    ghost_win_rate = round(ghost_wins / total_ghost * 100, 1) if total_ghost else 0
    shadow_win_rate = round(shadow_wins / total_shadow * 100, 1) if total_shadow else 0
    total_pnl = round(sum(b["profit"] or 0 for b in real_bets), 2)
    shadow_pnl = round(sum(b["profit"] or 0 for b in shadow_bets), 2)

    # ── Streak calculations (last 30 resolved bets, all types) ──
    # Use most-recent-first order for streak calc
    recent_30 = list(reversed(resolved_bets))[:30]

    # Current streak: consecutive W or L from most recent
    current_streak = 0
    streak_type = None  # 'W' or 'L'
    for b in recent_30:
        w = bool(b["won"])
        if streak_type is None:
            streak_type = 'W' if w else 'L'
            current_streak = 1
        elif (w and streak_type == 'W') or (not w and streak_type == 'L'):
            current_streak += 1
        else:
            break

    # Best win streak and worst loss streak in last 30
    best_win_streak = 0
    worst_loss_streak = 0
    cur_w = 0
    cur_l = 0
    for b in recent_30:
        if b["won"]:
            cur_w += 1
            cur_l = 0
            best_win_streak = max(best_win_streak, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            worst_loss_streak = max(worst_loss_streak, cur_l)

    # Last 30 stats
    r30_wins = sum(1 for b in recent_30 if b["won"])
    r30_total = len(recent_30)
    r30_wr = round(r30_wins / r30_total * 100, 1) if r30_total else 0
    r30_pnl = round(sum((dict(b) if not isinstance(b, dict) else b).get("profit", 0) or 0
                        for b in recent_30
                        if not b["is_ghost"] and not _is_explore(b)
                        and not b["is_shadow"]), 2)

    # ML readiness: need ghost + real data diversity
    # Score based on total samples, ghost coverage, and outcome balance
    ml_samples = total_all
    if ml_samples >= 200:
        ml_readiness = "EXCELENTE"
        ml_pct = 100
    elif ml_samples >= 100:
        ml_readiness = "BUENA"
        ml_pct = 75
    elif ml_samples >= 50:
        ml_readiness = "ACEPTABLE"
        ml_pct = 50
    elif ml_samples >= 20:
        ml_readiness = "EN PROGRESO"
        ml_pct = 25
    else:
        ml_readiness = "INSUFICIENTE"
        ml_pct = max(5, round(ml_samples / 20 * 25))

    # ── Full bet list for table (most recent first) ──
    # Include both resolved and pending bets
    table_bets = []
    for r in reversed(list(all_bets)):
        b = dict(r)
        # Skip cancelled bets (dead windows from restarts)
        if b["outcome"] == "CANCELLED":
            continue
        is_pending = b["outcome"] is None
        table_bets.append({
            "window_id": b["window_id"],
            "side": b["side"],
            "amount": round(b["amount"] or 0, 2),
            "odds_cents": b["odds_cents"],
            "profit": round(b["profit"] or 0, 2),
            "won": bool(b["won"]) if b["won"] is not None else None,
            "is_ghost": bool(b["is_ghost"]) or _is_explore(b),
            "is_explore": _is_explore(b),
            "is_shadow": bool(b["is_shadow"]),
            "is_pending": is_pending,
            "ai_model": b["ai_model"] or "deepseek",
            "strategy_name": b.get("strategy_name") or "smc_liquidity",
            "rag_strategy": b.get("rag_strategy") or "",
            "open_price": b["open_price"],
            "close_price": b["close_price"],
            "outcome": b["outcome"],
            "created_at": b["created_at"],
            "ai_confidence": b.get("ai_confidence"),
            "ai_prediction": b.get("ai_prediction"),
            "ml_oracle_prob": b.get("ml_oracle_prob"),
            "ml_skipped": bool(b.get("ml_skipped")),
            "signal_quality": b.get("signal_quality"),
            "odds_source": b.get("odds_source") or "internal_fallback",
        })

    return {
        "pnl_series": pnl_series,
        "win_rate_series": win_rate_series,
        "summary": {
            "total_real": total_real,
            "total_ghost": total_ghost,
            "total_shadow": total_shadow,
            "total_pending": total_pending,
            "total_all": total_all,
            "real_win_rate": real_win_rate,
            "ghost_win_rate": ghost_win_rate,
            "shadow_win_rate": shadow_win_rate,
            "shadow_pnl": shadow_pnl,
            "total_pnl": total_pnl,
            "ml_readiness": ml_readiness,
            "ml_pct": ml_pct,
            "current_streak": current_streak,
            "streak_type": streak_type or '-',
            "best_win_streak": best_win_streak,
            "worst_loss_streak": worst_loss_streak,
            "r30_win_rate": r30_wr,
            "r30_pnl": r30_pnl,
            "r30_total": r30_total,
        },
        "table": table_bets,
    }


# ── Agent Portfolio Management ───────────────────────────────

def get_agent_portfolio(agent_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM agent_portfolios WHERE agent_id=?", (agent_id,)
    ).fetchone()
    return dict(row) if row else None


def upsert_agent_portfolio(agent_id: str, display_name: str = "",
                           strategy: str = "", initial_balance: float = 100.0):
    conn = _get_conn()
    existing = conn.execute(
        "SELECT agent_id FROM agent_portfolios WHERE agent_id=?", (agent_id,)
    ).fetchone()
    if existing:
        conn.execute(
            """UPDATE agent_portfolios SET display_name=?, strategy=?,
               updated_at=datetime('now') WHERE agent_id=?""",
            (display_name, strategy, agent_id),
        )
    else:
        conn.execute(
            """INSERT INTO agent_portfolios
               (agent_id, display_name, strategy, balance, initial_balance, peak_balance)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (agent_id, display_name, strategy,
             initial_balance, initial_balance, initial_balance),
        )
    conn.commit()


def update_agent_balance(agent_id: str, profit: float, won: bool):
    conn = _get_conn()
    row = conn.execute(
        "SELECT balance, peak_balance FROM agent_portfolios WHERE agent_id=?",
        (agent_id,),
    ).fetchone()
    if not row:
        return
    new_balance = round(row["balance"] + profit, 2)
    new_peak = max(row["peak_balance"], new_balance)
    conn.execute(
        """UPDATE agent_portfolios SET
           balance=?, peak_balance=?,
           total_bets = total_bets + 1,
           wins = wins + ?,
           losses = losses + ?,
           total_profit = round(total_profit + ?, 2),
           updated_at=datetime('now')
           WHERE agent_id=?""",
        (new_balance, new_peak, int(won), int(not won), profit, agent_id),
    )
    conn.commit()


def get_all_agent_portfolios() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM agent_portfolios ORDER BY agent_id"
    ).fetchall()
    return [dict(r) for r in rows]


def reset_agent_portfolio(agent_id: str, new_balance: float = 100.0):
    conn = _get_conn()
    conn.execute(
        """UPDATE agent_portfolios SET
           balance=?, initial_balance=?, total_bets=0, wins=0, losses=0,
           total_profit=0.0, peak_balance=?,
           score=0, consecutive_losses=0, consecutive_wins=0,
           is_fatigued=0, rotation_idx=0, fatigue_since=0,
           updated_at=datetime('now')
           WHERE agent_id=?""",
        (new_balance, new_balance, new_balance, agent_id),
    )
    conn.commit()


def update_agent_score_and_streak(agent_id: str, won: bool, is_fatigued: bool):
    """Update score and win/loss streaks after a bet resolves.

    Scoring: +10 win, +15 win streak>=3, +20 win while fatigued,
             -5 loss, -10 loss streak>=3.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT score, consecutive_wins, consecutive_losses FROM agent_portfolios WHERE agent_id=?",
        (agent_id,),
    ).fetchone()
    if not row:
        return
    score = row["score"]
    c_wins = row["consecutive_wins"]
    c_losses = row["consecutive_losses"]

    if won:
        c_wins += 1
        c_losses = 0
        if is_fatigued:
            score += 20   # recovery bonus
        elif c_wins >= 3:
            score += 15   # streak bonus
        else:
            score += 10
    else:
        c_losses += 1
        c_wins = 0
        if c_losses >= 3:
            score -= 10   # fatigue penalty
        else:
            score -= 5

    conn.execute(
        """UPDATE agent_portfolios SET
           score=?, consecutive_wins=?, consecutive_losses=?,
           updated_at=datetime('now') WHERE agent_id=?""",
        (score, c_wins, c_losses, agent_id),
    )
    conn.commit()


def update_agent_fatigue_db(agent_id: str, is_fatigued: bool,
                            rotation_idx: int, fatigue_since: float):
    """Persist fatigue state to SQLite so it survives restarts."""
    conn = _get_conn()
    conn.execute(
        """UPDATE agent_portfolios SET
           is_fatigued=?, rotation_idx=?, fatigue_since=?,
           updated_at=datetime('now') WHERE agent_id=?""",
        (int(is_fatigued), rotation_idx, fatigue_since, agent_id),
    )
    conn.commit()


def load_all_fatigue_states() -> dict:
    """Load fatigue state for all agents from DB. Used at engine startup."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT agent_id, consecutive_losses, is_fatigued,
                  rotation_idx, fatigue_since
           FROM agent_portfolios WHERE is_active=1"""
    ).fetchall()
    states = {}
    for r in rows:
        states[r["agent_id"]] = {
            "losses": r["consecutive_losses"],
            "start_time": r["fatigue_since"] if r["fatigue_since"] > 0
                          else time.time(),
            "fatigued": bool(r["is_fatigued"]),
            "fatigue_since": r["fatigue_since"],
            "rotation_idx": r["rotation_idx"],
        }
    return states


def reset_agent_fatigue_db(agent_id: str):
    """Reset fatigue state for a single agent in DB."""
    conn = _get_conn()
    conn.execute(
        """UPDATE agent_portfolios SET
           is_fatigued=0, rotation_idx=0, fatigue_since=0,
           consecutive_losses=0, consecutive_wins=0,
           updated_at=datetime('now') WHERE agent_id=?""",
        (agent_id,),
    )
    conn.commit()


def get_agent_daily_stats(agent_id: str) -> dict:
    """Get today's P&L, bet count, and current losing streak for an agent.

    Returns dict with keys: daily_pnl, daily_bets, losing_streak.
    """
    conn = _get_conn()
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).strftime("%Y-%m-%d %H:%M:%S")

    # Today's P&L and bet count
    row = conn.execute(
        """SELECT COALESCE(SUM(profit), 0) as daily_pnl,
                  COUNT(*) as daily_bets
           FROM bets
           WHERE ai_model=? AND outcome IS NOT NULL
             AND is_ghost=0 AND is_shadow=0
             AND created_at >= ?""",
        (agent_id, today_start),
    ).fetchone()
    daily_pnl = row["daily_pnl"] if row else 0.0
    daily_bets = row["daily_bets"] if row else 0

    # Current losing streak (count consecutive losses from most recent)
    recent = conn.execute(
        """SELECT won FROM bets
           WHERE ai_model=? AND outcome IS NOT NULL
             AND is_ghost=0 AND is_shadow=0
           ORDER BY id DESC LIMIT 20""",
        (agent_id,),
    ).fetchall()
    losing_streak = 0
    for r in recent:
        if r["won"]:
            break
        losing_streak += 1

    return {
        "daily_pnl": round(daily_pnl, 2),
        "daily_bets": daily_bets,
        "losing_streak": losing_streak,
    }


def cancel_stale_bets(max_age_seconds: int = 600) -> int:
    """Cancel bets from windows that never resolved and are too old.

    These are from windows where the system restarted before the window
    could close — the window has no outcome and never will.
    Marks them with outcome='CANCELLED' so they don't pollute the history.

    Returns the number of stale bets cancelled.
    """
    conn = _get_conn()
    cutoff = time.time() - max_age_seconds
    cancelled = conn.execute(
        """UPDATE bets SET outcome='CANCELLED', won=0, profit=0, payout=0
           WHERE outcome IS NULL
             AND window_id IN (
               SELECT w.window_id FROM windows w
               WHERE w.outcome IS NULL AND w.open_time < ?
             )""",
        (cutoff,),
    ).rowcount
    if cancelled:
        conn.commit()
        log.info(f"[RECOVERY] Cancelled {cancelled} stale bets from "
                 f"dead windows (>{max_age_seconds}s old)")
    return cancelled


def resolve_orphaned_bets() -> int:
    """Resolve bets whose windows have been resolved but the bets were not.

    This happens when the system restarts mid-window — the window gets closed
    on the next run but the bet resolution was skipped.
    Calculates P&L using Polymarket formula and updates agent portfolios.

    Returns the number of orphaned bets resolved.
    """
    conn = _get_conn()
    orphans = conn.execute(
        """SELECT b.id, b.window_id, b.side, b.amount, b.odds_cents,
                  b.ai_model, b.is_ghost, b.is_shadow,
                  w.outcome, w.close_price
           FROM bets b
           JOIN windows w ON b.window_id = w.window_id
           WHERE b.outcome IS NULL AND w.outcome IS NOT NULL"""
    ).fetchall()

    if not orphans:
        return 0

    resolved = 0
    for row in orphans:
        won = row["side"] == row["outcome"]
        amount = row["amount"] or 0
        odds_cents = row["odds_cents"] or 50
        is_ghost = bool(row["is_ghost"])

        if is_ghost:
            payout, profit = 0.0, 0.0
        else:
            price = max(0.01, min(0.99, odds_cents / 100.0))
            tokens = amount / price
            if won:
                gross_profit = tokens - amount
                fee = gross_profit * 0.02
                payout = tokens - fee
                profit = payout - amount
            else:
                payout = 0.0
                profit = -amount

        conn.execute(
            """UPDATE bets SET close_price=?, outcome=?, payout=?, profit=?,
                      won=? WHERE id=?""",
            (row["close_price"], row["outcome"], round(payout, 4),
             round(profit, 4), int(won), row["id"]),
        )

        # Update agent portfolio if not ghost
        if not is_ghost and row["ai_model"]:
            try:
                update_agent_balance(row["ai_model"], round(profit, 4), won)
            except Exception:
                pass  # portfolio may not exist

        resolved += 1

    conn.commit()
    log.info(f"[RECOVERY] Resolved {resolved} orphaned bets from previous sessions")
    return resolved
