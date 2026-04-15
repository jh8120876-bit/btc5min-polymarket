#!/usr/bin/env python3
"""
Generate training dataset from btc5min SQLite database.
Joins predictions, windows, bets, market_context and liquidations
into a clean CSV with both classification and regression targets.

Usage:
    python -m ml.generate_dataset                  # export all data
    python -m ml.generate_dataset --last-days 7    # last 7 days only
    python -m ml.generate_dataset --output my.csv  # custom output path
"""

import argparse
import csv
import sqlite3
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(_PROJECT_ROOT / "btc5min.db")
DATA_DIR = _PROJECT_ROOT / "data"


def _default_output(model: str) -> str:
    """Generate output filename with model discriminator."""
    safe_name = model.replace("/", "_").replace("\\", "_")
    DATA_DIR.mkdir(exist_ok=True)
    return str(DATA_DIR / f"training_data_{safe_name}.csv")


def generate(db_path: str, output_path: str, last_days: int | None = None,
             model: str = "deepseek", include_ghosts: bool = False):
    if not Path(db_path).exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if model == "all":
        where = "WHERE w.outcome IS NOT NULL"
        params = []
    else:
        where = "WHERE w.outcome IS NOT NULL AND p.ai_model = ?"
        params = [model]
    if not include_ghosts:
        where += " AND COALESCE(b.is_ghost, 0) = 0"
    if last_days:
        cutoff = time.time() - (last_days * 86400)
        where += " AND w.open_time > ?"
        params.append(cutoff)

    query = f"""
        SELECT
            -- Window data
            w.window_id,
            w.open_price,
            w.close_price,
            w.outcome,
            CASE WHEN w.close_price > w.open_price THEN 1 ELSE 0 END as target,
            CASE WHEN w.open_price > 0
                 THEN (w.close_price - w.open_price) / w.open_price * 100
                 ELSE 0 END as actual_change_pct,
            w.target_magnitude as target_log_return,

            -- Prediction data
            p.prediction,
            p.confidence,
            p.original_confidence,
            p.final_confidence,
            p.calibrated_confidence,
            p.correct,
            p.risk_score,

            -- TA features
            p.rsi,
            p.macd_histogram,
            p.bb_pct,
            p.ema_cross,
            p.momentum,
            p.volatility,
            p.tick_buy_pressure,
            p.atr_pct,

            -- ML features
            p.hour_utc,
            p.day_of_week,
            p.price_change_pct,
            p.return_1m,
            p.return_5m,
            p.return_15m,
            p.price_acceleration,
            p.range_position,
            p.volatility_zscore,

            -- Multi-timeframe
            p.trend_slope_15m,
            p.trend_slope_1h,
            p.trend_alignment,

            -- Signal quality
            COALESCE(p.signal_quality, 'NORMAL') as signal_quality,
            p.quality_reason,

            -- Liquidation context
            COALESCE(p.liq_buy_vol_5m, 0) as liq_buy_vol_5m,
            COALESCE(p.liq_sell_vol_5m, 0) as liq_sell_vol_5m,
            COALESCE(p.reversal_alert, 0) as reversal_alert,

            -- Local AI ensemble features (double-brain)
            p.local_ai_prediction,
            p.local_ai_confidence,
            p.local_ai_reasoning,

            -- Market context (predictions snapshot)
            p.volume_24h as pred_volume_24h,
            p.funding_rate as pred_funding_rate,
            p.open_interest as pred_open_interest,

            -- Market context (dedicated table)
            mc.volume_24h,
            mc.volume_1h,
            mc.funding_rate,
            mc.open_interest,
            mc.oi_change_pct,
            mc.bid_ask_spread,

            -- Bet data
            b.side as bet_side,
            b.amount as bet_amount,
            b.odds_cents as bet_odds,
            b.profit as bet_profit,
            b.won as bet_won,
            b.is_early as bet_is_early,
            COALESCE(b.is_ghost, 0) as is_ghost,

            -- ML enrichment: latency, price snapshot, consensus
            p.ai_latency_ms,
            p.price_at_bet,
            p.consensus_agreement,

            -- SMC features (smartmoneyconcepts library)
            COALESCE(p.smc_fvg_active, 0) as smc_fvg_active,
            COALESCE(p.smc_bos_last, 0) as smc_bos_last,
            COALESCE(p.smc_choch_last, 0) as smc_choch_last,
            COALESCE(p.smc_ob_nearest, 0) as smc_ob_nearest,
            p.smc_ob_distance_pct,
            p.smc_ob_strength,
            COALESCE(p.smc_liq_nearest, 0) as smc_liq_nearest,
            p.smc_liq_distance_pct,
            p.smc_retracement_pct,
            COALESCE(p.smc_in_ote_zone, 0) as smc_in_ote_zone,
            p.smc_support,
            p.smc_resistance,
            p.smc_kill_zone,
            COALESCE(p.smc_in_kill_zone, 0) as smc_in_kill_zone,

            -- Order book imbalance
            mc.order_book_imbalance,

            -- RAG strategy
            p.rag_strategy,

            -- ML Oracle
            p.ml_oracle_prob,
            COALESCE(p.ml_skipped, 0) as ml_skipped,

            -- Layer alignment audit
            p.layer_alignment,

            -- ML Judge B-features
            mc.bid_ask_spread,
            p.volatility_regime,
            p.fear_greed,

            -- Sniper metadata (for Judge training)
            p.sniper_eval_count,
            p.sniper_time_to_bet_sec,
            COALESCE(p.sniper_flipped, 0) as sniper_flipped,

            -- Polymarket EV context
            p.polymarket_odds,
            p.expected_value,
            p.distance_to_strike,

            -- Temporal velocity features (history vectors)
            p.rsi_velocity,
            p.rsi_acceleration,
            p.momentum_velocity,
            p.cvd_velocity,

            -- API cost tracking (analytical)
            p.api_tokens_used,

            -- Advanced market features (CVD, GEX, MSS)
            p.cvd_imbalance_pct,
            p.cvd_aggressor_ratio,
            p.is_liquidity_swept_1m,
            p.mss_detected,
            p.post_sweep_reversal,
            p.gex_net_bias,
            p.gex_dist_call_wall_pct,
            p.gex_dist_put_wall_pct,
            p.liq_cluster_distance_atr,
            p.liq_cluster_magnitude_pct,
            p.liq_cluster_side,

            -- Metadata
            p.ai_model,
            COALESCE(p.strategy_name, 'smc_liquidity') as strategy_name,
            w.open_time,
            p.created_at as prediction_time

        FROM windows w
        INNER JOIN predictions p ON w.window_id = p.window_id
        LEFT JOIN market_context mc ON w.window_id = mc.window_id
        LEFT JOIN bets b ON w.window_id = b.window_id AND b.ai_model = p.ai_model
        {where}
        ORDER BY w.window_id ASC
    """

    try:
        rows = conn.execute(query, params).fetchall()
    except sqlite3.OperationalError as e:
        print(f"WARNING: Query error ({e}), trying simplified query...")
        query = f"""
            SELECT
                w.window_id, w.open_price, w.close_price, w.outcome,
                CASE WHEN w.close_price > w.open_price THEN 1 ELSE 0 END as target,
                p.prediction, p.confidence, p.original_confidence,
                p.correct, p.risk_score,
                p.rsi, p.macd_histogram, p.bb_pct, p.ema_cross,
                p.momentum, p.volatility, p.tick_buy_pressure, p.atr_pct,
                b.profit as bet_profit, b.won as bet_won,
                COALESCE(b.is_ghost, 0) as is_ghost,
                p.ai_model,
                COALESCE(p.strategy_name, 'smc_liquidity') as strategy_name,
                w.open_time, p.created_at as prediction_time
            FROM windows w
            INNER JOIN predictions p ON w.window_id = p.window_id
            LEFT JOIN bets b ON w.window_id = b.window_id AND b.ai_model = p.ai_model
            {where}
            ORDER BY w.window_id ASC
        """
        rows = conn.execute(query, params).fetchall()

    if not rows:
        print("No data found matching criteria.")
        sys.exit(0)

    columns = rows[0].keys()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[col] for col in columns])

    conn.close()

    print(f"Exported {len(rows)} rows to {output_path} (model={model})")
    print(f"Columns: {len(columns)}")
    print(f"  Features: {', '.join(columns)}")

    # Stats
    targets = [row["target"] for row in rows if row["target"] is not None]
    if targets:
        up_pct = sum(targets) / len(targets) * 100
        print(f"\nTarget distribution: UP={up_pct:.1f}%, DOWN={100-up_pct:.1f}%")
        print(f"Total samples: {len(targets)}")

        # Regression target coverage
        log_returns = [row["target_log_return"] for row in rows
                       if row["target_log_return"] is not None]
        if log_returns:
            print(f"Log-return target: {len(log_returns)}/{len(rows)} populated "
                  f"(mean={sum(log_returns)/len(log_returns):.4f}%, "
                  f"min={min(log_returns):.4f}%, max={max(log_returns):.4f}%)")

        # Signal quality distribution
        qualities = {}
        for row in rows:
            q = row["signal_quality"] or "NORMAL"
            qualities[q] = qualities.get(q, 0) + 1
        if qualities:
            print(f"Signal quality: {qualities}")

        # Agent/strategy distribution (multi-agent swarm)
        if "ai_model" in columns:
            agent_dist = {}
            for row in rows:
                m = row["ai_model"] or "deepseek"
                agent_dist[m] = agent_dist.get(m, 0) + 1
            if len(agent_dist) > 1:
                print(f"\nAgent distribution:")
                for m, cnt in sorted(agent_dist.items(), key=lambda x: -x[1]):
                    print(f"  {m}: {cnt} ({cnt/len(rows)*100:.1f}%)")
        if "strategy_name" in columns:
            strat_dist = {}
            for row in rows:
                s = row["strategy_name"] or "smc_liquidity"
                strat_dist[s] = strat_dist.get(s, 0) + 1
            if strat_dist:
                print(f"Strategy distribution: {strat_dist}")

        # NULL report
        null_counts = {}
        for col in columns:
            nulls = sum(1 for row in rows if row[col] is None)
            if nulls > 0:
                null_counts[col] = nulls
        if null_counts:
            print(f"\nColumns with NULLs (may need imputation):")
            for col, count in sorted(null_counts.items(),
                                     key=lambda x: x[1], reverse=True):
                print(f"  {col}: {count}/{len(rows)} "
                      f"({count/len(rows)*100:.0f}% missing)")


def main():
    parser = argparse.ArgumentParser(
        description="Export btc5min data as training CSV for LightGBM"
    )
    parser.add_argument(
        "--db", default=DB_PATH,
        help=f"Path to SQLite database (default: {DB_PATH})"
    )
    parser.add_argument(
        "--model", default="deepseek",
        help="AI model to filter by (default: deepseek). "
             "Use 'all' for multi-agent swarm data, or specific agent IDs "
             "like 'claude_opus_reversion', 'chatgpt_momentum', etc."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (default: training_data_<model>.csv)"
    )
    parser.add_argument(
        "--last-days", type=int, default=None,
        help="Only export data from the last N days"
    )
    parser.add_argument(
        "--include-ghosts", action="store_true", default=False,
        help="Include ghost trades (is_ghost=1, amount=0) in export. "
             "Default: exclude them to avoid $0 PnL contamination in sizing models."
    )
    args = parser.parse_args()
    output = args.output or _default_output(args.model)
    generate(args.db, output, args.last_days, args.model, args.include_ghosts)


if __name__ == "__main__":
    main()
