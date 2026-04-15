"""
Coliseum Evaluate — Weekly performance audit of swarm agent prompts.

Reads the last 7 days of bets from btc5min.db, groups by
(ai_model x strategy_name), and computes:
  - Winrate
  - Sharpe ratio (using per-bet P&L)
  - Profit factor
  - Total P&L

Exports a leaderboard CSV to data/coliseum_leaderboard_{date}.csv.

Usage:
    python -m scripts.coliseum_evaluate
"""

import csv
import math
import sqlite3
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "btc5min.db"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def evaluate(days: int = 7) -> list[dict]:
    """Evaluate all agent x strategy combos over the last N days.

    Returns a list of dicts sorted by sharpe descending:
      [{agent_id, strategy, trades, wins, losses, winrate, pnl, sharpe, profit_factor}]
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    rows = conn.execute("""
        SELECT ai_model, strategy_name, won, profit
        FROM bets
        WHERE created_at >= ?
          AND won IS NOT NULL
          AND is_ghost = 0
        ORDER BY ai_model, strategy_name
    """, (cutoff_str,)).fetchall()
    conn.close()

    if not rows:
        print(f"[COLISEUM] No resolved bets in the last {days} days")
        return []

    # Group by (ai_model, strategy_name)
    groups: dict[tuple, list] = {}
    for r in rows:
        key = (r["ai_model"] or "unknown", r["strategy_name"] or "unknown")
        groups.setdefault(key, []).append(r)

    results = []
    for (agent_id, strategy), bets in groups.items():
        profits = [b["profit"] for b in bets if b["profit"] is not None]
        wins = sum(1 for b in bets if b["won"])
        losses = len(bets) - wins
        total_pnl = sum(profits)
        winrate = wins / len(bets) * 100 if bets else 0

        # Sharpe ratio (annualized-ish: per-bet Sharpe * sqrt(trades/day))
        if len(profits) >= 2:
            mean_ret = statistics.mean(profits)
            std_ret = statistics.stdev(profits)
            sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        # Profit factor: gross_wins / abs(gross_losses)
        gross_wins = sum(p for p in profits if p > 0)
        gross_losses = abs(sum(p for p in profits if p < 0))
        profit_factor = (gross_wins / gross_losses
                         if gross_losses > 0 else float("inf"))

        results.append({
            "agent_id": agent_id,
            "strategy": strategy,
            "trades": len(bets),
            "wins": wins,
            "losses": losses,
            "winrate": round(winrate, 1),
            "pnl": round(total_pnl, 2),
            "sharpe": round(sharpe, 3),
            "profit_factor": round(profit_factor, 2),
        })

    # Sort by sharpe descending
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results


def export_csv(results: list[dict]) -> Path:
    """Write the leaderboard to a dated CSV."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    path = OUTPUT_DIR / f"coliseum_leaderboard_{date_str}.csv"

    fieldnames = ["agent_id", "strategy", "trades", "wins", "losses",
                  "winrate", "pnl", "sharpe", "profit_factor"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[COLISEUM] Leaderboard exported to {path}")
    return path


def main():
    print("[COLISEUM] Evaluating last 7 days of swarm performance...")
    results = evaluate(days=7)

    if not results:
        print("[COLISEUM] No data — nothing to export")
        return

    print(f"\n{'Agent':<25} {'Strategy':<30} {'Trades':>6} {'WR%':>6} "
          f"{'P&L':>8} {'Sharpe':>7} {'PF':>6}")
    print("-" * 95)
    for r in results:
        print(f"{r['agent_id']:<25} {r['strategy']:<30} {r['trades']:>6} "
              f"{r['winrate']:>5.1f}% {r['pnl']:>+7.2f} "
              f"{r['sharpe']:>7.3f} {r['profit_factor']:>6.2f}")

    export_csv(results)


if __name__ == "__main__":
    main()
