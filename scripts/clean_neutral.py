#!/usr/bin/env python3
"""
clean_neutral.py -- Remove NEUTRAL/HOLD/FLAT predictions and their linked bets.

Keeps only UP/DOWN predictions in the database for clean binary classification.
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "btc5min.db"


def main():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        sys.exit(0)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    # -- Diagnose non-binary predictions --
    invalid = conn.execute(
        """SELECT prediction, COUNT(*) as cnt
           FROM predictions
           WHERE prediction NOT IN ('UP', 'DOWN')
           GROUP BY prediction"""
    ).fetchall()

    if not invalid:
        print("No non-binary predictions found. Database is clean.")
        conn.close()
        return

    print("Non-binary predictions found:")
    total_invalid = 0
    for pred, cnt in invalid:
        print(f"  '{pred}': {cnt} rows")
        total_invalid += cnt

    # -- Find window_ids of neutral predictions --
    neutral_windows = conn.execute(
        """SELECT window_id FROM predictions
           WHERE prediction NOT IN ('UP', 'DOWN')"""
    ).fetchall()
    neutral_wids = [r[0] for r in neutral_windows]

    # -- Delete linked bets first (FK integrity) --
    if neutral_wids:
        placeholders = ",".join("?" * len(neutral_wids))
        cursor = conn.execute(
            f"DELETE FROM bets WHERE window_id IN ({placeholders})",
            neutral_wids,
        )
        bets_deleted = cursor.rowcount
        print(f"\nDeleted {bets_deleted} linked bets from neutral windows")
    else:
        bets_deleted = 0

    # -- Delete neutral predictions --
    cursor = conn.execute(
        "DELETE FROM predictions WHERE prediction NOT IN ('UP', 'DOWN')"
    )
    preds_deleted = cursor.rowcount
    conn.commit()

    print(f"Deleted {preds_deleted} non-binary predictions")

    # -- Verify --
    remaining = conn.execute(
        """SELECT COUNT(*) FROM predictions
           WHERE prediction NOT IN ('UP', 'DOWN')"""
    ).fetchone()[0]

    total_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    total_bets = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]

    print(f"\n-- Post-cleanup --")
    print(f"  predictions: {total_preds} rows (all UP/DOWN)")
    print(f"  bets: {total_bets} rows")
    print(f"  non-binary remaining: {remaining}")

    conn.close()

    if remaining == 0:
        print("\nDatabase sanitized. Zero neutral predictions remain.")
    else:
        print(f"\nWARNING: {remaining} non-binary predictions still remain!")


if __name__ == "__main__":
    main()
