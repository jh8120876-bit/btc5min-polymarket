#!/usr/bin/env python3
"""
clean_db.py — Deduplicate predictions and bets tables in btc5min.db.

For each window_id with multiple rows, keeps the row with the highest id
(most recent insert) and deletes all older clones.

Creates a backup before any modification.
"""

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "btc5min.db"


def main():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Nothing to clean — the bot hasn't created the DB yet.")
        sys.exit(0)

    # ── Backup first ──────────────────────────────────────
    backup = DB_PATH.with_suffix(
        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    )
    shutil.copy2(DB_PATH, backup)
    print(f"Backup created: {backup.name}")

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    # ── Diagnose ──────────────────────────────────────────
    for table in ("predictions", "bets"):
        dupes = conn.execute(f"""
            SELECT window_id, ai_model, COUNT(*) as cnt
            FROM {table}
            GROUP BY window_id, ai_model
            HAVING cnt > 1
            ORDER BY window_id
        """).fetchall()

        if not dupes:
            print(f"\n[{table}] No duplicates found.")
            continue

        total_duped_rows = sum(cnt for _, _, cnt in dupes)
        rows_to_delete = sum(cnt - 1 for _, _, cnt in dupes)
        print(f"\n[{table}] {len(dupes)} window_ids with duplicates "
              f"({total_duped_rows} total rows, {rows_to_delete} to remove):")
        for wid, model, cnt in dupes[:20]:
            print(f"  window_id={wid}, ai_model={model}: {cnt} rows")
        if len(dupes) > 20:
            print(f"  ... and {len(dupes) - 20} more")

        # ── Clean: keep highest id per window_id ──────────
        cursor = conn.execute(f"""
            DELETE FROM {table}
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM {table}
                GROUP BY window_id, ai_model
            )
        """)
        deleted = cursor.rowcount
        conn.commit()
        print(f"  Deleted {deleted} duplicate rows from [{table}].")

    # ── Final stats ───────────────────────────────────────
    print("\n-- Post-cleanup stats --")
    for table in ("predictions", "bets", "windows"):
        try:
            row = conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()
            print(f"  {table}: {row[0]} rows")
        except Exception:
            pass

    # Verify no duplicates remain
    clean = True
    for table in ("predictions", "bets"):
        dupes = conn.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT window_id, ai_model FROM {table}
                GROUP BY window_id, ai_model HAVING COUNT(*) > 1
            )
        """).fetchone()[0]
        if dupes > 0:
            print(f"  WARNING: {table} still has {dupes} duplicated window_ids!")
            clean = False

    conn.close()

    if clean:
        print("\nAll clean. Zero duplicates remain.")
    print("Done.")


if __name__ == "__main__":
    main()
