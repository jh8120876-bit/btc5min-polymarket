#!/usr/bin/env python3
"""
Generate AI_CONTEXT_NOW.md — dynamic state snapshot for AI assistants.

Reads btc5min.db and outputs a compact (~20 line) summary of the current
system state: trust, win rate, streak, daily P&L, ATR, ghost count, top lesson.

Usage:
    python generate_ai_context.py              # writes AI_CONTEXT_NOW.md
    python generate_ai_context.py --stdout     # print to stdout instead
"""

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(_PROJECT_ROOT / "btc5min.db")
OUTPUT_PATH = str(_PROJECT_ROOT / "AI_CONTEXT_NOW.md")


def generate() -> str:
    if not Path(DB_PATH).exists():
        return "# AI_CONTEXT_NOW\n\nDatabase not found. Run `python app.py` first.\n"

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    lines = ["# AI_CONTEXT_NOW — Auto-generated dynamic state",
             f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
             ""]

    # ── Trust score ──
    trust = 50.0
    try:
        row = conn.execute(
            "SELECT value FROM ai_memory WHERE key='trust_score'"
        ).fetchone()
        if row:
            trust = float(json.loads(row["value"]))
    except Exception:
        pass

    # ── Sessions count ──
    sessions = 0
    try:
        row = conn.execute(
            "SELECT value FROM ai_memory WHERE key='sessions_count'"
        ).fetchone()
        if row:
            sessions = int(json.loads(row["value"]))
    except Exception:
        pass

    # ── Last 20 windows win rate ──
    try:
        rows = conn.execute(
            """SELECT p.prediction, w.outcome, p.correct
               FROM windows w
               INNER JOIN predictions p ON w.window_id = p.window_id
               WHERE w.outcome IS NOT NULL AND p.ai_model='deepseek'
               ORDER BY w.window_id DESC LIMIT 20"""
        ).fetchall()
        # Fallback: try without ai_model filter
        if not rows:
            rows = conn.execute(
                """SELECT p.prediction, w.outcome, p.correct
                   FROM windows w
                   INNER JOIN predictions p ON w.window_id = p.window_id
                   WHERE w.outcome IS NOT NULL
                   ORDER BY w.window_id DESC LIMIT 20"""
            ).fetchall()
    except Exception:
        rows = []

    total = len(rows)
    wins = sum(1 for r in rows if r["correct"])
    wr = (wins / total * 100) if total > 0 else 0

    up_rows = [r for r in rows if r["prediction"] == "UP"]
    dn_rows = [r for r in rows if r["prediction"] == "DOWN"]
    up_w = sum(1 for r in up_rows if r["correct"])
    dn_w = sum(1 for r in dn_rows if r["correct"])

    # ── Current streak ──
    streak_type = ""
    streak_count = 0
    for r in rows:
        is_win = bool(r["correct"])
        if streak_count == 0:
            streak_type = "W" if is_win else "L"
            streak_count = 1
        elif (is_win and streak_type == "W") or (not is_win and streak_type == "L"):
            streak_count += 1
        else:
            break

    # ── Daily P&L (primary agent) ──
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).strftime("%Y-%m-%d %H:%M:%S")
    try:
        pnl_row = conn.execute(
            """SELECT COALESCE(SUM(profit), 0) as pnl
               FROM bets WHERE outcome IS NOT NULL
               AND is_ghost=0 AND is_shadow=0
               AND created_at >= ?""",
            (today_start,),
        ).fetchone()
        daily_pnl = pnl_row["pnl"] if pnl_row else 0
    except Exception:
        daily_pnl = 0

    # ── ATR baseline ──
    try:
        cutoff = time.time() - 86400
        atr_rows = conn.execute(
            """SELECT open_price, close_price FROM windows
               WHERE outcome IS NOT NULL AND open_time > ?
               ORDER BY open_time DESC""",
            (cutoff,),
        ).fetchall()
        if atr_rows:
            changes = [abs(r["close_price"] - r["open_price"]) / r["open_price"] * 100
                       for r in atr_rows if r["open_price"] and r["open_price"] > 0]
            atr_baseline = sum(changes) / len(changes) if changes else None
        else:
            atr_baseline = None
    except Exception:
        atr_baseline = None

    # ── Ghost trades count ──
    try:
        ghost_row = conn.execute(
            "SELECT COUNT(*) as cnt FROM bets WHERE is_ghost=1"
        ).fetchone()
        ghost_count = ghost_row["cnt"] if ghost_row else 0
    except Exception:
        ghost_count = 0

    # ── Total windows ──
    try:
        tw_row = conn.execute(
            "SELECT COUNT(*) as cnt FROM windows WHERE outcome IS NOT NULL"
        ).fetchone()
        total_windows = tw_row["cnt"] if tw_row else 0
    except Exception:
        total_windows = 0

    # ── Top lesson ──
    top_lesson = "N/A"
    try:
        row = conn.execute(
            "SELECT value FROM ai_memory WHERE key='learned_lessons'"
        ).fetchone()
        if row:
            lessons = json.loads(row["value"])
            if lessons:
                top_lesson = lessons[-1]
    except Exception:
        pass

    # ── Agent balances ──
    agent_info = []
    try:
        agents = conn.execute(
            "SELECT agent_id, balance, wins, losses FROM agent_portfolios"
        ).fetchall()
        for a in agents:
            agent_info.append(
                f"{a['agent_id']} ${a['balance']:.2f} "
                f"({a['wins']}W/{a['losses']}L)"
            )
    except Exception:
        pass

    # ── Bias ──
    bias_str = "None"
    try:
        row = conn.execute(
            "SELECT value FROM ai_memory WHERE key='bias_corrections'"
        ).fetchone()
        if row:
            bias = json.loads(row["value"])
            if bias:
                bias_str = "; ".join(bias.keys())
    except Exception:
        pass

    conn.close()

    # ── Build output ──
    lines.append(f"Trust: {trust:.1f} | Sessions: {sessions} | "
                 f"Total windows: {total_windows}")
    lines.append(f"Last 20: {wins}W/{total - wins}L ({wr:.0f}%) | "
                 f"UP: {up_w}W/{len(up_rows) - up_w}L | "
                 f"DOWN: {dn_w}W/{len(dn_rows) - dn_w}L")
    lines.append(f"Streak: {streak_count}{streak_type} | "
                 f"Daily P&L: ${daily_pnl:+.2f}")
    atr_str = f"{atr_baseline:.4f}%" if atr_baseline else "N/A"
    lines.append(f"ATR baseline (24h): {atr_str} | Ghost trades: {ghost_count}")
    lines.append(f"Bias: {bias_str}")
    if agent_info:
        lines.append(f"Agents: {' | '.join(agent_info)}")
    else:
        lines.append("Agents: no portfolio data yet")
    lines.append(f"Top lesson: {top_lesson}")
    lines.append("")

    return "\n".join(lines)


def main():
    content = generate()
    if "--stdout" in sys.argv:
        print(content)
    else:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Written to {OUTPUT_PATH}")
        print(content)


if __name__ == "__main__":
    main()
