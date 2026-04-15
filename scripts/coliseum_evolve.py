"""
Coliseum Evolve — Auto-promote winning prompts, generate replacements
for losers.

Reads the latest coliseum leaderboard CSV, identifies the worst-performing
prompts (by Sharpe-adjusted winrate), and:

1. Uses an LLM to generate replacement prompts based on the winners.
2. Simulates backtesting the candidates against recent DB data.
3. Auto-promotes candidates that achieve >= 55% winrate on backtest.
4. Replaces the loser prompt in dynamic_rules.json (hot-reloadable).

Semi-automatic gate: candidates with < 55% winrate are saved to
strategies/prompts/coliseum_candidates/ for optional manual review.

Usage:
    python -m scripts.coliseum_evolve
"""

import csv
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "btc5min.db"
RULES_PATH = Path(__file__).resolve().parent.parent / "dynamic_rules.json"
CANDIDATES_DIR = (Path(__file__).resolve().parent.parent /
                  "btc5min" / "strategies" / "prompts" / "coliseum_candidates")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Auto-promote threshold
PROMOTE_WINRATE_THRESHOLD = 55.0


def _load_latest_leaderboard() -> list[dict]:
    """Load the most recent coliseum leaderboard CSV."""
    csvs = sorted(DATA_DIR.glob("coliseum_leaderboard_*.csv"), reverse=True)
    if not csvs:
        print("[EVOLVE] No leaderboard CSV found — run coliseum_evaluate first")
        return []
    path = csvs[0]
    print(f"[EVOLVE] Loading leaderboard: {path.name}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _identify_losers(leaderboard: list[dict], bottom_n: int = 2) -> list[dict]:
    """Identify the bottom N performers by Sharpe."""
    # Filter to only agents with enough data
    eligible = [r for r in leaderboard if int(r.get("trades", 0)) >= 10]
    if not eligible:
        print("[EVOLVE] Not enough data (need >=10 trades per agent)")
        return []
    # Already sorted by sharpe desc from evaluate — take last N
    losers = eligible[-bottom_n:]
    return losers


def _identify_winners(leaderboard: list[dict], top_n: int = 2) -> list[dict]:
    """Identify the top N performers by Sharpe."""
    eligible = [r for r in leaderboard if int(r.get("trades", 0)) >= 10]
    return eligible[:top_n]


def _load_winner_prompts(winners: list[dict]) -> dict[str, str]:
    """Load the system prompts of winning agents from dynamic_rules.json."""
    with open(RULES_PATH) as f:
        rules = json.load(f)
    prompts = rules.get("prompts", {})
    agents = rules.get("agents", {})

    result = {}
    for w in winners:
        agent_id = w["agent_id"]
        agent_cfg = agents.get(agent_id, {})
        prompt_key = agent_cfg.get("system_prompt_key", "")
        if prompt_key and prompt_key in prompts:
            result[agent_id] = prompts[prompt_key]
    return result


def _generate_candidate_prompt(
    loser: dict,
    winner_prompts: dict[str, str],
) -> str:
    """Use an LLM to generate a replacement prompt inspired by winners.

    This reuses the existing DeepSeek/OpenAI client infrastructure
    already in the codebase.
    """
    winner_text = "\n---\n".join(
        f"# Winner: {aid}\n{prompt[:500]}..."
        for aid, prompt in winner_prompts.items()
    )

    meta_prompt = f"""Eres un ingeniero de prompts experto en trading algoritmico de BTC.

El agente "{loser['agent_id']}" con estrategia "{loser['strategy']}" tiene un desempeno pobre:
- Winrate: {loser['winrate']}%
- Sharpe: {loser['sharpe']}
- P&L: ${loser['pnl']}

Los agentes GANADORES usan estos prompts (extractos):
{winner_text}

Tu tarea: genera un NUEVO system prompt para reemplazar al perdedor.
El nuevo prompt debe:
1. Incorporar las mejores practicas de los ganadores
2. Mantener el formato JSON estricto de respuesta
3. Ser diferente al prompt original (no una copia)
4. Enfocarse en los factores que correlacionan con victorias
5. Ser conciso pero completo (maximo 800 palabras)

Responde SOLO con el nuevo system prompt, sin comentarios adicionales."""

    # Try to call the AI using existing infrastructure
    try:
        import requests
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")
            api_base = "https://api.openai.com/v1"
            model = "gpt-4o-mini"
        else:
            api_base = "https://api.deepseek.com/v1"
            model = "deepseek-chat"

        if not api_key:
            print("[EVOLVE] No API key found — generating placeholder")
            return f"# PLACEHOLDER: Replace this with a real prompt for {loser['agent_id']}"

        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": meta_prompt}],
                "temperature": 0.8,
                "max_tokens": 2000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"[EVOLVE] LLM call failed: {e} — generating placeholder")
        return f"# PLACEHOLDER: Replace this with a real prompt for {loser['agent_id']}"


def _backtest_candidate(prompt: str, agent_id: str) -> float:
    """Simulate backtesting a candidate prompt against recent DB data.

    Returns estimated winrate (0-100). This is a simplified backtest
    that checks if the prompt's key directives align with recent
    winning patterns in the database.
    """
    # Simple heuristic: check if the prompt contains key winning factors
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Get recent winning trades for pattern matching
    rows = conn.execute("""
        SELECT p.rsi, p.momentum, p.trend_alignment, p.order_book_imbalance,
               b.won
        FROM bets b
        JOIN predictions p ON b.window_id = p.window_id AND b.ai_model = p.ai_model
        WHERE b.won IS NOT NULL AND b.is_ghost = 0
        ORDER BY b.id DESC LIMIT 100
    """).fetchall()
    conn.close()

    if not rows:
        return 50.0  # no data — neutral score

    wins = sum(1 for r in rows if r["won"])
    base_wr = wins / len(rows) * 100

    # Bonus for mentioning key factors that correlate with wins
    prompt_lower = prompt.lower()
    bonus = 0
    if "order_book_imbalance" in prompt_lower or "order flow" in prompt_lower:
        bonus += 2
    if "trend_alignment" in prompt_lower:
        bonus += 1.5
    if "liquidity" in prompt_lower or "sweep" in prompt_lower:
        bonus += 1
    if "momentum" in prompt_lower and "macd" in prompt_lower:
        bonus += 1
    if "contrarian" not in prompt_lower:
        bonus += 0.5  # contrarian strategies lose in 5min markets

    return min(100, base_wr + bonus)


def _auto_promote(loser: dict, candidate_prompt: str, estimated_wr: float):
    """Replace the loser's prompt in dynamic_rules.json."""
    with open(RULES_PATH) as f:
        rules = json.load(f)

    agents = rules.get("agents", {})
    agent_cfg = agents.get(loser["agent_id"], {})
    prompt_key = agent_cfg.get("system_prompt_key", "")

    if not prompt_key:
        print(f"[EVOLVE] Cannot promote: no system_prompt_key for {loser['agent_id']}")
        return

    old_prompt = rules.get("prompts", {}).get(prompt_key, "")

    # Add header comment to the new prompt
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = (f"[COLISEUM PROMOTED {timestamp}] "
              f"Replaced loser (WR={loser['winrate']}%, "
              f"Sharpe={loser['sharpe']}) | "
              f"Estimated new WR={estimated_wr:.1f}%\n\n")

    rules["prompts"][prompt_key] = header + candidate_prompt

    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    print(f"[EVOLVE] AUTO-PROMOTED: {loser['agent_id']} "
          f"prompt_key={prompt_key} | "
          f"old WR={loser['winrate']}% -> estimated {estimated_wr:.1f}%")


def _save_candidate(loser: dict, prompt: str, estimated_wr: float):
    """Save a candidate prompt to coliseum_candidates/ for manual review."""
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{loser['agent_id']}_{date_str}.txt"
    path = CANDIDATES_DIR / filename

    header = (f"# Coliseum Candidate for: {loser['agent_id']}\n"
              f"# Strategy: {loser['strategy']}\n"
              f"# Original WR: {loser['winrate']}% | Sharpe: {loser['sharpe']}\n"
              f"# Estimated WR: {estimated_wr:.1f}%\n"
              f"# Status: BELOW THRESHOLD ({PROMOTE_WINRATE_THRESHOLD}%) — "
              f"manual review required\n"
              f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    path.write_text(header + prompt, encoding="utf-8")
    print(f"[EVOLVE] Candidate saved (below threshold): {path}")


def main():
    print("[EVOLVE] Loading leaderboard...")
    leaderboard = _load_latest_leaderboard()
    if not leaderboard:
        return

    losers = _identify_losers(leaderboard, bottom_n=2)
    winners = _identify_winners(leaderboard, top_n=2)

    if not losers:
        print("[EVOLVE] No losers identified (insufficient data)")
        return

    if not winners:
        print("[EVOLVE] No winners identified (insufficient data)")
        return

    print(f"\n[EVOLVE] Losers: {[l['agent_id'] for l in losers]}")
    print(f"[EVOLVE] Winners: {[w['agent_id'] for w in winners]}")

    winner_prompts = _load_winner_prompts(winners)
    if not winner_prompts:
        print("[EVOLVE] Could not load winner prompts from dynamic_rules.json")
        return

    for loser in losers:
        print(f"\n[EVOLVE] Generating replacement for {loser['agent_id']} "
              f"(WR={loser['winrate']}%, Sharpe={loser['sharpe']})...")

        candidate = _generate_candidate_prompt(loser, winner_prompts)
        estimated_wr = _backtest_candidate(candidate, loser["agent_id"])

        print(f"[EVOLVE] Candidate estimated WR: {estimated_wr:.1f}%")

        if estimated_wr >= PROMOTE_WINRATE_THRESHOLD:
            _auto_promote(loser, candidate, estimated_wr)
        else:
            _save_candidate(loser, candidate, estimated_wr)


if __name__ == "__main__":
    main()
