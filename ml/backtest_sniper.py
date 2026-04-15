#!/usr/bin/env python3
"""
Backtest the Sniper Consensus strategy on historical data.

Simulates the sniper consensus logic (2 consecutive same-direction
evaluations with rising/stable confidence -> fire) on historical
predictions to measure what-if performance.

Usage:
    python -m ml.backtest_sniper
    python -m ml.backtest_sniper --csv data/training_data_deepseek.csv
    python -m ml.backtest_sniper --min-confidence 60 --consensus-threshold 2
"""

import argparse
import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    """Load training CSV with required columns."""
    df = pd.read_csv(csv_path)
    print(f"CSV cargado: {len(df)} filas")

    required = ["prediction", "confidence", "actual_change_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Columnas faltantes: {missing}")
        sys.exit(1)

    df = df.dropna(subset=required)
    print(f"  Filas validas: {len(df)}")

    # Parse time if available
    for col in ["open_time", "prediction_time"]:
        if col in df.columns:
            df["_dt"] = pd.to_datetime(df[col])
            df = df.sort_values("_dt").reset_index(drop=True)
            break

    return df


def simulate_sniper(df: pd.DataFrame,
                    min_confidence: int = 55,
                    consensus_n: int = 2,
                    time_stop_windows: int = 3,
                    bet_amount: float = 2.0,
                    odds_cents: int = 50) -> dict:
    """
    Simulate sniper consensus on sequential predictions.

    Logic:
    - Accumulate consecutive same-direction predictions
    - When consensus_n consecutive same-direction + confidence >= min_confidence -> BET
    - If direction flips or time_stop_windows reached without consensus -> SKIP
    """
    balance = 100.0
    peak = balance
    trades = []
    skipped = 0
    flips = 0

    i = 0
    while i < len(df):
        # Start a new "window group" of up to time_stop_windows
        window_evals = []
        fired = False

        for j in range(time_stop_windows):
            idx = i + j
            if idx >= len(df):
                break
            row = df.iloc[idx]
            pred = row["prediction"]
            conf = row["confidence"]
            actual = row["actual_change_pct"]

            window_evals.append({
                "idx": idx, "prediction": pred,
                "confidence": conf, "actual": actual,
            })

            # Check consensus: last N evals same direction + confidence ok
            if len(window_evals) >= consensus_n:
                last_n = window_evals[-consensus_n:]
                same_dir = all(e["prediction"] == last_n[0]["prediction"]
                               for e in last_n)
                conf_ok = all(e["confidence"] >= min_confidence for e in last_n)
                # Confidence rising or stable
                conf_stable = all(
                    last_n[k]["confidence"] >= last_n[k - 1]["confidence"] - 2
                    for k in range(1, len(last_n))
                )

                if same_dir and conf_ok and conf_stable:
                    # FIRE
                    direction = last_n[-1]["prediction"]
                    avg_conf = np.mean([e["confidence"] for e in last_n])

                    # Determine win: prediction matches actual direction
                    won = (direction == "UP" and actual > 0) or \
                          (direction == "DOWN" and actual < 0)

                    # P&L simulation (simplified Polymarket)
                    price = odds_cents / 100.0
                    tokens = bet_amount / price
                    if won:
                        pnl = (tokens * 1.0 - bet_amount) * 0.98  # 2% fee
                    else:
                        pnl = -bet_amount

                    balance += pnl
                    peak = max(peak, balance)

                    trades.append({
                        "idx": idx,
                        "direction": direction,
                        "confidence": avg_conf,
                        "actual_pct": actual,
                        "won": won,
                        "pnl": round(pnl, 4),
                        "balance": round(balance, 2),
                        "evals_needed": len(window_evals),
                    })
                    fired = True
                    break

            # Check for flip (direction change)
            if len(window_evals) >= 2:
                if window_evals[-1]["prediction"] != window_evals[-2]["prediction"]:
                    flips += 1

        if not fired:
            skipped += 1

        # Advance past this group
        i += len(window_evals) if window_evals else 1

    return {
        "trades": trades,
        "skipped": skipped,
        "flips": flips,
        "final_balance": round(balance, 2),
        "peak_balance": round(peak, 2),
    }


def print_report(result: dict, bet_amount: float):
    """Print comprehensive backtest report."""
    trades = result["trades"]
    if not trades:
        print("\nNo se ejecutaron trades. Ajusta parametros.")
        return

    wins = [t for t in trades if t["won"]]
    losses = [t for t in trades if not t["won"]]
    total = len(trades)
    wr = len(wins) / total * 100

    total_pnl = sum(t["pnl"] for t in trades)
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

    # Max drawdown
    balances = [t["balance"] for t in trades]
    peak = 100.0
    max_dd = 0
    for b in balances:
        peak = max(peak, b)
        dd = (peak - b) / peak * 100
        max_dd = max(max_dd, dd)

    # Profit factor
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # Avg evals needed
    avg_evals = np.mean([t["evals_needed"] for t in trades])

    print(f"\n{'='*60}")
    print("BACKTEST SNIPER CONSENSUS")
    print(f"{'='*60}")
    print(f"  Trades ejecutados:   {total}")
    print(f"  Ventanas saltadas:   {result['skipped']}")
    print(f"  Flips detectados:    {result['flips']}")
    print(f"  Evals promedio/trade: {avg_evals:.1f}")

    print(f"\n  Win Rate:           {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  PnL Total:          {'+'if total_pnl>=0 else ''}{total_pnl:.2f}")
    print(f"  Balance Final:      ${result['final_balance']:.2f}")
    print(f"  Balance Pico:       ${result['peak_balance']:.2f}")

    print(f"\n  Avg Win:            +{avg_win:.4f}")
    print(f"  Avg Loss:           {avg_loss:.4f}")
    print(f"  Profit Factor:      {pf:.2f}")
    print(f"  Max Drawdown:       {max_dd:.1f}%")

    # Risk-adjusted
    if total >= 10:
        returns = [t["pnl"] / bet_amount for t in trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(288) if np.std(returns) > 0 else 0
        print(f"  Sharpe (annualized): {sharpe:.2f}")

    print(f"\n{'='*60}")
    print("VEREDICTO")
    print(f"{'='*60}")
    if wr >= 55 and pf >= 1.2 and max_dd < 20:
        print("  EXCELENTE — Sniper consensus agrega valor significativo")
    elif wr >= 50 and pf >= 1.0:
        print("  VIABLE — Sniper consensus es marginalmente beneficioso")
    else:
        print("  DEBIL — Considerar ajustar parametros o desactivar sniper")

    # Save detailed results
    out_path = _PROJECT_ROOT / "data" / "backtest_sniper_results.csv"
    pd.DataFrame(trades).to_csv(out_path, index=False)
    print(f"\n  Detalle guardado en: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtesting del Sniper Consensus sobre datos historicos"
    )
    parser.add_argument("--csv", default=str(_PROJECT_ROOT / "data" / "training_data_deepseek.csv"))
    parser.add_argument("--min-confidence", type=int, default=55,
                        help="Confianza minima para apostar (default: 55)")
    parser.add_argument("--consensus-threshold", type=int, default=2,
                        help="Evals consecutivas necesarias (default: 2)")
    parser.add_argument("--time-stop", type=int, default=3,
                        help="Max evals antes de skip (default: 3)")
    parser.add_argument("--bet-amount", type=float, default=2.0,
                        help="Monto por apuesta en USD (default: 2.0)")
    parser.add_argument("--odds", type=int, default=50,
                        help="Odds en centavos (default: 50)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV no encontrado: {csv_path}")
        sys.exit(1)

    df = load_data(str(csv_path))

    print(f"\nParametros:")
    print(f"  min_confidence: {args.min_confidence}%")
    print(f"  consensus_threshold: {args.consensus_threshold}")
    print(f"  time_stop: {args.time_stop} evals")
    print(f"  bet_amount: ${args.bet_amount}")
    print(f"  odds: {args.odds}c")

    result = simulate_sniper(
        df,
        min_confidence=args.min_confidence,
        consensus_n=args.consensus_threshold,
        time_stop_windows=args.time_stop,
        bet_amount=args.bet_amount,
        odds_cents=args.odds,
    )

    print_report(result, args.bet_amount)


if __name__ == "__main__":
    main()
