#!/usr/bin/env python3
"""
Walk-Forward Validation for XGBoost Judge.

Simulates how the model would have performed if retrained on a rolling
7-day window and tested on the next 1-day window. Prevents temporal
data leakage that inflates random-split accuracy.

Usage:
    python -m ml.train_walkforward
    python -m ml.train_walkforward --csv data/training_data_deepseek.csv
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

FEATURES = [
    "confidence", "rsi", "bb_pct", "volatility",
    "momentum", "atr_pct", "hour_utc", "price_change_pct",
    "trend_alignment", "funding_rate", "order_book_imbalance",
    "liq_buy_vol_5m", "liq_sell_vol_5m", "reversal_alert",
    "smc_bos_last", "smc_choch_last", "smc_in_kill_zone",
    "smc_ob_distance_pct", "consensus_agreement",
    "bid_ask_spread", "sniper_time_to_bet_sec",
    "volatility_regime", "fear_greed",
    # ── Temporal velocity features ──
    "rsi_velocity", "rsi_acceleration", "momentum_velocity", "cvd_velocity",
    # ── Liquidation cluster proximity ──
    "liq_cluster_distance_atr", "liq_cluster_magnitude_pct",
    "liq_cluster_side_enc",
]

TARGET = "bet_won"


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Load CSV, encode categoricals, sort by time."""
    df = pd.read_csv(csv_path)
    print(f"CSV cargado: {len(df)} filas")

    if TARGET not in df.columns:
        print(f"ERROR: Columna '{TARGET}' no encontrada.")
        sys.exit(1)

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    # Parse timestamp for temporal splitting
    time_col = None
    for col in ["open_time", "prediction_time", "created_at"]:
        if col in df.columns:
            time_col = col
            break
    if time_col is None:
        print("ERROR: No se encontro columna temporal (open_time/prediction_time).")
        sys.exit(1)

    df["_dt"] = pd.to_datetime(df[time_col])
    df = df.sort_values("_dt").reset_index(drop=True)

    # Encode volatility_regime
    if "volatility_regime" in df.columns:
        _vol_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2}
        df["volatility_regime"] = df["volatility_regime"].map(_vol_map).fillna(1)

    # Encode liq_cluster_side (ABOVE=+1, BELOW=-1, AT_SPOT/NEUTRAL=0)
    if "liq_cluster_side" in df.columns and "liq_cluster_side_enc" not in df.columns:
        _side_map = {"ABOVE": 1, "BELOW": -1, "AT_SPOT": 0, "NEUTRAL": 0}
        df["liq_cluster_side_enc"] = (
            df["liq_cluster_side"].astype(str).str.upper().map(_side_map).fillna(0)
        )

    # Select available features
    available = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  Features ignoradas (no en CSV): {missing}")
    if not available:
        print("ERROR: Ninguna feature disponible.")
        sys.exit(1)

    print(f"  Rango temporal: {df['_dt'].min()} -> {df['_dt'].max()}")
    print(f"  Features disponibles: {len(available)}/{len(FEATURES)}")
    print(f"  Filas con apuesta: {len(df)}")

    return df, available


def walk_forward(df: pd.DataFrame, features: list[str],
                 train_days: int = 7, test_days: int = 1):
    """Run walk-forward validation with rolling windows.

    Returns (fold_results, pooled_y_true, pooled_y_pred, pooled_y_prob).
    Pooled arrays concatenate OOS test predictions across every fold —
    fed into statistical_validation() for permutation + bootstrap.
    """
    results = []
    pooled_y_true: list = []
    pooled_y_pred: list = []
    pooled_y_prob: list = []
    start_date = df["_dt"].min()
    end_date = df["_dt"].max()
    current = start_date + pd.Timedelta(days=train_days)

    fold = 0
    while current + pd.Timedelta(days=test_days) <= end_date:
        train_start = current - pd.Timedelta(days=train_days)
        train_end = current
        test_end = current + pd.Timedelta(days=test_days)

        train_mask = (df["_dt"] >= train_start) & (df["_dt"] < train_end)
        test_mask = (df["_dt"] >= train_end) & (df["_dt"] < test_end)

        df_train = df[train_mask]
        df_test = df[test_mask]

        if len(df_train) < 10 or len(df_test) < 3:
            current += pd.Timedelta(days=test_days)
            continue

        fold += 1
        X_train = df_train[features].fillna(0)
        y_train = df_train[TARGET]
        X_test = df_test[features].fillna(0)
        y_test = df_test[TARGET]

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale = neg / pos if pos > 0 else 1.0

        model = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale, eval_metric="logloss",
            random_state=42, use_label_encoder=False, verbosity=0,
        )
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        result = {
            "fold": fold,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "test_date": train_end.strftime("%Y-%m-%d"),
            "n_train": len(df_train),
            "n_test": len(df_test),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "avg_prob": round(float(y_prob.mean()), 4),
            "win_rate_actual": round(float(y_test.mean()), 4),
        }
        results.append(result)
        pooled_y_true.extend(y_test.tolist())
        pooled_y_pred.extend(y_pred.tolist())
        pooled_y_prob.extend(y_prob.tolist())

        tag = "OK" if acc >= 0.55 else "WEAK"
        print(f"  Fold {fold:3d} | {result['test_date']} | "
              f"train={result['n_train']:4d} test={result['n_test']:3d} | "
              f"acc={acc:.2%} prec={prec:.2%} rec={rec:.2%} [{tag}]")

        current += pd.Timedelta(days=test_days)

    return results, np.array(pooled_y_true), np.array(pooled_y_pred), np.array(pooled_y_prob)


def statistical_validation(y_true: np.ndarray, y_pred: np.ndarray,
                           n_permutations: int = 2000,
                           n_bootstrap: int = 5000,
                           seed: int = 42) -> dict:
    """Monte Carlo permutation test + Bootstrap CI on pooled OOS predictions.

    Permutation test: H0 = predictions have no edge. Shuffle y_true, recompute
    accuracy. p-value = P(perm_acc >= observed_acc).

    Bootstrap: resample (y_true, y_pred) pairs with replacement. 95% CI on
    accuracy and on winrate-above-coinflip (edge).

    Returns dict with observed/pvalue/ci fields. Also derives a Sharpe-like
    ratio on unit bets (win=+1, loss=-1) with bootstrap CI.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {}

    observed_acc = float((y_true == y_pred).mean())
    # Unit-bet returns: +1 on win, -1 on loss. Assumes 1:1 odds; true Polymarket
    # payoff varies per bet, but this is a distribution-free edge test.
    returns = np.where(y_true == y_pred, 1.0, -1.0)
    observed_sharpe = float(returns.mean() / returns.std(ddof=1)) if returns.std(ddof=1) > 0 else 0.0
    observed_edge = observed_acc - 0.5

    # ── Permutation test ──
    perm_hits = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(y_true)
        if float((shuffled == y_pred).mean()) >= observed_acc:
            perm_hits += 1
    pvalue = (perm_hits + 1) / (n_permutations + 1)

    # ── Bootstrap CI ──
    boot_accs = np.empty(n_bootstrap)
    boot_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        boot_accs[i] = (yt == yp).mean()
        r = np.where(yt == yp, 1.0, -1.0)
        s = r.std(ddof=1)
        boot_sharpes[i] = r.mean() / s if s > 0 else 0.0

    acc_ci = (float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5)))
    sharpe_ci = (float(np.percentile(boot_sharpes, 2.5)), float(np.percentile(boot_sharpes, 97.5)))
    edge_ci = (acc_ci[0] - 0.5, acc_ci[1] - 0.5)

    return {
        "n_samples": n,
        "observed_accuracy": observed_acc,
        "observed_edge": observed_edge,
        "observed_sharpe": observed_sharpe,
        "permutation_pvalue": pvalue,
        "permutation_n": n_permutations,
        "accuracy_ci95": acc_ci,
        "edge_ci95": edge_ci,
        "sharpe_ci95": sharpe_ci,
        "bootstrap_n": n_bootstrap,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Validation para XGBoost Judge"
    )
    parser.add_argument(
        "--csv", default=str(_PROJECT_ROOT / "data" / "training_data_deepseek.csv"),
        help="CSV de entrenamiento",
    )
    parser.add_argument(
        "--train-days", type=int, default=7,
        help="Dias de ventana de entrenamiento (default: 7)",
    )
    parser.add_argument(
        "--test-days", type=int, default=1,
        help="Dias de ventana de test (default: 1)",
    )
    parser.add_argument(
        "--permutations", type=int, default=2000,
        help="Iteraciones permutation test (default: 2000)",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=5000,
        help="Iteraciones bootstrap CI (default: 5000)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV no encontrado: {csv_path}")
        print("  Ejecuta primero: python -m ml.generate_dataset")
        sys.exit(1)

    df, features = load_and_prepare(str(csv_path))

    total_days = (df["_dt"].max() - df["_dt"].min()).days
    if total_days < args.train_days + args.test_days:
        print(f"\nERROR: Solo {total_days} dias de datos. "
              f"Necesitas al menos {args.train_days + args.test_days}.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD: {args.train_days}d train → {args.test_days}d test")
    print(f"{'='*60}\n")

    results, y_true_pool, y_pred_pool, y_prob_pool = walk_forward(
        df, features, args.train_days, args.test_days
    )

    if not results:
        print("\nNo se generaron folds. Datos insuficientes.")
        sys.exit(1)

    # Summary
    accs = [r["accuracy"] for r in results]
    precs = [r["precision"] for r in results]
    recs = [r["recall"] for r in results]
    good_folds = sum(1 for a in accs if a >= 0.55)

    print(f"\n{'='*60}")
    print("RESUMEN WALK-FORWARD")
    print(f"{'='*60}")
    print(f"  Folds totales:       {len(results)}")
    print(f"  Accuracy promedio:   {np.mean(accs):.2%} "
          f"(std={np.std(accs):.2%})")
    print(f"  Precision promedio:  {np.mean(precs):.2%}")
    print(f"  Recall promedio:     {np.mean(recs):.2%}")
    print(f"  Folds >= 55% acc:    {good_folds}/{len(results)} "
          f"({good_folds/len(results)*100:.0f}%)")
    print(f"  Mejor fold:          {max(accs):.2%}")
    print(f"  Peor fold:           {min(accs):.2%}")

    if np.mean(accs) >= 0.55:
        print("\n  VEREDICTO: El modelo GENERALIZA bien temporalmente.")
    elif np.mean(accs) >= 0.50:
        print("\n  VEREDICTO: Marginal — considerar mas features o datos.")
    else:
        print("\n  VEREDICTO: El modelo NO generaliza — random-split puede estar inflado.")

    # Save results CSV
    out_path = _PROJECT_ROOT / "data" / "walkforward_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n  Resultados guardados en: {out_path}")

    # ── Statistical rigor: permutation + bootstrap on pooled OOS ──
    print(f"\n{'='*60}")
    print("VALIDACION ESTADISTICA (pooled OOS)")
    print(f"{'='*60}")
    stats = statistical_validation(y_true_pool, y_pred_pool,
                                   n_permutations=args.permutations,
                                   n_bootstrap=args.bootstrap)
    if stats:
        print(f"  Muestras pooled OOS:   {stats['n_samples']}")
        print(f"  Accuracy observada:    {stats['observed_accuracy']:.2%}")
        print(f"  Edge vs coinflip:      {stats['observed_edge']*100:+.2f}pp")
        print(f"  Sharpe (unit-bet):     {stats['observed_sharpe']:.3f}")
        print(f"  Permutation p-value:   {stats['permutation_pvalue']:.4f} "
              f"(n={stats['permutation_n']})")
        lo, hi = stats["accuracy_ci95"]
        print(f"  Accuracy 95% CI:       [{lo:.2%}, {hi:.2%}]")
        lo, hi = stats["edge_ci95"]
        print(f"  Edge 95% CI:           [{lo*100:+.2f}pp, {hi*100:+.2f}pp]")
        lo, hi = stats["sharpe_ci95"]
        print(f"  Sharpe 95% CI:         [{lo:.3f}, {hi:.3f}]")

        p = stats["permutation_pvalue"]
        edge_lo = stats["edge_ci95"][0]
        if p < 0.01 and edge_lo > 0:
            verdict = "EDGE REAL — p<0.01 y CI inferior positivo"
        elif p < 0.05 and edge_lo > 0:
            verdict = "EDGE PROBABLE — p<0.05, CI inferior positivo"
        elif p < 0.05:
            verdict = "AMBIGUO — p significativo pero CI cruza cero"
        else:
            verdict = "SIN EDGE — indistinguible de azar"
        print(f"\n  VEREDICTO ESTADISTICO: {verdict}")

        stats_out = _PROJECT_ROOT / "data" / "walkforward_stats.json"
        import json
        with open(stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats guardadas en:    {stats_out}")


if __name__ == "__main__":
    main()
