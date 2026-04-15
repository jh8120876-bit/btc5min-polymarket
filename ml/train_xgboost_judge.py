#!/usr/bin/env python3
"""
Train XGBoost "Juez Supremo" — filters DeepSeek AI predictions.

Reads training CSV exported by ml.generate_dataset, trains an
XGBClassifier to predict whether a bet will win (bet_won=1) or lose (0),
and exports the model to models/ml_judge_model.pkl.

Usage:
    python -m ml.train_xgboost_judge
    python -m ml.train_xgboost_judge --csv data/training_data_deepseek.csv
    python -m ml.train_xgboost_judge --output models/ml_judge_model.pkl
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# Features the judge uses to decide if a prediction is trustworthy
# Must stay in sync with engine.py _JUDGE_FEATURES
FEATURES = [
    "confidence",
    "rsi",
    "bb_pct",
    "volatility",
    "momentum",
    "atr_pct",
    "hour_utc",
    "price_change_pct",
    "trend_alignment",
    "funding_rate",
    "order_book_imbalance",
    "liq_buy_vol_5m",
    "liq_sell_vol_5m",
    "reversal_alert",
    "smc_bos_last",
    "smc_choch_last",
    "smc_in_kill_zone",
    "smc_ob_distance_pct",
    "bid_ask_spread",
    "liq_cluster_distance_atr",
    "liq_cluster_magnitude_pct",
    "liq_cluster_side_enc",
]

TARGET = "correct"


def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, select features + target, handle NaNs."""
    df = pd.read_csv(csv_path)
    print(f"CSV cargado: {len(df)} filas, {len(df.columns)} columnas")

    # Verify target exists
    if TARGET not in df.columns:
        print(f"ERROR: Columna '{TARGET}' no encontrada en el CSV.")
        print(f"  Columnas disponibles: {list(df.columns)}")
        sys.exit(1)

    # Drop rows where target is NaN (no bet was placed)
    before = len(df)
    df = df.dropna(subset=[TARGET])
    dropped = before - len(df)
    if dropped:
        print(f"  Eliminadas {dropped} filas sin {TARGET} (sin apuesta)")
    print(f"  Filas con apuesta: {len(df)}")

    # Derive encoded liq_cluster_side (ABOVE=+1, BELOW=-1, AT_SPOT/NEUTRAL=0).
    if "liq_cluster_side" in df.columns and "liq_cluster_side_enc" not in df.columns:
        _side_map = {"ABOVE": 1, "BELOW": -1, "AT_SPOT": 0, "NEUTRAL": 0}
        df["liq_cluster_side_enc"] = (
            df["liq_cluster_side"].astype(str).str.upper().map(_side_map).fillna(0)
        )

    # Check which features are available
    available = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  AVISO: Features no encontradas (ignoradas): {missing}")
    if not available:
        print("ERROR: Ninguna feature disponible en el CSV.")
        sys.exit(1)

    X = df[available].copy()
    y = df[TARGET].astype(int)

    # Encode categorical: volatility_regime -> numeric (LOW=0, NORMAL=1, HIGH=2)
    if "volatility_regime" in X.columns:
        _vol_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2}
        X["volatility_regime"] = X["volatility_regime"].map(_vol_map).fillna(1)

    # Impute remaining NaNs with median
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"  NaN encontrados en: {nan_cols} -> imputando con mediana")
        X = X.fillna(X.median())

    # Target distribution
    wins = y.sum()
    losses = len(y) - wins
    print(f"\n  Target: {wins} wins ({wins/len(y)*100:.1f}%) / "
          f"{losses} losses ({losses/len(y)*100:.1f}%)")

    return X, y


def train(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> XGBClassifier:
    """Train XGBClassifier with TimeSeriesSplit (temporal, no data leakage)."""

    # ── Temporal split: TimeSeriesSplit preserves chronological order ──
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_accuracies = []

    print(f"\nTimeSeriesSplit: {n_splits} folds (temporal, sin data leakage)")

    # Use the LAST fold as the final train/test for model export
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold_acc = None

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale = neg / pos if pos > 0 else 1.0

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            eval_metric="logloss",
            random_state=seed,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        fold_acc = (y_pred == y_test).mean()
        fold_accuracies.append(fold_acc)
        print(f"  Fold {fold}: train={len(X_train)} test={len(X_test)} "
              f"acc={fold_acc:.3f}")

    print(f"\n  Media temporal: {np.mean(fold_accuracies):.3f} "
          f"(std={np.std(fold_accuracies):.3f})")
    if np.mean(fold_accuracies) < 0.50:
        print("  AVISO: Accuracy temporal < 50% — modelo NO generaliza")

    # ── Final evaluation on last fold ──
    print(f"\nFinal split (fold {n_splits}): "
          f"{len(X_train)} train / {len(X_test)} test")

    y_pred = model.predict(X_test)

    print("\n" + "=" * 50)
    print("MATRIZ DE CONFUSION (ultimo fold temporal)")
    print("=" * 50)
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Loss (0)", "Win (1)"]
    print(f"{'':>12} {'Pred Loss':>10} {'Pred Win':>10}")
    for i, row_label in enumerate(labels):
        print(f"{row_label:>12} {cm[i][0]:>10} {cm[i][1]:>10}")

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=labels))

    # ── Retrain final model on ALL data for deployment ──
    print("Retraining final model on ALL data for deployment...")
    neg_all = (y == 0).sum()
    pos_all = (y == 1).sum()
    scale_all = neg_all / pos_all if pos_all > 0 else 1.0

    final_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_all,
        eval_metric="logloss",
        random_state=seed,
    )
    final_model.fit(X, y, verbose=False)

    # --- Feature Importance (from final model) ---
    print("=" * 50)
    print("IMPORTANCIA DE VARIABLES (Feature Importance)")
    print("=" * 50)
    importances = final_model.feature_importances_
    feat_imp = sorted(
        zip(X.columns, importances), key=lambda x: x[1], reverse=True
    )
    for feat, imp in feat_imp:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<20} {imp:.4f}  {bar}")

    return final_model


def main():
    parser = argparse.ArgumentParser(
        description="Entrena el Juez Supremo XGBoost para filtrar predicciones"
    )
    parser.add_argument(
        "--csv", default=str(_PROJECT_ROOT / "data" / "training_data_deepseek.csv"),
        help="CSV de entrenamiento",
    )
    parser.add_argument(
        "--output", "-o", default=str(_PROJECT_ROOT / "models" / "ml_judge_model.pkl"),
        help="Ruta del modelo exportado",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: No se encontro el CSV en '{csv_path}'")
        print("  Primero ejecuta: python -m ml.generate_dataset")
        sys.exit(1)

    X, y = load_data(str(csv_path))

    if len(y) < 20:
        print(f"\nAVISO: Solo {len(y)} muestras. Resultados poco fiables.")
        print("  Recomendado: al menos 200+ ventanas con apuesta.")

    model = train(X, y, seed=args.seed)

    # Save
    output_path = Path(args.output)
    joblib.dump(model, output_path)
    print(f"\nModelo guardado en: {output_path}")
    print(f"  Features usadas: {list(X.columns)}")
    print(f"  Para cargar: model = joblib.load('{output_path}')")


if __name__ == "__main__":
    main()
