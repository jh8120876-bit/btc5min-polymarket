#!/usr/bin/env python3
"""
Train regression model to predict actual_change_pct for dynamic bet sizing.

Instead of binary win/lose, this predicts the magnitude of price movement,
enabling Kelly-optimal position sizing based on expected return.

Usage:
    python -m ml.train_regressor
    python -m ml.train_regressor --csv data/training_data_deepseek.csv
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

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
]

TARGET = "actual_change_pct"


def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, prepare features and regression target."""
    df = pd.read_csv(csv_path)
    print(f"CSV cargado: {len(df)} filas, {len(df.columns)} columnas")

    if TARGET not in df.columns:
        print(f"ERROR: Columna '{TARGET}' no encontrada.")
        print(f"  Columnas disponibles: {list(df.columns)}")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=[TARGET])
    dropped = before - len(df)
    if dropped:
        print(f"  Eliminadas {dropped} filas sin {TARGET}")
    print(f"  Filas validas: {len(df)}")

    # Encode volatility_regime
    if "volatility_regime" in df.columns:
        _vol_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2}
        df["volatility_regime"] = df["volatility_regime"].map(_vol_map).fillna(1)

    available = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  Features ignoradas: {missing}")
    if not available:
        print("ERROR: Ninguna feature disponible.")
        sys.exit(1)

    X = df[available].copy()
    y = df[TARGET].astype(float)

    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"  NaN en: {nan_cols} -> imputando con mediana")
        X = X.fillna(X.median())

    # Target stats
    print(f"\n  Target ({TARGET}):")
    print(f"    Media:  {y.mean():.4f}%")
    print(f"    Std:    {y.std():.4f}%")
    print(f"    Min:    {y.min():.4f}%  Max: {y.max():.4f}%")

    return X, y


def train(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> XGBRegressor:
    """Train XGBRegressor with TimeSeriesSplit (temporal, no data leakage)."""

    # ── Temporal split: TimeSeriesSplit preserves chronological order ──
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    print(f"\nTimeSeriesSplit: {n_splits} folds (temporal, sin data leakage)")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed, verbosity=0,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        sign_match = ((y_pred > 0) == (y_test > 0)).mean()
        fold_metrics.append({"mae": mae, "sign_acc": sign_match})
        print(f"  Fold {fold}: train={len(X_train)} test={len(X_test)} "
              f"MAE={mae:.4f} sign_acc={sign_match:.3f}")

    avg_mae = np.mean([m["mae"] for m in fold_metrics])
    avg_sign = np.mean([m["sign_acc"] for m in fold_metrics])
    print(f"\n  Media temporal: MAE={avg_mae:.4f} sign_acc={avg_sign:.3f}")

    # ── Final evaluation on last fold ──
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print("METRICAS DE REGRESION (ultimo fold temporal)")
    print(f"{'='*50}")
    print(f"  MAE  (error absoluto medio): {mae:.4f}%")
    print(f"  RMSE (raiz error cuadratico): {rmse:.4f}%")
    print(f"  R²   (coef. determinacion):  {r2:.4f}")

    sign_match = ((y_pred > 0) == (y_test > 0)).mean()
    print(f"\n  Precision de direccion: {sign_match:.2%}")

    corr = np.corrcoef(np.abs(y_pred), np.abs(y_test))[0, 1]
    print(f"  Correlacion |pred| vs |actual|: {corr:.4f}")

    # ── Retrain final model on ALL data for deployment ──
    print("\nRetraining final model on ALL data for deployment...")
    final_model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, verbosity=0,
    )
    final_model.fit(X, y, verbose=False)

    # Feature importance
    print(f"\n{'='*50}")
    print("IMPORTANCIA DE VARIABLES")
    print(f"{'='*50}")
    importances = final_model.feature_importances_
    feat_imp = sorted(
        zip(X.columns, importances), key=lambda x: x[1], reverse=True
    )
    for feat, imp in feat_imp:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<25} {imp:.4f}  {bar}")

    # Sizing recommendation
    print(f"\n{'='*50}")
    print("RECOMENDACION DE USO")
    print(f"{'='*50}")
    if r2 > 0.05 and avg_sign > 0.55:
        print("  VIABLE para sizing dinamico")
    elif avg_sign > 0.50:
        print("  MARGINAL — solo usar para reducir tamaño en baja confianza")
    else:
        print("  NO VIABLE — el modelo no predice mejor que random")

    return final_model


def main():
    parser = argparse.ArgumentParser(
        description="Entrena modelo de regresion para sizing dinamico"
    )
    parser.add_argument(
        "--csv", default=str(_PROJECT_ROOT / "data" / "training_data_deepseek.csv"),
        help="CSV de entrenamiento",
    )
    parser.add_argument(
        "--output", "-o", default=str(_PROJECT_ROOT / "models" / "ml_regressor_model.pkl"),
        help="Ruta del modelo exportado",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV no encontrado: {csv_path}")
        sys.exit(1)

    X, y = load_data(str(csv_path))

    if len(y) < 20:
        print(f"\nAVISO: Solo {len(y)} muestras — resultados poco fiables.")

    model = train(X, y, seed=args.seed)

    output_path = Path(args.output)
    joblib.dump(model, output_path)
    print(f"\nModelo guardado en: {output_path}")
    print(f"  Features: {list(X.columns)}")


if __name__ == "__main__":
    main()
