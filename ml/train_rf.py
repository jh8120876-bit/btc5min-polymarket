# Pipeline de ML para evaluar la fiabilidad del Swarm y las estrategias RAG
#
# Entrena un RandomForestClassifier sobre la tabla `predictions` de btc5min.db
# usando TimeSeriesSplit (sin data leakage). Optimiza Precision porque es
# preferible no operar a operar y perder.
#
# Uso:
#   python -m ml.train_rf              # entrena y guarda modelo
#   python -m ml.train_rf --min-rows 100  # mínimo de filas requeridas

import os
import sys
import sqlite3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _PROJECT_ROOT / "btc5min.db"
MODEL_DIR = _PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "oracle_rf.joblib"

# ── Feature groups ───────────────────────────────────────────
# Numéricas del usuario + TA snapshot + ML features ya almacenadas
NUMERIC_FEATURES = [
    # Core request
    "ai_latency_ms",
    "consensus_agreement",
    "price_at_bet",
    # TA snapshot (already in DB)
    "rsi",
    "macd_histogram",
    "bb_pct",
    "momentum",
    "volatility",
    "atr_pct",
    "tick_buy_pressure",
    # ML features
    "confidence",
    "original_confidence",
    "price_change_pct",
    "return_1m",
    "return_5m",
    "return_15m",
    "price_acceleration",
    "range_position",
    "volatility_zscore",
    # Market microstructure
    "volume_24h",
    "funding_rate",
    "open_interest",
    "trend_slope_15m",
    "trend_slope_1h",
    "trend_alignment",
    "liq_buy_vol_5m",
    "liq_sell_vol_5m",
    "reversal_alert",
    # Temporal velocity features
    "rsi_velocity",
    "rsi_acceleration",
    "momentum_velocity",
    "cvd_velocity",
]

TEMPORAL_FEATURES = ["hour_utc", "day_of_week"]
CATEGORICAL_FEATURES = ["rag_strategy", "signal_quality", "prediction"]


def load_data() -> pd.DataFrame:
    """Extract resolved predictions from SQLite into a DataFrame."""
    if not DB_PATH.exists():
        print(f"[ERROR] Base de datos no encontrada: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE correct IS NOT NULL ORDER BY id ASC",
        conn,
    )
    conn.close()
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build X matrix and y target from raw predictions DataFrame.

    Returns (X, y, feature_names).
    """
    # ── Target ───────────────────────────────────────────────
    df["target"] = df["correct"].astype(int)

    # ── Temporal: extract from created_at if hour_utc/day_of_week are null ──
    if df["hour_utc"].isna().sum() > len(df) * 0.5:
        try:
            ts = pd.to_datetime(df["created_at"], errors="coerce")
            df["hour_utc"] = ts.dt.hour.fillna(12).astype(int)
            df["day_of_week"] = ts.dt.dayofweek.fillna(3).astype(int)
        except Exception:
            df["hour_utc"] = df["hour_utc"].fillna(12).astype(int)
            df["day_of_week"] = df["day_of_week"].fillna(3).astype(int)

    # ── Cyclical encoding for hour (captures 23→0 continuity) ──
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_utc"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_utc"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── Categorical: rag_strategy ──
    df["rag_strategy"] = df["rag_strategy"].fillna("Base_Instinct")
    df["signal_quality"] = df["signal_quality"].fillna("NORMAL")
    df["prediction"] = df["prediction"].fillna("UP")

    # One-hot encode categoricals
    cat_dummies = pd.get_dummies(
        df[CATEGORICAL_FEATURES], prefix_sep="_", dtype=int,
    )

    # ── Numeric: fill NaN with median (robust to outliers) ──
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    df_num = df[numeric_cols].copy()
    for col in df_num.columns:
        median_val = df_num[col].median()
        df_num[col] = df_num[col].fillna(median_val if pd.notna(median_val) else 0)

    # ── Assemble feature matrix ──
    cyclical = df[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]]
    X = pd.concat([df_num, cyclical, cat_dummies], axis=1)

    # Drop any columns that are 100% constant (zero variance)
    nunique = X.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"[INFO] Dropping {len(constant_cols)} constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)

    feature_names = X.columns.tolist()
    return X, df["target"], feature_names


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    n_splits: int = 5,
) -> RandomForestClassifier:
    """Train RandomForest with TimeSeriesSplit. Returns the final model."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Accumulate out-of-fold predictions for aggregate metrics
    oof_preds = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros(len(y))
    fold_metrics = []

    print("\n" + "=" * 65)
    print("  ENTRENAMIENTO — RandomForest + TimeSeriesSplit")
    print("=" * 65)
    print(f"  Filas totales: {len(X)} | Features: {len(feature_names)}")
    print(f"  Clase 1 (win): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Clase 0 (loss): {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")
    print(f"  Folds: {n_splits} (TimeSeriesSplit)")
    print("-" * 65)

    best_model = None
    best_precision = -1

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features (fit on train only to avoid leakage)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:, 1]

        oof_preds[test_idx] = preds
        oof_proba[test_idx] = proba

        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        acc = accuracy_score(y_test, preds)

        fold_metrics.append({"precision": prec, "recall": rec, "f1": f1, "acc": acc})
        print(f"  Fold {fold_i}: Precision={prec:.3f}  Recall={rec:.3f}  "
              f"F1={f1:.3f}  Acc={acc:.3f}  "
              f"(train={len(train_idx)}, test={len(test_idx)})")

        # Keep the model with best precision (our optimization target)
        if prec > best_precision:
            best_precision = prec
            best_model = model
            best_scaler = scaler

    # ── Aggregate out-of-fold report ──
    valid_mask = oof_preds >= 0
    y_valid = y[valid_mask]
    p_valid = oof_preds[valid_mask]

    print("\n" + "=" * 65)
    print("  REPORTE AGREGADO (Out-of-Fold)")
    print("=" * 65)
    print(classification_report(
        y_valid, p_valid,
        target_names=["LOSS (0)", "WIN (1)"],
        digits=3,
        zero_division=0,
    ))

    cm = confusion_matrix(y_valid, p_valid)
    print("  Confusion Matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    # ── Cross-fold averages ──
    avg = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
    std = {k: np.std([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
    print(f"\n  Promedio folds:")
    print(f"    Precision = {avg['precision']:.3f} +/- {std['precision']:.3f}")
    print(f"    Recall    = {avg['recall']:.3f} +/- {std['recall']:.3f}")
    print(f"    F1        = {avg['f1']:.3f} +/- {std['f1']:.3f}")
    print(f"    Accuracy  = {avg['acc']:.3f} +/- {std['acc']:.3f}")

    # ── Feature importance ──
    print("\n" + "=" * 65)
    print("  FEATURE IMPORTANCE (Top 20)")
    print("=" * 65)
    importances = best_model.feature_importances_
    fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for rank, (name, imp) in enumerate(fi[:20], 1):
        bar = "#" * int(imp * 200)
        print(f"  {rank:2d}. {name:28s} {imp:.4f}  {bar}")

    # ── Retrain final model on ALL data for deployment ──
    print("\n[INFO] Reentrenando modelo final sobre TODOS los datos...")
    final_scaler = StandardScaler()
    X_all_s = final_scaler.fit_transform(X)
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_all_s, y)

    return final_model, final_scaler, feature_names, fi


def save_model(model, scaler, feature_names, feature_importances):
    """Persist model + metadata to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "version": "1.0",
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"\n[OK] Modelo guardado: {MODEL_PATH} ({size_kb:.0f} KB)")
    print(f"     Features: {len(feature_names)}")
    print(f"     Para cargar: joblib.load('{MODEL_PATH}')")


def main():
    parser = argparse.ArgumentParser(description="BTC5min ML Oracle — Train Pipeline")
    parser.add_argument("--min-rows", type=int, default=50,
                        help="Minimum resolved predictions required (default: 50)")
    parser.add_argument("--splits", type=int, default=5,
                        help="TimeSeriesSplit folds (default: 5)")
    args = parser.parse_args()

    print("=" * 65)
    print("  BTC5MIN — ML ORACLE TRAINING PIPELINE")
    print("=" * 65)
    print(f"  DB: {DB_PATH}")

    # ── 1. Load data ──
    df = load_data()
    print(f"  Predicciones resueltas: {len(df)}")

    if len(df) < args.min_rows:
        print(f"\n[WARNING] Solo hay {len(df)} registros resueltos.")
        print(f"          Se requieren al menos {args.min_rows} para entrenar.")
        print(f"          Deja el bot corriendo para acumular mas datos.")
        sys.exit(0)

    # ── 2. Feature engineering ──
    X, y, feature_names = engineer_features(df)
    print(f"  Features construidas: {len(feature_names)}")

    # ── 3. Train + evaluate ──
    model, scaler, fnames, fi = train_and_evaluate(X, y, feature_names, n_splits=args.splits)

    # ── 4. Save ──
    save_model(model, scaler, fnames, fi)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETADO")
    print("=" * 65)


if __name__ == "__main__":
    main()
