#!/usr/bin/env python3
"""
Train LSTM Temporal Judge — captures intra-candle microstructure dynamics.

Unlike XGBoost (cross-sectional snapshot), this model ingests a SEQUENCE of
feature observations sampled every 2 seconds within each 5-min candle,
capturing the temporal evolution of order flow, CVD, price momentum, and
TA shifts that precede algorithmic HFT patterns.

Input shape: (batch, timesteps, features)
  - timesteps: up to 150 (300s / 2s sampling)
  - features: price_delta, cvd_net, cvd_imbalance, rsi, momentum, bb_pct,
              buy_pressure, volume_delta, bid_ask_spread, ob_imbalance

Output: P(window_outcome = UP) ∈ [0, 1]

Usage:
    python -m ml.train_lstm_judge
    python -m ml.train_lstm_judge --db btc5min.db --seq-len 150 --epochs 50
    python -m ml.train_lstm_judge --export models/lstm_judge.pt
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _PROJECT_ROOT / "btc5min.db"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "models" / "lstm_judge.pt"

# ── Temporal features extracted per 2s tick ──
# These are the columns we build from raw ticks + interpolated TA
TEMPORAL_FEATURES = [
    "price_delta_pct",      # (price - open_price) / open_price * 100
    "price_velocity",       # 1st derivative of price (normalized)
    "price_acceleration",   # 2nd derivative of price
    "cvd_cumulative_net",   # Running CVD within the candle
    "cvd_imbalance_pct",    # Buy% - Sell% of recent trades
    "rsi_interpolated",     # RSI (recomputed or interpolated)
    "momentum_instant",     # Instantaneous momentum
    "bb_position",          # Bollinger Band %B position
    "volume_burst",         # Volume spike relative to candle average
    "bid_ask_spread",       # Current spread (if available)
    "ob_imbalance",         # Order book imbalance
    "seconds_elapsed",      # Normalized time within candle [0, 1]
]

NUM_FEATURES = len(TEMPORAL_FEATURES)
DEFAULT_SEQ_LEN = 150   # 300s / 2s = 150 steps max
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3


def build_sequences_from_db(db_path: str,
                            seq_len: int = DEFAULT_SEQ_LEN,
                            min_ticks: int = 30) -> tuple:
    """
    Build temporal sequences from the database.

    For each resolved window:
    1. Query price_ticks within [open_time, close_time]
    2. Resample to 2s buckets
    3. Compute temporal features at each bucket
    4. Pad/truncate to seq_len
    5. Label = 1 if outcome == "UP", 0 if "DOWN"

    Returns:
        X: np.ndarray of shape (n_windows, seq_len, num_features)
        y: np.ndarray of shape (n_windows,) binary labels
        window_ids: list of window_ids for reference
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get resolved windows with outcome
    windows = conn.execute("""
        SELECT window_id, open_time, close_time, open_price, close_price, outcome
        FROM windows
        WHERE outcome IS NOT NULL AND open_price > 0 AND close_price > 0
        ORDER BY window_id
    """).fetchall()

    print(f"Found {len(windows)} resolved windows")

    X_list = []
    y_list = []
    wid_list = []
    skipped = 0

    for w in windows:
        wid = w["window_id"]
        open_time = w["open_time"]
        close_time = w["close_time"]
        open_price = w["open_price"]
        outcome = w["outcome"]

        if not open_time or not close_time or not open_price:
            skipped += 1
            continue

        # Query raw price ticks within this window
        ticks = conn.execute("""
            SELECT timestamp, price, source
            FROM price_ticks
            WHERE timestamp >= ? AND timestamp < ?
            ORDER BY timestamp
        """, (open_time, close_time)).fetchall()

        if len(ticks) < min_ticks:
            skipped += 1
            continue

        # Build 2s-bucketed sequence
        seq = _build_temporal_sequence(ticks, open_price, open_time,
                                       close_time, seq_len)
        if seq is None:
            skipped += 1
            continue

        X_list.append(seq)
        y_list.append(1 if outcome == "UP" else 0)
        wid_list.append(wid)

    conn.close()

    if not X_list:
        print("ERROR: No valid sequences built. Need more price_ticks data.")
        sys.exit(1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"Built {len(X)} sequences ({skipped} skipped), "
          f"shape: {X.shape}, UP: {y.sum():.0f}, DOWN: {len(y) - y.sum():.0f}")

    return X, y, wid_list


def _build_temporal_sequence(ticks: list, open_price: float,
                             open_time: float, close_time: float,
                             seq_len: int) -> np.ndarray | None:
    """
    Convert raw price ticks into a fixed-length temporal feature sequence.

    Resamples to 2-second buckets via forward-fill, computes features
    at each timestep.
    """
    window_duration = close_time - open_time
    if window_duration <= 0:
        return None

    bucket_sec = 2.0
    n_buckets = int(window_duration / bucket_sec)
    if n_buckets < 10:
        return None

    # Forward-fill prices into 2s buckets
    prices_bucketed = np.zeros(n_buckets)
    tick_idx = 0
    last_price = open_price

    for b in range(n_buckets):
        bucket_time = open_time + b * bucket_sec
        # Advance tick_idx to latest tick <= bucket_time
        while (tick_idx < len(ticks)
               and ticks[tick_idx]["timestamp"] <= bucket_time + bucket_sec):
            last_price = ticks[tick_idx]["price"]
            tick_idx += 1
        prices_bucketed[b] = last_price

    # Compute features at each timestep
    features = np.zeros((n_buckets, NUM_FEATURES), dtype=np.float32)

    for i in range(n_buckets):
        p = prices_bucketed[i]

        # 1. price_delta_pct
        features[i, 0] = (p - open_price) / open_price * 100 if open_price > 0 else 0

        # 2. price_velocity (1st derivative, normalized)
        if i >= 1:
            features[i, 1] = (prices_bucketed[i] - prices_bucketed[i - 1]) / open_price * 1e4
        else:
            features[i, 1] = 0

        # 3. price_acceleration (2nd derivative)
        if i >= 2:
            v1 = prices_bucketed[i] - prices_bucketed[i - 1]
            v0 = prices_bucketed[i - 1] - prices_bucketed[i - 2]
            features[i, 2] = (v1 - v0) / open_price * 1e6
        else:
            features[i, 2] = 0

        # 4. cvd_cumulative_net (placeholder — filled from aggTrade if available)
        # For training from historical data, approximate from price direction
        if i >= 1:
            direction = 1 if prices_bucketed[i] > prices_bucketed[i - 1] else -1
            features[i, 3] = features[i - 1, 3] + direction  # cumulative direction proxy

        # 5. cvd_imbalance_pct (proxy from price momentum)
        window = min(15, i + 1)  # ~30s lookback
        if window >= 3:
            recent = prices_bucketed[i - window + 1:i + 1]
            ups = sum(1 for j in range(1, len(recent)) if recent[j] > recent[j - 1])
            features[i, 4] = (ups / (window - 1) - 0.5) * 100

        # 6. rsi_interpolated (fast RSI on bucketed prices)
        rsi_window = min(70, i + 1)  # ~14-period at 2s = 28s
        if rsi_window >= 5:
            deltas = np.diff(prices_bucketed[max(0, i - rsi_window + 1):i + 1])
            gains = np.maximum(deltas, 0)
            losses = np.maximum(-deltas, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features[i, 5] = 100 - (100 / (1 + rs))
            else:
                features[i, 5] = 100 if avg_gain > 0 else 50

        # 7. momentum_instant
        lookback = min(15, i)
        if lookback >= 3:
            features[i, 6] = (p - prices_bucketed[i - lookback]) / open_price * 1e4

        # 8. bb_position (20-period BB on bucketed data)
        bb_window = min(100, i + 1)  # ~200s lookback
        if bb_window >= 10:
            window_prices = prices_bucketed[i - bb_window + 1:i + 1]
            mean = np.mean(window_prices)
            std = np.std(window_prices)
            if std > 0:
                features[i, 7] = (p - (mean - 2 * std)) / (4 * std) * 100
            else:
                features[i, 7] = 50

        # 9. volume_burst (proxy: absolute velocity relative to average)
        if i >= 5:
            recent_vols = np.abs(np.diff(prices_bucketed[max(0, i - 30):i + 1]))
            if len(recent_vols) > 0:
                avg_vol = np.mean(recent_vols)
                curr_vol = abs(prices_bucketed[i] - prices_bucketed[i - 1])
                features[i, 8] = curr_vol / avg_vol if avg_vol > 0 else 1.0

        # 10. bid_ask_spread (placeholder — 0 for historical)
        features[i, 9] = 0

        # 11. ob_imbalance (placeholder — 0.5 for historical)
        features[i, 10] = 0.5

        # 12. seconds_elapsed (normalized [0, 1])
        features[i, 11] = i / max(n_buckets - 1, 1)

    # Pad or truncate to seq_len
    if n_buckets >= seq_len:
        return features[-seq_len:]  # Take last seq_len steps
    else:
        padded = np.zeros((seq_len, NUM_FEATURES), dtype=np.float32)
        padded[-n_buckets:] = features  # Right-align (pad left with zeros)
        return padded


def train_model(X: np.ndarray, y: np.ndarray,
                epochs: int = 50,
                output_path: str = str(_DEFAULT_OUTPUT),
                validation_split: float = 0.2):
    """Train LSTM model on temporal sequences."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("ERROR: PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    # Temporal split (no data leakage)
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Train UP%: {y_train.mean() * 100:.1f}%, "
          f"Val UP%: {y_val.mean() * 100:.1f}%")

    # Normalize features (per-feature z-score from training set)
    # Shape: (n, seq_len, features) → compute mean/std over (n, seq_len) per feature
    train_flat = X_train.reshape(-1, NUM_FEATURES)
    feat_mean = train_flat.mean(axis=0)
    feat_std = train_flat.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0  # Prevent division by zero

    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = LSTMJudge(
        input_dim=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(model)

    # Class weights for imbalanced data
    pos_weight = torch.tensor(
        [(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)],
        dtype=torch.float32,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    best_state = None
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == yb).sum().item()
            train_total += len(yb)

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(xb)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_total += len(yb)

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100 if val_total > 0 else 0
        train_avg_loss = train_loss / train_total
        val_avg_loss = val_loss / val_total if val_total > 0 else 0

        print(f"Epoch {epoch + 1:3d}/{epochs} | "
              f"Train Loss: {train_avg_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_avg_loss:.4f} Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1} "
                      f"(best val acc: {best_val_acc:.1f}%)")
                break

    # Save best model + normalization params
    if best_state is not None:
        model.load_state_dict(best_state)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "seq_len": X.shape[1],
        },
        "normalization": {
            "mean": feat_mean,
            "std": feat_std,
        },
        "feature_names": TEMPORAL_FEATURES,
        "best_val_acc": best_val_acc,
    }, output_path)

    print(f"\nModel saved to {output_path}")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")

    return model


class LSTMJudge(torch.nn.Module if 'torch' in dir() else object):
    """
    Bidirectional LSTM with attention for intra-candle prediction.

    Architecture:
    1. Bidirectional LSTM captures temporal patterns in both directions
    2. Attention layer weights important timesteps (e.g., sweep moments)
    3. FC head outputs logit for P(UP)
    """
    pass  # Defined below after torch import check


def _define_model():
    """Define the LSTMJudge model class with PyTorch dependency."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    class _LSTMJudge(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64,
                     num_layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            # Bidirectional LSTM
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
            )

            # Temporal attention
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

            # Classification head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            # lstm_out: (batch, seq_len, hidden_dim * 2)

            # Attention weights
            attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)

            # Weighted sum of LSTM outputs
            context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_dim * 2)

            # Classify
            logit = self.classifier(context)  # (batch, 1)
            return logit

    return _LSTMJudge


# Late-bind the model class
_model_cls = _define_model()
if _model_cls is not None:
    LSTMJudge = _model_cls


# ── Inference helper (used by engine.py at runtime) ──

def load_lstm_judge(model_path: str = str(_DEFAULT_OUTPUT)):
    """
    Load a trained LSTM Judge for inference.

    Returns:
        (model, config, norm_params) or (None, None, None) if unavailable.
    """
    try:
        import torch
    except ImportError:
        return None, None, None

    path = Path(model_path)
    if not path.exists():
        return None, None, None

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["model_config"]

        model_cls = _define_model()
        if model_cls is None:
            return None, None, None

        model = model_cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=0,  # No dropout at inference
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        norm = checkpoint["normalization"]
        return model, config, norm
    except Exception as e:
        print(f"[LSTM] Failed to load model: {e}")
        return None, None, None


def predict_with_lstm(model, norm: dict, sequence: np.ndarray) -> float:
    """
    Run inference on a single temporal sequence.

    Args:
        model: Loaded LSTMJudge model
        norm: {"mean": np.ndarray, "std": np.ndarray}
        sequence: np.ndarray of shape (seq_len, num_features)

    Returns:
        P(UP) probability [0, 1]
    """
    import torch

    # Normalize
    seq_norm = (sequence - norm["mean"]) / norm["std"]
    # Add batch dimension
    x = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logit = model(x).squeeze()
        prob = torch.sigmoid(logit).item()

    return prob


# ── CLI Entry Point ──

def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM Temporal Judge on intra-candle sequences"
    )
    parser.add_argument("--db", default=str(_DEFAULT_DB),
                        help="Path to btc5min.db")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help="Sequence length (timesteps)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--export", default=str(_DEFAULT_OUTPUT),
                        help="Output model path")
    parser.add_argument("--min-ticks", type=int, default=30,
                        help="Minimum ticks per window to include")
    args = parser.parse_args()

    print("=" * 60)
    print("LSTM Temporal Judge — Intra-Candle Microstructure Model")
    print("=" * 60)

    t0 = time.time()
    X, y, wids = build_sequences_from_db(
        args.db, seq_len=args.seq_len, min_ticks=args.min_ticks
    )
    print(f"Data built in {time.time() - t0:.1f}s")

    train_model(X, y, epochs=args.epochs, output_path=args.export)


if __name__ == "__main__":
    main()
