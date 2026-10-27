"""Phase 5B: LSTM (PyTorch) on next-day log-return sequences.

- Uses causal v2 features from processed_v2/ and target Log_Return_Next.
- Sequences of the previous 30 days predict the next-day log return.
- Scaler fit on TRAIN rows only. Chronological splits. Early stopping on validation.
- Selects configuration by validation MAE. Does NOT use the test set for tuning.
- Writes only to models_v2/lstm/ and reports/model_v2/lstm/.
"""

import argparse
import copy
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import train_models as tm  # reuse metric contract

SEQ_LEN = 30
TARGET = "Log_Return_Next"
SEED = 42

CONFIGS = {
    "lstm_a": dict(hidden=32, layers=1, dropout=0.10, lr=1e-3, weight_decay=1e-5,
                   batch_size=64, max_epochs=300, patience=20),
    "lstm_b": dict(hidden=64, layers=1, dropout=0.20, lr=5e-4, weight_decay=1e-4,
                   batch_size=64, max_epochs=300, patience=20),
}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_write(path, force):
    if os.path.exists(path) and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --force)")
    os.makedirs(os.path.dirname(path), exist_ok=True)


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_v2(processed_v2):
    """Load train/val/test and concatenate in chronological order (contiguous splits)."""
    frames = {}
    for name in ("train", "val", "test"):
        frames[name] = pd.read_csv(os.path.join(processed_v2, f"{name}.csv"), parse_dates=["Date"])
    full = pd.concat([frames["train"], frames["val"], frames["test"]], ignore_index=True)
    split_label = np.array(["train"] * len(frames["train"]) +
                           ["val"] * len(frames["val"]) +
                           ["test"] * len(frames["test"]))
    return frames, full, split_label


def make_windows(features, targets, seq_len):
    """Window ending at row i = features[i-seq_len+1 : i+1]; target = targets[i].

    Only rows <= i are used, so no future information enters the input.
    """
    n = len(features)
    xs, ys, ends = [], [], []
    for i in range(seq_len - 1, n):
        xs.append(features[i - seq_len + 1:i + 1])
        ys.append(targets[i])
        ends.append(i)
    return (np.asarray(xs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32),
            np.asarray(ends, dtype=np.int64))


def fit_scaler(train_features):
    """Fit StandardScaler on training rows only (leakage-safe)."""
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=32, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.drop(last)).squeeze(-1)


def train_lstm(cfg, X_train, y_train, X_val, y_val, seed=SEED):
    set_seed(seed)
    n_features = X_train.shape[2]
    model = LSTMRegressor(n_features, hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)

    n = len(Xtr)
    gen = torch.Generator().manual_seed(seed)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = 0
    history = []

    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, cfg["batch_size"]):
            idx = perm[start:start + cfg["batch_size"]]
            opt.zero_grad()
            pred = model(Xtr[idx])
            loss = loss_fn(pred, ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva), yva))
        history.append(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model, best_val, history


def predict_return(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().astype(float)


def price_metrics(close, actual_price, pred_price):
    return tm.regression_metrics(actual_price, pred_price, close)


def main():
    parser = argparse.ArgumentParser(description="Phase 5B LSTM training (val selection)")
    parser.add_argument("--processed-v2", default="processed_v2")
    parser.add_argument("--models-v2", default=os.path.join("models_v2", "lstm"))
    parser.add_argument("--reports-v2", default=os.path.join("reports", "model_v2", "lstm"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    processed_v2 = os.path.join(here, args.processed_v2)
    models_dir = os.path.join(here, args.models_v2)
    reports_dir = os.path.join(here, args.reports_v2)

    with open(os.path.join(processed_v2, "feature_metadata_v2.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    features = list(meta["features"])
    assert meta["target"] == TARGET

    section("1. LOAD v2 SPLITS (test not used for tuning)")
    frames, full, split_label = load_v2(processed_v2)
    for name in ("train", "val", "test"):
        f = frames[name]
        print(f"{name:<5} rows={len(f):>5}  {f['Date'].min().date()} -> {f['Date'].max().date()}")

    section("2. SCALE (fit on TRAIN only)")
    scaler = fit_scaler(frames["train"][features].to_numpy(dtype=float))
    F = scaler.transform(full[features].to_numpy(dtype=float))
    y = full[TARGET].to_numpy(dtype=float)
    close = full["Close"].to_numpy(dtype=float)
    actual_price = full["Close_Next"].to_numpy(dtype=float)

    # scale target with train stats only
    y_mean = float(frames["train"][TARGET].mean())
    y_std = float(frames["train"][TARGET].std())
    y_scaled = (y - y_mean) / y_std
    print(f"feature scaler fit on {len(frames['train'])} train rows")
    print(f"target scaled with train mean={y_mean:.6f} std={y_std:.6f}")

    section("3. BUILD 30-DAY SEQUENCES")
    X, yseq, ends = make_windows(F, y_scaled, SEQ_LEN)
    end_split = split_label[ends]
    masks = {name: end_split == name for name in ("train", "val", "test")}
    for name in ("train", "val", "test"):
        print(f"{name:<5} sequences: {int(masks[name].sum())}")
    assert int(masks["val"].sum()) == len(frames["val"]), "val sequence count mismatch"
    assert int(masks["test"].sum()) == len(frames["test"]), "test sequence count mismatch"

    Xtr, ytr = X[masks["train"]], yseq[masks["train"]]
    Xva, yva = X[masks["val"]], yseq[masks["val"]]

    section("4. TRAIN CANDIDATES + SELECT BY VALIDATION MAE")
    results = {}
    trained = {}
    for name, cfg in CONFIGS.items():
        model, best_val_loss, history = train_lstm(cfg, Xtr, ytr, Xva, yva)
        # convert val sequences back to price
        val_end = ends[masks["val"]]
        pred_ret = predict_return(model, Xva) * y_std + y_mean
        pred_price = close[val_end] * np.exp(pred_ret)
        m = price_metrics(close[val_end], actual_price[val_end], pred_price)
        results[name] = {"params": cfg, "epochs_run": len(history), "best_val_loss": best_val_loss,
                         "metrics": m}
        trained[name] = model
        print(f"  {name:<8} epochs={len(history):>3}  val MAE={m['MAE']:>10.2f}  RMSE={m['RMSE']:>10.2f}  "
              f"R2={m['R2']:>7.4f}  DirAcc={m['DirectionalAccuracy']:.3f}")

    # baseline on val
    val_end = ends[masks["val"]]
    base_val = price_metrics(close[val_end], actual_price[val_end], close[val_end])
    print(f"  {'baseline':<8} val MAE={base_val['MAE']:>10.2f}  DirAcc={base_val['DirectionalAccuracy']:.3f}")

    best_name = min(results, key=lambda k: results[k]["metrics"]["MAE"])
    print(f"\nselected LSTM config by val MAE: {best_name}")

    section("5. SAVE MODEL + VALIDATION REPORT")
    model_path = os.path.join(models_dir, "lstm_logreturn.pt")
    scaler_path = os.path.join(models_dir, "lstm_scaler.joblib")
    config_path = os.path.join(models_dir, "lstm_config.json")
    safe_write(model_path, args.force)
    torch.save(trained[best_name].state_dict(), model_path)
    safe_write(scaler_path, args.force)
    joblib.dump(scaler, scaler_path)
    safe_write(config_path, args.force)
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump({"selected": best_name, "params": CONFIGS[best_name], "seq_len": SEQ_LEN,
                   "features": features, "target": TARGET,
                   "target_mean": y_mean, "target_std": y_std}, fh, indent=2)
    print(f"saved {model_path}\n{scaler_path}\n{config_path}")

    report = {
        "phase": "5B",
        "target": TARGET,
        "seq_len": SEQ_LEN,
        "features": features,
        "selection_metric": "val price MAE",
        "selected_model": best_name,
        "baseline_val": base_val,
        "results": results,
        "test_used_for_tuning": False,
        "train_sequences": int(masks["train"].sum()),
        "val_sequences": int(masks["val"].sum()),
    }
    report_path = os.path.join(reports_dir, "validation_report_lstm.json")
    safe_write(report_path, args.force)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved {report_path}")

    section("SUMMARY")
    print(f"selected: {best_name}  (val MAE {results[best_name]['metrics']['MAE']:.2f})")
    print("Test set untouched. No test-based tuning.")


if __name__ == "__main__":
    main()
