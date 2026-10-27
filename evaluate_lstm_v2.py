"""Phase 5B evaluation: single historical test-set evaluation of the saved LSTM.

Loads models_v2/lstm/ artifacts (no retraining/tuning), rebuilds 30-day sequences,
converts predicted log returns to prices, and compares with the persistence baseline
and the existing v2 XGBoost results on comparable dates. Writes NEW files only.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

import lstm_v2 as L
import train_models as tm

TARGET = "Log_Return_Next"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_write(path, force):
    if os.path.exists(path) and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --force)")
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 5B LSTM test evaluation")
    parser.add_argument("--processed-v2", default="processed_v2")
    parser.add_argument("--models-v2", default=os.path.join("models_v2", "lstm"))
    parser.add_argument("--reports-v2", default=os.path.join("reports", "model_v2"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    processed_v2 = os.path.join(here, args.processed_v2)
    models_dir = os.path.join(here, args.models_v2)
    reports_dir = os.path.join(here, args.reports_v2)
    lstm_reports = os.path.join(reports_dir, "lstm")

    with open(os.path.join(processed_v2, "feature_metadata_v2.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    features = list(meta["features"])
    with open(os.path.join(models_dir, "lstm_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    seq_len = cfg["seq_len"]
    y_mean, y_std = cfg["target_mean"], cfg["target_std"]
    assert cfg["target"] == TARGET

    section("1. REBUILD SEQUENCES (test again; historical)")
    frames, full, split_label = L.load_v2(processed_v2)
    scaler = joblib.load(os.path.join(models_dir, "lstm_scaler.joblib"))
    F = scaler.transform(full[features].to_numpy(dtype=float))
    y = full[TARGET].to_numpy(dtype=float)
    close = full["Close"].to_numpy(dtype=float)
    actual_price = full["Close_Next"].to_numpy(dtype=float)

    X, _, ends = L.make_windows(F, y, seq_len)
    end_split = split_label[ends]
    test_mask = end_split == "test"
    Xte = X[test_mask]
    te_end = ends[test_mask]
    print(f"test sequences: {int(test_mask.sum())}  "
          f"{full['Date'].iloc[te_end].min().date()} -> {full['Date'].iloc[te_end].max().date()}")

    section("2. LOAD SAVED LSTM")
    model = L.LSTMRegressor(len(features), hidden=cfg["params"]["hidden"],
                            layers=cfg["params"]["layers"], dropout=cfg["params"]["dropout"])
    state = torch.load(os.path.join(models_dir, "lstm_logreturn.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"loaded lstm_logreturn.pt (config {cfg['selected']})")

    section("3. PREDICT + EVALUATE (prices)")
    pred_ret = L.predict_return(model, Xte) * y_std + y_mean
    assert np.all(np.isfinite(pred_ret)), "non-finite predictions"
    pred_price = close[te_end] * np.exp(pred_ret)
    baseline_price = close[te_end].copy()

    results = {
        "persistence_baseline": {"type": "baseline",
                                 "metrics": tm.regression_metrics(actual_price[te_end], baseline_price, close[te_end])},
        "lstm_v2_logreturn": {"type": "model",
                              "metrics": tm.regression_metrics(actual_price[te_end], pred_price, close[te_end]),
                              "return_metrics": tm.regression_metrics(y[te_end], pred_ret, np.zeros(len(te_end)))},
    }
    for name, info in results.items():
        m = info["metrics"]
        print(f"  {name:<22} MAE={m['MAE']:>10.2f}  RMSE={m['RMSE']:>10.2f}  R2={m['R2']:>7.4f}  "
              f"DirAcc={m['DirectionalAccuracy']:.3f}")

    section("4. COMPARE vs XGBoost v2 (same 444 test dates)")
    xgb = None
    xgb_path = os.path.join(reports_dir, "test_report_v2.json")
    if os.path.isfile(xgb_path):
        with open(xgb_path, encoding="utf-8") as fh:
            xrep = json.load(fh)
        xgb = xrep["results"]["xgboost_v2_logreturn"]["metrics"]
        print(f"  {'xgboost_v2':<22} MAE={xgb['MAE']:>10.2f}  RMSE={xgb['RMSE']:>10.2f}  "
              f"R2={xgb['R2']:>7.4f}  DirAcc={xgb['DirectionalAccuracy']:.3f}")

    base_mae = results["persistence_baseline"]["metrics"]["MAE"]
    lstm_mae = results["lstm_v2_logreturn"]["metrics"]["MAE"]
    beats_base = bool(lstm_mae < base_mae)
    beats_xgb = bool(xgb and lstm_mae < xgb["MAE"])
    print(f"\nLSTM beats baseline: {beats_base}   LSTM beats XGBoost v2: {beats_xgb}")

    section("5. SAVE REPORTS")
    report = {
        "phase": "5B",
        "evaluation_type": "historical_test",
        "historical_note": ("Dataset ends 2021-07-06. The historical test period was already "
                            "examined in Phases 4/5A, so this is NOT an independent final test."),
        "no_retraining": True,
        "no_tuning_on_test": True,
        "target": TARGET,
        "seq_len": seq_len,
        "features": features,
        "test_sequences": int(test_mask.sum()),
        "results": results,
        "xgboost_v2_test_metrics": xgb,
        "lstm_beats_baseline": beats_base,
        "lstm_beats_xgboost_v2": beats_xgb,
        "test_used_for_tuning": False,
    }
    report_path = os.path.join(lstm_reports, "test_report_lstm.json")
    safe_write(report_path, args.force)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved {report_path}")

    preds = pd.DataFrame({
        "Date": full["Date"].iloc[te_end].dt.strftime("%Y-%m-%d").to_numpy(),
        "Close": close[te_end],
        "Actual_Close_Next": actual_price[te_end],
        "pred_price_persistence_baseline": baseline_price,
        "pred_price_lstm_v2": pred_price,
        "pred_return_lstm_v2": pred_ret,
    })
    preds_path = os.path.join(lstm_reports, "test_predictions_lstm.csv")
    safe_write(preds_path, args.force)
    preds.to_csv(preds_path, index=False)
    print(f"saved {preds_path}")

    section("SUMMARY (historical test)")
    print(f"baseline MAE={base_mae:.2f}  lstm MAE={lstm_mae:.2f}  beats_baseline={beats_base}  beats_xgb={beats_xgb}")
    print("HISTORICAL only; not an independent final test. Dataset ends 2021-07-06.")


if __name__ == "__main__":
    main()
