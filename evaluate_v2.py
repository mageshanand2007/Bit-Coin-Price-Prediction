"""Phase 5A evaluation: single, final historical test-set evaluation.

Loads the saved v2 model (no retraining/tuning) and evaluates on processed_v2/test.csv.
Converts predicted log returns to prices. Writes NEW files only.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

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
    parser = argparse.ArgumentParser(description="Phase 5A final test evaluation")
    parser.add_argument("--processed-v2", default="processed_v2")
    parser.add_argument("--models-v2", default="models_v2")
    parser.add_argument("--reports-v2", default=os.path.join("reports", "model_v2"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    processed_v2 = os.path.join(here, args.processed_v2)
    models_v2 = os.path.join(here, args.models_v2)
    reports_v2 = os.path.join(here, args.reports_v2)

    with open(os.path.join(processed_v2, "feature_metadata_v2.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    features = list(meta["features"])
    target = meta["target"]
    assert target == TARGET

    section("1. LOAD TEST SPLIT (historical; single evaluation)")
    test = pd.read_csv(os.path.join(processed_v2, "test.csv"), parse_dates=["Date"])
    assert list(test[features].columns) == features, "feature order mismatch"
    print(f"test rows: {len(test)}  {test['Date'].min().date()} -> {test['Date'].max().date()}")

    close = test["Close"].to_numpy(dtype=float)
    actual_price = test["Close_Next"].to_numpy(dtype=float)
    actual_ret = test[target].to_numpy(dtype=float)

    section("2. LOAD SAVED MODEL (no retraining)")
    model_path = os.path.join(models_v2, "xgboost_v2_logreturn.joblib")
    assert os.path.isfile(model_path), f"missing model: {model_path}"
    model = joblib.load(model_path)
    print(f"loaded {model_path}")

    section("3. PREDICT + EVALUATE (prices)")
    pred_ret = np.asarray(model.predict(test[features]), dtype=float)
    assert np.all(np.isfinite(pred_ret)), "non-finite predictions"
    pred_price = close * np.exp(pred_ret)
    baseline_price = close.copy()

    results = {
        "persistence_baseline": {
            "type": "baseline",
            "metrics": tm.regression_metrics(actual_price, baseline_price, close),
        },
        "xgboost_v2_logreturn": {
            "type": "model",
            "metrics": tm.regression_metrics(actual_price, pred_price, close),
            "return_metrics": tm.regression_metrics(actual_ret, pred_ret, np.zeros_like(actual_ret)),
        },
    }
    for name, info in results.items():
        m = info["metrics"]
        print(f"  {name:<22} MAE={m['MAE']:>10.2f}  RMSE={m['RMSE']:>10.2f}  "
              f"R2={m['R2']:>7.4f}  DirAcc={m['DirectionalAccuracy']:.3f}")

    base_mae = results["persistence_baseline"]["metrics"]["MAE"]
    model_mae = results["xgboost_v2_logreturn"]["metrics"]["MAE"]
    beats = bool(model_mae < base_mae)
    print(f"\nbaseline test MAE : {base_mae:.2f}")
    print(f"xgb_v2  test MAE  : {model_mae:.2f}")
    print(f"xgb_v2 beats baseline on test MAE: {beats}")

    section("4. VALIDATION vs TEST")
    val_report_path = os.path.join(reports_v2, "validation_report_v2.json")
    val_cmp = None
    if os.path.isfile(val_report_path):
        with open(val_report_path, encoding="utf-8") as fh:
            vrep = json.load(fh)
        sel = vrep["selected_model"]
        val_mae = vrep["results"][sel]["metrics"]["MAE"]
        val_dir = vrep["results"][sel]["metrics"]["DirectionalAccuracy"]
        test_dir = results["xgboost_v2_logreturn"]["metrics"]["DirectionalAccuracy"]
        val_cmp = {"selected_model": sel, "val_MAE": val_mae, "test_MAE": model_mae,
                   "val_DirAcc": val_dir, "test_DirAcc": test_dir}
        print(f"  selected {sel}: val_MAE={val_mae:.2f} -> test_MAE={model_mae:.2f}")
        print(f"  DirAcc: val={val_dir:.3f} -> test={test_dir:.3f}")

    section("5. SAVE REPORTS")
    report = {
        "phase": "5A",
        "evaluation_type": "historical_test",
        "historical_note": ("Dataset ends 2021-07-06. Historical evaluation only; "
                            "not indicative of current or future performance."),
        "no_retraining": True,
        "no_tuning_on_test": True,
        "target": target,
        "features": features,
        "test_rows": int(len(test)),
        "test_range": [str(test["Date"].min().date()), str(test["Date"].max().date())],
        "results": results,
        "beats_baseline_on_test_mae": beats,
        "val_vs_test": val_cmp,
    }
    report_path = os.path.join(reports_v2, "test_report_v2.json")
    safe_write(report_path, args.force)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved {report_path}")

    preds = pd.DataFrame({
        "Date": test["Date"].dt.strftime("%Y-%m-%d"),
        "Close": close,
        "Actual_Close_Next": actual_price,
        "pred_price_persistence_baseline": baseline_price,
        "pred_price_xgboost_v2": pred_price,
        "pred_return_xgboost_v2": pred_ret,
    })
    preds_path = os.path.join(reports_v2, "test_predictions_v2.csv")
    safe_write(preds_path, args.force)
    preds.to_csv(preds_path, index=False)
    print(f"saved {preds_path}")

    section("SUMMARY (historical test)")
    print(f"baseline MAE={base_mae:.2f}  xgb_v2 MAE={model_mae:.2f}  beats={beats}")
    print("HISTORICAL only; dataset ends 2021-07-06. No retraining/tuning.")


if __name__ == "__main__":
    main()
