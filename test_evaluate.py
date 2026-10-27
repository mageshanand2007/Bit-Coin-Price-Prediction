"""Phase 4: Final historical test-set evaluation.

- Loads saved models from ./models/ (NO retraining, NO tuning).
- Evaluates on exactly the same chronological rows in processed/test.csv.
- Reuses the metric contract from train_models.regression_metrics.
- Writes NEW report files; never overwrites existing reports.
- Results are HISTORICAL: the dataset ends 2021-07-06.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

import train_models as tm

TARGET = "Target_Close_Next"
CURRENT_CLOSE = "Current_Close"
MODEL_FILES = {
    "linear_regression": "linear_regression.joblib",
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost.joblib",
    "xgboost_shallow": "xgboost_shallow.joblib",
}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_write(path, force):
    if os.path.exists(path) and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --force)")
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_metadata(processed_dir):
    with open(os.path.join(processed_dir, "feature_metadata.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_test_with_current_close(processed_dir, target):
    """Load val+test to reconstruct the current-day Close for test rows only.

    test's first Current_Close comes from val's last Target (the previous day's
    known close), so val is needed only to seed that one value. No test rows
    beyond the test split are used as features.
    """
    val = pd.read_csv(os.path.join(processed_dir, "val.csv"), parse_dates=["Date"])
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"), parse_dates=["Date"])

    combined = pd.concat([val, test], ignore_index=True)
    combined[CURRENT_CLOSE] = combined[target].shift(1)

    test = combined.iloc[len(val):].reset_index(drop=True)
    assert test[CURRENT_CLOSE].notna().all(), "test Current_Close has gaps"

    # Independent cross-check: current Close == next row's Close_Lag_1.
    cross = combined["Close_Lag_1"].shift(-1)
    mask = cross.notna() & combined[CURRENT_CLOSE].notna()
    assert np.allclose(combined.loc[mask, CURRENT_CLOSE], cross[mask]), "Current_Close cross-check failed"
    return test


def load_saved_models(models_dir):
    loaded = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(models_dir, fname)
        assert os.path.isfile(path), f"missing model artifact: {path}"
        loaded[name] = joblib.load(path)
    return loaded


def evaluate(models, test, features, target):
    X = test[features]
    y = test[target].to_numpy(dtype=float)
    current = test[CURRENT_CLOSE].to_numpy(dtype=float)

    results = {}
    predictions = {}

    # Naive persistence baseline: next Close = current Close.
    baseline_pred = current.copy()
    results["persistence_baseline"] = {
        "type": "baseline",
        "metrics": tm.regression_metrics(y, baseline_pred, current),
        "predictions_finite": bool(np.all(np.isfinite(baseline_pred))),
    }
    predictions["persistence_baseline"] = baseline_pred

    for name, model in models.items():
        pred = np.asarray(model.predict(X), dtype=float)
        assert pred.shape[0] == len(y), f"{name}: prediction length mismatch"
        assert np.all(np.isfinite(pred)), f"{name}: non-finite predictions"
        results[name] = {
            "type": "model",
            "metrics": tm.regression_metrics(y, pred, current),
            "predictions_finite": True,
        }
        predictions[name] = pred

    baseline_mae = results["persistence_baseline"]["metrics"]["MAE"]
    for info in results.values():
        info["beats_baseline_mae"] = bool(info["metrics"]["MAE"] < baseline_mae)

    return results, predictions


def load_val_metrics(reports_dir):
    path = os.path.join(reports_dir, "validation_report.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("results")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 test-set evaluation")
    parser.add_argument("--processed", default="processed")
    parser.add_argument("--models", default="models")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--force", action="store_true", help="allow overwriting output files")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(here, args.processed)
    models_dir = os.path.join(here, args.models)
    reports_dir = os.path.join(here, args.reports)

    section("1. METADATA")
    meta = load_metadata(processed_dir)
    features = list(meta["features"])
    target = meta["target"]
    assert target == TARGET, f"unexpected target {target}"
    print(f"features ({len(features)}): {features}")
    print(f"target    : {target}")

    section("2. LOAD TEST SPLIT (historical)")
    test = load_test_with_current_close(processed_dir, target)
    # Feature order must match training contract exactly.
    assert list(test[features].columns) == features, "feature order mismatch"
    print(f"test rows: {len(test)}  {test['Date'].min().date()} -> {test['Date'].max().date()}")
    print("feature order verified against metadata -> OK")

    forbidden = {"SNo", "Name", "Symbol", "Date", target, CURRENT_CLOSE}
    assert not (set(features) & forbidden), f"leaky columns: {set(features) & forbidden}"

    section("3. LOAD SAVED MODELS (no retraining)")
    models = load_saved_models(models_dir)
    for name in models:
        print(f"loaded {name}")

    section("4. EVALUATE ON TEST ROWS")
    results, predictions = evaluate(models, test, features, target)

    ranked = sorted(results.items(), key=lambda kv: kv[1]["metrics"]["MAE"])
    for name, info in ranked:
        m = info["metrics"]
        print(
            f"  {name:<22} MAE={m['MAE']:>10.2f}  RMSE={m['RMSE']:>10.2f}  "
            f"R2={m['R2']:>7.4f}  DirAcc={m['DirectionalAccuracy']:>6.3f}"
        )

    baseline_mae = results["persistence_baseline"]["metrics"]["MAE"]
    best_name = ranked[0][0]
    trained = [n for n in results if n != "persistence_baseline"]
    best_trained = min(trained, key=lambda n: results[n]["metrics"]["MAE"])
    any_beats = any(results[n]["beats_baseline_mae"] for n in trained)

    print(f"\nbest by test MAE        : {best_name}")
    print(f"best trained model      : {best_trained}")
    print(f"any trained model beats baseline: {any_beats}")

    section("5. VALIDATION vs TEST COMPARISON")
    val_results = load_val_metrics(reports_dir)
    comparison = {}
    if val_results:
        for name in results:
            vm = val_results.get(name, {}).get("metrics", {})
            tst = results[name]["metrics"]
            comparison[name] = {
                "val_MAE": vm.get("MAE"),
                "test_MAE": tst["MAE"],
                "MAE_change_pct": (
                    None if not vm.get("MAE") else (tst["MAE"] - vm["MAE"]) / vm["MAE"] * 100.0
                ),
                "val_R2": vm.get("R2"),
                "test_R2": tst["R2"],
                "val_DirAcc": vm.get("DirectionalAccuracy"),
                "test_DirAcc": tst["DirectionalAccuracy"],
            }
            print(
                f"  {name:<22} val_MAE={vm.get('MAE'):>9.2f}  test_MAE={tst['MAE']:>9.2f}  "
                f"chg%={comparison[name]['MAE_change_pct']:>7.1f}"
            )
    else:
        print("validation_report.json not found; skipping comparison")

    section("6. SAVE REPORTS")
    test_report = {
        "phase": 4,
        "evaluation_type": "historical_test",
        "historical_note": (
            "Dataset ends 2021-07-06. These test metrics are a historical evaluation "
            "of saved models and are NOT indicative of current or future performance."
        ),
        "no_retraining": True,
        "no_tuning_on_test": True,
        "features": features,
        "target": target,
        "excluded_from_features": sorted(forbidden),
        "test_rows": int(len(test)),
        "test_range": [str(test["Date"].min().date()), str(test["Date"].max().date())],
        "baseline_note": (
            "Persistence baseline predicts next Close = current Close; directional "
            "accuracy ~0 because it always predicts 'no change'."
        ),
        "results": results,
        "best_model_by_test_mae": best_name,
        "best_trained_model_by_test_mae": best_trained,
        "any_trained_beats_baseline": any_beats,
        "val_vs_test": comparison,
    }
    report_path = os.path.join(reports_dir, "test_report.json")
    safe_write(report_path, args.force)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(test_report, fh, indent=2)
    print(f"saved {report_path}")

    preds = pd.DataFrame({"Date": test["Date"].dt.strftime("%Y-%m-%d"),
                          CURRENT_CLOSE: test[CURRENT_CLOSE],
                          "Actual_Close_Next": test[target]})
    for name, pred in predictions.items():
        preds[f"pred_{name}"] = pred
    preds_path = os.path.join(reports_dir, "test_predictions.csv")
    safe_write(preds_path, args.force)
    preds.to_csv(preds_path, index=False)
    print(f"saved {preds_path}")

    section("SUMMARY (historical test evaluation)")
    print(f"test rows              : {len(test)}")
    print(f"best by test MAE       : {best_name}")
    print(f"trained models beat baseline: {any_beats}")
    print("HISTORICAL evaluation only; dataset ends 2021-07-06. No retraining/tuning.")


if __name__ == "__main__":
    main()
