"""Phase 5A training: XGBoost regressor on next-day log return.

- Trains on processed_v2/train.csv, selects configuration on processed_v2/val.csv ONLY.
- Converts predicted log returns to prices via Close*exp(pred_return).
- Does NOT touch the test set. Does NOT modify Phase 1-4 artifacts.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

import train_models as tm  # reuse the exact metric contract
from xgboost import XGBRegressor

TARGET = "Log_Return_Next"
FEATURE_META = "feature_metadata_v2.json"

XGB_CANDIDATES = {
    # small, fixed grid; selection by validation MAE only
    "xgb_v2_a": dict(n_estimators=400, learning_rate=0.03, max_depth=3,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=5),
    "xgb_v2_b": dict(n_estimators=600, learning_rate=0.02, max_depth=2,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=3),
    "xgb_v2_c": dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                     subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, min_child_weight=5),
}
RANDOM_STATE = 42


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_write(path, force):
    if os.path.exists(path) and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --force)")
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_metadata(processed_v2):
    with open(os.path.join(processed_v2, FEATURE_META), encoding="utf-8") as fh:
        return json.load(fh)


def make_model(params):
    return XGBRegressor(
        **params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        objective="reg:squarederror", tree_method="hist",
    )


def price_from_return(close, pred_return):
    return close * np.exp(pred_return)


def evaluate_split(df, features, target, predict_return):
    close = df["Close"].to_numpy(dtype=float)
    actual_ret = df[target].to_numpy(dtype=float)
    actual_price = df["Close_Next"].to_numpy(dtype=float)
    pred_ret = np.asarray(predict_return, dtype=float)
    pred_price = price_from_return(close, pred_ret)
    return {
        "price": tm.regression_metrics(actual_price, pred_price, close),
        "return": tm.regression_metrics(actual_ret, pred_ret, np.zeros_like(actual_ret)),
        "actual_price": actual_price,
        "pred_price": pred_price,
        "actual_return": actual_ret,
        "pred_return": pred_ret,
        "close": close,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 5A XGBoost training (val selection)")
    parser.add_argument("--processed-v2", default="processed_v2")
    parser.add_argument("--models-v2", default="models_v2")
    parser.add_argument("--reports-v2", default=os.path.join("reports", "model_v2"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    processed_v2 = os.path.join(here, args.processed_v2)
    models_v2 = os.path.join(here, args.models_v2)
    reports_v2 = os.path.join(here, args.reports_v2)

    section("1. METADATA")
    meta = load_metadata(processed_v2)
    features = list(meta["features"])
    target = meta["target"]
    assert target == TARGET, f"unexpected target {target}"
    print(f"features ({len(features)}): {features}")
    print(f"target: {target}")

    section("2. LOAD TRAIN/VAL (test never loaded)")
    train = pd.read_csv(os.path.join(processed_v2, "train.csv"), parse_dates=["Date"])
    val = pd.read_csv(os.path.join(processed_v2, "val.csv"), parse_dates=["Date"])
    print(f"train rows: {len(train)}  {train['Date'].min().date()} -> {train['Date'].max().date()}")
    print(f"val   rows: {len(val)}  {val['Date'].min().date()} -> {val['Date'].max().date()}")

    forbidden = {"SNo", "Name", "Symbol", "Date", target, "Close", "Close_Next"}
    assert not (set(features) & forbidden), f"leaky features: {set(features) & forbidden}"

    X_train = train[features]
    y_train = train[target].to_numpy(dtype=float)
    X_val = val[features]

    section("3. VALIDATION: candidate selection (by price MAE)")
    results = {}
    # baseline: persistence predicts zero return -> price unchanged
    base = evaluate_split(val, features, target, np.zeros(len(val)))
    results["persistence_baseline"] = {"type": "baseline", "params": {}, "metrics": base["price"],
                                       "return_metrics": base["return"]}
    print(f"persistence_baseline val MAE={base['price']['MAE']:.2f}  DirAcc={base['price']['DirectionalAccuracy']:.3f}")

    fitted = {}
    for name, params in XGB_CANDIDATES.items():
        model = make_model(params)
        model.fit(X_train, y_train)
        pred_ret = model.predict(X_val)
        assert np.all(np.isfinite(pred_ret)), f"{name}: non-finite predictions"
        res = evaluate_split(val, features, target, pred_ret)
        results[name] = {
            "type": "model",
            "params": params,
            "metrics": res["price"],
            "return_metrics": res["return"],
            "importances": {f: float(v) for f, v in zip(features, model.feature_importances_)},
        }
        fitted[name] = model
        print(f"  {name:<12} val MAE={res['price']['MAE']:>10.2f}  RMSE={res['price']['RMSE']:>10.2f}  "
              f"R2={res['price']['R2']:>7.4f}  DirAcc={res['price']['DirectionalAccuracy']:.3f}")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["metrics"]["MAE"])
    best_name = ranked[0][0]
    best_model_name = min(fitted, key=lambda k: results[k]["metrics"]["MAE"])
    print(f"\nbest by val MAE: {best_name}")
    print(f"selected XGBoost config: {best_model_name}")

    section("4. SAVE MODEL + VALIDATION REPORT")
    model_path = os.path.join(models_v2, "xgboost_v2_logreturn.joblib")
    safe_write(model_path, args.force)
    joblib.dump(fitted[best_model_name], model_path)
    print(f"saved {model_path}")

    report = {
        "phase": "5A",
        "target": target,
        "target_definition": meta["target_definition"],
        "features": features,
        "excluded_from_features": sorted(forbidden),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_used": False,
        "train_range": [str(train["Date"].min().date()), str(train["Date"].max().date())],
        "val_range": [str(val["Date"].min().date()), str(val["Date"].max().date())],
        "selection_metric": "val price MAE",
        "selected_model": best_model_name,
        "best_by_val_mae": best_name,
        "results": results,
        "saved_model": os.path.relpath(model_path, here),
    }
    report_path = os.path.join(reports_v2, "validation_report_v2.json")
    safe_write(report_path, args.force)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved {report_path}")

    section("SUMMARY")
    print(f"candidates: {list(results)}")
    print(f"selected: {best_model_name}  (best by val MAE: {best_name})")
    print("Test set untouched. No test-based tuning.")


if __name__ == "__main__":
    main()
