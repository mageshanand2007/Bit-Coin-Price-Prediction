"""Phase 3: Train and validate candidate models for Bitcoin next-day Close prediction.

- Trains ONLY on processed/train.csv.
- Compares models on processed/val.csv (does NOT read/evaluate test.csv).
- Uses exactly the feature columns documented in processed/feature_metadata.json.
- Saves candidate models to ./models/ and a validation report to ./reports/.
- No frontend/backend, no final-accuracy claims.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

RANDOM_STATE = 42
TARGET = "Target_Close_Next"
CURRENT_CLOSE = "Current_Close"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred, current_close):
    """MAE, RMSE, R2 and directional accuracy (relative to current Close)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    current_close = np.asarray(current_close, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    actual_dir = np.sign(y_true - current_close)
    pred_dir = np.sign(y_pred - current_close)
    directional_accuracy = float(np.mean(actual_dir == pred_dir))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "DirectionalAccuracy": directional_accuracy,
    }


# ---------------------------------------------------------------- data
def load_metadata(processed_dir):
    with open(os.path.join(processed_dir, "feature_metadata.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return meta


def load_splits(processed_dir, features, target):
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"), parse_dates=["Date"])
    val = pd.read_csv(os.path.join(processed_dir, "val.csv"), parse_dates=["Date"])

    for name, df in (("train", train), ("val", val)):
        missing = [c for c in features + [target] if c not in df.columns]
        assert not missing, f"{name}.csv missing columns: {missing}"

    # Reconstruct the CURRENT day's Close (needed for the persistence baseline
    # and for directional accuracy) without touching test.csv.
    #   Current_Close[d] = Target_Close_Next[d-1]  (previous row's target)
    combined = pd.concat([train, val], ignore_index=True)
    combined[CURRENT_CLOSE] = combined[target].shift(1)

    # Cross-check against an independent feature: Close_Lag_1 shifted back one row
    # equals the current day's Close. (Last row has no successor -> NaN, ignored.)
    cross = combined["Close_Lag_1"].shift(-1)
    mask = cross.notna() & combined[CURRENT_CLOSE].notna()
    assert np.allclose(combined.loc[mask, CURRENT_CLOSE], cross[mask]), "Current_Close cross-check failed"

    n_train = len(train)
    train = combined.iloc[:n_train].reset_index(drop=True)
    val = combined.iloc[n_train:].reset_index(drop=True)

    # val's first Current_Close comes from train's last target -> must be present.
    assert val[CURRENT_CLOSE].notna().all(), "val Current_Close has gaps"
    return train, val


# ---------------------------------------------------------------- models
def build_models():
    models = {}
    models["linear_regression"] = Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression())]
    )
    models["random_forest"] = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    if HAS_XGB:
        # Small, fixed grid: select by validation MAE only (no test use).
        models["xgboost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
            objective="reg:squarederror", tree_method="hist",
        )
        models["xgboost_shallow"] = XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=2,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
            objective="reg:squarederror", tree_method="hist",
        )
    return models


def feature_importances(model, features):
    est = model
    if isinstance(model, Pipeline):
        est = model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_).ravel()
    else:
        return None
    order = np.argsort(imp)[::-1]
    return {features[i]: float(imp[i]) for i in order}


def safe_write(path, force):
    if os.path.exists(path) and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --force)")
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 model training + validation")
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
    print(f"features ({len(features)}): {features}")
    print(f"target    : {target}")
    assert target == TARGET, f"unexpected target {target}"

    section("2. LOAD SPLITS (train + val only)")
    train, val = load_splits(processed_dir, features, target)
    print(f"train rows: {len(train)}  {train['Date'].min().date()} -> {train['Date'].max().date()}")
    print(f"val   rows: {len(val)}  {val['Date'].min().date()} -> {val['Date'].max().date()}")
    print(f"test.csv NOT loaded (reserved for Phase 4)")

    # No leakage: identifiers/date/target are never inputs.
    forbidden = {"SNo", "Name", "Symbol", "Date", target, CURRENT_CLOSE}
    assert not (set(features) & forbidden), f"leaky feature columns: {set(features) & forbidden}"
    print("feature guard: no id/date/target columns in inputs -> OK")

    X_train, y_train = train[features], train[target].to_numpy()
    X_val, y_val = val[features], val[target].to_numpy()
    current_close_val = val[CURRENT_CLOSE].to_numpy()

    section("3. TRAIN + VALIDATE")
    results = {}

    # Naive persistence baseline: predict next Close = current Close.
    baseline_pred = current_close_val.copy()
    results["persistence_baseline"] = {
        "type": "baseline",
        "metrics": regression_metrics(y_val, baseline_pred, current_close_val),
    }
    print(f"persistence_baseline MAE={results['persistence_baseline']['metrics']['MAE']:.2f}")

    models = build_models()
    fitted = {}
    for name, model in models.items():
        print(f"\ntraining {name} ...")
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_val), dtype=float)

        assert pred.shape[0] == len(y_val), f"{name}: prediction length mismatch"
        assert np.all(np.isfinite(pred)), f"{name}: non-finite predictions"

        m = regression_metrics(y_val, pred, current_close_val)
        results[name] = {
            "type": "model",
            "params": {k: v for k, v in model.get_params().items()}
            if not isinstance(model, Pipeline)
            else {"pipeline": "StandardScaler+LinearRegression"},
            "metrics": m,
            "importances": feature_importances(model, features),
            "predictions_finite": True,
        }
        fitted[name] = model
        print(
            f"  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
            f"R2={m['R2']:.4f}  DirAcc={m['DirectionalAccuracy']:.4f}"
        )

    # Hyperparameter selection among xgboost variants by validation MAE.
    xgb_variants = [k for k in fitted if k.startswith("xgboost")]
    best_xgb = min(xgb_variants, key=lambda k: results[k]["metrics"]["MAE"]) if xgb_variants else None

    section("4. VALIDATION COMPARISON (sorted by MAE)")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["metrics"]["MAE"])
    for name, info in ranked:
        m = info["metrics"]
        print(
            f"  {name:<22} MAE={m['MAE']:>10.2f}  RMSE={m['RMSE']:>10.2f}  "
            f"R2={m['R2']:>7.4f}  DirAcc={m['DirectionalAccuracy']:>6.3f}"
        )
    best_name = ranked[0][0]
    print(f"\nbest model by validation MAE: {best_name}")
    print(f"best xgboost variant         : {best_xgb}")

    section("5. SAVE MODELS + REPORT")
    saved_models = []
    for name, model in fitted.items():
        path = os.path.join(models_dir, f"{name}.joblib")
        safe_write(path, args.force)
        joblib.dump(model, path)
        saved_models.append(os.path.relpath(path, here))
        print(f"saved {path}")

    report = {
        "phase": 3,
        "random_state": RANDOM_STATE,
        "features": features,
        "target": target,
        "excluded_from_features": list(forbidden),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_used": False,
        "train_range": [str(train["Date"].min().date()), str(train["Date"].max().date())],
        "val_range": [str(val["Date"].min().date()), str(val["Date"].max().date())],
        "baseline_note": (
            "Persistence baseline predicts next Close = current Close; its directional "
            "accuracy is ~0 because it always predicts 'no change' (sign 0)."
        ),
        "results": results,
        "best_model_by_val_mae": best_name,
        "best_xgboost_variant": best_xgb,
        "saved_models": saved_models,
    }
    report_path = os.path.join(reports_dir, "validation_report.json")
    safe_write(report_path, args.force)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved {report_path}")

    section("SUMMARY")
    print(f"models evaluated : {list(results)}")
    print(f"best by val MAE  : {best_name}")
    print("No test-set evaluation performed. No final accuracy claimed.")


if __name__ == "__main__":
    main()
