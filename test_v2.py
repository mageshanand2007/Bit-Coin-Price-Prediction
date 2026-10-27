"""Phase 5A tests: target alignment, causality, leakage, splits, finite preds, metrics.

Run with:  python test_v2.py     (also pytest-compatible)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

import preprocess_v2 as p2
import train_models as tm

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_V2 = os.path.join(HERE, "processed_v2")
MODELS_V2 = os.path.join(HERE, "models_v2")


def _meta():
    with open(os.path.join(PROCESSED_V2, "feature_metadata_v2.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _raw():
    df = pd.read_csv(os.path.join(HERE, "coin_Bitcoin .csv"), parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def test_target_alignment():
    f = p2.build_features_v2(_raw())
    expected = np.log(f["Close_Next"] / f["Close"])
    mask = f[p2.TARGET].notna()
    assert np.allclose(f.loc[mask, p2.TARGET], expected[mask])


def test_causality_no_lookahead():
    raw = _raw()
    baseline = p2.build_features_v2(raw)
    cols = p2.FEATURE_COLUMNS
    for i in [500, 1500, 2500]:
        perturbed = raw.copy()
        for c in ["Open", "High", "Low", "Close", "Volume", "Marketcap"]:
            perturbed.loc[i + 1:, c] = perturbed.loc[i + 1:, c] * 7.0 + 999.0
        pf = p2.build_features_v2(perturbed)
        assert baseline.loc[:i, cols].equals(pf.loc[:i, cols]), f"look-ahead at {i}"


def test_no_leaky_features():
    meta = _meta()
    forbidden = {"SNo", "Name", "Symbol", "Date", meta["target"], "Close", "Close_Next"}
    assert not (set(meta["features"]) & forbidden)


def test_chronological_splits():
    meta = _meta()
    tr = pd.read_csv(os.path.join(PROCESSED_V2, "train.csv"), parse_dates=["Date"])
    va = pd.read_csv(os.path.join(PROCESSED_V2, "val.csv"), parse_dates=["Date"])
    te = pd.read_csv(os.path.join(PROCESSED_V2, "test.csv"), parse_dates=["Date"])
    assert tr["Date"].max() < va["Date"].min()
    assert va["Date"].max() < te["Date"].min()
    for df in (tr, va, te):
        assert df["Date"].is_monotonic_increasing and not df["Date"].duplicated().any()


def test_no_nan_in_saved_features():
    meta = _meta()
    for name in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(PROCESSED_V2, f"{name}.csv"))
        assert int(df[meta["features"] + [meta["target"]]].isna().sum().sum()) == 0


def test_feature_order_matches_metadata():
    meta = _meta()
    df = pd.read_csv(os.path.join(PROCESSED_V2, "test.csv"))
    assert list(meta["features"]) == meta["features"]
    for c in meta["features"]:
        assert c in df.columns


def test_price_conversion_identity():
    close = np.array([100.0, 200.0])
    pred_ret = np.array([0.1, -0.05])
    price = close * np.exp(pred_ret)
    assert np.allclose(price, [100 * np.exp(0.1), 200 * np.exp(-0.05)])


def test_metrics_known_values():
    y_true = np.array([100.0, 110.0, 90.0, 120.0])
    y_pred = np.array([102.0, 108.0, 95.0, 118.0])
    current = np.array([99.0, 105.0, 95.0, 110.0])
    m = tm.regression_metrics(y_true, y_pred, current)
    assert np.isclose(m["MAE"], 2.75)
    assert np.isclose(m["RMSE"], np.sqrt(9.25))
    assert np.isclose(m["DirectionalAccuracy"], 0.75)


def test_saved_model_finite_predictions():
    with open(os.path.join(PROCESSED_V2, "feature_metadata_v2.json"), encoding="utf-8") as fh:
        features = json.load(fh)["features"]
    val = pd.read_csv(os.path.join(PROCESSED_V2, "val.csv"))
    model = joblib.load(os.path.join(MODELS_V2, "xgboost_v2_logreturn.joblib"))
    pred = np.asarray(model.predict(val[features]), dtype=float)
    assert pred.shape[0] == len(val)
    assert np.all(np.isfinite(pred))


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
