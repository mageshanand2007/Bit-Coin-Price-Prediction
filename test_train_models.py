"""Phase 3 tests: feature consistency, target alignment, metrics, finite predictions.

Run with:  python test_train_models.py     (also pytest-compatible)
"""

import os

import numpy as np
import pandas as pd

import train_models as tm

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "processed")


def _load():
    meta = tm.load_metadata(PROCESSED)
    return meta, meta["features"], meta["target"]


def test_metrics_known_values():
    y_true = np.array([100.0, 110.0, 90.0, 120.0])
    y_pred = np.array([102.0, 108.0, 95.0, 118.0])
    current = np.array([99.0, 105.0, 95.0, 110.0])
    m = tm.regression_metrics(y_true, y_pred, current)

    assert np.isclose(m["MAE"], np.mean([2, 2, 5, 2]))
    assert np.isclose(m["RMSE"], np.sqrt(np.mean([4, 4, 25, 4])))
    # actual dirs: +,+,-,+ ; pred dirs: +,+,0,+ -> 3 of 4 match
    assert np.isclose(m["DirectionalAccuracy"], 0.75)


def test_directional_accuracy_known():
    y_true = np.array([110.0, 90.0])
    y_pred = np.array([90.0, 110.0])
    current = np.array([100.0, 100.0])
    m = tm.regression_metrics(y_true, y_pred, current)
    assert np.isclose(m["DirectionalAccuracy"], 0.0)


def test_feature_consistency():
    meta, features, target = _load()
    train = pd.read_csv(os.path.join(PROCESSED, "train.csv"))
    val = pd.read_csv(os.path.join(PROCESSED, "val.csv"))
    for df in (train, val):
        assert list(features) == meta["features"]
        for c in features + [target]:
            assert c in df.columns, f"missing {c}"
    assert target == "Target_Close_Next"


def test_no_leaky_columns():
    _, features, target = _load()
    forbidden = {"SNo", "Name", "Symbol", "Date", target, "Current_Close"}
    assert not (set(features) & forbidden)


def test_target_alignment_and_current_close():
    meta, features, target = _load()
    train, val = tm.load_splits(PROCESSED, features, target)
    combined = pd.concat([train, val], ignore_index=True)

    # Current_Close[d] must equal the next row's Close (via Close_Lag_1 shift).
    cross = combined["Close_Lag_1"].shift(-1)
    mask = cross.notna() & combined["Current_Close"].notna()
    assert np.allclose(combined.loc[mask, "Current_Close"], cross[mask])

    # Target at row d must equal Current_Close at row d+1.
    assert np.allclose(
        combined[target].to_numpy()[:-1],
        combined["Current_Close"].to_numpy()[1:],
    )
    assert val["Current_Close"].notna().all()


def test_finite_predictions_linear_regression():
    meta, features, target = _load()
    train, val = tm.load_splits(PROCESSED, features, target)
    model = tm.build_models()["linear_regression"]
    model.fit(train[features], train[target].to_numpy())
    pred = model.predict(val[features])
    assert pred.shape[0] == len(val)
    assert np.all(np.isfinite(pred))


def test_split_row_counts_match_metadata():
    meta, features, target = _load()
    train, val = tm.load_splits(PROCESSED, features, target)
    assert len(train) == meta["splits"]["train"]["rows"]
    assert len(val) == meta["splits"]["val"]["rows"]


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
