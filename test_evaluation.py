"""Phase 4 tests: evaluation logic, feature order, target alignment, metrics, finite preds.

Run with:  python test_evaluation.py     (also pytest-compatible)
"""

import os

import numpy as np
import pandas as pd

import test_evaluate as te
import train_models as tm

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "processed")
MODELS = os.path.join(HERE, "models")


def _setup():
    meta = te.load_metadata(PROCESSED)
    features, target = list(meta["features"]), meta["target"]
    test = te.load_test_with_current_close(PROCESSED, target)
    models = te.load_saved_models(MODELS)
    return meta, features, target, test, models


def test_metrics_known_values():
    y_true = np.array([100.0, 110.0, 90.0, 120.0])
    y_pred = np.array([102.0, 108.0, 95.0, 118.0])
    current = np.array([99.0, 105.0, 95.0, 110.0])
    m = tm.regression_metrics(y_true, y_pred, current)
    assert np.isclose(m["MAE"], 2.75)
    assert np.isclose(m["RMSE"], np.sqrt(9.25))
    assert np.isclose(m["DirectionalAccuracy"], 0.75)


def test_feature_contract_and_order():
    meta, features, target, test, _ = _setup()
    assert features == meta["features"]
    assert list(test[features].columns) == features
    assert target == "Target_Close_Next"
    for c in features + [target]:
        assert c in test.columns


def test_no_leaky_columns():
    _, features, target, _, _ = _setup()
    forbidden = {"SNo", "Name", "Symbol", "Date", target, "Current_Close"}
    assert not (set(features) & forbidden)


def test_current_close_alignment():
    _, _, target, test, _ = _setup()
    # Target at row d equals Current_Close at row d+1 within the test split.
    assert np.allclose(test[target].to_numpy()[:-1], test["Current_Close"].to_numpy()[1:])
    assert test["Current_Close"].notna().all()


def test_baseline_predicts_current_close():
    _, features, target, test, models = _setup()
    results, preds = te.evaluate(models, test, features, target)
    assert np.allclose(preds["persistence_baseline"], test["Current_Close"].to_numpy())
    assert results["persistence_baseline"]["metrics"]["DirectionalAccuracy"] == 0.0


def test_all_predictions_finite_and_aligned():
    _, features, target, test, models = _setup()
    results, preds = te.evaluate(models, test, features, target)
    for name, pred in preds.items():
        assert pred.shape[0] == len(test), f"{name} length mismatch"
        assert np.all(np.isfinite(pred)), f"{name} non-finite"
    assert set(results) == {"persistence_baseline", "linear_regression",
                            "random_forest", "xgboost", "xgboost_shallow"}


def test_evaluate_metrics_match_direct_computation():
    _, features, target, test, models = _setup()
    results, preds = te.evaluate(models, test, features, target)
    y = test[target].to_numpy(float)
    current = test["Current_Close"].to_numpy(float)
    for name, pred in preds.items():
        expected = tm.regression_metrics(y, pred, current)
        got = results[name]["metrics"]
        for k in ("MAE", "RMSE", "R2", "DirectionalAccuracy"):
            assert np.isclose(got[k], expected[k]), f"{name}:{k} mismatch"


def test_beats_baseline_flag_consistent():
    _, features, target, test, models = _setup()
    results, _ = te.evaluate(models, test, features, target)
    base = results["persistence_baseline"]["metrics"]["MAE"]
    for name, info in results.items():
        assert info["beats_baseline_mae"] == (info["metrics"]["MAE"] < base)


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
