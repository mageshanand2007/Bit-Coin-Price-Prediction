"""Phase 5B tests: sequence construction, alignment, splits, scaler leakage, metrics.

Run with:  python test_lstm_v2.py     (also pytest-compatible)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

import lstm_v2 as L
import train_models as tm

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_V2 = os.path.join(HERE, "processed_v2")
MODELS = os.path.join(HERE, "models_v2", "lstm")


def _data():
    frames, full, split_label = L.load_v2(PROCESSED_V2)
    with open(os.path.join(PROCESSED_V2, "feature_metadata_v2.json"), encoding="utf-8") as fh:
        features = json.load(fh)["features"]
    return frames, full, split_label, features


def test_make_windows_shapes_and_no_future():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(20, 3))
    y = rng.normal(size=20)
    seq = 5
    X, ys, ends = L.make_windows(F, y, seq)
    assert X.shape == (20 - seq + 1, seq, 3)
    assert len(ys) == len(ends)
    for k, i in enumerate(ends):
        # window contains exactly rows i-seq+1..i (only past/current, never future)
        assert np.allclose(X[k], F[i - seq + 1:i + 1])
        assert np.isclose(ys[k], y[i])

    # perturbing future rows must not change windows ending at/before a probe index
    probe = 10
    F2 = F.copy()
    F2[probe + 1:] += 100.0
    X2, _, ends2 = L.make_windows(F2, y, seq)
    for k, i in enumerate(ends2):
        if i <= probe:
            assert np.allclose(X[k], X2[k])


def test_target_alignment_real_data():
    frames, full, split_label, features = _data()
    F = full[features].to_numpy(float)
    y = full[L.TARGET].to_numpy(float)
    X, ys, ends = L.make_windows(F, y, L.SEQ_LEN)
    assert np.allclose(ys, y[ends])
    # last step of each window equals the feature row at the target end index
    for k in (0, 100, len(ends) - 1):
        assert np.allclose(X[k, -1, :], F[ends[k]])


def test_chronological_sequence_splits():
    frames, full, split_label, features = _data()
    F = full[features].to_numpy(float)
    y = full[L.TARGET].to_numpy(float)
    _, _, ends = L.make_windows(F, y, L.SEQ_LEN)
    end_split = split_label[ends]
    tr_end = ends[end_split == "train"]
    va_end = ends[end_split == "val"]
    te_end = ends[end_split == "test"]
    assert tr_end.max() < va_end.min()
    assert va_end.max() < te_end.min()
    # every val/test row is predicted (comparable to XGBoost/baseline)
    assert int((end_split == "val").sum()) == len(frames["val"])
    assert int((end_split == "test").sum()) == len(frames["test"])


def test_scaler_fit_on_train_only():
    frames, full, _, features = _data()
    train_feat = frames["train"][features].to_numpy(float)
    full_feat = full[features].to_numpy(float)
    scaler = L.fit_scaler(train_feat)
    assert np.allclose(scaler.mean_, train_feat.mean(axis=0))
    # must NOT match statistics of the full series (would indicate leakage)
    assert not np.allclose(scaler.mean_, full_feat.mean(axis=0))
    # transform uses train stats
    z = scaler.transform(train_feat)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-6)


def test_saved_model_finite_predictions():
    frames, full, split_label, features = _data()
    with open(os.path.join(MODELS, "lstm_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    scaler = joblib.load(os.path.join(MODELS, "lstm_scaler.joblib"))
    F = scaler.transform(full[features].to_numpy(float))
    y = full[L.TARGET].to_numpy(float)
    X, _, ends = L.make_windows(F, y, cfg["seq_len"])
    model = L.LSTMRegressor(len(features), hidden=cfg["params"]["hidden"],
                            layers=cfg["params"]["layers"], dropout=cfg["params"]["dropout"])
    model.load_state_dict(torch.load(os.path.join(MODELS, "lstm_logreturn.pt"), map_location="cpu"))
    pred = L.predict_return(model, X[:64])
    assert pred.shape[0] == 64
    assert np.all(np.isfinite(pred))


def test_price_conversion_identity():
    close = np.array([100.0, 200.0])
    pred_ret = np.array([0.1, -0.05])
    assert np.allclose(close * np.exp(pred_ret), [100 * np.exp(0.1), 200 * np.exp(-0.05)])


def test_metrics_known_values():
    y_true = np.array([100.0, 110.0, 90.0, 120.0])
    y_pred = np.array([102.0, 108.0, 95.0, 118.0])
    current = np.array([99.0, 105.0, 95.0, 110.0])
    m = tm.regression_metrics(y_true, y_pred, current)
    assert np.isclose(m["MAE"], 2.75)
    assert np.isclose(m["RMSE"], np.sqrt(9.25))
    assert np.isclose(m["DirectionalAccuracy"], 0.75)


def test_reproducible_seeding():
    L.set_seed(123)
    a = torch.randn(5)
    L.set_seed(123)
    b = torch.randn(5)
    assert torch.allclose(a, b)


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
