"""Phase 5A preprocessing: causal technical indicators + next-day log-return target.

Outputs to processed_v2/ only (never touches Phase 1-4 artifacts).
Target = log(Close[t+1] / Close[t]); predictions are converted back to prices later.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

DATE_COL = "Date"
TARGET = "Log_Return_Next"
HELPER_COLS = ["Close", "Close_Next"]
ID_COLS = ["SNo", "Name", "Symbol", "Symbol"]
CANDIDATE_FILES = ["coin_Bitcoin .csv", "coin_Bitcoin.csv"]

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_7", "ret_14",
    "rsi_14",
    "macd_rel", "macd_signal_rel", "macd_hist_rel",
    "close_over_sma7", "sma7_over_sma30", "sma14_over_sma30",
    "bb_pctb", "bb_bandwidth",
    "volatility_7", "volatility_14", "volatility_30",
    "log_volume_rel_7", "log_volume_rel_30", "volume_ma_ratio",
    "Volume_Zero_Flag",
    "hl_range_rel", "close_pos_in_range",
]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_write(path, force):
    if os.path.exists(path) and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --force)")
    os.makedirs(os.path.dirname(path), exist_ok=True)


def locate_source(explicit=None):
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ([explicit] if explicit else CANDIDATE_FILES):
        path = os.path.join(here, name)
        if os.path.isfile(path):
            return path
    sys.exit("No source dataset found.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features_v2(df):
    """Causal features + next-day log-return target. Row t uses info <= day t only."""
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]

    logret = np.log(close / close.shift(1))
    out["ret_1"] = logret
    out["ret_3"] = np.log(close / close.shift(3))
    out["ret_7"] = np.log(close / close.shift(7))
    out["ret_14"] = np.log(close / close.shift(14))

    out["rsi_14"] = rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_rel"] = macd / close
    out["macd_signal_rel"] = signal / close
    out["macd_hist_rel"] = (macd - signal) / close

    sma7 = close.rolling(7, min_periods=7).mean()
    sma14 = close.rolling(14, min_periods=14).mean()
    sma20 = close.rolling(20, min_periods=20).mean()
    sma30 = close.rolling(30, min_periods=30).mean()
    out["close_over_sma7"] = close / sma7 - 1
    out["sma7_over_sma30"] = sma7 / sma30 - 1
    out["sma14_over_sma30"] = sma14 / sma30 - 1

    sd20 = close.rolling(20, min_periods=20).std()
    upper = sma20 + 2 * sd20
    lower = sma20 - 2 * sd20
    out["bb_pctb"] = (close - lower) / (upper - lower)
    out["bb_bandwidth"] = (upper - lower) / sma20

    out["volatility_7"] = logret.rolling(7, min_periods=7).std()
    out["volatility_14"] = logret.rolling(14, min_periods=14).std()
    out["volatility_30"] = logret.rolling(30, min_periods=30).std()

    logv = np.log1p(volume)
    lv7 = logv.rolling(7, min_periods=7).mean()
    lv30 = logv.rolling(30, min_periods=30).mean()
    out["log_volume_rel_7"] = logv - lv7
    out["log_volume_rel_30"] = logv - lv30
    vol_ma7 = volume.rolling(7, min_periods=7).mean()
    vol_ma30 = volume.rolling(30, min_periods=30).mean()
    out["volume_ma_ratio"] = vol_ma7 / vol_ma30.replace(0, np.nan)
    out["Volume_Zero_Flag"] = (volume == 0).astype(int)

    rng = (high - low)
    out["hl_range_rel"] = rng / close
    out["close_pos_in_range"] = (close - low) / rng.replace(0, np.nan)

    out["Close_Next"] = close.shift(-1)
    out[TARGET] = np.log(close.shift(-1) / close)
    return out


def load_phase2_boundaries(processed_dir):
    with open(os.path.join(processed_dir, "feature_metadata.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    s = meta["splits"]
    return {
        "train_end": s["train"]["end"],
        "val_start": s["val"]["start"],
        "val_end": s["val"]["end"],
        "test_start": s["test"]["start"],
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 5A preprocessing v2")
    parser.add_argument("--source", default=None)
    parser.add_argument("--processed", default="processed")
    parser.add_argument("--outdir", default="processed_v2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, args.outdir)

    src = locate_source(args.source)
    section("1. SOURCE (read-only)")
    print(f"file   : {src}")
    print(f"sha256 : {sha256(src)}")

    raw = pd.read_csv(src)
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
    raw = raw.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    print(f"raw rows: {len(raw)}  {raw[DATE_COL].min().date()} -> {raw[DATE_COL].max().date()}")

    section("2. BUILD CAUSAL FEATURES")
    feat = build_features_v2(raw)
    print(f"feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    section("3. TARGET ALIGNMENT CHECK")
    expected = np.log(feat["Close_Next"] / feat["Close"])
    mask = feat[TARGET].notna()
    assert np.allclose(feat.loc[mask, TARGET], expected[mask]), "target misaligned"
    print("Log_Return_Next == log(Close_Next/Close) -> OK")

    section("4. DROP WARM-UP / TAIL NaN")
    keep = [DATE_COL] + HELPER_COLS + FEATURE_COLUMNS + [TARGET]
    clean = feat.dropna(subset=FEATURE_COLUMNS + [TARGET]).reset_index(drop=True)
    print(f"rows before: {len(feat)}  after: {len(clean)}  removed: {len(feat) - len(clean)}")
    print(f"remaining NaN: {int(clean[FEATURE_COLUMNS + [TARGET]].isna().sum().sum())}")

    section("5. CHRONOLOGICAL SPLITS (reuse Phase 2 date boundaries)")
    b = load_phase2_boundaries(os.path.join(here, args.processed))
    print(f"boundaries: {b}")
    date = clean[DATE_COL].dt.normalize()
    train_end = pd.Timestamp(b["train_end"]).normalize()
    val_start = pd.Timestamp(b["val_start"]).normalize()
    val_end = pd.Timestamp(b["val_end"]).normalize()
    test_start = pd.Timestamp(b["test_start"]).normalize()
    train = clean[date <= train_end].reset_index(drop=True)
    val = clean[(date >= val_start) & (date <= val_end)].reset_index(drop=True)
    test = clean[date >= test_start].reset_index(drop=True)

    splits = {"train": train, "val": val, "test": test}
    for name, part in splits.items():
        print(
            f"{name:<5} rows={len(part):>5}  {part[DATE_COL].min().date()} -> {part[DATE_COL].max().date()}  "
            f"close[{part['Close'].min():.2f},{part['Close'].max():.2f}]"
        )
    assert train[DATE_COL].max() < val[DATE_COL].min() < val[DATE_COL].max() < test[DATE_COL].min()
    print("split ordering: PASS (chronological, no overlap)")

    section("6. SAVE OUTPUTS")
    full_path = os.path.join(outdir, "features_full_v2.csv")
    safe_write(full_path, args.force)
    clean[keep].to_csv(full_path, index=False)
    print(f"saved {full_path}")

    for name, part in splits.items():
        p = os.path.join(outdir, f"{name}.csv")
        safe_write(p, args.force)
        part[keep].to_csv(p, index=False)
        print(f"saved {p}")

    meta = {
        "phase": "5A",
        "source_file": os.path.basename(src),
        "source_sha256": sha256(src),
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(clean)),
        "features": FEATURE_COLUMNS,
        "target": TARGET,
        "target_definition": "log(Close[t+1] / Close[t]); convert via Close[t]*exp(pred)",
        "helper_columns": HELPER_COLS,
        "excluded_from_features": ID_COLS + [DATE_COL, TARGET, "Close", "Close_Next"],
        "split_boundaries": b,
        "split_rows": {k: int(len(v)) for k, v in splits.items()},
        "notes": [
            "All indicators are causal (use only data up to the prediction day).",
            "Features are relative/stationary to avoid price-level extrapolation.",
            "Volume zero days (2013) flagged via Volume_Zero_Flag; no imputation.",
        ],
    }
    meta_path = os.path.join(outdir, "feature_metadata_v2.json")
    safe_write(meta_path, args.force)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"saved {meta_path}")

    section("SUMMARY")
    print(f"clean rows: {len(clean)}  features: {len(FEATURE_COLUMNS)}  target: {TARGET}")
    print(f"train/val/test: {len(train)}/{len(val)}/{len(test)}")


if __name__ == "__main__":
    main()
