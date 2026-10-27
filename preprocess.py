"""Phase 2: Preprocessing + feature engineering for Bitcoin next-day Close prediction.

Read-only with respect to the source datasets. Writes prepared outputs to ./processed/.
No model training. All features are strictly causal (use only data <= prediction date).
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

DATE_COL = "Date"
TARGET = "Target_Close_Next"
ID_COLS = ["SNo", "Name", "Symbol"]
RAW_PRICE_COLS = ["Open", "High", "Low", "Close"]

# Candidate source filenames; note the real file has a space before ".csv".
CANDIDATE_FILES = ["coin_Bitcoin .csv", "coin_Bitcoin.csv"]

# Explicit, ordered feature contract. Excludes IDs, raw Date and the target.
FEATURE_COLUMNS = [
    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_3",
    "Close_Lag_7",
    "Close_Lag_14",
    "Daily_Return",
    "MA_7",
    "MA_14",
    "MA_30",
    "Volatility_7",
    "Volatility_14",
    "Volatility_30",
    "Volume_Lag_1",
    "Volume_Lag_7",
    "High_Low_Range",
    "Volume_Zero_Flag",
]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def locate_source(explicit=None):
    here = os.path.dirname(os.path.abspath(__file__))
    names = [explicit] if explicit else CANDIDATE_FILES
    for name in names:
        path = os.path.join(here, name)
        if os.path.isfile(path):
            return path
    sys.exit(f"No source dataset found. Looked for: {names}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path):
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[DATE_COL])
    if len(df) != before:
        print(f"[warn] dropped {before - len(df)} rows with unparsable dates")
    df = df.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)
    return df


def validate_ohlcv(df):
    issues = {
        "non_positive_price": int((df[RAW_PRICE_COLS] <= 0).sum().sum()),
        "high_lt_low": int((df["High"] < df["Low"]).sum()),
        "oc_outside_hl": int(
            (
                (df["Close"] > df["High"])
                | (df["Close"] < df["Low"])
                | (df["Open"] > df["High"])
                | (df["Open"] < df["Low"])
            ).sum()
        ),
        "negative_volume": int((df["Volume"] < 0).sum()),
        "zero_volume": int((df["Volume"] == 0).sum()),
        "duplicate_dates": int(df[DATE_COL].duplicated().sum()),
    }
    print("OHLCV validation:")
    for k, v in issues.items():
        print(f"  {k:<20}: {v}")
    return issues


def build_features(df):
    """Return a new frame with causal features + next-day Close target.

    Row d represents prediction time = end of day d. Every feature uses only
    values available on or before day d. The target is Close on day d+1.
    """
    out = df.copy()

    # Lagged Close (strictly past values; lag_k = Close of d-k)
    for k in (1, 2, 3, 7, 14):
        out[f"Close_Lag_{k}"] = out["Close"].shift(k)

    # Daily return of the current day vs previous day (available end of day d)
    out["Daily_Return"] = out["Close"].pct_change()

    # Moving averages of Close over trailing windows ending at day d
    for w in (7, 14, 30):
        out[f"MA_{w}"] = out["Close"].rolling(window=w, min_periods=w).mean()

    # Rolling volatility of daily returns (trailing window ending at day d)
    for w in (7, 14, 30):
        out[f"Volatility_{w}"] = out["Daily_Return"].rolling(window=w, min_periods=w).std()

    # Lagged Volume (yesterday / a week ago) - no imputation of zero-volume days
    out["Volume_Lag_1"] = out["Volume"].shift(1)
    out["Volume_Lag_7"] = out["Volume"].shift(7)

    # Same-day High-Low range
    out["High_Low_Range"] = out["High"] - out["Low"]

    # Flag zero-volume days instead of fabricating/ imputing volume
    out["Volume_Zero_Flag"] = (out["Volume"] == 0).astype(int)

    # Target: next day's Close
    out[TARGET] = out["Close"].shift(-1)
    return out


def check_target_alignment(featured):
    """Target at row i must equal Close at row i+1 in the same sorted frame."""
    expected = featured["Close"].shift(-1)
    valid = featured.index[:-1]
    mismatches = int((featured.loc[valid, TARGET] != expected.loc[valid]).sum())
    print(f"target alignment mismatches: {mismatches}")
    return mismatches == 0


def causality_probe(raw_sorted, index):
    """Perturb all data AFTER `index`; features at rows <= index must be unchanged.

    Generic look-ahead leakage test that does not duplicate feature logic.
    """
    baseline = build_features(raw_sorted)
    perturbed = raw_sorted.copy()
    for col in ["Open", "High", "Low", "Close", "Volume", "Marketcap"]:
        perturbed.loc[index + 1:, col] = perturbed.loc[index + 1:, col] * 7.0 + 12345.0
    perturbed_feat = build_features(perturbed)

    cols = [c for c in FEATURE_COLUMNS if c in baseline.columns]
    left = baseline.loc[:index, cols]
    right = perturbed_feat.loc[:index, cols]
    same = left.equals(right)
    return same


def run_causality_checks(raw_sorted):
    n = len(raw_sorted)
    probes = sorted({n // 4, n // 2, (3 * n) // 4, n - 2})
    ok = True
    print("causality / look-ahead probes (perturb future rows):")
    for i in probes:
        res = causality_probe(raw_sorted, i)
        ok = ok and res
        print(f"  index {i:>5} ({raw_sorted.loc[i, DATE_COL].date()}): {'PASS' if res else 'FAIL'}")
    return ok


def split_chronological(clean, train_frac, val_frac):
    n = len(clean)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return {
        "train": clean.iloc[:train_end].reset_index(drop=True),
        "val": clean.iloc[train_end:val_end].reset_index(drop=True),
        "test": clean.iloc[val_end:].reset_index(drop=True),
    }


def main():
    parser = argparse.ArgumentParser(description="Bitcoin preprocessing + feature engineering (Phase 2)")
    parser.add_argument("--source", default=None, help="explicit source CSV path/name")
    parser.add_argument("--outdir", default="processed", help="output directory")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    src = locate_source(args.source)
    section("1. SOURCE (read-only)")
    print(f"file   : {src}")
    print(f"sha256 : {sha256(src)}")

    raw = load_raw(src)
    section("2. LOAD + SORT")
    print(f"rows after sort: {len(raw)}")
    print(f"date range     : {raw[DATE_COL].min()} -> {raw[DATE_COL].max()}")
    print(f"monotonic dates: {raw[DATE_COL].is_monotonic_increasing}")

    section("3. OHLCV VALIDATION (no values fabricated)")
    issues = validate_ohlcv(raw)

    section("4. FEATURE ENGINEERING (causal)")
    featured = build_features(raw)
    print(f"feature columns ({len(FEATURE_COLUMNS)}):")
    for c in FEATURE_COLUMNS:
        print(f"  - {c}")
    print(f"target: {TARGET}")

    section("5. TARGET ALIGNMENT")
    aligned = check_target_alignment(featured)

    section("6. DROP WARM-UP / TAIL ROWS WITH NaN")
    clean = featured.dropna(subset=FEATURE_COLUMNS + [TARGET]).reset_index(drop=True)
    print(f"rows before drop: {len(featured)}")
    print(f"rows after drop : {len(clean)}")
    print(f"rows removed    : {len(featured) - len(clean)}")
    print(f"clean date range: {clean[DATE_COL].min()} -> {clean[DATE_COL].max()}")
    remaining_na = int(clean[FEATURE_COLUMNS + [TARGET]].isna().sum().sum())
    print(f"remaining NaN in feature/target matrix: {remaining_na}")

    section("7. LEAKAGE / CAUSALITY CHECKS")
    causal_ok = run_causality_checks(raw)
    print(f"\ntarget alignment ok : {aligned}")
    print(f"causality ok        : {causal_ok}")
    print(f"ID columns excluded : {ID_COLS} (not in FEATURE_COLUMNS)")
    print(f"Date excluded from model features: {'Date' not in FEATURE_COLUMNS}")

    section("8. CHRONOLOGICAL SPLITS")
    splits = split_chronological(clean, args.train_frac, args.val_frac)
    keep_cols = [DATE_COL] + FEATURE_COLUMNS + [TARGET]
    split_meta = {}
    for name, part in splits.items():
        part = part[keep_cols]
        splits[name] = part
        ordered = part[DATE_COL].is_monotonic_increasing and not part[DATE_COL].duplicated().any()
        print(
            f"{name:<5} rows={len(part):>5}  "
            f"{str(part[DATE_COL].min().date())} -> {str(part[DATE_COL].max().date())}  "
            f"ordered&unique={ordered}"
        )
        split_meta[name] = {
            "rows": len(part),
            "start": str(part[DATE_COL].min().date()),
            "end": str(part[DATE_COL].max().date()),
        }

    # Assert no temporal overlap and correct ordering between splits
    assert splits["train"][DATE_COL].max() < splits["val"][DATE_COL].min(), "train/val overlap"
    assert splits["val"][DATE_COL].max() < splits["test"][DATE_COL].min(), "val/test overlap"
    print("split ordering assertions: PASS (no overlap, chronological)")

    section("9. SAVE OUTPUTS")
    full_path = os.path.join(outdir, "bitcoin_features_full.csv")
    featured[keep_cols].to_csv(full_path, index=False)
    print(f"saved {full_path}")

    for name in ("train", "val", "test"):
        p = os.path.join(outdir, f"{name}.csv")
        splits[name].to_csv(p, index=False)
        print(f"saved {p}")

    meta = {
        "source_file": os.path.basename(src),
        "source_sha256": sha256(src),
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(clean)),
        "features": FEATURE_COLUMNS,
        "target": TARGET,
        "excluded_columns": ID_COLS + ["Date"],
        "notes": [
            "All features are causal (use only data available on/ before prediction date).",
            "Target_Close_Next = Close of the following day (row d -> Close of d+1).",
            "Zero-volume days (242, all 2013) are flagged via Volume_Zero_Flag; no imputation.",
            "Splits are chronological and non-overlapping; test period reserved for later evaluation.",
        ],
        "splits": split_meta,
    }
    meta_path = os.path.join(outdir, "feature_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"saved {meta_path}")

    section("SUMMARY")
    print(f"clean rows      : {len(clean)}")
    print(f"model features  : {len(FEATURE_COLUMNS)}")
    print(f"target          : {TARGET}")
    print(f"train/val/test  : {split_meta['train']['rows']}/{split_meta['val']['rows']}/{split_meta['test']['rows']}")
    print("done. No models trained.")


if __name__ == "__main__":
    main()
