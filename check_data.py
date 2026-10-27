"""Phase 1: Inspect the Bitcoin dataset. Read-only, no models, no writes."""

import hashlib
import os
import sys

import numpy as np
import pandas as pd

CANDIDATES = ["coin_Bitcoin .csv", "coin_Bitcoin.csv", "data"]


def locate_dataset():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in CANDIDATES:
        path = os.path.join(here, name)
        if os.path.isfile(path):
            return path, name
    sys.exit("No dataset file found.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    path, name = locate_dataset()
    section("1. DATASET LOCATION")
    print(f"file      : {path}")
    print(f"size      : {os.path.getsize(path):,} bytes")
    print(f"sha256    : {sha256(path)}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    section("2. STRUCTURE")
    print(f"rows      : {len(df)}")
    print(f"columns   : {list(df.columns)}")
    print("\ndtypes:")
    print(df.dtypes.to_string())
    print("\nhead:")
    print(df.head(3).to_string())
    print("\ntail:")
    print(df.tail(3).to_string())

    section("3. MISSING VALUES / DUPLICATES / CONSTANTS")
    miss = df.isna().sum()
    print("missing per column:")
    print(miss.to_string())
    print(f"total missing cells: {int(miss.sum())}")
    print(f"fully duplicated rows: {int(df.duplicated().sum())}")
    print(f"duplicate dates: {int(df['Date'].duplicated().sum())}")
    print(f"duplicate SNo: {int(df['SNo'].duplicated().sum())}")
    for col in ["Name", "Symbol"]:
        print(f"{col} unique values: {df[col].unique().tolist()}")

    section("4. DATE RANGE")
    print(f"min date  : {df['Date'].min()}")
    print(f"max date  : {df['Date'].max()}")
    print(f"span      : {(df['Date'].max() - df['Date'].min()).days} days")

    section("5. PRICE / VOLUME VALIDITY (OHLCV)")
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        s = df[col]
        print(
            f"{col:<6} min={s.min():>16,.4f} max={s.max():>16,.4f} "
            f"<=0:{int((s <= 0).sum())} NaN:{int(s.isna().sum())}"
        )
    bad_hl = int((df["High"] < df["Low"]).sum())
    bad_range = int(
        (
            (df["Close"] > df["High"])
            | (df["Close"] < df["Low"])
            | (df["Open"] > df["High"])
            | (df["Open"] < df["Low"])
        ).sum()
    )
    neg_vol = int((df["Volume"] < 0).sum())
    zero_vol = int((df["Volume"] == 0).sum())
    print(f"High < Low            : {bad_hl}")
    print(f"Open/Close outside H/L: {bad_range}")
    print(f"Volume < 0            : {neg_vol}   Volume == 0 : {zero_vol}")
    print("\nzero-volume rows per calendar year:")
    print(df.loc[df["Volume"] == 0, "Date"].dt.year.value_counts().sort_index().to_string())

    section("6. DAILY & CHRONOLOGICAL COMPLETENESS")
    ordered = df["Date"].is_monotonic_increasing
    strictly = df["Date"].is_monotonic_increasing and not df["Date"].duplicated().any()
    print(f"chronologically sorted (increasing): {ordered}")
    print(f"strictly increasing & unique dates : {strictly}")

    full = pd.date_range(df["Date"].min().normalize(), df["Date"].max().normalize(), freq="D")
    present = set(df["Date"].dt.normalize())
    missing = [d for d in full if d not in present]
    print(f"expected calendar days: {len(full)}")
    print(f"days present          : {len(present)}")
    print(f"missing days          : {len(missing)}")
    if missing:
        print("first 10 missing days:")
        for d in missing[:10]:
            print("  ", d.date())
        print("missing days per year:")
        print(pd.Series([d.year for d in missing]).value_counts().sort_index().to_string())

    grp = df["Date"].dt.normalize().value_counts()
    print(f"days with >1 row: {int((grp > 1).sum())}")
    gaps = df["Date"].diff().dropna()
    print(f"most common gap: {gaps.mode().iloc[0] if not gaps.empty else 'n/a'}")
    print(f"gaps > 1 day   : {int((gaps > pd.Timedelta(days=1)).sum())}")

    section("7. SUFFICIENCY FOR NEXT-DAY CLOSE PREDICTION")
    years = (df["Date"].max() - df["Date"].min()).days / 365.25
    print(f"observations : {len(df)}")
    print(f"period       : {years:.2f} years")
    print(f"target col   : Close (next-day)")
    print("verdict      : inspect above; see report.")
    print("\nNULL-feature caveat: raw OHLCV+Marketcap only (feature engineering needed later).")


if __name__ == "__main__":
    main()
