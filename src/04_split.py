"""Module 4: join labels + features and assign train / val / test.

    python src/04_split.py

Train 2010-2018, val 2019-2020, test 2021-2025. The last GAP (=20) trading
days of train and of val are dropped: their 20-day labels look into the
next period, which would leak future information.
Output: data/panel.parquet (date, ticker, close, FEATURES, fwd_min_ret, label, split)
"""
import pandas as pd

from common import (FEATURES, FEATURES_PATH, GAP, LABELS_PATH, PANEL_PATH, TEST,
                    TRAIN, VAL)


def assign_split(dates: pd.Series) -> pd.Series:
    cal = pd.Series(sorted(dates.unique()))
    split = pd.Series("drop", index=dates.index)
    for name, (a, b) in [("train", TRAIN), ("val", VAL), ("test", TEST)]:
        in_period = cal[(cal >= a) & (cal <= b)]
        if name != "test":
            in_period = in_period.iloc[:-GAP]          # purge the gap
        split[dates.isin(in_period)] = name
    return split


def main():
    labels = pd.read_parquet(LABELS_PATH)
    feats = pd.read_parquet(FEATURES_PATH).drop(columns=["close"])
    df = labels.merge(feats, on=["date", "ticker"], how="inner")
    df["split"] = assign_split(df["date"])
    df = df[df["split"] != "drop"].dropna(subset=FEATURES)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df.to_parquet(PANEL_PATH, index=False)

    print(f"Saved {len(df):,} rows to {PANEL_PATH.name}\n")
    summary = df.groupby("split").agg(
        first=("date", "min"), last=("date", "max"),
        rows=("label", "size"), crash_rate=("label", "mean"))
    summary = summary.loc[["train", "val", "test"]]
    summary["first"] = summary["first"].dt.date
    summary["last"] = summary["last"].dt.date
    summary["crash_rate"] = summary["crash_rate"].map("{:.1%}".format)
    print(summary.to_string())


if __name__ == "__main__":
    main()
