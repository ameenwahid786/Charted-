"""Module 2: crash labels.

    python src/02_labels.py

For each stock on each day t:
    fwd_min = lowest close over days t+1 .. t+20
    label   = 1 if fwd_min / close_t - 1 <= -10%, else 0

The last 20 days of each stock have no full future window, so they are dropped.
Output: data/labels.parquet  (date, ticker, close, fwd_min_ret, label)
"""
import numpy as np
import pandas as pd

from common import DROP, HORIZON, LABELS_PATH, load_all


def forward_min(close: pd.Series, horizon: int) -> pd.Series:
    """Lowest value over the NEXT `horizon` rows (excludes today)."""
    # reverse, take a trailing rolling min, reverse back, then step one day ahead
    rev_min = close[::-1].rolling(horizon, min_periods=horizon).min()[::-1]
    return rev_min.shift(-1)


def main():
    prices = load_all()
    prices["fwd_min"] = prices.groupby("ticker")["close"].transform(
        lambda s: forward_min(s, HORIZON))
    prices["fwd_min_ret"] = prices["fwd_min"] / prices["close"] - 1
    out = prices.dropna(subset=["fwd_min_ret"]).copy()
    out["label"] = (out["fwd_min_ret"] <= DROP).astype(np.int8)
    out = out[["date", "ticker", "close", "fwd_min_ret", "label"]]
    out.to_parquet(LABELS_PATH, index=False)

    print(f"Saved {len(out):,} rows to {LABELS_PATH.name}")
    print(f"Overall crash rate: {out['label'].mean():.1%}  "
          f"(this is the 'no skill' line on the PR curve)")
    print("\nCrash rate by year:")
    by_year = out.groupby(out["date"].dt.year)["label"].mean()
    print(by_year.map("{:.1%}".format).to_string())
    by_stock = out.groupby("ticker")["label"].mean().sort_values()
    print(f"\nLowest-risk stocks:  {by_stock.head(3).map('{:.1%}'.format).to_dict()}")
    print(f"Highest-risk stocks: {by_stock.tail(3).map('{:.1%}'.format).to_dict()}")


if __name__ == "__main__":
    main()
