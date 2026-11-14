"""Module 5: baseline scores (higher = more risky).

    python src/05_baselines.py

  never warn   score 0 everywhere -> PR-AUC = crash rate (the "no skill" line)
  volatility   chance of touching -10% in 20 days if the stock moved like a
               random walk with today's 20-day volatility (normal distribution):
                   P = 2 * Phi( ln(0.9) / (vol_20 * sqrt(20)) )
               Rule version: warn if vol_20 is above the TRAINING 80th percentile.
  200-day MA   score = how far the price is below its 200-day average
               Rule version: warn if close < 200-day average.

Prints VALIDATION results only. Test results are looked at once, on day 2.
Output: data/preds.parquet (date, ticker, split, label, score_*, rule_*)
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, precision_score, recall_score

from common import DROP, HORIZON, PREDS_PATH, load_panel


def main():
    df = load_panel()
    out = df[["date", "ticker", "split", "label"]].copy()

    out["score_never"] = 0.0
    out["score_vol"] = 2 * norm.cdf(np.log(1 + DROP) / (df["vol_20"] * np.sqrt(HORIZON)))
    out["score_ma200"] = (-df["dist_ma200"]).clip(lower=0)

    vol_cut = df.loc[df["split"] == "train", "vol_20"].quantile(0.80)
    out["rule_vol"] = (df["vol_20"] > vol_cut).astype(np.int8)
    out["rule_ma200"] = (df["dist_ma200"] < 0).astype(np.int8)
    out["rule_never"] = np.int8(0)

    out.to_parquet(PREDS_PATH, index=False)
    print(f"Saved {len(out):,} rows to {PREDS_PATH.name}")
    print(f"Volatility rule threshold (train 80th pct): daily vol {vol_cut:.2%} "
          f"= {vol_cut * np.sqrt(252):.0%} a year\n")

    v = out[out["split"] == "val"]
    rows = []
    for name in ["never", "vol", "ma200"]:
        warn = v[f"rule_{name}"]
        rows.append({
            "method": name,
            "PR-AUC": average_precision_score(v["label"], v[f"score_{name}"]),
            "precision": precision_score(v["label"], warn, zero_division=0),
            "recall": recall_score(v["label"], warn),
            "accuracy": (warn == v["label"]).mean(),
            "warn_rate": warn.mean(),
        })
    res = pd.DataFrame(rows).set_index("method")
    print("VALIDATION (2019-2020) -- baselines only")
    print(res.to_string(float_format="{:.3f}".format))
    print(f"\nNo-skill PR-AUC = crash rate = {v['label'].mean():.3f}")


if __name__ == "__main__":
    main()
