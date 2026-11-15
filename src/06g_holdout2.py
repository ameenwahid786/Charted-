"""Module 6g: the pre-registered holdout-2 test (results/v4_preregistration.md).

    python src/01_download.py --holdout2   # once, needs internet
    python src/06g_holdout2.py

Run ONCE; the output is committed whatever it shows. No v4 was selected (no candidate beat
v3 on validation, see results/v4_experiments_val.csv), so this is a second, independent
test of v3 against the volatility rule on 95 more stocks never used anywhere.
Thresholds are the ones fixed on validation (results/thresholds_v3.json).
Outputs: results/holdout2_metrics.csv, holdout2_bootstrap.json, holdout2_by_year.csv,
holdout2_pr_curves.csv, holdout2_summary.json
"""
import importlib
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score)

from common import FEATURES, HOLDOUT2_TICKERS, RESULTS_DIR, ROOT, csv_path
from model_v3 import predict_v3, vol_prob

holdout_mod = importlib.import_module("06e_holdout")
N_BOOT = 1000
V3, VOL = "Model v3 (rule + corrections)", "Volatility rule"


def main():
    df = holdout_mod.build_holdout(HOLDOUT2_TICKERS)
    y = df["label"].to_numpy()
    th = json.loads((RESULTS_DIR / "thresholds_v3.json").read_text())
    v1 = lgb.Booster(model_file=str(ROOT / "models" / "lgbm.txt"))
    v1_th = json.loads((RESULTS_DIR / "thresholds.json").read_text())["model_red"]
    scores = {V3: predict_v3(df), VOL: vol_prob(df["vol_20"]),
              "200-day average rule": (-df["dist_ma200"]).clip(lower=0).to_numpy(),
              "Model v1": v1.predict(df[FEATURES])}
    warn = {V3: scores[V3] >= th["red"], VOL: df["vol_20"].to_numpy() > th["vol_rule_cut_vol20"],
            "200-day average rule": df["dist_ma200"].to_numpy() < 0,
            "Model v1": scores["Model v1"] >= v1_th}
    rows = [{"method": k, "PR_AUC": average_precision_score(y, scores[k]),
             "precision": precision_score(y, warn[k]), "recall": recall_score(y, warn[k]),
             "F1": f1_score(y, warn[k]), "accuracy": float((warn[k] == y).mean()),
             "warn_rate": float(warn[k].mean())} for k in scores]
    rows.append({"method": "Never warn", "PR_AUC": y.mean(), "precision": 0, "recall": 0,
                 "F1": 0, "accuracy": float((y == 0).mean()), "warn_rate": 0})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "holdout2_metrics.csv", index=False)

    month = df["date"].dt.to_period("M").to_numpy()
    groups = [np.where(month == m)[0] for m in np.unique(month)]
    rng = np.random.default_rng(0)
    s3, sv = scores[V3], scores[VOL]
    gaps = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        gaps.append(average_precision_score(y[idx], s3[idx]) - average_precision_score(y[idx], sv[idx]))
    gaps = np.array(gaps)
    boot = {"win_share": float((gaps > 0).mean()), "gap_mean": float(gaps.mean()),
            "ci_low": float(np.quantile(gaps, 0.025)), "ci_high": float(np.quantile(gaps, 0.975)),
            "n_boot": N_BOOT, "n_months": len(groups)}
    (RESULTS_DIR / "holdout2_bootstrap.json").write_text(json.dumps(boot, indent=2))

    yr = pd.DataFrame([{"year": yv, "crash_rate": y[g.index].mean(),
                        "v3": average_precision_score(y[g.index], s3[g.index]),
                        "volatility_rule": average_precision_score(y[g.index], sv[g.index])}
                       for yv, g in df.groupby(df["date"].dt.year)])
    yr.to_csv(RESULTS_DIR / "holdout2_by_year.csv", index=False)
    won = [average_precision_score(y[g.index], s3[g.index]) >
           average_precision_score(y[g.index], sv[g.index])
           for _, g in df.groupby("ticker") if y[g.index].sum() >= 10]

    curves = []
    for k, sc in scores.items():
        p, r, _ = precision_recall_curve(y, sc)
        i = np.unique(np.linspace(0, len(r) - 1, 400).astype(int))
        curves.append(pd.DataFrame({"method": k, "recall": r[i], "precision": p[i]}))
    pd.concat(curves).to_csv(RESULTS_DIR / "holdout2_pr_curves.csv", index=False)

    missing = sorted(t for t in HOLDOUT2_TICKERS if not csv_path(t).exists())
    summary = {"n_stocks": int(df["ticker"].nunique()), "n_rows": int(len(df)),
               "crash_rate": float(y.mean()), "missing": missing,
               "years_won": int((yr["v3"] > yr["volatility_rule"]).sum()), "n_years": len(yr),
               "stocks_won": int(sum(won)), "stocks_compared": len(won),
               "note": "No v4 was selected (no candidate beat v3 on validation), so holdout 2 "
                       "is a second independent test of v3. Run once."}
    (RESULTS_DIR / "holdout2_summary.json").write_text(json.dumps(summary, indent=2))

    m = metrics.set_index("method")
    print(f"HOLDOUT 2: {summary['n_stocks']} stocks, {len(df):,} stock-days, crash rate {y.mean():.1%}"
          f" (missing: {missing})\n")
    print(m.to_string(float_format="{:.3f}".format))
    print(f"\nBootstrap: v3 beats the volatility rule in {boot['win_share']:.0%}; gap "
          f"{boot['gap_mean']:+.3f} (95% CI {boot['ci_low']:+.3f} to {boot['ci_high']:+.3f})")
    print(f"Years won {summary['years_won']} of {len(yr)}; stocks won {sum(won)} of {len(won)}")
    print(yr.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
