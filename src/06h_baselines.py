"""Module 6h: v3 against EVERY baseline, year by year, on both fresh holdouts.

    python src/06h_baselines.py

Reporting only: no model, threshold or setting is chosen here. Scores are the same as in
06e/06g (v3, volatility rule, 200-day average rule, model v1, no skill).
Outputs: results/baselines_by_year.csv, results/baselines_bootstrap.json
"""
import importlib
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import FEATURES, HOLDOUT2_TICKERS, HOLDOUT_TICKERS, RESULTS_DIR, ROOT, load_preds
from model_v3 import predict_v3, vol_prob

holdout_mod = importlib.import_module("06e_holdout")
N_BOOT = 1000
METHODS = ["v3", "volatility_rule", "ma200_rule", "v1"]


def scores_for(df, v1):
    return {"v3": predict_v3(df), "volatility_rule": vol_prob(df["vol_20"]),
            "ma200_rule": (-df["dist_ma200"]).clip(lower=0).to_numpy(),
            "v1": v1.predict(df[FEATURES])}


def bootstrap(df, y, a, b):
    month = df["date"].dt.to_period("M").to_numpy()
    groups = [np.where(month == m)[0] for m in np.unique(month)]
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        gaps.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
    gaps = np.array(gaps)
    return {"win_share": float((gaps > 0).mean()), "gap_mean": float(gaps.mean()),
            "ci_low": float(np.quantile(gaps, 0.025)), "ci_high": float(np.quantile(gaps, 0.975))}


def main():
    v1 = lgb.Booster(model_file=str(ROOT / "models" / "lgbm.txt"))
    rows, boots = [], {}
    for name, tickers in [("Holdout 1", HOLDOUT_TICKERS), ("Holdout 2", HOLDOUT2_TICKERS)]:
        df = holdout_mod.build_holdout(tickers)
        y = df["label"].to_numpy()
        sc = scores_for(df, v1)
        years = df["date"].dt.year.to_numpy()
        for yr in sorted(set(years)) + ["All"]:
            m = np.ones(len(df), bool) if yr == "All" else years == yr
            row = {"dataset": name, "year": yr, "no_skill": float(y[m].mean())}
            row.update({k: average_precision_score(y[m], sc[k][m]) for k in METHODS})
            rows.append(row)
        boots[name] = {f"v3_vs_{k}": bootstrap(df, y, sc["v3"], sc[k])
                       for k in ["volatility_rule", "ma200_rule"]}
        boots[name]["n_stocks"] = int(df["ticker"].nunique())
    # original 2021-25 test set (99 stocks; looked at before, disclosed): same scores
    t = load_preds()
    t = t[t["split"] == "test"]
    yt = t["label"].to_numpy()
    cols = {"v3": "score_model_v3", "volatility_rule": "score_vol", "ma200_rule": "score_ma200",
            "v1": "score_model"}
    years = t["date"].dt.year.to_numpy()
    for yr in sorted(set(years)) + ["All"]:
        m = np.ones(len(t), bool) if yr == "All" else years == yr
        row = {"dataset": "Original test", "year": yr, "no_skill": float(yt[m].mean())}
        row.update({k: average_precision_score(yt[m], t[c].to_numpy()[m]) for k, c in cols.items()})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "baselines_by_year.csv", index=False)
    (RESULTS_DIR / "baselines_bootstrap.json").write_text(json.dumps(boots, indent=2))
    print(out.to_string(index=False, float_format="{:.3f}".format))
    print(json.dumps(boots, indent=2))


if __name__ == "__main__":
    main()
