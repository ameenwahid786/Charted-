"""Module 6e: test model v3 on the FRESH holdout (50 unseen stocks, 2021-2025).

    python src/01_download.py --holdout   # once, needs internet
    python src/06e_holdout.py

Nothing is tuned here. Thresholds come from validation (2019-2020):
  * volatility rule: vol_20 above the TRAINING 80th percentile (as in module 5)
  * model v3: warns on the same share of validation days as the volatility rule
Outputs in results/: holdout_metrics.csv, holdout_bootstrap.json,
holdout_by_year.csv, holdout_pr_curves.csv, holdout_pr_curve.png
"""
import importlib
import json

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score)

from common import (DROP, csv_path, FEATURES, FEATURES_PATH, HOLDOUT_TICKERS, HORIZON, RESULTS_DIR,
                    ROOT, TEST, load_all, load_panel, load_preds)
from model_v3 import add_v3_features, predict_v3, vol_prob

labels_mod = importlib.import_module("02_labels")
features_mod = importlib.import_module("03_features")

N_BOOT = 1000
COLORS = {"Model v3 (rule + corrections)": "#2a78d6", "Volatility rule": "#eb6834",
          "200-day average rule": "#1baf7a", "Model v1": "#eda100"}
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def build_holdout():
    available = [t for t in HOLDOUT_TICKERS if csv_path(t).exists()]
    missing = sorted(set(HOLDOUT_TICKERS) - set(available))
    if missing:
        print(f"Skipping holdout stocks that failed to download: {missing}")
    prices = load_all(available)
    feats = features_mod.build(prices)
    prices["fwd_min"] = prices.groupby("ticker")["close"].transform(
        lambda s: labels_mod.forward_min(s, HORIZON))
    lab = prices.assign(label=(prices["fwd_min"] / prices["close"] - 1 <= DROP).astype(int),
                        known=prices["fwd_min"].notna())[["date", "ticker", "label", "known"]]
    df = feats.merge(lab, on=["date", "ticker"])
    df = df[df["known"] & df["date"].between(*TEST)].dropna(subset=FEATURES)
    ref = pd.read_parquet(FEATURES_PATH)
    ref = ref[ref["date"].between(*TEST)]
    return add_v3_features(df, reference=ref).reset_index(drop=True)


def main():
    df = build_holdout()
    y = df["label"].to_numpy()
    panel, preds = load_panel(), load_preds()
    val = preds[preds["split"] == "val"]

    scores = {
        "Model v3 (rule + corrections)": predict_v3(df),
        "Volatility rule": vol_prob(df["vol_20"]),
        "200-day average rule": (-df["dist_ma200"]).clip(lower=0).to_numpy(),
        "Model v1": lgb.Booster(model_file=str(ROOT / "models" / "lgbm.txt")).predict(df[FEATURES]),
    }
    vol_cut = panel.loc[panel["split"] == "train", "vol_20"].quantile(0.80)
    rate = val["rule_vol"].mean()
    warn = {
        "Model v3 (rule + corrections)": scores["Model v3 (rule + corrections)"]
        >= np.quantile(val["score_model_v3"], 1 - rate),
        "Volatility rule": df["vol_20"].to_numpy() > vol_cut,
        "200-day average rule": df["dist_ma200"].to_numpy() < 0,
        "Model v1": scores["Model v1"] >= np.quantile(val["score_model"], 1 - rate),
    }
    rows = [{"method": k, "PR_AUC": average_precision_score(y, scores[k]),
             "precision": precision_score(y, warn[k]), "recall": recall_score(y, warn[k]),
             "F1": f1_score(y, warn[k]), "accuracy": float((warn[k] == y).mean()),
             "warn_rate": float(warn[k].mean())} for k in scores]
    rows.append({"method": "Never warn", "PR_AUC": y.mean(), "precision": 0, "recall": 0,
                 "F1": 0, "accuracy": float((y == 0).mean()), "warn_rate": 0})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "holdout_metrics.csv", index=False)

    # --- monthly block bootstrap: v3 minus volatility rule --------------------------
    month = df["date"].dt.to_period("M").to_numpy()
    groups = [np.where(month == m)[0] for m in np.unique(month)]
    rng = np.random.default_rng(0)
    s3, sv = scores["Model v3 (rule + corrections)"], scores["Volatility rule"]
    gaps = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        gaps.append(average_precision_score(y[idx], s3[idx]) - average_precision_score(y[idx], sv[idx]))
    gaps = np.array(gaps)
    boot = {"win_share": float((gaps > 0).mean()), "gap_mean": float(gaps.mean()),
            "ci_low": float(np.quantile(gaps, 0.025)), "ci_high": float(np.quantile(gaps, 0.975)),
            "n_boot": N_BOOT, "n_months": len(groups)}
    (RESULTS_DIR / "holdout_bootstrap.json").write_text(json.dumps(boot, indent=2))

    # --- by year and by stock --------------------------------------------------------
    yr = []
    for year, g in df.groupby(df["date"].dt.year):
        i = g.index.to_numpy()
        yr.append({"year": year, "crash_rate": y[i].mean(),
                   "v3": average_precision_score(y[i], s3[i]),
                   "volatility_rule": average_precision_score(y[i], sv[i])})
    yr = pd.DataFrame(yr)
    yr.to_csv(RESULTS_DIR / "holdout_by_year.csv", index=False)
    st = []
    for t, g in df.groupby("ticker"):
        i = g.index.to_numpy()
        if y[i].sum() >= 10:
            st.append(average_precision_score(y[i], s3[i]) > average_precision_score(y[i], sv[i]))

    # --- v3 on the ORIGINAL test set (third look, disclosed) --------------------------
    t = preds[preds["split"] == "test"]
    orig = {"v3": average_precision_score(t["label"], t["score_model_v3"]),
            "volatility_rule": average_precision_score(t["label"], t["score_vol"])}

    # --- PR curves -----------------------------------------------------------------
    curves = []
    for k, sc in scores.items():
        p, r, _ = precision_recall_curve(y, sc)
        i = np.unique(np.linspace(0, len(r) - 1, 400).astype(int))
        curves.append(pd.DataFrame({"method": k, "recall": r[i], "precision": p[i]}))
    curves = pd.concat(curves)
    curves.to_csv(RESULTS_DIR / "holdout_pr_curves.csv", index=False)
    m = metrics.set_index("method")
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for k in scores:
        c = curves[curves["method"] == k]
        ax.plot(c["recall"], c["precision"], color=COLORS[k], lw=2,
                ls="--" if k == "Model v1" else "-", label=f"{k}  PR-AUC {m.loc[k, 'PR_AUC']:.3f}")
    ax.axhline(y.mean(), color=INK2, lw=1.5, ls=":", label=f"No skill  PR-AUC {y.mean():.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.45)
    ax.set_xlabel("Recall: share of real 10% drops that got a warning", color=INK2)
    ax.set_ylabel("Precision: share of warnings that were right", color=INK2)
    ax.set_title("Fresh holdout: 50 unseen stocks, 2021-2025", color=INK, loc="left",
                 fontsize=12, fontweight="bold")
    ax.grid(color=GRID, lw=0.8); ax.tick_params(colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    ax.text(0, -0.16, "This project is a prototype and is not financial advice.",
            transform=ax.transAxes, fontsize=7.5, color=INK2)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "holdout_pr_curve.png", facecolor=SURFACE)

    print(f"FRESH HOLDOUT: {df['ticker'].nunique()} stocks, {len(df):,} stock-days, "
          f"crash rate {y.mean():.1%}\n")
    print(m.to_string(float_format="{:.3f}".format))
    print(f"\nBootstrap ({N_BOOT}x, monthly blocks): v3 beats the volatility rule in "
          f"{boot['win_share']:.0%} of resamples; gap {boot['gap_mean']:+.3f} "
          f"(95% CI {boot['ci_low']:+.3f} to {boot['ci_high']:+.3f})")
    print(f"Years won: {(yr['v3'] > yr['volatility_rule']).sum()} of {len(yr)}")
    print(yr.to_string(index=False, float_format="{:.3f}".format))
    print(f"Stocks won (with >=10 crashes): {sum(st)} of {len(st)}")
    print(f"\nOriginal 2021-25 test set (third look, disclosed): v3 {orig['v3']:.3f} "
          f"vs volatility rule {orig['volatility_rule']:.3f}")


if __name__ == "__main__":
    main()
