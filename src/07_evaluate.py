"""Module 7: the one-time test-set evaluation (2021-2025) + the headline graph.

    python src/07_evaluate.py

Nothing here is tuned on the test set:
  * the model's warning threshold is fixed on VALIDATION so the model raises
    the same share of warnings as the volatility rule did on validation;
  * the volatility-rule cut-off was fixed on TRAINING data in module 5.

Outputs (all read by the web app, all committed):
  results/metrics_test.csv   PR-AUC, precision, recall, F1, accuracy, warn rate
  results/matched.csv        recall at the vol rule's precision, and vice versa
  results/pr_curves.csv      precision/recall points for the interactive chart
  results/thresholds.json    the model's warning threshold (+ amber level)
  results/pr_curve.png       static version of the headline graph
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score)

from common import RESULTS_DIR, load_preds

# v1 is the pre-registered model and the one the app uses; v2 is the declared
# post-hoc refit (06b), reported for transparency.
METHODS = {  # name shown to judges -> score column
    "Model v1 (trained 2010-18)": "score_model",
    "Volatility rule": "score_vol",
    "200-day average rule": "score_ma200",
    "Model v2 (trained 2010-20)": "score_model_v2",
}
MODELS = ["Model v1 (trained 2010-18)", "Model v2 (trained 2010-20)"]
COLORS = {"Model v1 (trained 2010-18)": "#2a78d6", "Volatility rule": "#eb6834",
          "200-day average rule": "#1baf7a", "Model v2 (trained 2010-20)": "#eda100"}
STYLES = {"Model v2 (trained 2010-20)": "--"}
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def best_recall_at_precision(y, s, p_min):
    prec, rec, _ = precision_recall_curve(y, s)
    ok = prec >= p_min
    return float(rec[ok].max()) if ok.any() else 0.0


def best_precision_at_recall(y, s, r_min):
    prec, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r_min
    return float(prec[ok].max()) if ok.any() else 0.0


def main():
    preds = load_preds()
    val = preds[preds["split"] == "val"]
    test = preds[preds["split"] == "test"]
    y = test["label"].to_numpy()
    base_rate = y.mean()

    # --- thresholds fixed on validation --------------------------------------
    # Each model warns on the same share of validation days as the volatility rule.
    vol_warn_rate_val = val["rule_vol"].mean()
    thr = {name: float(np.quantile(val[METHODS[name]], 1 - vol_warn_rate_val))
           for name in MODELS}
    amber = float(np.quantile(val["score_model"], 1 - 2 * vol_warn_rate_val))
    (RESULTS_DIR / "thresholds.json").write_text(json.dumps({
        "model_red": thr[MODELS[0]], "model_amber": amber,
        "model_v2_red": thr[MODELS[1]], "score_column": "score_model",
        "note": "red = same warning budget as the volatility rule on validation; "
                "amber = twice that budget.",
        "base_rate_test": float(base_rate)}, indent=2))

    warn = {name: (test[METHODS[name]] >= thr[name]).astype(int) for name in MODELS}
    warn.update({
        "Volatility rule": test["rule_vol"],
        "200-day average rule": test["rule_ma200"],
        "Never warn": test["rule_never"],
    })
    rows = []
    for name, w in warn.items():
        col = METHODS.get(name)
        rows.append({
            "method": name,
            "PR_AUC": average_precision_score(y, test[col]) if col else base_rate,
            "precision": precision_score(y, w, zero_division=0),
            "recall": recall_score(y, w),
            "F1": f1_score(y, w, zero_division=0),
            "accuracy": float((w.to_numpy() == y).mean()),
            "warn_rate": float(w.mean()),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "metrics_test.csv", index=False)

    # --- matched comparisons (practical metric) -------------------------------
    vol_p = metrics.set_index("method").loc["Volatility rule", "precision"]
    vol_r = metrics.set_index("method").loc["Volatility rule", "recall"]
    rows = []
    for name in MODELS:
        sc = test[METHODS[name]]
        rows += [{"model_version": name, "measure": "recall",
                  "question": f"Share of crashes caught when warnings are right {vol_p:.0%} of the time",
                  "model": best_recall_at_precision(y, sc, vol_p), "volatility_rule": float(vol_r)},
                 {"model_version": name, "measure": "precision",
                  "question": f"Share of warnings that are right when catching {vol_r:.0%} of crashes",
                  "model": best_precision_at_recall(y, sc, vol_r), "volatility_rule": float(vol_p)}]
    matched = pd.DataFrame(rows)
    matched.to_csv(RESULTS_DIR / "matched.csv", index=False)

    # --- PR curves: data for the app + static PNG --------------------------------
    curves = []
    for name, col in METHODS.items():
        prec, rec, _ = precision_recall_curve(y, test[col])
        idx = np.unique(np.linspace(0, len(rec) - 1, 400).astype(int))
        curves.append(pd.DataFrame({"method": name, "recall": rec[idx], "precision": prec[idx]}))
    pd.concat(curves).to_csv(RESULTS_DIR / "pr_curves.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    m = metrics.set_index("method")
    for name, col in METHODS.items():
        c = pd.concat(curves).query("method == @name")
        ax.plot(c["recall"], c["precision"], color=COLORS[name], lw=2,
                ls=STYLES.get(name, "-"),
                label=f"{name}  PR-AUC {m.loc[name, 'PR_AUC']:.3f}")
        ax.scatter(m.loc[name, "recall"], m.loc[name, "precision"], s=60,
                   color=COLORS[name], edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.axhline(base_rate, color=INK2, lw=1.5, ls="--",
               label=f"No skill (guessing)  PR-AUC {base_rate:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.45)
    ax.set_xlabel("Recall: share of real 10% drops that got a warning", color=INK2)
    ax.set_ylabel("Precision: share of warnings that were right", color=INK2)
    ax.set_title("Which method spots 10% drops better? Unseen data, 2021-2025",
                 color=INK, loc="left", fontsize=12, fontweight="bold")
    ax.grid(color=GRID, lw=0.8); ax.tick_params(colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    ax.text(0, -0.16, "Dots = each method's actual yes/no warning setting. "
            "This project is a prototype and is not financial advice.",
            transform=ax.transAxes, fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pr_curve.png", facecolor=SURFACE)

    # --- report -------------------------------------------------------------------
    print(f"TEST SET 2021-2025: {len(test):,} stock-days, crash rate {base_rate:.1%}\n")
    print(metrics.set_index("method").to_string(float_format="{:.3f}".format))
    print("\nMatched comparison:")
    for _, r in matched.iterrows():
        print(f"  [{r['model_version']}] {r['question']}: model {r['model']:.1%} "
              f"vs volatility rule {r['volatility_rule']:.1%}")
    for name in MODELS:
        gap = m.loc[name, "PR_AUC"] - m.loc["Volatility rule", "PR_AUC"]
        print(f"PR-AUC gap, {name} minus volatility rule: {gap:+.3f}")


if __name__ == "__main__":
    main()
