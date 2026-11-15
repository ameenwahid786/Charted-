"""Module 6: LightGBM crash-risk model.

    python src/06_model.py

1. Train on 2010-2018 only.
2. Try a few settings; pick the one with the best PR-AUC on VALIDATION (2019-2020).
   Validation is also used for early stopping. The test split is never touched.
3. Write score_model (probability of a 10%+ drop in 20 days) into preds.parquet
   for every row, and save the model + feature importance.

Prints VALIDATION results only. Test results are looked at once, in module 7.
"""
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import FEATURES, PREDS_PATH, RESULTS_DIR, ROOT, load_panel, load_preds

warnings.filterwarnings("ignore", message=".*eval_set.*")  # newer lightgbm rename

MODEL_DIR = ROOT / "models"
SEED = 42

# Small, deliberately conservative grid: crashes cluster in time, so a model
# that is too flexible memorises particular years instead of general patterns.
GRID = [
    {"num_leaves": 7,  "min_child_samples": 1000},
    {"num_leaves": 15, "min_child_samples": 1000},
    {"num_leaves": 15, "min_child_samples": 3000},
    {"num_leaves": 31, "min_child_samples": 1000},
    {"num_leaves": 31, "min_child_samples": 3000},
]
BASE = dict(objective="binary", learning_rate=0.03, n_estimators=2000,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1)


def fit(params, X_tr, y_tr, X_va, y_va):
    model = lgb.LGBMClassifier(**BASE, **params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    p = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
    return model, average_precision_score(y_va, p)


def main():
    panel = load_panel()
    preds = load_preds()
    assert (panel[["date", "ticker"]].values == preds[["date", "ticker"]].values).all()

    tr = panel["split"] == "train"
    va = panel["split"] == "val"
    X, y = panel[FEATURES], panel["label"]

    print(f"Training on {tr.sum():,} rows (2010-2018), tuning on {va.sum():,} rows (2019-2020)\n")
    results = []
    best = None
    for params in GRID:
        model, ap = fit(params, X[tr], y[tr], X[va], y[va])
        results.append({**params, "trees": model.best_iteration_, "val_PR_AUC": ap})
        print(f"  leaves={params['num_leaves']:>2}  min_leaf={params['min_child_samples']:>4}"
              f"  trees={model.best_iteration_:>4}  val PR-AUC={ap:.3f}")
        if best is None or ap > best[1]:
            best = (model, ap, params)
    model, best_ap, best_params = best

    # Score every row with the chosen model (train rows are in-sample: don't evaluate on them)
    preds["score_model"] = model.predict_proba(X, num_iteration=model.best_iteration_)[:, 1]
    preds.to_parquet(PREDS_PATH, index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODEL_DIR / "lgbm.txt"), num_iteration=model.best_iteration_)
    (MODEL_DIR / "lgbm_params.json").write_text(json.dumps(
        {**BASE, **best_params, "n_estimators": int(model.best_iteration_),
         "features": FEATURES}, indent=2))
    pd.DataFrame(results).to_csv(RESULTS_DIR / "model_tuning_val.csv", index=False)
    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    imp.rename("gain_share").to_csv(RESULTS_DIR / "feature_importance.csv")

    v = preds[va]
    vol_ap = average_precision_score(v["label"], v["score_vol"])
    ma_ap = average_precision_score(v["label"], v["score_ma200"])
    print(f"\nChosen: {best_params}, {model.best_iteration_} trees")
    print("\nVALIDATION PR-AUC (2019-2020)")
    print(f"  model            {best_ap:.3f}")
    print(f"  volatility rule  {vol_ap:.3f}")
    print(f"  200-day average  {ma_ap:.3f}")
    print(f"  no skill         {v['label'].mean():.3f}")
    print(f"  -> model {'BEATS' if best_ap > vol_ap else 'does NOT beat'} the volatility rule "
          f"on validation ({best_ap - vol_ap:+.3f})")
    print("\nTop features (share of total gain):")
    print(imp.head(6).map("{:.0%}".format).to_string())


if __name__ == "__main__":
    main()
