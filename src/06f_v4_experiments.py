"""Module 6f: model v4 candidates, chosen on VALIDATION ONLY (2019-20), then saved.

    python src/06f_v4_experiments.py

Candidates and the selection rule are fixed in results/v4_preregistration.md.
Test (2021-25) and holdout data are not touched here.
Writes results/v4_experiments_val.csv and, if a candidate beats V3 on validation,
models/lgbm_v4.txt + lgbm_v4.json.
"""
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from common import RESULTS_DIR, load_all, load_panel
from model_v3 import V3_FEATURES, V3_PARAMS
from model_v4 import (META_PATH, MODEL_PATH, V4_EXTRA, add_v4_features, leak_test,
                      ohlc_features)

warnings.filterwarnings("ignore")

CANDIDATES = {
    "V3": dict(rules=["close"], features=V3_FEATURES, leaves=3, min_child=2000),
    "P": dict(rules=["park"], features=V3_FEATURES, leaves=3, min_child=2000),
    "E": dict(rules=["close", "park", "ewm"], features=V3_FEATURES, leaves=3, min_child=2000),
    "X": dict(rules=["close", "park", "ewm"], features=V3_FEATURES + V4_EXTRA, leaves=3,
              min_child=2000),
    "X7": dict(rules=["close", "park", "ewm"], features=V3_FEATURES + V4_EXTRA, leaves=7,
               min_child=1000),
}


def build():
    prices = load_all()
    leak_test(prices)
    panel = load_panel()
    panel = panel[panel["split"].isin(["train", "val"])]
    df = panel.merge(ohlc_features(prices), on=["date", "ticker"], how="left")
    df = add_v4_features(df)
    l_cols = [f"l_{k}" for k in ["close", "park", "ewm"]]
    return df.dropna(subset=l_cols).reset_index(drop=True)


def fit(df, spec):
    tr, va = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy()
    y = df["label"].to_numpy()
    L = df[[f"l_{k}" for k in spec["rules"]]].to_numpy()
    cal = LogisticRegression().fit(L[tr], y[tr])
    init = L @ cal.coef_[0] + cal.intercept_[0]
    params = dict(V3_PARAMS, num_leaves=spec["leaves"], min_child_samples=spec["min_child"])
    m = lgb.LGBMClassifier(**params, n_estimators=2000)
    m.fit(df.loc[tr, spec["features"]], y[tr], init_score=init[tr],
          eval_set=[(df.loc[va, spec["features"]], y[va])], eval_init_score=[init[va]],
          eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
    raw = init[va] + m.predict(df.loc[va, spec["features"]], raw_score=True)
    ap = average_precision_score(y[va], raw)
    start_ap = average_precision_score(y[va], init[va])
    return m, cal, ap, start_ap


def main():
    df = build()
    va = df["split"] == "val"
    vol_ap = average_precision_score(df.loc[va, "label"], df.loc[va, "l_close"])
    rows, fitted = [], {}
    for name, spec in CANDIDATES.items():
        m, cal, ap, start_ap = fit(df, spec)
        fitted[name] = (m, cal, spec)
        rows.append({"candidate": name, "val_PR_AUC": ap, "start_only_val_PR_AUC": start_ap,
                     "trees": int(m.best_iteration_)})
        print(f"{name:3s} val PR-AUC {ap:.4f} (starting point alone {start_ap:.4f}, "
              f"{m.best_iteration_} trees)")
    res = pd.DataFrame(rows)
    res.loc[len(res)] = {"candidate": "Volatility rule (baseline)", "val_PR_AUC": vol_ap,
                         "start_only_val_PR_AUC": vol_ap, "trees": 0}
    res.to_csv(RESULTS_DIR / "v4_experiments_val.csv", index=False)
    print(f"Volatility rule val PR-AUC {vol_ap:.4f}")

    cands = res[res["candidate"].isin(CANDIDATES)]
    best = cands.loc[cands["val_PR_AUC"].idxmax(), "candidate"]
    v3_ap = float(cands.loc[cands["candidate"] == "V3", "val_PR_AUC"].iloc[0])
    if best == "V3":
        print("No candidate beats V3 on validation: there is no v4 (pre-registered rule).")
        return
    m, cal, spec = fitted[best]
    m.booster_.save_model(str(MODEL_PATH), num_iteration=m.best_iteration_)
    META_PATH.write_text(json.dumps({
        "candidate": best, "features": spec["features"], "rules": spec["rules"],
        "cal_coef": dict(zip(spec["rules"], map(float, cal.coef_[0]))),
        "cal_intercept": float(cal.intercept_[0]), "trees": int(m.best_iteration_),
        "params": dict(V3_PARAMS, num_leaves=spec["leaves"], min_child_samples=spec["min_child"]),
        "val_PR_AUC": float(cands["val_PR_AUC"].max()), "v3_val_PR_AUC": v3_ap,
        "vol_rule_val_PR_AUC": float(vol_ap)}, indent=2))
    print(f"Selected on validation: {best} -> saved as v4")


if __name__ == "__main__":
    main()
