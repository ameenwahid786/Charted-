"""Module 6c: try improved models -- chosen on VALIDATION ONLY (2019-2020).

    python src/06c_experiments.py

The 2021-2025 test set has already been used twice (v1, v2), so it is NOT used
here at all. The winner is then tested once on a fresh set of stocks the
project has never seen (module 6d).

Ideas:
  A  v1 as before (reference)
  B  "rule + corrections": LightGBM starts from the volatility rule's score
     (init_score) and only learns adjustments on top of it
  C  B with volatility-scaled and same-day ranked features (regime-robust)
  D  logistic regression on a handful of scaled features
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DROP, FEATURES, HORIZON, RESULTS_DIR, load_panel

warnings.filterwarnings("ignore")
SEED = 42


def vol_score(df):
    return 2 * norm.cdf(np.log(1 + DROP) / (df["vol_20"] * np.sqrt(HORIZON)))


def add_robust_features(df):
    """Scaled + same-day cross-sectional features (only same-day info, no look-ahead)."""
    df = df.copy()
    df["ret_20_z"] = df["ret_20"] / (df["vol_20"] * np.sqrt(20))
    df["ret_60_z"] = df["ret_60"] / (df["vol_60"] * np.sqrt(60))
    df["drawdown_z"] = df["drawdown_252"] / df["vol_60"]
    df["dist_ma200_z"] = df["dist_ma200"] / df["vol_60"]
    df["vol_vs_mkt"] = df["vol_20"] / df["mkt_vol_20"]
    g = df.groupby("date")
    for c in ["vol_20", "vol_60", "ret_20", "drawdown_252", "beta_60", "volume_ratio_20"]:
        df[f"{c}_rank"] = g[c].rank(pct=True)
    return df


ROBUST = ["ret_20_z", "ret_60_z", "drawdown_z", "dist_ma200_z", "vol_vs_mkt",
          "vol_20_rank", "vol_60_rank", "ret_20_rank", "drawdown_252_rank",
          "beta_60_rank", "volume_ratio_20_rank", "vol_ratio_20_60", "rsi_14",
          "vix", "vix_chg_5", "mkt_ret_20"]
LOGIT_FEATS = ["vol_20", "vol_60", "drawdown_252", "beta_60", "mkt_vol_20"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def lgbm(X_tr, y_tr, X_va, y_va, init_tr=None, init_va=None, leaves=7, min_leaf=1000):
    m = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, n_estimators=2000,
                           num_leaves=leaves, min_child_samples=min_leaf, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=SEED, verbose=-1)
    m.fit(X_tr, y_tr, init_score=init_tr, eval_set=[(X_va, y_va)],
          eval_init_score=[init_va] if init_va is not None else None,
          eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    raw = m.predict(X_va, raw_score=True, num_iteration=m.best_iteration_)
    if init_va is not None:
        raw = raw + init_va
    return m, average_precision_score(y_va, raw), m.best_iteration_


def main():
    df = add_robust_features(load_panel())
    tr, va = df["split"] == "train", df["split"] == "val"
    y = df["label"]
    # the rule's score, calibrated to log-odds on TRAIN (Platt scaling) so it is a sensible start
    vs = pd.DataFrame({"v": logit(np.asarray(vol_score(df)))}, index=df.index)
    cal = LogisticRegression().fit(vs[tr], y[tr])
    init = cal.decision_function(vs)

    res = []
    res.append(("Volatility rule (baseline)", average_precision_score(y[va], vol_score(df[va])), "-"))
    _, ap, n = lgbm(df.loc[tr, FEATURES], y[tr], df.loc[va, FEATURES], y[va])
    res.append(("A  v1 LightGBM (reference)", ap, n))
    for leaves, ml in [(3, 2000), (7, 1000), (7, 3000)]:
        _, ap, n = lgbm(df.loc[tr, FEATURES], y[tr], df.loc[va, FEATURES], y[va],
                        init[tr], init[va], leaves, ml)
        res.append((f"B  rule + corrections, 16 features (leaves={leaves}, min={ml})", ap, n))
        _, ap, n = lgbm(df.loc[tr, ROBUST], y[tr], df.loc[va, ROBUST], y[va],
                        init[tr], init[va], leaves, ml)
        res.append((f"C  rule + corrections, robust features (leaves={leaves}, min={ml})", ap, n))
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(df.loc[tr, LOGIT_FEATS], y[tr])
    res.append(("D  logistic regression, 5 features",
                average_precision_score(y[va], lr.predict_proba(df.loc[va, LOGIT_FEATS])[:, 1]), "-"))

    out = pd.DataFrame(res, columns=["model", "val_PR_AUC", "trees"])
    out.to_csv(RESULTS_DIR / "experiments_val.csv", index=False)
    print("VALIDATION PR-AUC (2019-2020) -- test set not used\n")
    print(out.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
