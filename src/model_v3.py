"""Model v3 = "volatility rule + learned corrections" (shared by 06d and the app).

The volatility rule's probability is turned into log-odds and calibrated on
training data (a, b). LightGBM then only learns small corrections on top, using
volatility-scaled features and same-day ranks against the 99-stock universe.

    p = sigmoid( a * logit(vol_rule_prob) + b + trees(features) )
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import DROP, HORIZON, ROOT

MODEL_PATH = ROOT / "models" / "lgbm_v3.txt"
META_PATH = ROOT / "models" / "lgbm_v3.json"

RANK_COLS = ["vol_20", "vol_60", "ret_20", "drawdown_252", "beta_60", "volume_ratio_20"]
V3_FEATURES = ["ret_20_z", "ret_60_z", "drawdown_z", "dist_ma200_z", "vol_vs_mkt",
               "vol_20_rank", "vol_60_rank", "ret_20_rank", "drawdown_252_rank",
               "beta_60_rank", "volume_ratio_20_rank", "vol_ratio_20_60", "rsi_14",
               "vix", "vix_chg_5", "mkt_ret_20"]
V3_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=3,
                 min_child_samples=2000, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1)


def vol_prob(vol_20):
    return 2 * norm.cdf(np.log(1 + DROP) / (np.asarray(vol_20) * np.sqrt(HORIZON)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def add_v3_features(df, reference=None):
    """Add scaled + rank features. Ranks are percentiles within the same day.

    reference=None: rank rows against each other (the 99-stock training universe).
    reference=DataFrame (date + RANK_COLS): rank each row against the reference
    universe on that date instead (used for holdout stocks and the app).
    """
    df = df.copy()
    df["ret_20_z"] = df["ret_20"] / (df["vol_20"] * np.sqrt(20))
    df["ret_60_z"] = df["ret_60"] / (df["vol_60"] * np.sqrt(60))
    df["drawdown_z"] = df["drawdown_252"] / df["vol_60"]
    df["dist_ma200_z"] = df["dist_ma200"] / df["vol_60"]
    df["vol_vs_mkt"] = df["vol_20"] / df["mkt_vol_20"]
    if reference is None:
        g = df.groupby("date")
        for c in RANK_COLS:
            df[f"{c}_rank"] = g[c].rank(pct=True)
    else:
        ref = {d: g for d, g in reference.groupby("date")}
        for c in RANK_COLS:
            out = np.full(len(df), np.nan)
            for d, idx in df.groupby("date").indices.items():
                if d not in ref:
                    continue
                vals = np.sort(ref[d][c].dropna().to_numpy())
                x = df[c].to_numpy()[idx]
                # percentile as if this stock were added to the reference universe
                out[idx] = (np.searchsorted(vals, x, side="right") + 1) / (len(vals) + 1)
            df[f"{c}_rank"] = out
    return df


def load_v3():
    meta = json.loads(META_PATH.read_text())
    return lgb.Booster(model_file=str(MODEL_PATH)), meta


def predict_v3(df, booster=None, meta=None):
    """df must already contain V3_FEATURES and vol_20."""
    if booster is None:
        booster, meta = load_v3()
    init = meta["cal_a"] * logit(vol_prob(df["vol_20"])) + meta["cal_b"]
    raw = init + booster.predict(df[V3_FEATURES], raw_score=True)
    return 1 / (1 + np.exp(-raw))
