"""Model v4 candidates (pre-registered in results/v4_preregistration.md).

Same recipe as v3 ("starting point + learned corrections"), but the starting point can use
better volatility estimates:
  * Parkinson volatility from daily high/low ranges (more precise than closes alone)
  * EWMA volatility (reacts faster to recent days)
Every feature uses prices up to and including day t only (see leak_test below).
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import ROOT
from model_v3 import V3_FEATURES, add_v3_features, logit, vol_prob

MODEL_PATH = ROOT / "models" / "lgbm_v4.txt"
META_PATH = ROOT / "models" / "lgbm_v4.json"

OHLC_COLS = ["park_20", "ewm_vol", "down_vol_20", "max_drop_20", "skew_60"]
V4_EXTRA = ["park_ratio", "down_ratio", "max_drop_z", "skew_60", "park_20_rank"]
RULE_COLS = {"close": "vol_20", "park": "park_20", "ewm": "ewm_vol"}


def ohlc_features(prices):
    """prices: date, ticker, open, high, low, close -> per-day volatility extras."""
    parts = []
    for t, g in prices.sort_values(["ticker", "date"]).groupby("ticker"):
        g = g.reset_index(drop=True)
        r = np.log(g["close"]).diff()
        hl2 = np.log(g["high"] / g["low"]) ** 2
        out = pd.DataFrame({"date": g["date"], "ticker": t})
        out["park_20"] = np.sqrt(hl2.rolling(20).mean() / (4 * np.log(2)))
        out["ewm_vol"] = r.ewm(halflife=10, min_periods=20).std()
        out["down_vol_20"] = np.sqrt((r.clip(upper=0) ** 2).rolling(20).mean())
        out["max_drop_20"] = r.rolling(20).min()
        out["skew_60"] = r.rolling(60).skew()
        parts.append(out)
    df = pd.concat(parts, ignore_index=True)
    return df.replace([np.inf, -np.inf], np.nan)


def _rank(df, col, reference=None):
    if reference is None:
        return df.groupby("date")[col].rank(pct=True)
    ref = {d: np.sort(g[col].dropna().to_numpy()) for d, g in reference.groupby("date")}
    out = np.full(len(df), np.nan)
    x = df[col].to_numpy()
    for d, idx in df.groupby("date").indices.items():
        if d in ref:
            vals = ref[d]
            out[idx] = (np.searchsorted(vals, x[idx], side="right") + 1) / (len(vals) + 1)
    return out


def add_v4_features(df, reference=None):
    """df: v3 base columns + OHLC_COLS. reference: 99-stock frame with the same columns
    (for ranking stocks outside the universe), or None to rank rows against each other."""
    df = add_v3_features(df, reference=reference)
    df["park_ratio"] = df["park_20"] / df["vol_20"]
    df["down_ratio"] = df["down_vol_20"] / df["vol_20"]
    df["max_drop_z"] = df["max_drop_20"] / df["vol_20"]
    df["park_20_rank"] = _rank(df, "park_20", reference)
    for k, c in RULE_COLS.items():
        df[f"l_{k}"] = logit(vol_prob(df[c]))
    return df


def load_v4():
    meta = json.loads(META_PATH.read_text())
    return lgb.Booster(model_file=str(MODEL_PATH)), meta


def start_logit(df, meta):
    z = np.full(len(df), meta["cal_intercept"], dtype=float)
    for k, w in meta["cal_coef"].items():
        z += w * df[f"l_{k}"].to_numpy()
    return z


def predict_v4(df, booster=None, meta=None, contrib=False):
    if booster is None:
        booster, meta = load_v4()
    z0 = start_logit(df, meta)
    if contrib:
        return z0, booster.predict(df[meta["features"]], pred_contrib=True)
    raw = z0 + booster.predict(df[meta["features"]], raw_score=True)
    return 1 / (1 + np.exp(-raw))


def leak_test(prices):
    """Halve prices after a cutoff: features up to the cutoff must not change."""
    p = prices[prices["ticker"] == prices["ticker"].iloc[0]].copy()
    cutoff = p["date"].iloc[1500]
    base = ohlc_features(p)
    shocked = p.copy()
    after = shocked["date"] > cutoff
    shocked.loc[after, ["open", "high", "low", "close"]] *= 0.5
    test = ohlc_features(shocked)
    keep = base["date"] <= cutoff
    same = np.allclose(base.loc[keep, OHLC_COLS].to_numpy(), test.loc[keep, OHLC_COLS].to_numpy(),
                       equal_nan=True)
    print(f"v4 leak test: {'PASS - no feature looks ahead' if same else 'FAIL'}")
    if not same:
        raise SystemExit(1)


__all__ = ["V3_FEATURES", "V4_EXTRA", "add_v4_features", "ohlc_features", "predict_v4",
           "load_v4", "leak_test"]
