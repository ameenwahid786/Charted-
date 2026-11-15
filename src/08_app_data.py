"""Module 8: precompute everything the web app needs (small, committed files).

    python src/08_app_data.py

The app must not train, download or rebuild the big parquet files, so this writes:
  data/app/app_history.parquet  last ~3 years per stock: close, v3 probability,
                                volatility-rule probability, warning level, label
  data/app/app_latest.parquet   one row per stock on the snapshot date: feature values
                                and exact per-feature contributions (LightGBM pred_contrib)
  data/app/app_meta.json        thresholds, base rate, snapshot date, names, sectors
  results/thresholds_v3.json    v3 warning thresholds, fixed on VALIDATION (2019-20)

Covers the 99 main stocks and the fresh holdout stocks. Nothing is tuned here.
Prices come from the app snapshot in data/raw_live/ (src/01b_download_live.py), which runs
to the latest close; the research data in data/raw/ is not used or changed here.
"""
import importlib
import json

import numpy as np
import pandas as pd

from common import (DATA_DIR, DROP, FEATURES, HOLDOUT_SECTORS, HOLDOUT_TICKERS, HORIZON,
                    RESULTS_DIR, TEST, TICKER_SECTOR, TICKERS, csv_path, load_all_live, load_live,
                    load_panel, load_preds)
from companies import COMPANY
from model_v3 import V3_FEATURES, add_v3_features, load_v3, logit, vol_prob

labels_mod = importlib.import_module("02_labels")
features_mod = importlib.import_module("03_features")

APP_DIR = DATA_DIR / "app"
HISTORY_START = "2023-01-01"
# raw values the app's plain-English templates quote
RAW_COLS = ["close", "vol_20", "vol_60", "ret_20", "ret_60", "drawdown_252", "dist_ma200",
            "beta_60", "volume_ratio_20", "vix", "vix_chg_5", "mkt_ret_20", "rsi_14"]


def thresholds():
    """red = v3 warns on the same share of validation days as the volatility rule;
    amber = twice that share. Fixed on 2019-20 only."""
    preds, panel = load_preds(), load_panel()
    val = preds[preds["split"] == "val"]
    rate = float(val["rule_vol"].mean())
    out = {
        "red": float(np.quantile(val["score_model_v3"], 1 - rate)),
        "amber": float(np.quantile(val["score_model_v3"], 1 - 2 * rate)),
        "val_warn_share_red": rate,
        "val_warn_share_amber": 2 * rate,
        "vol_rule_cut_vol20": float(panel.loc[panel["split"] == "train", "vol_20"].quantile(0.80)),
        "base_rate_test": float(preds.loc[preds["split"] == "test", "label"].mean()),
        "note": "Fixed on validation 2019-20. red = v3 warns on the same share of validation "
                "days as the volatility rule (vol_20 above the training 80th percentile); "
                "amber = twice that share. Never chosen using test or holdout data.",
    }
    (RESULTS_DIR / "thresholds_v3.json").write_text(json.dumps(out, indent=2))
    return out


def labels_for(prices):
    prices = prices.copy()
    prices["fwd_min"] = prices.groupby("ticker")["close"].transform(
        lambda s: labels_mod.forward_min(s, HORIZON))
    prices["label"] = np.where(prices["fwd_min"].isna(), np.nan,
                               (prices["fwd_min"] / prices["close"] - 1 <= DROP).astype(float))
    return prices[["date", "ticker", "label"]]


def main():
    th = thresholds()
    booster, meta = load_v3()

    # --- main 99 stocks: ranks within themselves, exactly as in training -----------------
    mp = load_all_live(TICKERS)
    ref = features_mod.build(mp, loader=load_live)
    ref = ref[ref["date"] >= HISTORY_START].dropna(subset=FEATURES)
    main_df = add_v3_features(ref)
    main_df = main_df.merge(labels_for(mp), on=["date", "ticker"], how="left").assign(
        universe="main")

    # --- holdout stocks: ranked against the 99-stock universe on the same day ------------
    avail = [t for t in HOLDOUT_TICKERS if csv_path(t).exists()]
    hp = load_all_live(avail)
    hf = features_mod.build(hp, loader=load_live)
    hf = hf[hf["date"] >= HISTORY_START].dropna(subset=FEATURES)
    hold_df = add_v3_features(hf, reference=ref)
    hold_df = hold_df.merge(labels_for(hp), on=["date", "ticker"], how="left").assign(
        universe="holdout")

    df = pd.concat([main_df, hold_df], ignore_index=True).dropna(subset=V3_FEATURES)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # --- v3 score = calibrated volatility log-odds + tree corrections --------------------
    start_logit = meta["cal_a"] * logit(vol_prob(df["vol_20"])) + meta["cal_b"]
    contrib = booster.predict(df[V3_FEATURES], pred_contrib=True)  # last col = tree bias
    raw = start_logit + contrib.sum(axis=1)
    df["p_v3"] = 1 / (1 + np.exp(-raw))
    df["p_vol_raw"] = vol_prob(df["vol_20"])          # the rule's own 2*Phi formula
    df["p_vol"] = 1 / (1 + np.exp(-start_logit))      # same rule, calibrated on 2010-18
    df["level"] = np.select([df["p_v3"] >= th["red"], df["p_v3"] >= th["amber"]],
                            ["red", "amber"], "green")
    df["vol_rule_warn"] = df["vol_20"] > th["vol_rule_cut_vol20"]

    APP_DIR.mkdir(parents=True, exist_ok=True)
    hist = df[["date", "ticker", "close", "p_v3", "p_vol", "p_vol_raw", "level",
               "vol_rule_warn", "label"]].copy()
    for c in ["close", "p_v3", "p_vol", "p_vol_raw", "label"]:
        hist[c] = hist[c].astype("float32")
    hist.to_parquet(APP_DIR / "app_history.parquet", index=False)

    # daily OHLCV for the stock pages (key stats, price chart)
    px = pd.concat([mp, hp], ignore_index=True)
    px = px[px["date"] >= HISTORY_START][["date", "ticker", "open", "high", "low", "close",
                                          "volume"]]
    for c in ["open", "high", "low", "close", "volume"]:
        px[c] = px[c].astype("float32")
    px.to_parquet(APP_DIR / "app_prices.parquet", index=False)

    last = df.groupby("ticker")["date"].transform("max") == df["date"]
    cols = ["date", "ticker", "universe", "p_v3", "p_vol", "p_vol_raw", "level", "vol_rule_warn"]
    cols += list(dict.fromkeys(RAW_COLS + V3_FEATURES))  # no duplicates
    latest = df.loc[last, cols].copy()
    c_last = contrib[last.to_numpy()]
    for i, f in enumerate(V3_FEATURES):
        latest[f"contrib_{f}"] = c_last[:, i]
    latest["contrib_bias"] = c_last[:, -1]
    latest["start_logit"] = start_logit[last.to_numpy()]
    latest.to_parquet(APP_DIR / "app_latest.parquet", index=False)

    sector = dict(TICKER_SECTOR)
    sector.update({t: s for s, g in HOLDOUT_SECTORS.items() for t in g})
    tickers = sorted(latest["ticker"])
    as_of = latest["date"].max()
    app_meta = {
        "as_of": str(as_of.date()),
        "history_start": HISTORY_START,
        "thresholds": th,
        "names": {t: COMPANY.get(t, t) for t in tickers},
        "sector": {t: sector.get(t, "") for t in tickers},
        "universe": dict(zip(latest["ticker"], latest["universe"])),
        "stale": sorted(latest.loc[latest["date"] < as_of, "ticker"]),
        "test_period": list(TEST),
    }
    (APP_DIR / "app_meta.json").write_text(json.dumps(app_meta, indent=2))

    print(f"Snapshot date {as_of.date()}: {len(tickers)} stocks "
          f"({(latest['universe'] == 'main').sum()} main + {(latest['universe'] == 'holdout').sum()} holdout)")
    print(f"History rows: {len(hist):,}  (from {HISTORY_START})")
    print(f"v3 thresholds (validation): red >= {th['red']:.3f}, amber >= {th['amber']:.3f}; "
          f"typical-stock base rate {th['base_rate_test']:.1%}")
    print(f"Mean tree bias in log-odds: {latest['contrib_bias'].mean():+.3f}")
    print("Warning levels today:", latest["level"].value_counts().to_dict())
    for f in sorted(APP_DIR.iterdir()):
        print(f"  {f.relative_to(DATA_DIR.parent)}  {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
