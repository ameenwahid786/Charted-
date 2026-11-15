"""Module 9: simple backtest of the warnings (2021-2025).

    python src/09_backtest.py      # after 08_app_data.py (needs results/thresholds_v3.json)

Equal-weight portfolio, rebalanced daily. Each stock is held, except that while its
warning is on at day t's close it sits in cash (0% return) for day t+1. Warnings use
thresholds fixed on validation (2019-20) only:
  * model v3: p_v3 >= red threshold (results/thresholds_v3.json)
  * volatility rule: vol_20 above the TRAINING 80th percentile
  * 200-day rule: close below its 200-day average
  * buy and hold: never in cash
No trading costs, taxes or dividends beyond adjusted closes. Nothing is tuned.

Universes: the 49 fresh holdout stocks and the original 99 stocks (test period).
Outputs: results/backtest_summary.csv, backtest_curves.csv, backtest_note.txt
"""
import importlib
import json

import numpy as np
import pandas as pd

from common import RESULTS_DIR, load_panel, load_preds
from model_v3 import predict_v3

holdout_mod = importlib.import_module("06e_holdout")

STRATS = ["v3", "vol_rule", "ma200", "buy_hold"]


def portfolio(df, th):
    """df: date, ticker, close, p_v3, vol_20, dist_ma200 -> daily portfolio returns."""
    df = df.sort_values(["ticker", "date"]).copy()
    df["next_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
    df = df.dropna(subset=["next_ret"])
    cash = {
        "v3": df["p_v3"] >= th["red"],
        "vol_rule": df["vol_20"] > th["vol_rule_cut_vol20"],
        "ma200": df["dist_ma200"] < 0,
        "buy_hold": pd.Series(False, index=df.index),
    }
    rets, in_cash = {}, {}
    for k, c in cash.items():
        r = df["next_ret"].where(~c, 0.0)
        # return earned on day t+1 is dated t+1
        nxt = df.groupby("ticker")["date"].shift(-1).loc[df.index]
        rets[k] = r.groupby(nxt).mean()
        in_cash[k] = float(c.mean())
    return pd.DataFrame(rets).sort_index(), in_cash


def summarise(rets, in_cash, universe):
    rows, curves = [], []
    for k in STRATS:
        r = rets[k]
        value = (1 + r).cumprod()
        years = len(r) / 252
        rows.append({"universe": universe, "strategy": k,
                     "total_return": value.iloc[-1] - 1,
                     "cagr": value.iloc[-1] ** (1 / years) - 1,
                     "max_drawdown": (value / value.cummax() - 1).min(),
                     "volatility": r.std() * np.sqrt(252),
                     "time_in_cash": in_cash[k]})
        curves.append(pd.DataFrame({"universe": universe, "strategy": k,
                                    "date": value.index, "value": value.to_numpy()}))
    return pd.DataFrame(rows), pd.concat(curves)


def main():
    th = json.loads((RESULTS_DIR / "thresholds_v3.json").read_text())

    hold = holdout_mod.build_holdout()
    hold["p_v3"] = predict_v3(hold)
    h_rets, h_cash = portfolio(hold[["date", "ticker", "close", "p_v3", "vol_20", "dist_ma200"]], th)

    panel, preds = load_panel(), load_preds()
    test = panel[panel["split"] == "test"][["date", "ticker", "close", "vol_20", "dist_ma200"]]
    test = test.merge(preds.loc[preds["split"] == "test", ["date", "ticker", "score_model_v3"]],
                      on=["date", "ticker"]).rename(columns={"score_model_v3": "p_v3"})
    m_rets, m_cash = portfolio(test, th)

    n_h, n_m = hold["ticker"].nunique(), test["ticker"].nunique()
    s1, c1 = summarise(h_rets, h_cash, f"Fresh holdout ({n_h} stocks)")
    s2, c2 = summarise(m_rets, m_cash, f"Original test ({n_m} stocks)")
    summary, curves = pd.concat([s1, s2]), pd.concat([c1, c2])
    curves["date"] = pd.to_datetime(curves["date"]).dt.strftime("%Y-%m-%d")
    curves["value"] = curves["value"].round(5)
    summary.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False)
    curves.to_csv(RESULTS_DIR / "backtest_curves.csv", index=False)

    first, last = curves["date"].min(), curves["date"].max()
    (RESULTS_DIR / "backtest_note.txt").write_text(
        f"Period {first} to {last}. Warning thresholds fixed on 2019-20 validation data; "
        f"nothing tuned on these results. Ignores trading costs and taxes, which would hurt "
        f"the strategies that switch in and out of cash. Uses dividend-adjusted closes. "
        f"Cash earns 0%. Past results do not predict future results.")

    pd.set_option("display.width", 140)
    print(summary.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
