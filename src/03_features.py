"""Module 3: features (only past data -- no peeking).

    python src/03_features.py

Every feature on day t uses prices up to and including day t, never later.
A leak test at the end proves it: changing a FUTURE price must not change
any feature today.
Output: data/features.parquet (date, ticker, close, <FEATURES>)
"""
import numpy as np
import pandas as pd

from common import FEATURES, FEATURES_PATH, load_all, load_raw


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI: 0-100, high = recent gains dominate."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features for ONE stock (rows sorted by date)."""
    c, v = df["close"], df["volume"]
    r = np.log(c).diff()
    out = pd.DataFrame(index=df.index)
    out["ret_5"] = c / c.shift(5) - 1
    out["ret_20"] = c / c.shift(20) - 1
    out["ret_60"] = c / c.shift(60) - 1
    out["vol_20"] = r.rolling(20).std()           # daily volatility
    out["vol_60"] = r.rolling(60).std()
    out["vol_ratio_20_60"] = out["vol_20"] / out["vol_60"]
    out["dist_ma50"] = c / c.rolling(50).mean() - 1
    out["dist_ma200"] = c / c.rolling(200).mean() - 1
    out["drawdown_252"] = c / c.rolling(252).max() - 1
    out["rsi_14"] = rsi(c)
    out["volume_ratio_20"] = v / v.rolling(20).mean()
    out["_r"] = r                                  # used for beta, dropped later
    return out


def market_features() -> pd.DataFrame:
    gspc = load_raw("^GSPC").set_index("date")["close"]
    vix = load_raw("^VIX").set_index("date")["close"]
    mr = np.log(gspc).diff()
    m = pd.DataFrame({
        "vix": vix,
        "vix_chg_5": vix / vix.shift(5) - 1,
        "mkt_ret_20": gspc / gspc.shift(20) - 1,
        "mkt_vol_20": mr.rolling(20).std(),
        "_mr": mr,
    })
    return m.reset_index()


def build(prices: pd.DataFrame) -> pd.DataFrame:
    market = market_features()
    parts = []
    for _, g in prices.sort_values(["ticker", "date"]).groupby("ticker"):
        g = g.reset_index(drop=True)
        f = pd.concat([g[["date", "ticker", "close"]], stock_features(g)], axis=1)
        f = f.merge(market, on="date", how="left")
        # 60-day beta to the S&P 500 (past data only)
        f["beta_60"] = f["_r"].rolling(60).cov(f["_mr"]) / f["_mr"].rolling(60).var()
        parts.append(f)
    df = pd.concat(parts, ignore_index=True).drop(columns=["_r", "_mr"])
    df = df.replace([np.inf, -np.inf], np.nan)
    return df[["date", "ticker", "close"] + FEATURES]


def leak_test(prices: pd.DataFrame):
    """Shock prices AFTER a cutoff; features up to the cutoff must not change."""
    t = "AAPL"
    p = prices[prices["ticker"] == t].copy()
    cutoff = p["date"].iloc[1500]
    base = build(p)
    shocked = p.copy()
    after = shocked["date"] > cutoff
    shocked.loc[after, ["close", "volume"]] *= 0.5
    test = build(shocked)
    keep = base["date"] <= cutoff
    a = base.loc[keep, FEATURES].to_numpy()
    b = test.loc[keep, FEATURES].to_numpy()
    same = np.allclose(a, b, equal_nan=True)
    print(f"Leak test (future prices halved after {cutoff.date()}): "
          f"{'PASS - no feature looks ahead' if same else 'FAIL - a feature uses future data!'}")
    if not same:
        raise SystemExit(1)


def main():
    prices = load_all()
    df = build(prices)
    df.to_parquet(FEATURES_PATH, index=False)
    print(f"Saved {len(df):,} rows x {len(FEATURES)} features to {FEATURES_PATH.name}")
    ready = df[df["date"] >= "2010-01-01"]
    missing = ready[FEATURES].isna().mean().sort_values(ascending=False)
    print(f"Missing values from 2010 on (should be ~0): max {missing.max():.2%} ({missing.index[0]})")
    leak_test(prices)


if __name__ == "__main__":
    main()
