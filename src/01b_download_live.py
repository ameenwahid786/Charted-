"""Module 1b: download the app's price snapshot (needs internet).

    python src/01b_download_live.py

The research pipeline uses data/raw/ (fixed, up to 27 Feb 2026) and is not touched.
This script refreshes only what the web app shows:
  * daily prices from 2022 to the latest close -> data/raw_live/<symbol>.csv
  * 1-minute bars for the last 5 sessions      -> data/app/app_intraday.parquet
    (1H / 1D / 1W charts; Yahoo keeps minute data for recent weeks only)
Covers the 99 main stocks, the first holdout stocks, ^GSPC and ^VIX.
"""
import importlib
import time

import pandas as pd

from common import (DATA_DIR, HOLDOUT_TICKERS, LIVE_DIR, LIVE_START, MARKET_TICKERS, TICKERS,
                    csv_path, live_csv_path)

dl = importlib.import_module("01_download")
APP_DIR = DATA_DIR / "app"
BATCH, PAUSE = 25, 2.0


def symbols():
    hold = [t for t in HOLDOUT_TICKERS if csv_path(t).exists()]  # same 148 stocks as the app
    return TICKERS + hold, MARKET_TICKERS


def daily(stocks):
    import yfinance as yf
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    failed = []
    for i in range(0, len(stocks), BATCH):
        batch = stocks[i:i + BATCH]
        raw = yf.download(batch, start=LIVE_START, end=end, auto_adjust=True, group_by="ticker",
                          threads=True, progress=False)
        got = dl.split_batch(raw, batch)
        for s in batch:
            if s in got:
                got[s].to_csv(live_csv_path(s), index=False)
            else:
                failed.append(s)
        time.sleep(PAUSE)
    return failed


def intraday(stocks):
    import yfinance as yf
    parts = []
    for i in range(0, len(stocks), BATCH):
        batch = stocks[i:i + BATCH]
        raw = yf.download(batch, period="5d", interval="1m", auto_adjust=True, group_by="ticker",
                          threads=True, progress=False, prepost=False)
        for s in batch:
            try:
                d = raw[s].copy()
            except KeyError:
                continue
            d.columns = [str(c).lower() for c in d.columns]
            d = d.dropna(subset=["close"])
            if d.empty:
                continue
            d.index = pd.to_datetime(d.index).tz_convert("America/New_York").tz_localize(None)
            d = d.rename_axis("ts").reset_index()[["ts", "open", "high", "low", "close", "volume"]]
            parts.append(d.assign(ticker=s))
        time.sleep(PAUSE)
    out = pd.concat(parts, ignore_index=True)
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = out[c].astype("float32")
    APP_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(APP_DIR / "app_intraday.parquet", index=False)
    return out


def main():
    stocks, market = symbols()
    failed = daily(stocks + market)
    if failed:
        print(f"Daily download failed for: {failed}")
    last = {s: pd.read_csv(live_csv_path(s), usecols=["date"])["date"].max()
            for s in stocks + market if live_csv_path(s).exists()}
    latest = max(last.values())
    behind = sorted(s for s, d in last.items() if d < latest)
    print(f"Daily: {len(last)} symbols, latest close {latest}. Behind latest: {behind or 'none'}")
    intr = intraday(stocks)
    print(f"Intraday: {intr['ticker'].nunique()} stocks, {len(intr):,} one-minute bars, "
          f"{intr['ts'].min()} to {intr['ts'].max()}")


if __name__ == "__main__":
    main()
