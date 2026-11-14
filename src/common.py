"""Shared constants and loaders for drawdown-guard.

Every other module should import from here instead of re-defining paths or
ticker lists, e.g.:

    from common import load_all, TICKERS, MARKET_TICKERS

(Scripts named 01_..., 02_... can't be imported, which is why this lives in
its own module.)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"
PANEL_PATH = DATA_DIR / "panel.parquet"
PREDS_PATH = DATA_DIR / "preds.parquet"
MANIFEST_PATH = RAW_DIR / "_manifest.csv"

# Fixed so every teammate gets identical data. END is exclusive and runs past
# 2025-12-31 so the 20-day forward labels for late-2025 test rows are known.
START = "2009-01-01"
END = "2026-03-01"

# ~100 large US stocks, 10 sectors. All were listed before 2009 so the
# 200-day features are ready by 2010. (Survivorship bias: these are today's
# survivors -- note it in the write-up.)
SECTORS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "ORCL", "CSCO", "INTC", "ADBE", "CRM",
                   "QCOM", "TXN", "IBM", "ACN"],
    "Communication": ["GOOGL", "T", "VZ", "CMCSA", "DIS", "NFLX", "EA", "OMC",
                      "IPG", "TTWO"],
    "Consumer Discretionary": ["AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT",
                               "BKNG", "TJX", "F"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MO", "KMB",
                         "GIS", "MDLZ"],
    "Health Care": ["JNJ", "UNH", "PFE", "MRK", "ABT", "TMO", "AMGN", "GILD",
                    "BMY", "MDT", "LLY", "CVS"],
    "Financials": ["JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "SCHW",
                   "USB", "V", "MA"],
    "Industrials": ["GE", "CAT", "HON", "UNP", "UPS", "MMM", "BA", "LMT", "DE",
                    "RTX"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "VLO", "HAL", "DVN",
               "WMB"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "Materials & Real Estate": ["APD", "SHW", "ECL", "NEM", "FCX", "AMT", "PLD",
                                "SPG"],
}
TICKERS = [t for group in SECTORS.values() for t in group]
TICKER_SECTOR = {t: s for s, group in SECTORS.items() for t in group}

# Market-wide series used as features, not as stocks to predict.
MARKET_TICKERS = ["^GSPC", "^VIX"]
ALL_SYMBOLS = TICKERS + MARKET_TICKERS

PRICE_COLS = ["open", "high", "low", "close", "volume"]


def csv_path(symbol: str) -> Path:
    """'^GSPC' -> data/raw/IDX_GSPC.csv (avoids '^' in file names)."""
    return RAW_DIR / f"{symbol.replace('^', 'IDX_')}.csv"


def load_raw(symbol: str) -> pd.DataFrame:
    """One symbol from disk: columns date, open, high, low, close, volume."""
    path = csv_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"{symbol}: {path} missing -- run src/01_download.py")
    return pd.read_csv(path, parse_dates=["date"])


def load_all(symbols=None) -> pd.DataFrame:
    """Long panel of stocks: one row per (date, ticker), sorted.

    Defaults to the stock list only (not ^GSPC/^VIX); load those with
    load_raw('^GSPC') and merge on date.
    """
    symbols = TICKERS if symbols is None else symbols
    frames = []
    for s in symbols:
        df = load_raw(s)
        df.insert(1, "ticker", s)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)
