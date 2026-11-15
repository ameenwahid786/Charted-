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
    "Communication": ["GOOGL", "T", "VZ", "CMCSA", "DIS", "NFLX", "TMUS", "OMC",
                      "LYV", "TTWO"],
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


# ---------------------------------------------------------------------------
# Modules 2-5: labels, features, split, baselines
# ---------------------------------------------------------------------------
LABELS_PATH = DATA_DIR / "labels.parquet"      # module 2 output (rebuildable)
FEATURES_PATH = DATA_DIR / "features.parquet"  # module 3 output (rebuildable)

HORIZON = 20      # trading days to look ahead
DROP = -0.10      # a "crash" = lowest close in the window is >= 10% below today

# Fixed split (set before seeing results). GAP trading days are dropped at the
# end of train and val so no training label peeks into the next period.
TRAIN = ("2010-01-01", "2018-12-31")
VAL = ("2019-01-01", "2020-12-31")
TEST = ("2021-01-01", "2025-12-31")
GAP = HORIZON

# Features the model is allowed to use (all computed from data up to day t).
FEATURES = [
    "ret_5", "ret_20", "ret_60",
    "vol_20", "vol_60", "vol_ratio_20_60",
    "dist_ma50", "dist_ma200", "drawdown_252",
    "rsi_14", "volume_ratio_20",
    "vix", "vix_chg_5",
    "mkt_ret_20", "mkt_vol_20", "beta_60",
]


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PANEL_PATH)


def load_preds() -> pd.DataFrame:
    return pd.read_parquet(PREDS_PATH)


# ---------------------------------------------------------------------------
# Fresh holdout (module 6d): 50 large US stocks NOT in TICKERS, never used for
# training, tuning or the earlier test looks. Chosen before model v3 was tested.
# ---------------------------------------------------------------------------
HOLDOUT_SECTORS = {
    "Technology": ["AMD", "MU", "AMAT", "LRCX", "KLAC", "ADI", "INTU", "ADSK", "CTSH", "HPQ"],
    "Consumer": ["ROST", "YUM", "MAR", "EBAY", "ORLY", "AZO", "DHI", "LEN", "KR", "SYY",
                 "HSY", "CLX"],
    "Health Care": ["ISRG", "BDX", "SYK", "BSX", "CI", "HUM", "REGN", "VRTX"],
    "Financials": ["PNC", "TRV", "AIG", "MET", "PRU", "BK", "CME", "AFL"],
    "Industrials & Materials": ["EMR", "ETN", "ITW", "GD", "NOC", "CSX", "FDX", "NUE"],
    "Energy & Utilities": ["APA", "OKE", "EXC", "XEL"],
}
HOLDOUT_TICKERS = [t for g in HOLDOUT_SECTORS.values() for t in g]
assert not set(HOLDOUT_TICKERS) & set(TICKERS), "holdout overlaps the main universe"


# ---------------------------------------------------------------------------
# Holdout 2 (pre-registered for model v4, see results/v4_preregistration.md):
# another set of large US stocks, in neither TICKERS nor HOLDOUT_TICKERS, fixed
# BEFORE any v4 candidate was built or scored. Used exactly once, by 06g.
# ---------------------------------------------------------------------------
HOLDOUT2_SECTORS = {
    "Technology": ["MCHP", "NTAP", "GLW", "APH", "MSI", "PAYX", "ADP", "FIS", "CDNS", "SNPS",
                   "AKAM", "TER", "SWKS", "FTNT"],
    "Consumer": ["BBY", "DG", "DLTR", "GPC", "HAS", "EL", "TSN", "CAG", "HRL", "MKC", "STZ",
                 "CHD", "LUV", "DAL", "CMG", "ULTA", "TSCO"],
    "Health Care": ["BAX", "BIIB", "CAH", "MCK", "ZBH", "EW", "LH", "RMD", "A", "IDXX", "HCA",
                    "CNC"],
    "Financials": ["COF", "FITB", "KEY", "MTB", "NTRS", "STT", "TROW", "SPGI", "MCO", "MMC",
                   "AON", "ALL", "PGR", "CB", "HIG", "CINF", "BEN", "RF"],
    "Industrials": ["CMI", "PCAR", "PH", "ROK", "DOV", "SWK", "NSC", "WM", "CTAS", "FAST",
                    "GWW", "EXPD", "JBHT", "AME", "ROP"],
    "Energy, Utilities, Real Estate & Materials": [
        "PSX", "MPC", "KMI", "ED", "PEG", "SRE", "WEC", "ES", "DTE", "ETR", "PSA", "EQIX",
        "WELL", "DLR", "IP", "PPG", "IFF", "ALB", "MLM", "VMC"],
}
HOLDOUT2_TICKERS = [t for g in HOLDOUT2_SECTORS.values() for t in g]
assert not set(HOLDOUT2_TICKERS) & (set(TICKERS) | set(HOLDOUT_TICKERS)), "holdout 2 overlaps"
assert len(set(HOLDOUT2_TICKERS)) == len(HOLDOUT2_TICKERS)
