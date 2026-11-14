# drawdown-guard

Warns when a large US stock is likely to fall 10% or more within the next 20 trading days, and checks whether that warning beats simple rules.

## Fixed decisions (set before seeing any results)

- **Target:** the lowest close in the next 20 trading days is at least 10% below today's close.
- **Split:** train 2010–2018, validate 2019–2020, test 2021–2025, with a 20-trading-day gap between each.
- **Main metric:** PR-AUC on the test set. **Practical metric:** recall when both methods are set to the same precision.
- **Universe:** 99 large US stocks across 10 sectors (see `src/common.py`), plus ^GSPC and ^VIX.
- **Data window:** 2009-01-01 to 2026-03-01 (exclusive). Both dates are fixed so everyone gets the same data, and the end date leaves room for the late-2025 labels.

## Setup

```bash
pip install -r requirements.txt
python src/01_download.py          # about 1–2 minutes; re-run if Yahoo rate-limits you
python src/01_download.py --check  # offline check: must print "DONE"
```

Only one person needs to download. They commit `data/raw/` and push it to GitHub; everyone else runs `git pull` so the whole team works from the same prices.

## Loading data in other modules

```python
from common import load_all, load_raw, TICKERS, TICKER_SECTOR
prices = load_all()          # columns: date, ticker, open, high, low, close, volume
gspc = load_raw("^GSPC")     # market series, merged on date
vix = load_raw("^VIX")
```

Each `*_score` column in `preds.parquet` must mean "higher = more risky".

*This project is a prototype and is not financial advice.*
