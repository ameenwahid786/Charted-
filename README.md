# Project name: Charted
**Track:** Finance
**Team:** Anees Ur-Rehman, Muhammad Ameen Wahid
## The problem
Many beginners buy or hold stocks without knowing how risky they are, which leaves their money exposed to losses they did not expect. Most risk tools are either too technical or give no explanation, so a beginner cannot tell which of their holdings is riskier than the others, leaving their investments vulnerable.

## What we built

A beginner enters a stock they plan to hold or buy. The tool returns the estimated chance that its close falls 10% or more below today’s price within 20 trading days as a risk metric, with a plain-English explanation of what is driving that number and how it compares with the typical stock. It helps someone see the risk before they commit their money, so they can make a safer decision. 

We compare our model against a simple baseline, a rule that warns when a stock’s 20-day volatility is in the top 20%, to see which predicts these drops better, using a PR-AUC.

This project is a prototype and is not financial advice

## How it works
Data: which dataset, how many examples, how we cleaned it.
Model: what we used and why.
## Results
Our baseline scored X. Our model scored Y on held-out test data.
Where it fails, and why we think so.
## What we'd do next
One or two concrete next steps with more time.
## Sources
Datasets, pretrained models, tutorials and AI tools used, with links.

---

## Running the code

### Fixed decisions (set before seeing any results)
- **Target:** the lowest close in the next 20 trading days is at least 10% below today's close.
- **Split:** train 2010–2018, validate 2019–2020, test 2021–2025, with a 20-trading-day gap between each.
- **Main metric:** PR-AUC on the test set. **Practical metric:** recall when both methods are set to the same precision.
- **Universe:** 99 large US stocks across 10 sectors (see `src/common.py`), plus ^GSPC and ^VIX, 2009-01-01 to 2026-03-01.

### Setup and data download
```bash
pip install -r requirements.txt
python src/01_download.py          # about 1–2 minutes; re-run if Yahoo rate-limits you
python src/01_download.py --check  # offline check: must print "DONE"
```

Only one person needs to download. They commit `data/raw/` and push it to GitHub; everyone else runs `git pull` so the whole team works from the same prices.

## Build the dataset (about 15 seconds, no internet)

```bash
python src/02_labels.py     # crash labels + crash rate per year
python src/03_features.py   # 16 past-only features + leak test (must PASS)
python src/04_split.py      # train 2010-18 / val 2019-20 / test 2021-25
python src/05_baselines.py  # baseline scores -> data/preds.parquet (val results only)
python src/06_model.py      # LightGBM, tuned on val -> score_model in preds.parquet
```

## Loading data in other modules

```python
from common import load_all, load_raw, TICKERS, TICKER_SECTOR
prices = load_all()          # columns: date, ticker, open, high, low, close, volume
gspc = load_raw("^GSPC")     # market series, merged on date
vix = load_raw("^VIX")
```

Each `*_score` column in `preds.parquet` must mean "higher = more risky".

*This project is a prototype and is not financial advice.*
