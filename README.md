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


### Model history (full transparency)
1. **v1** (LightGBM, 16 features, trained 2010-18, tuned on 2019-20) was fixed in advance. On the 2021-25 test set it **tied** the volatility rule (PR-AUC 0.192 vs 0.191).
2. **v2** (same model refit on 2010-20) was declared after seeing v1's result. It did worse (0.182).
3. **v3** ("volatility rule + learned corrections", volatility-scaled and same-day-ranked features) was chosen among 7 candidates **on validation only** (0.251 vs the rule's 0.218). Because the 2021-25 test set had already been looked at twice, v3's main test is a **fresh holdout: 50 different large US stocks over 2021-25** (`HOLDOUT_TICKERS` in `src/common.py`), fixed before v3 was ever scored on them. Pre-registered claim: v3 beats the volatility rule on holdout PR-AUC, with the monthly block bootstrap win share reported whatever it is. v3's score on the original test set is also reported, labelled as a third look.

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
python src/06_model.py      # LightGBM v1, tuned on val -> score_model
python src/06b_refit.py     # declared v2 refit on 2010-20 -> score_model_v2
python src/07_evaluate.py   # ONE-TIME test evaluation -> results/ (PR curve, metrics)
python src/06c_experiments.py  # candidate models, compared on validation only
python src/06d_train_v3.py     # train + save model v3 -> score_model_v3
python src/01_download.py --holdout  # needs internet: 50 fresh holdout stocks
python src/06e_holdout.py      # v3 vs volatility rule on the fresh holdout + bootstrap
python src/06f_v4_experiments.py  # pre-registered v4 candidates, validation only (none beat v3)
python src/01_download.py --holdout2  # needs internet: 96 more holdout stocks (pre-registered)
python src/06g_holdout2.py     # second fresh test of v3 (run once)
python src/06h_baselines.py    # v3 vs every baseline, year by year, all tests
python src/01b_download_live.py  # needs internet: app snapshot prices to the latest close
                                 # (data/raw_live/ + one-minute bars); data/raw/ is untouched
python src/08_app_data.py      # small precomputed app files -> data/app/ + results/thresholds_v3.json
python src/09_backtest.py      # warnings-as-cash backtest -> results/backtest_*.csv
python src/10_news.py          # needs internet, ~10 min: Google News headlines around big drops
```

`requirements.txt` is what the web app needs; `pip install -r requirements-dev.txt` adds
`yfinance` and `matplotlib` for re-running the whole pipeline.

## Web app

```bash
streamlit run app/app.py
```

The app only reads the small committed files in `data/app/` and `results/`. It never trains,
downloads prices or rebuilds the big parquet files, so it starts in a few seconds.
Theme: `.streamlit/config.toml`. LightGBM's Linux dependency: `packages.txt` (`libgomp1`).

**Deploy (Streamlit Community Cloud):** share.streamlit.io → *Create app* → repository
`ameenwahid786/Charted-`, branch `main`, main file `app/app.py` → *Deploy*. The repo must be
public (or the Streamlit account connected to GitHub with access to it). After any change to
the pipeline, re-run `08_app_data.py` and `09_backtest.py`, commit, and push; the hosted app
redeploys automatically.

## Loading data in other modules

```python
from common import load_all, load_raw, TICKERS, TICKER_SECTOR
prices = load_all()          # columns: date, ticker, open, high, low, close, volume
gspc = load_raw("^GSPC")     # market series, merged on date
vix = load_raw("^VIX")
```

Each `*_score` column in `preds.parquet` must mean "higher = more risky".

*This project is a prototype and is not financial advice.*
