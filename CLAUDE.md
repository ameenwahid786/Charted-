# CLAUDE.md — Charted (drawdown-guard) handoff

Read this whole file before doing anything. It summarises everything decided and built so far.

## The hackathon brief (non-negotiable)
Track: AI in finance, 2 days. "Build an AI tool that helps a person or a small business make a safer or fairer
money decision, and show that it beats a simple rule."
- Compare the model with at least one simple baseline and report BOTH results.
- Rare events → report precision and recall, not accuracy.
- Data has dates → train on earlier data, test on later data.
- The write-up AND the app must contain exactly: **"This project is a prototype and is not financial advice."**
- **A judge must be able to fully use the web app from a link, with nothing installed.** (Hosted on Streamlit Community Cloud.)

## The product
Crash-risk early warning for beginner investors. The user picks stocks they hold or plan to buy; the app shows the
estimated chance each one's close falls ≥10% below today's close within the next 20 trading days, explains why in
plain English, and compares it with the typical stock and with a simple rule.

Team: Muhammad Ameen Wahid (owner, GitHub `ameenwahid786`), Anees Ur-Rehman. Repo: github.com/ameenwahid786/Charted-
(local folder `~/Downloads/drawdown-guard`). Anees owns the write-up part of README.md; edit only the
"Running the code" section unless asked.

## Fixed decisions
- Target: label=1 if min close over days t+1..t+20 ≤ 0.90 × close_t (`src/02_labels.py`).
- Universe: 99 large US stocks (`TICKERS` in `src/common.py`), plus ^GSPC and ^VIX. Data 2009-01-02 → 2026-02-27 in `data/raw/*.csv` (committed).
- Split: train 2010–18, val 2019–20, test 2021–25, 20-trading-day purge gaps (`src/04_split.py`).
- Main metric PR-AUC; practical metric recall at matched precision. Thresholds are ALWAYS fixed on validation, never on test.
- Main baseline: **volatility rule** — score = 2·Φ(ln 0.9 / (vol_20·√20)); yes/no rule = vol_20 above the TRAINING 80th
  percentile. Also reported: 200-day moving-average rule, never-warn.

## Pipeline (all done, all working)
```
python src/01_download.py            # (+ --holdout for the 49 fresh stocks) needs internet
python src/02_labels.py              # crash rate 10.4%
python src/03_features.py            # 16 past-only features, leak test PASSES
python src/04_split.py
python src/05_baselines.py           # -> data/preds.parquet
python src/06_model.py               # v1
python src/06b_refit.py              # v2
python src/07_evaluate.py            # v1/v2 test eval -> results/ (pr_curve.png, metrics_test.csv ...)
python src/06c_experiments.py        # 7 candidates, VALIDATION ONLY
python src/06d_train_v3.py           # v3 -> models/lgbm_v3.txt + lgbm_v3.json, score_model_v3 in preds
python src/06e_holdout.py            # v3 on fresh holdout -> results/holdout_*
```
data/labels.parquet, features.parquet, panel.parquet are gitignored (rebuilt in ~15 s by 02–04).
Committed: data/raw/, data/preds.parquet, models/, results/.
`src/common.py` holds paths, tickers, FEATURES, split dates. `src/model_v3.py` holds `add_v3_features`, `predict_v3`,
`vol_prob`, `load_v3` — **reuse these, don't re-implement.** Scripts named `0X_*.py` can be imported with
`importlib.import_module("03_features")` (see 06e).

## Model history and results (tell this story honestly everywhere)
| | What | Result |
|---|---|---|
| v1 | LightGBM, 16 features, trained 2010–18, tuned on val | val 0.236; **test 2021–25: 0.192 vs vol rule 0.191 = tie** |
| v2 | v1 refit on 2010–20 (declared after seeing v1) | test 0.182 — worse; raw feature levels shift between market regimes |
| v3 | "volatility rule + learned corrections": LightGBM init_score = Platt-calibrated logit of the vol rule; features volatility-scaled + same-day percentile ranks; 3 leaves, min 2000/leaf, 54 trees; chosen among 7 candidates on validation only | val 0.251 vs 0.218 |

**Headline result — fresh holdout of 49 stocks never used anywhere (`HOLDOUT_TICKERS`, BK failed to download), 2021–25:**
v3 PR-AUC **0.233** vs volatility rule **0.216** vs v1 0.217 vs 200-day rule 0.145 vs no skill 0.120.
Monthly block bootstrap (1000×): v3 wins **99%**, gap +0.017, 95% CI +0.004 to +0.029. Years won **4 of 5** (2021 lost by
0.001). Stocks won **21 of 49** (report honestly: edge comes from ranking across stocks/dates, not within every stock).
Original test (third look, disclosed): v3 0.204 vs 0.191. Accuracy trap: never-warn has the best accuracy (0.88) and 0 recall.
**v4 attempt (pre-registered, `results/v4_preregistration.md`):** 4 candidates with Parkinson high-low / EWMA volatility
starting points (`src/model_v4.py`, `src/06f_v4_experiments.py`); none beat v3 on validation (best 0.241 vs 0.251) → no v4,
app stays on v3. **Holdout 2** (95 more unseen stocks, `HOLDOUT2_TICKERS`, MMC failed; `src/06g_holdout2.py`, run once):
v3 0.183 vs vol rule 0.169, bootstrap 99.8% (CI +0.007 to +0.022), years 5/5, stocks 39/95, precision 0.182 vs 0.172,
recall 0.524 vs 0.461. Report: `results/v4_holdout2_report.md`. Backtest (09): warnings cut max drawdown but lose to buy-and-hold.
Feature importance (v1): vol_60 42%, vol_20 24%, drawdown_252 9%, beta_60 8%.

## What is left to build
### A. `src/08_app_data.py` — precompute everything the app needs (committed, small)
The server must NOT train, download from Yahoo, or rebuild the big parquet files. Write a compact file, e.g.
`data/app/app_data.parquet` (+ json), for all 99 stocks AND the 49 holdout stocks:
- daily close, v3 probability, vol-rule probability, label (where known), for the last ~3 years
- v3 feature values and **per-feature contributions** via `booster.predict(X, pred_contrib=True)` (no `shap` package
  needed; it's exact for LightGBM). The starting point is the vol rule's calibrated log-odds; contributions are the
  corrections.
- **v3 thresholds fixed on validation**: red = v3 warns on the same share of val days as the vol rule; amber = twice that
  share. Save to `results/thresholds_v3.json` (the existing thresholds.json is for v1 — don't use it for the app).
- Typical-stock base rate (~12%) and each stock's sector.
"As of" date = last date with features (27 Feb 2026). Label it clearly as a data snapshot.

### B. `src/09_backtest.py` (module 9)
Equal-weight portfolio on the holdout stocks (and the 99 on the 2021–25 test) — hold a stock in cash while its warning
is on. Compare v3, volatility rule, 200-day rule, buy-and-hold: total return, max drawdown, % time in cash.
Save `results/backtest_*.csv` for the app. Report whatever it shows; no tuning.

### C. `app/app.py` — Streamlit web app, **dark black UI**
`.streamlit/config.toml`:
```
[theme]
base = "dark"
backgroundColor = "#000000"
secondaryBackgroundColor = "#111111"
textColor = "#f0efec"
primaryColor = "#3987e5"
font = "sans serif"
```
Plotly charts: `paper_bgcolor`/`plot_bgcolor` "#000000", grid "#262626", text "#c3c2b7", thin 2 px lines, hover
tooltips on every chart, a legend whenever ≥2 series. Series colours (dark-mode validated palette, fixed order):
v3 `#3987e5` (blue), volatility rule `#d95926` (orange), 200-day rule `#199e70` (aqua), v1 `#c98500` (yellow, dashed),
no skill grey dotted. Risk status: red / amber / green **always with an icon and a word** (never colour alone).

Pages (sidebar navigation):
1. **Check my stocks** — multiselect (search by ticker/company), default e.g. AAPL, NVDA, KO. One card per stock:
   v3 chance of a 10% drop in the next month (big number), status badge, "typical stock: 12%", "simple volatility
   rule says X%". Sort cards riskiest first.
2. **Stock detail** — price chart (last 2–3 years) with v3 red-warning periods shaded and actual 10%+ drops marked;
   "Why?" panel: start = "Price swings alone suggest X%" then the top 3 corrections in plain English from the
   contributions (template per feature, e.g. vol_20_rank → "It is more volatile than N% of large US stocks today";
   drawdown_z → "It is already well below its 1-year high"; vix → "The whole market is nervous (VIX = N)"), with arrows
   ↑/↓ for raising/lowering risk. A short "what this means for a beginner" line. No buy/sell advice.
3. **Does it beat a simple rule?** — interactive holdout PR curve (`results/holdout_pr_curves.csv`), metrics table
   (`holdout_metrics.csv`) with precision/recall/F1/accuracy and a note on why accuracy misleads, bootstrap result
   (`holdout_bootstrap.json`), by-year bars (`holdout_by_year.csv`), stocks-won honest note, the v1→v2→v3 story, the
   original-test results (`metrics_test.csv`, `results/pr_curve.png`), and the backtest.
4. **How it works** — beginner-friendly: what the 10%/20-day label means, what volatility is, what PR-AUC/precision/
   recall mean, train/val/test and the fresh holdout, limitations (survivorship bias: EA/IPG/BK dropped out of free data;
   overlapping 20-day labels; false alarms can make someone sell too early; data snapshot not live).
Footer on every page: "This project is a prototype and is not financial advice."
Load data with `@st.cache_data`; app must start in a few seconds. Add `src/` to `sys.path` from `app/app.py` via
`Path(__file__).resolve().parents[1] / "src"`. Use relative paths from `common.ROOT` only.

### D. Deployment (judge must be able to use it)
- `requirements.txt` (pandas, numpy, pyarrow, scipy, scikit-learn, lightgbm, plotly, streamlit; matplotlib only if used).
  Don't import yfinance/shap in the app. `packages.txt` already contains `libgomp1` (LightGBM on Linux).
- Ameen deploys at share.streamlit.io → New app → repo ameenwahid786/Charted-, branch main, file app/app.py.
  Repo may need to be public. Put the app link at the top of README.md.
- Test locally first: `streamlit run app/app.py`. Then check the hosted app in a fresh browser.
- Record a 1–2 min backup screen recording.

### E. Write-up (README results section, with Anees)
Fill "How it works / Results / Where it fails / What we'd do next / Sources" with the numbers above, both baselines,
precision & recall, the full v1/v2/v3 history, the holdout + bootstrap, limitations, and the required disclaimer line.
Sources: Yahoo Finance via yfinance, LightGBM, scikit-learn, Streamlit, Plotly, Claude (AI assistant).

## Environment & workflow
- Ameen's Mac: Python 3.9 in `.venv` (`source .venv/bin/activate`). No `X | Y` type hints, no `match`. Homebrew +
  `brew install libomp` done (LightGBM needs it). The urllib3 LibreSSL warning is harmless.
- Git: `git pull --rebase origin main` before starting, small commits, `git push` after each module. Never `--force`
  (Anees pushes too). Never commit `.venv/`.
- Integrity rules: never tune or pick anything using test or holdout results; every threshold comes from validation;
  report results whatever they are.
