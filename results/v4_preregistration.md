# Model v4: pre-registration

Written and committed **before** any v4 candidate was built, trained or scored, and before
any holdout-2 price was downloaded. Nothing below may be changed after seeing results; any
deviation must be disclosed in `results/v4_holdout2_report.md`.

## Why a v4
v3 beats the volatility rule on the fresh holdout (PR-AUC 0.233 vs 0.216, bootstrap 99%).
We want to know whether a better starting point helps further. Idea fixed in advance: the
volatility rule only uses closing prices; daily high/low ranges give a more precise volatility
estimate (Parkinson 1980), and recent-weighted (EWMA) volatility reacts faster.

## Candidates (all fixed now)
All use the v3 recipe: LightGBM `init_score` = calibrated starting log-odds, trained on
2010-18, early stopping on 2019-20 (AUC-PR), same 16 v3 features unless stated.
| Name | Starting point (calibrated on 2010-18 only) | Tree features | Trees |
|---|---|---|---|
| V3 | logit of close-to-close volatility rule (= current v3) | v3 16 | leaves 3, min 2000 |
| P | logit of the same 2Φ formula using 20-day Parkinson (high-low) volatility | v3 16 | leaves 3, min 2000 |
| E | logistic regression on logits of 3 rules: close vol_20, Parkinson 20, EWMA (halflife 10 days) | v3 16 | leaves 3, min 2000 |
| X | as E | v3 16 + 5 new: park_ratio, down_ratio, max_drop_z, skew_60, park_20_rank | leaves 3, min 2000 |
| X7 | as E | as X | leaves 7, min 1000 |

New features (each uses data up to day t only; a leak test must pass):
park_20 = sqrt(mean over 20 days of ln(high/low)^2 / (4 ln 2)); ewm_vol = EWMA std of daily
log returns (halflife 10); park_ratio = park_20 / vol_20; down_ratio = 20-day downside
semideviation / vol_20; max_drop_z = worst daily log return in 20 days / vol_20;
skew_60 = 60-day skew of daily log returns; park_20_rank = same-day percentile of park_20
among the 99 main stocks.

## Selection rule (validation only)
v4 = the candidate with the highest **2019-20 validation PR-AUC**. If no candidate beats V3 on
validation, there is no v4 and the app stays on v3. Test (2021-25) and holdout data are not
used for selection in any way.

## The one test: holdout 2
- 96 large US stocks in neither the 99-stock universe nor the first holdout
  (`HOLDOUT2_TICKERS` in `src/common.py`). Stocks that fail to download are dropped and listed.
- Period 2021-01-01 to 2025-12-31, same 10%/20-day label. Same-day ranks are computed against
  the 99 main stocks (as for holdout 1).
- Scored **once** by `src/06g_holdout2.py`; the output is committed whatever it shows.

## Claims and decision rule (fixed now)
- Primary: v4 PR-AUC > volatility rule PR-AUC on holdout 2; monthly block bootstrap (1000x,
  seed 0) win share and 95% CI reported.
- Also reported: v3 vs the volatility rule on holdout 2 (a fresh test for v3 too), v4 vs v3,
  precision/recall/F1/accuracy at thresholds fixed on validation (red = same share of 2019-20
  days warned as the volatility rule), years won, stocks won.
- **The app switches from v3 to v4 only if v4's holdout-2 PR-AUC beats both v3's and the
  volatility rule's.** Otherwise the app stays on v3 and the v4 result is reported as a
  negative result.
