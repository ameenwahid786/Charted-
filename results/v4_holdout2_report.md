# Model v4 and holdout 2: report

Follows `results/v4_preregistration.md` (committed before any v4 work). No deviations,
except that one holdout-2 stock (MMC) failed to download and was dropped, as the plan allows.

## v4 selection (validation 2019-20 only)
| Candidate | Val PR-AUC | Starting point alone |
|---|---|---|
| V3 (current v3, retrained: reproduces 0.251) | **0.2508** | 0.2181 |
| P  Parkinson high-low volatility start | 0.2411 | 0.2105 |
| E  3-rule calibrated start | 0.2391 | 0.2127 |
| X  E + 5 new features | 0.2397 | 0.2127 |
| X7 X with bigger trees | 0.2298 | 0.2127 |
| Volatility rule | 0.2181 | |

No candidate beat v3 on validation, so **there is no v4** and the app stays on v3. The better
volatility estimates did not help as starting points (even alone they scored below the
close-to-close rule on validation). This is a negative result, and we report it as one.

## Holdout 2: a second independent test of v3 (run once)
95 large US stocks in neither the main universe nor holdout 1, 2021-25, 119,225 stock-days,
crash rate 12.5%. Thresholds fixed on validation.

| Method | PR-AUC | Precision | Recall | F1 | Accuracy | Share warned |
|---|---|---|---|---|---|---|
| **Model v3** | **0.183** | **0.182** | **0.524** | **0.270** | 0.645 | 0.361 |
| Volatility rule | 0.169 | 0.172 | 0.461 | 0.251 | 0.655 | 0.335 |
| 200-day average rule | 0.151 | 0.157 | 0.473 | 0.236 | 0.616 | 0.377 |
| Model v1 | 0.180 | 0.180 | 0.492 | 0.264 | 0.656 | 0.341 |
| Never warn | 0.125 | 0 | 0 | 0 | 0.875 | 0 |

Monthly block bootstrap (1000x): v3 beats the volatility rule in **99.8%** of resamples, gap
+0.015 (95% CI +0.007 to +0.022). Years won **5 of 5**. Stocks won 39 of 95 (as on holdout
1, the edge comes from ranking across stocks and dates, not from winning on every stock).
