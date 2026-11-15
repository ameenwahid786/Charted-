**Live app: [https://charted.streamlit.app](https://charted.streamlit.app)** (opens in any browser, nothing to install)

# Project name: Charted
**Track:** Finance
**Team:** Anees Ur-Rehman, Muhammad Ameen Wahid
## The problem
Many beginners buy or hold stocks without knowing how risky they are, which leaves their money exposed to losses they did not expect. Most risk tools are either too technical or give no explanation, so a beginner cannot tell which of their holdings is riskier than the others, leaving their investments vulnerable. For example, both of our parents have strayed away from investing into stocks due to the risk involved - they need a tool that simply and clearly explains the risk, as well as a visual presentation to aid this.

## What we built

A beginner enters a stock they plan to hold or buy. The tool returns the estimated chance that its close falls 10% or more below today’s price within 20 trading days as a risk metric, with a plain-English explanation of what is driving that number and how it compares with the typical stock. It helps someone see the risk before they commit their money, so they can make a safer decision. 

We compare our model against 2 simple baselines, a rule that warns when a stock’s 20-day volatility is in the top 20%, and a 200-day moving average rule, to see which predicts these drops better, using a PR-AUC.

This project is a prototype and is not financial advice

# What we added

Our own idea

## How it works
**Data:** Our data that we trained and tested our model on comes from Yahoo Finance, using the yfinance library, and includes daily stock prices - high, low, open, close, and volume. The stocks were taken from the S&P 500 index, VIX, and 99 large US stocks across 10 sectors, with a time period of 2 Jan 2009 - 27 Feb 2026. 

The model was trained on stocks from 2010 - 2018 (222,156 examples), validated on stocks from 2019-2020(48,015 examples). The main test was carried out on data from 2021-2025(121,245 examples), and we conducted 2 additional tests on fresh test sets on stocks never encountered in training to see how the model performed. Both additional test sets were carried out on data from 2021-2025, with the stocks comprised of 49 other large US stocks for the first and 95 other additional stocks for the second (61,495 and 119,225 examples respectively). 

We included several checks to clean the data and prevent againast leakage:
   - Adjusted prices to prevent dividend and stock splits being interpreted as crashes
   - Validated the download to check for missing information, negaitve prices, and late starting dates
   - Included a warm up period (our data was collected from 2009 but we only started training on data from 2010 to ensure 1 year features have sufficient historical data)
   - Ensured that the final 20 trading days of each stock were excluded because they don't have a complete 20 day prediction window
   - Removed invalid values such as division by zero volume
   - Added 20 day gaps between training, validation, and test data periods to prevent data leakage

**Model:** We had 2 iterations of our model - our final model combines a simple volatility-based rule with a small LightGBM model to help learn corrections to its predictions. The volatility rule estimates the probability of a 10% drop, and the LightGBM model adjusts this probability upwards or downwards based on 16 additional signals. We chose LightGBM as it fits table-shaped data, which our model uses, better than alternatives, as well as being able to better explain the reasons behind its predictions (a key feature of our tool), as it can split each prediction into exact per-signal contributions.

Our v1 model only used the LightGBM model, and it matched the simple volatility baseline.

## Results
We used a Precision-Recall Area Under the Curve(PR-AUC) to evaluate our model against the baselines. Our results are as follows:

**What we predict:** will the stock's close fall 10% or more below today's close within the next 20 trading days? About 12% of days are "yes". Higher PR-AUC is better; guessing at random scores about 0.12. We use PR-AUC rather than accuracy because a model that never warns is 88% accurate and catches no crashes.

**Baselines:**
- **Volatility rule (main baseline):** a stock that swings a lot is more likely to drop 10%.
- **200-day average rule:** warns when the price is below its 200-day average.
- **Never warn:** never gives a warning.

### Model versions

| | v1 | v2 | v3 (final) |
|---|---|---|---|
| Model | LightGBM, 16 raw features | v1 retrained on more years | Volatility rule + LightGBM corrections, 16 features scaled by volatility and ranked against other stocks |
| Trained on | 2010–18 | 2010–20 | 2010–18 |
| Chosen how | Tuned on 2019–20 validation | Declared after seeing v1's test result | Best of 7 candidates, on validation only |
| Validation 2019–20 (vol rule 0.218) | 0.236 | n/a | **0.251** |
| Test 2021–25, 99 stocks (vol rule 0.191) | 0.192 (tie) | 0.182 (worse) | 0.204 (third look, disclosed) |
| Fresh holdout 1, 49 unseen stocks (vol rule 0.216) | 0.217 | not run | **0.233** |
| Fresh holdout 2, 95 unseen stocks (vol rule 0.169) | 0.180 | not run | **0.183** |

### Fresh holdout 1: 49 stocks never used anywhere, 2021–25 (main result)

| Method | PR-AUC | Precision | Recall | Accuracy |
|---|---|---|---|---|
| **Model v3** | **0.233** | 0.196 | **0.610** | 0.654 |
| Volatility rule | 0.216 | 0.198 | 0.596 | 0.663 |
| Model v1 | 0.217 | 0.191 | 0.563 | 0.662 |
| 200-day average rule | 0.145 | 0.153 | 0.411 | 0.657 |
| Never warn | 0.120 | 0 | 0 | **0.880** |

- **Resampling test:** we redrew the data by month 1,000 times. v3 beats the volatility rule in 98.8% of redraws. Average gap +0.017 (95% range +0.004 to +0.029).
- v3 won 4 of 5 years (it lost 2021 by 0.001) and had the higher score on 21 of the 49 stocks.
- v3 beats the 200-day rule in 100% of redraws.

### Fresh holdout 2: 95 more unseen stocks, 2021–25 (claim registered before running)

| Method | PR-AUC | Precision | Recall |
|---|---|---|---|
| **Model v3** | **0.183** | **0.182** | **0.524** |
| Volatility rule | 0.169 | 0.172 | 0.461 |
| Model v1 | 0.180 | 0.180 | 0.492 |
| 200-day average rule | 0.151 | 0.157 | 0.473 |
| Never warn | 0.125 | 0 | 0 |

- **Resampling test:** v3 wins 99.8% of redraws (95% range +0.007 to +0.022).
- v3 won 5 of 5 years and had the higher score on 39 of the 95 stocks.

### Original test set: 99 stocks, 2021–25

| Method | PR-AUC | Precision | Recall |
|---|---|---|---|
| Model v1 | 0.192 | 0.190 | 0.505 |
| Model v2 | 0.182 | 0.185 | 0.530 |
| Volatility rule | 0.191 | 0.190 | 0.506 |
| 200-day average rule | 0.138 | 0.142 | 0.428 |
| Never warn | 0.119 (accuracy 0.881) | 0 | 0 |
| Model v3 (third look) | 0.204 | n/a | n/a |

### v4 attempt
We registered 4 more ideas in advance, built on better volatility estimates. None beat v3 on validation (best 0.241 vs 0.251), so v3 stayed as the final model.

### Backtest: equal weight, a stock is held as cash while its warning is on

| Fresh holdout 1, 2021–25 | Total return | Worst fall (max drawdown) | Time in cash |
|---|---|---|---|
| Buy and hold | **+123%** | −17.6% | 0% |
| 200-day rule | +50% | −11.4% | 32% |
| Volatility rule | +45% | −6.4% | 36% |
| Model v3 | +37% | −6.8% | 37% |

Following the warnings cut the worst fall from −17.6% to under −7%, but it lost to simply holding, because the stocks recovered while the money sat in cash.

Although our model outperformed the baselines, it still has its limitations:

- It only had limited improvement over the volatility rule (+0.013 PR-AUC for the test data, +0.017 for the first set of additional test data, and +0.015 for the second set of additional test data), because the historical price volaitlity already contains most information associated with future 10% drops, so our model only offers limited improvement
- There was also a very high false-positive rate - approximately 80% of warnings were not followed by a 10%+ drop, as large price drops are relatively uncommon. A warning simply indicates an increased probablity, not certainty. To see the consequences of this, we conducted a stress test by selling whenever a warning was generated, and compared it to Buy and hold. Our results showed that although the worst fall was ony 7% compared to Buy and Hold's 18%, the final return was much lower at 37% in comparison to a 123% return from Buy and Hold.
- Out of the stocks analysed, most experienced their worst days due to company-specific events(e.g. UnitedHealth's fall of ~22% in April 2025 due to its earnings announcement report), whereas our model relies exclusively on market-price information. It cannot directly account for earnings announcements, regulatory developments, or unusual company news.
- There is also survivorship bias, as the dataset consists of companies that are large and established today - companies that previously failed or delisted were not adequetly represented. This could cause the model to underestimate real-world market risk

## What we'd do next
Although it would be hard to significantly improve the model's performance, some improvements to the model could be made with more time. Some examples are:
- Including company-specific information such as earnings annoucement dates, news sentiment, and announcements to help address one of the model's weaknesses as listed above.
- Addressing survivorship bias by reconstructing historical stock-index membership, and including companies that fell out or were delisted. Training the model on data including these stocks would produce a more representative assessment of historical market risk, improving its performance.

## Sources

**Data**
- **Yahoo Finance**, via the `yfinance` Python library: daily prices for 99 large US stocks plus 144 holdout stocks, the S&P 500 (^GSPC) and the VIX, 2009–2026. https://finance.yahoo.com and https://github.com/ranaroussi/yfinance
- **Google News RSS search**, for the News page headlines (free, no key). https://news.google.com

**Pretrained models:** none. All models were trained from scratch on the data above.

**Libraries and tools**
- LightGBM (gradient-boosted trees): https://lightgbm.readthedocs.io. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017.
- scikit-learn (PR-AUC, precision/recall, logistic-regression calibration): https://scikit-learn.org
- pandas, NumPy, SciPy, PyArrow: https://pandas.pydata.org, https://numpy.org, https://scipy.org, https://arrow.apache.org
- Streamlit and Streamlit Community Cloud (web app and hosting): https://streamlit.io
- Plotly (interactive charts): https://plotly.com/python
- IBM Plex Sans (font): https://fonts.google.com/specimen/IBM+Plex+Sans

**Methods we drew on**
- Platt scaling, for turning the volatility rule into a probability: Platt, J. (1999), "Probabilistic Outputs for Support Vector Machines".
- Gaps between train, validation and test to stop overlapping labels leaking: López de Prado, M. (2018), *Advances in Financial Machine Learning*.
- Month-block bootstrap for the significance test: Künsch, H. (1989), "The Jackknife and the Bootstrap for General Stationary Observations".

**AI tools**
- **Claude (Anthropic)**, through Claude Code, helped write the code, the app and the documentation. https://claude.ai

*This project is a prototype and is not financial advice.*
