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

Our baseline scored X. Our model scored Y on held-out test data.

Although our model outperformed the baselines, it still has its limitations:

- It only had limited improvement over the volatility rule (+0.013 PR-AUC for the test data, +0.017 for the first set of additional test data, and +0.015 for the second set of additional test data), because the historical price volaitlity already contains most information associated with future 10% drops, so our model only offers limited improvement
- There was also a very high false-positive rate - approximately 80% of warnings were not followed by a 10%+ drop, as large price drops are relatively uncommon. A warning simply indicates an increased probablity, not certainty. To see the consequences of this, we conducted a stress test by selling whenever a warning was generated, and compared it to Buy and hold. Our results showed that although the worst fall was ony 7% compared to Buy and Hold's 18%, the final return was much lower at 37% in comparison to a 123% return from Buy and Hold.
- Out of the stocks analysed, most experienced their worst days due to company-specific events, whereas our model relies exclusively on market-price information. It cannot directly account for earnings announcements, regulatory developments, or unusual company news.
- There is also survivorship bias, as the dataset consists of companies that are large and established today - companies that previously failed or delisted were not adequetly represented. This could cause the model to underestimate real-world market risk

## What we'd do next
Although it would be hard to significantly improve the model's performance, some improvements to the model could be made with more time. Some examples are:
- Including company-specific information such as earnings annoucement dates, news sentiment, and announcements to help address one of the model's weaknesses as listed above.
- Addressing survivorship bias by reconstructing historical stock-index membership, and including companies that fell out or were delisted. Training the model on data including these stocks would produce a more representative assessment of historical market risk, improving its performance.

## Sources
