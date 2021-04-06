# FINM33150 - Futures Spreads

## Dependencies

Create a virtual environment and install dependencies with

    pipenv install

## Analysis

The two futures spreads analyzed are:
* [10-Year T-Note Futures](https://www.cmegroup.com/trading/interest-rates/us-treasury/10-year-us-treasury-note_contractSpecs_futures.html) over the [Five-Year T-Note Futures](https://www.cmegroup.com/trading/interest-rates/us-treasury/5-year-us-treasury-note_contractSpecs_futures.html) (CBT:TY-FV throughout)
* [Brent Crude Futures](https://www.theice.com/products/219/Brent-Crude-Futures) over [Low Sulphur Gasoil Futures](https://www.theice.com/products/34361119/Low-Sulphur-Gasoil-Futures) (ICE:G-B throughout)

The prices are calculated using second month contracts (earliest expiration greater than 30 days) using quarterly contract expirations for the period from 2018-12-03 to 2020-08-31. Discussion and analysis of the following exhibits is included:
* Spread charts
* Summary statistics
* Histograms of daily returns compared to normal distributions
* Q-Q plots
* Rolling kurtosis charts
* Statistics of rolling average differences

### Spread Charts
A significant feature of both sets of charts is the relatively large change in performance that occurs beginning in the first quarter of 2020, presumably as a result of the pandemic. The effect is seen more clearly in the ICE:G-B spread than it is in the CBT:TY-FV spread, but is evident in the CBT:TY-FV spread nonetheless, and clearly evident in all of the underlying securities. If the objective of this analysis was to form a basis for a reversion to the mean trading strategy and the there was no expectation of there being similar shocks to the system during our investment horizon, it may be appropriate to exclude this period from the analysis.

### Summary Statistics
Across the entire period, CBT:TY-FV had a mean daily return of 0.001511 with a standard deviation of 0.018785. ICE:G-B was much more volatile, with a standard deviation of 0.13972 and a mean of -0.000871. That additional volatility can also be seen in the relative minimum and maximum daily returns. Whereas CBT:TY-FV had a min and max of -0.101946 and 0.084208, respectively, ICE:G-B had a min and max of -0.939851 and 0.719610, respectively. The range of ICE:G-B is 1.659461, which is 8.9 times greater than the range of 0.186154 for CBT:TY-FV. The range between the 25th percentile and the 75th percentile for ICE:G-B is still 4.4 times that of CBT:TY-FV. One last point to note is that ICE:G-B had only a slightly negative mean, even though it declined significantly over the period. This indicates there were a small number of observations with relatively large negative returns. For example, as can be seen in the chart above, ICE:G-B declined 51.0%, from $13.009530 on 2020-03-30 to $6.373356 on 2020-03-31.

### Distributions
These charts were constructed by creating a histogram of daily returns and then normalizing them to 100%. The normal distributions shown for comparison have the same parameters as the spread distributions and are similarly normalized to 100%. Summary statistics are also provided for reference, with the addition of skewness and kurtosis (normal = 0).

#### Entire Period
The distribution of daily returns for CBT:TY-FV does not exhibit a high degree of skewness and is slightly leptokurtic, with excess kurtosis of 5.0569. The normal distribution appears to fit reasonably well. The distribution of daily returns for ICE:G-B is much more peaked than that of CBT:TY-FV, with excess kurtosis of 13.0278.

#### Excluding Extraordinary Period
However, if we examine the distributions up to the end of 2019, before the onset of the pandemic, the returns for both spreads appear to be much more normally distributed and while both are still leptokurtic, kurtosis is lower for both. CBT:TY-FV has excess kurtosis of 0.8416 and ICE:G-B has excess kurtosis of 3.8181. The standard deviation of CBT:TY-FV is slightly decreased, from 0.0188 to 0.0166, whereas the standard deviation of ICE:G-B has decreased significantly, from 0.1469 to 0.0597.

Another potentially useful analysis is to examine the distributions before and after the potentially extraordinary period. If both periods have similar characteristics, that may make increase confidence in our expectations of the future. If they have different characteristics, that may decrease our confidence in our expectations of the future since there are various potential rationales for the difference, including (i) the more recent period is the new normal, (ii) the period prior to the extraordinary period is normal and the market has yet to return to normal or (iii) neither period is normal and the market is still transitioning to the new normal. Reviewing a longer history as well as more recent data may help increase confidence in our expectations.

When we look at the distribution of CBT:TY-FV for the period from 2020-06-01 to 2020-08-31 along side the distribution for the period from 2018-12-03 to 2019-12-31, we can see that distributions from the two different period do in fact have similar characteristics with respect to standard deviation, skewness and kurtosis.

However, when we look at the distributions for ICE:G-B for the tow different periods, we see that there is quite a bit more variance in the more recent period, but similar kurtosis.

### Q-Q Plots
To gain more insight into the ocurruence of outliers in the spread distributions, we can examine Q-Q plots. The plots are prepared by sorting normalized daily returns and plotting them against an equivalent number of divisions of the standard normal distribution from p = 0.001 to p = 0.999.

#### Entire Period
Here we can see that both distributions have some sinificant outliers, ICE:G-B to a greater extent, with some two observations greater than seven standard deviationsf from the mean. It is also interesting to note that, inside of the absolute of approximately 1.5 standard deviations, both distributions appear to be relatively normal.

#### Excluding Extraordinary Period
Here are the same plots including just the period up to the end of 2019. While there are still some outliers, there are far fewer of them and the distributions appear to be normal. The reduction in the quantity and magnitude of the outliers relative to the plot of the entire period appears to be consistent with the reduction in calculated kurtosis.

There are relatively few obserations in the more recent period, but for CBT:TY-FV, even though there appear to be a few outliers, they do not appear to be significant outliers, within three standard deviations of the mean. We can also see that the relatively high excess kurtosis of 6.5051 for ICE:G-B is likey due to the most extreme outlier at 4.0 standard deviations from the mean.

## Rolling Kurtosis
For a final evaluation of kurtosis, we examine kurtosis on a rolling basis over various window lengths. Over the shorter windows, kurtosis for both distributions is lower than it is over the longer windows, with an average of 0.43467 for CBT:TY-FV and 0.58609 for ICE:G-B for the 30 day window. This would seem to be because there are fewer opportunities for outliers to occurr over shorter periods of time and hence may not be a useful characteristic on its own. However, the fact that the higher kurtosis over longer periods appears to be as a result of a smaller number of outliers of a large magnitude, may indicate that it is possible to develop a good strategy if the outliers can be neutralized with stop loss limits or otherwise.

## Rolling Average Differences
The last analysis is of statistics of the differences between the spread daily returns and rolling averages over various window lengths. \[I am actually undertain how to interpret this analysis. Perhaps by looking at the relative differences in means and standard deviations can be helpful with determining propensity to revert to a mean.\]
