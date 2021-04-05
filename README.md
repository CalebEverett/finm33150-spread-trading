# FINM33150 - Futures Spreads

## Contents
1. [Installation](#installation)
2. [Fetching Data from Quandl](#fetching-data-from-quandl)
3. [Data Preparation](#data-preparation)
4. [Analysis](#analysis)
    1. [Spread Charts](#spread-charts)


## Installation

Create a virtual environment and install dependencies with

    pipenv install

#TODO: Add instructions on starting jupyter notebook


## Fetching Data from Quandl

Obtain data from Quandl to calculate second month futures prices.

* Super cool feature one.
* Super cool feature two.


## Data Preparation

* Super cool feature one.
* Super cool feature two.


## Analysis

The two futures spreads analyzed herein are:
* [10-Year T-Note Futures](https://www.cmegroup.com/trading/interest-rates/us-treasury/10-year-us-treasury-note_contractSpecs_futures.html) over the [Five-Year T-Note Futures](https://www.cmegroup.com/trading/interest-rates/us-treasury/5-year-us-treasury-note_contractSpecs_futures.html) (CBT:TY-FV throughout)
* [Brent Crude Futures](https://www.theice.com/products/219/Brent-Crude-Futures) over [Low Sulphur Gasoil Futures](https://www.theice.com/products/34361119/Low-Sulphur-Gasoil-Futures) (ICE:G-B throughout)

The prices are calculated using second month contracts (earliest expiration greater than 30 days) using quarterly contract expirations for the period from 2018-12-03 to 2020-08-31 and includes discussion and analysis of the following exhibits:
* Charts of underlying securities and spreads over the period
* Summary statistics
* Charts of distributions of daily returns relative to normal distributions
* Q-Q plots
* Charts of rolling kurtosis
* Statistics of differences to rolling averages of daily returns

### Spread Charts
A significant feature of both sets of charts is the relative large change in performance that occurs beginning in the first quarter of 2020, presumably as a result of the pandemic. The effect is seen more clearly in the ICE:G-B spread than it is in the CBT:TY-FV spread, but is evident nonetheless, and clearly evident in all of the underlying securities. If the objective were to form a basis for a reversion to the mean trading strategy, with the expectation that there will not be similar shocks to the system during our investment horizon, it may be appropriate to exclude this period from our analysis.

### Summary Statistics
Across the entire period, CBT:TY-FV had a mean daily return of 0.001708 with a standard deviation of 0.018756. ICE:G-B was much more volatile, with a standard deviation of 0.146384 and a mean of 0.007518. That additional volatility can also be seen in the relative minimum and maximum daily returns. Where as CBT:TY-FV had a min and max of -0.096922 and 0.08785, respectively, ICE:G-B had a min and max of -0.609314 and 1.053632, respectively. The range of ICE:G-B is 1.662946, which is 9.0 times greater than the range of 0.184777 for CBT:TY-FV. The range between the 25th percentile and the 75th percentile for ICE:G-B is still 4.4 times that of CBT:TY-FV. One last point to note is that ICE:G-B had a positive mean, even though it declined significantly over the period. This indicates there were a small number of observations with relatively large negative returns. For example, ICE:G-B declined 51.0%, from $13.009530 on 2020-03-30 to $6.373356 on 2020-03-31.

### Distributions
These charts were constructed by creating a histogram of daily returns and then normalizing them to 100%. The normal distributions show for comparison have the same parameters as the spread distributions and are similarly normalized to 100%. Summary statistics are also provided for reference, with the addition of skewness and kurtosis (Fisher's definition).

#### Entire Period
The distribution of daily returns for CBT:TY-FV does not exhibit a high degree of skewness and is slightly leptokurtic, with excess kurtosis of 1.7299. The normal distribution appears to fit it reasonably well. The distribution of daily returns for ICE:G-B is much more peaked than that of CBT:TY-FV, with excess kurtosis of 13.4635.

#### Excluding Extraordinary Period
However, if we examine the distributions up to the end of 2019, before the onset of the pandemic, the nature of the distributions is quite different.

The returns for both spreads appear to be much more normally distributed and both are platykurtic with respect to kurtosis. CBT:TY-FV has negative excess kurtosis of 2.0272 and ICE:G-B has negative excess kurtosis of 0.5653. The standard deviation of CBT:TY-FV has decreased slightly form 0.0188 to 0.0166, whereas the standard deviation of ICE:G-B has decreased significantly from 0.1469 to 0.0590.

Another potentially useful analysis is to examine the the distributions before and after the potentially extraordinary period. If both periods have similar characteristics that may make result in higher confidence in our expectations of the future. If they have different characteristics, that may result in lower confidence since there are could be various potential rationales for the difference, including (i) the more recent period is the new normal, (ii) the period prior to the extraordinary period is normal period and the market has yet to return to it or (iii) neither period is normal and the market is still transitioning to the new normal. Reviewing a longer history as well as more recent data would help increase confidence in our expectations.

When we look at the distribution of CBT:TY-FV for the period from 2020-06-01 to 2020-08-31 along side the distribution for the period from 2018-12-03 to 2019-12-31, we can see that they do in fact have similar characteristics with respect to standard deviation, skewness and kurtosis. However, when we look the distributions for ICE:G-B for the same periods, we see that there is quite a bit more variance in the more recent period as well as greater kurtosis.

### Q-Q Plots



### Key Questions
1. I would say the overriding objective of the analysis is to determine whether either of these spreads is an attractive candidate upon which to develop a reversion to the mean trading strategy.
2. With that in mind, the first question is what is the appropriate period of analysis. The historical data includes a period of extreme volatility, presumably as a result of the global pandemic. If we aren't expecting similar shocks in our investment horizon it may make sense to exclude this period from our analysis.
3. The next question then becomes, what is the propensity f
2. Within that period, what is the nature of the distributions of daily returns for each spread?
    1.
