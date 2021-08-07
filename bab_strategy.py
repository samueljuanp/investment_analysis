2# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 09:53:26 2021

@author: samueljuan
"""

# set github directory
import os
os.chdir(r'C:\Users\User\Documents\GitHub\investment_analysis')

# import libraries
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import pandas_datareader as wb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from statsmodels.regression.rolling import RollingOLS
#import investment_analysis as ia


#%%

# retrieve data from csv
data = pd.read_csv('spx_member_price.csv', index_col=0, parse_dates=True)
#tickers = data.columns.tolist()

# refresh data and overwrite csv
#new_data = wb.DataReader(tickers, 'yahoo', start='2000-01-01')['Adj Close']
#new_data.to_csv('spx_member_price.csv')

# read data from csv
#data = pd.read_csv('spx_member_price.csv', index_col=0, parse_dates=True)

# pull benchmark price
spx = wb.DataReader('^GSPC', 'yahoo', '2000-01-01', data.index[-1])[['Adj Close']]
spx.columns = ['SPX']

#%%

# define function to compute rolling beta
def get_rolling_beta(stock, window=500, cov_type='HCCM'):

    """
    Parameters
    ----------
    stock : str
        single stock name to query from data table.
    window : int
        rolling trading days to compute beta. The default is 500.
    cov_type : str
        choose 'nonrobust' to exclude heteroskedasticity impact.
        The default is 'HCCM'.

    Returns
    -------
    time-series of stock beta

    """
    # compute daily return
    stock_return = data[stock].pct_change()
    market_return = spx.pct_change()

    # fit into rolling regression framework
    model = RollingOLS(endog=stock_return, exog=market_return,
                       window=window, min_nobs=window, missing='drop')
    result = model.fit(cov_type=cov_type, params_only=True)

    # query beta and rename column
    rolling_beta = result.params
    rolling_beta.columns = [stock]

    return rolling_beta

#%%

# populate rolling 2Y beta for all stocks
#beta = []

#for stock in data.columns:
    #beta.append(get_rolling_beta(stock, window=500, cov_type='HCCM'))
#beta = pd.concat(beta, axis=1)

# save latest data to csv
#beta.to_csv('rolling_2y_beta.csv')

# read beta information from csv
beta = pd.read_csv('rolling_2y_beta.csv', index_col=0, parse_dates=True)

#%%

# generate allocation dataframe
allocation = beta.copy()
allocation.loc[:] = np.nan

# set quantile parameter
q = 0.05
low_high_ratio = 0.80

# loop over each day
for date in beta.index:

    # sort valid beta in ascending order
    beta_sorted = beta.loc[date].dropna().sort_values()

    # skip current loop if there is no valid data
    if beta_sorted.empty:
        continue
    else:
        # scale beta by its mean to avoid negative value
        beta_scaled = beta_sorted + beta_sorted.mean()

        # classify low and high beta
        low_beta = beta_scaled[beta_scaled <= beta_scaled.quantile(q)]
        high_beta = beta_scaled[beta_scaled >= beta_scaled.quantile(1-q)]

        # compute allocation based on signal strength
        long_low_beta = (1/low_beta) / (1/low_beta).sum()
        long_high_beta = high_beta / high_beta.sum()

        # assign final allocation
        allocation.loc[date, long_low_beta.index] = long_low_beta.values * low_high_ratio
        allocation.loc[date, long_high_beta.index] = long_high_beta.values * (1-low_high_ratio)

# sanity check for portfolio beta
#port_beta = (allocation * beta).dropna(how='all').sum(axis=1)
#port_beta.plot(title='Total Portfolio Beta', figsize=(12,7))
#plt.show()

# sanity check for portfolio weights
#port_weight = allocation.dropna(how='all').sum(axis=1)
#port_weight.plot(title='Total Portfolio Allocation', figsize=(12,7))
#plt.show()

#%%

# define backtest period
start = '2005'
end = None

# compute daily strategy and market return
strat_return = (data.pct_change() * allocation.shift(2)).dropna(how='all').sum(axis=1)
strat_return = strat_return.loc[start:end]
market_return = spx.pct_change().loc[strat_return.index[0]:strat_return.index[-1]]['SPX']

# include transaction cost in basis points
tc = 0
cost = tc / 10000
turnover = allocation.loc[start:end].shift().diff().fillna(0).abs()
strat_return = strat_return - (turnover * cost).sum(axis=1)

# define function to compute Sharpe ratio
def sharpe_ratio(daily_return):
    return daily_return.mean() / daily_return.std() * np.sqrt(252)

# define function to compute maximum drawdown
def max_drawdown(daily_return):
    rolling_max = (daily_return + 1).cumprod().expanding().max()
    monthly_drawdown = (daily_return + 1).cumprod() / rolling_max.values - 1.0
    max_drawdown = monthly_drawdown.abs().max() * 100
    return max_drawdown

# define function to compute annualized return
def annualized_return(daily_return):
    annual_return = np.prod(daily_return.resample('A').sum() + 1) \
                    ** (1/len(daily_return.resample('A').sum())) - 1
    annual_return = annual_return * 100
    return annual_return

# visualize performance
plt.figure(figsize=(15,8))
plt.title('Portfolio Performance Comparison', fontsize=13)
plt.ylabel('Net Asset Value', fontsize=13)

plt.plot((strat_return + 1).cumprod(),
         label=f"Long-Low-High-Beta | Annualized = {annualized_return(strat_return):.2f}% | Sharpe = {sharpe_ratio(strat_return):.2f} | Max.DD = {max_drawdown(strat_return):.2f}%")

plt.plot((market_return + 1).cumprod(),
         label=f"S&P500-Benchmark | Annualized = {annualized_return(market_return):.2f}% | Sharpe = {sharpe_ratio(market_return):.2f} | Max.DD = {max_drawdown(market_return):.2f}%")

plt.legend(fontsize=12)
plt.show()


















