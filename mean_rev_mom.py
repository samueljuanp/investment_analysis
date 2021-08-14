# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:57:48 2021

@author: samueljuan
"""

# import libraries
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

# import data
data = pd.read_csv('spx_member_price.csv', index_col='Date', parse_dates=True)

# create equal-weighted portfolio as benchmark
equal = data.copy()
equal.loc[:] = np.nan
for date in equal.index:
    valid_stocks = data.loc[date].dropna()
    equal.loc[date, valid_stocks.index] = 1 / len(valid_stocks)
equal_pnl = ((data.pct_change() * equal.shift()).sum(axis=1)).cumsum()

# set parameters
bollinger = 20
short = 20
medium = 40
long = 100
hold_period = 5

# construct bollinger bands to spot mean-reversion
upper_band = data.rolling(bollinger).mean() + 2*data.rolling(bollinger).std()
lower_band = data.rolling(bollinger).mean() - 2*data.rolling(bollinger).std()

# compute moving averages to determine trend / momentum
short_ma = data.rolling(short).mean()
medium_ma = data.rolling(medium).mean()
long_ma = data.rolling(long).mean()

# check for one ticker
ticker = 'UNP'

# determine long conditions
uptrend_momentum = (short_ma[ticker] > medium_ma[ticker]) | (long_ma.diff()[ticker] > 0)
uptrend_dip = data[ticker] < lower_band[ticker]
long_trigger = uptrend_momentum & uptrend_dip

# prepare signal dataframe
signal = long_trigger.copy()
signal.loc[:] = np.nan

# initialize trading backtest
in_trade = False

# loop over everyday
for curr_date in signal.index:
    # not in position and no long trigger
    if (in_trade == False) and (long_trigger.loc[curr_date] == False):
        pass

    # not in position and got long trigger
    elif (in_trade == False) and (long_trigger.loc[curr_date] == True):
        enter_date = curr_date
        signal.loc[curr_date] = 1
        in_trade = True

    # already in position and got long trigger
    elif (in_trade == True) and (long_trigger.loc[curr_date] == True):
        signal.loc[curr_date] = 1

    # already in position and no long trigger
    elif (in_trade == True) and (long_trigger.loc[curr_date] == False):
        if np.busday_count(enter_date.date(), curr_date.date()) < hold_period:
            signal.loc[curr_date] = 1
        else:
            signal.loc[curr_date] = np.nan
            in_trade = False

plt.figure(figsize=(13,7))
plt.plot((data[ticker].pct_change() * signal.replace(np.nan, 0).shift(2)).cumsum())
plt.show()
