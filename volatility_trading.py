# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 14:21:25 2021

@author: samueljuan
"""

import pandas as pd
pd.options.display.float_format = "{:,.2f}".format
import pandas_datareader as wb
import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm


#%%

class Volatility_Trading():

    # constructor
    def __init__(self, start_date=None, end_date=None):

        # set start attribute
        if start_date is None:
            self.start_date = '1994-01-03'
        else:
            if isinstance(start_date, int): start_date = str(start_date)
            self.start_date = start_date

        # set end attribute
        if end_date is None:
            self.end_date = None
        else:
            if isinstance(end_date, int): end_date = str(end_date)
            self.end_date = end_date

        # set public attributes
        self.vix = None
        self.spy = None
        self.signal = None
        self.daily_return = None
        self.sharpe = None
        self.max_dd = None
        self.annual_return = None
        self.summary = None


#%%

    # define method to retrieve price information
    def load_data(self, forward_fill=True):
        # fetch data from yahoo finance
        self.vix = wb.DataReader('^VIX', 'yahoo', self.start_date, self.end_date)
        self.spy = wb.DataReader('SPY', 'yahoo', self.start_date, self.end_date)

        # handle missing values
        if forward_fill:
            self.vix.fillna(method='pad', inplace=True)
            self.spy.fillna(method='pad', inplace=True)


#%%

    # define method to compute signal from stochastic oscillator
    def get_stochastic_signal(self, start=None, end=None, lookback=15, k_period=5, d_period=5, long_only=True):

        # fetch data if not available
        if self.vix is None:
            self.load_data()

        # copy vix ohlc dataframe
        if isinstance(start, int): start = str(start)
        if isinstance(end, int): end = str(end)
        ohlc = self.vix.loc[start:end].copy()

        # compute rolling high and low during lookback period
        ohlc['Rolling_High'] = ohlc['High'].rolling(window=lookback).max()
        ohlc['Rolling_Low'] = ohlc['Low'].rolling(window=lookback).min()

        # compute %K
        ohlc['%K'] = ((ohlc['Close']-ohlc['Rolling_Low']) / (ohlc['Rolling_High']-ohlc['Rolling_Low'])) * 100

        # compute full K and full D lines
        ohlc['Full_K'] = ohlc['%K'].rolling(window=k_period).mean()
        ohlc['Full_D'] = ohlc['Full_K'].rolling(window=d_period).mean()

        # create signal column
        ohlc['Signal'] = ohlc['Full_K'] - ohlc['Full_D']

        # handle long-only or long-short scenario
        if long_only:
            ohlc['Signal'] = ohlc['Signal'].apply(lambda x: 1 if x > 0 else 0 if x <= 0 else np.nan)
        else:
            ohlc['Signal'] = ohlc['Signal'].apply(lambda x: 1 if x > 0 else -1 if x <= 0 else np.nan)

        # capture signal in attributes
        self.signal = ohlc[['Signal']]
        #self.all_signal = pd.concat([self.all_signal, ohlc[['Signal']]])
        #self.all_signal = self.all_signal[~self.all_signal.index.duplicated(keep='first')]


#%%

    # define method to backtest strategy
    def backtest(self, tc=1, lag=1, plot=True):

        # compute strategy daily return
        df = pd.concat([self.spy['Adj Close'].pct_change(),
                        self.signal.shift(lag+1)], axis=1, join='inner').dropna()
        df.columns = ['Market_Return','Signal']
        self.daily_return = df['Market_Return'] * df['Signal']

        # include transaction cost
        cost = tc / 10000
        turnover = df['Signal'].shift(lag).diff().fillna(0).abs()
        self.daily_return = self.daily_return - (turnover * cost)

        # compute sharpe ratio and maximum drawdown
        sharpe_ratio = self.get_sharpe_ratio(self.daily_return)
        max_drawdown = self.get_max_drawdown(self.daily_return)
        annual_return = self.get_annualized_return(self.daily_return)

        # visualize performance
        if plot:
            plt.figure(figsize=(13,8))
            plt.plot((df['Market_Return'] + 1).cumprod(),
                     label=f"Buy & Hold | Annual Return: {self.get_annualized_return(df['Market_Return']):.2f}% | Sharpe: {self.get_sharpe_ratio(df['Market_Return']):.2f} | Max.DD: {self.get_max_drawdown(df['Market_Return']):.2f}%")
            plt.plot((self.daily_return + 1).cumprod(),
                     label=f"Vol Trading | Annual Return: {annual_return:.2f}% | Sharpe: {sharpe_ratio:.2f} | Max.DD: {max_drawdown:.2f}%")
            plt.legend()
            plt.show()

        # assign to attributes
        self.sharpe = sharpe_ratio
        self.max_dd = max_drawdown
        self.annual_return = annual_return


#%%

    # define method to compute portfolio analytics
    def get_sharpe_ratio(self, daily_return):
        return daily_return.mean() / daily_return.std() * np.sqrt(250)

    def get_max_drawdown(self, daily_return):
        rolling_max = (daily_return + 1).cumprod().expanding().max()
        daily_drawdown = (daily_return + 1).cumprod() / rolling_max.values - 1.0
        max_drawdown = daily_drawdown.abs().max() * 100
        return max_drawdown

    def get_annualized_return(self, daily_return):
        annual_return = np.prod(daily_return.dropna().resample('A').sum() + 1) \
                        ** (1/len(daily_return.dropna().resample('A').sum())) - 1
        return annual_return * 100


#%%

    # define method to optimize backtest parameters
    def optimize_params(self, start=None, end=None, regularization=2):

        # set parameters range
        lookback = range(5, 41, 1)
        k_period = range(3, 21, 1)
        d_period = range(3, 21, 1)

        # iterate all possible combinations
        param_combi = list(itertools.product(lookback, k_period, d_period))

        # introduce constraint to the combinations
        param_combi = [combi for combi in param_combi if ((combi[0] > combi[1]) & (combi[0] > combi[2]))]

        # prepare summary dataframe
        summary = pd.DataFrame(index=range(len(param_combi)),
                               columns=['lookback','k_period','d_period','sharpe','max_drawdown'])

        # grid search optimum parameters
        values = range(len(param_combi))
        with tqdm(total=len(values), position=0, leave=True) as pbar:
            for i in values:
                pbar.set_description(f"Evaluating combination {param_combi[i]}")
                pbar.update()
                look, k, d = param_combi[i]
                self.get_stochastic_signal(start=start, end=end, lookback=look, k_period=k, d_period=d)
                self.backtest(plot=False)
                summary.iloc[i,:] = [look, k, d, self.sharpe, self.max_dd]
        summary['ratio'] = summary['sharpe'] / summary['max_drawdown']

        # capture optimum parameters
        look_param = summary.sort_values(by='ratio', ascending=False).iloc[regularization-1]['lookback']
        k_param = summary.sort_values(by='ratio', ascending=False).iloc[regularization-1]['k_period']
        d_param = summary.sort_values(by='ratio', ascending=False).iloc[regularization-1]['d_period']
        print(f"\nOptimized parameter is ({look_param}, {k_param}, {d_param})")

        self.summary = summary













