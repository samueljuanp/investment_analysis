# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:57:48 2021

@author: samueljuan
"""

# import required libraries
import pandas as pd
pd.options.display.float_format = "{:,.3f}".format
import pandas_datareader as wb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


# define backtest engine function
def backtest_strategy (ticker, start='2000-01-01', end='2014-12-31', boll=20, short=20, medium=40, long=100,
                       stop_loss=0.02, take_profit=0.04, time_loss=40, tc=2, lag=0, long_short=False, verbose=False):

    # retrieve price data from Yahoo Finance
    data = wb.DataReader(ticker, 'yahoo', start, end)

    # construct bollinger bands to idenfity mean reversion
    upper_band = data['Adj Close'].rolling(boll).mean() + 2*data['Adj Close'].rolling(boll).std()
    lower_band = data['Adj Close'].rolling(boll).mean() - 2*data['Adj Close'].rolling(boll).std()

    # compute moving averages to determine trend or momentum
    short_ma = data['Adj Close'].rolling(short).mean()
    medium_ma = data['Adj Close'].rolling(medium).mean()
    long_ma = data['Adj Close'].rolling(long).mean()

    # determine long conditions
    uptrend_mom = (short_ma > medium_ma) | (long_ma.diff() > 0)
    uptrend_dip = data['Adj Close'] < lower_band
    long_trigger = uptrend_mom & uptrend_dip

    # determine short conditions
    if long_short:
        downtrend_mom = (short_ma < medium_ma) | (long_ma.diff() < 0)
        downtrend_spike = data['Adj Close'] > upper_band
        short_trigger = downtrend_mom & downtrend_spike

    # prepare signal dataframe
    signal = long_trigger.copy()
    signal.loc[:] = 0

    # initialize simulation
    long_trade = False
    short_trade = False
    stop_loss_counter = 0
    take_profit_counter = 0
    time_loss_counter = 0

    # loop over daily ticks
    for curr_date in signal.index:

        # do not enter long and short at the same time
        if short_trade is False:

            # not trading and no long trigger
            if (long_trade == False) and (long_trigger.loc[curr_date] == False):
                pass

            # not trading and got long trigger
            elif (long_trade == False) and (long_trigger.loc[curr_date] == True):
                long_date = curr_date
                long_spot = data.loc[curr_date,'Adj Close']
                signal.loc[curr_date] = 1
                long_trade = True
                if verbose: print(f"Long entry on {str(long_date.date())} at {long_spot:.2f}")

            # trading and got long trigger
            elif (long_trade == True) and (long_trigger.loc[curr_date] == True):
                # adjust trailing stop loss
                if data.loc[curr_date,'Adj Close'] > long_spot:
                    long_spot = data.loc[curr_date,'Adj Close']
                    if verbose: print(f"Adjust long trailing stop loss to {data.loc[curr_date,'Adj Close']:.2f}")
                signal.loc[curr_date] = 1

            # trading and no long trigger
            elif (long_trade == True) and (long_trigger.loc[curr_date] == False):

                # capture current price for ease for reference
                curr_long_price = data.loc[curr_date,'Adj Close']

                # hit stop loss
                if curr_long_price < (long_spot * (1-stop_loss)):
                    signal.loc[curr_date] = 0
                    long_trade = False
                    stop_loss_counter += 1
                    if verbose: print(f"Long exit on {str(curr_date.date())} at {curr_long_price:.2f} due to stop loss\n")

                # hit take profit
                elif curr_long_price > (long_spot * (1+take_profit)):
                    signal.loc[curr_date] = 0
                    long_trade = False
                    take_profit_counter += 1
                    if verbose: print(f"Long exit on {str(curr_date.date())} at {curr_long_price:.2f} due to take profit\n")

                # hit time loss
                elif np.busday_count(long_date.date(), curr_date.date()) > time_loss:
                    signal.loc[curr_date] = 0
                    long_trade = False
                    time_loss_counter += 1
                    if verbose: print(f"Long exit on {str(curr_date.date())} at {curr_long_price:.2f} due to time loss\n")

                else:
                    signal.loc[curr_date] = 1 # continue trade

        ########################## HANDLE LONG/SHORT CASE ##########################

        if long_short and (long_trade is False):

            # not trading and no short trigger
            if (short_trade == False) and (short_trigger.loc[curr_date] == False):
                pass

            # not trading and got short trigger
            elif (short_trade == False) and (short_trigger.loc[curr_date] == True):
                short_date = curr_date
                short_spot = data.loc[curr_date,'Adj Close']
                signal.loc[curr_date] = -1
                short_trade = True
                if verbose: print(f"Short entry on {str(short_date.date())} at {short_spot:.2f}")

            # trading and got short trigger
            elif (short_trade == True) and (short_trigger.loc[curr_date] == True):
                # adjust trailing stop loss
                if data.loc[curr_date,'Adj Close'] < short_spot:
                    short_spot = data.loc[curr_date,'Adj Close']
                    if verbose: print(f"Adjust short trailing stop loss to {data.loc[curr_date,'Adj Close']:.2f}")
                signal.loc[curr_date] = -1

            # trading and no short trigger
            elif (short_trade == True) and (short_trigger.loc[curr_date] == False):

                # capture current price for ease for reference
                curr_short_price = data.loc[curr_date,'Adj Close']

                # hit stop loss
                if curr_short_price > (short_spot * (1+stop_loss)):
                    signal.loc[curr_date] = 0
                    short_trade = False
                    stop_loss_counter += 1
                    if verbose: print(f"Short exit on {str(curr_date.date())} at {curr_short_price:.2f} due to stop loss\n")

                # hit take profit
                elif curr_short_price < (short_spot * (1-take_profit)):
                    signal.loc[curr_date] = 0
                    short_trade = False
                    take_profit_counter += 1
                    if verbose: print(f"Short exit on {str(curr_date.date())} at {curr_short_price:.2f} due to take profit\n")

                # hit time loss
                elif np.busday_count(short_date.date(), curr_date.date()) > time_loss:
                    signal.loc[curr_date] = 0
                    short_trade = False
                    time_loss_counter += 1
                    if verbose: print(f"Short exit on {str(curr_date.date())} at {curr_short_price:.2f} due to time loss\n")

                else:
                    signal.loc[curr_date] = -1 # continue trade


    # compute strategy return with specified execution lag
    daily_return = data['Adj Close'].pct_change() * signal.shift(1+lag)

    # include transaction cost
    cost = tc / 10000
    turnover = signal.shift(lag).diff().fillna(0).abs()
    daily_return = daily_return - (turnover * cost)

    # compute annualized sharpe ratio
    sharpe = daily_return.mean() / daily_return.std() * np.sqrt(252)

    # compute maximum drawdown
    cumulative_pnl = daily_return.cumsum(skipna=True)
    high_watermark = cumulative_pnl.cummax()
    daily_drawdown = cumulative_pnl - high_watermark
    max_drawdown = daily_drawdown.abs().max() * 100

    # compute annualized return
    annual_return = np.prod(daily_return.resample('A').sum() + 1) \
                    ** (1/len(daily_return.resample('A').sum())) - 1
    annual_return = annual_return * 100

    # visualize performance
    plt.figure(figsize=(16,9))
    plt.title(f"Strategy Performance of {ticker}", fontsize=13)
    plt.ylabel('Cumulative PnL (%)', fontsize=12)
    plt.plot(cumulative_pnl * 100,
             label=f"Annualized Return = {annual_return:.2f}% | Sharpe = {sharpe:.2f} | Max.DD = {max_drawdown:.2f}%")
    plt.legend(loc='upper left', fontsize=12)
    plt.show()

    return print(f"\nStop Loss: {stop_loss_counter}\nTake Profit: {take_profit_counter}\nTime Loss: {time_loss_counter}")
