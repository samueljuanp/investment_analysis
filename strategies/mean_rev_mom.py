# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:57:48 2021

@author: samueljuan
"""

# import required libraries
import pandas as pd
pd.options.display.float_format = "{:,.2f}".format
import pandas_datareader as wb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


#ticker='SPY'
#start='2000-01-01'
#end=None
#boll=20
#short=10
#medium=50
#long=150
#capital=10000
#stop_loss=0.02
#take_profit=0.03
#time_loss=40
#tc=1
#long_short=True
#verbose=True


# define backtest engine function
def backtest_strategy (ticker='SPY', start='2000-01-01', end='2014-12-31',
                       boll=20, short=10, medium=50, long=100, capital=10000,
                       stop_loss=0.02, take_profit=0.04, time_loss=40, tc=1,
                       long_short=True, verbose=False, plot=True, return_pnl=False):

    # retrieve price data from Yahoo Finance
    data = wb.DataReader(ticker, 'yahoo', start, end)

    # construct bollinger bands to idenfity mean reversion
    upper_band = data['Close'].rolling(boll).mean() + 2*data['Close'].rolling(boll).std()
    lower_band = data['Close'].rolling(boll).mean() - 2*data['Close'].rolling(boll).std()

    # compute moving averages to determine trend or momentum
    short_trend = data['Close'].rolling(short).mean()
    medium_trend = data['Close'].rolling(medium).mean()
    long_trend = data['Close'].rolling(long).mean()

    # determine long conditions
    uptrend_mom = (short_trend > medium_trend) & (long_trend.diff() > 0)
    uptrend_dip = data['Close'] < lower_band
    long_trigger = uptrend_mom & uptrend_dip

    # determine short conditions
    if long_short:
        downtrend_mom = (short_trend < medium_trend) & (long_trend.diff() < 0)
        downtrend_spike = data['Close'] > upper_band
        short_trigger = downtrend_mom & downtrend_spike

    # prepare signal dataframe
    pnl = long_trigger.copy()
    pnl.loc[:] = 0
    pnl.iloc[0] = capital

    # initialize backtest parameters
    long_trade = False
    short_trade = False
    stop_loss_counter = 0
    take_profit_counter = 0
    time_loss_counter = 0

    # loop over daily ticks
    for i in range(1, len(pnl)):

        # set dates for ease of reference
        prev_date = pnl.index[i-1]
        curr_date = pnl.index[i]


        # do not enter long and short at the same time
        if short_trade is False:

            # not trading and no long trigger
            if (long_trade == False) and (long_trigger.loc[prev_date] == False):
                pnl.loc[curr_date] = pnl.loc[prev_date]


            # not trading and got long trigger
            elif (long_trade == False) and (long_trigger.loc[prev_date] == True):
                if data.loc[curr_date,'Open'] > data.loc[prev_date,'Close']:
                    # update status
                    long_date = curr_date
                    long_enter = data.loc[curr_date,'Open']
                    long_sl = data.loc[prev_date,'Low']
                    distance = long_enter - long_sl
                    long_tp = long_enter + (take_profit / stop_loss) * distance
                    capital = pnl.loc[prev_date]
                    amount = np.min((stop_loss*capital)/distance, capital/long_enter)
                    long_trade = True
                    # update capital
                    pnl.loc[curr_date] = capital + (amount * (data.loc[curr_date,'Close'] - long_enter)) - tc
                    if verbose:
                        print(f"[LONG] Buy {amount:.2f} shares at {long_enter:.2f} on {str(long_date.date())}")
                        print(f"[LONG] Stop loss: {long_sl:.2f} and take profit: {long_tp:.2f}")
                else:
                    pnl.loc[curr_date] = pnl.loc[prev_date]


            # trading and got long trigger
            #elif (long_trade == True) and (long_trigger.loc[prev_date] == True):
                # update capital
                #pnl.loc[curr_date] = capital + (amount * (data.loc[curr_date,'Close'] - long_enter))


            # trading and no long trigger
            elif (long_trade == True): #and (long_trigger.loc[prev_date] == False):

                # capture current price for ease for reference
                curr_long_low_price = data.loc[curr_date,'Low']
                curr_long_high_price = data.loc[curr_date,'High']

                # hit stop loss
                if curr_long_low_price <= long_sl:
                    # update status
                    long_trade = False
                    stop_loss_counter += 1
                    # update capital
                    pnl.loc[curr_date] = capital + (amount * (long_sl - long_enter)) - tc
                    capital = pnl.loc[curr_date]
                    if verbose:
                        print(f"[LONG] Sell {amount:.2f} shares at {long_sl:.2f} on {str(curr_date.date())} due to stop loss")
                        print(f"[LONG] Loss: {(amount * (long_enter - long_sl)):.2f}\n")

                # hit take profit
                elif curr_long_high_price >= long_tp:
                    # update status
                    long_trade = False
                    take_profit_counter += 1
                    # update capital
                    pnl.loc[curr_date] = capital + (amount * (long_tp - long_enter)) - tc
                    capital = pnl.loc[curr_date]
                    if verbose:
                        print(f"[LONG] Sell {amount:.2f} shares at {long_tp:.2f} on {str(curr_date.date())} due to take profit")
                        print(f"[LONG] Profit: {(amount * (long_tp - long_enter)):.2f}\n")

                # hit time loss
                elif np.busday_count(long_date.date(), curr_date.date()) > time_loss:
                    # update status
                    long_trade = False
                    time_loss_counter += 1
                    # update capital
                    pnl.loc[curr_date] = capital + (amount * (data.loc[curr_date,'Open'] - long_enter)) - tc
                    capital = pnl.loc[curr_date]
                    if verbose:
                        print(f"[LONG] Sell {amount:.2f} shares at {data.loc[curr_date,'Open']:.2f} on {str(curr_date.date())} due to time loss")
                        print(f"[LONG] P/L: {(amount * (data.loc[curr_date,'Open'] - long_enter)):.2f}\n")

                # continue trade
                else:
                    pnl.loc[curr_date] = capital + (amount * (data.loc[curr_date,'Close'] - long_enter))


        ############################# HANDLE LONG/SHORT CASE #############################


        if long_short and (long_trade is False):

            # not trading and no short trigger
            if (short_trade == False) and (short_trigger.loc[prev_date] == False):
                pnl.loc[curr_date] = pnl.loc[prev_date]


            # not trading and got short trigger
            elif (short_trade == False) and (short_trigger.loc[prev_date] == True):
                # update status
                short_date = curr_date
                short_enter = data.loc[curr_date,'Open']
                short_sl = data.loc[prev_date,'High']
                short_tp = short_enter - (take_profit / stop_loss) * (short_sl - short_enter)
                capital = pnl.loc[prev_date]
                amount = (stop_loss * capital) / (short_sl - short_enter)
                short_trade = True
                # update capital
                pnl.loc[curr_date] = capital + (amount * (short_enter - data.loc[curr_date,'Close'])) - tc
                if verbose: print(f"[SHORT] Sell {amount:.2f} shares at {short_enter:.2f} on {str(short_date.date())}")


            # trading and got short trigger
            elif (short_trade == True) and (short_trigger.loc[prev_date] == True):
                # update capital
                pnl.loc[curr_date] = capital + (amount * (short_enter - data.loc[curr_date,'Close']))


            # trading and no short trigger
            elif (short_trade == True) and (short_trigger.loc[prev_date] == False):

                # capture current price for ease for reference
                curr_short_low_price = data.loc[curr_date,'Low']
                curr_short_high_price = data.loc[curr_date,'High']

                # hit stop loss
                if curr_short_high_price >= short_sl:
                    # update status
                    short_trade=False
                    stop_loss_counter += 1
                    # update capital
                    pnl.loc[curr_date] = capital + (amount * (short_enter - short_sl)) - tc
                    capital = pnl.loc[curr_date]
                    if verbose: print(f"[SHORT] Buy {amount:.2f} shares at {short_sl:.2f} on {str(curr_date.date())} due to stop loss\n")

                # hit take profit
                elif curr_short_low_price <= short_tp:
                    # update status
                    short_trade = False
                    take_profit_counter += 1
                    # update capital
                    pnl.loc[curr_date] = capital + (amount * (short_enter - short_tp)) - tc
                    capital = pnl.loc[curr_date]
                    if verbose: print(f"[SHORT] Buy {amount:.2f} shares at {short_tp:.2f} on {str(curr_date.date())} due to take profit\n")

                # hit time loss
                elif np.busday_count(short_date.date(), curr_date.date()) > time_loss:
                    # update status
                    short_trade = False
                    time_loss_counter += 1
                    # update capital
                    pnl.loc[curr_date] = capital + (amount * (short_enter - data.loc[curr_date,'Open'])) - tc
                    capital = pnl.loc[curr_date]
                    if verbose: print(f"[SHORT] Buy {amount:.2f} shares at {data.loc[curr_date,'Open']:.2f} on {str(curr_date.date())} due to time loss\n")

                # continue trade
                else:
                    pnl.loc[curr_date] = capital + (amount * (short_enter - data.loc[curr_date,'Close']))


    # compute daily return
    daily_return = pnl.diff()

    # compute annualized sharpe ratio
    average_gain = daily_return[daily_return > 0].mean()
    average_loss = daily_return[daily_return < 0].abs().mean()
    edge = average_gain / average_loss

    # compute take profit / stop loss ratio
    tp_sl_ratio = take_profit_counter / stop_loss_counter

    # compute maximum drawdown
    high_watermark = pnl.cummax()
    daily_drawdown = pnl - high_watermark
    max_drawdown = (daily_drawdown / high_watermark).abs().max() * 100

    # visualize performance
    if plot:
        plt.figure(figsize=(16,9))
        plt.title(f"Strategy Performance for {ticker}", fontsize=13)
        plt.ylabel('Cumulative PnL', fontsize=12)
        plt.plot(pnl, label=f"Trading Edge = {edge:.2f} | TP/SL Ratio = {tp_sl_ratio:.2f} | Max.DD = {max_drawdown:.2f}%")
        plt.legend(loc='upper left', fontsize=12)
        plt.show()

    # return pnl
    if return_pnl:
        return pnl
