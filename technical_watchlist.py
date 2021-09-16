# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 21:03:36 2021

@author: samueljuan
"""

import pandas as pd
pd.options.display.float_format = "{:,.2f}".format
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100)
from pandas.tseries.offsets import BDay
from yahoofinancials import YahooFinancials as yf

import pandas_datareader as wb
import datetime as dt
import numpy as np
import yagmail
import json
import os


#%%

class Technical_Watchlist():

    # set default watchlist
    watchlist = {
        'indices': ['ARKK','ASHR','DIA','EEM','GXC','KWEB','QQQ','RSP','SPY','^VIX'],

        'equities': ['AAPL','ACN','ADBE','AMZN','ASML','BA','BABA','BAC','CLX','CME',
                     'COST','CRM','DPZ','EL','FB','GOOGL','ICE','ILMN','INTU','JNJ','JPM',
                     'LMT','MA','MCD','MELI','MMM','MSFT','NKE','NVDA','PEP','PG',
                     '0700.HK','TSLA','TSM','UNH','UNP','V','VEEV','YUMC','ZM'],

        'reits': ['A17U.SI','AJBU.SI','AU8U.SI','BUOU.SI','BWCU.SI','J69U.SI',
                  'C2PU.SI','C38U.SI','K71U.SI','M44U.SI','N2IU.SI','ME8U.SI'],

        'bonds': ['SHY','IEI','IEF','TLT','BND','BNDX','LQD','EMB'],

        'commods': ['GLD','SLV','PDBC','DBC','COMT','CCRV','BDRY','XLE','XOP'],

        'cryptos': ['BTC-USD','ETH-USD','ADA-USD','BNB-USD','LTC-USD','XRP-USD','BCH-USD']
        }

    fields = ['Open','High','Low','Close']


    # constructor
    def __init__(self, tickers=None, start_date=None, end_date=None, verbose=True):

        # set tickers attribute
        if tickers is None:
            self.tickers = [ticker for key, value in zip(self.watchlist.keys(), self.watchlist.values()) \
                            for ticker in self.watchlist[key] if key not in ['cryptos']] # exclude cryptos first
        else:
            if isinstance(tickers, str): tickers = [tickers]
            self.tickers = tickers

        # set date attributes
        if end_date is None: self.end_date = dt.datetime.today()
        else: self.end_date = str(end_date)
        if start_date is None: self.start_date = self.end_date - dt.timedelta(730) # 2Y worth of data
        else: self.start_date = str(start_date)

        # set other attributes
        self.verbose = verbose
        self.daily = {}
        self.weekly = {}
        self.summary = {}
        self.fundamentals = {}
        self.html_tables = {}
        self.html_string = None
        self.file_path = None


#%%

    # run procedure
    def main(self, send=True):

        self.load_data()
        self.get_stochastic_signal()
        self.get_medium_trend()
        self.format_display()
        self.create_html_report()
        self.send_email(send=send)


#%%

    # load data from yahoo finance
    def load_data(self):

        # fetch data
        if self.verbose:
            print(f"\nRetrieving price data of {len([ticker for key in self.watchlist.keys() for ticker in self.watchlist[key]])} tickers..")
        daily = wb.DataReader(self.tickers, 'yahoo', self.start_date, self.end_date)[self.fields]
        crypto = wb.DataReader(self.watchlist['cryptos'], 'yahoo', self.start_date, self.end_date)[self.fields]
        daily = pd.merge(left=crypto, right=daily, how='outer', left_index=True, right_index=True)

        # include cryptos in tickers
        self.tickers += self.watchlist['cryptos']

        # drop unavailable tickers
        missing = []
        for ticker in daily.columns.levels[1]:
            # dataframe is empty
            if daily.xs(ticker, axis=1, level=1).dropna().empty is True:
                missing.append(ticker)
                self.tickers.remove(ticker)
            if missing != []:
                if self.verbose: print(f"Drop unavailable ticker(s): {', '.join(missing)}")
                daily.drop(missing, axis=1, level=1, inplace=True)

        # assign daily data to attribute for each ticker
        for ticker in self.tickers:
            self.daily[ticker] = daily.xs(ticker, axis=1, level=1).dropna()

            # resample data from daily to weekly
            weekly_open = self.daily[ticker]['Open'].resample('W-FRI').first()
            weekly_high = self.daily[ticker]['High'].resample('W-FRI').max()
            weekly_low = self.daily[ticker]['Low'].resample('W-FRI').min()
            weekly_close = self.daily[ticker]['Close'].resample('W-FRI').last()
            weekly = pd.concat([weekly_open, weekly_high, weekly_low, weekly_close],
                               axis=1, keys=['Open','High','Low','Close'])

            # assign to weekly attribute
            self.weekly[ticker] = weekly

            # change index data type to string
            self.daily[ticker].index = self.daily[ticker].index.astype(str)
            self.weekly[ticker].index = self.weekly[ticker].index.astype(str)

            # change the latest weekly date to avoid forward bias
            self.weekly[ticker].index = self.weekly[ticker].index[:-1].tolist() + [self.daily[ticker].index[-1]]


#%%

    # compute stochastic signal
    def get_stochastic_signal(self, ma='smoothed', lookback=9, k_period=3, d_period=3, oversold=25, overbought=85):

        # name parameters
        self.pct_k = ' %K(' + str(lookback) + ',' + str(k_period) + ')'
        self.pct_d = ' %D(' + str(d_period) + ')'

        # print status
        if self.verbose: print(f"Computing stochastic oscillator for {len(self.tickers)} tickers..")

        for ticker in self.tickers:

            # prepare dictionary to capture result
            if ticker not in self.summary.keys(): self.summary[ticker] = {}

            # loop over daily and weekly dataframe
            for i in range(2):

                # assign ohlc and timeframe
                if i == 0:
                    ohlc = self.daily[ticker]
                    timeframe = 'Daily'

                    # assign last price and 6M momentum
                    self.summary[ticker]['Last Price'] = ohlc.iloc[-1]['Close']
                    self.summary[ticker]['6M Change'] = ohlc['Close'].pct_change(120).iloc[-1]

                else:
                    ohlc = self.weekly[ticker]
                    timeframe = 'Weekly'

                # compute rolling high and low during lookback period
                ohlc['Rolling_High'] = ohlc['High'].rolling(window=lookback).max()
                ohlc['Rolling_Low'] = ohlc['Low'].rolling(window=lookback).min()

                # compute %K
                ohlc['%K'] = ((ohlc['Close']-ohlc['Rolling_Low']) / (ohlc['Rolling_High']-ohlc['Rolling_Low'])) * 100

                # compute full K and full D lines
                if ma == 'simple':
                    ohlc['Full_K'] = ohlc['%K'].rolling(window=k_period).mean()
                    ohlc['Full_D'] = ohlc['Full_K'].rolling(window=d_period).mean()

                elif ma == 'exponential':
                    ohlc['Full_K'] = ohlc['%K'].ewm(span=k_period).mean()
                    ohlc['Full_D'] = ohlc['Full_K'].ewm(span=d_period).mean()

                elif ma == 'smoothed':
                    smoothed_k = 2 * k_period - 1
                    smoothed_d = 2 * d_period - 1
                    ohlc['Full_K'] = ohlc['%K'].ewm(span=smoothed_k).mean()
                    ohlc['Full_D'] = ohlc['Full_K'].ewm(span=smoothed_d).mean()

                else:
                    print(f"Supported moving averages: simple, exponential, smoothed. {ma} was passed..")

                # compute crossover and change in signal
                ohlc['Stoch_Cross'] = ohlc['Full_K'] - ohlc['Full_D']
                ohlc['Stoch_Change'] = np.sign(ohlc['Stoch_Cross']).diff()
                ohlc['Stoch_Remark'] = ''

                # compute turning point
                for j in range(1, len(ohlc)):

                    # reference
                    date = ohlc.index[j]
                    change_now = ohlc.iloc[j]['Stoch_Change']
                    prev_k = ohlc.iloc[j-1]['Full_K']
                    prev_d = ohlc.iloc[j-1]['Full_D']

                    # bullish signal
                    if change_now > 0:
                        if (prev_k <= oversold) and (prev_d <= oversold):
                            ohlc.loc[date,'Stoch_Remark'] = 'Very Bullish'
                        else:
                            ohlc.loc[date,'Stoch_Remark'] = 'Bullish'

                    # bearish signal
                    elif change_now < 0:
                        if (prev_k >= overbought) and (prev_d >= overbought):
                            ohlc.loc[date,'Stoch_Remark'] = 'Very Bearish'
                        else:
                            ohlc.loc[date,'Stoch_Remark'] = 'Bearish'

                    # no confirmed signal
                    else: pass

                # assign to summary attribute
                self.summary[ticker][str(timeframe) + self.pct_k] = ohlc.iloc[-1]['Full_K'] # live
                self.summary[ticker][str(timeframe) + self.pct_d] = ohlc.iloc[-1]['Full_D'] # live
                self.summary[ticker][str(timeframe) + ' Remark'] = ohlc[ohlc['Stoch_Remark'] != ''].iloc[-1]['Stoch_Remark']
                self.summary[ticker][str(timeframe) + ' Time'] = ohlc[ohlc['Stoch_Remark'] != ''].index[-1]

                # assign weekly trend
                if i == 1:
                    if ohlc.iloc[-1]['Full_K'] > ohlc.iloc[-1]['Full_D']:
                        self.summary[ticker][str(timeframe) + ' Trend'] = 'Uptrend'
                    else:
                        self.summary[ticker][str(timeframe) + ' Trend'] = 'Downtrend'


#%%

    # compute moving averages
    def get_medium_trend(self):

        if self.verbose: print(f"Computing moving averages for {len(self.tickers)} tickers..")

        # loop over all tickers
        for ticker in self.tickers:
            # compute daily ema and sma
            self.daily[ticker]['EMA(20)'] = self.daily[ticker]['Close'].ewm(span=20).mean()
            self.daily[ticker]['EMA(40)'] = self.daily[ticker]['Close'].ewm(span=40).mean()
            self.daily[ticker]['SMA(50)'] = self.daily[ticker]['Close'].rolling(window=50).mean()
            self.daily[ticker]['SMA(100)'] = self.daily[ticker]['Close'].rolling(window=100).mean()
            self.daily[ticker]['SMA(150)'] = self.daily[ticker]['Close'].rolling(window=150).mean()
            self.daily[ticker]['SMA(200)'] = self.daily[ticker]['Close'].rolling(window=200).mean()

            # compute ma cross
            self.daily[ticker]['EMA_Cross'] = self.daily[ticker]['EMA(20)'] / self.daily[ticker]['EMA(40)'] - 1
            self.daily[ticker]['SMA_Cross'] = self.daily[ticker]['SMA(50)'] / self.daily[ticker]['SMA(150)'] - 1

            # assign latest value
            self.summary[ticker]['EMA(20) / (40)'] = self.daily[ticker]['EMA_Cross'].iloc[-1]
            self.summary[ticker]['SMA(50) / (150)'] = self.daily[ticker]['SMA_Cross'].iloc[-1]
            self.summary[ticker]['SMA(200)'] = self.daily[ticker]['SMA(200)'].iloc[-1]

            # assign medium trend remarks (CONDITIONS TO BE IMPROVED)
            if self.daily[ticker]['SMA_Cross'].iloc[-1] > 0:
                self.summary[ticker]['Medium Trend'] = 'Uptrend'
            else:
                self.summary[ticker]['Medium Trend'] = 'Downtrend'


#%%

    ############# UNDER DEVELOPMENT ##############

    # obtain equity fundamentals data
    def get_fundamentals(self, update=False):

        # loop over equity tickers
        for ticker in self.watchlist['equities']:

            # prepare dictionary
            self.fundamentals[ticker] = {}

            # retrieve data from yahoo financials (will take long time)
            bsheet = yf(ticker).get_financial_stmts('quarter','balance')
            income = yf(ticker).get_financial_stmts('quarter','income')
            cash = yf(ticker).get_financial_stmts('quarter','cash')

            # populate balance sheet data
            for item in bsheet['balanceSheetHistoryQuarterly'][ticker]:
                for date, bsheet_subitem in item.items():
                    self.fundamentals[ticker][date] = bsheet_subitem

            # populate income statement data
            for item in income['incomeStatementHistoryQuarterly'][ticker]:
                for date, income_subitem in item.items():
                    self.fundamentals[ticker][date].update(income_subitem)

            # populate cashflow data
            for item in cash['cashflowStatementHistoryQuarterly'][ticker]:
                for date, cash_subitem in item.items():
                    self.fundamentals[ticker][date].update(cash_subitem)

        # store data externally
        with open('fundamental_database.json','w') as f:
            json.dump(self.fundamentals, f)
        f.close()

        # retrieve stored data
        g = open('fundamental_database.json')
        self.fundamentals = json.load(g)
        g.close()

        # populate fcf/a and gp/a ratio
        df = pd.DataFrame(index=tech.watchlist['equities'], columns=['FCF/A','GP/A'])
        for ticker in self.watchlist['equities']:
            latest_date = list(self.fundamentals[ticker].keys())[0]
            asset = self.fundamentals[ticker][latest_date].get('totalAssets')
            gp = self.fundamentals[ticker][latest_date].get('grossProfit')
            cash = self.fundamentals[ticker][latest_date].get('totalCashFromOperatingActivities')
            capex = self.fundamentals[ticker][latest_date].get('capitalExpenditures')
            try:
                fcf = cash + capex
                df.loc[ticker,'FCF/A'] = fcf / asset
            except:
                df.loc[ticker,'FCF/A'] = cash / asset
            df.loc[ticker,'GP/A'] = gp / asset

    ############# UNDER DEVELOPMENT ##############


#%%

    # format display of summary dataframe
    def format_display(self):

        # transform dictionary to dataframe
        summary = pd.DataFrame(self.summary).T.copy()

        # rearrange columns
        summary = summary[['Last Price','6M Change', \
                           'Weekly'+self.pct_k,'Weekly'+self.pct_d,'Weekly Remark','Weekly Time', \
                           'EMA(20) / (40)','SMA(50) / (150)','SMA(200)','Medium Trend', \
                           'Daily'+self.pct_k,'Daily'+self.pct_d,'Daily Remark','Daily Time', \
                           ]]

        # format display
        dec_2dp = lambda x: str('{:.2f}'.format(x))
        pct_2dp = lambda x: str('{:.2f}%'.format(x*100))

        no_format = ['6M Change','Daily Remark','Daily Time','EMA(20) / (40)','SMA(50) / (150)',
                     'Medium Trend','Weekly Remark','Weekly Time']
        summary.loc[:, [col for col in summary.columns if col not in no_format]] = \
        summary.loc[:, [col for col in summary.columns if col not in no_format]].applymap(dec_2dp).values
        summary[['6M Change','EMA(20) / (40)','SMA(50) / (150)']] = \
        summary[['6M Change','EMA(20) / (40)','SMA(50) / (150)']].applymap(pct_2dp).values

        # split into different tabs
        self.summary = {}
        for key in self.watchlist.keys():
            self.summary[key] = summary[summary.index.isin(self.watchlist[key])].copy()

        # sort dataframe
        for tab in self.summary.keys():
            self.summary[tab].sort_values(by=['Weekly Time','Weekly Remark','Daily Time','Daily Remark'],
                                          ascending=[False,False,False, False], inplace=True)


#%%

    # generate html report
    def create_html_report(self):

        # generate html string for each category
        for key in self.summary.keys():
            self.html_tables[key] = self.summary[key].to_html().replace('<table border="1" class="dataframe">', f'<table id="{key}_table">')

        sgt = self.end_date.strftime('%d %B %Y %H:%M ') + 'Singapore'
        file_time = dt.datetime.now().strftime('%y%m%d_%H%M')

        self.html_string = '''
        <html>

          <head>

            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">

            <style>
              body {
                margin: 0 100;
                background: whitesmoke;
              }

              table, th, td {
                border-collapse: collapse;
                font-family: calibri;
                text-align: center;
                font-size: 13px;
                table-layout: fixed;
                width: 100px;
              }

              td, th {
                border: 1px solid #dddddd;
                padding: 5px;
              }

              th {
                background-color: #042c58;
                color: #ffffff;
              }

              td:nth-child(3) {border-right: 1.1px solid black}
              td:nth-child(7) {border-right: 1.1px solid black}
              td:nth-child(11) {border-right: 1.1px solid black}

              tr:nth-child(even) {background-color: #eee}
              tr:nth-child(odd) {background-color: #fff}

              tr:hover {background-color: #4f6bed50;}
              button:focus {background-color: #ff5f2b50;}

              .very_bullish {background-color: lime; color: dimgray; font-weight: bold}
              .bullish {background-color: darkgreen; color: white}
              .bearish {background-color: darkred; color: white}
              .very_bearish {background-color: red; color: white; font-weight: bold}
              .uptrend {background-color: green; font-weight: 400; color: white}
              .downtrend {background-color: firebrick; font-weight: 400; color: white}
              .old {opacity: 0.3}
            </style>

            <script>
              function toggleElements(showElement, hideElements) {
                document.querySelectorAll(hideElements).forEach(el => el.style.display = "none");
                document.querySelector(showElement).style.display = "block";
              }
            </script>

            <script>
              function color_code() {
                last_price = document.querySelectorAll("td:nth-child(2)");
                momentum = document.querySelectorAll("td:nth-child(3)");
                daily_k = document.querySelectorAll("td:nth-child(4)");
                daily_d = document.querySelectorAll("td:nth-child(5)");
                daily_remark = document.querySelectorAll("td:nth-child(6)");
                daily_time = document.querySelectorAll("td:nth-child(7)");
                ema_2040 = document.querySelectorAll("td:nth-child(8)");
                sma_50150 = document.querySelectorAll("td:nth-child(9)");
                sma_200 = document.querySelectorAll("td:nth-child(10)");
                medium_trend = document.querySelectorAll("td:nth-child(11)");
                weekly_k = document.querySelectorAll("td:nth-child(12)");
                weekly_d = document.querySelectorAll("td:nth-child(13)");
                weekly_remark = document.querySelectorAll("td:nth-child(14)");
                weekly_time = document.querySelectorAll("td:nth-last-child(1)");

                current_time = Date.parse(Date());
                one_day = 24*60*60*1000;
                oversold = 25; overbought = 85;

                for(i = 0 ; i < last_price.length ; i++) {

                  if(parseFloat(momentum[i].textContent) > 0) {
                    momentum[i].style.color = "forestgreen";
                    momentum[i].style.fontWeight = "bold";
                  }
                  if(parseFloat(momentum[i].textContent) < 0) {
                    momentum[i].style.color = "crimson";
                    momentum[i].style.fontWeight = "bold";
                  }

                  day_k = parseFloat(daily_k[i].textContent);
                  day_d = parseFloat(daily_d[i].textContent);
                  week_k = parseFloat(weekly_k[i].textContent);
                  week_d = parseFloat(weekly_d[i].textContent);
                  day_diff = Math.abs(day_k - day_d);
                  week_diff = Math.abs(week_k - week_d);

                  if(day_k <= oversold && day_d <= oversold) {
                    daily_k[i].bgColor = "#f7ff00"; daily_d[i].bgColor = "#f7ff00"
                  }

                  if(day_k >= overbought && day_d >= overbought) {
                    daily_k[i].bgColor = "lemonchiffon"; daily_d[i].bgColor = "lemonchiffon"
                  }

                  if(week_k <= oversold && week_d <= oversold) {
                    weekly_k[i].bgColor = "#f7ff00"; weekly_d[i].bgColor = "#f7ff00"
                  }

                  if(week_k >= overbought && week_d >= overbought) {
                    weekly_k[i].bgColor = "lemonchiffon"; weekly_d[i].bgColor = "lemonchiffon"
                  }

                  day_rmk = daily_remark[i].textContent
                  med_trd = medium_trend[i].textContent
                  week_rmk = weekly_remark[i].textContent

                  if(day_rmk == "Very Bullish") daily_remark[i].classList.add("very_bullish");
                  if(day_rmk == "Bullish") daily_remark[i].classList.add("bullish");
                  if(day_rmk == "Bearish") daily_remark[i].classList.add("bearish");
                  if(day_rmk == "Very Bearish") daily_remark[i].classList.add("very_bearish");

                  if(week_rmk == "Very Bullish") weekly_remark[i].classList.add("very_bullish");
                  if(week_rmk == "Bullish") weekly_remark[i].classList.add("bullish");
                  if(week_rmk == "Bearish") weekly_remark[i].classList.add("bearish");
                  if(week_rmk == "Very Bearish") weekly_remark[i].classList.add("very_bearish");

                  if(med_trd == "Uptrend") medium_trend[i].classList.add("uptrend");
                  if(med_trd == "Downtrend") medium_trend[i].classList.add("downtrend");

                  if(Math.abs(current_time - Date.parse(daily_time[i].textContent)) / one_day > 7.1) {
                    daily_remark[i].classList.add("old")
                  }

                  if(Math.abs(current_time - Date.parse(weekly_time[i].textContent)) / one_day > 7.1) {
                    weekly_remark[i].classList.add("old")
                  }

                }
              }
            </script>

          </head>

          <body>
            <h1> <u> Technical Watchlist </u> </h1>
            Generated on: ''' + sgt + '''
            <br>
            <br>
            <button onclick="toggleElements('#indices', '.technical')"> Global Indices </button>
            <button onclick="toggleElements('#equities', '.technical')"> Global Equities </button>
            <button onclick="toggleElements('#reits', '.technical')"> Global REITs </button>
            <button onclick="toggleElements('#bonds', '.technical')"> Global Bonds </button>
            <button onclick="toggleElements('#commods', '.technical')"> Commodities </button>
            <button onclick="toggleElements('#cryptos', '.technical')"> Cryptocurrencies </button>
            <br>

            <div class="technical" id="indices">
            <h2> Global Indices </h2>
            ''' + self.html_tables['indices'] + '''
            <br>
            </div>

            <div class="technical" id="equities" style="display: none;">
            <h2> Global Equities </h2>
            ''' + self.html_tables['equities'] + '''
            <br>
            </div>

            <div class="technical" id="reits" style="display: none;">
            <h2> Global REITs </h2>
            ''' + self.html_tables['reits'] + '''
            <br>
            </div>

            <div class="technical" id="bonds" style="display: none;">
            <h2> Global Bonds </h2>
            ''' + self.html_tables['bonds'] + '''
            <br>
            </div>

            <div class="technical" id="commods" style="display: none;">
            <h2> Commodities </h2>
            ''' + self.html_tables['commods'] + '''
            <br>
            </div>

            <div class="technical" id="cryptos" style="display: none;">
            <h2> Cryptocurrencies </h2>
            ''' + self.html_tables['cryptos'] + '''
            <br>
            </div>

          </body>

          <script>color_code()</script>
        </html>
        '''

        # output result to target directory
        file_name = 'tech_watch_' + file_time + '.html'
        self.file_path = os.path.join(os.getcwd(), 'reports', file_name)
        f = open(self.file_path, 'w')
        f.write(self.html_string)
        f.close()


#%%

    # send html report to designated email
    def send_email(self, send=True):

        yag = yagmail.SMTP('samuel.juan.prasetya')
        to = 'sjprasetya.2020@mqf.smu.edu.sg'
        subject = 'Technical Watchlist on ' + self.end_date.strftime('%Y-%m-%d')
        attachments = self.file_path

        today = dt.datetime.today().strftime('%Y-%m-%d')
        ytd = (dt.datetime.today() - BDay(1)).strftime('%Y-%m-%d')

        body = '<h2> Daily Stochastic Turns </h2>'

        for key in self.summary.keys():
            cond_one = self.summary[key]['Daily Time'] == ytd
            cond_two = self.summary[key]['Daily Time'] == today
            daily_turn = self.summary[key][(cond_one) | (cond_two)]
            daily_turn = daily_turn.sort_values(by='Daily Remark', ascending=False)

            if not daily_turn.empty:
                body += '<h3><u>' + key.capitalize() + ' </u></h3>'

                highlight = []
                for i in range(len(daily_turn)):
                    highlight.append(str(daily_turn.index[i] + ' (' + daily_turn.iloc[i]['Daily Remark'] + ')'))

                # attach to email body
                body += '<br>'.join(highlight) + '<br><br>'

        if send: yag.send(to=to, subject=subject, contents=body, attachments=attachments)


#%%

# run program
tech = Technical_Watchlist()
tech.main(True)

