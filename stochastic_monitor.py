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

import pandas_datareader as wb
import datetime as dt
import numpy as np
import yagmail
import os


#%%

class Stochastic_Monitor():

    # set default watchlist
    watchlist = {
        'indices': ['ASHR','DIA','EEM','GXC','KWEB','QQQ','RSP','SPY','^VIX'],

        'equities': ['AAPL','ACN','ADBE','AMZN','ASML','BA','BABA','BAC','CLX','CME',
                     'COST','CRM','EL','FB','GOOGL','ICE','ILMN','INTU','JNJ','JPM',
                     'LMT','MA','MCD','MELI','MMM','MSFT','NKE','PEP','PG','0700.HK',
                     'TSM','UNH','UNP','V','VEEV','YUMC'],

        'reits': ['A17U.SI','AJBU.SI','AU8U.SI','BUOU.SI','BWCU.SI',
                  'C2PU.SI','C38U.SI','K71U.SI','M44U.SI','N2IU.SI','ME8U.SI'],

        'commods': ['GLD','SLV','PDBC','DBC','COMT','CCRV','BDRY'],

        'bonds': ['SHY','IEI','IEF','TLT','BND','BNDX','LQD','EMB'],

        'cryptos': ['BTC-USD','ETH-USD','ADA-USD','BNB-USD','LTC-USD','XRP-USD','BCH-USD']
        }


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
        if start_date is None: self.start_date = self.end_date - dt.timedelta(700)
        else: self.start_date = str(start_date)

        # set other attributes
        self.verbose = verbose
        self.signals = {}
        self.summary = {}
        self.file_path = None
        self.html_string = None


#%%

    # run procedure
    def main(self):

        self.load_data()
        self.get_stochastic_signal()
        self.format_display()
        self.create_html_report()
        self.send_email()


#%%

    # load data from yahoo finance
    def load_data(self):

        # fetch data
        if self.verbose:
            print(f"\nFetching price data of {len([ticker for key in self.watchlist.keys() for ticker in self.watchlist[key]])} tickers..\n")
        daily = wb.DataReader(self.tickers, 'yahoo', self.start_date, self.end_date)[['Open','High','Low','Close']]
        crypto = wb.DataReader(self.watchlist['cryptos'], 'yahoo', self.start_date, self.end_date)[['Open','High','Low','Close']]
        daily = pd.merge(left=crypto, right=daily, how='outer', left_index=True, right_index=True)

        # include cryptos in tickers
        self.tickers += self.watchlist['cryptos']

        # check completeness
        missing = [ticker for ticker in daily.columns.levels[1] if ticker not in self.tickers]
        if missing != [] and self.verbose: print(f"Failed to load tickers: {', '.join(missing)}")

        # resample data from daily to weekly
        weekly_open = daily['Open'].resample('W-FRI').first()
        weekly_high = daily['High'].resample('W-FRI').max()
        weekly_low = daily['Low'].resample('W-FRI').min()
        weekly_close = daily['Close'].resample('W-FRI').last()
        weekly = pd.concat([weekly_open, weekly_high, weekly_low, weekly_close],
                           axis=1, keys=['Open','High','Low','Close'])

        # change index data type to string
        daily.index = daily.index.astype(str)
        weekly.index = weekly.index.astype(str)

        # change the latest weekly date to avoid forward bias
        weekly.index = weekly.index[:-1].tolist() + [daily.index[-1]]

        # assign to attributes
        self.daily = daily
        self.weekly = weekly


#%%

    # compute stochastic signal
    def get_stochastic_signal(self, ma_type='smoothed', lookback=9, k_period=3, d_period=3, oversold=25, overbought=75):

        # name parameters
        pct_k = '_%K(' + str(lookback) + ',' + str(k_period) + ')'
        pct_d = '_%D(' + str(d_period) + ')'

        for ticker in self.tickers:

            # print status
            if self.verbose: print(f"Evaluating stochastic signals for {ticker}..")

            # prepare dictionary to capture result
            if ticker not in self.summary.keys(): self.summary[ticker] = {}
            if ticker not in self.signals.keys(): self.signals[ticker] = {}

            # loop over daily and weekly dataframe
            for i in range(2):

                # assign ohlc and timeframe
                if i == 0:
                    ohlc = self.weekly.xs(ticker, axis=1, level=1).dropna().copy()
                    timeframe = 'Weekly'
                else:
                    ohlc = self.daily.xs(ticker, axis=1, level=1).dropna().copy()
                    timeframe = 'Daily'

                # compute rolling high and low during lookback period
                ohlc['Rolling_High'] = ohlc['High'].rolling(window=lookback).max()
                ohlc['Rolling_Low'] = ohlc['Low'].rolling(window=lookback).min()

                # compute %K
                ohlc['%K'] = ((ohlc['Close']-ohlc['Rolling_Low']) / (ohlc['Rolling_High']-ohlc['Rolling_Low'])) * 100

                # compute full K and full D lines
                if ma_type == 'simple':
                    ohlc['Full_K'] = ohlc['%K'].rolling(window=k_period).mean()
                    ohlc['Full_D'] = ohlc['Full_K'].rolling(window=d_period).mean()

                elif ma_type == 'exponential':
                    ohlc['Full_K'] = ohlc['%K'].ewm(span=k_period).mean()
                    ohlc['Full_D'] = ohlc['Full_K'].ewm(span=d_period).mean()

                elif ma_type == 'smoothed':
                    smoothed_k = 2 * k_period - 1
                    smoothed_d = 2 * d_period - 1
                    ohlc['Full_K'] = ohlc['%K'].ewm(span=smoothed_k).mean()
                    ohlc['Full_D'] = ohlc['Full_K'].ewm(span=smoothed_d).mean()

                else:
                    print(f"Supported ma_type: simple, exponential, smoothed. {ma_type} was passed..")

                # compute crossover and change in signal
                ohlc['Cross'] = ohlc['Full_K'] - ohlc['Full_D']
                ohlc['Change'] = np.sign(ohlc['Cross']).diff()
                ohlc['Remarks'] = ''

                # give verdict
                for i in range(1, len(ohlc)):

                    # reference
                    date = ohlc.index[i]
                    change_now = ohlc.iloc[i]['Change']
                    prev_k = ohlc.iloc[i-1]['Full_K']
                    prev_d = ohlc.iloc[i-1]['Full_D']

                    # bullish signal
                    if change_now > 0:
                        if (prev_k <= oversold) and (prev_d <= oversold):
                            ohlc.loc[date,'Remarks'] = 'Very Bullish'
                        else:
                            ohlc.loc[date,'Remarks'] = 'Bullish'

                    # bearish signal
                    elif change_now < 0:
                        if (prev_k >= overbought) and (prev_d >= overbought):
                            ohlc.loc[date,'Remarks'] = 'Very Bearish'
                        else:
                            ohlc.loc[date,'Remarks'] = 'Bearish'

                    # no confirmed signal
                    else: pass

                # assign to attributes
                self.signals[ticker][timeframe] = ohlc.iloc[2*lookback:]
                self.summary[ticker]['Last_Price'] = ohlc.iloc[-1]['Close']
                self.summary[ticker][str(timeframe) + pct_k] = ohlc[ohlc['Remarks'] != ''].iloc[-1]['Full_K']
                self.summary[ticker][str(timeframe) + pct_d] = ohlc[ohlc['Remarks'] != ''].iloc[-1]['Full_D']
                self.summary[ticker][str(timeframe) + '_Remarks'] = ohlc[ohlc['Remarks'] != ''].iloc[-1]['Remarks']
                self.summary[ticker][str(timeframe) + '_Timestamp'] = ohlc[ohlc['Remarks'] != ''].index[-1]


#%%

    # format display of summary dataframe
    def format_display(self):

        # transform dictionary to dataframe
        summary = pd.DataFrame(self.summary).T.copy()

        # format display
        dec_2dp = lambda x: str('{:.2f}'.format(x))
        no_format = ['Daily_Timestamp','Daily_Remarks','Weekly_Timestamp','Weekly_Remarks']
        summary.loc[:, [col for col in summary.columns if col not in no_format]] = \
        summary.loc[:, [col for col in summary.columns if col not in no_format]].applymap(dec_2dp).values

        # split into different tabs
        self.summary = {}
        for key in self.watchlist.keys():
            self.summary[key] = summary[summary.index.isin(self.watchlist[key])].copy()

        # sort dataframe
        for tab in self.summary.keys():
            self.summary[tab].sort_values(by=['Weekly_Timestamp','Daily_Timestamp','Weekly_Remarks','Daily_Remarks'],
                                          ascending=[False,False,False,False], inplace=True)


#%%

    # generate html report
    def create_html_report(self):

        # generate html string for each category
        indices = self.summary['indices'].to_html().replace('<table border="1" class="dataframe">', '<table id="indices_table">')
        equities = self.summary['equities'].to_html().replace('<table border="1" class="dataframe">', '<table id="equities_table">')
        reits = self.summary['reits'].to_html().replace('<table border="1" class="dataframe">', '<table id="reits_table">')
        bonds = self.summary['bonds'].to_html().replace('<table border="1" class="dataframe">', '<table id="bonds_table">')
        commods = self.summary['commods'].to_html().replace('<table border="1" class="dataframe">', '<table id="commods_table">')
        cryptos = self.summary['cryptos'].to_html().replace('<table border="1" class="dataframe">', '<table id="cryptos_table">')

        sgt = self.end_date.strftime('%d %b %Y ') + 'SGT'
        file_time = dt.datetime.now().strftime('%y%m%d')

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
                width: 120px;
              }

              td, th {
                border: 1px solid #dddddd;
                padding: 5px;
              }

              th {
                background-color: #042c58;
                color: #ffffff;
              }

              td:nth-last-child(4) {border-left: 1.1px solid black}
              td:nth-last-child(8) {border-left: 1.1px solid black}

              tr:nth-child(even) {background-color: #eee}
              tr:nth-child(odd) {background-color: #fff}

              tr:hover {background-color: #4f6bed50;}
              button:focus {background-color: #ff5f2b50;}
            </style>

            <script>
              function toggleElements(showElement, hideElements) {
                document.querySelectorAll(hideElements).forEach(el => el.style.display = "none");
                document.querySelector(showElement).style.display = "block";
              }
            </script>

          </head>

          <body>
            <h1> <u> Stochastic Crossover Monitor </u> </h1>
            <h4> Report is Generated on: ''' + sgt + ''' </h4>
            <br>
            <button onclick="toggleElements('#indices', '.stochastic')"> Global Indices </button>
            <button onclick="toggleElements('#equities', '.stochastic')"> Global Equities </button>
            <button onclick="toggleElements('#reits', '.stochastic')"> Global REITs </button>
            <button onclick="toggleElements('#bonds', '.stochastic')"> Global Bonds </button>
            <button onclick="toggleElements('#commods', '.stochastic')"> Commodities </button>
            <button onclick="toggleElements('#cryptos', '.stochastic')"> Cryptocurrencies </button>
            <br>

            <div class="stochastic" id="indices">
            <h2> Global Indices </h2>
            ''' + indices + '''
            <br>
            </div>

            <div class="stochastic" id="equities" style="display: none;">
            <h2> Global Equities </h2>
            ''' + equities + '''
            <br>
            </div>

            <div class="stochastic" id="reits" style="display: none;">
            <h2> Global REITs </h2>
            ''' + reits + '''
            <br>
            </div>

            <div class="stochastic" id="bonds" style="display: none;">
            <h2> Global Bonds </h2>
            ''' + bonds + '''
            <br>
            </div>

            <div class="stochastic" id="commods" style="display: none;">
            <h2> Commodities </h2>
            ''' + commods + '''
            <br>
            </div>

            <div class="stochastic" id="cryptos" style="display: none;">
            <h2> Cryptocurrencies </h2>
            ''' + cryptos + '''
            <br>
            </div>

          </body>

        </html>
        '''

        # output result to target directory
        file_name = 'stochastic_monitor_' + file_time + '.html'
        self.file_path = os.path.join(os.getcwd(), 'reports', file_name)
        f = open(self.file_path, 'w')
        f.write(self.html_string)
        f.close()


#%%

    # send html report to designated email
    def send_email(self):

        yag = yagmail.SMTP('samuel.juan.prasetya')
        to = 'sjprasetya.2020@mqf.smu.edu.sg'
        subject = 'Stochastic Monitor on ' + self.end_date.strftime('%Y-%m-%d')
        attachments = self.file_path

        body = """\
        Hello there,

        Sending daily stochastic monitor on global asset classes.

        Best Regards,

        Samuel Juan

        <i> This email is auto-generated using Python </i>
        """

        yag.send(to=to, subject=subject, contents=body, attachments=attachments)


#%%

# run program
stoch = Stochastic_Monitor()
stoch.main()







