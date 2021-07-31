# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 09:53:26 2021

@author: samueljuan
"""

# set github directory
import os
os.chdir(r'C:\Users\User\Documents\GitHub\investment_analysis')

# import libraries
import pandas as pd
pd.options.display.float_format = "{:,.2f}".format
import pandas_datareader as wb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.rolling import RollingOLS

#%%

# retrieve data
data = pd.read_csv('spx_member_price.csv', index_col=0, parse_dates=True)
beta = pd.read_csv('rolling_2y_beta.csv', index_col=0, parse_dates=True)
spx = wb.DataReader('^GSPC', 'yahoo', data.index[0], data.index[-1])[['Adj Close']]
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
#beta.to_csv('rolling_2y_beta.csv')

#%%


