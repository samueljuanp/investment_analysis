import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import datetime as dt
from pandas_datareader import data as wb
from yahoofinancials import YahooFinancials as yf
import matplotlib.pyplot as plt
import seaborn as sns
from currency_converter import CurrencyConverter

sns.set_style('darkgrid')
ratio = (18, 12)


def stock_return_risk(ticker, method='log'):
    """
    This function is to calculate the return and risk of a single stock by either using simple or natural logarithm method.
    Two arguments are required for this function: the stock ticker and the calculation method.
    By default, the calculation method for a single stock is using natural logarithm.
    """
    y = wb.DataReader(ticker, 'yahoo', '1998-01-01')
    y.dropna(inplace=True)

    if method == 'log':
        y['Log Return'] = np.log(y['Adj Close'] / y['Adj Close'].shift(1))
        log_risk = y['Log Return'].std() * math.sqrt(250) * 100
        print('Average standard deviation is ' + str(round(log_risk, 2)) + '%')
        print('Average log annual return is ' + str(round(y['Log Return'].mean() * 250 * 100, 2)) + '%')

    elif method == 'simple':
        y['Simple Return'] = y['Adj Close'] / y['Adj Close'].shift(1) - 1
        simple_risk = y['Simple Return'].std() * math.sqrt(250) * 100
        print('Average standard deviation is ' + str(round(simple_risk, 2)) + '%')
        print('Average simple annual return is ' + str(round(y['Simple Return'].mean() * 250 * 100, 2)) + '%')

    y['Adj Close'].plot(figsize=ratio)
    plt.title(str('Historical Price Performance of ' + ticker))


def portfolio_performance(start_date='1998-01-01', equal_dist=False):
    """
    This function is to calculate the return and risk of given stocks and specified allocation strategy.
    It will also do a benchmark against S&P500 Index to observe the performance over time.
    Starting period is, by default, from the year of 1998.
    If equal_dist is True, it will divide the portfolio allocation equally.
    """
    my_stocks = []
    allocation = np.array([])

    # state the number of stocks you would like to have
    n = int(input('How many stocks would you like to own? '))

    # creating a list of stocks
    for i in range(0, n):
        stock = input('Enter the ticker = ').upper()
        my_stocks.append(stock)
    print('')

    if equal_dist == False:
        # to input the weightage % allocated to each stock
        for i in range(0, n):
            weight = eval(input('For ' + my_stocks[i] + ', what is the weightage? '))
            allocation = np.append(allocation, weight)

        # checking the sum of allocation % - whether it is equal to 1
        check = round(np.sum(allocation), 5)

        while check != 1:
            print('')
            print('Allocation % is not equal to 1. Please repeat the input!')
            for i in range(0, n):
                weight = eval(input('For ' + my_stocks[i] + ', what is the weightage? '))
                allocation[i] = weight
            check = np.sum(allocation)

    elif equal_dist == True:
        # to equally divide the stock allocation percentage in the portfolio
        for i in range(0, n):
            weight = 1 / n
            allocation = np.append(allocation, weight)

    # creating dataframe and comparison visualization for the portfolio owned
    my_stocks.append('SPY')
    allocation = np.append(allocation, 0)

    my_table = pd.DataFrame()
    for stock in my_stocks:
        my_table[stock] = wb.DataReader(stock, 'yahoo', start_date)['Adj Close']
    my_table.dropna(inplace=True)
    my_table = my_table / my_table.iloc[0] * 100

    my_price = []
    for i in range(len(my_table)):
        my_price.append(np.dot(my_table.iloc[i], allocation))
    my_table['My Portfolio'] = my_price

    my_table[['SPY', 'My Portfolio']].plot(figsize=ratio)
    plt.title('The Performance of My Portfolio vs. S&P 500 Index')

    # calculating portfolio simple return based on % allocation
    pfolio_simple_ret = my_table.iloc[:, :-2].pct_change(1)
    my_portfolio_return = np.dot(pfolio_simple_ret.mean() * 250, allocation[:-1])
    my_portfolio_risk = math.sqrt(np.dot(allocation[:-1].T, np.dot(pfolio_simple_ret.cov() * 250, allocation[:-1])))
    print('')
    print('Average simple annual return of the portfolio is ' + str(round(my_portfolio_return * 100, 2)) + '%')
    print('Average standard deviation of the portfolio is ' + str(round(my_portfolio_risk * 100, 2)) + '%')
    print('Sharpe ratio of the portfolio is ' + str(round(my_portfolio_return / my_portfolio_risk, 2)))


def efficient_frontier():
    """
    This function is to calculate the optimum allocation of given stocks by simulating 10000 possible combinations and picking one that maximizes return and minimizes risk.
    No argument is required for this function.
    """
    my_stocks = []

    # state the number of stocks you would like to have
    n = int(input('How many stocks would you like to own? '))

    # creating a list of stocks
    for i in range(0, n):
        my_stocks.append(input('Enter the ticker = '))
    print('')

    # extracting price data from yahoo finance
    table = pd.DataFrame()
    for item in my_stocks:
        table[item] = wb.DataReader(item, 'yahoo', '1998-01-01')['Adj Close']

    # replacing NaN values with the mean return of the stock
    simple_returns = table.pct_change(1)
    for item in my_stocks:
        simple_returns[item].fillna(simple_returns[item].mean(), inplace=True)

    # creating dataframes and 20000 possible allocation combinations
    pfolio_return = np.array([])
    pfolio_risk = np.array([])
    pfolio_ratio = np.array([])
    allocation = pd.DataFrame(columns=my_stocks)

    for x in range(20000):
        weights = np.random.random(n)
        weights /= np.sum(weights)  # to make sum of weights equals to 1
        allocation.loc[x, :] = weights
        ret_row = np.dot(simple_returns.mean() * 250, weights)
        risk_row = np.sqrt(np.dot(weights.T, np.dot(simple_returns.cov() * 250, weights)))

        pfolio_return = np.append(pfolio_return, ret_row)
        pfolio_risk = np.append(pfolio_risk, risk_row)
        pfolio_ratio = np.append(pfolio_ratio, ret_row / risk_row)

    # manipulating the dataframes
    portfolios = pd.DataFrame({'Risk': pfolio_risk, 'Return': pfolio_return, 'Ratio': pfolio_ratio})
    portfolios = pd.concat([portfolios, allocation], axis=1, sort=False)

    # pinpointing return and risk that gives maximum Sharpe ratio
    max_pf_ret = pfolio_return[pfolio_ratio.argmax()]
    min_pf_risk = pfolio_risk[pfolio_ratio.argmax()]

    # printing the desired values
    print('The maximum Sharpe ratio of the portfolio is ' + str(round(pfolio_ratio.max(), 3)))
    print('The corresponding return is ' + str(round(max_pf_ret * 100, 2)) + '%')
    print('The corresponding risk is ' + str(round(min_pf_risk * 100, 2)) + '%')
    print('')

    for item in my_stocks:
        print('The optimum allocation for ' + item + ' is ' + str(
            round(portfolios[item].iloc[pfolio_ratio.argmax()] * 100, 2)) + '%')

    # creating visualization of efficient frontier chart
    plt.figure(figsize=(19, 12))
    plt.scatter(x=pfolio_risk, y=pfolio_return, c=pfolio_ratio, cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Expected Risk')
    plt.ylabel('Expected Return')
    plt.title('The Efficient Frontier')
    plt.scatter(x=min_pf_risk, y=max_pf_ret, c='red', s=75, edgecolors='black')


def monte_carlo(x):
    """
    This function is to predict 5-year future prices of a stock by simulating 1000 combinations of daily return, which is made up of drift and volatility.
    An argument is required for this function. Do pass a stock ticker to check its possible future mean, median, max, and min price.
    """
    data = pd.DataFrame()
    data[x] = wb.DataReader(x, 'yahoo', '1990-01-01')['Adj Close']
    data['Log Return'] = np.log(data[x] / data[x].shift(1))

    # calculating the required component: drift
    u = data['Log Return'].mean()
    var = data['Log Return'].var()
    stdev = data['Log Return'].std()
    drift = u - 0.5 * var
    drift = np.array(drift)
    stdev = np.array(stdev)

    # calculating the required component: volatility
    interval = 5 * 250
    iterations = 1000
    volatility = stdev * norm.ppf((np.random.rand(interval, iterations)))

    # calculating daily returns
    daily_ret = np.exp(drift + volatility)

    # simulating possible future prices
    latest_price = data[x].iloc[-1]
    price_list = np.zeros_like(daily_ret)
    price_list[0] = latest_price

    for i in range(1, interval):
        price_list[i] = price_list[i - 1] * daily_ret[i]

    plt.figure(figsize=ratio)
    plt.plot(price_list)
    plt.title('Monte Carlo Simulation')
    plt.xlabel('Five-Year Time Horizon')
    plt.ylabel('Possible Future Prices')

    print('')
    print('The latest stock price = ' + str(round(latest_price, 2)))
    print('The median simulated price in 5 years = ' + str(round(np.median(price_list[interval - 1]), 2)))
    print('The mean simulated price in 5 years = ' + str(round(price_list[interval - 1].mean(), 2)))
    print('The maximum simulated price in 5 years = ' + str(round(price_list[interval - 1].max(), 2)))
    print('The minimum simulated price in 5 years = ' + str(round(price_list[interval - 1].min(), 2)))


def compare_stocks(freq='W', start_date='1998-01-01'):
    """
    This function is to compare risk, return, and normalized price performance of several stocks over a period of time.
    Default frequency is B. Other available frequencies are W, BQ, M, and A.

    """
    my_stocks = []
    my_columns = []

    # state the number of stocks you would like to have
    n = int(input('How many stocks would you like to view? '))

    # creating a list of stocks and the respective columns
    for i in range(0, n):
        stock = input('Enter the ticker = ')
        stock_ret = str(stock + '_Return')
        my_stocks.append(stock)
        my_columns.append(stock)
        my_columns.append(stock_ret)
    print('')

    # creating dataframe
    my_table = pd.DataFrame(columns=my_columns)
    for stock in my_stocks:
        my_table[stock] = wb.DataReader(stock, 'yahoo', start_date)['Adj Close']
    for i in range(1, n * 2, 2):
        my_table.iloc[:, i] = np.log(my_table.iloc[:, i - 1] / my_table.iloc[:, i - 1].shift(1))
    my_table.dropna(inplace=True)

    # creating comparison table
    my_data = pd.DataFrame(index=['Average Return', 'Average Risk', 'Return/Risk'], columns=my_stocks)

    # populating average risk and return row
    ris = []
    ret = []
    rat = []
    for i in range(1, n * 2, 2):
        var_ret = str(round(my_table.iloc[:, i].mean() * 250 * 100, 2)) + '%'
        var_risk = str(round(my_table.iloc[:, i].std() * np.sqrt(250) * 100, 2)) + '%'
        var_rat = round((my_table.iloc[:, i].mean() * 250 * 100) / (my_table.iloc[:, i].std() * np.sqrt(250) * 100), 3)

        ret.append(var_ret)
        ris.append(var_risk)
        rat.append(var_rat)

    my_data.iloc[0] = ret
    my_data.iloc[1] = ris
    my_data.iloc[2] = rat
    print(my_data, end='\n\n')

    # getting correlation table
    temp1 = my_table[my_stocks]
    print('Below is the correlation table:', end='\n\n')
    print(temp1.corr())

    # plotting a comparison graph based on resampled frequency
    temp1 = temp1 / temp1.iloc[0] * 100
    temp1.resample(rule=freq).mean().plot(figsize=ratio, lw=2)
    plt.legend(loc=2)
    plt.title('Comparison of Price Performance')


def best_x_dji(x):
    """
    This function is to find out the top performing Dow Jones listed companies based on the latest normalized stock prices.
    It will also show the performance comparison table against Dow Jones Index.
    One argument is required for this function: best of what?
    For example, x = 20 means top 20 companies in S&P500 index based on their latest normalized stock prices.
    """

    # getting listed companies inside Dow Jones index
    website = 'https://ycharts.com/companies/DIA/holdings'
    data = pd.read_html(website)
    tickers = list(data[0]['Symbol'])
    tickers.append('DIA')

    # creating a dataframe consisting normalized stock prices of those listed companies
    my_table = pd.DataFrame()
    for stock in tickers:
        my_table[stock] = wb.DataReader(stock, 'yahoo', '1998-01-01')['Adj Close']
    my_table.dropna(inplace=True)
    my_table = my_table / my_table.iloc[0] * 100

    # finding out the x best companies based on latest normalized stock price
    mod_table = pd.DataFrame()
    mod_table['Last Price'] = my_table.T.iloc[:, -1]
    mod_table['Ticker'] = my_table.T.index
    mod_table.set_index('Last Price', inplace=True)

    x_best = []
    for i in range(x):
        stock = mod_table.loc[sorted(mod_table.index)[-i - 1], 'Ticker']
        x_best.append(stock)
    x_best.append('DIA')

    # plotting performance comparison chart
    print('')
    print('The {} best companies inside Dow Jones Index are {}'.format(x, x_best[:-1]))
    my_table[x_best].plot(figsize=ratio)
    plt.legend(loc=2)
    plt.title('Performance Comparison against Dow Jones Index')


def etf_information(x):
    website = 'https://ycharts.com/companies/' + x.upper()
    stock = pd.read_html(website)
    print('')
    print(stock[10].iloc[1,0], end='\n\n')
    print('-------------- Key Information -------------', end='\n\n')
    print('Expense Ratio = {}%'.format(stock[0].iloc[0, 0]))
    print('Total AUM = {}'.format(stock[0].iloc[0, 2]))
    print('Average Daily Volume = {}'.format(stock[0].iloc[0, 3]))
    print('Dividend Yield (TTM) = {}%'.format(stock[13].iloc[2, 1]))
    print('Weighted Average PE Ratio = {}'.format(stock[13].iloc[7, 1]))
    print('')
    print('--------------------------------------------', end='\n\n')

    # getting asset allocation table
    stock[8].columns = ['Type', 'Allocation', '% Long', '% Short']
    print(stock[8][['Type','Allocation']].set_index('Type'), end='\n\n')

    # getting top 10 holdings table
    stock[9].columns = ['Company Name', 'Weightage', 'Price', '% Change']
    print(stock[9][['Company Name', 'Weightage']].set_index('Company Name'), end='\n\n')


def fundamental_analysis(ticker):
    """
    This function is to extract and compare business bottomline.
    One argument is required: the ticker of the company

    """
    # to pull income statement and cash flow of the company from yahoo finance
    income_statement = yf(ticker).get_financial_stmts('annual', 'income')
    cash_flow = yf(ticker).get_financial_stmts('annual', 'cash')
    balance_sheet = yf(ticker).get_financial_stmts('annual', 'balance')

    # to extract information from YahooFinancials JSON format
    year = []
    total_revenue = []
    net_income = []
    op_cashflow = []
    stockholder_eq = []
    current_asset = []
    current_liab = []
    total_liab = []
    interest_exp = []
    short_debt = []
    long_debt = []
    ebit = []
    gross_profit = []

    for item in income_statement['incomeStatementHistory'][ticker]:
        for key, val in item.items():
            year.append(key[0:4])  # to get the year only
            total_revenue.append(val['totalRevenue'] / 1000000)
            net_income.append(val['netIncome'] / 1000000)
            # to handle None type data
            interest_exp.append(int(0 if val['interestExpense'] is None else val['interestExpense']) / 1000000)
            ebit.append(val['ebit'] / 1000000)
            gross_profit.append(val['grossProfit'] / 1000000)

    for item in cash_flow['cashflowStatementHistory'][ticker]:
        for key, val in item.items():
            op_cashflow.append(val['totalCashFromOperatingActivities'] / 1000000)

    for item in balance_sheet['balanceSheetHistory'][ticker]:
        for key, val in item.items():
            stockholder_eq.append(val['totalStockholderEquity'] / 1000000)
            current_asset.append(val['totalCurrentAssets'] / 1000000)
            current_liab.append(val['totalCurrentLiabilities'] / 1000000)
            total_liab.append(val['totalLiab'] / 1000000)

    # to handle missing information
    for i in range(len(year)):
        x = list(balance_sheet['balanceSheetHistory'][ticker][i].values())
        x = pd.DataFrame(x)

        if 'shortLongTermDebt' in x.columns:
            short_debt.append(int(x['shortLongTermDebt']) / 1000000)
        else:
            short_debt.append(0)

        if 'longTermDebt' in x.columns:
            long_debt.append(int(x['longTermDebt']) / 1000000)
        else:
            long_debt.append(0)

    # to check if revenue, profit, and operating CF are consistently increasing
    step_one = pd.DataFrame(columns=year,
                            index=['Revenue', 'Net Income', 'Cash Flow'])

    step_one.iloc[0] = total_revenue
    step_one.iloc[1] = net_income
    step_one.iloc[2] = op_cashflow

    # to normalize the value so as to visualize the trend better in one chart
    trend = step_one.copy()
    for i in range(len(trend.columns)):
        trend.iloc[:, i] = trend.iloc[:, i] / abs(trend.iloc[:, -1])

    # to set the width of the bar
    width = 0.25

    # to set the position of each bar on x-axis
    r1 = np.arange(len(trend.loc['Revenue']))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]

    print('')
    print('Step 1: Revenue, Net Income and Operating Cash Flow are Consistently Increasing', end='\n')

    # to make the bar plots
    plt.figure(figsize=ratio, dpi=100)
    plt.bar(r1, trend.loc['Revenue'], color='purple', width=width, edgecolor='white', label='Total Revenue', alpha=0.65)
    plt.bar(r2, trend.loc['Net Income'], color='orange', width=width, edgecolor='white', label='Net Income', alpha=0.65)
    plt.bar(r3, trend.loc['Cash Flow'], color='green', width=width, edgecolor='white', label='Operating Cash Flow',
            alpha=0.65)

    # to add year label on the middle of the group bars
    plt.xticks([r + width for r in range(len(trend.loc['Revenue']))], year)
    plt.legend()
    plt.title(str('Normalized Trend of Total Revenue, Net Income, and Operating Cash Flow of ' + ticker))

    # to rearrange the information is a dataframe format
    roe = []
    current_ratio = []
    de_ratio = []
    debt_serv_ratio = []
    net_profit_margin = []
    debt_ebit_ratio = []
    gross_profit_margin = []

    for i in range(len(net_income)):
        roe.append(net_income[i] / stockholder_eq[i])
        current_ratio.append(current_asset[i] / current_liab[i])
        de_ratio.append(total_liab[i] / stockholder_eq[i])
        debt_serv_ratio.append(abs(interest_exp[i]) / op_cashflow[i])
        net_profit_margin.append(net_income[i] / total_revenue[i])
        gross_profit_margin.append(gross_profit[i] / total_revenue[i])

        if ebit[i] != 0:
            debt_ebit_ratio.append((short_debt[i] + long_debt[i]) / ebit[i])
        else:
            debt_ebit_ratio.append((short_debt[i] + long_debt[i]) / net_income[i])

    step_two = pd.DataFrame(columns=year,
                            index=['Current Ratio > 1.00', 'Liability-to-Equity Ratio < 1.00',
                                   'Total-Debt-to-EBIT Ratio < 2.50', 'Debt Servicing Ratio < 0.30'])

    step_two.iloc[0] = ['%.2f' % elem for elem in current_ratio]
    step_two.iloc[1] = ['%.2f' % elem for elem in de_ratio]
    step_two.iloc[2] = ['%.2f' % elem for elem in debt_ebit_ratio]
    step_two.iloc[3] = ['%.2f' % elem for elem in debt_serv_ratio]

    print('')
    print('Step 2: Conservative Debt Level', end='\n')
    print('')
    print(step_two)

    step_three = pd.DataFrame(columns=year,
                              index=['ROE > 0.15', 'Gross Profit Margin', 'Net Profit Margin'])

    step_three.iloc[0] = ['%.2f' % elem for elem in roe]
    step_three.iloc[1] = ['%.2f' % elem for elem in gross_profit_margin]
    step_three.iloc[2] = ['%.2f' % elem for elem in net_profit_margin]

    print('')
    print('Step 3: High Return on Equity and Sustainable Competitive Advantage', end='\n\n')
    print(step_three)
    print('')

    # to get EPS growth rate = PE ratio / PEG --> TO FURTHER REFINE
    peg = yf(ticker).get_key_statistics_data()[ticker]['pegRatio']
    peg = float(0.00 if peg is None else peg)
    pe = yf(ticker).get_pe_ratio()
    pe = float(0.00 if pe is None else pe)

    eps = pe / peg
    eps = '%.2f' % eps

    print('Step 4: Positive EPS Long-Term Growth Rate ({}%)'.format(eps), end='\n\n')


def stock_valuation(ticker, method='DCF', terminal_year=10, country='US', print_info=False):
    """
    This function is to automatically calculate intrinsic value of a company.

    Two methodology options: 'Discounted Operating Cash Flow' or 'Discounted Net Income'.
    By default, method is set as 'DCF'.

    Terminal value can be specified up to 20 years.
    By default, it is set as 10.

    Two country options: US or CH.
    By default, it is set as 'US'.
    'CH' is general for either companies listed in HK or SS.
    Changing to 'CH' will impact discount rate and GDP growth rate.

    Everything is in terms of millions.
    """

    # to get user's input on growth rate forecast
    print('')
    growth_rate_1_to_5 = float(input('The forecasted growth rate for the 1st to 5th year (%) = ')) / 100
    growth_rate_6_to_10 = float(input('The forecasted growth rate for the 6th to 10th year (%) = ')) / 100

    # to get beta of the stock
    beta = yf(ticker).get_beta()
    beta = 1.00 if beta == None else beta  # if no beta is found, assume same as market
    beta = 0.80 if beta < 0.80 else beta  # to be conservative, min. beta = 0.80

    # to calculate discount rate based on different markets (US vs CH)
    # average US market risk premium = 5.00%
    # US market risk free rate = 0.64%
    # average CH market risk premium = 6.60&
    # US market risk free rate = 0.60%

    # US long term GDP growth rate + inflation = 4.18%
    # CH long term GDP growth rate + inflation = 8.00%

    # discount rate = risk free rate + (beta * market risk premium)

    if country.upper() == 'US':
        discount_rate = 0.0064 + (beta * 0.05)
        growth_rate_11_to_20 = 0.0418

    elif country.upper() == 'CH':
        discount_rate = 0.006 + (beta * 0.066)
        growth_rate_11_to_20 = 0.08

    print('')
    print('The forecasted growth rate for the 11th to 20th year (%) = {}'.format(round(growth_rate_11_to_20 * 100, 2)))

    # to get quarerly and TTM data
    income_statement = yf(ticker).get_financial_stmts('quarterly', 'income')
    cash_flow = yf(ticker).get_financial_stmts('quarterly', 'cash')
    balance_sheet = yf(ticker).get_financial_stmts('quarterly', 'balance')

    # to change JSON format to list
    op_cashflow = []
    net_income = []

    for item in cash_flow['cashflowStatementHistoryQuarterly'][ticker]:
        for key, val in item.items():
            op_cashflow.append(val['totalCashFromOperatingActivities'] / 1000000)

    for item in income_statement['incomeStatementHistoryQuarterly'][ticker]:
        for key, val in item.items():
            net_income.append(val['netIncome'] / 1000000)

    ttm_op_cashflow = sum(op_cashflow)
    ttm_net_income = sum(net_income)

    # to handle missing values in extracting last quarter total cash and debt
    x = list(balance_sheet['balanceSheetHistoryQuarterly'][ticker][0].values())
    x = pd.DataFrame(x)

    if 'cash' in x.columns:
        last_quarter_cash = x['cash']
    else:
        last_quarter_cash = 0

    if 'shortTermInvestments' in x.columns:
        last_quarter_short_inv = x['shortTermInvestments']
    else:
        last_quarter_short_inv = 0

    if 'longTermDebt' in x.columns:
        last_quarter_long_debt = x['longTermDebt']
    else:
        last_quarter_long_debt = 0

    if 'shortLongTermDebt' in x.columns:
        last_quarter_short_debt = x['shortLongTermDebt']
    else:
        last_quarter_short_debt = 0

    last_quarter_total_cash = (last_quarter_cash + last_quarter_short_inv) / 1000000
    last_quarter_total_debt = (last_quarter_long_debt + last_quarter_short_debt) / 1000000

    # to get discount factor and 20-year projected operating cashflow for dataframe creation

    if method == 'DCF':
        starting_position = ttm_op_cashflow

    elif method == 'DNI':
        starting_position = ttm_net_income

    current_year = dt.datetime.today().year
    year = []
    discount_factor = []
    projected_op_cashflow = []

    for i in range(1, 21):
        x = current_year + i
        year.append(x)

        y = round(1 / ((1 + discount_rate) ** i), 2)
        discount_factor.append(y)

        if i <= 5:
            z = round(starting_position * ((1 + growth_rate_1_to_5) ** i), 2)
            projected_op_cashflow.append(z)

        elif i <= 10:
            z = round(projected_op_cashflow[-1] * (1 + growth_rate_6_to_10), 2)
            projected_op_cashflow.append(z)

        elif i <= 20:
            z = round(projected_op_cashflow[-1] * (1 + growth_rate_11_to_20), 2)
            projected_op_cashflow.append(z)

            # to make the dataframe
    df = pd.DataFrame(index=year,
                      columns=['Projected Operating Cash Flow', 'Discount Factor', 'Discounted Value'])

    df['Projected Operating Cash Flow'] = projected_op_cashflow
    df['Discount Factor'] = discount_factor
    df['Discounted Value'] = df['Discount Factor'] * df['Projected Operating Cash Flow']

    # to obtain shares outstanding and currency
    shares_outstanding = yf(ticker).get_num_shares_outstanding(price_type='average') / 1000000
    currency = yf(ticker).get_currency()

    # to calculate sum of present value based on terminal year input
    total_present_value = df['Discounted Value'][0:terminal_year].sum()

    # to calculate intrinsic value
    final_intrinsic_value = float(
        (total_present_value + last_quarter_total_cash - last_quarter_total_debt) / shares_outstanding)

    # to convert intrinsic value to the currency of quote price

    if country == 'CH' and ticker.split('.')[-1] == ticker:  # chinese companies listed in US
        final_intrinsic_value = final_intrinsic_value * CurrencyConverter().convert(1, 'CNY', 'USD')

    elif country == 'CH' and ticker.split('.')[-1] == 'HK':  # chinese companies listed in HK
        final_intrinsic_value = final_intrinsic_value * CurrencyConverter().convert(1, 'CNY', 'HKD')

    elif country == 'CH' and ticker.split('.')[-1] == 'TW':  # chinese companies listed in TW
        final_intrinsic_value = final_intrinsic_value * CurrencyConverter().convert(1, 'TWD', 'TWD')

    elif country == 'CH' and ticker.split('.')[-1] == 'SS':  # chinese companies list in CH
        final_intrinsic_value = final_intrinsic_value * CurrencyConverter().convert(1, 'CNY', 'CNY')

    # to get latest close price
    current_price = yf(ticker).get_current_price()

    # to print results
    print('')
    print('===================================================================')
    print('')
    print('Intrinsic value of {} is around {} {}'.format(ticker, currency, round(final_intrinsic_value, 2)))
    print('')
    print('Latest close price of {} is at {} {}'.format(ticker, currency, round(current_price, 2)))
    print('')

    delta = abs(round(((current_price - final_intrinsic_value) / final_intrinsic_value) * 100, 2))

    if final_intrinsic_value < current_price:
        print('It is overvalued by ' + str(delta) + '%. Do not enter any positions!')
    elif final_intrinsic_value > current_price:
        print('It is undervalued by ' + str(delta) + '%. Check technicals before investing!')
    elif final_intrinsic_value == current_price:
        print('It is fairly valued. Wait for a better discount!')
    print('')
    print('===================================================================')

    # to print out the automated input if print_info is True

    if print_info == False:
        pass

    else:
        print('')
        if method == 'DCF':
            print('Latest operating cash flow (TTM) = {} {}'.format(currency, ttm_op_cashflow))
        elif method == 'DNI':
            print('Latest net income (TTM) = {} {}'.format(currency, ttm_net_income))

        print('')
        print('Total debt from last quarter = {} {}'.format(currency, round(float(last_quarter_total_debt), 2)))
        print('')
        print('Total cash from last quarter = {} {}'.format(currency, round(float(last_quarter_total_cash), 2)))
        print('')
        print('No. of shares outstanding (millions) = {}'.format(round(shares_outstanding, 2)))
        print('')
        print('Discount rate (%) = {}'.format(round(discount_rate * 100, 2)))


def pb_valuation(ticker):
    """
    This function is used to value bank stocks or companies with negative earnings.

    For companies that are making losses, P/B ratio must be less than 0.50 before investing (risky).
    For banks, P/B ratio of less than 1.00 can be generally considered as undervalued.
    Exception for JPM (largest US bank), it is fairly valued if P/B ratio is around 1.40.

    """
    pb_ratio = yf(ticker).get_key_statistics_data()[ticker]['priceToBook']
    pb_ratio = float('%.2f' % pb_ratio)

    website = str('https://ycharts.com/companies/' + ticker.upper() + '/price_to_book_value')
    pb_ratio_trend = pd.read_html(website)

    min_pb_5_year = float('%.2f' % pb_ratio_trend[3].iloc[0, 1])
    max_pb_5_year = float('%.2f' % pb_ratio_trend[3].iloc[1, 1])
    ave_pb_5_year = float('%.2f' % pb_ratio_trend[3].iloc[2, 1])

    delta = abs(round(((pb_ratio - ave_pb_5_year) / ave_pb_5_year) * 100, 2))

    current_price = yf(ticker).get_current_price()
    currency = yf(ticker).get_currency()

    print('')
    print('The latest close price of {} is at {} {}'.format(ticker, currency, current_price), end='\n\n')
    print('Current P/B ratio is {}, while five-year average P/B ratio is {}'.format(pb_ratio, ave_pb_5_year))

    if pb_ratio < ave_pb_5_year:
        print('It is undervalued by about {}%. Check technicals before investing!'.format(delta), end='\n\n')

    elif pb_ratio > ave_pb_5_year:
        print('It is overvalued by about {}%. Do not enter any positions!'.format(delta), end='\n\n')

    elif pb_ratio == ave_pb_5_year:
        print('It is fairly valued. Wait for a better discount!', end='\n\n')

    print('Five-year minimum and maximum P/B ratio are {} and {} respectively'.format(min_pb_5_year, max_pb_5_year),
          end='\n\n')
    print('Source: {}'.format(website))


def check_projected_growth_rate(modify=False):
    growth_rate_table = r'C:\Users\User\OneDrive\Python_Programming\growth_rate.csv'
    x = pd.read_csv(growth_rate_table, index_col=0)

    if modify is False:  # user only wants to check the table content
        print('')
        print(x)

    elif modify is True:  # user wants to check the content and modify

        n = int(input('How many tickers would you like to modify? '))

        for i in range(n):
            print('')
            ticker = input('Please enter ticker #{} to modify: '.format(i + 1)).upper()
            first_growth_rate = round(float(input('The first five-year growth rate for {} (%): '.format(ticker))), 2)
            second_growth_rate = round(float(input('The next five-year growth rate for {} (%): '.format(ticker))), 2)
            terminal_year = int(input('The terminal year of {}: '.format(ticker)))

            x.loc[ticker, 'First_Growth_%'] = first_growth_rate
            x.loc[ticker, 'Second_Growth_%'] = second_growth_rate
            x.loc[ticker, 'Terminal_Year'] = terminal_year

        x.to_csv(growth_rate_table)
        print('')
        print('=========================================================================')
        print('Growth rate table has been successfully modified! Below is the new table.')
        print('')
        x = pd.read_csv(growth_rate_table, index_col=0)
        print(x)
        print('')
        print('=========================================================================')


def technical_analysis(ticker, start_date='2015-01-01', freq='W'):
    """
    This function is to generate Moving Averages and Full Stochastic of a stock based on a chosen time frame.
    Default frequency is B. Other available frequencies are W, BQ, M, and A.
    """
    window_length = 10
    slowing_period = 3

    stock_data = wb.DataReader(ticker.upper(), 'yahoo', start_date)
    stock_data = stock_data.resample(rule=freq).mean()

    # create exponentially weighted moving averages (EWMA) information
    stock_data['50-EWMA'] = stock_data['Adj Close'].ewm(span=50).mean()
    stock_data['150-EWMA'] = stock_data['Adj Close'].ewm(span=150).mean()
    stock_data['200-EWMA'] = stock_data['Adj Close'].ewm(span=200).mean()

    # to get the minimum price point from specified window length period
    stock_data[('L' + str(window_length))] = stock_data['Low'].rolling(window=window_length).min()
    # to get the maximum price point from specified window length period
    stock_data[('H' + str(window_length))] = stock_data['High'].rolling(window=window_length).max()

    numerator = stock_data['Close'] - stock_data[('L' + str(window_length))]
    denominator = stock_data[('H' + str(window_length))] - stock_data[('L' + str(window_length))]

    stock_data['% K'] = numerator / denominator * 100
    stock_data['Full K'] = stock_data['% K'].ewm(span=slowing_period).mean()
    stock_data.dropna(inplace=True)

    # to plot the chart
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=ratio, gridspec_kw={'height_ratios':[2,1.1]})

    stock_data['Adj Close'].plot(ax=axes[0], c='black', alpha=0.7, label='Close Price of ' + ticker)
    stock_data['50-EWMA'].plot(ax=axes[0], c='blue', ls='--', label='50-Span Exponentially Weighted Moving Average')
    stock_data['150-EWMA'].plot(ax=axes[0], c='green', ls='--', label='150-Span Exponentially Weighted Moving Average')
    stock_data['200-EWMA'].plot(ax=axes[0], c='red', ls='--', label='200-Span Exponentially Weighted Moving Average')
    axes[0].set_title('Close Price and Moving Averages of ' + ticker + ' (' + freq + ')')
    axes[0].legend()

    stock_data['Full K'].plot(ax=axes[1], c='purple', ylim=[0, 100], label='Oscillator Line')
    axes[1].set_title('Full Stochastic Oscillator' + ' (' + freq + ')')
    axes[1].axhline(80, c='red', ls='--', label='Overbought')
    axes[1].axhline(20, c='green', ls='--', label='Oversold')
    axes[1].legend()

    plt.tight_layout()


def rsi_indicator(ticker):
    """
    This function is to generate exponentially-weighted RSI indicator with 14-day window period.
    One argument is required: the ticker of the company
    """
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=250)
    window_length = 10

    stock = wb.DataReader(ticker, 'yahoo', start_date, end_date)['Adj Close']
    # to get the delta of close price every day
    close_delta = stock.diff()[1:]
    # to create a copy from close price delta for data manipulation
    close_up, close_down = close_delta.copy(), close_delta.copy()
    # to zero out those days with down movements (including no movement) in close_up series
    close_up[close_up <= 0] = 0
    # to zero out those days with up movements in close_down series
    close_down[close_down > 0] = 0
    # to calculate EMA of the ups and downs
    roll_up_ewm = close_up.ewm(span=window_length).mean()
    roll_down_ewm = close_down.abs().ewm(span=window_length).mean()
    # to calculate the EWM RSI
    RSI_ewm = 100.00 - (100.00 / (1.00 + roll_up_ewm / roll_down_ewm))

    # to make the RSI plot
    plt.figure(figsize=(22,9), dpi=100)
    plt.ylim(10, 90)
    RSI_ewm[2:].plot(label='RSI EWM')
    plt.axhline(80, c='r')
    plt.axhline(70, c='r', ls='--', lw=0.9, label='Overbought')
    plt.axhline(30, c='g', ls='--', lw=0.9, label='Oversold')
    plt.axhline(20, c='g')
    plt.title(str('EWM RSI Indicator of ' + ticker))
    plt.legend()
