import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
# import statsmodels.api as sm

def read_csv_save_hdf():

    try:

        dataframe = pd.read_csv("C:/Python27/Examples/Quandl_Daily_Data/WIKI_PRICES.csv", index_col = 'date', parse_dates=True)

        dataframe[['ticker', 'adj_close']].to_hdf("C:/Python27/Examples/Quandl_Daily_Data/adj_close.h5", 'table')
        dataframe[['ticker', 'open']].to_hdf("C:/Python27/Examples/Quandl_Daily_Data/open.h5", 'table')
        dataframe[['ticker', 'high']].to_hdf("C:/Python27/Examples/Quandl_Daily_Data/high.h5", 'table')
        dataframe[['ticker', 'low']].to_hdf("C:/Python27/Examples/Quandl_Daily_Data/low.h5", 'table')
        dataframe[['ticker', 'volume']].to_hdf("C:/Python27/Examples/Quandl_Daily_Data/volume.h5", 'table')
        dataframe[['ticker', 'ex-dividend']].to_hdf("C:/Python27/Examples/Quandl_Daily_Data/exdividend.h5", 'table')
        dataframe[['ticker', 'adj_open']].to_hdf("C:/Python27/Examples/Quandl_Daily_Data/adj_open.h5", 'table')
        dataframe[['ticker', 'adj_high']].to_hdf("C:/Python27/Examples/Quandl_Daily_Data/adj_high.h5", 'table')
        dataframe[['ticker', 'adj_low']].to_hdf("C:/Python27/Examples/Quandl_Daily_Data/adj_low.h5", 'table')
        dataframe[['ticker', 'close']].to_hdf("C:/Python27/Examples/Quandl_Daily_Data/close.h5", 'table')


    except Exception as e:
        print ("Error occured in the read_csv_save_hdf() method ", e)



def read_adj_prices(px_field):

    adj_price_df = pd.read_hdf("C:/Python27/Examples/Quandl_Daily_Data/"+px_field+".h5", 'table')

    adj_price_df['Temp'] = adj_price_df.index

    idx  = np.arange(0, len(adj_price_df))

    adj_price_df.set_index(idx)

    pivoted_data = adj_price_df.pivot(index='Temp', columns='ticker', values= px_field)

    adj_price_df.set_index(idx)

    pivoted_data.to_hdf("C:/Python27/Examples/Quandl_Daily_Data/clean_"+px_field+".h5", 'table')

    # return pivoted_data

def s_and_p_tickers_check():

    missing_tickers = []
    sp_data = pd.read_csv("C:/Python27/Examples/S&P 500/sp500Stocks.csv")

    aclose= pd.read_hdf("C:/Python27/Examples/Quandl_Daily_Data/clean_adj_close.h5", 'table')
    aopen = pd.read_hdf("C:/Python27/Examples/Quandl_Daily_Data/clean_adj_open.h5", 'table')
    alow = pd.read_hdf("C:/Python27/Examples/Quandl_Daily_Data/clean_adj_low.h5", 'table')
    ahigh = pd.read_hdf("C:/Python27/Examples/Quandl_Daily_Data/clean_adj_high.h5", 'table')
    close = pd.read_hdf("C:/Python27/Examples/Quandl_Daily_Data/clean_close.h5", 'table')
    vol = pd.read_hdf("C:/Python27/Examples/Quandl_Daily_Data/clean_volume.h5", 'table')

    sp_ticker = sp_data.Symbol.tolist()

    for i, v in enumerate(sp_ticker):
        if v == 'BF.B':
            sp_ticker[i] == 'BF_B'
        if v == 'BRK.B':
            sp_ticker[i] = 'BRK_B'

    q_ticker = aclose.columns.tolist()

    for s in sp_ticker:

        if s not in q_ticker:
            missing_tickers.append(s)
    print (missing_tickers)

    if len(missing_tickers) ==0:

        aclose = aclose[sp_ticker].fillna(method = 'ffill')
        aclose.index.names = ['Date']
        aclose.to_csv("C:/Python27/Examples/S&P 500/aclose.csv")

        aopen = aopen[sp_ticker].fillna(method='ffill')
        aopen.index.names = ['Date']
        aopen.to_csv("C:/Python27/Examples/S&P 500/aopen.csv")

        alow = alow[sp_ticker].fillna(method='ffill')
        alow.index.names = ['Date']
        alow.to_csv("C:/Python27/Examples/S&P 500/alow.csv")

        ahigh = ahigh[sp_ticker].fillna(method='ffill')
        ahigh.index.names = ['Date']
        ahigh.to_csv("C:/Python27/Examples/S&P 500/ahigh.csv")

        close = close[sp_ticker].fillna(method='ffill')
        close.index.names = ['Date']
        close.to_csv("C:/Python27/Examples/S&P 500/close.csv")

        vol = vol[sp_ticker].fillna(method='ffill')
        vol.index.names = ['Date']
        vol.to_csv("C:/Python27/Examples/S&P 500/volume.csv")



def RSI(s, pers = 'B'):
    ''':param  takes single symbol and calculate sthe RSI'''

    # Window length for moving average
    window_length = 14
    delta = s.diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = delta[1:]
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(ignore_na=False, min_periods=0, adjust=True, com = window_length).mean()
    roll_down1 = down.abs().ewm(ignore_na=False, min_periods=0, adjust=True, com=window_length).mean()

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    return RSI1

def MACD(s, pers = 'B'):
    try:

        '''method takes in a stock as input and returns the MACD histogram'''
        macd_line = s.ewm(ignore_na=False, span=12, min_periods=0, adjust=True).mean() - s.ewm(ignore_na=False, span=26, min_periods=0, adjust=True).mean()
        sig_line = macd_line.ewm(ignore_na=False, span=9, min_periods=0, adjust=True).mean()
        macd_hist = macd_line - sig_line
        macd_hist = pd.rolling_mean(macd_hist, window = 10)
        return  macd_hist.resample(pers).last()

    except Exception as e:
        print("Error Occured in MACD method", e)



def money_flow(window = 5, pers = 'B'):
    try:
        aclose = pd.read_csv("C:/Python27/Examples/S&P 500/aclose.csv", index_col='Date', parse_dates=True)
        ahigh = pd.read_csv("C:/Python27/Examples/S&P 500/ahigh.csv", index_col='Date', parse_dates=True)
        alow = pd.read_csv("C:/Python27/Examples/S&P 500/alow.csv", index_col='Date', parse_dates=True)
        volume = pd.read_csv("C:/Python27/Examples/S&P 500/volume.csv", index_col='Date', parse_dates=True)

        typical_price = (aclose + ahigh + alow)/3.0
        moneyFlow = typical_price * volume
        avg_mf = moneyFlow.rolling(window).mean()
        mf_sigal = avg_mf/ volume
        return mf_sigal.resample(pers).last()

    except Exception as e:
        print ("Error Occured in money_flow method", e)

def stochastic_oscillator(window = 14, pers = 'B'):
    aclose = pd.read_csv("C:/Python27/Examples/S&P 500/aclose.csv", index_col='Date', parse_dates=True)
    ahigh = pd.read_csv("C:/Python27/Examples/S&P 500/ahigh.csv", index_col='Date', parse_dates=True)
    alow = pd.read_csv("C:/Python27/Examples/S&P 500/alow.csv", index_col='Date', parse_dates=True)
    lowest_low = alow.rolling(window).min()
    highest_high = ahigh.rolling(window).max()
    kwik_line = (aclose - lowest_low) / (highest_high - lowest_low)
    return kwik_line.resample(pers).last()

def industry_relative_returns(window = 5, pers = 'B'):

    sp = pd.read_csv("C:/Python27/Examples/S&P 500/sp500Stocks.csv")
    adj_close = pd.read_csv("C:/Python27/Examples/S&P 500/aclose.csv", index_col= 'Date', parse_dates=True)
    spdata = adj_close[sp.Ticker]
    spdata.index = pd.to_datetime(spdata.index)
    grped = sp.groupby('Industry')
    industry_list = grped.describe().index

    def industry_avg(s):
        temp = spdata[sp[sp.Industry == s]['Ticker']]
        temp = temp.pct_change()
        new_df = temp.sub(temp.mean(axis=1), axis=0)
        return new_df

    frames = [industry_avg(s) for s in industry_list ]
    result = pd.concat(frames, axis = 1)
    result = result.sort_index(axis=1)
    result_5d = result.rolling(5).mean()
    result_4w = result.rolling(20).mean()
    result_5d.to_csv("C:/Python27/Examples/S&P 500/relative_5d.csv")
    result_4w.to_csv("C:/Python27/Examples/S&P 500/relative_4w.csv")


def signal_test(signal_data,returnFrame, counter):
    def mq(x, i):
        # i = args[0]
        lower_bound = x.quantile(i)
        upper_bound = x.quantile(i + 0.1)
        # filtered_signal = np.where(((x>lower_bound) & (x<=upper_bound)), x, np.NAN)
        # filtered_signal = filtered_signal.mean(axis = 1)
        # return filtered_signal
        return np.where(((x > lower_bound) & (x <= upper_bound)), x, np.NAN)

    # signal_data = pd.read_csv("C:/Python27/Examples/S&P 500/macdsignal.csv", index_col = 'Date', parse_dates=True)
    # signal_data = signal_data.fillna(method='ffill')
    # adjClose = pd.read_csv("C:/Python27/Examples/S&P 500/aclose.csv", index_col='Date', parse_dates=True)
    # returnFrame = adjClose.pct_change()
    # returnFrame.fillna(method='ffill',inplace = True)


    filtered_frame = signal_data.apply(mq, i = counter, axis =1)
    filtered_frame = filtered_frame.shift(1)
    temp = returnFrame[~filtered_frame.isnull()]
    return temp.mean(axis = 1)






def calculate_indicators(resamp):
    aclose = pd.read_csv("C:/Python27/Examples/S&P 500/aclose.csv", index_col='Date', parse_dates=True)
    # rsi_frame = pd.DataFrame({s: RSI(aclose[s], resamp) for s in aclose.columns}, index=aclose.resample(resamp).last().index)
    # rsi_frame.to_csv("C:/Python27/Examples/S&P 500/rsi.csv")

    # macd_hist = pd.DataFrame({x: MACD(aclose[x], resamp) for x in aclose.columns}, index=aclose.resample(resamp).last().index)
    # macd_hist.to_csv("C:/Python27/Examples/S&P 500/macdsignal.csv")

    # moneyFlowSignal = money_flow()
    # moneyFlowSignal.to_csv("C:/Python27/Examples/S&P 500/moneyflow.csv")

    # kwikline = stochastic_oscillator()
    # kwikline.to_csv("C:/Python27/Examples/S&P 500/stochastic.csv")
    # print(kwikline)

    # industry_relative_returns()




if __name__ == "__main__":

    pers = "BM"
    # s_and_p_tickers_check()
    #  read_csv_save_hdf()
    # read_adj_prices('adj_close')
    # read_adj_prices('adj_high')
    # read_adj_prices('adj_low')
    # read_adj_prices('adj_open')

    # read_adj_prices('close')
    # read_adj_prices('volume')
    # calculate_indicators("B")

    wt_stoch = 0.05
    wt_macd = -0.10
    wt_rsi = 0.15
    wt_4w = -0.15
    wt_5d = -0.30
    wt_mflow= -0.25


    signal_stoch = pd.read_csv("C:/Python27/Examples/S&P 500/stochastic.csv", index_col='Date', parse_dates=True)
    signal_stoch = signal_stoch.fillna(method='ffill')
    signal_stoch = signal_stoch.resample(pers).last()

    signal_macd = pd.read_csv("C:/Python27/Examples/S&P 500/macdsignal.csv", index_col='Date', parse_dates=True)
    signal_macd = signal_macd.fillna(method='ffill')
    signal_macd = signal_macd.resample(pers).last()

    signal_mflow = pd.read_csv("C:/Python27/Examples/S&P 500/moneyflow.csv", index_col='Date', parse_dates=True)
    signal_mflow = signal_mflow.fillna(method='ffill')
    signal_mflow = signal_mflow.resample(pers).last()

    signal_4wr = pd.read_csv("C:/Python27/Examples/S&P 500/relative_4w.csv", index_col='Date', parse_dates=True)
    signal_4wr = signal_4wr.fillna(method='ffill')
    signal_4wr = signal_4wr.resample(pers).last()
    #
    signal_5dr = pd.read_csv("C:/Python27/Examples/S&P 500/relative_5d.csv", index_col='Date', parse_dates=True)
    signal_5dr = signal_5dr.fillna(method='ffill')
    signal_5dr = signal_5dr.resample(pers).last()
    #
    signal_rsi = pd.read_csv("C:/Python27/Examples/S&P 500/rsi.csv", index_col='Date', parse_dates=True)
    signal_rsi = signal_rsi.fillna(method='ffill')
    signal_rsi = signal_rsi.resample(pers).last()
    signal_data = (wt_stoch * signal_stoch) + (wt_macd * signal_macd) +(wt_mflow * signal_mflow) + (wt_4w * signal_4wr) +(wt_5d * signal_5dr) + (wt_rsi * signal_rsi)
    signal_data.to_csv("C:/Python27/Examples/S&P 500/combo.csv")
    # signal_data = signal_rsi


    signal_data = signal_data['2000':]
    adjClose = pd.read_csv("C:/Python27/Examples/S&P 500/aclose.csv", index_col='Date', parse_dates=True)
    returnFrame = adjClose.pct_change()
    returnFrame.fillna(method='ffill', inplace=True)
    returnFrame = returnFrame.resample(pers).last()
    returnFrame = returnFrame['2000':]
    sig = pd.DataFrame({"Q_"+str(i) : signal_test(signal_data, returnFrame, counter = i) for i in np.arange(0,1,0.1)})

    data = pd.read_csv("C:/Python27/Examples/S&P 500/Return_Data.csv", index_col=['Date'], parse_dates=True)
    data = data.resample(pers).last()
    data = data.pct_change()

    sig['BM'] = data['2000':]
    sig.to_csv("C:/Python27/Examples/S&P 500/bucketReturns.csv")

    print(sig.describe())
    sig.cumsum().plot()
    plt.grid()
    plt.show()

    # delta = sig.subtract(sig['Q_0.9'], axis = 'index' )
    # dm = delta.mean()
    # print(dm)
    # dm.plot(kind = 'bar')
    # plt.show()
