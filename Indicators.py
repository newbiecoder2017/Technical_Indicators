import pandas as pd
import numpy as np
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override() # <== that's all it takes :-)
from scipy import stats
import matplotlib.pyplot as plt
pd.set_option('precision',4)
pd.options.display.float_format = '{:.3f}'.format


def pull_data(s):
    data_df =  pdr.get_data_yahoo(s, start="2000-11-30", end="2018-05-31")
    return data_df['Open'],data_df['High'], data_df['Low'], data_df['Close'], data_df['Adj Close'], data_df['Volume']

def read_price_file(frq = 'BM', cols=[]):
    df_price = pd.read_csv("C:/Python27/Git/Technical_Indicators/aclose.csv", index_col='Date', parse_dates=True)
    df_price = df_price.resample(frq, closed='right').last()
    df_price = df_price[cols]
    return df_price

def stock_data_csv(universeList):
    open_data = []
    high_data = []
    low_data = []
    close_data = []
    aclose_data = []
    volume_data = []

    for s in universeList:
        #request OHLC data from yahoo for the universe
        print("****************************** " + s + " ************************")
        dfo, dfh, dfl, dfc, df_ac, dfv = pull_data(s)
        open_data.append(dfo)
        high_data.append(dfh)
        low_data.append(dfl)
        close_data.append(dfc)
        aclose_data.append(df_ac)
        volume_data.append(dfv)

    #concat data for the universe
    open_data = pd.concat(open_data, axis = 1)
    high_data = pd.concat(high_data, axis = 1)
    low_data = pd.concat(low_data, axis = 1)
    close_data = pd.concat(close_data, axis = 1)
    aclose_data = pd.concat(aclose_data, axis = 1)
    volume_data = pd.concat(volume_data, axis = 1)

    # rename columns
    open_data.columns = universe_list
    high_data.columns = universe_list
    low_data.columns = universe_list
    close_data.columns = universe_list
    aclose_data.columns = universe_list
    volume_data.columns = universe_list

    #save the dataframes as csv
    open_data.to_csv("C:/Python27/Git/Technical_Indicators/open.csv")
    high_data.to_csv("C:/Python27/Git/Technical_Indicators/high.csv")
    low_data.to_csv("C:/Python27/Git/Technical_Indicators/low.csv")
    close_data.to_csv("C:/Python27/Git/Technical_Indicators/close.csv")
    aclose_data.to_csv("C:/Python27/Git/Technical_Indicators/aclose.csv")
    volume_data.to_csv("C:/Python27/Git/Technical_Indicators/volume.csv")

def RSI(s, window_length):
    ''':param  takes single symbol and calculate sthe RSI'''

    # Window length for moving average
    window_length = window_length
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

def signal_generation(indicator,slope=2,buy_filter = 50.0, sell_filter=70.0):
    trade_signal = []
    is_buy = 0
    is_sell=0
    is_hold = 0
    indicator.iloc[0]=0.0
    slope_df = indicator.diff(slope)
    slope_df.loc[0:2] = -1
    for rows in indicator.iterrows():
        ind_val = rows[1].values[0]
        if ((is_buy==0) & (ind_val<=buy_filter) & (slope_df.loc[rows[0]][0]>=0.0)):
            trade_signal.append(1)
            is_buy= 1
            is_sell=0

        elif ((is_buy==1) & (ind_val>= sell_filter) & (slope_df.loc[rows[0]][0] <0.0)):
            trade_signal.append(-1)
            is_buy=0
            is_sell=1

        elif ((is_buy==1)&(is_sell==0)):
            trade_signal.append(1)
        else:
            trade_signal.append(0)
    signal_frame = pd.DataFrame(trade_signal, index=indicator.index)
    return signal_frame, slope_df


    return indicator
if __name__ == "__main__":

    pers = "W"
    universe_list = ['SPY']
    # #pull historical data
    # stock_data_csv(universe_list)
    # signal_stoch = pd.read_csv("C:/Python27/Examples/S&P 500/stochastic.csv", index_col='Date', parse_dates=True)
    # signal_stoch = signal_stoch.fillna(method='ffill')
    # signal_stoch = signal_stoch.resample(pers).last()
    #
    # signal_macd = pd.read_csv("C:/Python27/Examples/S&P 500/macdsignal.csv", index_col='Date', parse_dates=True)
    # signal_macd = signal_macd.fillna(method='ffill')
    # signal_macd = signal_macd.resample(pers).last()
    #
    # signal_mflow = pd.read_csv("C:/Python27/Examples/S&P 500/moneyflow.csv", index_col='Date', parse_dates=True)
    # signal_mflow = signal_mflow.fillna(method='ffill')
    # signal_mflow = signal_mflow.resample(pers).last()
    #
    # signal_4wr = pd.read_csv("C:/Python27/Examples/S&P 500/relative_4w.csv", index_col='Date', parse_dates=True)
    # signal_4wr = signal_4wr.fillna(method='ffill')
    # signal_4wr = signal_4wr.resample(pers).last()
    # #
    # signal_5dr = pd.read_csv("C:/Python27/Examples/S&P 500/relative_5d.csv", index_col='Date', parse_dates=True)
    # signal_5dr = signal_5dr.fillna(method='ffill')
    # signal_5dr = signal_5dr.resample(pers).last()
    price_df = pd.read_csv("C:/Python27/Git/Technical_Indicators/aclose.csv", index_col='Date',parse_dates=True)
    resample_price = price_df.resample(pers,closed='right').last()
    signal_rsi = RSI(resample_price,4)
    signal_rsi['Signal'],signal_rsi['Slope'] = signal_generation(signal_rsi,slope=1,buy_filter=40.0,sell_filter=90.0)

    signal_rsi['Returns'] = resample_price.pct_change()
    signal_rsi['PortReturn'] = signal_rsi['Signal']*(signal_rsi['Returns'].shift(-1))
    # print(signal_rsi.head(50))
    signal_rsi[['Returns','PortReturn']].cumsum().plot()
    plt.show()
    print(signal_rsi[['Returns', 'PortReturn']].describe())
    print(signal_rsi[['Returns','PortReturn']].groupby(signal_rsi.index.year).mean())

