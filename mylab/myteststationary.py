# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
__all__ = ["draw_trend","draw_ts","test_stationarity","draw_acf_pacf"]


from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# 移动平均图
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
#    rol_weighted_mean = pd.DataFrame.ewm(timeSeries,span=size).mean()
    rolstd = timeSeries.rolling( window=size).std()

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rolstd.plot(color='black', label='Rolling Standard')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard')
    plt.show()

def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()

'''
　　Unit Root Test
   The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
   root, with the alternative that there is no unit root. That is to say the
   bigger the p-value the more reason we assert that there is a unit root
'''
def test_stationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white',figsize = (10,3))
    ax1 = f.add_subplot(121)
    plot_acf(ts, lags=lags, ax=ax1)
    ax2 = f.add_subplot(122)
    plot_pacf(ts, lags=lags, ax=ax2)
    plt.show()