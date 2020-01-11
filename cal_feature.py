# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:36:13 2020

@author: gejianwen
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt



def cal_feature(df, col):
    '''
    input: dataframe, columns
    output: extract statistic features of every columns
    '''
    x = df[col].values
    
    x_min = df[col].min()
    x_max = df[col].max()
    x_range = x_max-x_min
    x_mean = df[col].mean()
    x_mode = df[col].mode()[0]
    x_median = df[col].median()
    x_quantile25 = df[col].quantile( q=0.25)
    x_quantile75 = df[col].quantile( q=0.75)
    x_std = df[col].std()
    x_var = df[col].var()
    x_skew = df[col].skew()
    x_sem = df[col].sem()                   # standard error of the mean
    x_mad = df[col].mad()                   # the mean absolute deviation of the values
    x_kurt = df[col].kurt()                 # 偏度是三阶中心矩和标准差的三次方的比值 峭度是四阶中心矩和标准差的四次方的比值
    x_msv = np.mean(df[col].pow(2))         # 均方值(mean-square value) 代表了信号的能量 msv
    x_rms = np.sqrt(x_msv)                  # 均方根，均方值的根 root of mean square valve  rmsv
    x_arv = np.mean(df[col].abs())          # 绝对均值(整流平均值arv-Average rectified value，或者叫mean of absolute value) mav
    x_sra = df[col].abs().pow(0.5).mean()**2  # (square root amplitude)   smrv 方根幅值
    x_I = x_max/x_arv                         # 脉冲因子
    x_sf = x_rms/x_arv                        # 波形因子(Form factor&shape factor) 有效值（RMS）与整流平均值的比值
    x_MI = x_range/x_sra                      # 裕度因子 信号峰值与方根幅值的比值
    x_CF = x_range/x_rms                      # 峰值因子(Crest factor),代表峰值在波形中的极端程度。
    x_llr = np.mean(np.log(np.abs(x)+1))/np.log(x_std)     # log-log ratio
    x_pi = x_max/x_mean                       # pulse indicators
    x_sdif = x_std/x_arv                      # SDIF
    x_cpt1 = max(abs(x))/x_sra                #CPT1
    feature_cols = ['x_min', 'x_max', 'x_range', 'x_mean', 'x_mode' ,'x_median' ,'x_quantile25', 'x_quantile75', 'x_std', 'x_var',
                    'x_skew', 'x_sem', 'x_mad', 'x_kurt',  'x_msv', 'x_rms',  'x_arv' ,  'x_sra',   'x_I',    'x_sf',
                    'x_MI',  'x_CF',   'x_llr', 'x_pi',   'x_sdif', 'x_cpt1']
    feature_cols = [s.replace("x_",col+"_") for s in feature_cols]
    feature_list = [x_min, x_max, x_range, x_mean, x_mode ,x_median ,x_quantile25, x_quantile75, x_std, x_var,
                    x_skew, x_sem, x_mad, x_kurt,  x_msv,  x_rms,   x_arv ,   x_sra,    x_I,   x_sf, 
                    x_MI,   x_CF,  x_llr,  x_pi,  x_sdif,  x_cpt1]
    feature_df = pd.DataFrame(data = feature_list).T
    feature_df.columns = feature_cols
    
    return feature_df