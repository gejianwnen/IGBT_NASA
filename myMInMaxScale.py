# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:34:41 2020

@author: gejianwen
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn import (manifold, datasets, decomposition, ensemble,
             discriminant_analysis, random_projection)
from sklearn.preprocessing import MinMaxScaler

cols = ['num_cycle' ,
     'up_collectorEmitterVoltage_llr',
     'up_collectorEmitterVoltage_std',
     'down_P_mean',
     'down_collectorEmitterVoltage_max',
     'down_gateEmitterVoltage_kurt',
     'down_P_sf',
     'down_P_max' ,
     'down_P_quantile75' ,
     'down_P_MI',
     'down_gateEmitterVoltage_mad' ,
     'down_P_I',
     'down_gateEmitterVoltage_msv',
     'down_gateEmitterVoltage_quantile75',
     'down_collectorEmitterVoltage_quantile75' ,
     'down_gateEmitterVoltage_sdif',
     'up_gateEmitterVoltage_skew' ,
     'up_collectorEmitterVoltage_sf',
     'down_collectorEmitterVoltage_median' ,
     'down_gateEmitterVoltage_std',
     'down_gateEmitterVoltage_sra',
    ]

if 'temperature' not in cols:
    cols.append('temperature')

# train_df = pd.read_csv('./temp/train_feature_df.csv')
train_df = pd.read_csv('./temp/train_feature_df_outliers.csv')
cols = train_df.columns
feature_df = train_df.loc[:,cols]

# std
# feature_df = pd.read_csv('./temp/useful_feature_df.csv')
target = ["temperature","num_cycle"]
cols_to_use = [c for c in cols if c not in target]
fliter = MinMaxScaler()
for col in cols_to_use:
    feature_df[col] = fliter.fit_transform(feature_df[col].values.reshape(-1, 1))
    
feature_df.to_csv("temp/train_df_scale.csv",index = False)
X = feature_df[cols_to_use].values
feature_df.head()









