# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:35:37 2020

@author: gejianwen
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
    
def plot_feature2(df, base_col, save_dir):
    plt.figure(figsize=(12, 6))
    cols = [c for c in  df.columns if c!=base_col ]
    for col in cols:
        plt.plot(df[base_col].values ,df[col].values)
        plt.title(col,fontsize = 'x-large')
        plt.tight_layout()
        plt.savefig(save_dir+'filtered_'+col+'.png')
        plt.clf()

