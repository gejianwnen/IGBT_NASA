# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:25:15 2020

@author: gejianwen
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def myScatter(map_df,suffix_str):
    # plot it
    cm=plt.cm.inferno  # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    x = map_df['x'].values
    y = map_df['y'].values
    color = map_df['num_cycle'].values
    plt.figure(figsize = (16,7))
    plt.subplot(121)
    sc = plt.scatter(x, y, c=color, vmin=0, vmax=430, s=35, cmap=cm)
    plt.colorbar(sc)

    plt.subplot(122)
    color = map_df['temperature'].values
    sc = plt.scatter(x, y, c=color, vmin=326, vmax=331, s=35, cmap=cm)
    plt.colorbar(sc)

    plt.tight_layout()
    plt.savefig("output/"+"map_"+suffix_str+".png")