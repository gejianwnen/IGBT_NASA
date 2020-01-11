# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:32:30 2020

@author: gejianwen
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

    

def myHeatMap(harvest,col1,col2,file_name = 'correlationship_usefue_features'):
    farmers = col1
    vegetables = col2
    fig,ax = plt.subplots(figsize = (9,9))
    im = ax.imshow(harvest)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(farmers)))
    ax.set_yticks(np.arange(len(vegetables)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(farmers)
    ax.set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Correlationship")
    fig.tight_layout()
    plt.savefig('temp/'+file_name+'.png')
    plt.show()
    
    
    