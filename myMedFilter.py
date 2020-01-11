# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:36:54 2020

@author: gejianwen
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt



def myMedFilter(s):
    s[2:-2] = np.median([s[0:-4],s[1:-3],s[2:-2],s[3:-1],s[4:]],axis = 0)
    s[1] = np.median(s[:3])    
    s[-2] = np.median(s[-3:])
    
    return s
    
