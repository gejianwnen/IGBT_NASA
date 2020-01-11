# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:31:20 2020

@author: gejianwen
"""

import numpy as np
import torch
from torch import nn


# Linear Regression Model
class linearRegression(nn.Module):
    def __init__(self, num_input, num_output):
        super(linearRegression, self).__init__()
        self.linear1 = nn.Linear(num_input, num_output)  # input and output is 1 dimension
        self.relu1 = nn.ReLU(True)

    def forward(self, x):
        out = self.relu1(self.linear1(x))
        return out