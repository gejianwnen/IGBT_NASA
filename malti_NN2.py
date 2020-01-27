# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 09:59:45 2020

@author: gejianwen
"""
import numpy as np
import torch
from torch import nn

class malti_NN2(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim,droprate = 0.5):
        super(malti_NN2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.Dropout(p = droprate),
            nn.ReLU(True))
        self.layer2_1 = nn.Sequential(
            nn.Linear(n_hidden_1, out_dim),
            nn.ReLU(True))
        self.layer2_2 = nn.Sequential(
            nn.Linear(n_hidden_1, out_dim),
            nn.ReLU(True))


    def forward(self, x):
        x = self.layer1(x)
        n = self.layer2_1(x)
        T = self.layer2_2(x)
        return n,T
