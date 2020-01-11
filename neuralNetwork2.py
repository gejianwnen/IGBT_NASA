# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 09:59:45 2020

@author: gejianwen
"""
import numpy as np
import torch
from torch import nn

class neuralNetwork2(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(neuralNetwork2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, out_dim),
            nn.ReLU(True))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
