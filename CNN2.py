# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:54:50 2020

@author: gejianwen
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


# 定义 Convolution Network 模型
class CNN2(nn.Module):
    def __init__(self, in_dim, out_dim, n_feature1 = 8,n_feature2 = 8, droprate = 0.4):
        super(CNN2, self).__init__()
        self.conv1 = nn.Sequential(
            # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv1d(in_dim, n_feature1, 16, stride=8, padding=0), # (1024-16)/8+1 = 127
            nn.ReLU(True),
            )
        self.conv2 = nn.Sequential(
            # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv1d(n_feature1, n_feature2, 16, stride=8, padding=1), # (127-16)/8+1
            nn.ReLU(True),
            )
        self.fc = nn.Sequential(
            nn.Linear(15*n_feature2*in_dim, out_dim),
            nn.Dropout(p = droprate),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out