# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:04:16 2020

@author: gejianwen
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


# 定义 Convolution Network 模型
class CNN1(nn.Module):
    def __init__(self, in_dim, out_dim, n_feature1 = 16, droprate = 0.4):
        super(CNN1, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv1d(in_dim, n_feature1, 128, stride=16, padding=0), # (1024-128)/16+1
            nn.ReLU(True),
            )
        self.fc = nn.Sequential(
            nn.Linear(57*n_feature1*in_dim, out_dim),
            nn.Dropout(p = droprate),
            )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out