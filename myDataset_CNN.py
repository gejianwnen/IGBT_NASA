# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:51:12 2020

@author: gejianwen
"""
import numpy as np
import torch
from torch import nn

from torch.utils.data import Dataset

class myDataset_CNN(Dataset):
    def __init__(self, x_train, y_train, shape_size = 1024,transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.shape_size = shape_size
        self.transform = transform
        self.size = x_train.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        traindata = self.x_train[idx].view(-1,self.shape_size).unsqueeze_(0)   # use skitimage
        label = self.y_train[idx]

        sample = {'traindata': traindata, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample