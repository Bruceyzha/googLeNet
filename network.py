'''
this script is for the Task 1 general network of Project 2.

-------------------------------------------
'''


import argparse
import logging
import sys
import time
import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class AddCla(nn.Module):
    def __init__(self, input_filter):
        super(AddCla, self).__init__()
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(5, 3),
            nn.Conv2d(input_filter, 128, 1, 1),
        )
        self.layer2 = nn.Linear(128*4*4, 1024)
        self.layer3 = nn.Linear(1024, 1000)
 
    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer2(x)
        x = self.layer3(x)
        output = F.softmax(x, dim=1)
        return output

class Network(nn.Module):
    def __init__(self, num_class=17):
        super(Network, self).__init__()
        
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer2_3 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.layer4_5 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.layer6_7 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.layer8_9 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.layer10_11 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.layer12_13 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.layer14_15 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.layer16_17 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.layer18_19 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.layer20_21 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.pooling = nn.AvgPool2d(7, 1)
        self.layer22 = nn.Linear(1024, 1000)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.maxpool(x)
        x = self.layer2_3(x)
        x = self.maxpool(x)
        x = self.layer4_5(x)
        x = self.layer6_7(x)
        x = self.maxpool(x)
        x = self.layer8_9(x)
        x = self.layer10_11(x)
        x = self.layer12_13(x)
        x = self.layer14_15(x)
        x = self.layer16_17(x)
        x = self.maxpool(x)
        x = self.layer18_19(x)
        x = self.layer20_21(x)
        x = self.pooling(x)
        x = nn.Dropout(0.4)(x)
        x = x.view(x.size(0), -1)
        x = self.layer22(x)
        output = F.softmax(x, dim=1)
        return output