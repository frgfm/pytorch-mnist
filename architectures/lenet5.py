#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
PyTorch implementation of LeNet5
'''

__author__ = 'François-Guillaume Fernandez'
__license__ = 'MIT License'
__version__ = '0.1'
__maintainer__ = 'François-Guillaume Fernandez'
__status__ = 'Development'

import torch.nn as nn
from collections import OrderedDict


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class LeNet5(nn.Module):
    """
    [1@28x28] Input
    [6@28x28] CONV1 (5x5), stride 1, pad 2
    [6@14x14] POOL1 (2x2) stride 2
    [16@10x10] CONV2 (5x5), stride 1, pad 0
    [16@5x5] POOL2 (2x2) stride 2
    [120] FC
    [84] FC
    [10] Softmax
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1, padding=2)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('conv2', nn.Conv2d(6, 16, (5, 5), 1, 0)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d((2, 2), 2)),
            ('flatten', Flatten()),
            ('fc3', nn.Linear(in_features=16 * 5 * 5, out_features=120)),
            ('relu3', nn.ReLU()),
            ('fc4', nn.Linear(in_features=120, out_features=84)),
            ('relu4', nn.ReLU()),
            ('fc5', nn.Linear(84, 10)),
            ('prob', nn.LogSoftmax(dim=1))]))

    def forward(self, img):
        output = self.model(img)
        return output

    def name(self):
        return 'LeNet5'
