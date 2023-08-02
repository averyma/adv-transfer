"""
AM: Aug28, 2019 rename model l1, l2 and l3 as linear1, linear2 and linear 3
    edited main.py to reflect change, for binary_mnist_l2*, i changed their 
    name to binary_mnist_linear2 in both the actual weight file and the record
    in the log.txt file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb

class fc2(nn.Module):
    def __init__(self):
        super(fc2, self).__init__()
        self.l = nn.Linear(784, 1, bias = True)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = x.view(-1, 784)
        return self.sig(self.l(x))

class fc5(nn.Module):
    """
    Implementation of the "2-hidden-layer ReLU network with 1000 hidden units" used in the 
    adversarial sphere paper: https://arxiv.org/abs/1801.02774
    """
    def __init__(self):
        super(fc5, self).__init__()
        self.layer1 = nn.Linear(500, 1000, bias = True)
        self.layer2 = nn.Linear(1000, 1000, bias = True)
        self.readout = nn.Linear(1000, 2, bias = True)
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(1000)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.readout(x)
    
        return x
    
class quad1(nn.Module):
    """
    Implementation of the "quadratic network" used in the 
    adversarial sphere paper: https://arxiv.org/abs/1801.02774
    """
    def __init__(self):
        super(quad1, self).__init__()
        self.layer1 = nn.Linear(500, 1000, bias = False)
        self.readout = nn.Linear(1, 1, bias = True)

    def act(self, x):
        return x**2
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = x.sum(dim = 1, keepdim = True)
        x = self.readout(x)
    
        return x.squeeze()
    
class fc6(nn.Module):
    def __init__(self):
        super(fc6, self).__init__()
        self.l1 = nn.Linear(784, 1000, bias = True)
        self.l2 = nn.Linear(1000, 10, bias = True)

    def forward(self, x):
        x = F.relu(self.l1(x.view(-1, 784)))
        return self.l2(x)
    
class fc7(nn.Module):
    def __init__(self):
        super(fc7, self).__init__()
        self.l1 = nn.Linear(784, 32, bias = False)
        self.l2 = nn.Linear(32, 10, bias = False)

    def forward(self, x):
        x = F.relu(self.l1(x.view(-1, 784)))
        return self.l2(x)
    
class fc8(nn.Module):
    def __init__(self):
        super(fc8, self).__init__()
        self.l1 = nn.Linear(784, 16, bias = False)
        self.l2 = nn.Linear(16, 10, bias = False)

    def forward(self, x):
        x = F.relu(self.l1(x.view(-1, 784)))
        return self.l2(x)
