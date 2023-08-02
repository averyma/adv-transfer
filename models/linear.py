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

class linear_flatten(nn.Module):
    def __init__(self):
        super(linear_flatten, self).__init__()
        self.l = nn.Linear(784, 10, bias = False)
        
    def forward(self, x):
        x = x.view(-1, 784)
        return self.l(x)

class linear_conv(nn.Module):
    def __init__(self, input_dim, output_dim, init='default', bias = False):
        super(linear_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, output_dim, input_dim, 1, bias = bias)
        # if init=='default': 
            # # https://discuss.pytorch.org/t/what-is-the-default-initialization-of-a-conv2d-layer-and-linear-layer/16055
            # stdv = 1./28
            # torch.nn.init.uniform_(self.conv1.weight,a=-stdv, b=stdv)
        # elif init =='std_normal':
            # torch.nn.init.normal_(self.conv1.weight,mean=0.0, std=1.0)
        # elif init =='kaiming_normal':
            # torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
            
    def forward(self, x):
        output = self.conv1(x)[:,:,0,0]
        return output

class LR_model(nn.Module):
    """
    Typically, during initialization, weights are sampled from 
    1. Normal distribution with zero mean and std^2 where std is computed using 1/sqrt(features)
        std is inversely proportional to the dim
        with large nn, we can argue that std is very small, so weights are iniatilized around 0
    2. Uniform{-k, k}, where k = 1/sqrt(input features)
    """
    def __init__(self, dim):
        super(LR_model, self).__init__()
        self.linear = nn.Linear(dim, 1, bias = False)
        torch.nn.init.normal_(self.linear.weight,mean=0.0, std=0.001)
        
        
#         torch.nn.init.normal_(self.linear.weight,mean=0.0, std=1.0)
#         print(self.linear.weight.shape)
#         torch.nn.init.kaiming_normal_(self.linear.weight)
    
#         k = 0.5*torch.ones_like(self.linear.weight.data)
#         print(self.linear.weight.data)
#         self.linear.weight.data = self.linear.weight.data + k
#         print(self.linear.weight.data)
#         self.linear.weight[0].data
#         print(self.linear.weight.data)
        
    def forward(self, x):
        output = self.linear(x.t())
        return output

