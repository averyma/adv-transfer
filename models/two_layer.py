import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb

class two_layer_flatten(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation = "sigmoid", bias=False):
        super(two_layer_flatten, self).__init__()
        self.input_dim = input_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim, bias = bias)
        # torch.nn.init.normal_(self.linear1.weight,mean=0.0, std=1.0)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias = bias)
        # torch.nn.init.normal_(self.linear2.weight,mean=0.0, std=1.0)
        
        if activation in ["sigmoid", "Sigmoid"]:
            self.activation = torch.nn.Sigmoid()
        elif activation in ["Relu", "relu"]:
            self.activation = torch.nn.ReLU()
        elif activation in ["tanh", "Tanh"]:
            self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        output = self.activation(self.linear1(x))
        output = self.linear2(output)

        return output
    
class two_layer_conv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation = "sigmoid", init='default'):
        super(two_layer_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim, input_dim, 1, bias = False)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias = False)
#         torch.nn.init.xavier_uniform(self.conv1.weight)

        if init=='default': # https://discuss.pytorch.org/t/what-is-the-default-initialization-of-a-conv2d-layer-and-linear-layer/16055
            stdv = 1./28
            torch.nn.init.uniform_(self.conv1.weight,a=-stdv, b=stdv)
            stdv = 1. / np.sqrt(output_dim)
            torch.nn.init.uniform_(self.linear2.weight,a=-stdv, b=stdv)
        elif init =='std_normal':
            torch.nn.init.normal_(self.conv1.weight,mean=0.0, std=1.0)
            torch.nn.init.normal_(self.linear2.weight,mean=0.0, std=1.0)
        elif init =='kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
            
        if activation in ["sigmoid", "Sigmoid"]:
            self.activation = torch.nn.Sigmoid()
        elif activation in ["Relu", "relu"]:
            self.activation = torch.nn.ReLU()
        elif activation in ["tanh", "Tanh"]:
            self.activation = torch.nn.Tanh()

    def forward(self, x):
        output = self.activation(self.conv1(x)[:,:,0,0])
#         output = self.linear2(output)
        output = self.linear2(output)
        return output

