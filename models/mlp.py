import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class mlp1(nn.Module):
    # def __init__(self, in_dim):
        # super(mlp1, self).__init__()
        # self.in_dim = in_dim[0] * in_dim[1]**2
        # self.h1 = nn.Linear(self.in_dim, 300, bias = True)
        # self.output = nn.Linear(300, 10, bias = True)
        # self.activation = nn.ReLU()
    # def forward(self, x):
        # x = x.view(-1, self.in_dim)
        # h1 = self.activation(self.h1(x))
        # return self.output(h1)

# class mlp2(nn.Module):
    # def __init__(self, in_dim):
        # super(mlp2, self).__init__()
        # self.in_dim = in_dim[0] * in_dim[1]**2
        # self.h1 = nn.Linear(self.in_dim, 400, bias = True)
        # self.h2 = nn.Linear(400, 100, bias = True)
        # self.output = nn.Linear(100, 10, bias = True)
        # self.activation = nn.ReLU()
    # def forward(self, x):
        # x = x.view(-1, self.in_dim)
        # h1 = self.activation(self.h1(x))
        # h2 = self.activation(self.h2(h1))
        # return self.output(h2)

class mlp3(nn.Module):
    def __init__(self):
        super(mlp3, self).__init__()
        self.h1 = nn.Linear(784, 500, bias = True)
        self.h2 = nn.Linear(500, 300, bias = True)
        self.h3 = nn.Linear(300, 100, bias = True)
        self.output = nn.Linear(100, 10, bias = True)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 784)
        h1 = self.activation(self.h1(x))
        h2 = self.activation(self.h2(h1))
        h3 = self.activation(self.h3(h2))
        return self.output(h3)

# class mlp4(nn.Module):
    # def __init__(self, in_dim):
        # super(mlp4, self).__init__()
        # self.in_dim = in_dim[0] * in_dim[1]**2
        # self.h1 = nn.Linear(self.in_dim, 600, bias = True)
        # self.h2 = nn.Linear(600, 300, bias = True)
        # self.h3 = nn.Linear(300, 150, bias = True)
        # self.h4 = nn.Linear(150, 75, bias = True)
        # self.output = nn.Linear(75, 10, bias = True)
        # self.activation = nn.ReLU()
    # def forward(self, x):
        # x = x.view(-1, self.in_dim)
        # h1 = self.activation(self.h1(x))
        # h2 = self.activation(self.h2(h1))
        # h3 = self.activation(self.h3(h2))
        # h4 = self.activation(self.h4(h3))
        # return self.output(h4)

# class mlp5(nn.Module):
    # def __init__(self, in_dim):
        # super(mlp5, self).__init__()
        # self.in_dim = in_dim[0] * in_dim[1]**2
        # self.h1 = nn.Linear(self.in_dim, 600, bias = True)
        # self.h2 = nn.Linear(600, 400, bias = True)
        # self.h3 = nn.Linear(400, 200, bias = True)
        # self.h4 = nn.Linear(200, 100, bias = True)
        # self.h5 = nn.Linear(100, 50, bias = True)
        # self.output = nn.Linear(50, 10, bias = True)
        # self.activation = nn.ReLU()
    # def forward(self, x):
        # x = x.view(-1, self.in_dim)
        # h1 = self.activation(self.h1(x))
        # h2 = self.activation(self.h2(h1))
        # h3 = self.activation(self.h3(h2))
        # h4 = self.activation(self.h4(h3))
        # h5 = self.activation(self.h5(h4))
        # return self.output(h5)
