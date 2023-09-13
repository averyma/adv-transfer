'''
TeacNet and StudNet are from the following paper:
https://github.com/yccm/EGA/blob/main/models/stud_net.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacNet(nn.Module):
    def __init__(self, f_dim, feature_dim):
        # FC layers after feature extraction
        # f_dim is the feature dimension after clip.(depending on which pre-trained clip model is used.) eg.512, 1024.
        # feature_dim is the dimension of joint feature space. eg. 256 or 512.
        super(TeacNet, self).__init__()
        self.fit_dim_Net = nn.Linear(f_dim, feature_dim)
        self.cl_Net = nn.Linear(feature_dim, 1000) 
    def forward(self, x):
        x = self.fit_dim_Net(x)
        ft = x
        x = self.cl_Net(x)
        logit = x        
        return ft, logit

class StudNet(nn.Module):
    def __init__(self, f_dim, feature_dim):
        # FC layers after feature extraction
        # f_dim is the feature dimension after student resnet.(depending on the number of filters of student resnet.)
        # feature_dim is the dimension of joint feature space. eg. 256 or 512.
        super(StudNet, self).__init__()
        self.fit_dim_Net = nn.Linear(f_dim, feature_dim)
        self.cl_Net = nn.Linear(feature_dim, 1000)        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):        
        x = self.fit_dim_Net(x)
        fs = x
        x = self.cl_Net(x)
        logit = x        
        return fs, logit

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProjection, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    def forward(self, input_feature):
        output_feature = self.projection(input_feature)
        return output_feature
