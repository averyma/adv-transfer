from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

# class DistillKL(nn.Module):
    # """Distilling the Knowledge in a Neural Network"""
    # def __init__(self, T):
        # super(DistillKL, self).__init__()
        # self.T = T

    # def forward(self, y_s, y_t):
        # p_s = F.log_softmax(y_s/self.T, dim=1)
        # p_t = F.softmax(y_t/self.T, dim=1)
        # loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        # return loss

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T, reduction='average'):
        super(DistillKL, self).__init__()
        self.T = T
        self.reduction = reduction
        assert reduction in ['average', 'max', 'min']

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=2)

        loss_per_dim_per_sample_per_model = F.kl_div(p_s, p_t, reduction='none') * (self.T**2)
        loss_per_sample_per_model = loss_per_dim_per_sample_per_model.sum(dim=2)

        if self.reduction == 'average':
            # take the average over models:
            loss_per_sample = loss_per_sample_per_model.mean(dim=0)
        elif self.reduction == 'max':
            # take the max over models:
            loss_per_sample = loss_per_sample_per_model.max(dim=0)[0]
        elif self.reduction == 'min':
            # take the min over models:
            loss_per_sample = loss_per_sample_per_model.min(dim=0)[0]

        loss = loss_per_sample.mean()
        return loss

class SymmetricKL(nn.Module):
    '''This is my own implementation of distillation using kl'''
    def __init__(self):
        super(SymmetricKL, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, p_s, p_w):
        yp_s = F.log_softmax(p_s, dim=1)
        yp_w = F.log_softmax(p_w, dim=1)
        loss = self.kl_loss(yp_s, yp_w) + self.kl_loss(yp_w, yp_s)
        return loss
