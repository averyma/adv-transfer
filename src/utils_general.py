import random
import os
import operator as op
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from models import PreActResNet18, PreActResNet50, Wide_ResNet, resnet50, resnet18, VGG
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import grad
from warmup_scheduler import GradualWarmupScheduler

def seed_everything(manual_seed):
    # set benchmark to False for EXACT reproducibility
    # when benchmark is true, cudnn will run some tests at
    # the beginning which determine which cudnn kernels are
    # optimal for opertions
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model(argu, device=None):

    if argu.dataset == 'cifar100':
        num_classes=100
    elif argu.dataset in ['cifar10', 'svhn', 'mnist', 'fashionmnist', 'imagenette','fmd']:
        num_classes=10
    elif argu.dataset == 'tiny':
        num_classes=200
    elif argu.dataset == 'dtd':
        num_classes=47
    elif argu.dataset in ['caltech', 'food']:
        num_classes=101
    elif argu.dataset == 'flowers':
        num_classes=102
    elif argu.dataset == 'cars':
        num_classes=196
    elif argu.dataset in ['imagenet','dummy']:
        num_classes=1000
    else:
        raise ValueError('dataset unspecified!')

    if argu.arch == 'preactresnet18':
        model = PreActResNet18(argu.dataset, num_classes, argu.input_normalization, argu.enable_batchnorm)
    elif argu.arch == 'preactresnet50':
        model = PreActResNet50(argu.dataset, num_classes, argu.input_normalization, argu.enable_batchnorm)
    elif argu.arch == 'wrn28':
        model = Wide_ResNet(28, 10, 0.3, num_classes, argu.input_normalization)
    elif argu.arch == 'resnet18':
        model = resnet18()
    elif argu.arch == 'resnet50':
        model = resnet50()
    elif argu.arch == 'vgg19':
        model = VGG('VGG19', argu.dataset, num_classes, argu.input_normalization, argu.enable_batchnorm)
    else:
        raise NotImplementedError("model not included")

    # if argu.pretrain:
        # model.load_state_dict(torch.load(argu.pretrain, map_location=device))
        # model.to(device)
        # print("\n ***  pretrain model loaded: "+ argu.pretrain + " *** \n")

    if device is not None:
        model.to(device)

    return model

def get_optim(model, argu):
    """
    recommended setup:
    SGD_step: initial lr:0.1, momentum: 0.9, weight_decay: 0.0002, miliestones: [100, 150]
    Adam_step: initial lr:0.1, milestones: [80,120,160,180]
    others: constant lr at 0.001 should be sufficient
    """
    opt = optim.SGD(
            model.parameters(),
            lr = argu.lr,
            momentum = argu.momentum,
            weight_decay = argu.weight_decay,
            nesterov=argu.nesterov)

    # check if milestone is an empty array
    if argu.lr_scheduler_type == "multistep":
        _milestones = [argu.epoch/ 2, argu.epoch * 3 / 4]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=_milestones, gamma=0.1)
    elif argu.lr_scheduler_type == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=argu.epoch, eta_min=0.)
    elif argu.lr_scheduler_type == "fixed":
        lr_scheduler = None
    else:
        raise ValueError('invalid lr_schduler=%s' % argu.lr_scheduler_type)

    if argu.warmup:
        lr_scheduler = GradualWarmupScheduler(
            opt,
            multiplier=argu.warmup_multiplier,
            total_epoch=argu.warmup_epoch,
            after_scheduler=lr_scheduler
        )

    return opt, lr_scheduler

class DictWrapper(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]

def ep2itr(epoch, loader):
    try:
        data_len = loader.dataset.data.shape[0]
    except AttributeError:
        data_len = loader.dataset.tensors[0].shape[0]
    batch_size = loader.batch_size
    iteration = epoch * np.ceil(data_len/batch_size)
    return iteration
