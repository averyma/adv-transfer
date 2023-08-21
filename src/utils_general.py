'''
CIFAR10/100: model defintions are from https://github.com/kuangliu/pytorch-cifar/
Imagenet: from torchvision 0.13.1

The configuration for vit on cifar10/100 follows:
https://github.com/kentaroy47/vision-transformers-cifar10
'''

import random
import os
import operator as op
import matplotlib.pyplot as plt
import warnings
import torch, torchvision
import torch.nn as nn
import numpy as np
import torch.optim as optim
from models import PreActResNet18, PreActResNet50, Wide_ResNet, VGG
from vit_pytorch.vit_for_small_dataset import ViT
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

def get_model(args, device=None):

    if args.dataset.startswith('cifar'):
        num_classes = 10 if args.dataset == 'cifar10' else 100
        if args.arch == 'preactresnet18':
            model = PreActResNet18(num_classes)
        elif args.arch == 'preactresnet50':
            model = PreActResNet50(num_classes)
        elif args.arch == 'wrn28':
            model = Wide_ResNet(28, 10, 0.3, num_classes)
        elif args.arch == 'vgg19':
            model = VGG('VGG19', num_classes)
        elif args.arch == 'vit_small':
            model = ViT(
                image_size=32,
                patch_size=4,
                num_classes=num_classes,
                dim=512,
                depth=6,
                heads=8,
                mlp_dim=512,
                dropout=0.1,
                emb_dropout=0.1
                        )
        else:
            raise NotImplementedError("model not included")
    else:
        model = torchvision.models.get_model(args.arch)

    # if args.pretrain:
        # model.load_state_dict(torch.load(args.pretrain, map_location=device))
        # model.to(device)
        # print("\n ***  pretrain model loaded: "+ args.pretrain + " *** \n")

    if device is not None:
        model.to(device)

    return model

def get_optim(model, args):
    """
    recommended setup:
    SGD_step: initial lr:0.1, momentum: 0.9, weight_decay: 0.0002, miliestones: [100, 150]
    Adam_step: initial lr:0.1, milestones: [80,120,160,180]
    others: constant lr at 0.001 should be sufficient
    """
    if args.optim.startswith("sgd"):
        opt = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov)
    elif args.optim == "adamw":
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, AdamW are supported.")

    # check if milestone is an empty array
    if args.lr_scheduler_type == "multistep":
        _milestones = [args.epoch/ 2, args.epoch * 3 / 4]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=_milestones, gamma=0.1)
    elif args.lr_scheduler_type == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch, eta_min=0.)
    elif args.lr_scheduler_type == "fixed":
        lr_scheduler = None
    else:
        raise ValueError('invalid lr_schduler=%s' % args.lr_scheduler_type)

    if args.warmup:
        lr_scheduler = GradualWarmupScheduler(
            opt,
            multiplier=args.warmup_multiplier,
            total_epoch=args.warmup_epoch,
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
