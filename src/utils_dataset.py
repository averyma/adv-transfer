import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.transforms.functional import InterpolationMode
import ipdb
from typing import Any, Callable, List, Optional, Union, Tuple
import os
from PIL import Image
import math

from src.utils_augmentation import CustomAugment
# from data.Caltech101.caltech_dataset import Caltech
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Subset

data_dir = '/scratch/ssd001/home/ama/workspace/data/'

def load_dataset(dataset, batch_size=128, op_name='Identity', op_prob=1., op_magnitude=9, workers=4, distributed=False):
    
    # default augmentation
    if dataset.startswith('cifar') or dataset == 'svhn':
        if dataset == 'cifar10':
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
        elif dataset == 'cifar100':
            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]
        elif dataset == 'svhn':
            mean = [0.4376821, 0.4437697, 0.47280442]
            std = [0.19803012, 0.20101562, 0.19703614]

        transform_train = transforms.Compose([
            # using 0.75 has a similar effect as pad 4 and randcrop
            # April 4 commented because it seems to cause NaN in training
            # transforms.RandomResizedCrop(32, scale=(0.75, 1.0), interpolation=Image.BICUBIC), 
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
    elif dataset == 'imagenet':
        # mean/std obtained from: https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
        # detail: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/7
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset == 'dummy':
        pass
    elif dataset in ['imagenet-a', 'imagenet-o', 'imagenet-r']:
        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError('invalid dataset name=%s' % dataset)
    
    # apply custom augmentation if op_name is not identity
    if op_name != 'Identity' and dataset in ['cifar10', 'cifar100', 'imagenet']:
        transform_train.transforms.insert(1 if dataset.startswith('cifar') else 2,
                CustomAugment(op_name=op_name, op_prob=op_prob, magnitude=op_magnitude))
        print(transform_train)
    
    # load dataset
    if dataset == 'cifar10':
        data_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        data_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        data_train = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
        data_test = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)
    elif dataset == 'svhn':
        data_train = datasets.SVHN(data_dir+"SVHN", split='train', download = True, transform=transform_train)
        data_test = datasets.SVHN(data_dir+"SVHN", split='test', download = True, transform=transform_test)
    elif dataset == 'dummy':
        data_train = datasets.FakeData(5000, (3, 224, 224), 1000, transforms.ToTensor())
        data_test = datasets.FakeData(1000, (3, 224, 224), 1000, transforms.ToTensor())
    elif dataset == 'imagenet':
        dataroot = '/scratch/ssd002/datasets/imagenet'
        traindir = os.path.join(dataroot, 'train')
        valdir = os.path.join(dataroot, 'val')
        data_train = datasets.ImageFolder(traindir,transform_train)
        data_test = datasets.ImageFolder(valdir,transform_test)
    elif dataset in ['imagenet-a', 'imagenet-o', 'imagenet-r']:
        dataroot = '/scratch/ssd002/datasets/{}/'.format(dataset)
        data_test = datasets.ImageFolder(dataroot,transform_test)

    if distributed:
        if dataset not in ['imagenet-a', 'imagenet-o','imagenet-r']:
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
        else:
            train_sampler = None
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_test, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    if dataset not in ['imagenet-a', 'imagenet-o', 'imagenet-r']:
        train_loader = torch.utils.data.DataLoader(
            data_train, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler)
    else:
        train_loader = None

    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)

    return train_loader, test_loader, train_sampler, val_sampler

def load_IMAGENET_C(batch_size=32, distortion_name='brightness', severity=1, workers=4, distributed=False):
    
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    data_root = '/scratch/ssd002/datasets/imagenet-c/' + distortion_name + '/' + str(severity) + '/'
    
    distorted_dataset = datasets.ImageFolder(
            root=data_root,
            transform=transform)
    
    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_test, shuffle=False, drop_last=True)
    else:
        val_sampler = None

    test_loader = torch.utils.data.DataLoader(
        distorted_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    return test_loader
