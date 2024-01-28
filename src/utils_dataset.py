import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.transforms.functional import InterpolationMode
import ipdb
from typing import Any, Callable, List, Optional, Union, Tuple
import os
from PIL import Image
import math, random

from src.utils_augmentation import CustomAugment
from src.sampler import RASampler
from torchvision.transforms.functional import InterpolationMode
# from data.Caltech101.caltech_dataset import Caltech
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Subset

data_dir = '/scratch/ssd001/home/ama/workspace/data/'

def load_dataset(dataset, batch_size=128, workers=4, distributed=False, auto_augment=None, ra_magnitude=9, interpolation='bilinear', ra_sampler=False, ra_reps=3, random_erase_prob=0.0, augmix_severity=3):

    if interpolation == 'bilinear':
        _interpolation = InterpolationMode.BILINEAR
    elif interpolation == 'bicubic':
        _interpolation = InterpolationMode.BICUBIC

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

        transform_list = [transforms.RandomCrop(32, padding=2),
                          transforms.RandomHorizontalFlip()]
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean, std))

        transform_train = transforms.Compose(transform_list)
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
    elif dataset == 'imagenet':
        # mean/std obtained from: https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
        # detail: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/7
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_list = [transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=_interpolation),
                          transforms.RandomHorizontalFlip()]
        if auto_augment is not None:
            if auto_augment == 'ra':
                transform_list.append(transforms.RandAugment(
                    interpolation=_interpolation,
                    magnitude=ra_magnitude))
            elif auto_augment == 'ta_wide':
                transform_list.append(transforms.TrivialAugmentWide(
                    interpolation=_interpolation))
            elif auto_augment == 'augmix':
                transform_list.append(transforms.AugMix(
                    interpolation=_interpolation,
                    severity=augmix_severity))
            else:
                aa_policy = transforms.AutoAugmentPolicy('imagenet')
                transform_list.append(transforms.AutoAugment(
                    policy=aa_policy,
                    interpolation=_interpolation))

        transform_list.extend([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])

        if random_erase_prob > 0:
            transform_list.append(transforms.RandomErasing(p=random_erase_prob))

        transform_train = transforms.Compose(transform_list)

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=_interpolation),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset == 'dummy':
        pass
    else:
        raise ValueError('invalid dataset name=%s' % dataset)
    
    # load dataset
    if dataset == 'cifar10':
        data_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        data_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        data_train = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
        data_test = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)
    elif dataset == 'svhn':
        data_train = datasets.SVHN(data_dir+"SVHN", split='train', download=True, transform=transform_train)
        data_test = datasets.SVHN(data_dir+"SVHN", split='test', download=True, transform=transform_test)
    elif dataset == 'dummy':
        data_train = datasets.FakeData(5000, (3, 224, 224), 1000, transforms.ToTensor())
        data_test = datasets.FakeData(1000, (3, 224, 224), 1000, transforms.ToTensor())
    elif dataset == 'imagenet':
        dataroot = '/scratch/ssd002/datasets/imagenet'
        traindir = os.path.join(dataroot, 'train')
        valdir = os.path.join(dataroot, 'val')
        data_train = datasets.ImageFolder(traindir, transform_train)
        data_test = datasets.ImageFolder(valdir, transform_test)

    if distributed:
        if ra_sampler:
            train_sampler = RASampler(data_train, shuffle=True, repetitions=ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_test, shuffle=False, drop_last=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(data_train)
        val_sampler = torch.utils.data.SequentialSampler(data_test)

    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)

    return train_loader, test_loader, train_sampler, val_sampler

def load_imagenet_test_1k(batch_size=128, workers=4, selection='random', distributed=False):

    # default augmentation
    # mean/std obtained from: https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
    # detail: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/7
    if selection not in ['random', 'fixed']:
        raise ValueError('{} not supported!'.format(selection))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # load dataset
    dataroot = '/scratch/ssd002/datasets/imagenet'
    valdir = os.path.join(dataroot, 'val')
    data_test = datasets.ImageFolder(valdir, transform_test)

    if selection == 'random':
        # randomly sample from the testset
        random.seed(27)
        sample_indices = random.sample(range(len(data_test)), 1000)
    elif selection == 'fixed':
        # select the first sample of every class, ensure every class is present
        # during evaluation
        sample_indices = range(0, len(data_test), 50)

    data_test_1k = Subset(data_test, sample_indices)

    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_test_1k,
                                                                    shuffle=False,
                                                                    drop_last=False)
    else:
        val_sampler = torch.utils.data.SequentialSampler(data_test_1k)

    test_loader = torch.utils.data.DataLoader(
        data_test_1k, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)

    return test_loader, val_sampler


def load_imagenet_test_shuffle(batch_size=128, workers=4, distributed=False):
    '''
    this is the loader used for cvpr results: entire test set
    '''

    # default augmentation
    # mean/std obtained from: https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
    # detail: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/7
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # load dataset
    dataroot = '/scratch/ssd002/datasets/imagenet'
    valdir = os.path.join(dataroot, 'val')
    data_test = datasets.ImageFolder(valdir,transform_test)

    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_test)
    else:
        val_sampler = torch.utils.data.RandomSampler(data_test)

    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=(val_sampler is None),
        num_workers=workers, pin_memory=True, sampler=val_sampler)

    return test_loader, val_sampler
