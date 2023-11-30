import os
import sys
import logging
import shutil
import time
from enum import Enum
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

import numpy as np

from src.args import get_args, print_args
from src.evaluation import test_clean, test_AA, eval_corrupt, eval_CE, test_gaussian, CORRUPTIONS_IMAGENET_C

from src.utils_dataset import load_dataset, load_imagenet_test_shuffle, load_imagenet_test_1k
from src.utils_log import metaLogger, rotateCheckpoint, wandbLogger, saveModel, delCheckpoint
from src.utils_general import seed_everything, get_model, get_optim, remove_module
from src.transforms import get_mixup_cutmix
import copy
from src.attacks import pgd
from src.context import ctx_noparamgrad_and_eval
import torch.nn.functional as F
import ipdb
# from src.evaluation import validate, eval_transfer, eval_transfer_bi_direction, eval_transfer_bi_direction_two_metric
from src.transfer import model_align, model_align_feature_space
from distiller_zoo import RKDLoss, EGA, PKT, DistillKL, HintLoss, NCELoss, SymmetricKL
from src.evaluation import accuracy


import torch
import torch.nn as nn
from src.attacks import pgd
from src.context import ctx_noparamgrad_and_eval
import ipdb
from tqdm import trange
from autoattack import AutoAttack
import numpy as np
from src.utils_log import Summary, AverageMeter, ProgressMeter
import time
from torch.utils.data import Subset
import torchattacks
from torch.autograd import grad
from src.utils_analysis import measure_smoothness

root_dir = '/scratch/ssd001/home/ama/workspace/adv-transfer/ckpt/'
model_ckpt = {
        'imagenet': {
            'resnet18': '20230726-imagenet-resnet18-256-4',
            'resnet50': '20230726-imagenet-resnet50-256-4',
            'resnet101': '20230928-4gpu-rtx6000,t4v2-imagenet-resnet101-256-4',
            'vgg19_bn': '20230810-4gpu-rtx6000-imagenet-vgg19_bn-256-4',
            'densenet121': '20230928-4gpu-rtx6000,t4v2-imagenet-densenet121-256-4',
            'inception_v3': '20230928-4gpu-rtx6000,t4v2-imagenet-inception_v3-256-4',
            'swin_t': '20230926-8gpu-t4v2-imagenet-swin_t-1024-4',
            'vit_t_16': '20230929-8gpu-t4v2-imagenet-vit_t_16-1024-4',
            'vit_s_16': '20230929-8gpu-t4v2-imagenet-vit_s_16-1024-4',
            'vit_b_16': '20231010-8gpu-t4v2-imagenet-vit_b_16-1024-4',
            }
        }

def ddp_setup(dist_backend, dist_url, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    dist.init_process_group(backend=dist_backend, world_size=world_size,
                            rank=rank, init_method=dist_url)

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

def ddp_cleanup():
    dist.destroy_process_group()

def main():
    args = get_args()

    print_args(args)

    seed_everything(args.seed)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args), start_method='spawn', join=True)
    else:
        # Simply call main_worker function
        args.gpu = 0 if torch.cuda.is_available() else None
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):

    args.ngpus_per_node = ngpus_per_node
    args.ncpus_per_node = len(os.sched_getaffinity(0))
    args.gpu = gpu
    device = torch.device('cuda:{}'.format(args.gpu))

    if args.seed == 0:
        source_idx, witness_idx, target_idx = 0, 1, 2
    elif args.seed == 1:
        source_idx, witness_idx, target_idx = 1, 2, 0
    elif args.seed == 2:
        source_idx, witness_idx, target_idx = 2, 0, 1

    args.arch = args.source_arch
    source_model = get_model(args)
    source_model_dir = os.path.join(
        root_dir, args.dataset, args.source_arch,
        model_ckpt[args.dataset][args.source_arch]+str(source_idx), 'model/best_model.pt'
        )
    ckpt = torch.load(source_model_dir, map_location=device)
    try:
        source_model.load_state_dict(ckpt)
    except RuntimeError:
        source_model.load_state_dict(remove_module(ckpt))
    print('{}: Load source model from {}.'.format(device, source_model_dir))

    args.arch = args.source_arch
    target_model = get_model(args)
    target_model_dir = os.path.join(
        root_dir, args.dataset, args.source_arch,
        model_ckpt[args.dataset][args.source_arch]+str(target_idx), 'model/best_model.pt'
        )
    ckpt = torch.load(target_model_dir, map_location=device)
    try:
        target_model.load_state_dict(ckpt)
    except RuntimeError:
        target_model.load_state_dict(remove_module(ckpt))
    print('{}: Load target model from {}.'.format(device, target_model_dir))

    if args.source_arch == 'resnet50':
        aligned_model_dir = os.path.join(root_dir,
                'aligned',
                '20231027-2gpu-a40-imagenet-kl-S-resnet50-W-resnet18-none-1ep-0.001-seed'+str(source_idx),
                'model/final_model.pt')
    elif args.source_arch == 'vit_b_16':
        aligned_model_dir = os.path.join(root_dir,
                'aligned',
                '20231027-2gpu-a40-imagenet-kl-S-vit_b_16-W-vit_t_16-none-1ep-0.1-seed'+str(source_idx),
                'model/final_model.pt')

    args.arch = args.source_arch
    aligned_model = get_model(args)
    ckpt = torch.load(aligned_model_dir, map_location=device)
    try:
        aligned_model.load_state_dict(ckpt)
    except RuntimeError:
        aligned_model.load_state_dict(remove_module(ckpt))
    print('{}: Load aligned model from {}.'.format(device, aligned_model_dir))

    result = {}

    torch.cuda.set_device(args.gpu)
    source_model = source_model.cuda(args.gpu)
    aligned_model = aligned_model.cuda(args.gpu)
    target_model = target_model.cuda(args.gpu)

    # train_loader and test_loader are the original loader for imagenet
    # train_sampler is necessary for alignment
    # val_sampler is removed so we can use the one from test_loader_shuffle
    train_loader, test_loader, train_sampler, _ = load_dataset(
                args.dataset,
                args.batch_size,
                args.workers,
                args.distributed
                )

    # test_loader_1k contains exactly 1 sample from each of the 1000 class
    test_loader_1k = load_imagenet_test_1k(
                batch_size=32,
                workers=0,
                distributed=args.distributed
                )
    # test_loader_shuffle is contains the same number of data as the original
    # but data is randomly shuffled, this is for evaluating transfer attack
    test_loader_shuffle, val_sampler = load_imagenet_test_shuffle(
                batch_size=32,
                workers=0,
                distributed=args.distributed
                )

    print('{}: len(train_loader): {}\t'
          'len(test_loader): {}\t'
          'len(test_loader_1k): {}\t'
          'len(test_loader_shuffle): {}'.format(
           device,
           len(train_loader)*args.batch_size,
           len(test_loader)*args.batch_size,
           len(test_loader_1k)*(32 if args.dataset == 'imagenet' else args.batch_size),
           len(test_loader_shuffle)*(32 if args.dataset == 'imagenet' else args.batch_size)))

    print('{}: Dataloader compelete! Ready for alignment!'.format(device))

##########################################################
###################### Training begins ###################
##########################################################
    # test_acc1, test_acc5 = validate(test_loader_1k, source_model, criterion_cls, args, is_main_task, False)
    # test_acc1, test_acc5 = validate(test_loader_1k, modified_source_model, criterion_cls, args, is_main_task, False)
    # min_dist_source_model, min_dist_modified_source = compute_min_perturbation(test_loader_1k, source_model, modified_source_model, args, is_main_task)

    s = measure_smoothness(test_loader_1k, source_model, args)
    s = measure_smoothness(test_loader_1k, aligned_model, args)
    # compute_dist2db(test_loader_1k, source_model, aligned_model, target_model, args)
    return 0

##########################################################
###################### Training ends #####################
##########################################################

    # if args.distributed:
        # dist.barrier()

    # # delete slurm checkpoints
    # if is_main_task:
        # delCheckpoint(args.j_dir, args.j_id)

    # if args.distributed:
        # ddp_cleanup()

def validate(val_loader, model, criterion, args, is_main_task, whitebox=False):
    if whitebox:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        atk = torchattacks.PGD(
            model,
            eps=4/255,
            alpha=1/255,
            steps=20,
            random_start=True)
        atk.set_normalization_used(mean=mean, std=std)

    def run_validate(loader, base_progress=0):
        end = time.time()
        for i, (images, target) in enumerate(loader):
            i = base_progress + i
            if args.gpu is not None and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.backends.mps.is_available():
                images = images.to('mps')
                target = target.to('mps')
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            if whitebox:
                with ctx_noparamgrad_and_eval(model):
                    delta = atk(images, target) - images
            else:
                delta = 0

            # compute output
            with torch.no_grad():
                output = model(images+delta)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and is_main_task:
                progress.display(i + 1)
            if args.debug:
                break

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    if is_main_task:
        progress.display_summary()

    return top1.avg, top5.avg

def compute_min_perturbation(val_loader, model_a, model_b, args, is_main_task):
    steps = 1 if args.debug else 10
    atk_a = torchattacks.CW(model_a, steps=steps, lr=0.001)
    atk_b = torchattacks.CW(model_b, steps=steps, lr=0.001)

    atk_a.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    atk_b.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def run_validate_one_iteration(images, target):
        end = time.time()
        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.backends.mps.is_available():
            images = images.to('mps')
            target = target.to('mps')
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.no_grad():
            output_a = model_a(images)
            output_b = model_b(images)

        # first, we idenitfy those classified correctly by the original source model
        correct_a = output_a.argmax(dim=1) == target
        correct_b = output_b.argmax(dim=1) == target
        correct = torch.logical_and(correct_a, correct_b)

        if correct.sum().item() != 0:
            images, target = images[correct], target[correct]
            
            with ctx_noparamgrad_and_eval(model_a):
                adv_images_a = atk_a(images, target)

            # second, we identify successful whitebox attacks on those correctly-classified samples
            with torch.no_grad():
                adv_output_a = model_a(adv_images_a)

            adv_success_a = adv_output_a.argmax(dim=1) != target
            
            if adv_success_a.sum().item() != 0:
                _min_dist_a = torch.norm(
                        (adv_images_a - images)[adv_success_a],
                        p=2,
                        dim=[1, 2, 3])

                with ctx_noparamgrad_and_eval(model_b):
                    adv_images_b = atk_b(images[adv_success_a], target[adv_success_a])

                _min_dist_b = torch.norm(
                        adv_images_b - images[adv_success_a],
                        p=2,
                        dim=[1, 2, 3])

                min_dist_a.update(_min_dist_a.sum().item(), adv_success_a.sum().item())
                min_dist_b.update(_min_dist_b.sum().item(), adv_success_a.sum().item())
                total_eval.update(adv_success_a.sum().item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    min_dist_a = AverageMeter('original/l2_dist', ':6.2f', Summary.AVERAGE)
    min_dist_b = AverageMeter('modified/l2_dist', ':6.2f', Summary.AVERAGE)
    total_eval = AverageMeter('Evaluated', ':6.2f', Summary.SUM)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, min_dist_a, min_dist_b, total_eval],
        prefix='Using {}: '.format(args.method))

    # switch to evaluate mode
    model_a.eval()
    model_b.eval()

    for i, (images, target) in enumerate(val_loader):
        run_validate_one_iteration(images, target)

        if (i % args.print_freq == 0 and is_main_task) or args.debug:
            progress.display(i + 1)

        if args.distributed:
            total_eval.all_reduce()

        if total_eval.sum > 500:
            break

    if args.distributed:
        min_dist_a.all_reduce()
        min_dist_b.all_reduce()

    if is_main_task:
        progress.display_summary()

    return min_dist_a.avg, min_dist_b.avg

def compute_dist2db(val_loader, model_a, model_b, model_c, args):
    '''
    model_a: original model
    model_b: modified model(source)
    model_c: target model
    setting: first, find samples that can be both correctly classified by model a and model c
    generate adv on model a, find those which:
        1. can misclassfy on a
        2. cannot transfer to c
    compute their distance to db of model c using CW
    Finally, on the same samples, generate adv using b, compute their distance to c
    '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    pgd_a = torchattacks.PGD(
        model_a,
        eps=args.pgd_eps,
        alpha=args.pgd_alpha,
        steps=1 if args.debug else args.pgd_itr,
        random_start=True)
    pgd_a.set_normalization_used(mean=mean, std=std)

    pgd_b = torchattacks.PGD(
        model_b,
        eps=args.pgd_eps,
        alpha=args.pgd_alpha,
        steps=1 if args.debug else args.pgd_itr,
        random_start=True)
    pgd_b.set_normalization_used(mean=mean, std=std)

    steps = 1 if args.debug else 10
    cw_a = torchattacks.CW(model_a, steps=steps, lr=0.001)
    cw_b = torchattacks.CW(model_b, steps=steps, lr=0.001)

    cw_a.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cw_b.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    num_eval = 1000 if args.dataset == 'imagenet' else 10000
    num_eval = 100 if args.debug else num_eval

    def run_validate_one_iteration(images, target):
        end = time.time()
        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.backends.mps.is_available():
            images = images.to('mps')
            target = target.to('mps')
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            p_a = model_a(images)
            p_b = model_b(images)

        both_correct_pred_on_clean = return_correct_on_a_b(p_a, p_b, target)
        if both_correct_pred_on_clean.sum().item() != 0:
            images = images[both_correct_pred_on_clean, ::]
            target = target[both_correct_pred_on_clean]

            with ctx_noparamgrad_and_eval(model_a):
                delta_a = pgd_a(images, target) - images
            # compute output
            with torch.no_grad():
                p_adv_a = model_a(images+delta_a)
                p_adv_c = model_c(images+delta_a)
                qualified = return_misclassify_on_a_and_cannot_transfer_to_c(
                        p_adv_a, p_adv_c, target)
            
            num_qualified=qualified.sum().item()
            if num_qualified != 0:
                images = images[qualified, ::]
                target = target[qualified]
                 
                with ctx_noparamgrad_and_eval(model_a):
                    delta_cw_a = cw_a(images, target) - images
                _min_dist_a = torch.norm(
                        delta_cw_a, p=2, dim=[1,2,3]
                        )

                with ctx_noparamgrad_and_eval(model_b):
                    delta_cw_b = cw_b(images, target) - images
                _min_dist_b = torch.norm(
                        delta_cw_b, p=2, dim=[1,2,3]
                        )

                min_dist_a.update(_min_dist_a.sum().item(), num_qualified)
                min_dist_b.update(_min_dist_b.sum().item(), num_qualified)
                total_qualified.update(num_qualified)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    min_dist_a = AverageMeter('original/l2_dist-2-c', ':6.2f', Summary.AVERAGE)
    min_dist_b = AverageMeter('source/l2_dist-2-c', ':6.2f', Summary.AVERAGE)
    total_qualified = AverageMeter('Evaluated', ':6.2f', Summary.SUM)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, min_dist_a, min_dist_b, total_qualified],
        prefix='Transfer: ')

    # switch to evaluate mode
    model_a.eval()
    model_b.eval()

    for i, (images, target) in enumerate(val_loader):
        run_validate_one_iteration(images, target)

        progress.display(i + 1)

        if args.distributed:
            total_qualified.all_reduce()

        if total_qualified.sum > (num_eval/args.ngpus_per_node):
            break

    if args.distributed:
        min_dist_a.all_reduce()
        min_dist_b.all_reduce()

    progress.display_summary()

    return min_dist_a.avg, min_dist_b.avg

def return_correct_on_a_b(p_0, p_1, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        _, pred_0 = p_0.topk(1, 1, True, True)
        _, pred_1 = p_1.topk(1, 1, True, True)

        pred_0 = pred_0.t()
        pred_1 = pred_1.t()

        correct_0 = pred_0.eq(target.view(1, -1).expand_as(pred_0)).squeeze()
        correct_1 = pred_1.eq(target.view(1, -1).expand_as(pred_0)).squeeze()
        qualified = correct_0.eq(correct_1)

        return qualified

def return_misclassify_on_a_and_cannot_transfer_to_c(p_0, p_1, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        _, pred_0 = p_0.topk(1, 1, True, True)
        _, pred_1 = p_1.topk(1, 1, True, True)

        pred_0 = pred_0.t()
        pred_1 = pred_1.t()

        incorrect_0 = pred_0.ne(target.view(1, -1).expand_as(pred_0)).squeeze()
        correct_1 = pred_1.eq(target.view(1, -1).expand_as(pred_0)).squeeze()
        qualified = incorrect_0.eq(correct_1)

        return qualified

# def return_correct_on_a_b(p_0, p_1, p_adv_0, p_adv_1, target):
    # """Computes the accuracy over the k top predictions for the specified values of k"""
    # with torch.no_grad():
        # _, pred_0 = p_0.topk(1, 1, True, True)
        # _, pred_1 = p_1.topk(1, 1, True, True)
        # _, pred_adv_0 = p_adv_0.topk(1, 1, True, True)
        # _, pred_adv_1 = p_adv_1.topk(1, 1, True, True)

        # pred_0 = pred_0.t()
        # pred_1 = pred_1.t()
        # pred_adv_0 = pred_adv_0.t()
        # pred_adv_1 = pred_adv_1.t()

        # correct_0 = pred_0.eq(target.view(1, -1).expand_as(pred_0)).squeeze()
        # correct_1 = pred_1.eq(target.view(1, -1).expand_as(pred_0)).squeeze()
        # incorrect_0 = pred_adv_0.ne(target.view(1, -1).expand_as(pred_0)).squeeze()
        # incorrect_1 = pred_adv_1.ne(target.view(1, -1).expand_as(pred_0)).squeeze()
        # qualified = correct_0.eq(correct_1).eq(incorrect_0).eq(incorrect_1)

        # return qualified

if __name__ == "__main__":
    main()


