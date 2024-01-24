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

from src.args import get_args, print_args, get_base_model_dir

from src.utils_dataset import load_dataset, load_imagenet_test_1k
from src.utils_log import metaLogger, rotateCheckpoint, wandbLogger, saveModel, delCheckpoint
from src.utils_general import seed_everything, get_model, get_optim, remove_module
from src.transforms import get_mixup_cutmix
import copy
from src.attacks import pgd
from src.context import ctx_noparamgrad_and_eval
import torch.nn.functional as F
import ipdb
from src.evaluation import validate, eval_transfer_ensemble, eval_transfer
from src.align import align_feature_space
from distiller_zoo import RKDLoss, EGA, PKT, DistillKL, HintLoss, NCELoss, SymmetricKL


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

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        ddp_setup(args.dist_backend, args.dist_url, args.rank, args.world_size)
        dist.barrier()
    '''
    For Imagenet: available seeds: 40 ~ 42
        source model:  seed 40
        target model:  seed 41
        witness model: seed 42
    For CIFAR10/100: available seeds: 40 ~ 48
        source model:  seed 40
        target model:  seed 41
        witness model: seed 42
    '''
    if args.dataset.startswith('cifar'):
        list_target_arch = ['preactresnet18', 'preactresnet50', 'vgg19', 'vit_small']
    else:
        if args.debug:
            list_target_arch = ['resnet18']
        else:
            list_target_arch = ['resnet18', 'resnet50', 'resnet101',
                                'densenet121', 'inception_v3', 'vgg19_bn',
                                'swin_t', 'swin_s', 'vit_t_16', 'vit_s_16', 'vit_b_16']

    base_model_dir = get_base_model_dir(
            '/h/ama/workspace/adv-transfer/options/ckpt_summary.yaml'
            )

    if args.source_arch == 's1':
        ensemble_dir = ['/h/ama/workspace/adv-transfer/ckpt/aligned/20231027-2gpu-a40-imagenet-kl-S-resnet50-W-resnet18-none-1ep-0.001-seed0/model/final_model.pt']
        ensemble_arch = ['resnet50']
    elif args.source_arch == 's2':
        ensemble_dir = ['/h/ama/workspace/adv-transfer/ckpt/imagenet/resnet50/20230726-imagenet-resnet50-256-40/model/best_model.pt',
                      '/h/ama/workspace/adv-transfer/ckpt/imagenet/resnet18/20230726-imagenet-resnet18-256-40/model/best_model.pt']
        ensemble_arch = ['resnet50', 'resnet18']
    elif args.source_arch == 's3':
        ensemble_dir = ['/h/ama/workspace/adv-transfer/ckpt/aligned/20231027-2gpu-a40-imagenet-kl-S-resnet50-W-resnet18-none-1ep-0.001-seed0/model/final_model.pt',
                      '/h/ama/workspace/adv-transfer/ckpt/aligned/20231027-2gpu-a40-imagenet-kl-S-resnet50-W-resnet18-none-1ep-0.001-seed1/model/final_model.pt']
        ensemble_arch = ['resnet50', 'resnet50']
    elif args.source_arch == 's4':
        ensemble_dir = ['/h/ama/workspace/adv-transfer/ckpt/imagenet/resnet50/20230726-imagenet-resnet50-256-40/model/best_model.pt',
                      '/h/ama/workspace/adv-transfer/ckpt/imagenet/resnet18/20230726-imagenet-resnet18-256-40/model/best_model.pt',
                      '/h/ama/workspace/adv-transfer/ckpt/imagenet/resnet50/20230726-imagenet-resnet50-256-41/model/best_model.pt',
                      '/h/ama/workspace/adv-transfer/ckpt/imagenet/resnet18/20230726-imagenet-resnet18-256-41/model/best_model.pt']
        ensemble_arch = ['resnet50', 'resnet18', 'resnet50', 'resnet18']
    elif args.source_arch == 's5':
        ensemble_dir = ['/h/ama/workspace/adv-transfer/ckpt/aligned/20231027-2gpu-a40-imagenet-kl-S-vit_b_16-W-vit_t_16-none-1ep-0.1-seed0/model/final_model.pt']
        ensemble_arch = ['vit_b_16']
    elif args.source_arch == 's6':
        ensemble_dir = ['/h/ama/workspace/adv-transfer/ckpt/imagenet/vit_b_16/20231010-8gpu-t4v2-imagenet-vit_b_16-1024-40/model/best_model.pt',
                      '/h/ama/workspace/adv-transfer/ckpt/imagenet/vit_t_16/20230929-8gpu-t4v2-imagenet-vit_t_16-1024-40/model/best_model.pt']
        ensemble_arch = ['vit_b_16', 'vit_t_16']
    elif args.source_arch == 's7':
        ensemble_dir = ['/h/ama/workspace/adv-transfer/ckpt/aligned/20231027-2gpu-a40-imagenet-kl-S-vit_b_16-W-vit_t_16-none-1ep-0.1-seed0/model/final_model.pt',
                      '/h/ama/workspace/adv-transfer/ckpt/aligned/20231027-2gpu-a40-imagenet-kl-S-vit_b_16-W-vit_t_16-none-1ep-0.1-seed1/model/final_model.pt']
        ensemble_arch = ['vit_b_16', 'vit_b_16']
    elif args.source_arch == 's8':
        ensemble_dir = ['/h/ama/workspace/adv-transfer/ckpt/imagenet/vit_b_16/20231010-8gpu-t4v2-imagenet-vit_b_16-1024-40/model/best_model.pt',
                      '/h/ama/workspace/adv-transfer/ckpt/imagenet/vit_t_16/20230929-8gpu-t4v2-imagenet-vit_t_16-1024-40/model/best_model.pt',
                      '/h/ama/workspace/adv-transfer/ckpt/imagenet/vit_b_16/20231010-8gpu-t4v2-imagenet-vit_b_16-1024-41/model/best_model.pt',
                      '/h/ama/workspace/adv-transfer/ckpt/imagenet/vit_t_16/20230929-8gpu-t4v2-imagenet-vit_t_16-1024-41/model/best_model.pt']
        ensemble_arch = ['vit_b_16', 'vit_t_16', 'vit_b_16', 'vit_t_16']
    elif args.source_arch == 's9':
        ensemble_dir = ['/h/ama/workspace/adv-transfer/ckpt/imagenet/resnet50/20230726-imagenet-resnet50-256-40/model/best_model.pt']
        ensemble_arch = ['resnet50']
    elif args.source_arch == 's10':
        ensemble_dir = ['/h/ama/workspace/adv-transfer/ckpt/imagenet/resnet18/20230726-imagenet-resnet18-256-40/model/best_model.pt']
        ensemble_arch = ['resnet18']
    elif args.source_arch == 's11':
        ensemble_dir = ['/h/ama/workspace/adv-transfer/ckpt/imagenet/vit_b_16/20231010-8gpu-t4v2-imagenet-vit_b_16-1024-40/model/best_model.pt']
        ensemble_arch = ['vit_b_16']
    elif args.source_arch == 's12':
        ensemble_dir = ['/h/ama/workspace/adv-transfer/ckpt/imagenet/vit_t_16/20230929-8gpu-t4v2-imagenet-vit_t_16-1024-40/model/best_model.pt']
        ensemble_arch = ['vit_t_16']

    ensemble = nn.ModuleList([])
    for _ensemble_arch, _ensemble_dir in zip(ensemble_arch, ensemble_dir):
        args.arch = _ensemble_arch
        source_model = get_model(args)
        ckpt = torch.load(_ensemble_dir, map_location=device)
        try:
            source_model.load_state_dict(ckpt)
        except RuntimeError:
            source_model.load_state_dict(remove_module(ckpt))
        print('{}: Load source model from {}.'.format(device, _ensemble_dir))
        ensemble.append(copy.deepcopy(source_model))

    result = {}
    for target_arch in list_target_arch:
        result[target_arch] = None

    if not torch.cuda.is_available():
        # print('using CPU, this will be slow')
        print('This should not be run on CPU!!!!!')
        return 0
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.gpu)
        ensemble.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.workers = args.ncpus_per_node//max(args.ngpus_per_node, 1)
        print("GPU: {}, batch_size: {}, ncpus_per_node: {}, ngpus_per_node: {}, workers: {}".format(
            args.gpu, args.batch_size, args.ncpus_per_node, args.ngpus_per_node, args.workers))
    else:
        torch.cuda.set_device(args.gpu)
        ensemble = ensemble.cuda(args.gpu)

    is_main_task = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    print('{}: is_main_task: {}'.format(device, is_main_task))

    ckpt_dir = os.path.join(args.j_dir, 'ckpt')
    ckpt_location_curr = os.path.join(ckpt_dir, "ckpt_curr.pth")
    ckpt_location_prev = os.path.join(ckpt_dir, "ckpt_prev.pth")

    valid_checkpoint = False
    for ckpt_location in [ckpt_location_prev, ckpt_location_curr]:
        if os.path.exists(ckpt_location):
            load_ckpt_retry = 0
            load_ckpt_successful = False
            while not load_ckpt_successful and load_ckpt_retry < 5:
                load_ckpt_retry += 1
                print("{}: Checkpoint found at {}".format(device, ckpt_location))
                print("{}: Loading ckpt. Attempt: {}".format(device, load_ckpt_retry))
                try:
                    torch.load(ckpt_location)
                except:
                    print("{}: Corrupted ckpt!".format(device))
                else:
                    print("{}: Checkpoint verified!".format(device))
                    load_ckpt_successful = True
                    valid_checkpoint = True
                    load_this_ckpt = ckpt_location
    if args.distributed:
        dist.barrier()

    ckpt_epoch = 1

    if valid_checkpoint and os.path.exists(load_this_ckpt):
        ckpt = torch.load(load_this_ckpt, map_location=device)
        result = ckpt['result']
        print("{}: CHECKPOINT LOADED!".format(device))
        del ckpt
        torch.cuda.empty_cache()
    else:
        print('{}: NO CHECKPOINT LOADED, FRESH START!'.format(device))

    if args.distributed:
        dist.barrier()

    if is_main_task:
        print('{}: This is the device for the main task!'.format(device))
        # was hanging on wandb init on wandb 0.12.9, fixed after upgrading to 0.15.7
        if args.enable_wandb:
            print('{}: wandb logger created!'.format(device))
            wandb_logger = wandbLogger(args)
        print('{}: local logger created!'.format(device))
        logger = metaLogger(args)
        logging.basicConfig(
            filename=args.j_dir+ "/log/log.txt",
            format='%(asctime)s %(message)s', level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # train_loader and test_loader are the original loader for imagenet
    # train_sampler is necessary for alignment
    # val_sampler is removed so we can use the one from test_loader_shuffle
    # train_loader, test_loader, train_sampler, _ = load_dataset(args.dataset,
                                                               # args.batch_size,
                                                               # args.workers,
                                                               # args.distributed)

    if args.dataset == 'imagenet':
        # test_loader_random_1k contains 1000 randomly selected samples from
        # the test set. The random seed is fixed to 27 to ensure the same random
        # data is used during evaluations.
        test_loader_random_1k, val_sampler = load_imagenet_test_1k(batch_size=32,
                                                                   workers=0,
                                                                   selection='random',
                                                                   distributed=args.distributed)
    else:
        test_loader_random_1k = test_loader

    print('{}: Dataloader compelete! Ready for alignment!'.format(device))
##########################################################
###################### Training begins ###################
##########################################################
##########################################################
###################### Training ends #####################
##########################################################

    for target_arch in list_target_arch:
        if result[target_arch] is None:
            args.arch = target_arch
            target_model = get_model(args)
            target_model_dir = os.path.join(
                base_model_dir['root'], args.dataset, target_arch,
                base_model_dir[args.dataset][target_arch][args.target_idx],
                'model/best_model.pt')
            print('{}: Load target model from {}.'.format(device, target_model_dir))
            ckpt = torch.load(target_model_dir, map_location=device)
            try:
                target_model.load_state_dict(ckpt)
            except RuntimeError:
                target_model.load_state_dict(remove_module(ckpt))
            target_model.cuda(args.gpu)
            if args.distributed:
                target_model = torch.nn.parallel.DistributedDataParallel(target_model,
                                                                         device_ids=[args.gpu])

            if args.distributed:
                dist.barrier()
                if args.dataset == 'imagenet':
                    val_sampler.set_epoch(27)
            acc1_source2target = eval_transfer_ensemble(test_loader_random_1k,
                                                        model_a=target_model,
                                                        ensemble=ensemble,
                                                        args=args,
                                                        is_main_task=is_main_task)

            _result_source2target = 100.-acc1_source2target
            if args.distributed:
                dist.barrier()
            if is_main_task:
                result[target_arch] = _result_source2target
                print(' *  {}: {:.2f}'.format(target_arch, _result_source2target))
                ckpt = {'result': result}
                rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                logger.save_log()

    if args.distributed:
        dist.barrier()

    # Logging and checkpointing only at the main task (rank0)
    if is_main_task:
        for key in result.keys():
            logger.add_scalar(key, result[key], 1)
            logging.info("{}: {:.2f}\t".format(key, result[key]))
    if args.distributed:
        dist.barrier()

    # upload runs to wandb:
    if is_main_task:
        if args.enable_wandb:
            save_wandb_retry = 0
            save_wandb_successful = False
            while not save_wandb_successful and save_wandb_retry < 5:
                print('Uploading runs to wandb...')
                try:
                    wandb_logger.upload(logger, 1)
                except:
                    save_wandb_retry += 1
                    print('Retry {} times'.format(save_wandb_retry))
                else:
                    save_wandb_successful = True

            if not save_wandb_successful:
                print('Failed at uploading runs to wandb.')
            else:
                wandb_logger.finish()

        logger.save_log(is_final_result=True)

    if args.distributed:
        dist.barrier()

    # delete slurm checkpoints
    if is_main_task:
        delCheckpoint(ckpt_dir)

    if args.distributed:
        ddp_cleanup()

if __name__ == "__main__":
    main()
