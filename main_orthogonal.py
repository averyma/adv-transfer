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
from src.evaluation import validate, eval_transfer, eval_transfer_bi_direction, eval_transfer_bi_direction_two_metric, eval_transfer_orthogonal
from src.transfer import model_align, model_align_feature_space
from distiller_zoo import RKDLoss, EGA, PKT, DistillKL, HintLoss, NCELoss, SymmetricKL

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

    if args.debug:
        list_atk_method = ['linbp']
    else:
        # temp_list_atk_method = ['pgd', 'mi', 'ni', 'vni', 'vmi', 'sini', 'ti', 'di']
        temp_list_atk_method = ['linbp']
        list_atk_method = []
        for atk in temp_list_atk_method:
            list_atk_method.append(atk)
            list_atk_method.append(atk+'-strong')

    for _metric in ['pre', 'post', 'diff']:
        for _atk in list_atk_method:
            result[_atk+'/'+_metric] = None

    if not torch.cuda.is_available():
        # print('using CPU, this will be slow')
        print('This should not be run on CPU!!!!!')
        return 0
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.gpu)
        source_model.cuda(args.gpu)
        aligned_model.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.workers = args.ncpus_per_node//max(args.ngpus_per_node, 1)
        print("GPU: {}, batch_size: {}, ncpus_per_node: {}, ngpus_per_node: {}, workers: {}".format(args.gpu, args.batch_size, args.ncpus_per_node, args.ngpus_per_node, args.workers))
        source_model = torch.nn.parallel.DistributedDataParallel(source_model, device_ids=[args.gpu])
        aligned_model = torch.nn.parallel.DistributedDataParallel(aligned_model, device_ids=[args.gpu])
    else:
        torch.cuda.set_device(args.gpu)
        source_model = source_model.cuda(args.gpu)
        aligned_model = aligned_model.cuda(args.gpu)

    is_main_task = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    print('{}: is_main_task: {}'.format(device, is_main_task))

    ckpt_dir = os.path.join(args.j_dir, 'ckpt')
    log_dir = os.path.join(args.j_dir, 'log')
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
    train_loader, test_loader, train_sampler, _ = load_dataset(
                args.dataset,
                args.batch_size,
                args.workers,
                args.distributed,
                )

    if args.dataset == 'imagenet':
        # test_loader_1k contains exactly 1 sample from each of the 1000 class
        test_loader_1k = load_imagenet_test_1k(
                    batch_size=32,
                    workers=0,
                    # workers=args.workers,
                    distributed=args.distributed
                    )
        # test_loader_shuffle is contains the same number of data as the original
        # but data is randomly shuffled, this is for evaluating transfer attack
        test_loader_shuffle, val_sampler = load_imagenet_test_shuffle(
                    batch_size=32,
                    workers=0,
                    # workers=args.workers,
                    distributed=args.distributed
                    )
    else:
        test_loader_1k = test_loader
        test_loader_shuffle = test_loader

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
##########################################################
###################### Training ends #####################
##########################################################

    for (prefix, model) in zip(['pre', 'post'], [source_model, aligned_model]):

        for atk_method in list_atk_method:
            if result[atk_method + '/' + prefix] is None:
                # load target model
                args.arch = args.target_arch
                target_model = get_model(args)
                target_model_dir = os.path.join(
                    root_dir, args.dataset, args.target_arch,
                    model_ckpt[args.dataset][args.target_arch]+str(target_idx), 'model/best_model.pt'
                    )
                print('{}: Load target model from {}.'.format(device, target_model_dir))
                ckpt = torch.load(target_model_dir, map_location=device)
                try:
                    target_model.load_state_dict(ckpt)
                except RuntimeError:
                    target_model.load_state_dict(remove_module(ckpt))
                target_model.cuda(args.gpu)
                if args.distributed:
                    target_model = torch.nn.parallel.DistributedDataParallel(target_model, device_ids=[args.gpu])
                # target model loaded

                if args.distributed:
                    dist.barrier()
                    if args.dataset == 'imagenet':
                        val_sampler.set_epoch(27)
                acc1_source2target = eval_transfer_orthogonal(
                        test_loader_shuffle,
                        model_a=target_model,
                        model_b=model,
                        args=args,
                        atk_method=atk_method,
                        is_main_task=is_main_task)

                _result_source2target = 100.-acc1_source2target
                if args.distributed:
                    dist.barrier()
                if is_main_task:
                    result[atk_method+'/'+prefix] = _result_source2target
                    print(' *  {}: {:.2f}'.format(atk_method+'/'+prefix, _result_source2target))

                    ckpt = { 'result': result}
                    rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                    logger.save_log()

    if args.distributed:
        dist.barrier()

    # Logging and checkpointing only at the main task (rank0)
    if is_main_task:
        for _atk in list_atk_method:
            result['{}/diff'.format(_atk)] = result['{}/post'.format(_atk)] - result['{}/pre'.format(_atk)]

        for key in result.keys():
            num_align_iteration=1
            logger.add_scalar(key, result[key], num_align_iteration)
            logging.info("{}: {:.2f}\t".format(key, result[key]))

    if args.distributed:
        dist.barrier()

    # upload runs to wandb:
    if is_main_task:
        if args.save_modified_model:
            print('Saving final model!')
            saveModel(args.j_dir+"/model/", "final_model", source_model.state_dict())
        if args.enable_wandb:
            save_wandb_retry = 0
            save_wandb_successful = False
            while not save_wandb_successful and save_wandb_retry < 5:
                print('Uploading runs to wandb...')
                try:
                    wandb_logger.upload(logger, num_align_iteration)
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


