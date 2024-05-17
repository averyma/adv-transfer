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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

import numpy as np

from src.args import get_args, print_args, get_base_model_dir

from src.utils_dataset import load_dataset, load_imagenet_test_shuffle
from src.utils_log import metaLogger, rotateCheckpoint, wandbLogger, saveModel, delCheckpoint
from src.utils_general import seed_everything, get_model, get_optim, remove_module
from src.transforms import get_mixup_cutmix
import copy
import torch.nn.functional as F
import ipdb
from src.evaluation import validate, eval_transfer, eval_transfer_ensemble
from src.align import align_feature_space, align_feature_space_mixed
from distiller_zoo import RKDLoss, EGA, PKT, DistillKL, HintLoss, NCELoss, SymmetricKL
from models import LinearProjection

import timm

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

    # Set the indices for source, witness, and target models
    if args.debug:
        list_target_arch = ['resnet18']
    else:
        list_target_arch = ['resnet18', 'resnet50', 'resnet101',
                            'densenet121', 'inception_v3', 'vgg19_bn',
                            'swin_t', 'swin_s',
                            'vit_t_16', 'vit_s_16', 'vit_b_16']

    # Load source model
    # ECCV EXPERIMENTS
    ####################################################################################################################
    ####################################################################################################################
    if args.dataset == 'cars':
        if args.source_arch == 'resnet50':
            source_model = timm.create_model("hf_hub:anonauthors/stanford_cars-timm-resnet50", pretrained=True)
            witness_model = timm.create_model("hf_hub:anonauthors/stanford_cars-timm-convnext_base.fb_in1k", pretrained=True)
            list_target_arch = ['vit', 'swin']
        elif args.source_arch == 'convnext':
            source_model = timm.create_model("hf_hub:anonauthors/stanford_cars-timm-convnext_base.fb_in1k", pretrained=True)
            witness_model = timm.create_model("hf_hub:anonauthors/stanford_cars-timm-resnet50", pretrained=True)
            list_target_arch = ['vit', 'swin']
        elif args.source_arch == 'swin':
            source_model = timm.create_model("hf_hub:anonauthors/stanford_cars-timm-swin_base_patch4_window7_224.ms_in22k_ft_in1k", pretrained=True)
            witness_model = timm.create_model("hf_hub:anonauthors/stanford_cars-timm-resnet50", pretrained=True)
            list_target_arch = ['vit', 'convnext']

    elif args.dataset == 'food101':
        if args.source_arch == 'resnet50':
            source_model = timm.create_model("hf_hub:anonauthors/food101-resnet50", pretrained=True)
            witness_model = timm.create_model("hf_hub:anonauthors/food101-timm-convnext_base.fb_in1k", pretrained=True)
            list_target_arch = ['vit', 'swin']
        elif args.source_arch == 'convnext':
            source_model = timm.create_model("hf_hub:anonauthors/food101-timm-convnext_base.fb_in1k", pretrained=True)
            witness_model = timm.create_model("hf_hub:anonauthors/food101-resnet50", pretrained=True)
            list_target_arch = ['vit', 'swin']
        elif args.source_arch == 'swin':
            source_model = timm.create_model("hf_hub:anonauthors/food101-timm-swin_base_patch4_window7_224.ms_in22k_ft_in1k", pretrained=True)
            witness_model = timm.create_model("hf_hub:anonauthors/food101-resnet50", pretrained=True)
            list_target_arch = ['vit', 'convnext']

    list_witness_model = nn.ModuleList([])
    list_witness_model.append(copy.deepcopy(witness_model))
    ###################################################################################################################
    ####################################################################################################################

    # Project source embedding to witness embedding when method is not KL or SymKL
    args.proj_emb = False

    result = {'loss': np.zeros(args.epoch),
              'loss_cls': np.zeros(args.epoch),
              'loss_align': np.zeros(args.epoch),
              'test-err': None,
              'whitebox-err': None,
              'avg-err': 0}
    for target_arch in list_target_arch:
        result[target_arch] = None

    # Sending the model to the device
    if not torch.cuda.is_available():
        print('This should not be run on CPU!!!!!')
        return 0
    elif args.distributed:
        # Compute batch size and workers for distributed training
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.workers = args.ncpus_per_node//max(args.ngpus_per_node, 1)
        print("GPU: {}, batch_size: {}, ncpus_per_node: {},"
              "ngpus_per_node: {}, workers: {}".format(
                  args.gpu, args.batch_size, args.ncpus_per_node,
                  args.ngpus_per_node, args.workers))

        torch.cuda.set_device(args.gpu)
        source_model.cuda(args.gpu)
        source_model = DDP(source_model, device_ids=[args.gpu])

        # orig_source_model.cuda(args.gpu)
        list_witness_model.cuda(args.gpu)
    else:
        torch.cuda.set_device(args.gpu)
        source_model = source_model.cuda(args.gpu)
        # orig_source_model = orig_source_model.cuda(args.gpu)
        list_witness_model = list_witness_model.cuda(args.gpu)

    # Set the main task for the main process
    is_main_task = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)
    print('{}: is_main_task: {}'.format(device, is_main_task))

    # Define the loss function: cls for classification, kd for alignment
    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_kd = DistillKL(args.kl_temp, args.kl_reduction).to(device)

    # Define the optimizer and learning rate scheduler
    list_trainable = nn.ModuleList([])
    list_trainable.append(source_model)

    opt, lr_scheduler = get_optim(list_trainable.parameters(), args)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if is_main_task:
        print('{}: agrs.amp: {}, scaler: {}'.format(device, args.amp, scaler))

    # Load checkpoint if exists
    ckpt_dir = os.path.join(args.j_dir, 'ckpt')
    ckpt_location_curr = os.path.join(ckpt_dir, "ckpt_curr.pth")
    ckpt_location_prev = os.path.join(ckpt_dir, "ckpt_prev.pth")

    # Verify the checkpoint
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

    # Load the checkpoint
    if valid_checkpoint and os.path.exists(load_this_ckpt):
        ckpt = torch.load(load_this_ckpt, map_location=device)
        source_model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        result = ckpt['result']
        ckpt_epoch = ckpt['ckpt_epoch']
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        if scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        print("{}: CHECKPOINT LOADED!".format(device))
        del ckpt
        torch.cuda.empty_cache()
    else:
        # Start from scratch
        ckpt_epoch = 1
        print('{}: NO CHECKPOINT LOADED, FRESH START!'.format(device))

    if args.distributed:
        dist.barrier()

    # Create loggers
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
    train_loader, test_loader, train_sampler, val_sampler = load_dataset(args.dataset,
                                                               args.batch_size,
                                                               args.workers,
                                                               args.distributed)

    print('{}: Dataloader compelete! Ready for alignment!'.format(device))

##########################################################
###################### Training begins ###################
##########################################################
    if args.distributed:
        dist.barrier()

    if not args.eval_only:
        for _epoch in range(ckpt_epoch, args.epoch+1):
            if args.distributed:
                train_sampler.set_epoch(_epoch)
            train_acc1, train_acc5, loss, loss_history = align_feature_space(train_loader,
                                                                             list_trainable,
                                                                             list_witness_model,
                                                                             criterion_kd,
                                                                             criterion_cls,
                                                                             opt,
                                                                             lr_scheduler,
                                                                             scaler,
                                                                             _epoch,
                                                                             device,
                                                                             args,
                                                                             is_main_task)
            # checkpointing for preemption
            if is_main_task:
                result['loss'][_epoch-1] = loss
                result['loss_cls'][_epoch-1] = loss_history[1].mean()
                result['loss_align'][_epoch-1] = loss_history[2].mean()
                ckpt = {"optimizer": opt.state_dict(),
                        "state_dict": source_model.state_dict(),
                        'result': result,
                        'ckpt_epoch': _epoch+1}
                if scaler is not None:
                    ckpt["scaler"] = scaler.state_dict()
                if lr_scheduler is not None:
                    ckpt["lr_scheduler"] = lr_scheduler.state_dict()
                rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                logger.save_log()

            if args.distributed:
                dist.barrier()

    if args.eval_model is not None:
        ckpt = torch.load(args.eval_model, map_location=device)
        try:
            source_model.load_state_dict(ckpt)
        except RuntimeError:
            source_model.load_state_dict(remove_module(ckpt))
        source_model.cuda(args.gpu)
        if args.distributed:
            source_model = DDP(source_model, device_ids=[args.gpu])
        print('{}: No alignment. Evaluating model from {}.'.format(device, args.eval_model))

    del train_loader
    del list_witness_model
    torch.cuda.empty_cache()
##########################################################
###################### Training Ends #####################
###################### Evaluation Begins #################
##########################################################

    if result['test-err'] is None or result['whitebox-err'] is None:
        if args.distributed:
            dist.barrier()
        test_acc1, test_acc5 = validate(test_loader, source_model, criterion_cls, args, is_main_task)
        whitebox_acc1, whitebox_acc5 = validate(test_loader, source_model, criterion_cls,args, is_main_task, whitebox=True)
        test_err1, whitebox_err1 = 100.-test_acc1, 100.-whitebox_acc1
        if args.distributed:
            dist.barrier()
        result['test-err'], result['whitebox-err'] = test_err1, whitebox_err1
        if is_main_task:
            print(' *  {}: {:.2f}\n *  {}: {:.2f}'.format('test-err', test_err1, 'whitebox-err', whitebox_err1))
            ckpt = {"state_dict": source_model.state_dict(), 'result': result, 'ckpt_epoch': args.epoch+1}
            rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
            logger.save_log()

    for target_arch in list_target_arch:
        if result[target_arch] is None:
            # Load target model
            if args.dataset == 'cars':
                if target_arch == 'resnet50':
                    target_model = timm.create_model("hf_hub:anonauthors/stanford_cars-timm-resnet50", pretrained=True)
                elif target_arch == 'convnext':
                    target_model = timm.create_model("hf_hub:anonauthors/stanford_cars-timm-convnext_base.fb_in1k", pretrained=True)
                elif target_arch == 'swin':
                    target_model = timm.create_model("hf_hub:anonauthors/stanford_cars-timm-swin_base_patch4_window7_224.ms_in22k_ft_in1k", pretrained=True)
                elif target_arch == 'vit':
                    target_model = timm.create_model("hf_hub:anonauthors/stanford_cars-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)

            elif args.dataset == 'food101':
                if target_arch == 'resnet50':
                    target_model = timm.create_model("hf_hub:anonauthors/food101-resnet50", pretrained=True)
                elif target_arch == 'convnext':
                    target_model = timm.create_model("hf_hub:anonauthors/food101-timm-convnext_base.fb_in1k", pretrained=True)
                elif target_arch == 'swin':
                    target_model = timm.create_model("hf_hub:anonauthors/food101-timm-swin_base_patch4_window7_224.ms_in22k_ft_in1k", pretrained=True)
                elif target_arch == 'vit':
                    target_model = timm.create_model("hf_hub:anonauthors/food101-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)

            target_model.cuda(args.gpu)
            if args.distributed:
                target_model = DDP(target_model, device_ids=[args.gpu])
                dist.barrier()
                val_sampler.set_epoch(27)

            # Evaluate transferability
            acc1_transfer = eval_transfer(test_loader, source_model, target_model, args, is_main_task)
            err1_transfer = 100.-acc1_transfer
            if args.distributed:
                dist.barrier()
            if is_main_task:
                result[target_arch] = err1_transfer
                print(' *  {}: {:.2f}'.format(target_arch, err1_transfer))

                ckpt = {"state_dict": source_model.state_dict(), 'result': result, 'ckpt_epoch': args.epoch+1}
                rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                logger.save_log()
##########################################################
###################### Evaluation Ends ###################
##########################################################

    if args.distributed:
        dist.barrier()
    # Logging and checkpointing only at the main task (rank0)
    if is_main_task:
        print('result: {}'.format(result))
        for target_arch in list_target_arch:
            result['avg-err'] += result[target_arch]/len(list_target_arch)

        for key in result.keys():
            if 'loss' not in key:
                logger.add_scalar(key, result[key], args.epoch)
                logging.info("{}: {:.2f}\t".format(key, result[key]))
            else:
                num_align_iteration = len(result['loss'])
                for i in range(num_align_iteration):
                    logger.add_scalar(key, result[key][i], i+1)
                    logging.info("{}: {:.2f}\t".format(key, result[key][i]))
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


