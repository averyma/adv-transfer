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
from src.attacks import pgd
from src.context import ctx_noparamgrad_and_eval
import torch.nn.functional as F
import ipdb
from src.evaluation import validate, eval_transfer
from src.align import align_feature_space
from distiller_zoo import RKDLoss, EGA, PKT, DistillKL, HintLoss, NCELoss, SymmetricKL
from models import LinearProjection


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

    assert args.method in ['kl', 'nce', 'hint', 'pkt', 'ega', 'rkd', 'symkl']

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
        list_target_arch = ['resnet18', 'resnet50']
    else:
        list_target_arch = ['resnet18', 'resnet50', 'resnet101',
                            'densenet121', 'inception_v3', 'vgg19_bn',
                            'swin_t', 'swin_s',
                            'vit_t_16', 'vit_s_16', 'vit_b_16']

    args.source_idx, args.target_idx = 0, 2
    #because i am considering source and witness from different architectures, so having the same idx is okay
    if args.num_witness == 1:
        witness_idx = [0]
    elif args.num_witness == 3:
        witness_idx = [0, 1, 3]

    print('Source model idx: {}\n'
          'Witness model idx: {}\n'
          'Target model idx: {}'.format(
            args.source_idx, witness_idx, args.target_idx))

    base_model_dir = get_base_model_dir(
            '/h/ama/workspace/adv-transfer/options/ckpt_summary.yaml'
            )

    # Load source model
    args.arch = args.source_arch
    source_model = get_model(args)
    source_model_dir = os.path.join(
        base_model_dir['root'], args.dataset, args.source_arch,
        base_model_dir[args.dataset][args.source_arch][args.source_idx],
        'model/best_model.pt')
    ckpt = torch.load(source_model_dir, map_location=device)
    try:
        source_model.load_state_dict(ckpt)
    except RuntimeError:
        source_model.load_state_dict(remove_module(ckpt))
    print('{}: Load source model from {}.'.format(device, source_model_dir))

    # Load witness model
    list_witness_model = nn.ModuleList([])
    for w_idx in witness_idx:
        args.arch = args.witness_arch
        witness_model = get_model(args)
        witness_model_dir = os.path.join(
            base_model_dir['root'], args.dataset, args.witness_arch,
            base_model_dir[args.dataset][args.witness_arch][w_idx],
            'model/best_model.pt')
        ckpt = torch.load(witness_model_dir, map_location=device)
        try:
            witness_model.load_state_dict(ckpt)
        except RuntimeError:
            witness_model.load_state_dict(remove_module(ckpt))
        print('{}: Load witness model from {}.'.format(device, witness_model_dir))
        list_witness_model.append(copy.deepcopy(witness_model))

    # Project source embedding to witness embedding when method is not KL or SymKL
    if args.method not in ['kl', 'symkl'] and args.source_arch != args.witness_arch:
        # Only the combination of resnet50/resnet18 and vit_b_16/vit_t_16 are supported
        if args.source_arch == 'resnet50' and args.witness_arch == 'resnet18':
            args.proj_emb = True
            dim_emb_source = 2048
            dim_emb_witness = 512
        elif args.source_arch == 'vit_b_16' and args.witness_arch == 'vit_t_16':
            args.proj_emb = True
            dim_emb_source = 768
            dim_emb_witness = 192

        if args.proj_emb:
            # Project the source embedding to the witness embedding
            source_projection = LinearProjection(dim_emb_source, dim_emb_witness)
            print('{}: Use linear projection: {}'.format(device, args.proj_emb))
        else:
            raise ValueError('Invalid source and witness model combination!')
    else:
        args.proj_emb = False

    result = {'loss': np.array([None] * args.epoch),
              'loss_cls': np.array([None] * args.epoch),
              'loss_align': np.array([None] * args.epoch),
            #   'test-err': np.array([None] * args.epoch),
            #   'whitebox-err': np.array([None] * args.epoch),
              'avg-err': np.array([0] * args.epoch)}
    for target_arch in list_target_arch:
        result[target_arch] = np.array([None] * args.epoch)

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
        if args.proj_emb:
            source_projection.cuda(args.gpu)
            source_projection = DDP(source_projection, device_ids=[args.gpu])
    else:
        torch.cuda.set_device(args.gpu)
        source_model = source_model.cuda(args.gpu)
        # orig_source_model = orig_source_model.cuda(args.gpu)
        list_witness_model = list_witness_model.cuda(args.gpu)
        if args.proj_emb:
            source_projection = source_projection.cuda(args.gpu)

    # Set the main task for the main process
    is_main_task = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)
    print('{}: is_main_task: {}'.format(device, is_main_task))

    # Define the loss function: cls for classification, kd for alignment
    criterion_cls = nn.CrossEntropyLoss().to(device)
    if args.method == 'rkd':
        criterion_kd = RKDLoss(args.rkd_dist_ratio, args.rkd_angle_ratio).to(device)
    elif args.method == 'ega':
        criterion_kd = EGA(args.ega_node_weight, args.ega_edge_weight).to(device)
    elif args.method == 'pkt':
        criterion_kd = PKT().to(device)
    elif args.method == 'hint':
        criterion_kd = HintLoss(args.hint_weight).to(device)
    elif args.method == 'nce':
        criterion_kd = NCELoss(args.nce_temp).to(device)
    elif args.method == 'kl':
        criterion_kd = DistillKL(args.kl_temp, args.kl_reduction).to(device)
    elif args.method == 'symkl':
        criterion_kd = SymmetricKL().to(device)
    else:
        raise ValueError('Invalid align menthod: {}!'.format(args.method))

    # Define the optimizer and learning rate scheduler
    list_trainable = nn.ModuleList([])
    list_trainable.append(source_model)
    if args.proj_emb:
        list_trainable.append(source_projection)

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
        if args.proj_emb:
            source_projection.load_state_dict(ckpt['projection'])
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
    train_loader, test_loader, train_sampler, _ = load_dataset(args.dataset,
                                                               args.batch_size,
                                                               args.workers,
                                                               args.distributed)

    test_loader_shuffle, val_sampler = load_imagenet_test_shuffle(batch_size=32,
                                                                  workers=0,
                                                                  distributed=args.distributed)

    print('{}: Dataloader compelete! Ready for alignment!'.format(device))

    if is_main_task:
        print('Modifying {} with {} using {}!'.format(args.source_arch, args.witness_arch, args.method))
##########################################################
###################### Training begins ###################
##########################################################
    if args.distributed:
        dist.barrier()

    for _epoch in range(ckpt_epoch, args.epoch+1):
        if args.distributed:
            train_sampler.set_epoch(_epoch)
        if result['loss'][_epoch-1] is not None:
            print('Epoch {} already aligned!'.format(_epoch))
        else:
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
                if args.proj_emb:
                    ckpt['projection'] = source_projection.state_dict()
                rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                logger.save_log()

        if args.distributed:
            dist.barrier()

        for target_arch in list_target_arch:
            if result[target_arch][_epoch-1] is None:
                # Load target model
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
                    target_model = DDP(target_model, device_ids=[args.gpu])
                    dist.barrier()
                    val_sampler.set_epoch(27)

                # Evaluate transferability
                acc1_transfer = eval_transfer(test_loader_shuffle, source_model, target_model, args, is_main_task)

                err1_transfer = 100.-acc1_transfer
                if args.distributed:
                    dist.barrier()
                if is_main_task:
                    result[target_arch][_epoch-1] = err1_transfer
                    print(' *  {}: {:.2f}'.format(target_arch, err1_transfer))

                    ckpt = {"optimizer": opt.state_dict(),
                        "state_dict": source_model.state_dict(),
                        'result': result,
                        'ckpt_epoch': _epoch+1}
                    if scaler is not None:
                        ckpt["scaler"] = scaler.state_dict()
                    if lr_scheduler is not None:
                        ckpt["lr_scheduler"] = lr_scheduler.state_dict()
                    if args.proj_emb:
                        ckpt['projection'] = source_projection.state_dict()
                    rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                    logger.save_log()

                del target_model
                torch.cuda.empty_cache()

    del train_loader
    del list_witness_model
    torch.cuda.empty_cache()
##########################################################
###################### Training Ends #####################
##########################################################

    if args.distributed:
        dist.barrier()
    # Logging and checkpointing only at the main task (rank0)
    if is_main_task:
        print('result: {}'.format(result))
        for _epoch in range(args.epoch):
            for target_arch in list_target_arch:
                result['avg-err'][_epoch] += result[target_arch][_epoch]/len(list_target_arch)

        for key in result.keys():
            # if 'loss' not in key:
            #     logger.add_scalar(key, result[key], args.epoch)
            #     logging.info("{}: {:.2f}\t".format(key, result[key]))
            # else:
            #     num_align_iteration = len(result['loss'])
            for i in range(args.epoch):
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
                    wandb_logger.upload(logger, args.epoch)
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


