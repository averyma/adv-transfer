'''
reference: https://github.com/pytorch/examples/blob/main/imagenet/main.py
'''
import os
import sys
import logging
import shutil
import time
from enum import Enum
import ipdb
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.models
import torch.nn as nn
from torch.utils.data import Subset

import numpy as np

from src.args import get_args, print_args

from src.utils_dataset import load_dataset
from src.utils_log import metaLogger, rotateCheckpoint, wandbLogger, saveModel, delCheckpoint
from src.utils_general import seed_everything, get_model, get_optim, remove_module, set_weight_decay
from src.transforms import get_mixup_cutmix
# import dill as pickle
from src.train import train
from src.evaluation import validate

best_acc1 = 0

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

    if args.debug:
        print('*** DEBUG MODE ***')

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

    global best_acc1
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

    model = get_model(args)

    if not torch.cuda.is_available():
        # print('using CPU, this will be slow')
        print('This should not be run on CPU!!!!!')
        return 0
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = args.ncpus_per_node//max(args.ngpus_per_node, 1)
        # args.workers = 4
        print("GPU: {}, batch_size: {}, ncpus_per_node: {}, ngpus_per_node: {}, workers: {}".format(args.gpu, args.batch_size, args.ncpus_per_node, args.ngpus_per_node, args.workers))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    parameters = set_weight_decay(
        model,
        args.weight_decay,
    )

    is_main_task = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    print('{}: is_main_task: {}'.format(device, is_main_task))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    opt, lr_scheduler = get_optim(parameters, args)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if is_main_task:
        print('{}: agrs.amp: {}, scaler: {}'.format(device, args.amp, scaler))
    ckpt_epoch = 1

    ckpt_dir = os.path.join(args.j_dir, 'ckpt')
    log_dir = os.path.join(args.j_dir, 'log')
    ckpt_location_curr = os.path.join(ckpt_dir, "ckpt_curr.pth")
    ckpt_location_prev = os.path.join(ckpt_dir, "ckpt_prev.pth")

    '''
    when no ckpt saved at the curr dir and resume_from_ckpt is enabled,
    we copy the ckpt and log files from path specified by resume_from_ckpt to the curr dir
    '''
    if is_main_task and args.resume_from_ckpt is not None:
        if not (os.path.exists(ckpt_location_prev) or os.path.exists(ckpt_location_curr)):
            resume_ckpt_dir = os.path.join(args.resume_from_ckpt, 'ckpt')
            print('Resume from a prev ckpt at {}'.format(args.resume_from_ckpt))
            resume_log_dir = os.path.join(args.resume_from_ckpt, 'log')
            print('Also copying log files in {}'.format(resume_log_dir))

            ckpt_prev_curr = os.path.join(resume_ckpt_dir, "ckpt_curr.pth")
            ckpt_prev_prev = os.path.join(resume_ckpt_dir, "ckpt_prev.pth")

            # only copying if there is still ckpt in the path spepcified by resume_from_ckpt
            if os.path.isfile(ckpt_prev_curr) or os.path.isfile(ckpt_prev_prev):

                log_prev_txt = os.path.join(resume_log_dir, "log.txt")
                log_prev_curr = os.path.join(resume_log_dir, "log_curr.pth")
                log_prev_prev = os.path.join(resume_log_dir, "log_prev.pth")

                ckpt_curr_curr = ckpt_location_curr
                ckpt_curr_prev = ckpt_location_prev

                log_curr_txt = os.path.join(log_dir, "log.txt")
                log_curr_curr = os.path.join(log_dir, "log_curr.pth")
                log_curr_prev = os.path.join(log_dir, "log_prev.pth")

                for from_path, to_path in zip(
                        [ckpt_prev_curr, ckpt_prev_prev, log_prev_txt, log_prev_curr, log_prev_prev],
                        [ckpt_curr_curr, ckpt_curr_prev, log_curr_txt, log_curr_curr, log_curr_prev]):
                    if os.path.isfile(from_path):
                        print("copying {} to {}".format(from_path, to_path))
                        cmd = "cp {} {}".format(from_path, to_path)
                        os.system(cmd)
                        if to_path.endswith('.pth'):
                            try:
                                torch.load(to_path)
                            except:
                                print("Corrupted file at {}".format(to_path))
                            else:
                                print("Copied file verified at {}".format(to_path))
            else:
                print('No ckpt found at {}'.format(args.resume_from_ckpt))
        else:
            print('Ckpt already exists at {}. No Resuming.'.format(ckpt_dir))

    if args.distributed:
        dist.barrier()

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
        try:
            model.load_state_dict(ckpt["state_dict"])
        except RuntimeError:
            model.load_state_dict(remove_module(ckpt['state_dict']))
        opt.load_state_dict(ckpt["optimizer"])
        ckpt_epoch = ckpt["epoch"]
        best_acc1 = ckpt['best_acc1']
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        if scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        print("{}: CHECKPOINT LOADED!".format(device))
        del ckpt
        torch.cuda.empty_cache()
    else:
        print('{}: NO CHECKPOINT LOADED, FRESH START!'.format(device))

    if args.distributed:
        dist.barrier()

    actual_trained_epoch = args.epoch

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

    train_loader, test_loader, train_sampler, val_sampler = load_dataset(
                args.dataset,
                args.batch_size,
                args.workers,
                args.distributed,
                args.auto_augment,
                args.ra_magnitude,
                args.interpolation,
                args.ra_sampler,
                args.ra_reps,
                args.random_erase,
                args.augmix_severity
                )

    if args.dataset == 'imagenet':
        num_classes = 1000
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'cifar10':
        num_classes = 10
    mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            num_categories=num_classes,
            use_v2=args.use_v2
            )

    print('{}: Dataloader compelete! Ready for training!'.format(device))
##########################################################
###################### Training begins ###################
##########################################################
    if args.distributed:
        dist.barrier()
    for _epoch in range(ckpt_epoch, args.epoch+1):
        if args.distributed:
            train_sampler.set_epoch(_epoch)

        # train for one epoch
        if args.distributed:
            dist.barrier()
        train_acc1, train_acc5, loss = train(train_loader, model, criterion, opt, _epoch, device, args, is_main_task, scaler, mixup_cutmix)
        if args.distributed:
            dist.barrier()
        test_acc1, test_acc5 = validate(test_loader, model, criterion, args, is_main_task, False)
        if args.distributed:
            dist.barrier()
        lr_scheduler.step()

        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1, best_acc1)

        # Logging and checkpointing only at the main task (rank0)
        if is_main_task:
            logger.add_scalar("train/top1_acc", train_acc1, _epoch)
            logger.add_scalar("train/top5_acc", train_acc5, _epoch)
            logger.add_scalar("train/loss", loss, _epoch)
            logger.add_scalar("lr", opt.param_groups[0]['lr'], _epoch)
            logger.add_scalar("test/top1_acc", test_acc1, _epoch)
            logger.add_scalar("test/top5_acc", test_acc5, _epoch)
            logger.add_scalar("test/best_top1_acc", best_acc1, _epoch)
            logging.info(
                "Epoch: [{0}]\t"
                "lr: {lr:.6f}\t"
                "Train Loss: {loss:.6f}\t"
                "Train Accuracy(top1): {train_acc1:.2f}\t"
                "Train Accuracy(top5): {train_acc5:.2f}\t"
                "Test Accuracy(top1): {test_acc1:.2f}\t"
                "Test Accuracy(top5): {test_acc5:.2f}\t".format(
                    _epoch,
                    lr=opt.param_groups[0]['lr'],
                    loss=loss,
                    train_acc1=train_acc1,
                    train_acc5=train_acc5,
                    test_acc1=test_acc1,
                    test_acc5=test_acc5,
                    ))

            # checkpointing for preemption
            if _epoch % args.ckpt_freq == 0:
                # since preemption would happen in the next epoch, so we want to start from {_epoch+1}
                ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "epoch": _epoch+1,
                        "best_acc1": best_acc1
                        }
                if scaler is not None:
                    ckpt["scaler"] = scaler.state_dict()
                if lr_scheduler is not None:
                    ckpt["lr_scheduler"] = lr_scheduler.state_dict()

                rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                logger.save_log()

            # save best model
            if is_best and _epoch > int(args.epoch*3/4):
                saveModel(args.j_dir+"/model/", "best_model", model.state_dict())

        # Early terminate training when half way thru training and test accuracy still below 20%
        if (np.isnan(loss) or (_epoch > int(args.epoch/2) and test_acc1 < 20)) and not args.debug:
            print('{}: Early stopping at epoch {}.'.format(device, _epoch))
            actual_trained_epoch = _epoch
            saveModel(args.j_dir+"/model/", "final_model", model.state_dict())
            break # break the training for-loop
    if args.distributed:
        dist.barrier()
##########################################################
###################### Training ends #####################
##########################################################

    # load best model
    # try:
        # loc = 'cuda:{}'.format(args.gpu)
        # ckpt_best_model = torch.load(args.j_dir+"/model/best_model.pt", map_location=loc)
    # except:
        # print("Problem loading best_model ckpt at {}/model/best_model.pt!".format(args.j_dir))
        # print("Evaluating using the model from the last epoch!")
    # else:
        # model.load_state_dict(ckpt_best_model)
        # print("LOADED THE BEST CHECKPOINT")

    # upload runs to wandb:
    if is_main_task:
        print('Saving final model!')
        saveModel(args.j_dir+"/model/", "final_model", model.state_dict())
        print('Final model saved!')
        print('Final model trained for {} epochs, test accuracy: {}%'.format(actual_trained_epoch, test_acc1))
        print('Best model has a test accuracy of {}%'.format(best_acc1))
        if args.enable_wandb:
            save_wandb_retry = 0
            save_wandb_successful = False
            while not save_wandb_successful and save_wandb_retry < 5:
                print('Uploading runs to wandb...')
                try:
                    wandb_logger.upload(logger, actual_trained_epoch)
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

    # delete slurm checkpoints
    if is_main_task:
        delCheckpoint(ckpt_dir)

    if args.distributed:
        ddp_cleanup()

if __name__ == "__main__":
    main()
