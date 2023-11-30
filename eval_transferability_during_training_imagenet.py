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
from src.evaluation import test_clean, test_AA, eval_corrupt, eval_CE, test_gaussian, CORRUPTIONS_IMAGENET_C

from src.utils_dataset import load_dataset, load_imagenet_test_shuffle, load_imagenet_test_1k
from src.utils_log import metaLogger, rotateCheckpoint, wandbLogger, saveModel, delCheckpoint
from src.utils_general import seed_everything, get_model, get_optim
from src.transforms import get_mixup_cutmix
# import dill as pickle
from src.evaluation import validate, eval_transfer_bi_direction
from src.train import train

best_acc1 = 0
root_dir = '/scratch/hdd001/home/ama/improve-transferability/'
model_ckpt = {
        'imagenet': {
            'resnet18': '2023-07-26/20230726-imagenet-resnet18-256-4',
            'resnet50': '2023-07-26/20230726-imagenet-resnet50-256-4',
            'vgg19_bn': '2023-08-10/20230810-4gpu-rtx6000-imagenet-vgg19_bn-256-4',
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
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):

    global best_acc1
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

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
    '''
    load models for evaluation
    # source model:  seed 41
    # target model:  seed 41
    '''
    args.arch = 'resnet18'
    model_resnet18 = get_model(args)
    model_dir = root_dir + model_ckpt[args.dataset][args.arch] + '1/model/best_model.pt'
    loc = 'cuda:{}'.format(args.gpu)
    ckpt = remove_module(torch.load(model_dir, map_location=loc))
    model_resnet18.load_state_dict(ckpt)
    print('Load resnet18 from {} to gpu{}'.format(model_dir, args.gpu))

    args.arch = 'resnet50'
    model_resnet50 = get_model(args)
    model_dir = root_dir + model_ckpt[args.dataset][args.arch] + '1/model/best_model.pt'
    loc = 'cuda:{}'.format(args.gpu)
    ckpt = remove_module(torch.load(model_dir, map_location=loc))
    model_resnet50.load_state_dict(ckpt)
    print('Load resnet50 from {} to gpu{}'.format(model_dir, args.gpu))

    args.arch = 'vgg19_bn'
    model_vgg19_bn = get_model(args)
    model_dir = root_dir + model_ckpt[args.dataset][args.arch] + '1/model/best_model.pt'
    loc = 'cuda:{}'.format(args.gpu)
    ckpt = remove_module(torch.load(model_dir, map_location=loc))
    model_vgg19_bn.load_state_dict(ckpt)
    print('Load vgg19_bn from {} to gpu{}'.format(model_dir, args.gpu))

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                model_resnet18.cuda(args.gpu)
                model_resnet50.cuda(args.gpu)
                model_vgg19_bn.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                # args.workers = mp.cpu_count()//max(ngpus_per_node, 1)
                args.workers = 4
                print("GPU: {}, batch_size: {}, workers: {}".format(args.gpu, args.batch_size, args.workers))
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                model_resnet18 = torch.nn.parallel.DistributedDataParallel(model_resnet18, device_ids=[args.gpu])
                model_resnet50 = torch.nn.parallel.DistributedDataParallel(model_resnet50, device_ids=[args.gpu])
                model_vgg19_bn = torch.nn.parallel.DistributedDataParallel(model_vgg19_bn, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print('device: {}'.format(device))

    is_main_task = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    opt, lr_scheduler = get_optim(model, args)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    print('agrs.amp: {}, scaler: {}'.format(args.amp, scaler))
    ckpt_epoch = 1

    ckpt_dir = args.j_dir+"/"+str(args.j_id)+"/"
    ckpt_location_curr = os.path.join(ckpt_dir, "ckpt_curr.pth")
    ckpt_location_prev = os.path.join(ckpt_dir, "ckpt_prev.pth")

    '''
    when no ckpt saved at the curr dir and resume_from_ckpt is enabled,
    we copy the ckpt and log files from path specified by resume_from_ckpt to the curr dir
    '''
    if is_main_task and args.resume_from_ckpt is not None:
        if not (os.path.exists(ckpt_location_prev) or os.path.exists(ckpt_location_curr)):
            print('Resume from a prev ckpt at {}'.format(args.resume_from_ckpt))

            log_dir = args.resume_from_ckpt[:-1] if args.resume_from_ckpt[-1] == '/' else args.resume_from_ckpt
            while log_dir[-1] != '/':
                log_dir = log_dir[:-1]
            log_dir += 'log/'
            print('Also copying log files in {}'.format(log_dir))

            ckpt_prev_curr = os.path.join(args.resume_from_ckpt, "ckpt_curr.pth")
            ckpt_prev_prev = os.path.join(args.resume_from_ckpt, "ckpt_prev.pth")

            # only copying if there is still ckpt in the path spepcified by resume_from_ckpt
            if os.path.isfile(ckpt_prev_curr) or os.path.isfile(ckpt_prev_prev):

                log_prev_txt = os.path.join(log_dir, "log.txt")
                log_prev_curr = os.path.join(log_dir, "log_curr.pth")
                log_prev_prev = os.path.join(log_dir, "log_prev.pth")

                ckpt_curr_curr = os.path.join(args.j_dir, str(args.j_id), "ckpt_curr.pth")
                ckpt_curr_prev = os.path.join(args.j_dir, str(args.j_id), "ckpt_prev.pth")

                log_curr_txt = os.path.join(args.j_dir, "log", "log.txt")
                log_curr_curr = os.path.join(args.j_dir, "log", "log_curr.pth")
                log_curr_prev = os.path.join(args.j_dir, "log", "log_prev.pth")

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
    dist.barrier()

    valid_checkpoint = False
    for ckpt_location in [ckpt_location_prev, ckpt_location_curr]:
        if os.path.exists(ckpt_location):
            load_ckpt_retry = 0
            load_ckpt_successful = False
            while not load_ckpt_successful and load_ckpt_retry < 5:
                load_ckpt_retry += 1
                print("Checkpoint found at {}\n"
                      "Loading ckpt to device {}\n"
                      "Attempt: {}".format(ckpt_location, device, load_ckpt_retry))
                try:
                    torch.load(ckpt_location)
                except:
                    print("Corrupted ckpt at {}".format(ckpt_location_curr))
                else:
                    print("Checkpoint verified at {}".format(ckpt_location))
                    load_ckpt_successful = True
                    valid_checkpoint = True
                    load_this_ckpt = ckpt_location
    dist.barrier()

    if valid_checkpoint and os.path.exists(load_this_ckpt):
        loc = 'cuda:{}'.format(args.gpu)
        ckpt = torch.load(load_this_ckpt, map_location=loc)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        ckpt_epoch = ckpt["epoch"]
        best_acc1 = ckpt['best_acc1']
        if lr_scheduler is not None:
            for _dummy in range(ckpt_epoch-1):
                lr_scheduler.step()
        if scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        print("CHECKPOINT LOADED to device: {}".format(device))
        del ckpt
        torch.cuda.empty_cache()
    else:
        print('NO CHECKPOINT LOADED, FRESH START!')
    dist.barrier()

    actual_trained_epoch = args.epoch

    if is_main_task:
        print('This is the device: {} for the main task!'.format(device))
        # was hanging on wandb init on wandb 0.12.9, fixed after upgrading to 0.15.7
        if args.enable_wandb:
            wandb_logger = wandbLogger(args)
        logger = metaLogger(args)
        logging.basicConfig(
            filename=args.j_dir+ "/log/log.txt",
            format='%(asctime)s %(message)s', level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    train_loader, test_loader, train_sampler, val_sampler = load_dataset(
                args.dataset,
                args.batch_size,
                args.op_name,
                args.op_prob,
                args.op_magnitude,
                args.workers,
                args.distributed
                )

    # test_loader_1k contains exactly 1 sample from each of the 1000 class
    test_loader_1k = load_imagenet_test_1k(
                batch_size=args.batch_size,
                workers=0,
                distributed=args.distributed
                )

    # test_loader_shuffle is contains the same number of data as the original
    # but data is randomly shuffled, this is for evaluating transfer attack
    test_loader_shuffle, val_sampler = load_imagenet_test_shuffle(
                batch_size=args.batch_size,
                workers=0,
                distributed=args.distributed
                )


    num_classes = 1000
    mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            num_categories=num_classes,
            use_v2=args.use_v2
            )
    print('finished data loader')
##########################################################
###################### Training begins ###################
##########################################################
    dist.barrier()
    for _epoch in range(ckpt_epoch, args.epoch+1):
        if args.distributed:
            train_sampler.set_epoch(_epoch)

        # train for one epoch
        dist.barrier()
        train_acc1, train_acc5, loss = train(train_loader, model, criterion, opt, _epoch, device, args, is_main_task, scaler, mixup_cutmix)

        dist.barrier()
        test_acc1, test_acc5 = validate(test_loader, model, criterion, args, is_main_task, whitebox=False)

        dist.barrier()
        test_acc1, test_acc5 = validate(test_loader_1k, model, criterion, args, is_main_task, whitebox=True)

        # from other models to the trained model
        dist.barrier()
        if args.distributed:
            val_sampler.set_epoch(_epoch)
        acc1_model_to_resnet18, acc1_resnet18_to_model = eval_transfer_bi_direction(
                test_loader_shuffle,
                model,
                model_resnet18,
                args,
                is_main_task)

        dist.barrier()
        if args.distributed:
            val_sampler.set_epoch(_epoch)
        acc1_model_to_resnet50, acc1_resnet50_to_model = eval_transfer_bi_direction(
                test_loader_shuffle,
                model,
                model_resnet50,
                args,
                is_main_task)

        dist.barrier()
        if args.distributed:
            val_sampler.set_epoch(_epoch)
        acc1_model_to_vgg19_bn, acc1_vgg19_bn_to_model = eval_transfer_bi_direction(
                test_loader_shuffle,
                model,
                model_vgg19_bn,
                args,
                is_main_task)
        dist.barrier()

        lr_scheduler.step()

        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1, best_acc1)

        # Logging and checkpointing only at the main task (rank0)
        if is_main_task:
            logger.add_scalar("train/top1_err", 100-train_acc1, _epoch)
            logger.add_scalar("train/top5_err", 100-train_acc5, _epoch)
            logger.add_scalar("train/loss", loss, _epoch)
            logger.add_scalar("lr", opt.param_groups[0]['lr'], _epoch)
            logger.add_scalar("test/top1_err", 100-test_acc1, _epoch)
            logger.add_scalar("test/top5_err", 100-test_acc5, _epoch)
            logger.add_scalar("test/best_top1_err", 100-best_acc1, _epoch)

            logger.add_scalar("transfer/from_resnet18", 100-acc1_resnet18_to_model, _epoch)
            logger.add_scalar("transfer/from_resnet50", 100-acc1_resnet50_to_model, _epoch)
            logger.add_scalar("transfer/from_vgg19_bn", 100-acc1_vgg19_bn_to_model, _epoch)

            logger.add_scalar("transfer/to_resnet18", 100-acc1_model_to_resnet18, _epoch)
            logger.add_scalar("transfer/to_resnet50", 100-acc1_model_to_resnet50, _epoch)
            logger.add_scalar("transfer/to_vgg19_bn", 100-acc1_model_to_vgg19_bn, _epoch)

            logging.info(
                "Epoch: [{0}]\t"
                "Train Loss: {loss:.6f}\t"
                "Train Accuracy(top1): {train_acc1:.2f}\t"
                "Train Accuracy(top5): {train_acc5:.2f}\t"
                "Test Accuracy(top1): {test_acc1:.2f}\t"
                "Test Accuracy(top5): {test_acc5:.2f}\t".format(
                    _epoch,
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
                        "best_acc1":best_acc1
                        }
                if scaler is not None:
                    ckpt["scaler"] = scaler.state_dict()

                rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                logger.save_log()

            # save best model
            if is_best and _epoch > int(args.epoch*3/4):
                saveModel(args.j_dir+"/model/", "best_model", model.state_dict())

        # Early terminate training when half way thru training and test accuracy still below 20%
        if np.isnan(loss) or (_epoch > int(args.epoch/2) and test_acc1 < 20):
            print('Early stopping at epoch {}.'.format(_epoch))
            actual_trained_epoch = _epoch
            saveModel(args.j_dir+"/model/", "final_model", model.state_dict())
            break # break the training for-loop
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

    # Evaluation on imagenet-a, imagenet-o, imagenet-r
    # _, imagenet_a_loader, _, _ = load_dataset('imagenet-a', args.batch_size, args.workers, args.distributed)
    # imagenet_a_acc1, imagenet_a_acc5 = validate(imagenet_a_loader, model, criterion, args)
    # _, imagenet_o_loader, _, _ = load_dataset('imagenet-o', args.batch_size, args.workers, args.distributed)
    # imagenet_o_acc1, imagenet_o_acc5 = validate(imagenet_o_loader, model, criterion, args)
    # _, imagenet_r_loader, _, _ = load_dataset('imagenet-r', args.batch_size, args.workers, args.distributed)
    # imagenet_r_acc1, imagenet_r_acc5 = validate(imagenet_r_loader, model, criterion, args)
    
    # if is_main_task:
        # logger.add_scalar("imagenet-a/top1_acc", imagenet_a_acc1, args.epoch)
        # logger.add_scalar("imagenet-a/top5_acc", imagenet_a_acc5, args.epoch)
        # logger.add_scalar("imagenet-o/top1_acc", imagenet_o_acc1, args.epoch)
        # logger.add_scalar("imagenet-o/top5_acc", imagenet_o_acc5, args.epoch)
        # logger.add_scalar("imagenet-r/top1_acc", imagenet_r_acc1, args.epoch)
        # logger.add_scalar("imagenet-r/top5_acc", imagenet_r_acc5, args.epoch)

    # Evaluation on imagenet-c
    # for _severity in [1, 3, 5]:
        # corruption_acc1 = []
        # corruption_acc5 = []
        # for corruption in CORRUPTIONS_IMAGENET_C:
            # try:
                # imagenet_c_loader = load_IMAGENET_C(args.batch_size,
                                                # corruption,
                                                # _severity,
                                                # args.workers,
                                                # args.distributed)
            # except:
                # print("failed to load {}({})".format(corruption, _severity))
            # else:
                # print("{}({}) loaded".format(corruption, _severity))
                # imagenet_c_acc1, imagenet_c_acc5 = validate(imagenet_c_loader, model, criterion, args)
                # corruption_acc1.append(imagenet_c_acc1)
                # corruption_acc5.append(imagenet_c_acc5)
                # if is_main_task:
                    # logger.add_scalar('imagenet-c/{}-{}/top1_acc'.format(corruption, _severity),
                            # imagenet_c_acc1, args.epoch)
                    # logger.add_scalar('imagenet-c/{}-{}/top5_acc'.format(corruption, _severity),
                            # imagenet_c_acc5, args.epoch)
        # if is_main_task:
            # logger.add_scalar('imagenet-c/mCC-{}/top1_acc'.format(_severity),
                    # np.array(corruption_acc1).mean(), args.epoch)
            # logger.add_scalar('imagenet-c/mCC-{}/top5_acc'.format(_severity),
                    # np.array(corruption_acc5).mean(), args.epoch)

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
    delCheckpoint(args.j_dir, args.j_id)
    ddp_cleanup()



def remove_module(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

if __name__ == "__main__":
    main()

