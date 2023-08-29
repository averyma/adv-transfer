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
from torch.utils.data import Subset

import numpy as np

from src.args import get_args, print_args
from src.evaluation import test_clean, test_AA, eval_corrupt, eval_CE, test_gaussian, CORRUPTIONS_IMAGENET_C

from src.utils_dataset import load_dataset, load_imagenet_test_shuffle, load_imagenet_test_1k
from src.utils_log import metaLogger, rotateCheckpoint, wandbLogger, saveModel, delCheckpoint
from src.utils_general import seed_everything, get_model, get_optim
from src.transforms import get_mixup_cutmix
import copy
from src.attacks import pgd
from src.context import ctx_noparamgrad_and_eval
import torch.nn.functional as F
import ipdb

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
        args.gpu=0
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
    '''
    load source model (to be aligned) and witness model
    # source model:  seed 40
    # target model:  seed 41
    # witness model: seed 42
    '''
    args.arch = args.source_arch
    source_model = get_model(args)
    source_model_dir = root_dir + model_ckpt[args.dataset][args.source_arch] + '0/model/best_model.pt'
    loc = 'cuda:{}'.format(args.gpu)
    ckpt = remove_module(torch.load(source_model_dir, map_location=loc))
    source_model.load_state_dict(ckpt)
    orig_source_model = copy.deepcopy(source_model)
    print('Load source model from {} to gpu{}'.format(source_model_dir, args.gpu))

    args.arch = args.witness_arch
    witness_model = get_model(args)
    witness_model_dir = root_dir + model_ckpt[args.dataset][args.witness_arch] + '2/model/best_model.pt'
    loc = 'cuda:{}'.format(args.gpu)
    ckpt = remove_module(torch.load(witness_model_dir, map_location=loc))
    witness_model.load_state_dict(ckpt)
    print('Load witness model from {} to gpu{}'.format(witness_model_dir, args.gpu))

    result = {
            'pre-test': None,
            'post-test': None,
            'pre-rob': None,
            'post-rob': None,
            'pre-rob-resnet18': None,
            'post-rob-resnet18': None,
            'pre-rob-resnet50': None,
            'post-rob-resnet50': None,
            'pre-rob-vgg19_bn': None,
            'post-rob-vgg19_bn': None,
            'pre-tnsf-resnet18': None,
            'post-tnsf-resnet18': None,
            'pre-tnsf-resnet50': None,
            'post-tnsf-resnet50': None,
            'pre-tnsf-vgg19_bn': None,
            'post-tnsf-vgg19_bn': None,
            'pre-mean-rob': None,
            'post-mean-rob': None,
            'pre-mean-tnsf': None,
            'post-mean-tnsf': None,
            'mean-rob-diff': None,
            'mean-tnsf-diff': None,
                }

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                source_model.cuda(args.gpu)
                orig_source_model.cuda(args.gpu)
                witness_model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                # args.workers = mp.cpu_count()//max(ngpus_per_node, 1)
                args.workers = 4
                print("GPU: {}, batch_size: {}, workers: {}".format(args.gpu, args.batch_size, args.workers))
                source_model = torch.nn.parallel.DistributedDataParallel(source_model, device_ids=[args.gpu])
                orig_source_model = torch.nn.parallel.DistributedDataParallel(orig_source_model, device_ids=[args.gpu])
                witness_model = torch.nn.parallel.DistributedDataParallel(witness_model, device_ids=[args.gpu])
            else:
                source_model.cuda()
                witness_model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                source_model = torch.nn.parallel.DistributedDataParallel(source_model)
                witness_model = torch.nn.parallel.DistributedDataParallel(witness_model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        source_model = source_model.cuda(args.gpu)
        witness_model = witness_model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        source_model = source_model.to(device)
        witness_model = witness_model.to(device)
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

    opt, lr_scheduler = get_optim(source_model, args)

    ckpt_dir = args.j_dir+"/"+str(args.j_id)+"/"
    ckpt_location_curr = os.path.join(ckpt_dir, "ckpt_curr.pth")
    ckpt_location_prev = os.path.join(ckpt_dir, "ckpt_prev.pth")

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
        source_model.load_state_dict(ckpt["state_dict"])
        result = ckpt['result']
        print("CHECKPOINT LOADED to device: {}".format(device))
        del ckpt
        torch.cuda.empty_cache()
    else:
        print('NO CHECKPOINT LOADED, FRESH START!')
    dist.barrier()

    actual_trained_epoch = 1
    _epoch = 1

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
 
    # train_loader and test_loader are the original loader for imagenet
    # train_sampler is necessary for alignment
    # val_sampler is removed so we can use the one from test_loader_shuffle
    train_loader, test_loader, train_sampler, _ = load_dataset(
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

    num_classes = 1000
    print('finished data loader')
    print('Modifying {} using {}!'.format(args.source_arch, args.witness_arch))
##########################################################
###################### Training begins ###################
##########################################################
    dist.barrier()
    if not valid_checkpoint:
        # for _epoch in range(ckpt_epoch, args.epoch+1):
        if args.distributed:
            train_sampler.set_epoch(_epoch)
        train_acc1, train_acc5, loss = match_kl(train_loader,
                                                source_model,
                                                witness_model,
                                                criterion,
                                                opt,
                                                _epoch,
                                                device,
                                                args,
                                                is_main_task)
        del train_loader
    dist.barrier()
##########################################################
###################### Training ends #####################
##########################################################

    # checkpointing for preemption
    if is_main_task:
        ckpt = { "state_dict": source_model.state_dict(), 'result': result}
        rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
        logger.save_log()
    dist.barrier()
    
    for (prefix, model) in zip(['pre', 'post'], [orig_source_model, source_model]):

        if result[prefix + '-test'] is None:
            dist.barrier()
            test_acc1, test_acc5 = validate(test_loader, model, criterion, args, is_main_task)
            dist.barrier()
            result[prefix + '-test'] = test_acc1
            if is_main_task:
                ckpt = { "state_dict": source_model.state_dict(), 'result': result}
                rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                logger.save_log()

        if result[prefix + '-rob'] is None:
            dist.barrier()
            test_acc1, test_acc5 = validate(test_loader_1k, model, criterion, args, is_main_task, whitebox=True)
            dist.barrier()
            result[prefix + '-rob'] = test_acc1
            if is_main_task:
                print('{}: {:.2f}'.format(prefix + '-rob', test_acc1))
                ckpt = { "state_dict": source_model.state_dict(), 'result': result}
                rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                logger.save_log()

        for target_arch in ['resnet18', 'resnet50', 'vgg19_bn']:
            if result[prefix + '-rob-' + target_arch] is None:
                args.arch = target_arch
                target_model = get_model(args)
                target_model_dir = root_dir + model_ckpt[args.dataset][target_arch] + '1/model/best_model.pt'
                loc = 'cuda:{}'.format(args.gpu)
                ckpt = remove_module(torch.load(target_model_dir, map_location=loc))
                target_model.load_state_dict(ckpt)
                target_model.cuda(args.gpu)
                target_model = torch.nn.parallel.DistributedDataParallel(target_model, device_ids=[args.gpu])

                dist.barrier()
                if args.distributed:
                    val_sampler.set_epoch(27)
                if is_main_task:
                    test_acc1, test_acc5 = eval_transfer(test_loader_shuffle, target_model, source_model, args, is_main_task)
                    result[prefix + '-rob-' + target_arch] = test_acc1
                    print('{}: {:.2f}'.format(prefix + '-rob-' + target_arch, test_acc1))
                    ckpt = { "state_dict": source_model.state_dict(), 'result': result}
                    rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                    logger.save_log()

            if result[prefix + '-tnsf-' + target_arch] is None:
                args.arch = target_arch
                target_model = get_model(args)
                target_model_dir = root_dir + model_ckpt[args.dataset][target_arch] + '1/model/best_model.pt'
                loc = 'cuda:{}'.format(args.gpu)
                ckpt = remove_module(torch.load(target_model_dir, map_location=loc))
                target_model.load_state_dict(ckpt)
                target_model.cuda(args.gpu)
                target_model = torch.nn.parallel.DistributedDataParallel(target_model, device_ids=[args.gpu])

                dist.barrier()
                if args.distributed:
                    val_sampler.set_epoch(27)
                if is_main_task:
                    test_acc1, test_acc5 = eval_transfer(test_loader_shuffle, source_model, target_model, args, is_main_task)
                    result[prefix + '-tnsf-' + target_arch] = test_acc1
                    print('{}: {:.2f}'.format(prefix + '-tnsf-' + target_arch, test_acc1))
                    ckpt = { "state_dict": source_model.state_dict(), 'result': result}
                    rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                    logger.save_log()

    dist.barrier()

    # Logging and checkpointing only at the main task (rank0)
    if is_main_task:
        result['pre-mean-rob'] = (result['pre-rob-resnet18']
                                +result['pre-rob-resnet50']
                                +result['pre-rob-vgg19_bn'])/3
        result['post-mean-rob'] = (result['post-rob-resnet18']
                                +result['post-rob-resnet50']
                                +result['post-rob-vgg19_bn'])/3
        result['pre-mean-tnsf'] = (result['pre-tnsf-resnet18']
                                    +result['pre-tnsf-resnet50']
                                    +result['pre-tnsf-vgg19_bn'])/3
        result['post-mean-tnsf'] = (result['post-tnsf-resnet18']
                                    +result['post-tnsf-resnet50']
                                    +result['post-tnsf-vgg19_bn'])/3
        result['mean-rob-diff'] = result['pre-mean-rob'] - result['post-mean-rob']
        result['mean-tnsf-diff'] = result['pre-mean-tnsf'] - result['post-mean-tnsf']

        for key in result.keys():
            logger.add_scalar(key, result[key], _epoch)
            logging.info("{}: {:.2f}\t".format(key, result[key]))

    dist.barrier()

    # upload runs to wandb:
    if is_main_task:
        print('Saving final model!')
        saveModel(args.j_dir+"/model/", "final_model", model.state_dict())
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

def match_kl(train_loader, source_model, witness_model, criterion, optimizer, epoch, device, args, is_main_task):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    source_model.train()
    witness_model.eval()
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    if args.noise_type != 'none':
        param = {'ord': np.inf,
              'epsilon': 4./255.,
              'alpha': 1./255.,
              'num_iter': 4,
              'restarts': 1,
              'rand_init': True,
              'clip': True,
              'loss_fn': nn.CrossEntropyLoss(),
              'dataset': args.dataset}
        if args.noise_type == 'rand_init':
            param['num_iter'] = 0
        elif args.noise_type.startswith('pgd'):
            _itr = args.noise_type[3:-6] if args.noise_type.endswith('indep') else args.noise_type[3::]
            param['num_iter'] = int(_itr)
        attacker = pgd(**param)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if args.noise_type != 'none':
            with ctx_noparamgrad_and_eval(source_model):
                delta = attacker.generate(source_model, images, target)
        else:
            delta = 0

        # compute output
        p_s = source_model(images+delta)
        yp_s = F.log_softmax(p_s, dim=1)

        with ctx_noparamgrad_and_eval(witness_model):
            yp_w = F.log_softmax(witness_model(images+delta), dim=1)
            if args.misalign:
                loss = -(kl_loss(yp_s, yp_w) + kl_loss(yp_w, yp_s))
            else:
                loss = (kl_loss(yp_s, yp_w) + kl_loss(yp_w, yp_s))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(p_s, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and is_main_task:
            progress.display(i + 1)

    return top1.avg, top5.avg, losses.avg

def validate(val_loader, model, criterion, args, is_main_task, whitebox=False):
    if whitebox:
        param = {'ord': np.inf,
              'epsilon': 4./255.,
              'alpha': 1./255.,
              'num_iter': 4,
              'restarts': 1,
              'rand_init': True,
              'clip': True,
              'loss_fn': nn.CrossEntropyLoss(),
              'dataset': args.dataset}
        attacker = pgd(**param)

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
                    delta = attacker.generate(model, images, target)
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
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    if is_main_task:
        progress.display_summary()

    return top1.avg, top5.avg

def eval_transfer(val_loader, source_model, target_model, args, is_main_task):
    param = {'ord': np.inf,
          'epsilon': 4./255.,
          'alpha': 1./255.,
          'num_iter': 4,
          'restarts': 1,
          'rand_init': True,
          'clip': True,
          'loss_fn': nn.CrossEntropyLoss(),
          'dataset': args.dataset}
    attacker = pgd(**param)
    num_eval = 500

    def run_validate(loader, base_progress=0):
        total_qualified = 0
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

            with ctx_noparamgrad_and_eval(source_model):
                delta_s = attacker.generate(source_model, images, target)

            with ctx_noparamgrad_and_eval(target_model):
                delta_t = attacker.generate(target_model, images, target)

            # compute output
            with torch.no_grad():
                p_s = source_model(images)
                p_t = target_model(images)
                p_adv_s = source_model(images+delta_s)
                p_adv_t = target_model(images+delta_t)
                transfer_qualified = return_qualified(p_s, p_t, p_adv_s, p_adv_t, target)

                p_transfer = target_model((images+delta_s)[transfer_qualified, ::])

            # measure accuracy and record loss
            num_qualified = transfer_qualified.sum().item()
            acc1, acc5 = accuracy(p_transfer, target[transfer_qualified], topk=(1, 5))
            top1.update(acc1[0], num_qualified)
            top5.update(acc5[0], num_qualified)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and is_main_task:
                progress.display(i + 1)

            total_qualified += num_qualified
            if total_qualified > num_eval:
                break

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    source_model.eval()
    target_model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    if is_main_task:
        progress.display_summary()

    return top1.avg, top5.avg

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def return_qualified(p_0, p_1, p_adv_0, p_adv_1, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        _, pred_0 = p_0.topk(1, 1, True, True)
        _, pred_1 = p_1.topk(1, 1, True, True)
        _, pred_adv_0 = p_adv_0.topk(1, 1, True, True)
        _, pred_adv_1 = p_adv_1.topk(1, 1, True, True)

        pred_0 = pred_0.t()
        pred_1 = pred_1.t()
        pred_adv_0 = pred_adv_0.t()
        pred_adv_1 = pred_adv_1.t()

        correct_0 = pred_0.eq(target.view(1, -1).expand_as(pred_0)).squeeze()
        correct_1 = pred_1.eq(target.view(1, -1).expand_as(pred_0)).squeeze()
        incorrect_0 = pred_adv_0.ne(target.view(1, -1).expand_as(pred_0)).squeeze()
        incorrect_1 = pred_adv_1.ne(target.view(1, -1).expand_as(pred_0)).squeeze()
        qualified = correct_0.eq(correct_1).eq(incorrect_0).eq(incorrect_1)

        return qualified

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
