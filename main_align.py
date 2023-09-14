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
from src.utils_general import seed_everything, get_model, get_optim
from src.transforms import get_mixup_cutmix
import copy
from src.attacks import pgd
from src.context import ctx_noparamgrad_and_eval
import torch.nn.functional as F
import ipdb
from src.evaluation import validate, eval_transfer, eval_transfer_bi_direction
from src.transfer import model_align, model_align_feature_space
from distiller_zoo import RKDLoss, EGA, PKT, DistillKL, HintLoss, NCELoss

root_dir = '/scratch/hdd001/home/ama/improve-transferability/'
model_ckpt = {
        'cifar10': {
            'preactresnet18': '2023-07-19/cifar10/cosine/20230719-cifar10-preactresnet18-0.1-4',
            'preactresnet50': '2023-07-21/cifar10/cosine/20230721-cifar10-preactresnet50-0.1-4',
            'vgg19': '2023-07-27/20230727-cifar10-vgg19-0.1-4',
            'vit_small': '2023-08-02/20230802-cifar10-vit_small-0.1-4'
            },
        'cifar100': {
            'preactresnet18': '2023-07-19/cifar100/cosine/20230719-cifar100-preactresnet18-0.1-4',
            'preactresnet50': '2023-07-21/cifar100/cosine/20230721-cifar100-preactresnet50-0.1-4',
            'vgg19': '2023-07-27/20230727-cifar100-vgg19-0.1-4',
            'vit_small': '2023-08-02/20230802-cifar100-vit_small-0.1-4'
            },
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
        args.gpu = 0 if torch.cuda.is_available() else None
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):

    args.ngpus_per_node = ngpus_per_node
    args.gpu = gpu
    device = torch.device('cuda:{}'.format(args.gpu))

    assert args.method in ['kl', 'nce', 'hint', 'pkt', 'ega', 'rkd']

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
        list_target_arch = ['resnet18', 'resnet50', 'vgg19_bn']

    args.arch = args.source_arch
    source_model = get_model(args)
    source_model_dir = root_dir + model_ckpt[args.dataset][args.source_arch] + '0/model/best_model.pt'
    ckpt = torch.load(source_model_dir, map_location=device)
    try:
        source_model.load_state_dict(ckpt)
    except RuntimeError:
        source_model.load_state_dict(remove_module(ckpt))
    orig_source_model = copy.deepcopy(source_model)
    print('{}: Load source model from {}.'.format(device, source_model_dir))

    args.arch = args.witness_arch
    witness_model = get_model(args)
    witness_model_dir = root_dir + model_ckpt[args.dataset][args.witness_arch] + '2/model/best_model.pt'
    ckpt = torch.load(witness_model_dir, map_location=device)
    try:
        witness_model.load_state_dict(ckpt)
    except RuntimeError:
        witness_model.load_state_dict(remove_module(ckpt))
    print('{}: Load witness model from {}.'.format(device, witness_model_dir))

    if args.method != 'kl':
        if args.source_arch != args.witness_arch:
            args.project_source_embedding = True
        else:
            args.project_source_embedding = args.no_proj_when_dim_matches
    else:
        args.project_source_embedding = False


    if args.project_source_embedding:
        from models import LinearProjection

        if args.source_arch in ['resnet18', 'preactresnet18', 'vgg19', 'vit_small']:
            dim_emb_source = 512
        elif args.source_arch in ['resnet50', 'preactresnet50']:
            dim_emb_source = 2048
        elif args.source_arch in 'vgg19_bn':
            dim_emb_source = 25088

        if args.witness_arch in ['resnet18', 'preactresnet18', 'vgg19', 'vit_small']:
            dim_emb_witness = 512
        elif args.witness_arch in ['resnet50', 'preactresnet50']:
            dim_emb_witness = 2048
        elif args.witness_arch in 'vgg19_bn':
            dim_emb_witness = 25088

        source_projection = LinearProjection(dim_emb_source, dim_emb_witness)
    print('{}: Use linear projection: {}'.format(device, args.project_source_embedding))

    result = {
            'pre/test-err': None,
            'post/test-err': None,
            'diff/test-err': None,
            'pre/whitebox-err': None,
            'post/whitebox-err': None,
            'diff/whitebox-err': None,
            # 'pre/transfer-from-resnet18': None,
            # 'post/transfer-from-resnet18': None,
            # 'diff/transfer-from-resnet18': None,
            # 'pre/transfer-from-resnet50': None,
            # 'post/transfer-from-resnet50': None,
            # 'diff/transfer-from-resnet50': None,
            # 'pre/transfer-from-vgg19_bn': None,
            # 'post/transfer-from-vgg19_bn': None,
            # 'diff/transfer-from-vgg19_bn': None,
            'pre/avg-transfer-from': None,
            'post/avg-transfer-from': None,
            'diff/avg-transfer-from': None,
            # 'pre/transfer-to-resnet18': None,
            # 'post/transfer-to-resnet18': None,
            # 'diff/transfer-to-resnet18': None,
            # 'pre/transfer-to-resnet50': None,
            # 'post/transfer-to-resnet50': None,
            # 'diff/transfer-to-resnet50': None,
            # 'pre/transfer-to-vgg19_bn': None,
            # 'post/transfer-to-vgg19_bn': None,
            # 'diff/transfer-to-vgg19_bn': None,
            'pre/avg-transfer-to': None,
            'post/avg-transfer-to': None,
            'diff/avg-transfer-to': None,
                }
    for _arch in list_target_arch:
        for _metric in ['pre', 'post', 'diff']:
            for _to_from in ['to', 'from']:
                result[_metric+'/transfer-'+_to_from+'-'+_arch] = None

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
        orig_source_model.cuda(args.gpu)
        witness_model.cuda(args.gpu)
        # if args.method != 'kl':
        if args.project_source_embedding:
            # teac_net.cuda(args.gpu)
            source_projection.cuda(args.gpu)
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
        # if args.method != 'kl':
        if args.project_source_embedding:
            # teac_net = torch.nn.parallel.DistributedDataParallel(teac_net, device_ids=[args.gpu])
            source_projection = torch.nn.parallel.DistributedDataParallel(source_projection, device_ids=[args.gpu])
    else:
        torch.cuda.set_device(args.gpu)
        source_model = source_model.cuda(args.gpu)
        orig_source_model = orig_source_model.cuda(args.gpu)
        witness_model = witness_model.cuda(args.gpu)
        # if args.method != 'kl':
        if args.project_source_embedding:
            source_projection = source_projection.cuda(args.gpu)

    is_main_task = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    print('{}: is_main_task: {}'.format(device, is_main_task))

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
        criterion_kd = DistillKL(args.kl_temp).to(device)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_kd)

    module_list = nn.ModuleList([])
    module_list.append(source_model)
    module_list.append(witness_model)

    trainable_list = nn.ModuleList([])
    trainable_list.append(source_model)

    # if args.method != 'kl':
    if args.project_source_embedding:
        module_list.append(source_projection)
        trainable_list.append(source_projection)

    opt, _ = get_optim(trainable_list, args)

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
        source_model.load_state_dict(ckpt["state_dict"])
        result = ckpt['result']
        print("{}: CHECKPOINT LOADED!".format(device))
        del ckpt
        torch.cuda.empty_cache()
    else:
        print('{}: NO CHECKPOINT LOADED, FRESH START!'.format(device))
    if args.distributed:
        dist.barrier()

    actual_trained_epoch = 1
    _epoch = 1

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
                args.op_name,
                args.op_prob,
                args.op_magnitude,
                args.workers,
                args.distributed
                )

    if args.dataset == 'imagenet':
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
    else:
        test_loader_1k = test_loader
        test_loader_shuffle = test_loader

    print('{}: Dataloader compelete! Ready for alignment!'.format(device))
    if is_main_task:
        print('Modifying {} with {} using {}!'.format(args.source_arch, args.witness_arch, args.method))
##########################################################
###################### Training begins ###################
##########################################################
    if args.distributed:
        dist.barrier()
    if not valid_checkpoint:
        # for _epoch in range(ckpt_epoch, args.epoch+1):
        if args.distributed:
            train_sampler.set_epoch(_epoch)
        train_acc1, train_acc5, loss = model_align_feature_space(train_loader,
                                                                module_list,
                                                                criterion_list,
                                                                opt,
                                                                device,
                                                                args,
                                                                is_main_task)
        del train_loader
        return 0
    if args.distributed:
        dist.barrier()
##########################################################
###################### Training ends #####################
##########################################################

    # checkpointing for preemption
    if is_main_task:
        ckpt = { "state_dict": source_model.state_dict(), 'result': result}
        rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
        logger.save_log()
    if args.distributed:
        dist.barrier()
    
    for (prefix, model) in zip(['pre/', 'post/'], [orig_source_model, source_model]):

        if result[prefix + 'test-err'] is None:
            if args.distributed:
                dist.barrier()
            test_acc1, test_acc5 = validate(test_loader, model, criterion_cls, args, is_main_task)
            _result = 100.-test_acc1
            if args.distributed:
                dist.barrier()
            result[prefix + 'test-err'] = _result
            if is_main_task:
                print('{}: {:.2f}'.format(prefix + 'test-err', _result))
                ckpt = { "state_dict": source_model.state_dict(), 'result': result}
                rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                logger.save_log()

        if result[prefix + 'whitebox-err'] is None:
            if args.distributed:
                dist.barrier()
            test_acc1, test_acc5 = validate(test_loader_1k, model, criterion_cls, args, is_main_task, whitebox=True)
            _result = 100.-test_acc1
            if args.distributed:
                dist.barrier()
            result[prefix + 'whitebox-err'] = _result
            if is_main_task:
                print('{}: {:.2f}'.format(prefix + 'whitebox-err', _result))
                ckpt = { "state_dict": source_model.state_dict(), 'result': result}
                rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                logger.save_log()

        for target_arch in list_target_arch:
            if result[prefix + 'transfer-from-' + target_arch] is None:
                args.arch = target_arch
                target_model = get_model(args)
                target_model_dir = root_dir + model_ckpt[args.dataset][target_arch] + '1/model/best_model.pt'
                ckpt = torch.load(target_model_dir, map_location=device)
                try:
                    target_model.load_state_dict(ckpt)
                except RuntimeError:
                    target_model.load_state_dict(remove_module(ckpt))
                target_model.cuda(args.gpu)
                if args.distributed:
                    target_model = torch.nn.parallel.DistributedDataParallel(target_model, device_ids=[args.gpu])

                if args.distributed:
                    dist.barrier()
                    val_sampler.set_epoch(27)
                test_acc1_target2source, test_acc1_source2target = eval_transfer_bi_direction(
                                                                    test_loader_shuffle,
                                                                    model_a=target_model,
                                                                    model_b=source_model,
                                                                    args=args,
                                                                    is_main_task=is_main_task)
                _result_target2source = 100.-test_acc1_target2source
                _result_source2target = 100.-test_acc1_source2target
                if args.distributed:
                    dist.barrier()
                if is_main_task:
                    result[prefix + 'transfer-from-' + target_arch] = _result_target2source
                    print('{}: {:.2f}'.format(prefix + 'transfer-from-' + target_arch, _result_target2source))
                    result[prefix + 'transfer-to-' + target_arch] = _result_source2target
                    print('{}: {:.2f}'.format(prefix + 'transfer-to-' + target_arch, _result_source2target))
                    ckpt = { "state_dict": source_model.state_dict(), 'result': result}
                    rotateCheckpoint(ckpt_dir, "ckpt", ckpt)
                    logger.save_log()

    if args.distributed:
        dist.barrier()

    # Logging and checkpointing only at the main task (rank0)
    if is_main_task:
        for _metric in ['test-err', 'whitebox-err', 'transfer-from-', 'transfer-to-']:
            if _metric in ['transfer-from-', 'transfer-to-']:
                for _arch in list_target_arch:
                    result['diff/{}{}'.format(_metric, _arch)] = result['post/{}{}'.format(_metric, _arch)] - result['pre/{}{}'.format(_metric, _arch)]
            else:
                result['diff/{}'.format(_metric)] = result['post/{}'.format(_metric)] - result['pre/{}'.format(_metric)]

        for _metric in ['avg-transfer-from', 'avg-transfer-to']:
            for _prefix in ['pre/', 'post/', 'diff/']:
                _result = 0
                for _arch in list_target_arch:
                    _result += result[_prefix+_metric[4:]+'-'+_arch]/3.
                result[_prefix+_metric] = _result

        for key in result.keys():
            logger.add_scalar(key, result[key], _epoch)
            logging.info("{}: {:.2f}\t".format(key, result[key]))

    if args.distributed:
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
    if is_main_task:
        delCheckpoint(args.j_dir, args.j_id)
    if args.distributed:
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

