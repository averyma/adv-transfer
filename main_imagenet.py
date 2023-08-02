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

from src.utils_dataset import load_dataset, load_IMAGENET_C
from src.utils_log import metaLogger, rotateCheckpoint, wandbLogger, saveModel, delCheckpoint
from src.utils_general import seed_everything, get_model, get_optim

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
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                args.workers = mp.cpu_count()//max(ngpus_per_node, 1)
                print("GPU: {}, batch_size: {}, workers: {}".format(args.gpu, args.batch_size, args.workers))
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
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

    criterion = nn.CrossEntropyLoss().to(device)

    opt, lr_scheduler = get_optim(model, args)

    ckpt_epoch = 1

    ckpt_dir = args.j_dir+"/"+str(args.j_id)+"/"
    ckpt_location_curr = os.path.join(ckpt_dir, "ckpt_curr.pth")
    ckpt_location_prev = os.path.join(ckpt_dir, "ckpt_prev.pth")

    valid_checkpoint = False
    for ckpt_location in [ckpt_location_curr, ckpt_location_prev]:
        if os.path.exists(ckpt_location):
            print("Checkpoint found at {}!".format(ckpt_location))
            try:
                torch.load(ckpt_location)
            except:
                print("Corrupted ckpt at {}".format(ckpt_location_curr))
            else:
                print("Checkpoint verified at {}!".format(ckpt_location))
                valid_checkpoint = True
                load_this_ckpt = ckpt_location
                break

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
        print("CHECKPOINT LOADED to device: {}".format(device))
    else:
        print('NO CHECKPOINT LOADED, FRESH START!')

    actual_trained_epoch = args.epoch

    is_main_task = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    if is_main_task:
        print('This is the device: {} for the main task!'.format(device))
        # was hanging on wandb init on wandb 0.12.9, fixed after upgrading to 0.15.7
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

##########################################################
###################### Training begins ###################
##########################################################
    dist.barrier()
    for _epoch in range(ckpt_epoch, args.epoch+1):
        if args.distributed:
            train_sampler.set_epoch(_epoch)

        # train for one epoch
        if args.optimize_cluster_param:
            test_acc1, test_acc5 = 99, 99
            # train_acc1, train_acc5, loss = 99, 99, 0
            dist.barrier()
            train_acc1, train_acc5, loss = train(train_loader, model, criterion, opt, _epoch, device, args, is_main_task)
            dist.barrier()
        else:
            dist.barrier()
            train_acc1, train_acc5, loss = train(train_loader, model, criterion, opt, _epoch, device, args, is_main_task)
            dist.barrier()
            test_acc1, test_acc5 = validate(test_loader, model, criterion, args, is_main_task)
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
                rotateCheckpoint(ckpt_dir, "custom_ckpt", model, opt, _epoch+1, best_acc1)
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

def train(train_loader, model, criterion, optimizer, epoch, device, args, is_main_task):
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
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and is_main_task:
            progress.display(i + 1)

    return top1.avg, top5.avg, losses.avg

def validate(val_loader, model, criterion, args, is_main_task):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
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

                # compute output
                output = model(images)
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

if __name__ == "__main__":
    main()
