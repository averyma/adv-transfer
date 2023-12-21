import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from src.attacks import pgd, Linf_ball_projection
from src.context import ctx_noparamgrad_and_eval
import ipdb
from tqdm import trange
import numpy as np
from src.utils_log import Summary, AverageMeter, ProgressMeter
from src.evaluation import accuracy
import time
from distiller_zoo import RKDLoss, EGA, PKT, DistillKL, HintLoss, NCELoss

def align_feature_space(train_loader, list_trainable, list_witness_model, criterion_kd, criterion_cls, optimizer, lr_scheduler, scaler, epoch, device, args, is_main_task):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Align Epoch: [{}]".format(epoch))

    len_history = len(train_loader) if not args.debug else 3
    loss_history = np.empty(len_history)
    loss_cls_history = np.empty(len_history)
    loss_align_history = np.empty(len_history)

    source_model = list_trainable[0]

    # switch to train mode
    if args.source_in_eval_mode:
        source_model.eval()
    else:
        source_model.train()
    list_witness_model.eval()

    if args.project_source_embedding:
        source_projection = list_trainable[1]
        source_projection.train()

    for param in list_witness_model.parameters():
        param.requires_grad = False

    if args.noise_type != 'none':
        param = {'ord': np.inf,
                 'epsilon': args.pgd_eps,
                 'alpha': args.pgd_alpha,
                 'num_iter': 10,
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

        p_s = source_model(images+delta)
        p_w = torch.zeros([args.num_witness, p_s.shape[0], p_s.shape[1]], device=device)

        for idx, witness_model in enumerate(list_witness_model):
            with ctx_noparamgrad_and_eval(witness_model):
                p_w[idx, ::] = witness_model(images+delta)

        loss_align = criterion_kd(p_s, p_w)

        loss_cls = criterion_cls(p_s, target)
        loss = args.lambda_kd * loss_align + args.lambda_cls * loss_cls

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(source_model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(source_model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step((i+1)/len(train_loader))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(p_s, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        loss_history[i] = loss.item()
        loss_cls_history[i] = loss_cls.item()
        loss_align_history[i] = loss_align.item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0 and is_main_task) or args.debug:
            progress.display(i + 1)
        if args.debug and i == 2:
            break

    return top1.avg, top5.avg, losses.avg, [loss_history, loss_cls_history, loss_align_history]
