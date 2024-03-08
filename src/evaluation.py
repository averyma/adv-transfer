import torch
import torch.nn as nn
from src.attacks import pgd, pgd_linbp, pgd_ensemble
from src.context import ctx_noparamgrad_and_eval
import ipdb
from tqdm import trange
from autoattack import AutoAttack
import numpy as np
from src.utils_log import Summary, AverageMeter, ProgressMeter
import time
from torch.utils.data import Subset
import torchattacks
from models.ensemble import EnsembleTwo, EnsembleFour

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

def return_qualified_ensemble(p_clean, p_adv, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    '''
    A given input is qualified if it can be correctly classified by all models
    AND its perturbed version needs to be incorrectly classified by the originating model
    '''
    qualified = torch.ones(p_clean.size(0), device=p_clean.device)
    with torch.no_grad():
        for i in range(p_clean.size(2)):
            pred_clean = p_clean[:, :, i].topk(1, 1, True, True)[1].t()
            pred_adv = p_adv[:, :, i].topk(1, 1, True, True)[1].t()
            correct = pred_clean.eq(target.view(1, -1).expand_as(pred_clean)).squeeze()
            incorrect = pred_adv.ne(target.view(1, -1).expand_as(pred_clean)).squeeze()
            qualified *= correct
            qualified *= incorrect
    return qualified == 1.

def validate(val_loader, model, criterion, args, is_main_task, whitebox=False):
    if whitebox:
        atk = get_attack(args.dataset, args.atk, model, args.pgd_eps, args.pgd_alpha, args.pgd_itr if not args.debug else 1)

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
                    delta = atk(images, target) - images
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
            if args.debug:
                break

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ' if not whitebox else 'Whitebox: ')

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

def get_attack(dataset, atk_method, model, eps, alpha, steps, random=True):
    if dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]

    if atk_method == 'pgd':
        attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=random)
    elif atk_method == 'mi':
        attack = torchattacks.MIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'ni':
        attack = torchattacks.NIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'vni':
        attack = torchattacks.VNIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'vmi':
        attack = torchattacks.VMIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'sini':
        attack = torchattacks.SINIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'ti':
        attack = torchattacks.TIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'di':
        attack = torchattacks.DIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'linbp':
        param = {'ord': np.inf, 'epsilon': eps, 'alpha': alpha, 'num_iter': steps,
                 'restarts': 1, 'rand_init': True, 'clip': True, 'loss_fn': nn.CrossEntropyLoss(),
                 'dataset': 'imagenet'}
        attack = pgd_linbp(**param)

    if atk_method != 'linbp':
        attack.set_normalization_used(mean=mean, std=std)
        attack.set_normalization_used(mean=mean, std=std)
    return attack

def eval_transfer(val_loader, source_model, target_model, args, is_main_task):
    # Define total number of qualified samples to be evaluated
    num_eval = 100 if args.debug else 1000

    if args.atk == 'linbp':
        atk = get_attack(args.dataset, args.atk, source_model, args.pgd_eps, args.pgd_alpha, args.pgd_itr if not args.debug else 1)
    else:
        atk_source = get_attack(args.dataset, args.atk, source_model, args.pgd_eps, args.pgd_alpha, args.pgd_itr if not args.debug else 1)

        atk_target = get_attack(args.dataset, args.atk, target_model, args.pgd_eps, args.pgd_alpha, args.pgd_itr if not args.debug else 1)

    def run_validate_one_iteration(images, target):
        end = time.time()
        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.backends.mps.is_available():
            images = images.to('mps')
            target = target.to('mps')
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        with ctx_noparamgrad_and_eval(source_model) and ctx_noparamgrad_and_eval(target_model):
            if args.atk == 'linbp':
                delta_source = atk.generate(source_model, images, target)
                delta_target = atk.generate(target_model, images, target)
            else:
                delta_source = atk_source(images, target) - images
                delta_target = atk_target(images, target) - images

        # compute output
        with torch.no_grad():
            p_source = source_model(images)
            p_target = target_model(images)
            p_adv_source = source_model(images+delta_source)
            p_adv_target = target_model(images+delta_target)
            qualified = return_qualified(p_source, p_target, p_adv_source, p_adv_target, target)

        # measure accuracy and record loss
        num_qualified = qualified.sum().item()
        p_source2target = target_model((images+delta_source)[qualified, ::])

        acc1, acc5 = accuracy(p_source2target, target[qualified], topk=(1, 5))

        top1.update(acc1[0], num_qualified)
        top5.update(acc5[0], num_qualified)
        total_qualified.update(num_qualified)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    total_qualified = AverageMeter('Qualified', ':6.2f', Summary.SUM)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, top1, top5, total_qualified],
        prefix='Transfer({}): '.format(args.atk))

    # switch to evaluate mode
    source_model.eval()
    target_model.eval()

    for i, (images, target) in enumerate(val_loader):
        run_validate_one_iteration(images, target)

        if (i % args.print_freq == 0 and is_main_task) or args.debug:
            progress.display(i + 1)

        if args.distributed:
            total_qualified.all_reduce()

        if total_qualified.sum > (num_eval/args.ngpus_per_node):
            break

    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if is_main_task:
        progress.display_summary()

    return top1.avg

def eval_transfer_ensemble(val_loader, source_ensemble, target_model, args, is_main_task):
    '''
    Within entire imagenet test data, evaluation based on correctly classified samples,
    and the adv perturbations must lead to misclassification for their originating models.
    We will exhaust all testset images to find the qualified examples, it should be closer
    to 1000 compared to v1 and v2.
    '''
    num_eval = 100 if args.debug else 1000

    if len(source_ensemble) == 2:
        ensemble_model = EnsembleTwo(source_ensemble[0], source_ensemble[1])
    else:
        ensemble_model = EnsembleFour(source_ensemble[0], source_ensemble[1], source_ensemble[2], source_ensemble[3])

    atk_ensemble = get_attack(args.dataset, args.atk, ensemble_model, args.pgd_eps, args.pgd_alpha, args.pgd_itr if not args.debug else 1)

    atk_target = get_attack(args.dataset, args.atk, target_model, args.pgd_eps, args.pgd_alpha, args.pgd_itr if not args.debug else 1)

    def run_validate_one_iteration(images, target):
        end = time.time()
        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.backends.mps.is_available():
            images = images.to('mps')
            target = target.to('mps')
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)


        # compute output
        with torch.no_grad():
            p_clean = target_model(images).unsqueeze(2)

        with ctx_noparamgrad_and_eval(target_model):
            delta_target = atk_target(images, target) - images
        p_adv = target_model(images+delta_target).unsqueeze(2)

        for model in source_ensemble:
            with torch.no_grad():
                p_clean = torch.cat([p_clean, model(images).unsqueeze(2)], dim=2)
            with ctx_noparamgrad_and_eval(model):
                atk_model = get_attack(args.dataset, args.atk, model, args.pgd_eps, args.pgd_alpha, args.pgd_itr if not args.debug else 1)
                delta = atk_model(images, target) - images
            p_adv = torch.cat([p_adv, model(images+delta).unsqueeze(2)], dim=2)

        qualified = return_qualified_ensemble(p_clean, p_adv, target)

        images, target = images[qualified, ::], target[qualified]

        with ctx_noparamgrad_and_eval(ensemble_model):
            delta_ensemble = atk_ensemble(images, target) - images

        # measure accuracy and record loss
        num_qualified = qualified.sum().item()
        p_b2a = target_model(images + delta_ensemble)

        acc1_b2a, acc5_b2a = accuracy(p_b2a, target, topk=(1, 5))

        top1.update(acc1_b2a[0], num_qualified)
        total_qualified.update(num_qualified)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    total_qualified = AverageMeter('Qualified', ':6.2f', Summary.SUM)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, top1, total_qualified],
        prefix='Transfer: ')

    # switch to evaluate mode
    target_model.eval()
    source_ensemble.eval()

    for i, (images, target) in enumerate(val_loader):
        run_validate_one_iteration(images, target)

        if (i % args.print_freq == 0 and is_main_task) or args.debug:
            progress.display(i + 1)

        if args.distributed:
            total_qualified.all_reduce()

        if total_qualified.sum > (1000/args.ngpus_per_node):
            break

        if args.debug:
            break

    if args.distributed:
        top1.all_reduce()

    if is_main_task:
        progress.display_summary()

    return top1.avg

