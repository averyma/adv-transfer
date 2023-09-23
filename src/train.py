import torch
import torch.nn as nn

from tqdm import trange
import time

from src.utils_log import Summary, AverageMeter, ProgressMeter
from src.evaluation import accuracy

AVOID_ZERO_DIV = 1e-6

def train_standard(loader, model, opt, device, epoch=1, lr_scheduler=None):
    total_loss, total_correct = 0., 0.
    # curr_itr = ep2itr(epoch, loader)
    total_steps = len(loader)
    steps = 0
    with trange(len(loader)) as t:
        for X, y in loader:
            steps+=1
            model.train()
            X, y = X.to(device), y.to(device)

            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)

            batch_correct = (yp.argmax(dim=1) == y).sum().item()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100),
                          lr=opt.param_groups[0]['lr'])
            t.update()

            if lr_scheduler is not None:
                lr_scheduler.step(epoch-1+float(steps)/total_steps)

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)
    return acc, total_loss

def train(train_loader, model, criterion, optimizer, epoch, device, args, is_main_task, scaler, mixup_cutmix):
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

        # mixup-cutmix
        orig_target = target.clone().detach()
        if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
            images, target = mixup_cutmix(images, target)

        # compute output
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, orig_target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and is_main_task:
            progress.display(i + 1)
        if args.debug and i == 2:
            break

    return top1.avg, top5.avg, losses.avg
