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
# from src.rkd import RkdDistance, RKdAngle
from distiller_zoo import RKDLoss, EGA, PKT, DistillKL, HintLoss, NCELoss

def rand_init(dataset, x, epsilon = 8./255.):
        # imagenet normalization
    if dataset == 'cifar10':
        mean = torch.tensor([x / 255 for x in [125.3, 123.0, 113.9]], device=x.device)
        std = torch.tensor([x / 255 for x in [63.0, 62.1, 66.7]], device=x.device)
    elif dataset == 'cifar100':
        mean = torch.tensor([x / 255 for x in [129.3, 124.1, 112.4]], device=x.device)
        std = torch.tensor([x / 255 for x in [68.2, 65.4, 70.4]], device=x.device)

    projector = Linf_ball_projection(mean, std)

    delta = torch.rand_like(x, requires_grad=False)
    delta.data[:,0,:,:].mul_(2. * epsilon/std[0]).add_(- epsilon/std[0])
    delta.data[:,1,:,:].mul_(2. * epsilon/std[1]).add_(- epsilon/std[1])
    delta.data[:,2,:,:].mul_(2. * epsilon/std[2]).add_(- epsilon/std[2])
    delta.data = projector(x + delta) - x.data

    return delta


def match_kl(loader, opt, args, source_model, list_witness_model, device):

    total_loss = 0.

    source_model.train()
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    if args.noise_type != 'none':
        param = {'ord': np.inf,
              'epsilon': 8./255.,
              'alpha': 2./255.,
              'num_iter': 20,
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

    with trange(len(loader)) as t:
        for X, y in loader:
            loss = 0
            X, y = X.to(device), y.to(device)

            if args.noise_type != 'none':
                with ctx_noparamgrad_and_eval(source_model):
                    delta_s = attacker.generate(source_model, X, y)
            else:
                delta_s = 0

            p_s = source_model(X+delta_s)
            yp_s = F.log_softmax(p_s, dim=1)

            for witness_model in list_witness_model:
                witness_model.eval()
                with ctx_noparamgrad_and_eval(witness_model):
                    if args.noise_type.endswith('indep'):
                        delta_w = attacker.generate(witness_model, X, y)
                    else:
                        delta_w = delta_s

                    yp_w = F.log_softmax(witness_model(X+delta_w), dim=1)
                    if args.misalign:
                        loss -= (kl_loss(yp_s, yp_w) + kl_loss(yp_w, yp_s))
                    else:
                        loss += (kl_loss(yp_s, yp_w) + kl_loss(yp_w, yp_s))

            loss *= 1/args.num_witness
            if args.ce_regularized:
                loss += nn.CrossEntropyLoss()(p_s, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item())
            t.update()

    total_loss = total_loss / len(loader.dataset)

def match_jacobian(loader, opt, args, source_model, list_witness_model, device):

    total_loss = 0.

    source_model.train()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    _dim = 3*32*32

    if args.noise_type != 'none':
        param = {'ord': np.inf,
              'epsilon': 8./255.,
              'alpha': 2./255.,
              'num_iter': 20,
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

    with trange(len(loader)) as t:
        for X, y in loader:
            loss = 0
            X, y = X.to(device), y.to(device)

            if args.noise_type != 'none':
                with ctx_noparamgrad_and_eval(source_model):
                    delta_s = attacker.generate(source_model, X, y)
            else:
                delta_s = 0

            X_s = (X+delta_s).clone().detach()
            X_s.requires_grad = True

            loss_s = nn.CrossEntropyLoss()(source_model(X_s), y)
            dldx_s = len(X) * grad(loss_s, X_s, create_graph=True)[0].view(-1, _dim)

            for witness_model in list_witness_model:
                witness_model.eval()
                with ctx_noparamgrad_and_eval(witness_model):
                    if args.noise_type.endswith('indep'):
                        delta_w = attacker.generate(witness_model, X, y)
                    else:
                        delta_w = delta_s

                    X_w = (X+delta_w).clone().detach()
                    X_w.requires_grad = True

                    loss_w = nn.CrossEntropyLoss()(witness_model(X_w), y)
                    dldx_w = len(X) * grad(loss_w, X_w, create_graph=False)[0].view(-1, _dim)
                    if args.misalign:
                        loss += cos(dldx_s, dldx_w).mean()
                    else:
                        loss -= cos(dldx_s, dldx_w).mean()

            loss *= 1/args.num_witness
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item())
            t.update()

    total_loss = total_loss / len(loader.dataset)

def match_kl_jacobian(loader, opt, args, source_model, list_witness_model, device):

    total_loss = 0.

    source_model.train()
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    _dim = 3*32*32

    if args.noise_type != 'none':
        param = {'ord': np.inf,
              'epsilon': 8./255.,
              'alpha': 2./255.,
              'num_iter': 20,
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

    with trange(len(loader)) as t:
        for X, y in loader:
            loss = 0
            X, y = X.to(device), y.to(device)

            if args.noise_type != 'none':
                with ctx_noparamgrad_and_eval(source_model):
                    delta_s = attacker.generate(source_model, X, y)
            else:
                delta_s = 0

            X_s = (X+delta_s).clone().detach()
            X_s.requires_grad = True

            p_s = source_model(X_s)
            yp_s = F.log_softmax(p_s, dim=1)
            loss_s = nn.CrossEntropyLoss()(p_s, y)
            dldx_s = len(X) * grad(loss_s, X_s, create_graph=True)[0].view(-1, _dim)

            for witness_model in list_witness_model:
                witness_model.eval()
                with ctx_noparamgrad_and_eval(witness_model):
                    if args.noise_type.endswith('indep'):
                        delta_w = attacker.generate(witness_model, X, y)
                    else:
                        delta_w = delta_s

                    X_w = (X+delta_w).clone().detach()
                    X_w.requires_grad = True

                    p_w = witness_model(X_w)
                    yp_w = F.log_softmax(p_w, dim=1)
                    loss_w = nn.CrossEntropyLoss()(p_w, y)
                    dldx_w = len(X) * grad(loss_w, X_w, create_graph=True)[0].view(-1, _dim)

                    loss += -cos(dldx_s, dldx_w).mean()
                    loss += kl_loss(yp_s, yp_w) + kl_loss(yp_w, yp_s)
                    if args.misalign:
                        loss += cos(dldx_s, dldx_w).mean()
                        loss -= (kl_loss(yp_s, yp_w) + kl_loss(yp_w, yp_s))
                    else:
                        loss -= cos(dldx_s, dldx_w).mean()
                        loss += (kl_loss(yp_s, yp_w) + kl_loss(yp_w, yp_s))

            loss *= 1/args.num_witness
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item())
            t.update()

    total_loss = total_loss / len(loader.dataset)

def model_align(train_loader, source_model, witness_model, optimizer, device, args, is_main_task):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Align Epoch: [{}]".format(0))

    # switch to train mode
    source_model.train()
    witness_model.eval()
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

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

        if args.method == 'kl':
            # compute output
            p_s = source_model(images+delta)
            yp_s = F.log_softmax(p_s, dim=1)

            with ctx_noparamgrad_and_eval(witness_model):
                p_w = witness_model(images+delta)
                yp_w = F.log_softmax(p_w, dim=1)
            loss = kl_loss(yp_s, yp_w) + kl_loss(yp_w, yp_s)

        elif args.method == 'jacobian':
            dim = images.size(1)*images.size(2)*images.size(3)

            images_s = (images+delta).clone().detach()
            images_s.requires_grad=True

            p_s = source_model(images_s)
            loss_s = nn.CrossEntropyLoss()(p_s, target)
            dldx_s = images.size(0) * grad(loss_s, images_s, create_graph=True)[0].view(-1, dim)

            images_w = images_s.clone().detach()
            images_w.requires_grad=True

            with ctx_noparamgrad_and_eval(witness_model):
                p_w = witness_model(images_w)
                loss_w = nn.CrossEntropyLoss()(p_w, target)
                dldx_w = images.size(0) * grad(loss_w, images_w, create_graph=True)[0].view(-1, dim)
            loss = cos(dldx_s, dldx_w).mean()

        elif args.method == 'kl-jacobian':
            dim = images.size(1)*images.size(2)*images.size(3)

            images_s = (images+delta).clone().detach()
            images_s.requires_grad=True

            p_s = source_model(images_s)
            yp_s = F.log_softmax(p_s, dim=1)
            loss_s = nn.CrossEntropyLoss()(p_s, target)
            dldx_s = images.size(0) * grad(loss_s, images_s, create_graph=True)[0].view(-1, dim)

            images_w = images_s.clone().detach()
            images_w.requires_grad=True

            with ctx_noparamgrad_and_eval(witness_model):
                p_w = witness_model(images_w)
                yp_w = F.log_softmax(p_w, dim=1)
                loss_w = nn.CrossEntropyLoss()(p_w, target)
                dldx_w = images.size(0) * grad(loss_w, images_w, create_graph=True)[0].view(-1, dim)
            loss = cos(dldx_s, dldx_w).mean()
            loss += kl_loss(yp_s, yp_w) + kl_loss(yp_w, yp_s)

        loss *= -1 if args.misalign else 1
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
        if args.debug:
            break

    return top1.avg, top5.avg, losses.avg

def model_align_feature_space(train_loader, module_list, criterion_list, optimizer, device, args, is_main_task):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Align Epoch: [{}]".format(0))

    source_model = module_list[0]
    witness_model = module_list[1]
    criterion_cls = criterion_list[0]
    criterion_kd = criterion_list[1]

    # switch to train mode
    source_model.train()
    witness_model.eval()

    if args.project_source_embedding:
        source_projection = module_list[2]
        source_projection.train()

    for param in witness_model.parameters():
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

        if args.method != 'kl':
            features={}
            def get_features(name):
                def hook(model, input, output):
                    features[name] = output
                return hook

            try:
                source_model_hook = source_model.module
            except:
                source_model_hook = source_model

            try:
                witness_model_hook = witness_model.module
            except:
                witness_model_hook = witness_model

            if args.dataset == 'imagenet':
                source_model_hook.avgpool.register_forward_hook(get_features('feat_s'))
                witness_model_hook.avgpool.register_forward_hook(get_features('feat_w'))
            elif args.dataset.startswith('cifar'):
                if args.source_arch.startswith('preactresnet'):
                    source_model_hook.avgpool.register_forward_hook(get_features('feat_s'))
                elif args.source_arch.startswith('vgg'):
                    source_model_hook.module.features.register_forward_hook(get_features('feat_s'))
                elif args.source_arch.startswith('vit'):
                    source_model_hook.to_latent.register_forward_hook(get_features('feat_s'))
                if args.witness_arch.startswith('preactresnet'):
                    witness_model_hook.avgpool.register_forward_hook(get_features('feat_w'))
                elif args.witness_arch.startswith('vgg'):
                    witness_model_hook.features.register_forward_hook(get_features('feat_w'))
                elif args.witness_arch.startswith('vit'):
                    witness_model_hook.to_latent.register_forward_hook(get_features('feat_w'))

        p_s = source_model(images+delta)
        with ctx_noparamgrad_and_eval(witness_model):
            p_w = witness_model(images+delta)

        if args.method != 'kl':
            feat_s = features['feat_s'].view(images.size(0), -1)
            feat_w = features['feat_w'].view(images.size(0), -1)

            if args.project_source_embedding:
                feat_s = source_projection(feat_s)
        else:
            feat_s = p_s
            feat_w = p_w

        if args.method == 'nce':
            loss_kd = criterion_kd(feat_s, feat_w, target)
        else:
            loss_kd = criterion_kd(feat_s, feat_w)

        loss_cls = criterion_cls(p_s, target)
        loss = args.lambda_kd * loss_kd + args.lambda_cls * loss_cls

        loss *= -1 if args.misalign else 1
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
        if args.debug:
            break

    return top1.avg, top5.avg, losses.avg
