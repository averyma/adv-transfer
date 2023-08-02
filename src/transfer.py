import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from src.attacks import pgd, Linf_ball_projection
from src.context import ctx_noparamgrad_and_eval
import ipdb
from tqdm import trange
import numpy as np

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

            yp_s = F.log_softmax(source_model(X+delta_s), dim=1)

            for witness_model in list_witness_model:
                witness_model.eval()
                with ctx_noparamgrad_and_eval(witness_model):
                    if args.noise_type.endswith('indep'):
                        delta_w = attacker.generate(witness_model, X, y)
                    else:
                        delta_w = delta_s

                    yp_w = F.log_softmax(witness_model(X+delta_w), dim=1)
                    loss += kl_loss(yp_s, yp_w) + kl_loss(yp_w, yp_s)

            loss *= 1/args.num_witness
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
                    loss += -cos(dldx_s, dldx_w).mean()

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

            loss *= 1/args.num_witness
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item())
            t.update()

    total_loss = total_loss / len(loader.dataset)
