import torchvision
import torch.nn as nn
import torch
import os
from src.utils_general import remove_module
from src.utils_dataset import load_dataset, load_imagenet_test_shuffle, load_imagenet_test_1k
import torchattacks
from src.context import ctx_noparamgrad_and_eval
import numpy as np
from src.utils_log import Summary, AverageMeter, ProgressMeter
from torch.autograd import grad
import torchattacks

def measure_smoothness(val_loader, noise_type, model, isVit, device):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if noise_type != None:
        if noise_type == 'rand':
            steps=0
        elif noise_type == 'pgd10':
            steps=10
        elif noise_type == 'pgd20':
            steps=20
        
        atk = torchattacks.PGD(model,eps=4/255,alpha=1/255,steps=steps,random_start=True)
        atk.set_normalization_used(mean=mean, std=std)
        
    def run_validate(loader, base_progress=0):
        for i, (images, target) in enumerate(loader):
            i = base_progress + i
            images = images.cuda(device)
            target = target.cuda(device)
            
            if noise_type != None:
                with ctx_noparamgrad_and_eval(model):
                    images = atk(images, target)
                
            images.requires_grad = True
            # compute output
            # with torch.no_grad():
            output = model(images)
            loss = nn.CrossEntropyLoss()(output, target)

            batch_lambda_max, batch_grad_norm = returnGradNorm_maxEigenH(images, loss, isVit, device)

            # measure accuracy and record loss
            lambda_max.update(batch_lambda_max.mean().item(), images.size(0))
            grad_norm.update(batch_grad_norm.mean().item(), images.size(0))
            
            if i%10==0:
                progress.display(i + 1)

    lambda_max = AverageMeter('max lambda', ':.4e', Summary.AVERAGE)
    grad_norm = AverageMeter('grad norm', ':.4e', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [grad_norm, lambda_max],
        prefix='Smooth: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    return grad_norm.avg, lambda_max.avg

def returnOrthogonal(vector):
    vector_flatten = vector.flatten()/torch.norm(vector, p=2)
    
    rand_flatten = torch.rand_like(vector_flatten)
    rand_flatten /= torch.norm(rand_flatten, p=2)
    
    projection = torch.matmul(vector_flatten, rand_flatten) * vector_flatten
    
    rand_flatten -= projection
#     print('Innerproduct: {}'.format(torch.matmul(rand_flatten,vector_flatten)))
    rand = rand_flatten.unflatten(dim=0, sizes=vector.size())
    
    return rand


def returnGradNorm_maxEigenH(X, loss, isVit,device):
    '''
    Computes maximum eigenvalue of input hessian matrix using power iteration
    '''
    _dim = 3*224*224 
    input_grad = grad(loss, X, create_graph=True)
    r_k = torch.ones_like(input_grad[0])
    if not isVit:
        dldx = len(X) * list(input_grad)[0]
        flatten_dldx = dldx.view(-1)

        x_k = torch.rand_like(X.view(-1).detach(),device = device)
        x_next = torch.rand_like(x_k.detach(), requires_grad = True, device = device)

        i = 0
        diff_norm = torch.tensor(100)
        while torch.all(diff_norm >= 10e-6) and i < 10:
            # end = time.time()
            i = i+1
            hvp_k = grad([flatten_dldx @ x_k], X, allow_unused=True, retain_graph=True)[0]
            hvp_k_norm = torch.norm(hvp_k.detach(), p = 2, dim = (1,2,3), keepdim = True).clamp(min = 1e-12)

            x_next.data = (hvp_k / hvp_k_norm).view(-1).detach()

            hvp_next = grad([flatten_dldx @ x_next], X, create_graph = True, allow_unused=True, retain_graph=True)[0]
            r_k = torch.matmul(x_next.view(len(X),1,_dim), hvp_next.view(len(X),_dim,1))

            diff = hvp_k - r_k.view(len(X),1,1,1) * x_k.view(len(X), X.shape[1], X.shape[2], X.shape[3])
            diff_norm = torch.norm(diff.detach(), p =2, dim = (1,2,3))
            x_k.data = x_next.view(-1).detach()
            # print('[{:6.2f}s], i: {}, threshold: {:6.3f}'.format(time.time()-end, i, diff_norm.mean().item()))
    
    grad_norm = torch.norm(input_grad[0],p=2,dim=[1,2,3])
    return r_k, grad_norm