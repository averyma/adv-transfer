import torch
import torch.nn as nn
from src.attacks import pgd, pgd_linbp
from src.context import ctx_noparamgrad_and_eval
import ipdb
from tqdm import trange
from autoattack import AutoAttack
import numpy as np
from src.utils_log import Summary, AverageMeter, ProgressMeter
import time
from torch.utils.data import Subset
import torchattacks

CORRUPTIONS_CIFAR10=['brightness',
                     'gaussian_noise',
                     'saturate',
                     'contrast',
                     'glass_blur',
                     'shot_noise',
                     'defocus_blur',
                     'impulse_noise',
                     'snow',
                     'elastic_transform',
                     'jpeg_compression',
                     'spatter',
                     'fog',
                     'speckle_noise',
                     'frost',
                     'motion_blur',
                     'zoom_blur',
                     'gaussian_blur',
                     'pixelate']
 
CORRUPTIONS_IMAGENET_C=['brightness',
                        'contrast',
                        'defocus_blur',
                        'elastic_transform',
                        'fog',
                        'frost',
                        'gaussian_blur',
                        'gaussian_noise',
                        'glass_blur',
                        'impulse_noise',
                        'jpeg_compression',
                        'motion_blur',
                        'pixelate',
                        'saturate',
                        'shot_noise',
                        'snow',
                        'spatter',
                        'speckle_noise',
                        'zoom_blur'
                       ]

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

def test_clean(loader, model, device):
    total_loss, total_correct = 0., 0.
    total_correct_5 = 0
    for x,y in loader:
        model.eval()
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            y_hat = model(x)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            # batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
            # batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
            batch_acc = accuracy(y_hat, y, topk=(1,5))
            batch_correct = batch_acc[0].sum().item()*x.shape[0]/100
            batch_correct_5 = batch_acc[1].sum().item()*x.shape[0]/100
        # print(accuracy(y_hat, y, topk=(1,5)), batch_correct/128*100)
        
        total_correct += batch_correct
        total_correct_5 += batch_correct_5
        total_loss += loss.item() * x.shape[0]
    # ipdb.set_trace()
    test_acc = total_correct / len(loader.dataset) * 100
    test_acc_5 = total_correct_5 / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss, test_acc_5

def test_gaussian(loader, model, var, device):
    total_loss, total_correct = 0., 0.
    total_correct_5 = 0
    for x,y in loader:
        model.eval()
        x, y = x.to(device), y.to(device)
        
        noise = (var**0.5)*torch.randn_like(x, device = x.device)

        with torch.no_grad():
            y_hat = model(x+noise)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            # batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
            batch_acc = accuracy(y_hat, y, topk=(1,5))
#             ipdb.set_trace()
            batch_correct = batch_acc[0].item()*x.shape[0]/100
            batch_correct_5 = batch_acc[1].item()*x.shape[0]/100
        
        total_correct += batch_correct
        total_correct_5 += batch_correct_5
        total_loss += loss.item() * x.shape[0]
        
    test_acc = total_correct / len(loader.dataset) * 100
    test_acc_5 = total_correct_5 / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss, test_acc_5

def test_transfer(loader, args, source_model, target_model, device):
    total_loss, total_correct = 0., 0.
    total_eval = 0
    total_correct_5 = 0
    param = {'ord': np.inf,
          'epsilon': 8./255.,
          'alpha': 2./255.,
          'num_iter': 20,
          'restarts': 1,
          'rand_init': True,
          'clip': True,
          'loss_fn': nn.CrossEntropyLoss(),
          'dataset': args.dataset}
    attacker = pgd(**param)

    # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    # std = [x / 255 for x in [63.0, 62.1, 66.7]]
    with trange(len(loader)) as t:
        for x,y in loader:
            source_model.eval()
            target_model.eval()
            x, y = x.to(device), y.to(device)

            with ctx_noparamgrad_and_eval(source_model):
                delta = attacker.generate(source_model, x, y)
            x_adv = x+delta

            # for i in range(3):
                # print('channel: {}'.format(i))
                # print('***********before unnormalization************')
                # print('[x.min, x.max]: [{}, {}]'.format(
                    # x[:,i,:,:].min().item(),
                    # x[:,i,:,:].max().item()))
                # print('[delta.min, delta.max]: [{}, {}]'.format(
                    # delta[:,i,:,:].min().item(),
                    # delta[:,i,:,:].max().item()))
                # print('[x_adv.min, x_adv.max]: [{}, {}]'.format(
                    # x_adv[:,i,:,:].min().item(),
                    # x_adv[:,i,:,:].max().item()))
                # print('***********after unnormalization************')
                # print('[x.min, x.max]: [{}, {}]'.format(
                    # (x[:,i,:,:]*std[i]+mean[i]).min().item(),
                    # (x[:,i,:,:]*std[i]+mean[i]).max().item()))
                # print('[delta.min, delta.max]: [{}, {}]'.format(
                    # (delta[:,i,:,:]*std[i]).min().item(),
                    # (delta[:,i,:,:]*std[i]).max().item()))
                # print('[x_adv.min, x_adv.max]: [{}, {}]'.format(
                    # (x_adv[:,i,:,:]*std[i]+mean[i]).min().item(),
                    # (x_adv[:,i,:,:]*std[i]+mean[i]).max().item()))

            with torch.no_grad():
                y_hat = target_model(x_adv)
                loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                batch_acc = accuracy(y_hat, y, topk=(1,5))
                batch_correct = batch_acc[0].item()*x.shape[0]/100
                batch_correct_5 = batch_acc[1].item()*x.shape[0]/100

            total_correct += batch_correct
            total_correct_5 += batch_correct_5
            total_eval += len(x)
            total_loss += loss.item() * x.shape[0]

            t.set_postfix(acc='{0:.2f}%'.format(batch_acc[0].item()))
            t.update()
            # if total_eval > 1000:
                # break

    test_acc = total_correct / total_eval * 100
    test_acc_5 = total_correct_5 / total_eval * 100
    test_loss = total_loss / total_eval

    return test_acc, test_loss, test_acc_5

def test_transfer_multiple_target(loader, args, source_model, list_target_model, device):
    total_loss = np.zeros(len(list_target_model))
    total_correct = np.zeros(len(list_target_model))
    total_correct_5 = np.zeros(len(list_target_model))
    total_eval = 0
    param = {'ord': np.inf,
          'epsilon': 8./255.,
          'alpha': 2./255.,
          'num_iter': 20,
          'restarts': 1,
          'rand_init': True,
          'clip': True,
          'loss_fn': nn.CrossEntropyLoss(),
          'dataset': args.dataset}
    attacker = pgd(**param)

    with trange(len(loader)) as t:
        for x,y in loader:
            source_model.eval()
            x, y = x.to(device), y.to(device)

            with ctx_noparamgrad_and_eval(source_model):
                delta = attacker.generate(source_model, x, y)
                # delta = 0
            x_adv = x+delta
            for i, target_model in enumerate(list_target_model):
                target_model.eval()
                with torch.no_grad():
                    y_hat = target_model(x_adv)
                    loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                    batch_acc = accuracy(y_hat, y, topk=(1,5))
                    batch_correct = batch_acc[0].item()*x.shape[0]/100
                    batch_correct_5 = batch_acc[1].item()*x.shape[0]/100

                total_correct[i] += batch_correct
                total_correct_5[i] += batch_correct_5
                total_loss[i] += loss.item() * x.shape[0]

            total_eval += len(x)
            t.update()
            # if total_eval > 1000:
                # break

    test_acc = total_correct / total_eval * 100
    test_acc_5 = total_correct_5 / total_eval * 100
    test_loss = total_loss / total_eval

    return test_acc, test_loss, test_acc_5

def test_AA(loader, model, norm, eps, attacks_to_run=None, verbose=False):

    assert norm in ['L2', 'Linf']

    adversary = AutoAttack(model, norm=norm, eps=eps, version='standard', verbose=verbose)
    if attacks_to_run is not None:
        adversary.attacks_to_run = attacks_to_run

    lx, ly = [],[]
    for x,y in loader:
        lx.append(x)
        ly.append(y)
    x_test = torch.cat(lx, 0)
    y_test = torch.cat(ly, 0)

    # if attacks_to_run is None and len(y_test)>1000:
    x_test,y_test = x_test[:1000],y_test[:1000]
    bs = 32 if x_test.shape[2] == 224 else 128

    with torch.no_grad():

        result = adversary.run_standard_evaluation_return_robust_accuracy(x_test, y_test,
                                                                          bs=bs, return_perturb=True)
        total_correct = 0
        total_correct_5 = 0
        for i in range(20):
            x_adv = result[1][i*50:(i+1)*50,:].to('cuda')
            y_adv = y_test[i*50:(i+1)*50].to('cuda')
            y_hat = model(x_adv)
            batch_acc = accuracy(y_hat, y_adv,topk=(1,5))
            batch_correct = batch_acc[0].sum().item()*50/100
            batch_correct_5 = batch_acc[1].sum().item()*50/100
            total_correct += batch_correct
            total_correct_5 += batch_correct_5
        test_acc = total_correct / 1000 * 100
        test_acc_5 = total_correct_5 / 1000 * 100
    return test_acc, test_acc_5 

def eval_corrupt(model, dataset, severity, device):
    # dct_matrix = getDCTmatrix(28)
    acc = []
    acc_5 = []
    total_correct = 0
    total_correct_5 = 0
    
    corruptions_list = CORRUPTIONS_CIFAR10
    
    for _corrupt, corrupt_type in enumerate(corruptions_list):
        print("{0:.2f}%, {1}".format(_corrupt/len(corruptions_list)*100, corrupt_type))
        
        if dataset == 'cifar10':
            data_path = '/h/ama/workspace/ama-at-vector/freq-robust/data/CIFAR-10-C/' + corrupt_type + '.npy'
            label_path = '/h/ama/workspace/ama-at-vector/freq-robust/data/CIFAR-10-C/labels.npy'
            x = torch.tensor(np.transpose(np.load(data_path), (0, 3, 1, 2))/255.,dtype= torch.float32)[(severity-1)*10000:severity*10000].to(device)
            y = torch.tensor(np.load(label_path),dtype=torch.float32)[(severity-1)*10000:severity*10000].to(device)
            
            # mean = [0.49139968, 0.48215841, 0.44653091]
            # std = [0.24703223, 0.24348513, 0.26158784]
            # x[:,0,:,:] = (x[:,0,:,:]-mean[0])/std[0]
            # x[:,1,:,:] = (x[:,1,:,:]-mean[1])/std[1]
            # x[:,2,:,:] = (x[:,2,:,:]-mean[2])/std[2]

            for i in range(10):
                with torch.no_grad():
                    y_hat = model(x[1000*i:1000*(i+1)])
                    # total_correct += (y_hat.argmax(dim = 1) == y[1000*i:1000*(i+1)]).sum().item()
                    batch_acc = accuracy(y_hat, y[1000*i:1000*(i+1)], topk=(1,5))
                    total_correct += batch_acc[0].sum().item()*1000/100
                    total_correct_5 += batch_acc[1].sum().item()*1000/100

        elif dataset == 'cifar100':
            data_path = '/h/ama/workspace/ama-at-vector/freq-robust/data/CIFAR-100-C/' + corrupt_type + '.npy'
            label_path = '/h/ama/workspace/ama-at-vector/freq-robust/data/CIFAR-100-C/labels.npy'
            x = torch.tensor(np.transpose(np.load(data_path), (0, 3, 1, 2))/255.,dtype= torch.float32)[(severity-1)*10000:severity*10000].to(device)
            y = torch.tensor(np.load(label_path),dtype=torch.float32)[(severity-1)*10000:severity*10000].to(device)
            
            # mean = [0.50707516, 0.48654887, 0.44091784]
            # std = [0.26733429, 0.25643846, 0.27615047]
            # x[:,0,:,:] = (x[:,0,:,:]-mean[0])/std[0]
            # x[:,1,:,:] = (x[:,1,:,:]-mean[1])/std[1]
            # x[:,2,:,:] = (x[:,2,:,:]-mean[2])/std[2]

            for i in range(10):
                with torch.no_grad():
                    y_hat = model(x[1000*i:1000*(i+1)])
                    # total_correct += (y_hat.argmax(dim = 1) == y[1000*i:1000*(i+1)]).sum().item()
                    batch_acc = accuracy(y_hat, y[1000*i:1000*(i+1)], topk=(1,5))
                    total_correct += batch_acc[0].sum().item()*1000/100
                    total_correct_5 += batch_acc[1].sum().item()*1000/100

        corrupt_acc = total_correct / len(y) * 100
        corrupt_acc_5 = total_correct_5 / len(y) * 100
        acc.append(corrupt_acc)
        acc_5.append(corrupt_acc_5)
        total_correct = 0
        total_correct_5 = 0
        
        del x,y 
    
    return acc, acc_5

def eval_CE(base_acc, f_acc):
    
    mCE = []
    rel_mCE = []
    for i in range(1,16):
        mCE.append((100-f_acc[i])/(100-base_acc[i]))
        rel_mCE.append(((100-f_acc[i])-(100-f_acc[0]))/((100-base_acc[i])-(100-base_acc[0])))

    return np.array(mCE).mean(), np.array(rel_mCE).mean()


def eval_transfer(val_loader, source_model, target_model, args, is_main_task):
    param = {'ord': np.inf,
             'epsilon': args.pgd_eps,
             'alpha': args.pgd_alpha,
             'num_iter': args.pgd_itr,
             'restarts': 1,
             'rand_init': True,
             'clip': True,
             'loss_fn': nn.CrossEntropyLoss(),
             'dataset': args.dataset}
    param['num_iter'] = 1 if args.debug else args.pgd_itr
    attacker = pgd(**param)
    num_eval = 100 if args.debug else 1000

    def run_validate_one_epoch(images, target, base_progress=0):
        end = time.time()
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
            qualified = return_qualified(p_s, p_t, p_adv_s, p_adv_t, target)

            p_transfer = target_model((images+delta_s)[qualified, ::])

        # measure accuracy and record loss
        num_qualified = qualified.sum().item()
        acc1, acc5 = accuracy(p_transfer, target[qualified], topk=(1, 5))
        top1.update(acc1[0], num_qualified)
        top5.update(acc5[0], num_qualified)
        total_qualified.update(num_qualified)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    total_qualified = AverageMeter('Total Qualified', ':6.2f', Summary.SUM)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, top1, top5, total_qualified],
        prefix='Transfer: ')

    # switch to evaluate mode
    source_model.eval()
    target_model.eval()

    for i, (images, target) in enumerate(val_loader):
        run_validate_one_epoch(images, target)

        if is_main_task:
            progress.display(i + 1)

        if args.distributed:
            total_qualified.all_reduce()

        if total_qualified.sum > (num_eval/args.ngpus_per_node):
            break

    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    # if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        # aux_val_dataset = Subset(val_loader.dataset,
                                 # range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        # aux_val_loader = torch.utils.data.DataLoader(
            # aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            # num_workers=args.workers, pin_memory=True)
        # run_validate(aux_val_loader, len(val_loader))

    if is_main_task:
        progress.display_summary()

    return top1.avg, top5.avg

def validate(val_loader, model, criterion, args, is_main_task, whitebox=False):
    if whitebox:
        if args.dataset == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif args.dataset == 'cifar10':
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
        elif args.dataset == 'cifar100':
            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]
        # param = {'ord': np.inf,
              # 'epsilon': args.pgd_eps,
              # 'alpha': args.pgd_alpha,
              # 'num_iter': args.pgd_itr,
              # 'restarts': 1,
              # 'rand_init': True,
              # 'clip': True,
              # 'loss_fn': nn.CrossEntropyLoss(),
              # 'dataset': args.dataset}
        # param['num_iter'] = 1 if args.debug else args.pgd_itr
        # attacker = pgd(**param)
        atk = torchattacks.PGD(
            model,
            eps=args.pgd_eps,
            alpha=args.pgd_alpha,
            steps=1 if args.debug else args.pgd_itr,
            random_start=True)
        atk.set_normalization_used(mean=mean, std=std)

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
                    # delta = attacker.generate(model, images, target)
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

def eval_transfer_bi_direction(val_loader, model_a, model_b, args, is_main_task):
    param = {'ord': np.inf,
             'epsilon': args.pgd_eps,
             'alpha': args.pgd_alpha,
             'num_iter': args.pgd_itr,
             'restarts': 1,
             'rand_init': True,
             'clip': True,
             'loss_fn': nn.CrossEntropyLoss(),
             'dataset': args.dataset}
    param['num_iter'] = 1 if args.debug else args.pgd_itr
    attacker = pgd(**param)
    num_eval = 100 if args.debug else 1000

    def run_validate_one_iteration(images, target, base_progress=0):
        end = time.time()
        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.backends.mps.is_available():
            images = images.to('mps')
            target = target.to('mps')
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        with ctx_noparamgrad_and_eval(model_a):
            delta_a = attacker.generate(model_a, images, target)

        with ctx_noparamgrad_and_eval(model_b):
            delta_b = attacker.generate(model_b, images, target)

        # compute output
        with torch.no_grad():
            p_a = model_a(images)
            p_b = model_b(images)
            p_adv_a = model_a(images+delta_a)
            p_adv_b = model_b(images+delta_b)
            qualified = return_qualified(p_a, p_b, p_adv_a, p_adv_b, target)

            p_b2a = model_a((images+delta_b)[qualified, ::])
            p_a2b = model_b((images+delta_a)[qualified, ::])

        # measure accuracy and record loss
        num_qualified = qualified.sum().item()
        acc1_b2a, acc5_b2a = accuracy(p_b2a, target[qualified], topk=(1, 5))
        acc1_a2b, acc5_a2b = accuracy(p_a2b, target[qualified], topk=(1, 5))
        top1_b2a.update(acc1_b2a[0], num_qualified)
        top5_b2a.update(acc5_b2a[0], num_qualified)
        top1_a2b.update(acc1_a2b[0], num_qualified)
        top5_a2b.update(acc5_a2b[0], num_qualified)
        total_qualified.update(num_qualified)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1_b2a = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top1_a2b = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5_b2a = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    top5_a2b = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    total_qualified = AverageMeter('Total Qualified', ':6.2f', Summary.SUM)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, top1_a2b, top5_a2b, top1_b2a, top5_b2a, total_qualified],
        prefix='Transfer: ')

    # switch to evaluate mode
    model_a.eval()
    model_b.eval()

    for i, (images, target) in enumerate(val_loader):
        run_validate_one_iteration(images, target)

        if is_main_task:
            progress.display(i + 1)

        if args.distributed:
            total_qualified.all_reduce()

        if total_qualified.sum > (num_eval/args.ngpus_per_node):
            break

    if args.distributed:
        top1_b2a.all_reduce()
        top1_a2b.all_reduce()
        top5_b2a.all_reduce()
        top5_a2b.all_reduce()

    if is_main_task:
        progress.display_summary()

    return top1_a2b.avg, top1_b2a.avg

def eval_transfer_bi_direction_two_metric(val_loader, model_a, model_b, args, is_main_task):
    if args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]

    # param = {'ord': np.inf,
             # 'epsilon': args.pgd_eps,
             # 'alpha': args.pgd_alpha,
             # 'num_iter': args.pgd_itr,
             # 'restarts': 1,
             # 'rand_init': True,
             # 'clip': True,
             # 'loss_fn': nn.CrossEntropyLoss(),
             # 'dataset': args.dataset}
    # param['num_iter'] = 1 if args.debug else args.pgd_itr
    # attacker = pgd(**param)
    atk_a = torchattacks.PGD(
        model_a,
        eps=args.pgd_eps,
        alpha=args.pgd_alpha,
        steps=1 if args.debug else args.pgd_itr,
        random_start=True)
    atk_a.set_normalization_used(mean=mean, std=std)

    atk_b = torchattacks.PGD(
        model_b,
        eps=args.pgd_eps,
        alpha=args.pgd_alpha,
        steps=1 if args.debug else args.pgd_itr,
        random_start=True)
    atk_b.set_normalization_used(mean=mean, std=std)

    num_eval = 1000 if args.dataset == 'imagenet' else 10000
    num_eval = 100 if args.debug else num_eval

    def run_validate_one_iteration(images, target):
        end = time.time()
        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.backends.mps.is_available():
            images = images.to('mps')
            target = target.to('mps')
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        with ctx_noparamgrad_and_eval(model_a):
            # delta_a = attacker.generate(model_a, images, target)
            delta_a = atk_a(images, target) - images

        with ctx_noparamgrad_and_eval(model_b):
            # delta_b = attacker.generate(model_b, images, target)
            delta_b = atk_b(images, target) - images

        # compute output
        with torch.no_grad():
            p_a = model_a(images)
            p_b = model_b(images)
            p_adv_a = model_a(images+delta_a)
            p_adv_b = model_b(images+delta_b)
            qualified = return_qualified(p_a, p_b, p_adv_a, p_adv_b, target)

            p_b2a_NS = model_a((images+delta_b))
            p_a2b_NS = model_b((images+delta_a))

        # measure accuracy and record loss
        num_qualified = qualified.sum().item()
        p_b2a = p_b2a_NS[qualified, ::]
        p_a2b = p_a2b_NS[qualified, ::]

        acc1_b2a, acc5_b2a = accuracy(p_b2a, target[qualified], topk=(1, 5))
        acc1_a2b, acc5_a2b = accuracy(p_a2b, target[qualified], topk=(1, 5))

        top1_b2a.update(acc1_b2a[0], num_qualified)
        top5_b2a.update(acc5_b2a[0], num_qualified)
        top1_a2b.update(acc1_a2b[0], num_qualified)
        top5_a2b.update(acc5_a2b[0], num_qualified)
        total_qualified.update(num_qualified)

        acc1_b2a_NS, acc5_b2a_NS = accuracy(p_b2a_NS, target, topk=(1, 5))
        acc1_a2b_NS, acc5_a2b_NS = accuracy(p_a2b_NS, target, topk=(1, 5))
        top1_b2a_NS.update(acc1_b2a_NS[0], images.size(0))
        top5_b2a_NS.update(acc5_b2a_NS[0], images.size(0))
        top1_a2b_NS.update(acc1_a2b_NS[0], images.size(0))
        top5_a2b_NS.update(acc5_a2b_NS[0], images.size(0))
        total_eval.update(images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1_b2a = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top1_a2b = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5_b2a = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    top5_a2b = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    top1_b2a_NS = AverageMeter('Acc@1(NS)', ':6.2f', Summary.AVERAGE)
    top1_a2b_NS = AverageMeter('Acc@1(NS)', ':6.2f', Summary.AVERAGE)
    top5_b2a_NS = AverageMeter('Acc@5(NS)', ':6.2f', Summary.AVERAGE)
    top5_a2b_NS = AverageMeter('Acc@5(NS)', ':6.2f', Summary.AVERAGE)
    total_qualified = AverageMeter('Qualified', ':6.2f', Summary.SUM)
    total_eval = AverageMeter('Evaluated(NS)', ':6.2f', Summary.SUM)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, top1_a2b, top1_a2b_NS, top1_b2a, top1_b2a_NS, total_qualified, total_eval],
        prefix='Transfer: ')

    # switch to evaluate mode
    model_a.eval()
    model_b.eval()

    for i, (images, target) in enumerate(val_loader):
        run_validate_one_iteration(images, target)

        if (i % args.print_freq == 0 and is_main_task) or args.debug:
            progress.display(i + 1)

        if args.distributed:
            total_qualified.all_reduce()

        if total_qualified.sum > (num_eval/args.ngpus_per_node):
            break

    if args.distributed:
        top1_b2a.all_reduce()
        top1_a2b.all_reduce()
        top5_b2a.all_reduce()
        top5_a2b.all_reduce()
        top1_b2a_NS.all_reduce()
        top1_a2b_NS.all_reduce()
        top5_b2a_NS.all_reduce()
        top5_a2b_NS.all_reduce()

    if is_main_task:
        progress.display_summary()

    return top1_a2b.avg, top1_b2a.avg, top1_a2b_NS.avg, top1_b2a_NS.avg

def eval_transfer_orthogonal(val_loader, model_a, model_b, args, atk_method, is_main_task):
    if args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]

    if args.debug:
        steps = 1
        eps = 4/255
        alpha = 1/255
        num_eval = 100
    else:
        if atk_method.endswith('strong'):
            steps = 40
            eps = 8/255
            alpha = 2/255
        else:
            steps = 20
            eps = 4/255
            alpha = 1/255
        num_eval = 1000

    if atk_method.startswith('linbp'):
        param = {'ord': np.inf,
                 'epsilon': eps,
                 'alpha': alpha,
                 'num_iter': steps,
                 'restarts': 1,
                 'rand_init': True,
                 'clip': True,
                 'loss_fn': nn.CrossEntropyLoss(),
                 'dataset': 'imagenet'}
        attacker = pgd_linbp(**param)
    else:
        if atk_method.startswith('pgd'):
            atk_a = torchattacks.PGD(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps,
                random_start=True)
            atk_b = torchattacks.PGD(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps,
                random_start=True)
        elif atk_method.startswith('mi'):
            atk_a = torchattacks.MIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)
            atk_b = torchattacks.MIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        elif atk_method.startswith('ni'):
            atk_a = torchattacks.NIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)
            atk_b = torchattacks.NIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        elif atk_method.startswith('vni'):
            atk_a = torchattacks.VNIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)
            atk_b = torchattacks.VNIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        elif atk_method.startswith('vmi'):
            atk_a = torchattacks.VMIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)
            atk_b = torchattacks.VMIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        elif atk_method.startswith('sini'):
            atk_a = torchattacks.SINIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)
            atk_b = torchattacks.SINIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        elif atk_method.startswith('ti'):
            atk_a = torchattacks.TIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)
            atk_b = torchattacks.TIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        elif atk_method.startswith('di'):
            atk_a = torchattacks.DIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)
            atk_b = torchattacks.DIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        atk_a.set_normalization_used(mean=mean, std=std)
        atk_b.set_normalization_used(mean=mean, std=std)

    def run_validate_one_iteration(images, target):
        end = time.time()
        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.backends.mps.is_available():
            images = images.to('mps')
            target = target.to('mps')
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        with ctx_noparamgrad_and_eval(model_a):
            if atk_method.startswith('linbp'):
                delta_a = attacker.generate(model_a, images, target)
            else:
                delta_a = atk_a(images, target) - images

        with ctx_noparamgrad_and_eval(model_b):
            if atk_method.startswith('linbp'):
                delta_b = attacker.generate(model_b, images, target)
            else:
                delta_b = atk_b(images, target) - images

        # compute output
        with torch.no_grad():
            p_a = model_a(images)
            p_b = model_b(images)
            p_adv_a = model_a(images+delta_a)
            p_adv_b = model_b(images+delta_b)
            qualified = return_qualified(p_a, p_b, p_adv_a, p_adv_b, target)

        # measure accuracy and record loss
        num_qualified = qualified.sum().item()
        p_b2a = model_a((images+delta_b)[qualified, ::])
        p_a2b = model_b((images+delta_a)[qualified, ::])

        acc1_b2a, acc5_b2a = accuracy(p_b2a, target[qualified], topk=(1, 5))
        acc1_a2b, acc5_a2b = accuracy(p_a2b, target[qualified], topk=(1, 5))

        top1_b2a.update(acc1_b2a[0], num_qualified)
        top5_b2a.update(acc5_b2a[0], num_qualified)
        top1_a2b.update(acc1_a2b[0], num_qualified)
        top5_a2b.update(acc5_a2b[0], num_qualified)
        total_qualified.update(num_qualified)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1_b2a = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top1_a2b = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5_b2a = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    top5_a2b = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    total_qualified = AverageMeter('Qualified', ':6.2f', Summary.SUM)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, top1_a2b, top1_b2a, total_qualified],
        prefix='Transfer: ')

    # switch to evaluate mode
    model_a.eval()
    model_b.eval()

    for i, (images, target) in enumerate(val_loader):
        run_validate_one_iteration(images, target)

        if (i % args.print_freq == 0 and is_main_task) or args.debug:
            progress.display(i + 1)

        if args.distributed:
            total_qualified.all_reduce()

        if total_qualified.sum > (num_eval/args.ngpus_per_node):
            break

    if args.distributed:
        top1_b2a.all_reduce()
        top1_a2b.all_reduce()
        top5_b2a.all_reduce()
        top5_a2b.all_reduce()

    if is_main_task:
        progress.display_summary()

    return top1_b2a.avg

