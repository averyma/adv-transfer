import torch
import torch.nn as nn
from src.attacks import pgd
from src.context import ctx_noparamgrad_and_eval
import ipdb
from tqdm import trange
from autoattack import AutoAttack
import numpy as np

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

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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

