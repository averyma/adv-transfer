import torch
import torch.nn as nn
import numpy as np
import ipdb

avoid_zero_div = 1e-12

class pgd(object):
    """ PGD attacks, with random initialization within the specified lp ball """
    def __init__(self, **kwargs):
        # define default attack parameters here:
        self.param = {'ord': np.inf,
                      'epsilon': 8./255.,
                      'alpha': 2./255.,
                      'num_iter': 20,
                      'restarts': 1,
                      'rand_init': True,
                      'clip': True,
                      'loss_fn': nn.CrossEntropyLoss(),
                      'dataset': 'cifar10'}
        # parse thru the dictionary and modify user-specific params
        self.parse_param(**kwargs)

    def generate(self, model, x, y):
        epsilon = self.param['epsilon']
        num_iter = self.param['num_iter']
        alpha = epsilon if num_iter == 1 else self.param['alpha']
        rand_init = self.param['rand_init']
        clip = self.param['clip']
        restarts = self.param['restarts']
        loss_fn = self.param['loss_fn']
        p_norm = self.param['ord']
        dataset = self.param['dataset']

        # implementation begins:
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        max_delta = torch.zeros_like(x)
        _dim = x.shape[1] * x.shape[2] * x.shape[3]

        # imagenet normalization
        if dataset == 'cifar10':
            # mean: [0.4913725490196078, 0.4823529411764706, 0.4466666666666667]
            # std: [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]
            mean = torch.tensor([x / 255 for x in [125.3, 123.0, 113.9]], device=x.device)
            std = torch.tensor([x / 255 for x in [63.0, 62.1, 66.7]], device=x.device)
        elif dataset == 'cifar100':
            mean = torch.tensor([x / 255 for x in [129.3, 124.1, 112.4]], device=x.device)
            std = torch.tensor([x / 255 for x in [68.2, 65.4, 70.4]], device=x.device)
        elif dataset == 'svhn':
            mean = torch.tensor([0.4376821, 0.4437697, 0.47280442], device=x.device)
            std = torch.tensor([0.19803012, 0.20101562, 0.19703614], device=x.device)

        '''
        before normalization:
        x \in [0,1] for all channels
        after normalization:
        x_c should be in [-mu[c]/std[c], (1-mu[c])/std[c]], where c denotes channel
        x_0 in (-1.9889, 2.0587)
        x_1 in (-1.9807, 2.1256)
        x_2 in (-1.7076, 2.1154)

        Linf norm of eps of 8/255 in each channel now becomes: (0.127, 0.128, 0.1199)
        '''
        # instead of [0,1], adjust ball size based on normalization
        projector = Linf_ball_projection(mean, std)

        for i in range(restarts):
            if p_norm == np.inf:

                adjust_eps = torch.ones_like(x)*epsilon
                adjust_eps[:,0,:,:].div_(std[0])
                adjust_eps[:,1,:,:].div_(std[1])
                adjust_eps[:,2,:,:].div_(std[2])

                adjust_alpha = torch.ones_like(x)*alpha
                adjust_alpha[:,0,:,:].div_(std[0])
                adjust_alpha[:,1,:,:].div_(std[1])
                adjust_alpha[:,2,:,:].div_(std[2])

                if rand_init:
                    delta = torch.rand_like(x, requires_grad=True)
                    delta.data[:,0,:,:].mul_(2. * epsilon/std[0]).add_(- epsilon/std[0])
                    delta.data[:,1,:,:].mul_(2. * epsilon/std[1]).add_(- epsilon/std[1])
                    delta.data[:,2,:,:].mul_(2. * epsilon/std[2]).add_(- epsilon/std[2])

                    if clip:
                        delta.data = projector(x + delta) - x.data
                else:
                    delta = torch.zeros_like(x, requires_grad=True)
                for t in range(num_iter):
                    model.zero_grad()
                    loss = loss_fn(model(x + delta), y)
                    loss.backward()
                    # first we need to make sure delta is within the specified lp ball
                    delta.data = (delta+adjust_alpha*delta.grad.detach().sign()).clamp(min=-adjust_eps,
                                                                                       max=adjust_eps)
                    # then we need to make sure x+delta in the next iteration is within the [0,1] range
                    if clip:
                        delta.data = projector(x + delta) - x.data
                    delta.grad.zero_()

            elif p_norm == 2:
                if rand_init:
                    # first we sample a random direction and normalize it
                    delta = torch.rand_like(x, requires_grad = True) # U[0,1]
                    delta.data = delta.data * 2.0 - 1.0 # U[-1,1]
                    delta_norm = torch.norm(delta.detach(), p = 2 , dim = (1,2,3), keepdim = True).clamp(min = avoid_zero_div) # get norm
                    # next, we get a random radius < epsilon
                    rand_radius = torch.rand(x.shape[0], requires_grad = False,device=x.device).view(x.shape[0],1,1,1) * epsilon # get random radius
                    
                    # finally we re-assign delta
                    delta.data = epsilon * delta.data/delta_norm

                    if clip:
                        delta.data = (x.data + delta.data).clamp(min = 0., max = 1.) - x.data
                else:
                    delta = torch.zeros_like(x, requires_grad = True)
                    
                for t in range(num_iter):
                    model.zero_grad()
                    loss = loss_fn(model(x + delta), y)
                    loss.backward()
                    
                    # computing norm of loss gradient wrt input
                    # To avoid zero division, cleverhans uses 1e-12, advertorch uses 1e-6
                    grad_norm = torch.norm(delta.grad.detach().view(-1, _dim), p = 2 , dim = 1).view(x.shape[0],1,1,1).clamp(min = avoid_zero_div)
                    # one step in the direction of normalized gradient (stepsize = alpha)
                    delta.data = delta + alpha*delta.grad.detach()/grad_norm
                    # computing the norm of the new delta term
                    # To avoid zero division, cleverhans uses 1e-12, advertorch uses 1e-6
                    delta_norm = torch.norm(delta.detach().view(-1, _dim), p = 2 , dim = 1).view(x.shape[0],1,1,1).clamp(min = avoid_zero_div)
                    # here the clip factor is used to **clip** to within the norm ball
                    # not to **normalize** onto the surface of the ball
                    factor = torch.min(epsilon/delta_norm, torch.tensor(1., device = x.device ))

                    delta.data = delta.data * factor
                    if clip:
                        delta.data = (x.data + delta.data).clamp(min = 0., max = 1.) - x.data

                    delta.grad.zero_()
            else: 
                error = "Only ord = inf and ord = 2 have been implemented"
                raise NotImplementedError(error)
            
            # added the if condition to cut 1 additional unnecessary foward pass
            if restarts > 1:
                all_loss = nn.CrossEntropyLoss(reduction='none')(model(x+delta),y)
                max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
                max_loss = torch.max(max_loss, all_loss)
            else:
                max_delta = delta.detach()
        return max_delta

        

    def parse_param(self, **kwargs):
        for key,value in kwargs.items():
            if key in self.param:
                self.param[key] = value

    def return_param(self):
        return self.param
    
class InputUnNormalize(torch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputUnNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        _device = x.device
        x_unnormalized = x*self.new_std.to(_device) + self.new_mean.to(_device)
        return x_unnormalized

class InputNormalize(torch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        _device = x.device
        x_normalized = (x-self.new_mean.to(_device))/self.new_std.to(_device)
        return x_normalized

class Linf_ball_projection(torch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(Linf_ball_projection, self).__init__()
        # new_std = new_std[..., None, None]
        # new_mean = new_mean[..., None, None]

        # self.register_buffer("new_mean", new_mean)
        # self.register_buffer("new_std", new_std)

        self.new_lower_bound = [-new_mean[0]/new_std[0],
                                -new_mean[1]/new_std[1],
                                -new_mean[2]/new_std[2]]
        self.new_upper_bound = [(1-new_mean[0])/new_std[0],
                                (1-new_mean[1])/new_std[1],
                                (1-new_mean[2])/new_std[2]]

    def forward(self, x):
        # _device = x.device
        x[:, 0, :, :].clamp_(min=self.new_lower_bound[0], max=self.new_upper_bound[0])
        x[:, 1, :, :].clamp_(min=self.new_lower_bound[1], max=self.new_upper_bound[1])
        x[:, 2, :, :].clamp_(min=self.new_lower_bound[2], max=self.new_upper_bound[2])

        return x
