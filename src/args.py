import yaml
import argparse
import os
from src.utils_general import DictWrapper
import distutils.util
    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method",
                        default=argparse.SUPPRESS)
    parser.add_argument("--dataset",
                        default=argparse.SUPPRESS)
    parser.add_argument("--arch",
                        default=argparse.SUPPRESS)
    parser.add_argument("--pretrain",
    			default=argparse.SUPPRESS)
    
    # hyper-param for optimization
    parser.add_argument("--lr",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--lr_scheduler_type",
    			default=argparse.SUPPRESS)
    parser.add_argument("--optim",
    			default='sgd', type=str)
    parser.add_argument("--momentum",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--weight_decay",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--nesterov",
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)

    parser.add_argument("--lr_warmup_type",
    			default=argparse.SUPPRESS)
    parser.add_argument("--lr_warmup_epoch",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--lr_warmup_decay",
    			default=argparse.SUPPRESS, type=float)

    parser.add_argument("--batch_size",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--seed",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--epoch",
    			default=argparse.SUPPRESS, type=int)

    # hyper-param for job_id, and ckpt
    parser.add_argument("--j_dir", required=True)
    parser.add_argument("--j_id",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--ckpt_freq",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--resume_from_ckpt",
    			default=None, type=str)

    # setup wandb logging
    parser.add_argument("--wandb_project",
    			default=argparse.SUPPRESS)
    parser.add_argument('--enable_wandb',
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)


    parser.add_argument('--eval_AA',
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)
    parser.add_argument('--eval_CC',
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)

    parser.add_argument('--input_normalization',
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)
    parser.add_argument('--enable_batchnorm',
                        default=argparse.SUPPRESS, type=distutils.util.strtobool)

    # imagenet training
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')
    parser.add_argument('--debug',
                        default=False, type=distutils.util.strtobool)
    parser.add_argument("--print_freq",
                        default=argparse.SUPPRESS, type=int)

    # added for vit training on imagenet
    parser.add_argument('--amp',
                        default=None, action="store_true")
    parser.add_argument("--label_smoothing",
                        default=0., type=float)
    parser.add_argument('--mixup_alpha',
                        default=0., type=float)
    parser.add_argument('--cutmix_alpha',
                        default=0., type=float)
    parser.add_argument('--clip_grad_norm',
                        default=None)
    parser.add_argument('--auto_augment',
                        default=None, type=str)
    parser.add_argument('--ra_magnitude',
                        default=9, type=int)
    parser.add_argument('--interpolation',
                        default='bilinear', type=str)
    parser.add_argument('--augmix_severity',
                        default=3, type=int)
    parser.add_argument('--random_erase',
                        default=0.0, type=float)

    parser.add_argument('--ra_sampler',
                        default=False, type=distutils.util.strtobool)
    parser.add_argument("--ra_reps",
                        default=3, type=int,
                        help="number of repetitions for Repeated Augmentation (default: 3)")
    parser.add_argument('--use-v2',
                        action="store_true")
    # parser.add_argument("--model_ema", action="store_true",
                        # help="enable tracking Exponential Moving Average of model parameters")
    # parser.add_argument("--model_ema_steps",
                        # default=32, type=int,
                        # help="the number of iterations that controls how often to update the EMA model (default: 32)")
    # parser.add_argument("--model_ema_decay",
                        # default=0.99998, type=float,
                        # help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)")


    # model alignment/misalignment
    parser.add_argument("--source_model",
                        default=argparse.SUPPRESS)
    parser.add_argument("--modified_source_model",
    			default=None, type=str)
    parser.add_argument("--target_model",
                        default=argparse.SUPPRESS)
    parser.add_argument("--source_arch",
                        default=argparse.SUPPRESS)
    parser.add_argument("--source_in_eval_mode",
                        default=False, type=distutils.util.strtobool)
    parser.add_argument("--target_arch",
                        default=argparse.SUPPRESS)
    parser.add_argument("--witness_arch",
                        default=argparse.SUPPRESS)
    parser.add_argument("--num_witness",
                        default=1, type=int)
    parser.add_argument("--noise_type",
                        default='none', type=str)
    parser.add_argument("--save_modified_model",
                        default=False, type=distutils.util.strtobool)
    # the following pgd params are used in whitebox and transfer evaluations
    # if alignment is performed at pgd perturbed datapoint, the same pgd_eps, pgd_alpha
    # will be used, the number of iteration is specified in ${noise_type}
    parser.add_argument("--pgd_itr",
                            default=20, type=int)
    parser.add_argument("--pgd_eps",
                            default=4./255., type=float)
    parser.add_argument("--pgd_alpha",
                            default=1./255., type=float)

    parser.add_argument("--rkd_dist_ratio",
                            default=1., type=float)
    parser.add_argument("--rkd_angle_ratio",
                            default=2., type=float)
    parser.add_argument("--ega_node_weight",
                            default=0.8, type=float)
    parser.add_argument("--ega_edge_weight",
                            default=0.24, type=float)
    parser.add_argument("--hint_weight",
                            default=1., type=float)
    parser.add_argument("--nce_temp",
                            default=0.1, type=float)
    parser.add_argument("--kl_temp",
                            default=4., type=float)
    parser.add_argument("--kl_reduction",
                            default='average', type=str)
    parser.add_argument("--always_proj",
                        default=True, type=distutils.util.strtobool)
    parser.add_argument("--lambda_cls",
                            default=0., type=float)
    parser.add_argument("--lambda_kd",
                            default=1., type=float)

    parser.add_argument('--source_idx', default=0, type=int)
    parser.add_argument('--target_idx', default=1, type=int)


    args = parser.parse_args()

    return args

def make_dir(args):
    _dir = str(args["j_dir"]+"/config/")
    try:
        os.makedirs(_dir)
    except os.error:
        pass

    if not os.path.exists(_dir + "/config.yaml"):
        f = open(_dir + "/config.yaml" ,"w+")
        f.write(yaml.dump(args))
        f.close()

def get_default(yaml_path):
    default = {}
    with open(yaml_path, 'r') as handle:
        default = yaml.load(handle, Loader=yaml.FullLoader)
    return default 

def get_base_model_dir(yaml_path):
    with open(yaml_path, 'r') as file:
        base_model_dir = yaml.safe_load(file)
    return base_model_dir

def get_args():
    args = parse_args()
    if args.dataset.startswith('cifar') or args.dataset == 'svhn':
        default = get_default('options/default_cifar.yaml')
    elif args.dataset == 'imagenet':
        default = get_default('options/default_imagenet.yaml')

    default.update(vars(args).items())
    
    make_dir(default)

    # if args.dataset.startswith('cifar'):
        # args_dict = DictWrapper(default)
        # return args_dict
    if default['clip_grad_norm'] == None:
        pass
    elif default['clip_grad_norm'] in ['none', 'None']:
        default['clip_grad_norm'] = None
    else:
        default['clip_grad_norm'] = float(default['clip_grad_norm'])

    return argparse.Namespace(**default)

def print_args(args):
    print("***********************************************************")
    print("************************ Arguments ************************")
    print("***********************************************************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("***********************************************************")
    print("***********************************************************")
