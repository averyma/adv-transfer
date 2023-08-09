import os
import sys
import logging

import torch
import numpy as np

from src.attacks import pgd
from src.train import train_standard
from src.evaluation import test_clean, test_AA, eval_corrupt, eval_CE, test_gaussian, CORRUPTIONS_CIFAR10, test_transfer, test_transfer_multiple_target
from src.args import get_args, print_args
from src.utils_dataset import load_dataset
from src.utils_log import metaLogger, rotateCheckpoint, wandbLogger, saveModel, delCheckpoint
from src.utils_general import seed_everything, get_model, get_optim
import ipdb, pdb
import copy, random
from src.transfer import match_kl, match_jacobian, match_kl_jacobian

best_acc1 = 0
root_dir = '/scratch/hdd001/home/ama/improve-transferability/'
model_ckpt = {
        'cifar10': {
            'preactresnet18': '2023-07-19/cifar10/cosine/20230719-cifar10-preactresnet18-0.1-4',
            'preactresnet50': '2023-07-21/cifar10/cosine/20230721-cifar10-preactresnet50-0.1-4',
            'vgg19': '2023-07-27/20230727-cifar10-vgg19-0.1-4',
            'vit_small': '2023-08-02/20230802-cifar10-vit_small-0.1-4'
            },
        'cifar100': {
            'preactresnet18': '2023-07-19/cifar100/cosine/20230719-cifar100-preactresnet18-0.1-4',
            'preactresnet50': '2023-07-21/cifar100/cosine/20230721-cifar100-preactresnet50-0.1-4',
            'vgg19': '2023-07-27/20230727-cifar100-vgg19-0.1-4',
            'vit_small': '2023-08-02/20230802-cifar100-vit_small-0.1-4'
            }
        }

def train(args, epoch, logger, loader, model, opt, lr_scheduler, device):
    """perform one epoch of training."""
    if args.method == "standard":
        train_log = train_standard(loader, model, opt, device, epoch, lr_scheduler)

    else:
        raise  NotImplementedError("Training method not implemented!")

    logger.add_scalar("train/acc_ep", train_log[0], epoch)
    logger.add_scalar("train/loss_ep", train_log[1], epoch)
    logging.info(
        "Epoch: [{0}]\t"
        "Loss: {loss:.6f}\t"
        "Accuracy: {acc:.2f}".format(
            epoch,
            loss=train_log[1],
            acc=train_log[0]))

    return train_log

def main():

    global best_acc1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_args()
    print_args(args)

    logger = metaLogger(args)
    logging.basicConfig(
        filename=args.j_dir+ "/log/log.txt",
        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    seed_everything(args.seed)
    train_loader, test_loader, _, _ = load_dataset(
                                    args.dataset,
                                    args.batch_size,
                                    args.op_name,
                                    args.op_prob,
                                    args.op_magnitude)
    
    '''
    # make a list of model ckpts, using the following rule:
    # source model:  seed 40,41,42 (randomly select one)
    # target model:  seed 43,44,45
    # witness model: seed 46,47,48
    # goal: improve target model using witness models
    '''
    target_model_dir = root_dir + model_ckpt[args.dataset][args.target_arch]
    list_target_model_ckpt = [target_model_dir + str(i) + '/model/best_model.pt' for i in range(3, 6)]
    print('target model ckpt: ', list_target_model_ckpt)

    witness_model_dir = root_dir + model_ckpt[args.dataset][args.witness_arch]
    list_witness_model_ckpt = [witness_model_dir + str(i) + '/model/best_model.pt' for i in range(6, 6+args.num_witness)]
    print('witness model ckpt: ', list_witness_model_ckpt)

    # load witness model
    list_witness_model = []
    for i in range(args.num_witness):
        args.arch = args.witness_arch
        witness_model = get_model(args, device)
        witness_model.load_state_dict(torch.load(list_witness_model_ckpt[i]))
        witness_model.to(device)
        list_witness_model.append(witness_model)

    epoch, actual_trained_epoch = 1, 1
    mean_pre_test_acc1 = 0
    mean_post_test_acc1 = 0

    # load source model, target model, do stuff to source models 
    print('Modifying {} using {}!'.format(args.target_arch, args.witness_arch))
    print('Performing {}.'.format('misalignment' if args.misalign else 'alignment'))
    list_target_model, list_misaligned_target_model = [],[]
    for i, target_model_ckpt in enumerate(list_target_model_ckpt):
        # load target model
        args.arch = args.target_arch
        target_model = get_model(args, device)
        target_model.load_state_dict(torch.load(target_model_ckpt))
        target_model.to(device)

        # evaluation before modifying the target model
        pre_test_log = test_clean(test_loader, target_model, device)
        logger.add_scalar("pre/test_acc1({})".format(i), pre_test_log[0], epoch)

        list_target_model.append(copy.deepcopy(target_model))

        # init optimizer based on target model
        opt, _ = get_optim(target_model, args)

        # modify target model
        for _epoch in range(args.epoch):
            if args.method == 'kl':
                match_kl(train_loader, opt, args, target_model, list_witness_model, device)
            elif args.method == 'jacobian':
                match_jacobian(train_loader, opt, args, target_model, list_witness_model, device)
            elif args.method == 'kl-jacobian':
                match_kl_jacobian(train_loader, opt, args, target_model, list_witness_model, device)
            else:
                raise ValueError('Unspecified method: {}'.format(args.method))

        # evaluation after modifying the target model
        post_test_log = test_clean(test_loader, target_model, device)
        logger.add_scalar("post/test_acc1({})".format(i), post_test_log[0], epoch)

        logging.info(
            "Target Model: [{}]\t"
            "Test Acc before: {:.2f}\t"
            "Test Acc after: {:.2f}\t".format(
                i, pre_test_log[0], post_test_log[0]))
        mean_pre_test_acc1 += pre_test_log[0]/3.
        mean_post_test_acc1 += post_test_log[0]/3
        list_misaligned_target_model.append(copy.deepcopy(target_model))
    print('Misalignment Compelete!')

    mean_pre_transfer_acc1_avg_over_source_arch = 0
    mean_post_transfer_acc1_avg_over_source_arch = 0

    for source_arch in ['preactresnet18', 'preactresnet50', 'vgg19', 'vit_small']:
        print('Evaluating original and misaligned models on {}'.format(source_arch))
        source_model_dir = root_dir + model_ckpt[args.dataset][source_arch]
        source_model_ckpt = source_model_dir + str(random.randint(0, 2))+'/model/best_model.pt'
        print('source model ckpt: ', source_model_ckpt)

        # load source model
        args.arch = source_arch
        source_model = get_model(args, device)
        source_model.load_state_dict(torch.load(source_model_ckpt))
        source_model.to(device)

        transfer_log = test_transfer_multiple_target(test_loader, args, source_model,
                list_target_model + list_misaligned_target_model, device)

        mean_pre_transfer_acc1 = transfer_log[0][:3].mean()
        mean_post_transfer_acc1 = transfer_log[0][3:].mean()

        mean_pre_transfer_acc1_avg_over_source_arch += mean_pre_transfer_acc1/4.
        mean_post_transfer_acc1_avg_over_source_arch += mean_post_transfer_acc1/4.

        logger.add_scalar("pre/mean_transfer_acc1_{}".format(source_arch),
                mean_pre_transfer_acc1, epoch)
        logger.add_scalar("post/mean_transfer_acc1_{}".format(source_arch),
                mean_post_transfer_acc1, epoch)

        logging.info(
                "Source model: {}\n"
                "Transfer Acc before: {:.2f}, {:.2f}, {:.2f}\t"
                "Transfer Acc after: {:.2f} {:.2f} {:.2f}".format(source_arch,
                    transfer_log[0][0], transfer_log[0][1], transfer_log[0][2],
                    transfer_log[0][3], transfer_log[0][4], transfer_log[0][5]))

    # reporting the average statistics
    logger.add_scalar("pre/mean_test_acc1", mean_pre_test_acc1, epoch)
    logger.add_scalar("post/mean_test_acc1", mean_post_test_acc1, epoch)

    logger.add_scalar("pre/mean_transfer_acc1", mean_pre_transfer_acc1_avg_over_source_arch, epoch)
    logger.add_scalar("post/mean_transfer_acc1", mean_post_transfer_acc1_avg_over_source_arch, epoch)

    mean_test_acc1_diff = mean_pre_test_acc1 - mean_post_test_acc1
    mean_transfer_acc1_diff = mean_pre_transfer_acc1_avg_over_source_arch - mean_post_transfer_acc1_avg_over_source_arch

    logger.add_scalar("mean_test_acc1_diff", mean_test_acc1_diff, epoch)
    logger.add_scalar("mean_transfer_acc1_diff", mean_transfer_acc1_diff, epoch)
    logging.info(
        "Mean Test Acc before: {:.2f}\t"
        "Mean Transfer Acc before: {:.2f}\n"
        "Mean Test Acc after: {:.2f}\t"
        "Mean Transfer Acc after: {:.2f}\n"
        "Mean Test Acc Diff (pre-post): {:.2f}\t"
        "Mean Transfer Acc Diff (pre-post): {:.2f}".format(
            mean_pre_test_acc1, mean_pre_transfer_acc1_avg_over_source_arch,
            mean_post_test_acc1, mean_post_transfer_acc1_avg_over_source_arch,
            mean_test_acc1_diff, mean_transfer_acc1_diff))

    if mean_transfer_acc1_diff > 0 and args.save_modified_model:
        print('Saving the last source model to {}/model/'.format(args.j_dir))
        saveModel(args.j_dir+"/model/", "modified_source_model", source_model.state_dict())

    # upload runs to wandb:
    if args.enable_wandb:
        save_wandb_retry = 0
        save_wandb_successful = False
        while not save_wandb_successful and save_wandb_retry < 5:
            print('Uploading runs to wandb...')
            try:
                wandb_logger = wandbLogger(args)
                wandb_logger.upload(logger, actual_trained_epoch)
            except:
                save_wandb_retry += 1
                print('Retry {} times'.format(save_wandb_retry))
            else:
                save_wandb_successful = True

        if not save_wandb_successful:
            print('Failed at uploading runs to wandb.')

    logger.save_log(is_final_result=True)

    # delete slurm checkpoints
    delCheckpoint(args.j_dir, args.j_id)

if __name__ == "__main__":
    main()
