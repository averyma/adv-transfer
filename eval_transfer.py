import os
import sys
import logging

import torch
import numpy as np

from src.attacks import pgd
from src.train import train_standard
from src.evaluation import test_clean, test_AA, eval_corrupt, eval_CE, test_gaussian, CORRUPTIONS_CIFAR10, test_transfer
from src.args import get_args, print_args
from src.utils_dataset import load_dataset
from src.utils_log import metaLogger, rotateCheckpoint, wandbLogger, saveModel, delCheckpoint
from src.utils_general import seed_everything, get_model, get_optim
import ipdb, pdb
import copy
from src.transfer import match_kl, match_jacobian

best_acc1 = 0
root_dir = '/scratch/hdd001/home/ama/improve-transferability/'
model_ckpt = {
        'cifar10': {
            'preactresnet18': '2023-07-19/cifar10/cosine/20230719-cifar10-preactresnet18-0.1-4',
            'preactresnet50': '2023-07-21/cifar10/cosine/20230721-cifar10-preactresnet50-0.1-4',
            'vgg19': '2023-07-27/20230727-cifar10-vgg19-0.1-4'
            },
        'cifar100': {
            'preactresnet18': '2023-07-19/cifar100/cosine/20230719-cifar100-preactresnet18-0.1-4',
            'preactresnet50': '2023-07-21/cifar100/cosine/20230721-cifar100-preactresnet50-0.1-4',
            'vgg19': '2023-07-27/20230727-cifar100-vgg19-0.1-4'
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
    # source model:  seed 40,41,42
    # target model:  seed 43,44,45
    # witness model: seed 46,47,48
    # goal: improve source model using witness models
    '''
    source_model_dir = root_dir + model_ckpt[args.dataset][args.source_arch]
    list_source_model_ckpt = [source_model_dir + str(i) + '/model/best_model.pt' for i in range(0, 3)]
    print('source model ckpt: ', list_source_model_ckpt)

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

    mean_pre_transfer_acc1 = 0
    mean_post_transfer_acc1 = 0

    # load source model, target model, do stuff to source models 
    print('Modifying {} using {}!'.format(args.source_arch, args.witness_arch))
    print('Transfering from {} to {}!'.format(args.source_arch, args.target_arch))
    for i, (source_model_ckpt, target_model_ckpt) in enumerate(zip(list_source_model_ckpt,
                                                                   list_target_model_ckpt)):
        # load source model
        args.arch = args.source_arch
        source_model = get_model(args, device)
        source_model.load_state_dict(torch.load(source_model_ckpt))
        source_model.to(device)

        # load target model
        args.arch = args.target_arch
        target_model = get_model(args, device)
        target_model.load_state_dict(torch.load(target_model_ckpt))
        target_model.to(device)

        # init optimizer based on source model
        opt, _ = get_optim(source_model, args)

        # evaluation before modifying the source model
        pre_test_log = test_clean(test_loader, source_model, device)
        pre_transfer_log = test_transfer(test_loader, args, source_model, target_model, device)
        logger.add_scalar("pre/test_acc1({})".format(i), pre_test_log[0], epoch)
        logger.add_scalar("pre/transfer_acc1({})".format(i), pre_transfer_log[0], epoch)
        logging.info(
            "Source Model: [{0}]\t"
            "Test Acc before: {test_acc:.2f}\t"
            "Transfer Acc before: {transfer_acc:.2f}".format(
                i,
                test_acc=pre_test_log[0],
                transfer_acc=pre_transfer_log[0]))
        mean_pre_test_acc1 += pre_test_log[0]/3.
        mean_pre_transfer_acc1 += pre_transfer_log[0]/3.

        # modify source model
        for _epoch in range(args.epoch):
            if args.method == 'kl':
                match_kl(train_loader, opt, args, source_model, list_witness_model, device)
            elif args.method == 'jacobian':
                match_jacobian(train_loader, opt, args, source_model, list_witness_model, device)

        # evaluation after modifying the source model
        post_test_log = test_clean(test_loader, source_model, device)
        post_transfer_log = test_transfer(test_loader, args, source_model, target_model, device)

        logger.add_scalar("post/test_acc1({})".format(i), post_test_log[0], epoch)
        logger.add_scalar("post/transfer_acc1({})".format(i), post_transfer_log[0], epoch)
        logging.info(
            "Source Model: [{0}]\t"
            "Test Acc after: {test_acc:.2f}\t"
            "Transfer Acc after: {transfer_acc:.2f}".format(
                i,
                test_acc=post_test_log[0],
                transfer_acc=post_transfer_log[0]))
        mean_post_test_acc1 += post_test_log[0]/3.
        mean_post_transfer_acc1 += post_transfer_log[0]/3.

    # reporting the average statistics
    logger.add_scalar("pre/mean_test_acc1", mean_pre_test_acc1, epoch)
    logger.add_scalar("pre/mean_transfer_acc1", mean_pre_transfer_acc1, epoch)
    logger.add_scalar("post/mean_test_acc1", mean_post_test_acc1, epoch)
    logger.add_scalar("post/mean_transfer_acc1", mean_post_transfer_acc1, epoch)

    mean_test_acc1_diff = mean_pre_test_acc1 - mean_post_test_acc1
    mean_transfer_acc1_diff = mean_pre_transfer_acc1 - mean_post_transfer_acc1

    logger.add_scalar("mean_test_acc1_diff", mean_test_acc1_diff, epoch)
    logger.add_scalar("mean_transfer_acc1_diff", mean_transfer_acc1_diff, epoch)
    logging.info(
        "Mean Test Acc before: {mean_pre_test_acc1:.2f}\t"
        "Mean Transfer Acc before: {mean_pre_transfer_acc1:.2f}\n"
        "Mean Test Acc after: {mean_post_test_acc1:.2f}\t"
        "Mean Transfer Acc after: {mean_post_transfer_acc1:.2f}\n"
        "Mean Test Acc Diff (pre-post): {mean_test_acc1_diff:.2f}\t"
        "Mean Transfer Acc Diff (pre-post): {mean_transfer_acc1_diff:.2f}".format(
            mean_pre_test_acc1=mean_pre_test_acc1,
            mean_pre_transfer_acc1=mean_pre_transfer_acc1,
            mean_post_test_acc1=mean_post_test_acc1,
            mean_post_transfer_acc1=mean_post_transfer_acc1,
            mean_test_acc1_diff=mean_test_acc1_diff,
            mean_transfer_acc1_diff=mean_transfer_acc1_diff))

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
