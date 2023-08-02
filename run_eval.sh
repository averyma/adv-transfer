#!/bin/bash
project="eval-transferability-cifar"
#gpu="rtx6000"
gpu="t4v2,rtx6000"
#gpu="t4v2"
#gpu="t4v1,p100,t4v2"
enable_wandb=true #true/false
eval_AA=false
eval_CC=false

method='kl'
date=`date +%Y%m%d`
epoch=5
lr_scheduler_type="fixed"
batch_size=128
#lr=0.1
#source_arch='vgg19'
#target_arch='preactresnet50'
#witness_arch='vgg19'
seed=40
#num_witness=1
#dataset='cifar10'

for dataset in 'cifar10' 'cifar100'; do
	for source_arch in 'vgg19' 'preactresnet18' 'preactresnet50'; do
		for target_arch in 'vgg19' 'preactresnet18' 'preactresnet50'; do
			for num_witness in 1 3; do
				for lr in 0.001; do
					for noise_type in 'none' 'rand_init' 'rand_init_indep' 'pgd7' 'pgd7_indep'; do
						witness_arch=${source_arch}
						j_name='eval-'${dataset}'-S-'${source_arch}'-T-'${target_arch}'-'${num_witness}'W-'${witness_arch}'-'${noise_type}'-'${epoch}'-'${lr}
						bash launch_slurm_job.sh ${gpu} ${j_name} 1 "python3 eval_transfer.py --method \"${method}\" --lr ${lr} --dataset \"${dataset}\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb} --seed ${seed} --batch_size ${batch_size} --lr_scheduler_type ${lr_scheduler_type} --warmup 0 --source_arch ${source_arch} --target_arch ${target_arch} --num_witness ${num_witness} --witness_arch ${witness_arch} --noise_type ${noise_type}"
						sleep 0.1
					done
				done
			done
		done
	done
done
