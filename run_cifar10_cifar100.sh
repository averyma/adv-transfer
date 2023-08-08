#!/bin/bash
project="vit-cifar-baselines"
gpu="t4v2,rtx6000"
#gpu="rtx6000"
#gpu="t4v2"
#gpu="t4v1,p100,t4v2"
enable_wandb=true #true/false
eval_AA=false
eval_CC=false

method='standard'
date=`date +%Y%m%d`
epoch=200
lr_scheduler_type="cosine"
batch_size=128
#lr=0.1
#arch='preactresnet18'
#arch='preactresnet50'
#arch='vgg19'
#arch='wrn28'


for seed in 40 41 42 43 44 45 46 47 48 49; do
	for dataset in 'cifar10' 'cifar100'; do
		for lr in 0.1 0.01 0.001; do
			for arch in 'vit' 'vit_small' 'simplevit'; do
			#for arch in 'vit'; do
				j_name=${date}'-'${dataset}'-'${arch}'-'${lr}'-'${seed}
				bash launch_slurm_job.sh ${gpu} ${j_name} 1 "python3 main.py --method \"${method}\" --lr ${lr} --dataset \"${dataset}\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb} --arch \"${arch}\" --seed ${seed} --eval_AA ${eval_AA} --eval_CC ${eval_CC} --batch_size ${batch_size} --lr_scheduler_type ${lr_scheduler_type}" 
				sleep 0.1
			done
		done
	done
done
