#!/bin/bash
project="imagenet-baselines"
#project="imagenet_cluster_optimization"
#gpu="t4v2,rtx6000"
#gpu="t4v2"
gpu="rtx6000"
#tv42: 4cpus-per-gpu with 8 gpus
#rtx6000: 8cpus-per-gpu with 4gpus
#gpu="t4v1,p100,t4v2"
enable_wandb=true #true/false
eval_AA=false
eval_CC=false

method='standard'
dataset='imagenet'
date=`date +%Y%m%d`
epoch=90
lr=0.1
#arch='resnet50'

batch_size=256
num_gpu=4
#seed=40
resume_from_ckpt='/scratch/hdd001/home/ama/improve-transferability/2023-08-06/20230806-imagenet-resnet18-256-48/10609428'
optimize_cluster_param=false

#for arch in 'resnet50'; do
	#for seed in 44 45 46 47 48; do
	#for seed in 44 45 48; do
	#for seed in 44; do
#for arch in 'vgg19_bn'; do
	#for seed in 40 41 42 43 44 45 46 47 48; do
for arch in 'resnet18'; do
	#for seed in 43 45 46 47 48; do
	#for seed in 47 48; do
	for seed in 48; do
		j_name=${date}'-'${dataset}'-'${arch}'-'${batch_size}'-'${seed}
		bash launch_slurm_job_single_node_multi_gpu.sh ${gpu} ${j_name} ${num_gpu} "python3 main_imagenet.py --method \"${method}\" --lr ${lr} --dataset \"${dataset}\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb} --arch \"${arch}\" --seed ${seed} --eval_AA ${eval_AA} --eval_CC ${eval_CC} --batch_size ${batch_size} --ckpt_freq 5 --optimize_cluster_param ${optimize_cluster_param} --resume_from_ckpt ${resume_from_ckpt}"
		sleep 0.5
	done
done
