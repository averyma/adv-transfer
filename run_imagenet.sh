#!/bin/bash
project="improve-transferability"
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

batch_size=128
num_gpu=4
seed=40
optimize_cluster_param=false

for arch in 'resnet18'; do
	for seed in 40 41 42 43 44; do
		j_name=${date}'-'${dataset}'-'${arch}'-'${batch_size}'-'${seed}
		#j_name='cluster-optim-'${gpu}'-'${workers}
		#j_name='debug'
		bash launch_slurm_job_single_node_multi_gpu.sh ${gpu} ${j_name} ${num_gpu} "python3 main_imagenet.py --method \"${method}\" --lr ${lr} --dataset \"${dataset}\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb} --arch \"${arch}\" --seed ${seed} --eval_AA ${eval_AA} --eval_CC ${eval_CC} --batch_size ${batch_size} --ckpt_freq 5 --optimize_cluster_param ${optimize_cluster_param}"
		sleep 0.5
	done
done
