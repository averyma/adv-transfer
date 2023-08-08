#!/bin/bash
project="eval-transferability-cifar-misalign"
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
save_modified_model=0
misalign=1

for noise_type in 'none' 'pgd7' 'rand_init'; do
	for dataset in 'cifar10'; do
	#for dataset in 'cifar10' 'cifar100'; do
		for source_arch in 'vgg19' 'preactresnet18' 'preactresnet50' 'vit_small'; do
		#for source_arch in 'vgg19' 'preactresnet18'; do
		#for source_arch in 'vgg19'; do
		#for source_arch in 'vit_small'; do
			#for target_arch in 'vgg19' 'preactresnet18' 'preactresnet50'; do
			#for target_arch in 'vgg19' 'preactresnet18'; do
			for target_arch in 'vgg19' 'preactresnet18' 'preactresnet50' 'vit_small'; do
			#for target_arch in 'vit_small'; do
				for num_witness in 1; do
					for lr in 0.001 0.000001; do
						for witness_arch in 'vgg19' 'preactresnet18' 'preactresnet50' 'vit_small'; do
						#for witness_arch in 'vit_small'; do
						#for witness_arch in 'vgg19' 'preactresnet18' 'preactresnet50'; do
							#if [ "${source_arch}" != "${witness_arch}" ]; then
							#witness_arch=${source_arch}
							j_name='eval-misalign-'${method}'-'${dataset}'-S-'${source_arch}'-T-'${target_arch}'-'${num_witness}'W-'${witness_arch}'-'${noise_type}'-'${epoch}'-'${lr}
							#j_name='eval-'${method}'-'${dataset}'-S-'${source_arch}'-T-'${target_arch}'-'${num_witness}'W-'${witness_arch}'-'${noise_type}'-'${epoch}'-'${lr}
							bash launch_slurm_job.sh ${gpu} ${j_name} 1 "python3 eval_transfer.py --method \"${method}\" --lr ${lr} --dataset \"${dataset}\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb} --seed ${seed} --batch_size ${batch_size} --lr_scheduler_type ${lr_scheduler_type} --warmup 0 --source_arch ${source_arch} --target_arch ${target_arch} --num_witness ${num_witness} --witness_arch ${witness_arch} --noise_type ${noise_type} --save_modified_model ${save_modified_model} --misalign ${misalign}"
							sleep 0.1
							#fi
						done
					done
				done
			done
		done
	done
done
