#!/bin/bash
date=`date +%Y%m%d`
enable_wandb=true #true/false
save_modified_model=0
debug=0

dataset='imagenet'
pgd_itr=20
pgd_eps=0.01568 # 4/255
pgd_alpha=0.003921 # 1/255
epoch=1

declare -A m=( ["qos"]="m" ["hour"]=12 ["account"]='vector')
declare -A m2=( ["qos"]="m2" ["hour"]=8 ["account"]='vector')
declare -A m3=( ["qos"]="m3" ["hour"]=4 ["account"]='vector')
declare -A deadline=( ["qos"]="deadline" ["hour"]=12 ["account"]='deadline')


counter=0
for param in 0 1 2 3 4 5; do
	for param2 in 0 1 2 3 4 5 6 7; do
		if [[ ${counter}%20 -lt 6 ]]
		then
			qos=${deadline[qos]}
			hour=${deadline[hour]}
			account=${deadline[account]}
		elif [[ ${counter}%20 -lt 8 ]]
		then
			qos=${m[qos]}
			hour=${m[hour]}
			account=${m[account]}
		elif [[ ${counter}%20 -lt 12 ]]
		then
			qos=${m2[qos]}
			hour=${m2[hour]}
			account=${m2[account]}
		elif [[ ${counter}%20 -lt 20 ]]
		then
			qos=${m3[qos]}
			hour=${m3[hour]}
			account=${m3[account]}
		fi
		echo ${counter}'-'${qos}'-'${hour}'-'${account}
		counter=$((${counter}+1))
	done
done
