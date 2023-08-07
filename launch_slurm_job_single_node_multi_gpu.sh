#!/bin/bash

d=`date +%Y-%m-%d`
partition=$1
j_name=$2
resource=$3
cmd=$4
hdd=/scratch/hdd001/home/$USER
ssd=/scratch/ssd001/home/$USER
j_dir=$hdd/improve-transferability/$d/$j_name

mkdir -p $j_dir/scripts

# build slurm script
mkdir -p $j_dir/log
mkdir -p $j_dir/slurm_out
mkdir -p $j_dir/config
mkdir -p $j_dir/model
echo "#!/bin/bash
#SBATCH --job-name=${j_name}
#SBATCH --output=${j_dir}/slurm_out/%j.out
#SBATCH --error=${j_dir}/slurm_out/%j.err
#SBATCH --partition=${partition}
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:${resource}
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --open-mode=append
export MASTER_PORT=\"$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')\"
bash ${j_dir}/scripts/${j_name}.sh
" > $j_dir/scripts/${j_name}.slrm

# build bash script
echo -n "#!/bin/bash
ln -s /checkpoint/$USER/\$SLURM_JOB_ID ${j_dir}/\$SLURM_JOB_ID
touch ${j_dir}/\$SLURM_JOB_ID/DELAYPURGE
wandb login 9bc64f558f249643c1805ff63ac9c55f0ef649c4
$cmd --j_dir ${j_dir} --j_id \$SLURM_JOB_ID --dist-url \"tcp://127.0.0.1:\$MASTER_PORT\" --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
" > $j_dir/scripts/${j_name}.sh

sbatch $j_dir/scripts/${j_name}.slrm
