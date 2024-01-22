#!/bin/bash
#SBATCH --job-name=tiny
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=11:30:00
#SBATCH --output=logs/tiny.log
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=16GB
echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
nvidia-smi
conda activate /home/nkx870/anaconda3/envs/monorama

# ro de en fr pl it es
for language in et
do
    python3.9 program.py --language $language --model_size tiny
done