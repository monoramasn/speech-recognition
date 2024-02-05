#!/bin/bash
#SBATCH --job-name=small
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/small.log
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=64GB
echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
nvidia-smi
conda activate /home/nkx870/anaconda3/envs/monorama

# ro de en fr pl it es
for language in et
do
    python3.9 Estonian.py --language $language --model_size small
done

