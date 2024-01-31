#!/bin/bash
#SBATCH --job-name=small
#SBATCH --time=1-00:00:00
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --output=logs/small.log
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=64GB
echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
nvidia-smi
conda activate /home/nkx870/anaconda3/envs/monorama

# ro de en fr pl it 
for language in lt
do
    python3.9 Lithuanian.py --language $language --model_size small
done
