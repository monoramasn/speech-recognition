#!/bin/bash
#SBATCH --job-name=large
#SBATCH --time=1-00:00:00
#SBATCH -p gpu --gres=gpu:a100:2
#SBATCH --output=logs/large.log
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=64GB
echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
nvidia-smi
conda activate /home/nkx870/anaconda3/envs/monorama

# ro de en fr pl it 
for language in hr
do
    python3.9 Croatian.py --language $language --model_size large
done
