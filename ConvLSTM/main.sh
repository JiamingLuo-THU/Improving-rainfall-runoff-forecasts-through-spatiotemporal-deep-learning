#!/bin/bash
#SBATCH -J python
#SBATCH -N 1
#SBATCH -p gnall
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=28 
#SBATCH --gres=gpu:8 
 
module load gpu/cuda/v12
source /home/zhudj/WORK/zhudj_work/miniconda3/bin/activate
conda activate ljm_torch

python -u main.py > Self_21_1_Mask_Self_multilr.log