#!/bin/bash
#SBATCH -J python
#SBATCH -N 1
#SBATCH -p cnall
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=56 

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
 
module load gpu/cuda/v12
source /home/zhudj/WORK/zhudj_work/miniconda3/bin/activate
conda activate ljm

python -u Daymet_Generate_ConvLSTM_data_mask.py > Daymet_Generate_ConvLSTM_data_mask_self.log
