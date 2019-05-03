#!/bin/bash

#SBATCH --job-name=train_rnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176

export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

srun python3 train.py