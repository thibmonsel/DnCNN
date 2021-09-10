#!/bin/sh
#SBATCH -J denoiser
#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH -t 24:01:00
#SBATCH -o logs.out
#SBATCH -e logs.err

srun python main_train.py

