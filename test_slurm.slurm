#!/bin/bash
#SBATCH --job-name=tr8b-104B-emb-norm
#SBATCH --constraint=v100-32g
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@gpu

set -x -e

# to debug - add echo (it exits and prints what it would have launched)
clear; srun --jobid $SLURM_JOBID bash -c "python process.py -src_lang zh -num_workers=$GPU_NUMBERS -cutoff 30"
