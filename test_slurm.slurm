#!/bin/bash
#SBATCH --account=six@gpu
#SBATCH --job-name=tr8b-104B-emb-norm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.err          # output file name
set -x -e

module load pytorch-gpu/py3/1.7.0

# to debug - add echo (it exits and prints what it would have launched)
srun bash -c "python process.py -src_lang zh -num_workers=$GPU_NUMBERS -cutoff 30"
