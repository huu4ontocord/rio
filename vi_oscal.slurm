#!/bin/bash
#SBATCH --account=six@gpu
#SBATCH --job-name=vi_oscar_all
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=vi_oscar_all_%x-%j.stdout           # output file name
#SBATCH --error=vi_oscar_all_%x-%j.stderr          # output file name
set -x -e

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
module load pytorch-gpu/py3/1.7.0

# to debug - add echo (it exits and prints what it would have launched)
srun bash -c "time  -hfdataset TurkuNLP/register_oscar,vi -src_lang vi -do_trans 0  -do_hf_ner 1 -do_spacy 1 -do_regex 1 -do_kenlm 1 -do_anonymization 1 -num_workers 4"
