## Set up the projet on Djeanzaï

0. Prepare symlicks:

 ```
Todo nltk, dataset, transformers badaboum :)
setup sentence transformers as well
ln -s $six_ALL_CCFRWORK/PII/cache_dir/.neuralcoref $HOME/.neuralcoref
ln -s $six_ALL_CCFRWORK/PII/cache_dir/.cache/transformers $HOME/.cache/transformers
ln -s $six_ALL_CCFRWORK/PII/cache_dir/.cache/huggingface $HOME/.cache/huggingface
ln -s $six_ALL_CCFRWORK/PII/cache_dir/nltk_data $HOME/nltk_data

 ```

1. Check current project:
 ```
idrproj
 ```
2. Set the HF project:
 ```
eval $(idrenv -d six)
 ```
3. Go in $six_ALL_CCFRWORK/PII
 ```
cd $six_ALL_CCFRWORK/PII 
 ```
4. Get the repo
 ```
git clone https://github.com/ontocord/muliwai.git
 ```

5. Set up python stuffs:
 ```
module load pytorch-gpu/py3/1.7.0 
cd muliwai
pip install -r requirements_pierre_spacy.txt
 ```
 
6. Check if everything works on the dev node with a single GPU
 ```
srun --pty --partition=prepost --account=six@gpu --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:0 --hint=nomultithread --time=1:00:00 bash
python process.py -src_lang zh -cutoff 30 -preload_cache
 ```


7. Check if everything works on the prod node without internet
 ```
srun --pty --partition=prepost --account=six@gpu --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:0 --hint=nomultithread --time=1:00:00 bash

export HF_DATASETS_OFFLINE=1 # ugly but Dataset and Transformers are bugged
export TRANSFORMERS_OFFLINE=1
 ```
 
 
8. Target more gpus 
 ```
export GPU_NUMBERS=3
srun --pty -A six@gpu --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:$GPU_NUMBERS --hint=nomultithread --time=60 bash

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python process.py -src_lang zh -num_workers=$GPU_NUMBERS -cutoff 30

 ```
