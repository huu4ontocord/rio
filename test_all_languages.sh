
export CUTOFF=150
export GPU_NUMBERS=4
export NODES=2
for LANG in ar bn ca en es eu fr hi id ig ny pt sn st sw ur vi xh yo zh zu;do
      EXP_NAME=${LANG}_${NODES}_${GPU_NUMBERS}_${CUTOFF}
      sbatch --job-name=${EXP_NAME} \
        --account=six@gpu \
        --gres=gpu:1 \
        --no-requeue \
        --nodes=${NODES} \
        --cpus-per-task=10 \
        --hint=nomultithread \
        --time=20:00:00 \
        --qos=qos_gpu-t4 \
        --output=jobinfo/${EXP_NAME}_%j.out \
        --error=jobinfo/${EXP_NAME}_%j.err \
        --wrap="module purge; module load pytorch-gpu/py3/1.7.0 ;export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1 ;time python process.py -src_lang ${LANG} -outfile=${EXP_NAME}.json -num_workers 2  -do_trans 0 -cutoff ${CUTOFF} -do_hf_ner 1 -do_spacy 1 -do_regex 1 -do_kenlm 1 -do_anonymization 1"
    done
