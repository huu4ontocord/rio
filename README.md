# Intro
Muliwai (pronounced: mu-lee-why, meaning river in Hawaiian) is a library for text pre-processing, augmentation, anonymization, synthesis and generalization. It is intended to be used to process text datasets for training NLP models.

# Installing
```
git clone https://github.com/ontocord/muliwai
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install spacy==2.1.0 protobuf neuralcoref cdifflib transformers datasets langid faker sentencepiece fsspec tqdm sentence-transformers nltk
python -m nltk.downloader punkt 
python -m spacy download en_core_web_sm
```

# Running
```
cd muliwai
python processor.py -src_lang zh
```


# Preloading the cache
```
cd muliwai
python processor.py -src_lang zh -preload_cache

```
