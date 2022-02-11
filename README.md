# Intro
Muliwai (pronounced: mu-lee-why, meaning river in Hawaiian) is a library for text pre-processing, augmentation, anonymization, synthesis and generalization. It is intended to be used to process text datasets for training NLP models.

# What is it
Muliwai was written in part to support the data-tooling efforts of the BigScience workshop, but has grown beyond this. There are several utilities for performing NER and assocaited augmentation and anonymization. In theory, Muliwai can do NER in most of the languages supported by XLMRoberta & M2M100 (100+ languages). However, we have not tested various languages beyond: ar, ur, bn, hi, eu, ca, vi, zh, fr, id, fa, es, pt, ig, sw, yo, zh, and zu. 

There are other features, and we will create documentation soon...

# How it works
We use a transformer NER model that is good enough for the current language - in this case, a specific model for the language, or a model with some cross-lingual capabilities. Muliwai tags using the transformer, then translates the sentence to a target_lang (e.g., English), and tests to see if the translation preserves the NER tagging, and discoutns or increases the weight of an NER decision accordingly. It then performs NER in the target_lang, and translates to the src_lang. It then matches the translate sentence to the original sentence, to determine which text spans in the *original* src_lang sentence should be NER tagged based on the target_lang NER.

# What it's meant for
The technique used are very compute heavy, and not intended to perform fast detection or anonymization of a dataset. Instead, it is intended to be used to create augmented training data to train a relatively fast model (e.g., spacy or transformer model) for languages where there is little or no NER data.

# Installing
```
git clone https://github.com/ontocord/muliwai
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install spacy==2.1.0 python-stdnum protobuf neuralcoref cdifflib transformers datasets langid faker sentencepiece fsspec tqdm sentence-transformers nltk
python -m nltk.downloader punkt 
python -m spacy download en_core_web_sm
```

# Running
```
cd muliwai
python processor.py -src_lang zh
```
If you have more than one GPU
```
cd muliwai
python processor.py -src_lang zh -num_workers=2
```

# Preloading the cache
```
cd muliwai
python processor.py -src_lang zh -preload_cache

```
