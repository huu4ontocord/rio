# Intro
Muliwai (pronounced: mu-lee-why, meaning river in Hawaiian) is a library for text pre-processing, augmentation, synthesis and generalization

# Installing
```
git clone https://github.com/ontocord/muliwai
pip install spacy==2.1.0 neuralcoref cdifflib transformers datasets langid faker sentencepiece fsspec tqdm sentence-transformers nltk
python -m nltk.downloader punkt 
python -m spacy download en_core_web_sm
cd muliwai
python processor.py -src_lang zh
```
