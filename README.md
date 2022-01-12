Muliwai text data processing 
```
pip install spacy==3.2
git clone https://github.com/bigscience-workshop/data_tooling
pip install cdifflib transformers datasets langid faker nltk sentencepiece fsspec tqdm sentence-transformers
python -m nltk.downloader punkt stopwords  wordnet
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
python -m spacy download pt_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download ca_core_news_sm
```
