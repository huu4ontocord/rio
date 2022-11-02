# Intro
Rio (spanish for river) is a library for text pre-processing, filtering, and augmentation. It is intended to be used to process text datasets for training NLP models. This is based on the original Muliwai repo but the PII code has been refactored to live in its own rep at https://www.github.com/piisa/muliwai. Howeover, Rio no longer does PII processing. Please use https://www.github.com/piisa/muliwai instead.

# Installing
If you want to be able to do gender detection and coref detection, you will need to load neuralcoref below. However, you will only be able to use spacy english if you load neural coref. You can also load a larger spacy model for more accuracy but more memory.
```
git clone https://github.com/ontocord/rio
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install spacy==2.1.0 regex==2022.3.2 dateparser python-stdnum protobuf neuralcoref cdifflib transformers datasets langid faker sentencepiece fsspec tqdm sentence-transformers nltk
python -m nltk.downloader punkt wordnet
```

# License
- The source code authored by Ontocord LLC and contributed by contributors of this project is licensed under Apache 2.0.

# Contributors

We welcome all contributions. Please feel free to send a PR. Please follow the code of conduct: https://github.com/ontocord/rio/blob/main/CODE_OF_CONDUCT.md 
Special thanks to these people not just for code contributions but for comments and reviews (in no particular order): 
- @dadelani
- @edugp 
- @vumichien
- @ianyu93
- @j-chim
- @justinphan3110
- @mapama247
- @paulovn
- @PierreColombo
- @piesauce
- @mmitchellai
- @shamikbose

# Acknowledgements

We heavily use the models trained by @dadelani and the excelent work by https://github.com/masakhane-io.
