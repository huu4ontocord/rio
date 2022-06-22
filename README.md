# Intro
Rio (spanish for river) is a library for text pre-processing, augmentation, anonymization, and synthesis. It is intended to be used to process text datasets for training NLP models. This was the original Muliwai repo but the PII code has been refactored to live in its own rep at https://www.github.com/muliwai. TODO: import the muliwai as a library.

# Disclaimer
While we have code to detect and anonymize PII in this library, the intention of the library is to create better text training datasets, and NOT to generally protect PII, and this library is NOT intended as a genearl PII protection engine. 

# How it works
We use a transformer NER model that is good enough for the current language - in this case, a specific model for the language, or a model with some cross-lingual capabilities. Rio tags using the transformer, then translates the sentence to a target_lang (e.g., English), and tests to see if the translation preserves the NER tagging, and discoutns or increases the weight of an NER decision accordingly. It then performs NER in the target_lang, and back translates to the src_lang. It then matches the translate sentence to the original sentence, to determine which text spans in the *original* src_lang sentence should be NER tagged based on the target_lang NER.

We also use spacy and regex as added signals for NER tags.

# What it's meant for
The translation techniques used are very compute heavy, and not intended to perform fast detection or anonymization of a dataset. Instead, it is intended to be used to create augmented training data to train a relatively fast model (e.g., spacy or transformer model) for languages where there is little or no NER data.

# Installing
If you want to be able to do gender detection and coref detection, you will need to load neuralcoref below. However, you will only be able to use spacy english if you load neural coref. You can also load a larger spacy model for more accuracy but more memory.
```
git clone https://github.com/ontocord/rio
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install spacy==2.1.0 dateparser python-stdnum protobuf neuralcoref cdifflib transformers datasets langid faker sentencepiece fsspec tqdm sentence-transformers nltk
python -m nltk.downloader punkt wordnet
python -m spacy download en_core_web_sm
```

If you don't need gender detection and coref detection, install the below which will enable spacy for other languages. Neuralcoref will not be installed. You can also load a larger spacy model for more accuracy but more memory.
```
git clone https://github.com/ontocord/rio
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install spacy==3.1.0 dateparser python-stdnum protobuf cdifflib transformers datasets langid faker sentencepiece fsspec tqdm sentence-transformers nltk tokenizers==0.11.3
python -m nltk.downloader punkt wordnet
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download ca_core_news_sm
python -m spacy download pt_core_news_sm
python -m spacy download zh_core_web_sm

```

To experimental with adress detection, you should also insall libpostal:

```
sudo apt-get install curl autoconf automake libtool pkg-config
git clone https://github.com/openvenues/libpostal
cd libpostal
make distclean
./bootstrap.sh
./configure --datadir=/content/libpostal_data
make -j4
sudo make install
pip install postal
cp /usr/local/lib/libpostal.so /usr/lib/libpostal.so.1
```
 
# Running
If no filenames are passed, the sample data from turkunlp_data/{src_lang}.jsonl.gz will be loaded. The below runs on a sample of 30 documents only.
```
cd rio
python process.py -src_lang zh -cutoff 30
```
If you have more than one GPU
```
cd rio
python process.py -src_lang zh -num_workers=2 -cutoff 30
```
# CLI
```
usage: process.py [-h] [-src_lang SRC_LANG] [-target_lang TARGET_LANG]
                  [-augment_lang AUGMENT_LANG] [-cutoff CUTOFF]
                  [-batch_size BATCH_SIZE] [-hfdataset HFDATASET]
                  [-infile INFILE] [-shard_range SHARD_RANGE]
                  [-max_docs MAX_DOCS] [-outfile OUTFILE]
                  [-num_workers NUM_WORKERS] [-do_spacy_only DO_SPACY_ONLY]
                  [-do_hf_ner_only DO_HF_NER_ONLY]
                  [-do_dictionary_only DO_ONTOLOGY_ONLY]
                  [-do_regex_only DO_REGEX_ONLY]
                  [-do_qg_rel_only DO_QG_REL_ONLY] [-do_spacy DO_SPACY]
                  [-do_skip_src_lang_processing DO_SKIP_SRC_LANG_PROCESSING]
                  [-do_hf_ner DO_HF_NER] [-do_dictionary DO_ONTOLOGY]
                  [-do_trans DO_TRANS] [-do_backtrans DO_BACKTRANS]
                  [-do_augment DO_AUGMENT]
                  [-do_anonymization DO_ANONYMIZATION] [-do_regex DO_REGEX]
                  [-do_cleanup DO_CLEANUP] [-do_marian_mt DO_MARIAN_MT]
                  [-do_docs_trim_for_person DO_DOCS_TRIM_FOR_PERSON]
                  [-do_docs_filter DO_DOCS_FILTER] [-do_kenlm DO_KENLM]
                  [-do_qg_rel DO_QG_REL]
                  [-num_words_per_chunk NUM_WORDS_PER_CHUNK]
                  [-dictionary_weight ONTOLOGY_WEIGHT]
                  [-spacy_weight SPACY_WEIGHT] [-hf_ner_weight HF_NER_WEIGHT]
                  [-regex_weight REGEX_WEIGHT]
                  [-backtrans_weight BACKTRANS_WEIGHT] [-aug_scope AUG_SCOPE]
                  [-anon_scope ANON_SCOPE] [-force_gpu FORCE_GPU]
                  [-force_cpu FORCE_CPU] [-preload_cache]

Text Annotation, Augmentation and Anonymization

optional arguments:
  -h, --help            show this help message and exit
  -src_lang SRC_LANG    Source Language(s), comma separated
  -target_lang TARGET_LANG
                        Target Language or Languages, comma separated
  -augment_lang AUGMENT_LANG
                        Translate to this Language for text augmentation
  -cutoff CUTOFF        Cutoff documents, -1 is none
  -batch_size BATCH_SIZE
                        batch size
  -hfdataset HFDATASET  dataset to load, comma separated for different subsets
  -infile INFILE        file to load
  -shard_range SHARD_RANGE
                        portion of file to load, e.g., 1/4, 2/4, etc. unless
                        the dataset is a hf dataset, max_docs must also be set
  -max_docs MAX_DOCS    the maximum number of documents in this dataset
  -outfile OUTFILE      file to save
  -num_workers NUM_WORKERS
                        Num of Workers
  -do_spacy_only DO_SPACY_ONLY
                        Wether to only apply a spacy model
  -do_hf_ner_only DO_HF_NER_ONLY
                        Wether to only apply a huggingface NER model
  -do_dictionary_only DO_ONTOLOGY_ONLY
                        Wether to only use an dictionary
  -do_regex_only DO_REGEX_ONLY
                        Wether to only apply regex models
  -do_qg_rel_only DO_QG_REL_ONLY
                        Wether to only infer a relationship between PII
                        entities based an question generation (EXPERIMENTAL)
  -do_spacy DO_SPACY    Wether or not to apply a spacy model
  -do_skip_src_lang_processing DO_SKIP_SRC_LANG_PROCESSING
                        Wether or not to skip NER for src_lang (assumes NER is
                        already perfored in the data provided)
  -do_hf_ner DO_HF_NER  Wether or not to apply a huggingface NER model
  -do_dictionary DO_ONTOLOGY
                        Wether or not to use a dictionary
  -do_trans DO_TRANS    Wether or not to do translation (setting to 0 will
                        make src_lang == target_lang)
  -do_backtrans DO_BACKTRANS
                        Wether or not to do back translation
  -do_augment DO_AUGMENT
                        Wether or not to do translation augmentation
  -do_anonymization DO_ANONYMIZATION
                        Wether or not to anonymize the src_lang
  -do_regex DO_REGEX    Wether or not to apply regex models
  -do_cleanup DO_CLEANUP
                        Wether or not to cleanup NERs that are just stopwords
                        or small number
  -do_marian_mt DO_MARIAN_MT
                        Wether or not to use marianMT for translation instead
                        of M2M100
  -do_docs_trim_for_person DO_DOCS_TRIM_FOR_PERSON
                        Wether or not to filter out documents with no mentions
                        of persons
  -do_docs_filter DO_DOCS_FILTER
                        Wether or not to filter out documents with high ratios
                        of junk, or CSAM
  -do_kenlm DO_KENLM    Wether or not to apply a KenLM model to decide if a
                        name is a common person name
  -do_qg_rel DO_QG_REL  Wether or not to infer a relationship between PII
                        entities based an question generation (EXPERIMENTAL)
  -num_words_per_chunk NUM_WORDS_PER_CHUNK
                        number of words per chunk
  -dictionary_weight ONTOLOGY_WEIGHT
                        Weight given to the dictionary model
  -spacy_weight SPACY_WEIGHT
                        weight given to a spacy decision
  -hf_ner_weight HF_NER_WEIGHT
                        weight given to a hf model decision
  -regex_weight REGEX_WEIGHT
                        weight given to a regex decision
  -backtrans_weight BACKTRANS_WEIGHT
                        weight given to back tranlation decisions
  -aug_scope AUG_SCOPE  tag types for augmentation
  -anon_scope ANON_SCOPE
                        tag types for anonymization
  -force_gpu FORCE_GPU  Force usage of GPU
  -force_cpu FORCE_CPU  Force usage of CPU
  -preload_cache        Preload the cache of models and data  
  
  ```

# Using the API

You can use the functions in regex_manager.py, ner_manager.py, and faker_manager.py to detect and anonymize PII types in text.

You can modify the regex_rulebase or pass in your own to *detect_ner_with_regex_and_context* to customize your detection. 

The following *apply_anonymization* for example will annonymize a given text in a target language. 

```
from rio.regex_manager import detect_ner_with_regex_and_context
from rio.pii_regexes_rulebase import regex_rulebase
from rio.ner_manager import detect_ner_with_hf_model
from rio.faker_manager import augment_anonymize
def apply_anonymization(
    sentence: str,
    lang_id: str,
    context_window: int = 20,
    anonymize_condition=None,
    tag_type={'IP_ADDRESS', 'KEY', 'ID', 'PHONE', 'USER', 'EMAIL', 'LICENSE_PLATE', 'PERSON'} ,
    device: str = "cpu",
) -> str:
    """
    Params:
    ==================
    sentence: str, the sentence to be anonymized
    lang_id: str, the language id of the sentence
    context_window: int, the context window size
    anonymize_condition: function, the anonymization condition
    tag_type: iterable, the tag types of the anonymization. By default: {'IP_ADDRESS', 'KEY', 'ID', 'PHONE', 'USER', 'EMAIL', 'LICENSE_PLATE', 'PERSON'} 
    device: cpu or cuda:{device_id}

    """
    if tag_type == None:
        tag_type = regex_rulebase.keys()
    lang_id = lang_id.split("_")[0]
    ner_ids = detect_ner_with_regex_and_context(
        sentence=sentence,
        src_lang=lang_id,
        context_window=context_window,
        tag_type=tag_type,
    )
    ner_persons = detect_ner_with_hf_model(
        sentence=sentence,
        src_lang=lang_id,
        device=device,
        tag_type=tag_type,
    )
    ner = list(set(ner_ids + ner_persons))
    ner.sort(key=lambda a: a[1])
    if anonymize_condition:
        new_sentence, new_ner, _ = augment_anonymize(sentence, lang_id, ner, )
        doc = {'text': new_sentence, 'ner': new_ner, 'orig_text': sentence, 'orig_ner': ner}
    else:
        new_sentence = sentence
        doc = {'text': new_sentence, 'ner': ner}
    return new_sentence, doc
    
 ```
 
# Preloading the cache
- For systems where there is limited access to the Internet, such as the JZ supercomptuers, you will want to preload the models.
- The below command will load the various models needed to run the code for the specific language. 
- The huggingface models will be stored in ~/.cache/huggingface and ~/.cache/transformers.
- The neuralcoref cache is stored in ~/.neuralcoref
- NOTE: the nlkt_data and en_core_web_sm are not stored in ~/.cache directory and will vary based on your system. See the documentation for spacy and nltk for their location.
```
cd rio
python processor.py -src_lang zh -preload_cache

```

# License
- The source code authored by Ontocord LLC and contributed by contributors of this project is licensed under Apache 2.0.
- The TurkuNLP sample data is based on OSCAR and mc4. See the information uder turkunlp_data for more details.

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
