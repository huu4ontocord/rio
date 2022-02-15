# Intro
Muliwai (pronounced: mu-lee-why, meaning river in Hawaiian) is a library for text pre-processing, augmentation, anonymization, and synthesis. It is intended to be used to process text datasets for training NLP models.

# What is it
Muliwai was written in part to support the data-tooling efforts of the BigScience workshop, but has grown beyond this. There are several utilities for performing NER and assocaited augmentation and anonymization. In theory, Muliwai can do NER in most of the languages supported by XLMRoberta & M2M100 (100+ languages). However, we have not tested various languages beyond: ar, ur, bn, hi, eu, ca, vi, zh, fr, id, es, pt,  sw, yo. 

There are other features, and we will create documentation soon...

# How it works
We use a transformer NER model that is good enough for the current language - in this case, a specific model for the language, or a model with some cross-lingual capabilities. Muliwai tags using the transformer, then translates the sentence to a target_lang (e.g., English), and tests to see if the translation preserves the NER tagging, and discoutns or increases the weight of an NER decision accordingly. It then performs NER in the target_lang, and back translates to the src_lang. It then matches the translate sentence to the original sentence, to determine which text spans in the *original* src_lang sentence should be NER tagged based on the target_lang NER.

We also use spacy and regex as added signals for NER tags.

# What it's meant for
The translation techniques used are very compute heavy, and not intended to perform fast detection or anonymization of a dataset. Instead, it is intended to be used to create augmented training data to train a relatively fast model (e.g., spacy or transformer model) for languages where there is little or no NER data.

# Installing
```
git clone https://github.com/ontocord/muliwai
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install spacy==2.1.0 dateparse python-stdnum protobuf neuralcoref cdifflib transformers datasets langid faker sentencepiece fsspec tqdm sentence-transformers nltk
python -m nltk.downloader punkt 
python -m spacy download en_core_web_sm
```

# Running
If no filenames are passed, the sample data from turkunlp_data/{src_lang}.jsonl.gz will be loaded. The below runs on a sample of 30 documents only.
```
cd muliwai
python processor.py -src_lang zh -cutoff 30
```
If you have more than one GPU
```
cd muliwai
python processor.py -src_lang zh -num_workers=2 -cutoff 30
```
# CLI

usage: process.py [-h] [-src_lang SRC_LANG] [-target_lang TARGET_LANG]
                  [-augment_lang AUGMENT_LANG] [-cutoff CUTOFF]
                  [-batch_size BATCH_SIZE] [-infile INFILE] [-outfile OUTFILE]
                  [-num_workers NUM_WORKERS] [-do_spacy_only DO_SPACY_ONLY]
                  [-do_hf_ner_only DO_HF_NER_ONLY]
                  [-do_regex_only DO_REGEX_ONLY]
                  [-do_qg_rel_only DO_QG_REL_ONLY] [-do_spacy DO_SPACY]
                  [-do_skip_src_lang_processing DO_SKIP_SRC_LANG_PROCESSING]
                  [-do_hf_ner DO_HF_NER] [-do_backtrans DO_BACKTRANS]
                  [-do_augment DO_AUGMENT]
                  [-do_anonymization DO_ANONYMIZATION] [-do_regex DO_REGEX]
                  [-do_cleanup DO_CLEANUP] [-do_marian_mt DO_MARIAN_MT]
                  [-do_docs_trim_for_person DO_DOCS_TRIM_FOR_PERSON]
                  [-do_docs_filter DO_DOCS_FILTER] [-do_kenlm DO_KENLM]
                  [-do_qg_rel DO_QG_REL]
                  [-num_words_per_chunk NUM_WORDS_PER_CHUNK]
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
  -infile INFILE        file to load
  -outfile OUTFILE      file to save
  -num_workers NUM_WORKERS
                        Num of Workers
  -do_spacy_only DO_SPACY_ONLY
                        Wether to only apply a spacy model
  -do_hf_ner_only DO_HF_NER_ONLY
                        Wether to only apply a huggingface NER model
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
  
# Preloading the cache
- For systems where there is limited access to the Internet, such as the JZ supercomptuers, you will want to preload the models.
- The below command will load the various models needed to run the code for the specific language. 
- The huggingface models will be stored in ~/.cache/huggingface and ~/.cache/transformers.
- NOTE: the nlkt_data and en_core_web_sm are not stored in ~/.cache directory and will vary based on your system. See the documentation for spacy and nltk for their location.
```
cd muliwai
python processor.py -src_lang zh -preload_cache

```
# License

- The source code authored by Ontocord LLC and contributed by contributors of this project is licensed under Apache 2.0.
- The TurkuNLP sample data is based on OSCAR and mc4. See the information uder turkunlp_data for more details.
- The ontology data is derived from Conceptnet and Yago and is mostly licensed under a CC license.

## Yago
Yago is licensed under CC BY 4.0. https://yago-knowledge.org/

## Conceptnet 5 Licensing Info

Below is information on the licensing of Conceptnet 5 from the authors of Conceptnet 5 generally under a CC BY SA 4.0 (http://conceptnet.io):
```
This work includes data from ConceptNet 5, which was compiled by the Commonsense Computing Initiative. ConceptNet 5 is freely available under the Creative Commons Attribution-ShareAlike license (CC BY SA 4.0) from http://conceptnet.io.

The included data was created by contributors to Commonsense Computing projects, contributors to Wikimedia projects, DBPedia, OpenCyc, Games with a Purpose, Princeton University's WordNet, Francis Bond's Open Multilingual WordNet, and Jim Breen's JMDict. Credits and acknowledgements ConceptNet has been developed by:

The MIT Media Lab, through various groups at different times:

Commonsense Computing Software Agents Digital Intuition The Commonsense Computing Initiative, a worldwide collaboration with contributions from:

National Taiwan University Universidade Federal de São Carlos Hokkaido University Tilburg University Nihon Unisys Labs Dentsu Inc. Kyoto University Yahoo Research Japan Luminoso Technologies, Inc.

Significant amounts of data were imported from:

WordNet, a project of Princeton University Open Multilingual WordNet, compiled by Francis Bond and Kyonghee Paik Wikipedia and Wiktionary, collaborative projects of the Wikimedia Foundation Luis von Ahn's "Games with a Purpose" JMDict, compiled by Jim Breen CC-CEDict, by MDBG The Unicode CLDR DBPedia Here is a short, incomplete list of people who have made significant contributions to the development of ConceptNet as a data resource, roughly in order of appearance:

Push Singh Catherine Havasi Hugo Liu Hyemin Chung Robyn Speer Ken Arnold Yen-Ling Kuo Joshua Chin Joanna Lowry-Duda Robert Beaudoin Naoki Otani Vanya Cohen Licenses for included resources Commonsense Computing The Commonsense Computing project originated at the MIT Media Lab and expanded worldwide. Tens of thousands of contributors have taken some time to teach facts to computers. Their pseudonyms can be found in the "sources" list found in ConceptNet's raw data and in its API.

Games with a Purpose Data collected from Verbosity, one of the CMU "Games with a Purpose", is used and released under ConceptNet's license, by permission from Luis von Ahn and Harshit Surana.

Verbosity players are anonymous, so in the "sources" list, data from Verbosity is simply credited to the pseudonym "verbosity".

Wikimedia projects ConceptNet uses data directly from Wiktionary, the free dictionary. It also uses data from Wikipedia, the free encyclopedia via DBPedia.

Wiktionary and Wikipedia are collaborative projects, authored by their respective online communities. They are currently released under the Creative Commons Attribution-ShareAlike license.

Wikimedia encourages giving attribution by providing links to the hosted pages that the data came from, and DBPedia asks for the same thing in turn. In addition to crediting the assertions that came from Wiktionary and DBPedia, we also provide "ExternalURL" edges pointing to the page that they came from. For example, the term /c/de/sprache has an ExternalURL link pointing to http://en.wiktionary.org/wiki/Sprache. Its list of individual contributors can be seen by following its "History" link.

The URLs of links to DBPedia are the same as the resource names that DBPedia uses, encouraging interoperability with their linked data.

WordNet WordNet is available under an unencumbered license: see http://wordnet.princeton.edu/wordnet/license/. Its text is reproduced below:

WordNet Release 3.0

This software and database is being provided to you, the LICENSEE, by Princeton University under the following license. By obtaining, using and/or copying this software and database, you agree that you have read, understood, and will comply with these terms and conditions.:

Permission to use, copy, modify and distribute this software and database and its documentation for any purpose and without fee or royalty is hereby granted, provided that you agree to comply with the following copyright notice and statements, including the disclaimer, and that the same appear on ALL copies of the software, database and documentation, including modifications that you make for internal use or for distribution.

WordNet 3.0 Copyright 2006 by Princeton University. All rights reserved.

THIS SOFTWARE AND DATABASE IS PROVIDED "AS IS" AND PRINCETON UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT NOT LIMITATION, PRINCETON UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES OF MERCHANT- ABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF THE LICENSED SOFTWARE, DATABASE OR DOCUMENTATION WILL NOT INFRINGE ANY THIRD PARTY PATENTS, COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS.

The name of Princeton University or Princeton may not be used in advertising or publicity pertaining to distribution of the software and/or database. Title to copyright in this software, database and any associated documentation shall at all times remain with Princeton University and LICENSEE agrees to preserve same.

Open Multilingual WordNet Open Multilingual WordNet was compiled by Francis Bond, Kyonghee Paik, and Ryan Foster, from data provided by many multilingual WordNet projects. Here is the complete list of references to the projects that created the data.
```
