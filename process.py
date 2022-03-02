"""
Copyright, 2021-2022 Ontocord, LLC, All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import re
import fsspec
import copy
from collections import Counter
from  datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaForTokenClassification, M2M100ForConditionalGeneration, M2M100Tokenizer, pipelines
import spacy
from tqdm import tqdm
import difflib
from transformers import pipeline, MarianMTModel, XLMRobertaForTokenClassification, BertForTokenClassification, ElectraForTokenClassification
import random
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import langid
import json
import os
import time
import gzip
from functools import partial
import argparse
import re, regex
import itertools
import torch
from torch import multiprocessing
import sys
from huggingface_hub import hf_hub_url, cached_download
import argparse
from torch import multiprocessing
import time
from functools import partial
from faker import Faker
from faker.providers import person, company, geo, address, ssn, internet

import logging

from transformers.utils.dummy_tf_objects import TFRagSequenceForGeneration
logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s : %(processName)s : %(levelname)s : %(message)s',
    level=logging.INFO)

try:
  import neuralcoref
except:
  neuralcoref = None
  pass
import sys
try:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))         
except:
    pass
 
from text_augment import *

if __name__ == "__main__":
  in_notebook = 'google.colab' in sys.modules
  if not in_notebook:
    try:
        get_ipython()
    except:
      in_notebook = False
  if not in_notebook:
    parser = argparse.ArgumentParser(description='Text Annotation, Augmentation and Anonymization')
    parser.add_argument('-src_lang', dest='src_lang', type=str, help='Source Language(s), comma separated', default=None)
    parser.add_argument('-target_lang', dest='target_lang', type=str, help='Target Language or Languages, comma separated', default="en")
    parser.add_argument('-augment_lang', dest='augment_lang', type=str, help='Translate to this Language for text augmentation', default="en")
    parser.add_argument('-cutoff', dest='cutoff', type=int, help='Cutoff documents, -1 is none', default=-1)
    parser.add_argument('-batch_size', dest='batch_size', type=int, help='batch size', default=5)
    parser.add_argument('-hfdataset', dest='hfdataset', type=str, help='dataset to load, comma separated for different subsets', default=None)
    parser.add_argument('-infile', dest='infile', type=str, help='file to load', default=None)
    parser.add_argument('-outfile', dest='outfile', type=str, help='file to save', default=None)
    parser.add_argument('-num_workers', dest='num_workers', type=int, help='Num of Workers', default = 1)
    parser.add_argument('-do_spacy_only', dest='do_spacy_only', type=int, help='Wether to only apply a spacy model', default = 0)
    parser.add_argument('-do_hf_ner_only', dest='do_hf_ner_only', type=int, help='Wether to only apply a huggingface NER model', default = 0)
    parser.add_argument('-do_dictionary_only', dest='do_ontology_only', type=int, help='Wether to only use an dictionary', default = 0)
    parser.add_argument('-do_regex_only', dest='do_regex_only', type=int, help='Wether to only  apply regex models', default = 0)
    parser.add_argument('-do_qg_rel_only', dest='do_qg_rel_only', type=int, help='Wether to only infer a relationship between PII entities based an question generation (EXPERIMENTAL)', default = 0)
    parser.add_argument('-do_spacy', dest='do_spacy', type=int, help='Wether or not to apply a spacy model', default = 1)
    parser.add_argument('-do_skip_src_lang_processing', dest='do_skip_src_lang_processing', type=int, help='Wether or not to skip NER for src_lang (assumes NER is already perfored in the data provided)', default = 0)
    parser.add_argument('-do_hf_ner', dest='do_hf_ner', type=int, help='Wether or not to apply a huggingface NER model', default = 1)
    parser.add_argument('-do_dictionary', dest='do_ontology', type=int, help='Wether or not to use a dictionary', default = 1)
    parser.add_argument('-do_trans', dest='do_trans', type=int, help='Wether or not to do translation (setting to 0 will make src_lang == target_lang)', default = 1)
    parser.add_argument('-do_backtrans', dest='do_backtrans', type=int, help='Wether or not to do back translation', default = 1)
    parser.add_argument('-do_augment', dest='do_augment', type=int, help='Wether or not to do translation augmentation', default = 0)
    parser.add_argument('-do_anonymization', dest='do_anonymization', type=int, help='Wether or not to anonymize the src_lang', default = 0)
    parser.add_argument('-do_regex', dest='do_regex', type=int, help='Wether or not to apply regex models', default = 1)
    parser.add_argument('-do_cleanup', dest='do_cleanup', type=int, help='Wether or not to cleanup NERs that are just stopwords or small number', default = 1)
    parser.add_argument('-do_marian_mt', dest='do_marian_mt', type=int, help='Wether or not to use marianMT for translation instead of M2M100', default = 0)
    parser.add_argument('-do_docs_trim_for_person', dest='do_docs_trim_for_person', type=int, help='Wether or not to filter out documents with no mentions of persons', default = 0)
    parser.add_argument('-do_docs_filter', dest='do_docs_filter', type=int, help='Wether or not to filter out documents with high ratios of junk, or CSAM', default = 0)
    parser.add_argument('-do_kenlm', dest='do_kenlm', type=int, help='Wether or not to apply a KenLM model to decide if a name is a common person name', default = 1)
    parser.add_argument('-do_qg_rel', dest='do_qg_rel', type=int, help='Wether or not to infer a relationship between PII entities based an question generation (EXPERIMENTAL)', default = 0)
    parser.add_argument('-num_words_per_chunk', dest='num_words_per_chunk', type=int, help='number of words per chunk', default=70)
    parser.add_argument('-dictionary_weight', dest='ontology_weight', type=float, help='Weight given to the dictionary model', default=0.85)
    parser.add_argument('-spacy_weight', dest='spacy_weight', type=float, help='weight given to a spacy decision', default=1.00)
    parser.add_argument('-hf_ner_weight', dest='hf_ner_weight', type=float, help='weight given to a hf model decision', default=1.25)
    parser.add_argument('-regex_weight', dest='regex_weight', type=float, help='weight given to a regex decision', default=1.5)
    parser.add_argument('-backtrans_weight', dest='backtrans_weight', type=float, help='weight given to back tranlation decisions', default=0.9)
    parser.add_argument('-aug_scope', dest='aug_scope', type=str, help='tag types for augmentation', default="ADDRESS,ORG,PERSON,LOC,ID")
    parser.add_argument('-anon_scope', dest='anon_scope', type=str, help='tag types for anonymization', default='PERSON,ID')
    parser.add_argument('-force_gpu', dest='force_gpu', type=int, help='Force usage of GPU', default = 0)
    parser.add_argument('-force_cpu', dest='force_cpu', type=int, help='Force usage of CPU', default = 0)
    parser.add_argument('-preload_cache', dest='preload_cache', action='store_true', help='Preload the cache of models and data', default = 0)
    args = parser.parse_args()
    if args.force_gpu:
        TextAugmentDeviceModel.available_device_models =[None]
        TextAugmentDeviceModel.available_devices=[0]
    elif args.force_cpu:
        TextAugmentDeviceModel.available_device_models =[None]
        TextAugmentDeviceModel.available_devices=[-1]
    if args.do_spacy_only:
      args.do_spacy = True
      args.do_hf_ner = False
      args.do_regex = False
      args.do_qg_rel = False
      args.do_ontology_only = False
      args.do_backtrans = False
      args.do_trans = False
      args.target_lang = args.src_lang
      args.do_anonymization = False
      args.do_augmentation = False
    elif args.do_regex_only:
      args.do_spacy = False
      args.do_hf_ner = False
      args.do_regex = True
      args.do_qg_rel = False
      args.do_ontology_only = False
      args.do_backtrans = False
      args.do_trans = False
      args.target_lang = args.src_lang
      args.do_anonymization = False
      args.do_augmentation = False
    elif args.do_hf_ner_only:
      args.do_spacy = False
      args.do_hf_ner = True
      args.do_regex = False
      args.do_qg_rel = False
      args.do_ontology_only = False
      args.do_backtrans = False
      args.do_trans = False
      args.target_lang = args.src_lang
      args.do_anonymization = False
      args.do_augmentation = False
    elif args.do_qg_rel_only:
      args.do_spacy = False
      args.do_hf_ner = False
      args.do_regex = False
      args.do_backtrans = False
      args.do_trans = False
      args.do_qg_rel = True
      args.do_ontology_only = False
      args.target_lang = args.src_lang
      args.do_anonymization = False
      args.do_augmentation = False
    elif args.do_ontology_only:
      args.do_spacy = False
      args.do_hf_ner = False
      args.do_regex = False
      args.do_backtrans = False
      args.do_trans = False
      args.do_qg_rel = False
      args.do_ontology_only = True
      args.target_lang = args.src_lang
      args.do_anonymization = False
      args.do_augmentation = False
    args.anon_scope = set(args.anon_scope.split(","))
    args.aug_scope = set(args.aug_scope.split(","))
    src_lang = args.src_lang
    if src_lang is not None:
      src_lang = src_lang.split(",")
    else:
      src_lang = ["en"]
    if not args.do_trans:
      do_backtrans = False
      target_lang = src_lang
    elif not args.target_lang:
        target_lang =["en"]
    else:
      target_lang = args.target_lang.split(",")
    if len(target_lang) < len(src_lang):
      target_lang.extend([target_lang[0]]*(len(src_lang)-len(target_lang)))
    cutoff = args.cutoff
    batch_size = args.batch_size
    infile = args.infile
    outfile = args.outfile
    num_workers = args.num_workers
    if cutoff <= 0:
      cutoff = None
    if outfile is None:
      if infile is not None:
        outfile = "out.jsonl"
    docs = TextAugment.deserialize_ner_items(infile=infile) if infile else None
    if args.preload_cache: 
      TextAugment.preload_cache(src_lang or ["en"], target_lang)
    #TODO - do multiprocessing
    elif src_lang is not None:
      if num_workers > 1:
        TextAugment.multiprocess_ner(docs,
                    outfile,
                    src_langs=src_lang,
                    target_langs=target_lang,
                    hfdataset=args.hfdataset,
                    do_spacy = args.do_spacy ,
                    do_hf_ner = args.do_hf_ner ,
                    do_ontology = args.do_ontology,
                    do_skip_src_lang_processing=args.do_skip_src_lang_processing,
                    do_backtrans=args.do_backtrans,
                    do_augment=args.do_augment,
                    do_anonymization=args.do_anonymization,
                    augment_lang=args.augment_lang,
                    do_cleanup=args.do_cleanup,
                    do_regex = args.do_regex ,
                    do_marian_mt = args.do_marian_mt,
                    num_words_per_chunk=args.num_words_per_chunk,
                    ontology_weight=args.ontology_weight,
                    spacy_weight=args.spacy_weight,
                    hf_ner_weight=args.hf_ner_weight,
                    regex_weight=args.regex_weight,
                    backtrans_weight=args.backtrans_weight,
                    do_docs_trim_for_person=args.do_docs_trim_for_person,
                    do_docs_filter=args.do_docs_filter,
                    do_qg_rel=args.do_qg_rel,
                    do_kenlm = args.do_kenlm,
                    cutoff=cutoff,
                    batch_size=batch_size,
                    num_workers=num_workers)
      else:
        processor = TextAugment(single_process=True)
        if args.hfdataset:
            all_docs = [(processor.get_docs(src_lang[0], hfdataset=args.hfdataset, cutoff=cutoff), src_lang[0], target_lang[0])]
        elif not docs:
            all_docs = [(processor.get_docs(sl, cutoff=cutoff), sl, tl) for sl, tl in zip(src_lang, target_lang)]
        else:
            all_docs = [([docs], src_lang[0], target_lang[0])]

        if outfile is not None:
            _file =  open(outfile, 'w', encoding='utf-8')
        else:
            _file = None
        for docs_iter, src_lang, target_lang in all_docs:
            if outfile is None:
                if _file is not None: _file.close()
                _file = open(f"{src_lang}_out.jsonl", 'w', encoding='utf-8')
            #print(docs_iter)
            for docs in tqdm(docs_iter):
                #print(docs)
                docs =  processor.process_ner(docs=docs, 
                    src_lang=src_lang,
                    target_lang=target_lang,
                    do_spacy = args.do_spacy ,
                    do_hf_ner = args.do_hf_ner ,
                    do_ontology = args.do_ontology,
                    do_skip_src_lang_processing=args.do_skip_src_lang_processing,
                    do_backtrans=args.do_backtrans,
                    do_augment=args.do_augment,
                    do_anonymization=args.do_anonymization,
                    augment_lang=args.augment_lang,
                    do_cleanup=args.do_cleanup,
                    do_regex = args.do_regex ,
                    do_marian_mt = args.do_marian_mt,
                    num_words_per_chunk=args.num_words_per_chunk,
                    ontology_weight=args.ontology_weight,
                    spacy_weight=args.spacy_weight,
                    hf_ner_weight=args.hf_ner_weight,
                    regex_weight=args.regex_weight,
                    backtrans_weight=args.backtrans_weight,
                    do_docs_trim_for_person=args.do_docs_trim_for_person,
                    do_docs_filter=args.do_docs_filter,
                    do_qg_rel=args.do_qg_rel,
                    do_kenlm = args.do_kenlm,
                    cutoff=cutoff,
                    batch_size=batch_size)
                for doc in processor.serialize_ner_items(docs):
                    doc = json.dumps(doc)
                    _file.write(f'{doc}\n')
        if _file is not None: _file.close()
