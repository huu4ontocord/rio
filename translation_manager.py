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
import logging
from marian_mt import *
logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s : %(processName)s : %(levelname)s : %(message)s',
    level=logging.INFO)

m2m100_lang = {
    ('en', 'yo'): "Davlan/m2m100_418M-eng-yor-mt",
    ('yo', 'en'): "Davlan/m2m100_418M-yor-eng-mt",
    ('en', 'zu'): "masakhane/m2m100_418M-en-zu-mt",
    ('zu', 'en'): "masakhane/m2m100_418M-zu-en-mt",
    ('*', '*') : "facebook/m2m100_418M"
    }

translation_pipelines= {}
translation_tokenizers = {}

def batchify(lst, n):
    """Generate batches"""
    lst = list(lst)
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

def do_translations(text, src_lang='en', target_lang='hi', device="cpu", device_id=-1, batch_size=16, do_marian_mt=False):
    if type(text) is str:
      texts = [text]
    else:
      texts = text
    if not do_marian_mt:
      m2m_model_name = m2m100_lang.get((src_lang, target_lang), m2m100_lang[('*', '*')])
      if m2m_model_name not in translation_tokenizers:
        m2m_tokenizer = translation_tokenizers[m2m_model_name] = M2M100Tokenizer.from_pretrained(m2m_model_name, model_max_length=512)
      else:
        m2m_tokenizer = translation_tokenizers[m2m_model_name] 
      if True: #try:
        target_lang_bos_token = m2m_tokenizer.get_lang_id(target_lang)
      else: # except:
        do_marian_mt = True
    if not do_marian_mt:   
      if m2m_model_name in  translation_pipelines:
        m2m_model =  translation_pipelines[m2m_model_name]
      else:
        if device == "cpu":
            translation_pipelines[m2m_model_name] = m2m_model = M2M100ForConditionalGeneration.from_pretrained(m2m_model_name).eval()
            translation_pipelines[m2m_model_name] = m2m_model = torch.quantization.quantize_dynamic(m2m_model, {torch.nn.Linear}, dtype=torch.qint8)
        else:
            translation_pipelines[m2m_model_name] = m2m_model = M2M100ForConditionalGeneration.from_pretrained(m2m_model_name).eval().half().to(device)
      translations = []
      for src_text_list in batchify(texts, batch_size):
          try:
            batch = m2m_tokenizer(src_text_list, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
          except:
            logger.info ("could not tokenize m2m batch. falling back to marian_mt")
            do_marian_mt = True
            break

          gen = m2m_model.generate(**batch, forced_bos_token_id=target_lang_bos_token, no_repeat_ngram_size=4, ) #
          outputs = m2m_tokenizer.batch_decode(gen, skip_special_tokens=True)
          translations.extend(outputs)
    if not do_marian_mt:
          return translations

    translations = []
    model_name = marian_mt.get((src_lang, target_lang))
    mt_pipeline = None
    if model_name is not None and model_name not in translation_pipelines:
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512,truncation=True)
        if self.device == "cpu":
          model = MarianMTModel.from_pretrained(model_name).eval()
          model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        else:
          model = MarianMTModel.from_pretrained(model_name).eval().half().to(device)
        if self.device == 'cpu':
          mt_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)
        else:
          mt_pipeline = pipeline("translation", model=model, tokenizer=tokenizer, device=device_id)
        if mt_pipeline is None:
          raise RuntimeError("no translation pipeline") # we could do multi-step translation where there are no pairs
        translation_pipelines[model_name] = mt_pipeline
      
    for src_text_list in batchify(texts, batch_size):
        outputs = [t['translation_text'] for t in mt_pipeline(src_text_list, batch_size=batch_size,  truncation=True, max_length=512)]
        translations.extend(outputs)
    if type(text) is str:
      return translations[0]
    return translations


