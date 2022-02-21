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
sys.path.append(os.path.abspath(os.path.dirname(__file__)))         
from marian_mt import marian_mt
from edugp_kenlm_model import *
from fake_names import *
from pii_regexes import detect_ner_with_regex_and_context
from ontology.ontology_manager import OntologyManager
try:
  if not stopwords:
    from stopwords import stopwords
except:
  try:
    from stopwords import stopwords
  except:
    stopwords = {}
try:
  if not english_flagged_words:
    from flagged_words import *
except:
  try:
    from flagged_words import *
  except:
    english_flagged_words = {}
    flagged_words = {}
import qg_pipeline
def try_decode(text):
   try:
     return text.decode().strip()
   except:
     return None

faker_list = [
    'ar_AA',
    'ar_PS',
    'ar_SA',
    'bg_BG',
    'cs_CZ',
    'de_AT',
    'de_CH',
    'de_DE',
    'dk_DK',
    'el_GR',
    'en_GB',
    'en_IE',
    'en_IN',
    'en_NZ',
    'en_TH',
    'en_US',
    'es_CA',
    'es_ES',
    'es_MX',
    'et_EE',
    'fa_IR',
    'fi_FI',
    'fr_CA',
    'fr_CH',
    'fr_FR',
    'fr_QC',
    'ga_IE',
    'he_IL',
    'hi_IN',
    'hr_HR',
    'hu_HU',
    'hy_AM',
    'id_ID',
    'it_IT',
    'ja_JP',
    'ka_GE',
    'ko_KR',
    'lt_LT',
    'lv_LV',
    'ne_NP',
    'nl_NL',
    'no_NO',
    'or_IN',
    'pl_PL',
    'pt_BR',
    'pt_PT',
    'ro_RO',
    'ru_RU',
    'sl_SI',
    'sv_SE',
    'ta_IN',
    'th_TH',
    'tr_TR',
    'tw_GH',
    'uk_UA',
    'zh_CN',
    'zh_TW']

faker_map = {}

for faker_lang in faker_list:
  lang, _ = faker_lang.split("_")
  faker_map[lang] = faker_map.get(lang, []) + [faker_lang]

trannum = str.maketrans("0123456789", "1111111111")

class TextAugmentDeviceModel:

  available_devices = [-1] if torch.cuda.device_count() == 0 else [torch.cuda.device(i).idx for i in range(torch.cuda.device_count())]
  available_device_models  = [None] if torch.cuda.device_count() == 0 else [None]* torch.cuda.device_count()
  
  def __init__(self, device_id=None, device=None):
    if device_id is not None:
      self.device_id = int(device_id)
      self.device = "cpu" if device_id < 0 else "cuda:"+str(device_id)
    elif device is not None:
      self.device=device
      self.device_id = -1 if device == "cpu" else int(device.split(":")[-1])
    else:
      self.device = None
      self.device_id = None
    self.labse = None
    self.qg  = None
    self.translation_pipelines = None
    self.ner_model_name2pipelines = None
    self.marian_mt = None

  @staticmethod
  def initializer_all(src_langs=["en"], target_langs=["en"], aug_langs=["en"]):

    for available_device_model, device_id in zip(TextAugmentDeviceModel.available_device_models, TextAugmentDeviceModel.available_devices):
        if available_device_model is None:  
          available_device_model = TextAugmentDeviceModel(device_id=device_id)
        available_device_model.initializer(src_langs=src_langs, target_langs=target_langs, aug_langs=aug_langs)
        TextAugmentDeviceModel.available_device_models[max(0,available_device_model.device_id)] = available_device_model

  def initializer(self, device_id=None, device=None, src_langs=["en"], target_langs=["en"], aug_langs=["en"]):
    if device_id is not None:
      self.device_id = int(device_id)
      self.device = "cpu" if device_id < 0 else "cuda:"+str(device_id)
    elif device is not None:
      self.device=device
      self.device_id = -1 if device == "cpu" else int(device.split(":")[-1])
    else:
      self.device = None
      self.device_id = None
    if not hasattr(self, 'qg') or self.qg is None: self.qg = qg_pipeline.pipeline("multitask-qa-qg", device=self.device) # TODO make sure it's running in half mode
    if not hasattr(self, 'labse') or self.labse is None: self.labse =  SentenceTransformer("sentence-transformers/LaBSE", cache_folder=os.path.expanduser ('~')+"/.cache").half().eval().to(self.device)
    if not hasattr(self, 'ner_model_name2pipelines') or self.ner_model_name2pipelines is None: self.ner_model_name2pipelines = {}
    if not hasattr(self, 'translation_pipelines') or self.translation_pipelines is None: 
      self.translation_pipelines  = {}
      self.translation_pipelines["facebook/m2m100_418M"] =  M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").eval().half().to(self.device)
    seen = {}
    pairs = list(set([(src_lang, target_lang) in zip(src_langs, target_langs+aug_langs)] + [(target_lang, src_lang) in zip(src_langs, target_langs+aug_langs)]))
    for pair in pairs: 
      if pair not in seen:
        model_name = marian_mt.get(pair)
        seen[pair] = 1
        if model_name is not None and model_name not in TextAugment.translation_pipelines:
          tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_len=512)
          if self.device == "cpu":
            model = MarianMTModel.from_pretrained(model_name).eval()
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
          else:
            model = MarianMTModel.from_pretrained(model_name).eval().half().to(self.device)
          if self.device == 'cpu':
            mt_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)
          else:
            mt_pipeline = pipeline("translation", model=model, tokenizer=tokenizer, device=self.device_id)
            self.translation_pipelines[model_name] = mt_pipeline                          

    for target_lang in list(set(target_langs + src_langs + aug_langs)):
      for model_name, model_cls, hf_ner_weight2 in TextAugment.hf_ner_model_map.get(target_lang, []):
          if model_name not in self.ner_model_name2pipelines:
            try:
              model = model_cls.from_pretrained(model_name).half().eval().to(self.device)
              tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_len=512)
              ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=self.device_id)
              self.ner_model_name2pipelines[model_name] = ner_pipeline
              logger.info("problems loading model and tokenizer for pipeline. attempting to load without passing in model")
            except:
              ner_pipeline = pipeline("ner",  model=model_name, tokenizer=(model_name, {"use_fast": True},), device=self.device_id)
              self.ner_model_name2pipelines[model_name] = ner_pipeline


class TextAugment:
  device_id = None
  device = None
  ner_model_name2pipelines = {}
  translation_pipelines = {}
  qg = None
  labse = None
  m2m_model_name = ""
  m2m_tokenizer = None
  en_spacy_nlp = None
  faker_en_list  = None
  kenlm_model = None
  
  # see https://www.researchgate.net/publication/259179064_Comparing_Methods_for_Detecting_Child_Exploitation_Content_Online for common CSAM words
  # http://antipaedo.lip6.fr/T12/keywords_cnrs.pdf - for top 20 from table 7 and 8 of the paper, plus other similar words , ignoring stopwords like "tuesday"
  # WARNING: Translations are probably not accurate. TODO to fix.
  # we are only doing our current covered langauges. TODO, do for all languages.
  banned_words = {
      'ar': {"فتاة صغيرة" , "فتاة صغيرة" , "فتى صغير", "مراهق" , "لولي" , "لولا" , "لوليتا" , "لولي" , "لولي" , "لوليتاغوي" , "فتاة" , "صبي" ,"طفل" , "حمام" },
      'ur': {'نوجوان لڑکی', 'چھوٹی لڑکی', 'چھوٹا لڑکا', 'نوجوان لڑکا','نوعمر','لولی','لولا','لولیتا','لولی','لولی','لولیتاگو','لڑکی','لڑکا','بچہ','بچہ','غسل', 'غسل', 'غسل'},
      'bn': {'ছোট মেয়ে', 'ছোট মেয়ে', 'ছোট ছেলে', 'কিশোর', 'ললি', 'লোলা', 'ললিতা', 'ললি', 'লোলি', 'ললিটাগুই', 'মেয়ে', 'ছেলে' , 'শিশু', 'স্নান',},
      'hi': {'युवा लड़की', 'छोटी लड़की', 'छोटा लड़का', 'किशोर', 'लॉली', 'लोला', 'लोलिता', 'लल्ली', 'लोली', 'लोलितागुय', 'लड़की', 'लड़का' , 'बच्चा', 'स्नान',},
      'eu': {'neska gaztea', 'neska txikia', 'mutil txikia', 'nerabea', 'neska', 'mutil' , 'haurra', 'bainua',},
      'ca': {'noia', 'nena', 'nen petit', 'nen' , 'nen', 'banyer',},
      'vi': {'thiếu niên', 'cậu nhỏ', 'cậu bé', 'cô gái trẻ', 'cô bé', 'cậu bé', 'trẻ em', 'tắm', },
      'zh': {'儿童','女孩','孩子', '小女孩', '小姑娘','小男孩', '年輕女孩','年轻女孩','年轻的女孩','洗澡','洛丽塔','洛麗塔','浴','男孩','萝拉','萝莉','萝莉塔吉','蘿拉','蘿莉','蘿莉塔','青少年'},
      'fr': {'jeune fille','petite fille','petit garçon','ado',  'fille', 'garçon' , 'enfant', 'bain',},
      'id': {'gadis muda','gadis kecil','anak laki-laki kecil','remaja',  'perempuan', 'laki-laki' , 'anak', 'mandi',},
      'fa': {'دختر جوان',  'دختر کوچولو',  'پسر کوچولو',  'نوجوان',  'لولی',  'لولا',  'لولیتا',  'لولی',  'لولی',  'لولیتاگو',  'دختر',  'پسر' ,'کودک',  'حمام', },
      'es': {'niña',  'niño', 'adolescente', 'baño',},
      'pt': {'menina', 'menino', 'adolescente', 'pirulito',  'criança', 'banho',},
      'ig': {'nwa agbọghọ', 'nwa agbọghọ', 'nwa agbọghọ',' iri na ụma', 'nwa agbọghọ', 'nwoke' , 'nwa', },
      'sw': {'msichana mdogo','msichana mdogo','kijana mdogo', 'mtoto', 'kuoga',},
      'yo': {'kekere', 'omobinrin', 'omokunrin', 'ọmọ', 'wẹwẹ',},
      'xh': {'intombazana encinci', 'intsha', 'umntwana', 'hlamba', 'inkwenkwe', },
      'zu': {'intombazane', 'intsha', 'intombazane encane',  'umfana omncane','geza', 'ingane', 'yomfana'},
      'default': {'young girl', 'little girl','little boy', 'young boy', 'teen', 'lolli', 'lola', 'lolita', 'lolly', 'loli', 'lolitaguy', 'girl', 'boy', 'child', 'kid',  \
                  'bath', 'baths', 'bathing', "pedo", 'nymphet', 'nimphet', 'babyj', 'voglia', 'eurololita', '349', 'hussyfan', 'kidzilla', 'raygold', 'ygold', 'qwerty', 'qqaazz', 'ptsc', \
                  'pthc', 'nn', 'tanta', 'mylola', 'arina', 'newstar', 'playtoy', 'imouto', 'lourinha', 'amateurz', 'kacy', 'vicky', 'lsm', 'sandra', \
                  'babyshivid', 'shiori', 'tvg', 'chiharu','kidzilla', 'izzy', 'rika', 'kdquality', 'cbaby', 'nablot', 'lso',  'kinderficker', \
                  'yo',  'yr',  }
  }
  # note that we do not have a transformer model for catalan, but  we use transfer learning from Davlan/xlm-roberta-base-ner-hrl. We could also use spacy's catalan model
  hf_ner_model_map = {
      "sn": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]], # consider using one of the smaller models
      "st": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]], # consider using one of the smaller models
      "ny": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]], # consider using one of the smaller models
      "xh": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]], # consider using one of the smaller models
      "zu": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]], # consider using one of the smaller models
      "sw": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]], # consider using one of the smaller models
      "yo": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0 ]],
      "ig": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0 ]],
      "ar": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0]],
      "en": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0], ["bioformers/bioformer-cased-v1.0-ncbi-disease", BertForTokenClassification, 1.0]], #["jplu/tf-xlm-r-ner-40-lang", None ],
      "es": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ]],
      "eu": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      "ca": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      "pt": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ]],
      "fr": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ]],
      "zh": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ]],
      'vi': [["lhkhiem28/COVID-19-Named-Entity-Recognition-for-Vietnamese", RobertaForTokenClassification, 1.0]],#["jplu/tf-xlm-r-ner-40-lang", None ],  # jplu/tf-xlm-r-ner-40-lang is breaking CPU mode
      'hi': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]], #["jplu/tf-xlm-r-ner-40-lang", None, 1.0 ]],  # jplu/tf-xlm-r-ner-40-lang is breaking CPU mode
      'bn': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]], #["jplu/tf-xlm-r-ner-40-lang", None, 1.0 ]],  # jplu/tf-xlm-r-ner-40-lang is breaking CPU mode
      'ur': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]], #["jplu/tf-xlm-r-ner-40-lang", None, 1.0 ]],  # jplu/tf-xlm-r-ner-40-lang is breaking CPU mode
      'id': [["cahya/bert-base-indonesian-NER", BertForTokenClassification, 1.0]],
      
      # NOT PART OF OUR LANGUAGE SET. EXPERIMENTAL
      'he': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]], #["jplu/tf-xlm-r-ner-40-lang", None, 1.0 ]], # jplu/tf-xlm-r-ner-40-lang is breaking CPU mode
      'hr': [["classla/bcms-bertic-ner", ElectraForTokenClassification, 1.0]],
      'bs': [["classla/bcms-bertic-ner", ElectraForTokenClassification, 1.0]],
      'sr': [["classla/bcms-bertic-ner", ElectraForTokenClassification, 1.0]],
      'cnr': [["classla/bcms-bertic-ner", ElectraForTokenClassification, 1.0]],
      'hbs': [["classla/bcms-bertic-ner", ElectraForTokenClassification, 1.0]],
      'da': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      'no': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      'nb': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      'nn': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      'sv': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      'fo': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      'is': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      }

  #wikipedia kenlm model based on prompt "f{s} (born"
  #TODO figure out actual numbers. Also, add languge specific kenlm models
  public_figure_kenlm_cutoff_map = {'en': 500,
                                    'yo': 500,
                                    'zu': 500,
                                    'sn': 500,
                                    'st': 500,
                                    'ny': 500,
                                    'xh': 500,
                                    'sw': 500,
                                    'ig': 500,
                                    'ar': 500,
                                    'en': 500,
                                    'es': 500,
                                    'eu': 500,
                                    'ca': 500,
                                    'pt': 500,
                                    'fr': 500,
                                    'zh': 500,
                                    'vi': 500,
                                    'hi': 500,
                                    'ur': 500,
                                    'id': 500,
                                    'bn': 500,
                                    }
  m2m100_lang = {
    ('en', 'yo'): "Davlan/m2m100_418M-eng-yor-mt",
    ('yo', 'en'): "Davlan/m2m100_418M-yor-eng-mt",
    ('en', 'zu'): "masakhane/m2m100_418M-en-zu-mt",
    ('zu', 'en'): "masakhane/m2m100_418M-zu-en-mt",
    ('*', '*') : "facebook/m2m100_418M"
    }

  strip_chars = " ,،、{}[]|()\"'“”《》«»!:;?。…．"
  punc_char = ".!:;?。…．"
  special_char = " ,{}[]()|\\\"'“”《》«»~!@#$%^&*{}[]()_+=-0987654321`<>,、،./?':;“”\"\t\n\\πه☆●¦″．۩۱（☛₨➩°・■↑☻、๑º‹€σ٪’Ø·−♥ıॽ،٥《‘©。¨﴿！★×✱´٬→±x：¹？£―▷ф¡Г♫∟™ª₪®▬「—¯；¼❖․ø•�」٣，٢◦‑←§١ー٤）˚›٩▼٠«¢¸٨³½˜٭ˈ¿¬ι۞⌐¥►†ƒ∙²»¤…﴾⠀》′ا✓→¶'"
  junk = set(",{}[]()|\\\"'“”《》«»~!@#$%^&*{}[]()_+=-0987654321`<>,、،./?':;“”\"\t\n\\πه☆●¦″．۩۱（☛₨➩°・■↑☻、๑º‹€σ٪’Ø·−♥ıॽ،٥《‘©。¨﴿！★×✱´٬→±x：¹？£―▷ф¡Г♫∟™ª₪®▬「—¯；¼❖․ø•�」٣，٢◦‑←§١ー٤）˚›٩▼٠«¢¸٨³½˜٭ˈ¿¬ι۞⌐¥►†ƒ∙²»¤…﴾⠀》′ا✓→¶'")
  #don't add a space for junk chars
  ontology_manager = None
  max_stoword_len_zh = max([0]+[len(a) for a in stopwords.get('zh', [])])
  max_stoword_len_ko = max([0]+[len(a) for a in stopwords.get('ko', [])])
  max_stoword_len_ja = max([0]+[len(a) for a in stopwords.get('ja', [])])
  stopwords_en = set(stopwords.get('en',[]))
  cache_dir = None

  def __init__(self, device=None, single_process=1, available_device_model=None, labse=None, ontology_manager=None, translation_pipelines=None, ner_model_name2pipelines=None, en_spacy_nlp=None, faker_en_list=None, qg=None, kenlm_model=None, cache_dir=None):
    if cache_dir is None: 
        cache_dir = os.path.expanduser ('~')+"/.cache"
    if TextAugment.cache_dir is None: 
        TextAugment.cache_dir = cache_dir
    if device is not None:
      TextAugment.device = device
      if device == "cpu": 
        TextAugment.device_id = -1
      else:
        TextAugment.device_id = int(device.split(":")[-1])
    else:
      if TextAugmentDeviceModel.available_devices:
        TextAugment.device_id = -1 if TextAugmentDeviceModel.available_devices[0] == -1 else random.choice(TextAugmentDeviceModel.available_devices)
        TextAugment.device = "cpu" if TextAugment.device_id == -1 else "cuda:"+str(TextAugment.device_id) 
      else:
        TextAugment.device_id = -1
        TextAugment.device = "cpu"  
    logger.info (('running on ', TextAugment.device))
    if single_process:
      self.initializer(available_device_model=available_device_model, device=TextAugment.device, labse=labse, ontology_manager=ontology_manager, translation_pipelines=translation_pipelines, ner_model_name2pipelines=ner_model_name2pipelines, en_spacy_nlp=en_spacy_nlp, faker_en_list=faker_en_list, qg=qg, kenlm_model=kenlm_model, cache_dir=cache_dir)
    
  def initializer(self, device_id_by_proess_id=True, all_available_device_model=None, available_device_model=None, device=None,  labse=None, ontology_manager=None, translation_pipelines=None, ner_model_name2pipelines=None, en_spacy_nlp=None, faker_en_list=None, qg=None, kenlm_model=None, cache_dir=None):
    if all_available_device_model is not None:
      TextAugmentDeviceModel.available_device_models   = all_available_device_model
      TextAugmentDeviceModel.available_devices = [d.device_id for d in all_available_device_model]
        
    if cache_dir is None: 
        cache_dir = os.path.expanduser ('~')+"/.cache"
    if TextAugment.cache_dir is None: 
        TextAugment.cache_dir = cache_dir
    if device is not None:
      TextAugment.device = device
      if device == "cpu": 
        TextAugment.device_id = -1
      else:
        TextAugment.device_id = int(device.split(":")[-1])
    else:
      if TextAugmentDeviceModel.available_devices:
        TextAugment.device_id = -1 if TextAugmentDeviceModel.available_devices[0] == -1 else random.choice(TextAugmentDeviceModel.available_devices)
        TextAugment.device = "cpu" if TextAugment.device_id == -1 else "cuda:"+str(TextAugment.device_id) 
      else:
        TextAugment.device_id = -1
        TextAugment.device = "cpu" 

    device = TextAugment.device
    if available_device_model is not None:
      TextAugmentDeviceModel.available_device_models  [max(0,available_device_model.device_id)] = available_device_model
      device_id = available_device_model.device_id
      TextAugment.device = device = available_device_model.device
      labse = available_device_model.labse
      qg = available_device_model.qg
      translation_pipelines = available_device_model.translation_pipelines
      ner_model_name2pipelines = available_device_model.ner_model_name2pipelines
    else:
      if device is None:
        if TextAugmentDeviceModel.available_devices and TextAugmentDeviceModel.available_devices[0].device_id >= 0:
          if device_id_by_proess_id:
            process_id = torch.multiprocessing.current_process().name.split("-")[-1]
            try:
              process_id = int(process_id)
            except:
              process_id = 0
            device_id = process_id % len(TextAugmentDeviceModel.available_devices)
          else:
            device_id = random.choice(TextAugmentDeviceModel.available_devices)
          TextAugment.device_id = device_id
          device = TextAugment.device = "cuda:"+str(TextAugment.device_id) 
        else:
          device_id = TextAugment.device_id = -1
          device = TextAugment.device = "cpu"  
      else:
        device_id = -1 if device == "cpu" else int(device.split(":")[-1])
      if True:
        #print (device_id)
        available_device_model = TextAugmentDeviceModel.available_device_models[max(0,device_id)]
        if available_device_model is None: 
          TextAugmentDeviceModel.available_device_models[max(0,device_id)] = available_device_model = TextAugmentDeviceModel(device=TextAugment.device)
        labse = available_device_model.labse
        qg = available_device_model.qg
        translation_pipelines = available_device_model.translation_pipelines
        ner_model_name2pipelines = available_device_model.ner_model_name2pipelines
    
    if labse is not None: TextAugment.labse = labse 
    if translation_pipelines is not None: TextAugment.translation_pipelines = translation_pipelines
    if ner_model_name2pipelines is not None: TextAugment.ner_model_name2pipelines = ner_model_name2pipelines
    if qg is not None: TextAugment.qg = qg
    if ontology_manager is not None: TextAugment.ontology_manager = ontology_manager
    if en_spacy_nlp is not None: TextAugment.en_spacy_nlp = en_spacy_nlp
    if faker_en_list is not None: TextAugment.faker_en_list = faker_en_list
    if kenlm_model is not None: TextAugment.kenlm_model = kenlm_model
    if TextAugment.en_spacy_nlp is None: TextAugment.en_spacy_nlp = spacy.load('en_core_web_sm')
    try:
        coref = neuralcoref.NeuralCoref(TextAugment.en_spacy_nlp.vocab)
        TextAugment.en_spacy_nlp.add_pipe(coref, name='neuralcoref')
        #we could potentially add new items to the vocabulary to improve coref.
    except:
        logger.info("Neuralcoref not loaded. Using normal spacy")
        pass

    if TextAugment.ontology_manager is None: TextAugment.ontology_manager = OntologyManager('en') #src_lang=src_lang
    #speed up loading if we don't use kenlm models
    #if TextAugment.kenlm_model is None: 
    #  TextAugment.load_kenlm_model()
    if TextAugment.faker_en_list is None:
      TextAugment.faker_en_list  = faker_en_list = [Faker(faker_lang) for faker_lang in faker_map["en"]]
      for faker_en in faker_en_list:
          faker_en.add_provider(person)
          faker_en.add_provider(ssn)
          faker_en.add_provider(address)
          faker_en.add_provider(geo)
          faker_en.add_provider(internet)
          faker_en.add_provider(company)
    #print ("finished load")
    #TODO - create an abstraction for faker, so when faker returns None, we fallback to faker_en


  #TODO: we also need a deserialize
  @staticmethod
  def serialize_ner_items(docs, ner_keys=None, outfile=""):
        #print ("serialize_ner_items")
        # serialize ner keys
        if ner_keys:
          ner_keys = [k + '_ner' for k in ner_keys if '_ner' not in k]
        else:
          ner_keys = []
        if type(docs) is dict:
          serialize_docs = list(docs.values())
        else:
          serialize_docs = docs

        serialize_docs.sort(key=lambda a: int(a.get('id', -1)))
        serialize_docs = copy.copy(serialize_docs)
        for doc in serialize_docs:
            for ner_key in ner_keys + ([] if ner_keys else [key for key in doc if key.endswith('_ner')]):
                ner_items = doc[ner_key]
                serialize_items = []
                for (text, start, end), ner_value in ner_items.items():
                    ner_value = list(ner_value.items())
                    ner_dict = [text, start, end, ner_value]
                    serialize_items.append(ner_dict)
                doc[ner_key] = serialize_items
        if outfile:       
          with open(outfile, 'w', encoding='utf-8') as file:
            for doc in serialize_docs:
              doc = json.dumps(doc)
              file.write(f'{doc}\n')
        return serialize_docs

  @staticmethod
  def deserialize_ner_items(docs=None, infile="", return_dict=False):
    def load_py_from_str(s, default=None):
      if not s.strip(): return default
      dat = None
      try:
        dat = json.loads(s)
      except:
        pass
      if dat is not None: return dat
      ret = {'__ret': None}
      #print (s)
      exec("__ret= "+s, ret)
      return ret['__ret']
      
    def deserialize_doc(doc):
      for ner_key in [key for key in doc if key.endswith('_ner')]:
                ner_items = doc[ner_key]
                if ner_items and type(ner_items) is list:
                  deserialize_items = {}
                  for item in ner_items:
                    mention = (item[0], item[1], item[2])
                    aHash = dict([(tuple(a[0]) if type(a[0]) is list else a[0], float(a[1])) for a in item[3]])
                    deserialize_items[mention] = aHash
                  doc[ner_key] = deserialize_items
      return doc

    #print ("deserialize_ner_items")
    if infile:
      docs= [load_py_from_str(s, {}) for s in open(infile, "rb").read().decode().split("\n")]
      #print (docs)
    elif docs:
      if type(docs) is dict:
        docs = copy.copy(docs.values())
        return_dict = True
      else:
        docs = copy.copy(docs)
    if return_dict:
      return dict([(int(doc.get('id', idx)), deserialize_doc(doc)) for idx, doc in enumerate(docs)])
    else:
      return [deserialize_doc(doc) for doc in docs]
    return docs

  @staticmethod
  def get_lang_groups(src_lang):
    lang_groups=[src_lang]
    if src_lang in ('ig', 'sn', 'ny', 'st', 'zu', 'xh', 'rw', 'sw', 'yo'):
      lang_groups = ['ig', 'sn', 'ny', 'st', 'zu', 'xh', 'rw', 'sw', 'yo']  
    elif src_lang in ('mr', 'ne', 'hi', ):
      lang_groups = ['mr', 'ne', 'hi', ]  
    elif src_lang in ('pt', 'gl'):
      lang_groups = ['pt','gl','la' ]  
    elif src_lang in ('fr', 'br'):
      lang_groups = ['fr','la', 'br' ]  
    elif src_lang in ('es', 'oc', 'ca', 'eu', 'an', 'gl' ):
      lang_groups = ['es', 'oc', 'ca', 'eu', 'an', 'gl', 'la' ]  
    elif src_lang in ('arz', 'ar', 'fa', 'ur', 'az', 'azb', 'ckb' ):
      lang_groups = ['arz', 'ar', 'fa', 'ur', 'az', 'azb', 'ckb' ]  
    elif src_lang in ('id', 'ms', ):
      lang_groups = ['id', 'ms',]  
    elif src_lang in ('as', 'bn', 'bpy'):
      lang_groups = ['as', 'bn', 'bpy']  
    elif src_lang in ('af', 'nl', ):
      lang_groups = ['af', 'nl',]  
    elif src_lang in ('bo', 'dz', ):
      lang_groups = ['bo', 'dz',]  
    elif src_lang in ('bs', 'hr', ):
      lang_groups = ['bs', 'hr',]  
    elif src_lang in ('bxr', 'mn', ):
      lang_groups = ['bxr', 'mn',]  
    elif src_lang in ('ceb', 'tl', ):
      lang_groups = ['ceb', 'tl',]  
    elif src_lang in ('cs', 'sk', ):
      lang_groups = ['cs', 'sk',]  
    elif src_lang in ('da', 'no', ):
      lang_groups = ['da', 'no',] 
    elif src_lang in ('eml', 'wa', ):
      lang_groups = ['eml', 'wa',] 
    elif src_lang in ('de', 'lb', 'pl', 'dsb'):
      lang_groups = ['de', 'lb', 'pl', 'dsb'] 
    elif src_lang in ('av', 'ru', 'bg', 'ba', 'kk', 'uk', 'be', 'ce', 'cv'):
      lang_groups = ['av', 'ru', 'bg', 'ba', 'kk', 'uk', 'be', 'ce', 'cv']  
    return set(lang_groups)

  @staticmethod
  def load_kenlm_model(store_model=True):
      if TextAugment.cache_dir == None:
        cache_dir = os.path.expanduser ('~')+"/.cache"
      else:
        cache_dir = TextAugment.cache_dir
      os.system(f"mkdir -p {cache_dir}/wikipedia")
      if not os.path.exists(f"{cache_dir}/wikipedia/en.arpa.bin"): 
        file_url= hf_hub_url(repo_id="edugp/kenlm", filename="wikipedia/en.arpa.bin")
        file = cached_download(file_url)
        os.system(f"ln -s {file} {cache_dir}/wikipedia/en.arpa.bin")
      if not os.path.exists(f"{cache_dir}/wikipedia/en.sp.model"): 
        file_url= hf_hub_url(repo_id="edugp/kenlm", filename="wikipedia/en.sp.model")
        file = cached_download(file_url)
        os.system(f"ln -s {file} {cache_dir}/wikipedia/en.sp.model")
      if not os.path.exists(f"{cache_dir}/wikipedia/en.sp.vocab"):
        file_url= hf_hub_url(repo_id="edugp/kenlm", filename="wikipedia/en.sp.vocab")
        file = cached_download(file_url)
        os.system(f"ln -s {file} {cache_dir}/wikipedia/en.sp.vocab")
      if store_model: TextAugment.kenlm_model = KenlmModel(f"{cache_dir}/wikipedia", "en")
  
  @staticmethod
  def check_good_sentence(s, src_lang, stopwords, show_err=False, lang_groups=[], ret_score=False, stopword_ratio_cutoff=0.06, bannedwords=None, flagged_words=None, badword_ratio_cutoff=0.15,  junk_ratio=0.16, max_badword_len=5):
    #basic dejunk
    # for flagged_words, only filter out if the ratio is exceeded AND there exists one banned word
    if bannedwords is None:
      bannedwords = TextAugment.banned_words.get(src_lang, TextAugment.banned_words['default'])
    default_bannedwords = TextAugment.banned_words['default']
    s = s.lower().strip()
    if not s: 
       return False
    jr = len([s2 for s2 in s if s2 in TextAugment.junk])/len(s)
    if jr >= junk_ratio:
      return False
    if src_lang in ("ja", "ko", "zh"):
      sArr = s
    else:
      sArr = [s2.strip(TextAugment.special_char) for s2 in s.lower().split() if s2.strip(TextAugment.special_char)]
    if len(sArr) == 0:
      return False
    bad_score = 0.0
    if flagged_words:
      if src_lang not in ("ja", "ko", "zh") and len([s2 for s2 in sArr if s2 in flagged_words])/len(sArr) > badword_ratio_cutoff:
        if any(s2 for s2 in sArr if s2 in bannedwords) or any(s2 for s2 in sArr if s2 in default_bannedwords):
          #print ('bw', len([s2 for s2 in sArr if s2 in flagged_words])/len(sArr))
          return False
        else:
          bad_score = len([s2 for s2 in sArr if s2 in flagged_words])/len(sArr)
      if src_lang in ("ja", "ko", "zh"):
        badword_ratio_cutoff /= 100
        len_s = len(s)
        bad_cnt = 0
        total_cnt = 0
        for i in range(len_s):
          for j in range(i+1,min(len_s, i+max_badword_len)):
            if s[i:j] in flagged_words:
              bad_cnt += 1
            total_cnt += 1
        bad_score = (bad_cnt/total_cnt)
        if bad_score > badword_ratio_cutoff:
          for bword in bannedwords:
            if bword in s:
              return False
          for bword in default_bannedwords:
            if bword in s:
              return False
          
    #stopword check
    if stopwords:
      #TODO: catch multi word with spaces
      if src_lang not in ("ja", "ko", "zh") and len([s2 for s2 in sArr if s2 in stopwords])/len(sArr) < stopword_ratio_cutoff:
        #print ('sw', len([s2 for s2 in sArr if s2 in stopwords])/len(sArr))
        return False
      if src_lang in ("ja", "ko", "zh"):
        if src_lang == "zh":
          max_stoword = TextAugment.max_stoword_len_zh
        elif src_lang == "ko":
          max_stoword = TextAugment.max_stoword_len_ko
        elif src_lang == "ja":
          max_stoword = TextAugment.max_stoword_len_ja
        len_s = len(s)
        stop_cnt = 0
        total_cnt = 0
        for i in range(len_s):
          for j in range(i+1,min(len_s, i+max_stoword)):
            if s[i:j] in stopwords:
              stop_cnt += 1
            total_cnt += 1
        #print ('stopword', (stop_cnt/total_cnt) )
        if (stop_cnt/total_cnt) < stopword_ratio_cutoff:
          return False
    #langid check
    try:
        lang =  langid.classify(s)[0]
    except:
        return True
    if show_err and lang != src_lang and lang not in lang_groups:
      logger.info ((src_lang, lang))
    if ret_score: return lang == src_lang or lang in lang_groups, bad_score
    return lang == src_lang or lang in lang_groups

  #WIP - we can use this question generation method to extract people, place and thing, and potentially age/date AND to get a relationship between a person and a PII info
  def generate_questions_answers_rel(self, docs, chunks, src_lang, default_answers=[], text_key=None, ner_key=None, rel_key=None, signal='qg_rel', weight=1.0):
    answers = {}

    if ner_key is None:
      ner_key = f'{src_lang}_signal_ner'
    if text_key is None:
      text_key = f'{src_lang}_text'
    if rel_key is None:
      rel_key = f'{src_lang}_rel'
    i= 0
    allqa = []
    for chunk in chunks:
      text = chunk[text_key]
      _id = chunk['id']
      ner = docs[_id][ner_key] = docs[_id].get(ner_key,{})
      rel = docs[_id][rel_key] = docs[_id].get(rel_key,{})
      default_answers = list(set([a[0] for a in ner.keys()]+default_answers))
      answers1={}
      #ti = time.time()
      text = text.replace("\n", " ").replace(",", " , ").replace("  ", " ").strip().replace(" , ", ", ") 
      aHash = self.qg(text , default_answers=default_answers)[0]

      allqa.append(aHash)
      #default_answers = list(set([a['answer'] for a in aHash]+default_answers))
      #print (aHash)
      for aHash1 in aHash:
        i+=1
        quest=aHash1['question'].lower().strip("?").replace("'s",  " 's").replace("  ", " ").split()
        question=aHash1['question'].lower()
        answer=aHash1['answer'].lower()
        label=None
        #TODO, use spacy_en to get NER and only fall back to "who", "when", "where" to determine ner if we find nothing
        if quest[0] == "who" and aHash1['answer'][-1] =='s':
          label="ORG"
        elif quest[0] == "who":
          label="PERSON"
          if "'s" in quest:
            for j in range(len(quest)):
              if j > 0 and quest[j-1]=="'s":
                label = "MISC"
                break
        elif quest[0] == "where":
          label="LOC"
        elif quest[0] == "when":
          label = "DATE"
          if "born" in quest or "died" in quest or "birthday" in quest or "birthdate" in quest:
            label="AGE"
        elif quest[0] == "why":
          label="EVENT"
        elif quest[0] == "how" and quest[1] in ("much", "many"):
          label="ORDINAL"
        elif quest[0] == "how":
          label="EVENT"
        elif quest[0] in ("which", "what") and quest[1] not in self.stopwords_en:
          label="MISC"
        else:
          label = None
        if label:
          mentions = [mention for mention in ner if (mention[0] == aHash1['answer'] or mention[0].startswith(aHash1['answer']) or aHash1['answer'].startswith(mention[0]))]
          if mentions:
            for mention in mentions:
              ner[mention][(label, signal)] = ner[mention].get((label, signal), 0) + weight
          else:
            pos = 0
            while aHash1['answer'] in text[pos:]:
              i = text[pos:].index(aHash1['answer'])
              start = pos + i
              end = start + len(aHash1['answer'])
              pos = end + 1
              ner[mention][(label, signal)] = ner[mention].get((label, signal), 0) + weight

          for mention in ner:
            ent = mention[0].lower()
            if ent in question:
              for mention0 in mentions:
                rel[question] = rel.get(question, []) + [(mention0, mention)]                               
    return docs  

  @staticmethod
  def get_aligned_text(sent1, sent2, src_lang, prefer_split_char="]"):
    """
    Given two sentences, find blocks of text that match and that don't match.
    return the blocks, and a matching score.
    Used to extract NER from original language sentence.
    """
    #print ("get_aligned_text")
    #will the below have a side-effect?
    sent1 = sent1.replace("。", ".").replace("،", ",").replace("、", ",").replace("`", "'").replace("“", "\"").replace("”", "\"").replace("《", "\"").replace("》", "\"").replace("«", "\"").replace("»", "\"")
    sent2 = sent2.replace("。", ".").replace("،", ",").replace("、", ",").replace("`", "'").replace("“", "\"").replace("”", "\"").replace("《", "\"").replace("》", "\"").replace("«", "\"").replace("»", "\"")
    if True: # src_lang in ("ja", "ko", "zh"):
      # splitting on spaces doesn't always work because some languages aren't space separated
      sep = ""
    else:
      sep = " "
      sent1 = sent1.split()
      sent2 = sent2.split()
    aMatch = difflib.SequenceMatcher(None,sent1, sent2)
    score = aMatch.ratio()
    blocks = aMatch.get_matching_blocks()
    blocks2 = []
    prevEndA = 0
    prevEndB = 0
    matchLen = 0
    nonMatchLen = 0
    #print (blocks)
    for blockI in range(len(blocks)):
      if blockI > 0 or (blockI==0 and (blocks[blockI][0] != 0 or blocks[blockI][1] != 0)):
        blocks3 = []
        if True:
          a, b = sep.join(sent1[prevEndA:blocks[blockI][0]]), sep.join(sent2[prevEndB:blocks[blockI][1]])
          if "]" in b:
            blocks3 = []
            a_arr = a.split(" ") if src_lang not in ("zh", "ja", "ki") else a
            len_a_arr = len(a_arr)
            b_cnt = b.count(prefer_split_char) + (1 if b.endswith(prefer_split_char) or b.endswith(prefer_split_char+" ") else 0)
            a_step = int(len(a)/b_cnt)
            #print (len(a), b_cnt)
            a_arr2 = []
            if a_step <= 0:
              a_arr2.extend([a]+['']*len(a))
            else:
              for rng in range(0, len_a_arr, a_step):
                a_arr2.append((" " if src_lang not in ("zh", "ja", "ki") else "").join(a_arr[rng:min(len_a_arr, rng+a_step)]))
            for a1, b1 in zip(a_arr2, b.split("]")):
                if src_lang not in ("zh", "ja", "ki"):
                  a1 = a1+" "
                b1 = b1+prefer_split_char
                blocks3.append([a1, b1, 0])

        if blocks3:
          blocks2.extend(blocks3)
        else:
          blocks2.append([sep.join(sent1[prevEndA:blocks[blockI][0]]), sep.join(sent2[prevEndB:blocks[blockI][1]]), 0])
        nonMatchLen += max(blocks[blockI][0] - prevEndA, blocks[blockI][1] - prevEndB)
      if blocks[blockI][2] != 0:
        blocks2.append([sep.join(sent1[blocks[blockI][0]:blocks[blockI][0]+blocks[blockI][2]]), sep.join(sent2[blocks[blockI][1]:blocks[blockI][1]+blocks[blockI][2]]), 1])
        prevEndA = blocks[blockI][0]+blocks[blockI][2]
        prevEndB = blocks[blockI][1]+blocks[blockI][2]
        matchLen += blocks[blockI][2]
    score = float(matchLen+1)/float(nonMatchLen+1)
    return (blocks2, score+score)

  def do_translations(self, texts, src_lang='en', target_lang='hi', batch_size=16, do_marian_mt=False):
    #print ("do_translations")
    #print ([len(t.split()) for t in texts])
    if not do_marian_mt:
      m2m_model_name = self.m2m100_lang.get((src_lang, target_lang), self.m2m100_lang[('*', '*')])
      if m2m_model_name != self.m2m_model_name or self.m2m_tokenizer is None:
        self.m2m_tokenizer = M2M100Tokenizer.from_pretrained(m2m_model_name, model_max_len=512)
      try:
        self.m2m_tokenizer.src_lang = src_lang
        target_lang_bos_token = self.m2m_tokenizer.get_lang_id(target_lang)
      except:
        do_marian_mt = True
        pass
      if not do_marian_mt:
        if m2m_model_name != self.m2m_model_name or self.m2m_model is None:
          if m2m_model_name in  TextAugment.translation_pipelines:
            self.m2m_model =  TextAugment.translation_pipelines[m2m_model_name]
          else:
            if self.device == "cpu":
                TextAugment.translation_pipelines[m2m_model_name] = self.m2m_model = M2M100ForConditionalGeneration.from_pretrained(m2m_model_name).eval()
                TextAugment.translation_pipelines[m2m_model_name] = self.m2m_model = torch.quantization.quantize_dynamic(self.m2m_model, {torch.nn.Linear}, dtype=torch.qint8)
            else:
                TextAugment.translation_pipelines[m2m_model_name] = self.m2m_model = M2M100ForConditionalGeneration.from_pretrained(m2m_model_name).eval().half().to(self.device)
        self.m2m_model_name = m2m_model_name
        translations = []
        for src_text_list in self.batch(texts, batch_size):
          try:
            batch = self.m2m_tokenizer(src_text_list, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
          except:
            logger.info ("could not tokenize m2m batch. falling back to marian_mt")
            do_marian_mt = True
            break

          gen = self.m2m_model.generate(**batch, forced_bos_token_id=target_lang_bos_token, no_repeat_ngram_size=4, ) #
          outputs = self.m2m_tokenizer.batch_decode(gen, skip_special_tokens=True)
          translations.extend(outputs)
        if not do_marian_mt:
          return translations

    translations = []
    #marian_mt = self.marian_mt
    model_name = marian_mt.get((src_lang, target_lang))
    mt_pipeline = None
    if model_name is not None and model_name not in TextAugment.translation_pipelines:
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_len=512)
        if self.device == "cpu":
          model = MarianMTModel.from_pretrained(model_name).eval()
          model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        else:
          model = MarianMTModel.from_pretrained(model_name).eval().half().to(self.device)
        if self.device == 'cpu':
          mt_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)
        else:
          mt_pipeline = pipeline("translation", model=model, tokenizer=tokenizer, device=self.device_id)
        TextAugment.translation_pipelines[model_name] = mt_pipeline
        if mt_pipeline is None:
          raise RuntimeError("no translation pipeline") # we could do multi-step translation where there are no pairs
    mt_pipeline = self.translation_pipelines[model_name]
    for src_text_list in self.batch(texts, batch_size):
        outputs = [t['translation_text'] for t in mt_pipeline(src_text_list, batch_size=batch_size,  truncation=True, max_length=512)]
        translations.extend(outputs)
    return translations

  @staticmethod
  def cjk_detect(texts):
    # korean
    if re.search("[\uac00-\ud7a3]", texts):
        return "ko"
    # japanese
    if re.search("[\u3040-\u30ff]", texts):
        return "ja"
    # chinese
    if re.search("[\u4e00-\u9FFF]", texts):
        return "zh"
    return None

  @staticmethod
  def batch(lst, n):
    """Generate batches"""
    lst = list(lst)
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

  def apply_regex_ner(self, src_lang, docs, context_window = 20, weight = 1.0, text_key=None, ner_key=None, signal='regex'):
    """
    apply regexes from the rulebase. if there is a context, check if the context is met in the context_window.
    """
    #print ("apply_regex_ner")
    global regex_rulebase
    if ner_key is None:
      ner_key = f'{src_lang}_signal_ner'
    if text_key is None:
      text_key = f'{src_lang}_text'
    for doc in docs.values():
      ner = doc[ner_key] = doc.get(ner_key, {})
      sentence = doc[text_key]
      all_ner = detect_ner_with_regex_and_context(sentence, src_lang, context_window=context_window, tag_type=None)
      for mention_tag in all_ner:
        ent, start, end, tag = mention_tag
        key = (ent, start, end)
        aHash = ner.get(key, {})
        aHash[(tag, signal)] = aHash.get((tag, signal), 0) + weight * (1.0 + len(ent)/100) # extra weight?
        ner[key] = aHash
      doc[ner_key] = ner
    #print (docs)
    return docs


  def hf_ner(self, hf_pipeline, src_lang, docs, chunks, stopwords=None, weight=1.5, text_key=None, ner_key=None, offset_key=None, signal='hf'):
    """
    run the text through a Huggingface ner pipeline.
    any tags found by this method will be weighted by the weight param
    TODO: use the predicted value of the logits to further weight prediction
    NOTE: we don't use results_arr = hf_pipeline([chunk[text_key] for chunk in chunks], grouped_entities=True)
    because grouped_entities does not properly group all entities as we do it below.
    """
    #print ("hf_ner")
    if stopwords is None:
      stopwords = set(stopwords.get(src_lang, []))
    if offset_key is None:
      offset_key = f'{src_lang}_offset'
    if ner_key is None:
      ner_key = f'{src_lang}_signal_ner'
    if text_key is None:
      text_key = f'{src_lang}_text'
    #print (chunks)
    results_arr = hf_pipeline([chunk[text_key] for chunk in chunks], batch_size=len(chunks))
    results_arr2 = []
    offset = 0
    for chunk, results in zip(chunks, results_arr):
      text = chunk[text_key]
      _id = chunk['id']
      ner = docs[_id][ner_key] = docs[_id].get(ner_key,{})
      offset = chunk[offset_key]
      len_text= len(text)
      results = [ner_result for ner_result in results if ner_result['word'] not in ("[UNK]", "<unk>")]
      if not results:
        results_arr2.append([])
        continue
      results2 = []
      if results[0]['start'] is not None: #TODO, test for the case where all the 'start' are '0'.
        results.sort(key=lambda a: a['start'])
      else:
        results.sort(key=lambda a: a['index'])
        i = 0
        for ner_result in results:
          ner_result['word'] = word = ner_result['word'].rstrip('@@')
          ner_result['start'] = text.index(word, i)
          i = ner_result['start'] + 1
          ner_result['end'] = ner_result['start'] + len(word)

      for ner_result in results:
        start = ner_result['start']
        if start >= len_text: continue
        if not self.cjk_detect(text[ner_result['start']:ner_result['end']]):
              if text[start] not in self.strip_chars:
                for j in range(1, start):
                  if start - j == -1 or text[start-j] in self.strip_chars:
                    start = max(start -j, 0)
                    break
              end = ner_result['end']
              if end < len_text and text[end] != ' ':
                end += len(text[end:].split(' ', 1)[0])
        else:
              start = ner_result['start']
              end = ner_result['end']
        while text[start] in self.strip_chars and start < len_text:
          start += 1
          if start >= end: break
        if start < len_text and start < end:
            end = start + len(text[start:end].strip(self.strip_chars))
            ner_result['word'] = text[start:end]
            ner_result['start'] = start+offset
            ner_result['end'] = end+offset
        if results2 and results2[-1]['end'] > ner_result['start']:
          continue
        if start < len_text and start < end:
            results2.append(ner_result)
      results_arr2.append(results2)
    results_arr = results_arr2
    for chunk, results in zip(chunks, results_arr):
        _id = chunk['id']
        ner = docs[_id][ner_key]
        text = docs[_id][text_key]
        len_text = len(text)
        results = [ner_result for ner_result in results if ner_result['word'] not in ("[UNK]", "<unk>")]
        if not results: continue
        prev_word = [0,0]
        prev_label = None
        prev_word2 = ""
        for ner_result in results:
          start = ner_result['start']
          if start is None:
            prev_word2 = ""
            continue
          end = ner_result['end']
          if text[start:end] != ner_result['word']:
            logger.info ('offset mismatch', text[start:end], ner_result['word'])
          if "-" in ner_result['entity']:
            _, label = ner_result['entity'].split('-')
          else:
            label = ner_result['entity']
          label = label.upper()
          if label in ('ADDRESS', 'STREET_ADDRESS'): label = 'ADDRESS'
          elif label in ('PUBLIC_FIGURE',): label = 'PUBLIC_FIGURE'
          elif label in ('NAME', 'PER', 'PERSON'): label = 'PERSON'
          elif label in ('LOCATION', 'LOC', 'GPE'): label = 'LOC'
          elif label in ('ORGANIZATION', 'ORG'): label = 'ORG'
          elif label in ('AGE',): label = 'AGE'
          elif label in ('NORP',): label = 'NORP'
          elif label in ('BIO', 'SYMPTOM_AND_DISEASE', 'DISEASE' ): label = 'DISEASE'
          elif label in ('PATIENT_ID', 'GOVT_ID' ): label = 'ID'
          elif label in ('USER_ID', 'ID'): label = 'ID'
          elif label in ('MISC', ) and '@' in ner_result['word']: label = 'ID'
          else: label = 'MISC'
          label = (label, signal)
          if prev_label is not None:
              if not ner_result['entity'].startswith('B-') and label == prev_label and (prev_word[1] >= start - 5):
                prev_word[1] =  max(prev_word[1], end)
                prev_word2 = prev_word2 + " " + ner_result['word']
              else:
                if ner_result['entity'].startswith('B-'):
                  if prev_word[1] > start:
                    prev_word[1] = start
                if prev_word[0] != prev_word[1]:
                  ner_word = text[prev_word[0]:prev_word[1]]
                  #if ner_word != prev_word2:
                  #  print (ner_word, '**', prev_word2)
                  #ner_word.strip(self.strip_chars)
                  mention = (ner_word, prev_word[0], prev_word[1])
                  if ner_word and ner_word.lower() not in stopwords:
                    aHash = ner.get(mention, {})
                    aHash[prev_label] = aHash.get(prev_label, 0) + weight * (1.0 + len(ner_word)/100)
                    ner[mention] = aHash
                  prev_word = [start, end]
                  prev_word2 = ner_result['word']
          elif prev_label is None:
            prev_word = [start, end]
            prev_word2 = ner_result['word']
          prev_label = label

        if prev_label is not None and prev_word[0] != prev_word[1]:
            ner_word = text[prev_word[0]:prev_word[1]]
            #if ner_word != prev_word2:
            #  print (ner_word, '**', prev_word2)
            mention = (ner_word, prev_word[0], prev_word[1])
            if ner_word and ner_word.lower() not in stopwords:
                aHash = ner.get(mention, {})
                aHash[prev_label] = aHash.get(prev_label, 0) + weight * (1.0 + len(ner_word)/100)
                ner[mention] = aHash

  def add_chunks_span(self, chunks, new_mention, old_mention, label, coref, chunk2ner, mention2ref, ref2mention):
    """ add a span to the chunks sequence and update the various ref and NER hashes """
    if old_mention in chunk2ner:
      del chunk2ner[old_mention]
    if label:
      chunk2ner[new_mention] = label
    if old_mention in mention2ref:
      old_ref = mention2ref[old_mention]
      ref2mention[old_ref].remove(old_mention)
      if not ref2mention[old_ref]:
        del ref2mention[old_ref]
      del mention2ref[old_mention]
    if new_mention in mention2ref and coref != mention2ref[new_mention]:
      old_ref = mention2ref[new_mention]
      ref2mention[old_ref].remove(new_mention)
      if not ref2mention[old_ref]:
        del ref2mention[old_ref]
      del mention2ref[new_mention]
    if coref:
      mention2ref[new_mention] = coref
      lst = ref2mention.get(coref, [])
      if new_mention not in lst:
        ref2mention[coref] = lst + [new_mention]
    chunks.append(new_mention)

  def del_ner_coref(self, old_mention, chunk2ner, mention2ref, ref2mention):
    """ remove an old_mention from the various NER and ref hashes """
    if old_mention in chunk2ner:
      del chunk2ner[old_mention]
    if old_mention in mention2ref:
      old_ref = mention2ref[old_mention]
      ref2mention[old_ref].remove(old_mention)
      if not ref2mention[old_ref]:
        del ref2mention[old_ref]
      del mention2ref[old_mention]

  def spacy_ner_coref(self, docs, nlp, stopwords, spacy_weight, src_lang, extra_weight=1.0, signal='neuralcoref', text_key=None, ner_key=None, connector="_", pronouns=("who", "whom", "whose", "our", "ours", "you", "your", "my", "i", "me", "mine", "he", "she", "his", "her", "him", "hers", "it", "its", "they", "their", "theirs", "them", "we")):
    """
    Use the spacy English model to create chunks for English text
    and gather NER and coreference information
    """
    #print ("spacy_ner_coref")
    if not nlp:
      return
    if stopwords is None:
      stopwords = set(stopwords.get(src_lang, []))
    offset_key=f'{src_lang}_offset'
    if ner_key is None:
      ner_key = f'{src_lang}_signal_ner'
    if text_key is None:
      text_key = f'{src_lang}_text'
    mention2ref_key = f'{src_lang}_mention2ref'
    ref2mention_key = f'{src_lang}_ref2mention'
    mention2pronoun_key = f'{src_lang}_mention2pronoun'
    for dat in docs.values():
      chunk2ner = {}
      ref2mention = {} # dat[ref2mention_key] =  dat.get(ref2mention_key,{})
      mention2ref = {} # dat[mention2ref_key] =  dat.get(mention2ref_key,{})
      mention2pronoun = dat[mention2pronoun_key] =  dat.get(mention2pronoun_key,{})
      ner =  dat[ner_key] =  dat.get(ner_key,{})
      text = dat[text_key]
      doc = nlp(text)
      entities = list(doc.ents)
      # spacy is not as high accuracy as transformers, but we use the spacey neuralcoref model so we can get pronoun coreference groups
      # to be able to do proper gender swapping. We can also expand NER tags based on coreferences.

      #store away NOUNs for potential label and coref reference
      #rule for promoting a noun span into one considered for further processing:
      # - length of the number of words > 2 or length of span > 2 and the span is all uppercase (for abbreviations)
      # coref candidates:
      # - create an abbreviation from noun phrases as a candidate coref.
      # - use either the last two words of a span as a candidate coref, or
      # - use the abbreviation as a candidate coref
      for entity in list(doc.noun_chunks) + list(doc.ents):
        chunk2ner[(entity.text, entity.start, entity.end)]= "NOUN"
        mention_lower = entity.text.lower()
        textArr = mention_lower.split()
        if len(textArr) > 2:
          short_span = " ".join(textArr[-2:])
          ref2mention[short_span] = ref2mention.get(short_span, []) + [(entity.text, entity.start, entity.end)]
          non_stopwords = [a for a in textArr if a not in self.stopwords_en]
          if len(non_stopwords) > 2:
            abrev = "".join([a[0] for a in non_stopwords])
            ref2mention[abrev] = ref2mention.get(abrev, []) + [(entity.text, entity.start, entity.end)]
        elif (len(entity.text) >=2 and entity.text == entity.text.upper()):
          ref2mention[entity.text.lower()] = ref2mention.get(entity.text.lower(), []) + [(entity.text, entity.start, entity.end)]

      #store away coref NOUNs for potential label and coref reference
      #same rule as above for promoting a noun span into one considered for further processing.
      for cl in list(doc._.coref_clusters):
        mentions = [(entity.text, entity.start, entity.end) for entity in cl.mentions]
        mentions.sort(key=lambda e: len(e[0]), reverse=True)
        textArr = mentions[0][0].lower().split()
        for key in mentions:
          chunk2ner[key]= "NOUN"
        for mention in mentions:
          mention_lower = mention[0].lower()
          textArr = mention_lower.split()
          if mention_lower not in self.stopwords_en:
            if len(textArr) > 1:
              short_span = " ".join(textArr[-2:])
            else:
              short_span = textArr[0]
            ref2mention[short_span] = ref2mention.get(short_span, []) + mentions
            non_stopwords = [a for a in textArr if a not in self.stopwords_en]
            if len(non_stopwords) > 2:
              abrev = "".join([a[0] for a in non_stopwords])
              ref2mention[abrev] = ref2mention.get(abrev, []) + mentions

      #cleanup the mention2ref, favoring large clusters with coref labels that are longer
      seen = {}
      corefs = [(a, list(set(b))) for a, b in ref2mention.items()]
      corefs.sort(key=lambda a: a[0].count(" ")+len(a[1]), reverse=True)
      for coref, spans in corefs:
        new_spans = []
        spans = list(set(spans))
        spans.sort(key=lambda a: a[1]+(1.0/(1.0+a[2]-a[1])))
        spans2 = []
        for span in spans:
          if spans2 and spans2[-1][1] >= span[1]:
            continue
          spans2.append(span)
        for span in spans2:
          if span in seen: continue
          seen[span] = 1
          new_spans.append(span)
        del ref2mention[coref]
        if new_spans:
          new_coref = [s[0] for s in new_spans]
          new_coref.sort(key=lambda a: len(a), reverse=True)
          ref2mention[new_coref[0].lower()] = list(set(list(ref2mention.get(new_coref[0].lower(), [])) + new_spans))

      mention2ref.clear()
      for a, b1 in ref2mention.items():
        for b in b1:
          mention2ref[b] = a

      # expand coref information by using the most common coref label in a cluster
      if True:
        for cl in list(doc._.coref_clusters):
          mentions = [(entity.text, entity.start, entity.end) for entity in cl.mentions]
          all_mentions = list(set(itertools.chain(*[ref2mention[mention2ref[mention]] for mention in mentions if mention in mention2ref])))
          corefs = [mention2ref[mention] for mention in mentions if mention in mention2ref]
          if corefs:
            coref = Counter(corefs).most_common()[0][0]
          else:
            coref = cl.main.text.lower()
          for mention in all_mentions:
            if mention not in chunk2ner:
              chunk2ner[mention] = 'NOUN'
            old_ref = mention2ref.get(mention)
            if old_ref and mention in ref2mention[old_ref]:
              ref2mention[old_ref].remove(mention)
              if not ref2mention[old_ref]:
                del ref2mention[old_ref]
            mention2ref[mention] = coref
            if mention not in ref2mention.get(coref,[]):
              ref2mention[coref] = ref2mention.get(coref,[])
              ref2mention[coref].append(mention)

      #expand ner labels based on coref matches
      for entity in list(doc.ents):
        mention = (entity.text, entity.start, entity.end)
        chunk2ner[mention]= entity.label_
        if mention in mention2ref:
          coref = mention2ref[mention]
          for mention in ref2mention[coref]:
            chunk2ner[mention] = entity.label_

      # overwrite all ner labels in the coref cluster to PERSON if there is a person pronoun
      if True:
        for cl in list(doc._.coref_clusters):
          cluster_text_list = set([m.text.lower() if m.text != 'US' else m.text for m in cl.mentions])
          if "us" in cluster_text_list or "you" in cluster_text_list or "your"  in cluster_text_list  or "yours"  in cluster_text_list  or  "we" in cluster_text_list  or 'i' in cluster_text_list  or 'my' in cluster_text_list  or 'mine' in cluster_text_list or 'me' in cluster_text_list or 'he' in cluster_text_list or "she" in cluster_text_list or "his" in cluster_text_list or "her" in cluster_text_list or "him" in cluster_text_list or "hers" in cluster_text_list:
            label = "PERSON"
            for m in cl.mentions:
              chunk2ner[(m.text, m.start, m.end)] = label

      # propogate the ner label to everything in the same coref group
      for coref, seq in ref2mention.items():
        labels = [chunk2ner[mention]  for mention in seq if mention in chunk2ner and chunk2ner[mention] != 'NOUN']
        if labels:
          label = Counter(labels).most_common()[0][0]
          for mention in seq:
            if mention in chunk2ner and  not (label == 'PERSON' or chunk2ner[mention] == 'PUBLIC_FIGURE'): chunk2ner[mention] = label

      #sort the chunks into order
      chunks = list(chunk2ner.items())
      chunks.sort(key=lambda a: a[0][1]+(1.0/(1.0+a[0][2]-a[0][1])))
      chunks2 = []

      #clear duplicates and subsumed mentions
      for mention, label in chunks:
        if not chunks2 or (chunks2[-1][2] <= mention[1]):
          if not chunks2 or chunks2[-1][2] < mention[1]:
            self.add_chunks_span(chunks2, (doc[0 if not chunks2 else chunks2[-1][2]: mention[1]].text, 0 if not chunks2 else chunks2[-1][2], mention[1]), \
                                 None, None, None, chunk2ner, mention2ref, ref2mention)
          self.add_chunks_span(chunks2, mention, None, label, mention2ref.get(mention), chunk2ner, mention2ref, ref2mention)
        elif chunks2[-1][2] > mention[1] and chunks2[-1][1] <= mention[1]:
          if chunk2ner.get(chunks2[-1]) not in (None, '', 'NOUN'):
            self.del_ner_coref(mention, chunk2ner, mention2ref, ref2mention)
            continue
          elif label in  (None, '', 'NOUN'):
            self.del_ner_coref(mention, chunk2ner, mention2ref, ref2mention)
            continue
          old_mention = chunks2.pop()
          oldSpan = old_mention[0]
          oldLabel = chunk2ner.get(old_mention)
          oldAnaphore = mention2ref.get(old_mention)
          sArr = oldSpan.split(mention[0], 1)
          self.del_ner_coref(old_mention, chunk2ner, mention2ref, ref2mention)
          s0 = sArr[0].strip()
          if s0:
            self.add_chunks_span(chunks2, (s0, old_mention[1], mention[1]), None, \
                                 oldLabel if s0 in pronouns or (len(s0) > 1 and s0 not in self.stopwords_en) else None, oldAnaphore  if s0 in pronouns or (len(s0) > 1 and s0 not in self.stopwords_en) else None, \
                                 chunk2ner, mention2ref, ref2mention)
          self.add_chunks_span(chunks2,  mention, None, label, oldAnaphore if not mention2ref.get(mention) else mention2ref.get(mention), chunk2ner, mention2ref, ref2mention)
          if len(sArr) > 1:
            s1 = sArr[1].strip()
            if s1:
              self.add_chunks_span(chunks2, (s1, mention[2], old_mention[2]), None,  \
                                   oldLabel if s1 in pronouns or (len(s1) > 1 and s1 not in self.stopwords_en) else None, oldAnaphore  if s1 in pronouns or (len(s1) > 1 and s1 not in self.stopwords_en) else None, \
                                   chunk2ner, mention2ref, ref2mention)
      len_doc = len(doc)
      if chunks2 and chunks2[-1][2] < len_doc:
        self.add_chunks_span(chunks2, (doc[chunks2[-1][2]:].text, chunks2[-1][2], len_doc), None, None, None, chunk2ner, mention2ref, ref2mention)

      #reset the indexes for chunks to be per character index.
      i = 0
      for spanIdx, mention in enumerate(chunks2):
          label = chunk2ner.get(mention)
          if label in ('CARDINAL', 'WORK_OF_ART', 'NOUN', None): continue
          if label in ('GPE', 'FAC'): label = 'LOC'

          ner_word = mention[0]
          if ner_word and ner_word.lower() not in stopwords:
              if not self.cjk_detect(ner_word):
                if ner_word not in text: continue
                i += text[i:].index(ner_word)
                ner_word = text[i:].split(" ", 1)[0]
              ner_word = ner_word.rstrip(self.strip_chars)
              if ner_word.lower() not in stopwords:
                mention2 = (ner_word, i, i+len(ner_word))
                aHash = ner.get(mention2, {})
                aHash[(label, signal)] = aHash.get((label, signal), 0) + spacy_weight * (1.0 + len(ner_word)/100) * extra_weight
                ner[mention2] = aHash
                if label in ('PERSON', 'PUBLIC_FIGURE'):
                  coref = set([a[0] for a in ref2mention.get(mention2ref.get(mention), [])])
                  if "he" in coref or "He" in coref or "him" in coref or "Him" in coref or "his" in coref or "His" in coref or "Mr." in coref or "Mr" in coref or "mr" in coref or "mr." in coref:
                    mention2pronoun[mention2] = "he"
                  elif "she" in coref or "She" in coref or "her" in coref or "Her" in coref or "hers" in coref or "Hers" in coref or "Miss" in coref or "miss" in coref or  "Mrs." in coref or "Mrs" in coref or "mrs" in coref or "mrs." in coref or "Ms." in coref or "Ms" in coref or "ms" in coref or "ms." in coref:
                    mention2pronoun[mention2] =  "she"
                  elif "they" in coref or "They" in coref or "Them" in coref or "them" in coref or "Their" in coref or "their" in coref or "We" in coref or "we" in coref or  "Us" in coref or "us" in coref:
                    mention2pronoun[mention2] =  "they"
                  else:
                    mention2pronoun[mention2] =  random.choice(["she", "he", "they"])
      #print (ner)
    return docs #text, chunks, chunk2ner, mention2ref, ref2mention

  def spacy_ner(self, docs, nlp, stopwords, spacy_weight, src_lang, extra_weight=1.0, text_key=None, ner_key=None, signal='spacy'):
      """
      Use the spacy models to create mentions w/ NER
      """
      #print ("spacy_ner")
      if neuralcoref is not None:
        return self.spacy_ner_coref(docs, nlp, stopwords, spacy_weight, src_lang, extra_weight=extra_weight, text_key=text_key, ner_key=ner_key, signal=signal)
      else:
        if not nlp:
          return
        if stopwords is None:
          stopwords = set(stopwords.get(src_lang, []))
        offset_key=f'{src_lang}_offset'
        if ner_key is None:
          ner_key = f'{src_lang}_signal_ner'
        if text_key is None:
          text_key = f'{src_lang}_text'
        for dat in docs.values():
          ner =  dat[ner_key] =  dat.get(ner_key,{})
          text = dat[text_key]
          doc = nlp(text)
          entities = list(doc.ents)
          ents = [(entity.text, entity.label_ if (entity.label_ in ('PERSON', 'GPE', 'ORG', 'NORP', 'FAC', 'LOC') and 'http:' not in entity.text) else 'MISC') for entity in entities]
          i = 0
          for ner_word, label in ents:
            if label in ('GPE', 'FAC'): label = 'LOC'
            ner_word = ner_word.strip(self.strip_chars)
            if ner_word and ner_word.lower() not in stopwords:
              if not self.cjk_detect(ner_word):
                if ner_word not in text: continue
                i += text[i:].index(ner_word)
                ner_word = text[i:].split(" ", 1)[0]
              ner_word = ner_word.strip(self.strip_chars)
              if ner_word.lower() not in stopwords:
                mention = (ner_word, i, i+len(ner_word))
                aHash = ner.get(mention, {})

                aHash[(label, signal)] = aHash.get((label, signal), 0) + spacy_weight * (1.0 + len(ner_word)/100) * extra_weight
                ner[mention] = aHash

  def trim_to_prefer_person(self, docs, chunks, prob=100):
      #print ("trim_to_prefer_person")
      # downsample to mostly docs with mentions of people, id/email
      # if there were no ner set, then don't downsample the doc
      len_docs = len(docs)
      do_ids = []
      for _id, doc in docs.items():
        if not any(key for key in doc if key.endswith('_ner')):
          do_ids.append(_id)
          continue
        found_ner = False
        for key in list(doc.keys()):
          if doc.get('has_person'):
            do_ids.append(_id)
            break
          if "_ner" in key:
            if not found_ner:
              found_ner = doc[key] != {}
            ner =  doc[key]
            #print (ner, key, doc)
            for aHash in ner.values():
              if type(aHash) is dict and 'PUBLIC_FIGURE' in aHash or 'PERSON' in aHash or 'ID' in aHash:
                doc['has_person'] = True
                do_ids.append(_id)
                break
        if doc.get('has_person'):
            do_ids.append(_id)
        elif not doc.get('has_person') and random.randint(0, prob) == 0:
            do_ids.append(_id)
      do_ids = set(do_ids)
      chunks2 = [chunk for chunk in chunks if chunk['id'] in do_ids]
      docs2 = dict([(doc['id'], doc) for doc in docs.values() if doc['id'] in do_ids])
      if len(docs2) == 0 or len_docs == len(docs2):
        return docs, chunks
      #print ('trim_to_prefer_person', (len_docs-len(docs2))/len_docs)
      return docs2, chunks2



  def collapse_ner(self, docs, ner_key, collapse_ner_key, text_key, stopwords, do_cleanup_only=False):
    #print ("collapse_ner")
    for doc in docs.values():
      text = doc.get(text_key, "")

      if True:
        #do some cleanups. we don't want any ner that are just short numbers (but what about govt id?), stopwords or single characters.
          ner =  doc[ner_key]
          for key in list(doc[ner_key].keys()):
            ner_word = key[0]
            try:
              if len(ner_word) < 4 and float(ner_word):
                #print ("deleting ", ner_word)
                del doc[ner_key][key]
                continue
            except:
              pass
            if ner_word.lower() in stopwords or (not self.cjk_detect(ner_word) and len(ner_word) <= 1):
              #print ("deleting ", ner_word)
              del doc[ner_key][key]

      if do_cleanup_only:
        continue
      #TODO - generalize long ner to rest of the sentences if not already tagged

      chunk2ner = doc.get(ner_key, {})
      chunks = list(chunk2ner.items())
      chunks.sort(key=lambda a: a[0][1]+(1.0/(1.0+a[0][2]-a[0][1])))
      chunks2 = []

      for mention, labelsHash in chunks:
        mention = list(mention)
        if not chunks2:
          chunks2.append([mention[0], mention[1], mention[2], labelsHash])
        # completely or mostly subsumed
        elif chunks2[-1][2] >= mention[2] or chunks2[-1][2] - mention[1] > 3:
          prev_ent, prev_start, prev_end, prev_labelsHash = chunks2[-1]
          for tag in labelsHash:
            prev_labelsHash[tag]  = prev_labelsHash.get(tag, 0) + labelsHash.get(tag, 0)
          chunks2[-1][2] = mention[2]
          #print (chunks2[-1], text)
          chunks2[-1][0] = text[chunks2[-1][1]:chunks2[-1][2]]
        elif chunks2[-1][2] < mention[1]:
          chunks2.append([mention[0], mention[1], mention[2], labelsHash])
        # partially subsumed
        else:
          if mention[2] - mention[1] > chunks2[-1][2] - chunks2[-1][1]:
              chunks2[-1][2] = mention[1] -1
              chunks2[-1][0] = text[chunks2[-1][1]:chunks2[-1][2]]
          else:
              mention[1] = chunks2[-1][2] + 1
              mention[0] = text[mention[1]:mention[2]]
          chunks2.append([mention[0], mention[1], mention[2], labelsHash])

      ner = {}
      for ent, start, end, labelsHash in chunks2:
        ent = ent.strip(self.strip_chars)
        if ent:
          mention = (ent, start, start + len(ent))
          labelsHash2 = {}
          for key, val in labelsHash.items():
            if type(key) is tuple:
              key = key[0]
            labelsHash2[key] = labelsHash2.get(key, 0) + val
          #do hypo collapse so that loc->address, person->public_figure when there are overlaps, date/id->date
          if 'PERSON' in labelsHash2 and 'PUBLIC_FIGURE' in labelsHash2:
            labelsHash2['PUBLIC_FIGURE'] = labelsHash2['PUBLIC_FIGURE'] + labelsHash2['PERSON']
            del labelsHash2['PERSON']
          if 'LOC' in labelsHash2 and 'ADDRESS' in labelsHash2:
            labelsHash2['ADDRESS'] = labelsHash2['ADDRESS'] + labelsHash2['LOC']
            del labelsHash2['LOC']
          if 'DATE' in labelsHash2 and 'AGE' in labelsHash2:
            labelsHash2['AGE'] = labelsHash2['AGE'] + labelsHash2['DATE']
            del labelsHash2['DATE']
          if 'DATE' in labelsHash2 and 'ID' in labelsHash2:
            del labelsHash2['ID'] # we prefe dates to ids?
          if 'CARDINAL' in labelsHash2 and 'ID' in labelsHash2:
            labelsHash2['ID'] = labelsHash2['ID'] + labelsHash2['CARDINAL']
            del labelsHash2['CARDINAL']

          ner[mention] = labelsHash2
      doc[collapse_ner_key] = ner


    return docs

  def create_augment_anon_context(self, docs, chunks, src_lang, faker_target_lang, faker_en, aug_scope={'ID', 'PERSON'}, target_lang=None, \
                                 items_key=None, context_key=None, ner_key=None):
        #print ("create_augment_anon_context")
        if target_lang is None: target_lang = src_lang
        if ner_key is None: ner_key = f'{src_lang}_signal_ner'
        if context_key is None: context_key = f'{src_lang}_aug_context'
        if items_key is None: items_key = f'{src_lang}_items'
        if faker_target_lang is not None and faker_en is not None:
          for doc in docs.values():
          # do augmentation in src_lang, and then translate to target_lang.
            context = doc[context_key] = doc.get(context_key, {})
            ner = doc.get(ner_key, {})
            src_items_sorted = copy.copy(doc[items_key])
            src_items_sorted.sort(key=lambda a: len(a[0]), reverse=True)

            for key in src_items_sorted:
              idx = key[-1]
              mention = tuple(key[:-1])
              if mention not in ner: continue
              ent = key[0]
              tag = max(Counter(ner[mention]))
              if ent in context: continue
              #TODO - do proper gender based aug and gender swap
              if 'PERSON' in aug_scope and tag == 'PERSON' and ent not in context:
                context[ent] = context.get(ent, faker_en.first_name() + " " + random.choice(bantu_surnames) if " " in ent and target_lang in ("yo", "sw","sn", "st", "ig", "ny", "xh",) else \
                                      random.choice(bantu_surnames) if target_lang in ("yo", "sw","sn", "st", "ig", "ny", "xh",) else \
                                      random.choice(vietnamese_surnames) + " " + random.choice(vietnamese_firstnames) if " " in ent and target_lang =="vi" else \
                                      random.choice(vietnamese_surnames) if  target_lang == "vi" else \
                                      faker_en.first_name() + " " + random.choice(bengali_surnames) if " " in ent and target_lang =="bn" else \
                                      random.choice(bengali_surnames) if target_lang == "bn" else \
                                      random.choice(urdu_firstnames)  + " " + random.choice(urdu_surnames) if " " in ent and target_lang =="ur" else \
                                      random.choice(urdu_surnames) if target_lang == "ur" else \
                                      faker_target_lang.name() if " " in ent else \
                                      faker_target_lang.first_name() )

              if 'LOC' in aug_scope and tag == 'LOC' and ent not in context:
                context[ent] = context.get(ent, faker_en.country() if  target_lang in ("yo", "sw","sn", "st", "ig", "ny", "xh", "bn", "ur", "vi", "eu") else \
                                      faker_target_lang.state() if target_lang != 'zh' else \
                                      faker_target_lang.province() if  target_lang == 'zh' else
                                      ent)
              if 'ORG' in aug_scope and tag == 'ORG' and ent not in context:
                try:
                  context[ent] = context.get(ent, faker_target_lang.company())
                except:
                  context[ent] = context.get(ent, faker_en.company())

              if 'ID' in aug_scope and tag == 'ID' and ent not in context:
                if '@' in ent:
                  context[ent] = context.get(ent, faker_target_lang.email())
                else:
                  context[ent] = context.get(ent, str(random.randrange(10000000,999999999)) if target_lang in ("yo", "sw","sn", "st", "ig", "ny", "xh", "bn", "ur", "vi", "eu")  else \
                                      faker_target_lang.ssn())

              if 'ADDRESS' in aug_scope and tag == 'ADDRESS' and ent not in context:
                context[ent] = context.get(ent, faker_en.address() if target_lang not in ("yo", "sw","sn", "st", "ig", "ny", "xh", "bn", "ur", "vi", "eu") else \
                                      faker_target_lang.address() )

              if tag in  ('PERSON', 'ORG') and tag in aug_scope :
                src_first, src_last = None, None
                target_first, target_last = None, None
                if src_lang in ("ja", "ko", "zh") and len(ent) > 1:
                  src_first, src_last = ent[0], ent[-1]
                elif " " in ent:
                  src_first, src_last =  ent.split()[0], ent.split()[-1]
                if target_lang in ("ja", "ko", "zh"):
                  target_first, target_last = context[ent][0], context[ent][-1]
                elif " " in context[ent]:
                  target_first, target_last = context[ent].split()[0], context[ent].split()[-1]
                if src_first and (src_lang  in ("ja", "ko", "zh") or len(src_first) > 3) and src_first not in context:
                  context[src_first] = target_first
                if src_last and (src_lang  in ("ja", "ko", "zh") or len(src_last) > 3) and src_last not in context:
                  context[src_last] = target_last
            #print (context_key, context)

  def replace_items_in_chunks(self, docs, chunks, src_lang, target_lang=None, lbracket="[", rbracket="]", replace_with_bracket=True, do_augment=False, \
                                                   context_key=None, ner_key=None, items_key=None, text_key=None, replace_text_key=None, offset_key=None):
        #print ("replace_items_in_chunks")
        if target_lang is None: target_lang = src_lang
        if ner_key is None: ner_key = f'{src_lang}_ner'
        if context_key is None: context_key = f'{src_lang}_aug_context'
        if items_key is None: items_key = f'{src_lang}_items'
        if text_key is None: text_key = f'{src_lang}_text'
        if replace_text_key is None: replace_text_key = f'{src_lang}_tmp_text'
        if offset_key is None: offset_key = f'{src_lang}_offset'
        for doc in docs.values():
          items = list(doc.get(ner_key, {}).keys())
          items.sort(key=lambda a: a[1])
          items = [list(key)+[idx] for idx, key in enumerate(items)]
          doc[items_key] = items

        for chunk in chunks:
          if replace_with_bracket:
            text = chunk[text_key].replace("(", "{").replace(")", "}")
          else:
            text = chunk[text_key]
          _id = chunk['id']
          offset = chunk[offset_key]
          doc = docs[_id]
          offset_end = offset + len(text)
          i = 0
          for idx, key in enumerate(doc[items_key]):
            if key[1] < offset:
              continue
            if key[2] > offset_end:
              break
            if len(key[0]) < 4 and not self.cjk_detect(key[0]):
              if " "+key[0]+" " in text[i:]:
                j = text.index(" "+key[0]+" ", i)
                text = text[:j]+(text[j:].replace(" "+key[0]+" ", f"  **{idx}**  ", 1))
                i = j
            else:
              if key[0] in text[i:]:
                j = text.index(key[0], i)
                text = text[:j]+(text[j:].replace(key[0], f"  **{idx}**  ", 1))
                i = j
          chunk[replace_text_key] = text

        for doc in docs.values():
          doc[items_key].sort(key=lambda a: len(a[0]), reverse=True)

        for chunk in chunks:
          text = chunk[replace_text_key]
          _id = chunk['id']
          doc = docs[_id]
          for key in doc[items_key]:
            idx = key[-1]
            if len(key[0]) < 5 and not self.cjk_detect(key[0]):
              text = text.replace(" "+key[0]+" ", f"  **{idx}**  ")
            else:
              text = text.replace(key[0], f" **{idx}** ")
          chunk[replace_text_key] = text


        for chunk in chunks:
          if do_augment:
            context = doc[context_key] = doc.get(context_key, {})
          else:
            context = {}
          text = chunk[replace_text_key]
          _id = chunk['id']
          doc = docs[_id]
          for key in doc[items_key]:
            idx  = key[-1]
            if do_augment:
              ent = context.get(key[0], key[0])
            else:
              ent = key[0]
            if replace_with_bracket:
              text = text.replace(f" **{idx}** ", f" {idx} {lbracket} {ent} {rbracket}")
            else:
              text = text.replace(f" **{idx}** ", ent)
          chunk[replace_text_key] = text.replace("  ", " ")

        for doc in docs.values():
          doc[items_key].sort(key=lambda a: a[-1])

        return docs, chunks


  def anonymize(self, docs, chunks, src_lang, faker_src_lang, faker_en, anon_scope = {'ID', 'PERSON'}, target_lang=None):

    #anonymization is very similar to augmentation, except we operate in the src_lang space, and don't require translation.
    #we will replace the words directly from {src_lang}_text to {src_lang}_text_anon.
    #we assume all ner process has been completed at this point.
    #anonymization will create a new {src_lang}_text_anon.
    #TODO: create a {src_lang}_signal_ner_anon field.
    #TODO: another  way to do anonymimzation is to pass the anonymized text through backtrans. TBD?
    #print ("anonymize")
    if target_lang is None: target_lang = src_lang

    self.create_augment_anon_context(docs, chunks, src_lang, faker_src_lang, faker_en, aug_scope=anon_scope, target_lang=src_lang, \
                                          items_key=f'{src_lang}_items', context_key=f'{src_lang}_anon_context', ner_key=f'{src_lang}_ner')

    docs, chunks = self.replace_items_in_chunks(docs, chunks, src_lang, replace_with_bracket=False, do_augment=True, \
                                                                         context_key=f'{src_lang}_non_context', \
                                                                         ner_key=f'{src_lang}_signal_ner', items_key=f'{src_lang}_items', \
                                                                         text_key=f'{src_lang}_text', replace_text_key=f'{src_lang}_text_anon', \
                                                                         offset_key=f'{src_lang}_offset')
    for doc in docs.values():
      doc[f'{src_lang}_text_anon']  = " ".join([chunk[f'{src_lang}_text_anon'] for chunk in doc['chunks']])

    return docs, chunks

  #TODO - refactor this method into parts
  def process_ner_chunks_with_trans(self,
                          src_lang,
                          docs,
                          chunks,
                          target_lang=None,
                          do_spacy = True,
                          do_hf_ner = True,
                          do_ontology = True,
                          do_backtrans=False,
                          do_augment=False,
                          do_anonymization=False,
                          do_regex = True,
                          do_cleanup=True,
                          do_marian_mt=True,
                          batch_size = 5,
                          num_words_per_chunk=70,
                          ontology_weight=0.85,
                          spacy_weight=1.00,
                          hf_ner_weight=1.25,
                          regex_weight=1.5,
                          backtrans_weight=0.9,
                          do_docs_trim_for_person=False,
                          do_public_figure_expansion=True,
                          do_kenlm = True,
                          do_qg_rel=False,
                          aug_scope={'ADDRESS', 'ORG', 'PERSON', 'LOC', 'ID'}, #TODO, public figure, age, norp and disease
                          anon_scope={'PERSON', 'ID'},):
    #print ("process_ner_chunks_with_trans:", self.device, self.device_id)
    if do_augment and do_backtrans:
      assert False, "Only augment or backtrans can be performed at a time, not both"
    if do_augment and do_anonymization:
      assert False, "Only augment or anonymization can be performed at a time, not both"
    if target_lang is None:
        target_lang = src_lang
    if (do_augment or do_anonymization) and target_lang != src_lang:
      if target_lang not in ("eu", "ca") and target_lang not in faker_map:
        faker_target_lang = random.choice(self.faker_en_list)
      else:
        faker_lang_list = faker_map["es" if target_lang in ("eu", "ca") else target_lang]
        faker_target_lang = Faker(random.choice(faker_lang_list))
        faker_target_lang.add_provider(person)
        faker_target_lang.add_provider(ssn)
        faker_target_lang.add_provider(address)
        faker_target_lang.add_provider(geo)
        faker_target_lang.add_provider(internet)
        faker_target_lang.add_provider(company)

      if src_lang not in ("eu", "ca") and src_lang not in faker_map:
        faker_src_lang = random.choice(self.faker_en_list)
      else:
        faker_lang_list = faker_map["es" if src_lang in ("eu", "ca") else src_lang]
        faker_src_lang = Faker(random.choice(faker_lang_list))
        faker_src_lang.add_provider(person)
        faker_src_lang.add_provider(ssn)
        faker_src_lang.add_provider(address)
        faker_src_lang.add_provider(geo)
        faker_src_lang.add_provider(internet)
        faker_src_lang.add_provider(company)

      faker_en = random.choice(self.faker_en_list)

    else:
      faker_target_lang = None
      faker_src_lang = None
      faker_en = None

    stopwords1 = set(stopwords.get(src_lang,[]))
    stopwords2 = set(stopwords.get(target_lang,[]))

    #init spacy pipeline
    spacy_nlp = None
    if do_spacy:
      if target_lang == 'en':
        spacy_nlp = self.en_spacy_nlp
      elif target_lang == 'zh':
        try:
          spacy_nlp = spacy.load('zh_core_web_sm')
        except:
          pass
      elif target_lang == 'pt':
        try:
          spacy_nlp = spacy.load('pt_core_news_sm')
        except:
          pass
      elif target_lang == 'fr':
        try:
          spacy_nlp = spacy.load('fr_core_news_sm')
        except:
          pass
      elif target_lang == 'ca':
        try:
          spacy_nlp = spacy.load('ca_core_news_sm')
        except:
          pass
    model = None
    ner_pipelines = []
    
    # init the kenlm pipeline
    if do_kenlm:
        if TextAugment.kenlm_model is None:
            TextAugment.load_kenlm_model()
            
    if target_lang != src_lang:
        if TextAugment.qg is None: TextAugment.qg = qg_pipeline.pipeline("multitask-qa-qg", TextAugment=self.device) # TODO make sure it's running in half mode
        if TextAugment.labse is None: 
            TextAugment.labse =  SentenceTransformer("sentence-transformers/LaBSE", cache_folder=os.path.expanduser ('~')+"/.cache").eval()
            if self.device == "cpu":
              TextAugment.labse  = torch.quantization.quantize_dynamic(TextAugment.labse , {torch.nn.Linear}, dtype=torch.qint8)
            else:
              TextAugment.labse  = TextAugment.labse.half().to(TextAugment.device)
        if "facebook/m2m100_418M" not in TextAugment.translation_pipelines:
          TextAugment.translation_pipelines["facebook/m2m100_418M"] =  M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").eval()
          if self.device == "cpu":
              TextAugment.translation_pipelines["facebook/m2m100_418M"] =  torch.quantization.quantize_dynamic(TextAugment.translation_pipelines["facebook/m2m100_418M"] , {torch.nn.Linear}, dtype=torch.qint8)
          else:
              TextAugment.translation_pipelines["facebook/m2m100_418M"] =  TextAugment.translation_pipelines["facebook/m2m100_418M"].half().to(TextAugment.device)
        if TextAugment.device_id >= 0:
            available_device_model = TextAugmentDeviceModel.available_device_models [TextAugment.device_id]
            if available_device_model is not None:
                if TextAugment.labse  is not None and available_device_model.labse is None: available_device_model.labse = TextAugment.labse 
                if TextAugment.qg is not None and available_device_model.qg  is None: available_device_model.qg = TextAugment.qg
                if TextAugment.translation_pipelines  is not None and not available_device_model.translation_pipelines : available_device_model.translation_pipelines = TextAugment.translation_pipelines 
                if TextAugment.ner_model_name2pipelines is not None and not available_device_model.ner_model_name2pipelines: available_device_model.ner_model_name2pipelines = TextAugment.ner_model_name2pipelines

    # init hf ner pipelines
    if do_hf_ner:
      for model_name, model_cls, hf_ner_weight2 in self.hf_ner_model_map.get(target_lang, []):
        if model_name not in self.ner_model_name2pipelines:
          #print ("setting")
          try:
            if self.device == 'cpu':
                model = model_cls.from_pretrained(model_name).eval()
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            else:
                model = model_cls.from_pretrained(model_name).half().eval().to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_len=512)
            if self.device == 'cpu':
              ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
            else:
              ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=self.device_id)
            self.ner_model_name2pipelines[model_name] = ner_pipeline
          except:
            logger.info("problems loading hf pipeline model and tokenizer. trying without passing in model and tokenizer")
            if self.device == 'cpu':
              ner_pipeline = pipeline("ner",  model=model_name, tokenizer=(model_name, {"use_fast": True},))
            else:
              ner_pipeline = pipeline("ner",  model=model_name, tokenizer=(model_name, {"use_fast": True},), device=self.device_id)
            self.ner_model_name2pipelines[model_name] = ner_pipeline
        ner_pipelines.append((model_name, self.ner_model_name2pipelines[model_name], hf_ner_weight2))
    target_is_cjk = target_lang in ('zh', 'ko', 'ja')
    src_is_cjk = src_lang in ('zh', 'ko', 'ja')
    lbracket = "[["
    rbracket = "]]"

    if do_augment:
      target_text_key = f'{target_lang}_text_aug'
      target_ner_key =  f'{target_lang}_signal_ner_aug'
      target_collapse_ner_key =  f'{target_lang}_ner_aug'
      target_offset_key = f'{target_lang}_offset_aug'
      target_src_sim_key = f'{src_lang}_2_{target_lang}_sim_aug'
    else:
      target_text_key = f'{target_lang}_text'
      target_ner_key = f'{target_lang}_signal_ner'
      target_collapse_ner_key = f'{target_lang}_ner'
      target_offset_key = f'{target_lang}_offset'
      target_src_sim_key = f'{src_lang}_2_{target_lang}_sim'

    public_figure_kenlm_cutoff = self.public_figure_kenlm_cutoff_map.get(target_lang, 450)

    docs = self.collapse_ner(docs, ner_key = f'{src_lang}_signal_ner', collapse_ner_key = f'{src_lang}_ner',  text_key = f'{src_lang}_text', stopwords=stopwords1)

    # do operations in the target_lang space
    if target_lang == src_lang:
      backtrans_weight = 1.0
      do_backtrans = False

    elif target_lang != src_lang:
        #translate from src_lang to target_lang and do ner in target_lang.  translation also acts as an error check and additional ner.
        #we check to see if the already tagged items in src lang should have scores for tags increased or are common words in target lang and should not be tagged.
        #we also add new labels for items that are already tagged in src_lang.


        if do_augment:
          self.create_augment_anon_context(docs, chunks, src_lang, faker_target_lang, faker_en, aug_scope=aug_scope, target_lang=target_lang, \
                                          items_key=f'{src_lang}_items', context_key=f'{src_lang}_aug_context', ner_key=f'{src_lang}_ner')
        docs, chunks = self.replace_items_in_chunks(docs, chunks,  src_lang, lbracket=lbracket, rbracket=rbracket, \
                                                                         replace_with_bracket=True, do_augment=do_augment, \
                                                                         context_key=f'{src_lang}_aug_context', \
                                                                         ner_key=f'{src_lang}_ner',  items_key=f'{src_lang}_items', \
                                                                         text_key=f'{src_lang}_text', replace_text_key=f'{src_lang}_tmp_text', \
                                                                         offset_key=f'{src_lang}_offset')
        chunks2 = [chunk[f'{src_lang}_tmp_text'] for chunk in chunks]
        text2 = self.do_translations(chunks2, src_lang=src_lang, target_lang=target_lang, batch_size=batch_size, do_marian_mt=do_marian_mt)
        for chunk, trans_text in zip(chunks, text2):
          #TODO: fix translations which sometimes doesn't split sentences on "."
          #langid check
          try:
            lang =  langid.classify(trans_text)[0]
          except:
            lang = target_lang
          if lang == target_lang:
            chunk[target_text_key] = trans_text.lstrip(" .").replace(")", "] ").replace("(", " [").replace(rbracket, "] ").replace(lbracket, " [").replace("}", ")").replace("{", "(").replace("[", " [ ").replace("]", " ] ").replace("  ", " ")
            chunk[target_text_key] = chunk[target_text_key].replace(",", ", ").replace("．", "． ").replace("。", "。").replace("、", "、 ").replace("《", " 《").replace("》", "》 ").replace("  ", " ")
          else:
            chunk[target_text_key] = " . . . "

        if len(chunks2) == 0:
          similarity = []
        else:
          all_embed = self.labse.encode(chunks2, convert_to_tensor=True)
          all_trans_embed = self.labse.encode([chunk[target_text_key] for chunk in chunks], convert_to_tensor=True)
          similarity = cosine_similarity(all_embed, all_trans_embed, dim=1)
        for chunk, sim_score in zip(chunks, similarity):
          trans_text = chunk[target_text_key]
          sim_score = sim_score.item()
          #print (sim_score, '**', trans_text, '**', chunk[f'{src_lang}_tmp_text'])
          _id = chunk['id']
          doc = docs[_id]
          if (do_augment and sim_score < 0.5) or (not do_augment and sim_score < 0.75):
            trans_text = chunk[target_text_key] = " . . . "
            if doc.get(target_text_key, ""):
              chunk[target_offset_key] = len(doc.get(target_text_key, "")) + 1
            else:
              chunk[target_offset_key] = 0
            doc[target_text_key] = (doc.get(target_text_key, "") + " " + trans_text).strip()
            chunk[target_src_sim_key] = 0.0
            continue
          chunk[target_src_sim_key] = sim_score
          len_items = len(doc[f'{src_lang}_items'])
          doc[f'{target_lang}_2_{src_lang}_tmp'] = doc.get(f'{target_lang}_2_{src_lang}_tmp', {})
          while "[" in trans_text:
            before, after = trans_text.split("[",1)
            before = before.strip()
            after = after.strip()
            before_arr = before.split()
            if "]" not in after or not before_arr:
              trans_text = before + " " if not target_is_cjk else "" + after
              continue
            idx = before_arr[-1]
            idx = re.findall(r'\d+$',idx)
            if idx:
              idx = idx[-1]
            else:
              idx = None
            ent, after = after.split("]", 1)
            ent = ent.strip()

            if True:
              try:
                idx = int(idx)
              except:
                idx = None
              if idx is not None and idx < len_items:
                before = " ".join(before_arr[:-1])
                key = doc[f'{src_lang}_items'][idx]
                mention = tuple(key[:-1])
                ent_lower = ent.lower()
                if ent_lower in stopwords2:
                  #reduce weight of target labels if this is translated into an target_lang stopword
                  if mention in doc[f'{src_lang}_signal_ner']:
                    aHash = doc[f'{src_lang}_signal_ner'][mention]
                    for key2 in list(aHash.keys()):
                      aHash[key2] /= 2.0
                else:
                  #vals = list(doc[f'{src_lang}_signal_ner'][mention].keys())
                  ent = ent.strip(self.strip_chars)
                  doc[f'{target_lang}_2_{src_lang}_tmp'][ent] = idx
            else: # except:
              pass
            trans_text = before + " " + ent + " " + after
          trans_text = chunk[target_text_key] = trans_text.replace("  ", " ").strip()
          if do_kenlm and target_lang == 'en':
              chunk[f'{target_lang}_kenlm'] = self.kenlm_model.get_perplexity(chunk[target_text_key])
          if doc.get(target_text_key, ""):
            chunk[target_offset_key] = len(doc.get(target_text_key, "")) + 1
          else:
            chunk[target_offset_key] = 0
          doc[target_text_key] = (doc.get(target_text_key, "") + " " + trans_text).strip()
    if do_kenlm and target_lang == 'en':
      for doc in docs.values():
        doc[f'{target_lang}_kenlm'] = self.kenlm_model.get_perplexity(doc[target_text_key].replace(" .", " "))

    if do_regex:
      docs = self.apply_regex_ner(target_lang, docs=docs, weight=regex_weight, text_key=target_text_key, ner_key=target_ner_key)

    if do_ontology and self.ontology_manager is not None:
        # dictionary matching context independent so has lower accuracies
        for doc in docs.values():
          doc[target_ner_key] = ner = doc.get(target_ner_key, {})
          if True:
            chunk2ner = self.ontology_manager.tokenize(doc[target_text_key])['chunk2ner']
            onto_items = []
            for c, label in chunk2ner.items():
              if label not in ("PERSON", "PUBLIC_FIGURE"): continue # hard coded to only do people for now
              ner_word  = c[0].replace(" ", "").replace("_", "").replace("_", "") if self.cjk_detect(c[0]) else c[0].replace("_", " ").replace("_", " ").rstrip(self.strip_chars)
              if ner_word.lower() not in stopwords2:
                if not self.cjk_detect(ner_word) and label in ('PERSON', 'PUBLIC_FIGURE', 'ORG') and " " not in ner_word: continue
                onto_items.append(((ner_word, c[1], c[1] + len(ner_word)), label))
            for ner_mention, label in list(set(onto_items)):
                aHash = ner.get(ner_mention, {})
                aHash[(label, 'onto')] = aHash.get((label, 'onto'), 0) + ontology_weight * (1.0 + len(ner_mention[0])/100) * backtrans_weight
                ner[ner_mention] = aHash

    if do_spacy:
        if spacy_nlp:
          # spacy
          self.spacy_ner(docs, spacy_nlp, stopwords2, spacy_weight, target_lang, extra_weight=backtrans_weight, text_key=target_text_key, ner_key=target_ner_key)

    if do_hf_ner:
        # transformer
        for model_name, ner_pipeline, hf_ner_weight2 in ner_pipelines:
          for a_batch in self.batch(chunks, batch_size):
            self.hf_ner(ner_pipeline, target_lang, docs, a_batch, stopwords=stopwords2, weight=hf_ner_weight*backtrans_weight*hf_ner_weight2, text_key=target_text_key, \
                        ner_key=target_ner_key, signal=model_name, offset_key=target_offset_key)

    if do_qg_rel and target_lang == 'en':
      docs = self.generate_questions_answers_rel(docs, chunks, target_lang, ner_key=target_ner_key)

    docs = self.collapse_ner(docs, target_ner_key, target_collapse_ner_key, target_text_key, stopwords2, do_cleanup_only=True)

    if do_docs_trim_for_person:
      docs, chunks = self.trim_to_prefer_person(docs, chunks)

    if do_kenlm and self.kenlm_model is not None:
      for doc in docs.values():
        ner = doc[target_ner_key]
        prev_public_figures = []
        for ent, aHash in ner.items():
          if any(key[0] == 'PUBLIC_FIGURE' for key in aHash.keys()):
            ent = ent[0].strip(".")
            if not target_is_cjk:
              ent_arr = ent.split()
              if len(ent_arr[-1]) == 1: continue
              if len(ent_arr) == 2 and len(ent_arr[0].strip(".")) == 1: continue
            prev_public_figures.append(ent)
        
        persons = []
        for ent, aHash in ner.items():
          if any(key[0] in ('PUBLIC_FIGURE', 'PERSON') for key in aHash.keys()):
            ent = ent[0].strip(".")
            if not target_is_cjk:
              ent_arr = ent.split()
              if not ent_arr: continue
              if len(ent_arr[-1]) == 1: continue
              if len(ent_arr) == 2 and len(ent_arr[0].strip(".")) == 1: continue
            persons.append(ent)
        persons += prev_public_figures
        prev_public_figures = set(prev_public_figures)
        public_figures = []
        for ent in list(set(persons)):
          ent2 = ent
          if not target_is_cjk and ent == ent.upper():
            ent2 = " ".join([a[0].upper()+a[1:] if len(a) > 1 else a.upper() for a in ent.lower().split()])
          kenlm_score = self.kenlm_model.get_perplexity(f"{ent2} (born")
          #logger.info((ent, kenlm_score))
          if kenlm_score <= public_figure_kenlm_cutoff:
            logger.info(("found public figure ", ent2, kenlm_score))
            public_figures.append(ent)
          else:
            if ent2 in prev_public_figures:
              logger.info(("**not public figure ", ent2, kenlm_score))
            else:
              logger.info(("not public figure ", ent2, kenlm_score))
            pass 
            
        public_figures = set(public_figures)
        for ent, aHash in ner.items():
          if ent[0].strip(".") in public_figures:
            #logger.info(("adding knelm public figure", ent))
            aHash[('PUBLIC_FIGURE', 'kenlm')] = aHash.get(('PUBLIC_FIGURE', 'kenlm'), 0) + 1.0 # use param kenlm_weight

    if do_public_figure_expansion:
      for doc in docs.values():
        text  = doc[target_text_key]
        len_text = len(text)
        ner = doc[target_ner_key]
        public_figures = []
        for ent, aHash in ner.items():
          if any(key[0] in ('PUBLIC_FIGURE',) for key in aHash.keys()):
            ent1 = ent[0].strip(".")
            if not target_is_cjk:
              for ent2 in ent1.split():
                if len(ent2.strip(".,")) <= 4: continue
                public_figures.append((ent2, ent[1], ent[2]))
        f = lambda x: x[0]
        for ent, group in itertools.groupby(sorted(public_figures, key=f), f):
          spans = [(a[1], a[2]) for a in group]
          label = 'PUBLIC_FIGURE'
          pos = 0
          while pos < len_text and ent in text[pos:]:
            i = text[pos:].index(ent)
            start = pos + i
            end = start + len(ent)
            pos = end+1
            if any(s for s in spans if s[0]<=start and s[1]>=end):  continue
            mention2 = (ent, start, end)
            if any(mention for mention in ner if mention[1] <= mention2[1] and mention[2] >= mention2[2]):
              ner[mention2] = aHash1 = ner.get(mention2, {})
              aHash1[(label, 'pf_expand')] = aHash1.get((label, 'pf_expand'), 0) + 1.0
              logger.info(("pf_expand", mention2))

    # this will mess up the items array and the other arrays that depends on mentions unles we do_cleanup_only
    docs = self.collapse_ner(docs, target_ner_key, target_collapse_ner_key, target_text_key, stopwords2, do_cleanup_only=True)

    if target_lang != src_lang and not do_augment:
          for doc in docs.values():
            ner =  doc[f'{target_lang}_signal_ner']
            src_ner = doc[f'{src_lang}_signal_ner']
            target2src_ner = doc.get(f'{target_lang}_2_{src_lang}_tmp', {})
            #increase weight of src ner_lang items if the target_lang translations indicate it's an NER
            for ent, idx in target2src_ner.items():
              key = doc[f'{src_lang}_items'][idx]
              mention = tuple(key[:-1])
              #NOTE that this is an unordered match
              ner_match = [key2 for key2 in ner if ent == key2[0]]
              if not ner_match and len(ent) > 3:
                ner_match = [key2 for key2 in ner if (ent in key2[0] or (len(key2[0]) > 3 and key2[0] in ent))]
              if ner_match:
                if mention in src_ner:
                  aHash = src_ner[mention]
                  all_labels = []
                  for key2 in ner_match:
                    all_labels.extend(list(ner[key2].keys()))
                  all_labels = set(all_labels)
                  found = False
                  for label in list(aHash.keys()):
                    if label in all_labels or 'MISC' in all_labels:
                      aHash[label] *= 1.1
                      #print ('increasing ', mention, label, aHash[label])
                      found = True
                  if not found:
                    pass
                    #print ('not found', mention, all_labels)

    if do_backtrans and target_lang != src_lang and not do_augment:
        #TBD: we could run the augmented text back to the original sentence create additional augmented data.
        #backtrans from src_lang to target_lang back to src_lang. this allows us to catch more NER using target lang NER tools.
        #then we tag in target_lang those items we haven't already found, and tranlsate back to match the original text.
        #NOTE: We do not modify the original text, but only use backtrans to do NER tagging and other analysis.
        if src_is_cjk:
            sep = ""
        else:
            sep = " "
        for doc in docs.values():
          doc[f'{target_lang}_2_{src_lang}_tmp'] = doc.get(f'{target_lang}_2_{src_lang}_tmp', {})
          aHash = doc[f'{target_lang}_2_{src_lang}_tmp']
          items = doc[f'{src_lang}_items']
          doc[f'{target_lang}_2_{src_lang}_context'] = dict([(a, items[b][0]) for a, b in aHash.items()])
        docs, chunks = self.replace_items_in_chunks(docs, chunks,  src_lang, lbracket=lbracket, rbracket=rbracket, \
                                                                         replace_with_bracket=True, do_augment=True, \
                                                                         ner_key=f'{target_lang}_signal_ner', items_key=f'{target_lang}_items', \
                                                                         text_key=f'{target_lang}_text', replace_text_key=f'{target_lang}_tmp_text', \
                                                                         offset_key=f'{target_lang}_offset', context_key=f'{target_lang}_2_{src_lang}_context')

        backtrans_text = self.do_translations([chunk[f'{target_lang}_tmp_text'] for chunk in chunks], src_lang=target_lang, target_lang=src_lang, batch_size=batch_size)
        for chunk, trans_text in zip(chunks, backtrans_text):
          #TODO: fix translations where there is no " " after a "." for some sentences.
          #langid check
          try:
            lang =  langid.classify(trans_text)[0]
          except:
            lang = src_lang
          #print (lang, '**', trans_text)
          if lang == src_lang:
            chunk[f'{src_lang}_text_backtrans_from_{target_lang}'] = trans_text.lstrip(" .").replace(")", "] ").replace("(", " [").replace(rbracket, "] ").replace(lbracket, " [").replace("}", ")").replace("{", "(").replace("[", " [ ").replace("]", " ] ").replace("  ", " ")
            chunk[f'{src_lang}_text_backtrans_from_{target_lang}'] =  chunk[f'{src_lang}_text_backtrans_from_{target_lang}'].replace(",", ", ").replace("．", "． ").replace("。", "。").replace("、", "、 ").replace("《", " 《").replace("》", "》 ").replace("  ", " ")
          else:
            chunk[f'{src_lang}_text_backtrans_from_{target_lang}'] = " . . . "

        if len(backtrans_text) == 0:
          similarity = []
        else:
          all_embed = self.labse.encode(backtrans_text, convert_to_tensor=True)
          all_trans_embed = self.labse.encode([chunk[f'{src_lang}_text'] for chunk in chunks], convert_to_tensor=True)
          similarity = cosine_similarity(all_embed, all_trans_embed, dim=1)
        for chunk, trans_text, sim_score in zip(chunks, backtrans_text, similarity):
          _id = chunk['id']
          doc = docs[_id]
          offset = chunk[f'{src_lang}_offset']
          src_text = chunk[f'{src_lang}_text']
          trans_text = chunk[f'{src_lang}_text_backtrans_from_{target_lang}']
          items = doc[f'{target_lang}_items']
          len_items = len(items)
          ner = doc[f'{target_lang}_signal_ner']
          bner = doc[f'{src_lang}_2_{target_lang}_backtrans_ner_tmp'] = doc.get(f'{src_lang}_2_{target_lang}_backtrans_ner_tmp', {})
          pos = 0
          sim_score = sim_score.item()
          #print ('backtrans match', sim_score, orig_text, '**', trans_text)
          if sim_score < 0.70:
            chunk[f'{src_lang}_text_backtrans_from_{target_lang}'] = " . . . "
            #TODO - save away sim_score ?
            continue
          orig_text = src_text.replace("。", ".").replace("،", ",").replace("、", ",").replace("`", "'").replace("“", "\"").replace("”", "\"").replace("《", "\"").replace("》", "\"").replace("«", "\"").replace("»", "\"")
          trans_text = trans_text.replace("。", ".").replace("،", ",").replace("、", ",").replace("`", "'").replace("“", "\"").replace("”", "\"").replace("《", "\"").replace("》", "\"").replace("«", "\"").replace("»", "\"")
          blocks, score =  self.get_aligned_text(orig_text, trans_text, src_lang)

          # check score?
          prev_t = None
          prev_o = None
          ner_word = ""
          ent2 = ""
          idx = None
          before = after = ""
          for o, t, _ in blocks:
            if "[" in t:
              ner_word = ""
              ent2 = ""
              before, after = t.split("[",1)
              before = before.strip()
              if before:
                idx = re.findall(r'\d+$',before)
                try:
                  idx = int(idx[-1])
                except:
                  idx = None
                if idx is None and prev_t and prev_t.strip():
                  idx = re.findall(r'\d+$',prev_t.strip())
                  try:
                    idx = int(idx[-1])
                  except:
                    idx = None

            if idx is not None and idx >= len_items:
              logger.info (('err idx out of range of items', idx, t, o))
            if idx is not None and idx < len_items:
              ner_word += o
              if "[" in t:
                ent2 += t.split("[",1)[-1].split("]")[0].strip()
              else:
                ent2 += t.split("]")[0].strip()
              if "]" in t:
                #print (idx, items)
                key = items[idx]
                mention = tuple(key[:-1])
                if True:
                  ner_word = ner_word.strip(self.strip_chars)
                  ent2 = ent2.strip(self.strip_chars)
                  if ent2 in ner_word:
                      ner_word = ent2
                  else:
                      if src_is_cjk:
                        ent2arr = list(ent2)
                        ner_wordarr = list(ner_word)
                      else:
                        ent2arr = ent2.split()
                        ner_wordarr = ner_word.split()
                      len_ent2arr = len(ent2arr)
                      if False:
                        found=False
                        if  len_ent2arr > 3:
                          ent3 = sep.join(ent2arr[:3])
                          if ent3 in new_word:
                            new_word = ner_word[ner_word.index(ent3):]
                            found=True
                        if not found:
                          if len_ent2arr < len(ner_wordarr):
                            new_word = sep.join(ner_wordarr[-len_ent2arr:])
                  #TODO, add ent2 if ent2 in orig_text
                  if ner_word and ner_word.lower() not in stopwords1:
                      ner_word = ner_word.strip(self.strip_chars+".。")
                      #print (ner_word, ner_word in orig_text, orig_text)
                      if ner_word not in orig_text[pos:]:
                        logger.info( ('cant find in orig_text ', ner_word, '**', orig_text[pos:], '**', orig_text))
                      else:
                        i = orig_text[pos:].index(ner_word)
                        start = pos + i 
                        len_nerword = len(ner_word)
                        pos = start + len_nerword
                        ner_word = src_text[offset + start:offset + start + len_nerword]
                        mention2 = (ner_word, offset + start, offset + start + len_nerword)
                        aHash = bner.get(mention2, {})
                        for label in ner[mention]:
                          #print (f'found new mention from {target_lang}', mention, mention2, label)
                          aHash[label] = aHash.get(label, backtrans_weight) + ner[mention][label]
                        bner[mention2] = aHash
                  idx = None
                  ner_word = ""
                  ent2 = ""
            prev_o, prev_t = o, t

        #print ('bner', bner)
        #*** the actual mapping from the target_lang ner back to the src_lang ner ***
        #copy into the src_lang ner score if we matched an ner in backtrans src_lang, increase scores if there was a partial match
        #TODO: for all persons and public figures we map from target_lang back to src_lang, also map the gender
        for doc in docs.values():
          bner = doc[f'{src_lang}_2_{target_lang}_backtrans_ner_tmp']
          ner = doc[f'{src_lang}_signal_ner']
          # partial match
          for key, aHash in bner.items():
            if key in ner: continue # we do the full match below
            ent = key[0]
            ner_match = [key2 for key2 in ner if ent == key2[0]]
            if not ner_match and len(ent) > 3:
              ner_match = [key2 for key2 in ner if (ent in key2[0] or (len(key2[0]) > 3 and key2[0] in ent))]
            all_keys = []
            for key2 in ner_match:
              all_keys.extend(list(ner[key2].keys()))
            all_keys = set(all_keys)
            #there was another ner item that partially matches this one and has the same label, so let's increase the
            #ner label score for this item.
            for label in list(aHash.keys()):
              if label in all_keys or 'MISC' in all_keys:
                    aHash[label] *= 1.1
                    #print ('increasing in backtrans ', key, label, aHash[label])
          #full match
          for key, aHash1 in bner.items():
            ner[key] = aHash2 = ner.get(key, {})
            for key2 in aHash1:
              aHash2[key2] = aHash2.get(key2, 0.0) + aHash1[key2]

    if target_lang != src_lang and not do_augment:
          for doc in docs.values():
            ner =  doc[f'{target_lang}_signal_ner']
            src_ner = doc[f'{src_lang}_signal_ner']
            target2src_ner = doc.get(f'{target_lang}_2_{src_lang}_tmp', {})
            #print (target2src_ner)
            #add labels from src ner to target ner for matches
            for ent, idx in target2src_ner.items():
              key = doc[f'{src_lang}_items'][idx]
              text = doc[f'{target_lang}_text']
              mention = tuple(key[:-1])
              if mention in src_ner:
                aHash = src_ner[mention]
                pos = 0
                len_text = len(text)
                while pos < len_text and ent in text[pos:]:
                  i = text[pos:].index(ent)
                  start = pos + i
                  end = start + len(ent)
                  pos = end+1
                  mention2 = (ent, start, end)
                  ner[mention2] = aHash1 = ner.get(mention2, {})
                  for label in aHash:
                    aHash1[label] = aHash1.get(label, 0) + aHash[label]

    #now src_lang ner and target_lang ner will have very similar ner items

    docs = self.collapse_ner(docs, target_ner_key, target_collapse_ner_key, target_text_key, stopwords2)
    if target_lang != src_lang:
      docs = self.collapse_ner(docs, ner_key = f'{src_lang}_signal_ner', collapse_ner_key = f'{src_lang}_ner',  text_key = f'{src_lang}_text', stopwords=stopwords1)

    if do_anonymization and faker_src_lang is not None and faker_en is not None:
      docs, chunks = self.anonymize(docs, chunks, src_lang, faker_src_lang, faker_en, anon_scope=anon_scope)

    #TODO: remove all _tmp fields

    return docs, chunks

  def process_ner(self,
              docs,
              src_lang = None,
              do_spacy = True,
              do_hf_ner = True,
              do_ontology = True,
              do_skip_src_lang_processing=False,
              do_backtrans=False,
              do_augment=False,
              do_anonymization=False,
              do_marian_mt=True,
              copy_anon_to_text=True, # if we do_anonymize, we will copy {src_lang}_text_anon -> text
              augment_lang="es",
              do_cleanup=True,
              do_regex = True,
              batch_size = 5,
              num_words_per_chunk=70,
              ontology_weight=0.85,
              spacy_weight=1.00,
              hf_ner_weight=1.25,
              regex_weight=1.5,
              backtrans_weight=0.9,
              do_docs_trim_for_person=False,
              do_docs_filter=False,
              do_qg_rel=False,
              do_kenlm = True,
              cutoff=None,
              target_lang=None,
              domain="",
              aug_scope={'ADDRESS', 'ORG', 'PERSON', 'LOC', 'ID'}, #TODO, public figure, age, norp and disease
              anon_scope={'PERSON', 'ID'}):
      """
      This is the main routine to perform crosslingual NER for a src_lang document with potentially no NER models.
      It uses a cross lingual NER model that is 'close enough', and also uses backtranslation (target_lang English) to do further NER, and then map back to src_lang.
      It can also create crosslingual augmented data to create additional data for training.
      This routine can also be used to do anonymization of the original src_lang text at the end of the NER pipeline.
      Note: This code will have a side-effect on the docs.
      """
      #print ("process_ner")
      if TextAugment.device is None:
        self.initializer()
      if type(docs) is tuple:
        docs, src_lang, target_lang = docs
      if target_lang is None:
        target_lang = "en"
      assert src_lang, "a source language needs to be specified"
      #print (self.ner_model_name2pipelines)
      src_is_cjk = src_lang in ('zh', 'ko', 'ja')
      if src_is_cjk:
        sep = ""
      else:
        sep = " "
      if type(docs) is dict:
        docs = list(docs.values())      
      #for testing only
      if cutoff is not None and cutoff > 0 and len(docs) > cutoff:
        docs = docs[:cutoff*3]
      #print (docs)
      len_docs = len(docs)
      _id = 0
      # use the 'text' field as the current working src_lang_text field unless there is one already
      for doc in docs:
        if 'text' in doc:
          doc['text'] = doc['text'].replace("．", "． ").replace("。", "。").replace("、", "、 ").replace("《", " 《").replace("》", "》 ").replace("  ", " ")
        if 'id' in doc:
          _id = max(_id, int(doc['id']))
        if f'{src_lang}_text' in doc:
          if 'text' not in doc:
            # we won't cleanup src_lang_text because that might mess up mention spans.
            doc['text'] = doc[f'{src_lang}_text']
          continue
        if 'text' in doc:
          doc[f'{src_lang}_text'] = doc['text']
        else:
          #print ('problem**', doc)
          doc['text'] = doc[f'{src_lang}_text'] = ' . . . '

      flagged_words1 = set([s for s in flagged_words.get(src_lang, []) if len(s) < 5])
      stopwords1 = set(stopwords.get(src_lang, []))
      if do_docs_filter:
        lang_groups=TextAugment.get_lang_groups(src_lang)
        docs = [doc for doc in docs if self.check_good_sentence(doc[f'{src_lang}_text'], src_lang, lang_groups=lang_groups, stopwords=stopwords1, flagged_words=flagged_words1)]
        #print ('trimmed junk', (len_docs-len(docs))/len_docs)
            #for testing only
      if cutoff is not None and cutoff > 0 and len(docs) > cutoff:
        docs = docs[:cutoff*3]
      len_docs = len(docs)
      
      counter = {}
      chunks = []
      for doc in docs:
        if 'id' not in doc or int(doc['id']) < 0:
          doc['id'] = str(_id)
          _id += 1
        doc[f'{src_lang}_text'] = doc[f'{src_lang}_text'].replace("[", "(").replace("]", ")") # we use [] as special chars
        doc['lang'] = doc.get('lang', src_lang)
        doc['domain'] = doc['domain'] if doc.get('domain') is not None else domain
        doc['chunks'] = doc.get('chunks', [])
        
        #simple multi-lingual tokenizer and sentence splitter
        offset = 0
        if src_is_cjk:
          text = list(doc[f'{src_lang}_text'].replace("。", "。 ").replace("  ", " "))
        else:
          textarr = doc[f'{src_lang}_text'].replace("  ", " ").split()
          text = []
          for t in textarr:
            len_t = len(t)
            if len_t == 1: 
              text.append(t)
              continue
            punc_found = [punc for punc in t if punc in self.punc_char]
            word1, word2 = "", ""
            if punc_found:
              tarr = t.split(punc_found[0])
              word1 = tarr[-2]
              word2 = tarr[-1]
            if punc_found and t[-1] not in self.punc_char and \
                              ((punc_found[0] not in ".。") or \
                               (t[0] not in "0123456789" and t[0] == t[0].lower()) or \
                               (word1 and word1[-1] in self.strip_chars) or \
                               (word2 and word2[0] in self.strip_chars)):
              w = t[t.index(punc_found[0])+1]
              if w == w.upper():
                t, t1 = t.split(punc_found[0],1)
                t = t+punc_found[0]
                text.append(t)
                text.append(t1)
                continue
            text.append(t)
        text[0] = text[0].lstrip()
        text[-1] = text[-1].rstrip()
        doc[f'{src_lang}_text'] = sep.join(text)
        len_text = len(text)
        src_text = ""
        while len_text > num_words_per_chunk:
            for j in range(num_words_per_chunk-1, len_text):
              if j > num_words_per_chunk * 2: break 
              if (src_is_cjk and text[j] in self.punc_char+' ') or \
                  (not src_is_cjk and text[j][-1] in self.punc_char):
                break
            text_str = sep.join(text[:j+1])
            chunks.append({f'{src_lang}_text': text_str, 'id': doc['id'], f'{src_lang}_offset': offset})
            doc['chunks'].append(chunks[-1])
            offset += len(text_str) + (0 if src_is_cjk else 1)
            text = text[j+1:]
            len_text = len(text)
        if text:
            text_str = sep.join(text)
            chunks.append({f'{src_lang}_text': text_str, 'id': doc['id'], f'{src_lang}_offset': offset})
            doc['chunks'].append(chunks[-1])
      
      # store as a dictionary for easy lookup
      docs = dict([(doc['id'], doc) for doc in docs])
      if do_docs_trim_for_person:
        docs2, chunks2 = self.trim_to_prefer_person(docs, chunks)
        do_docs_trim_for_person = len(docs2) == len(docs)
        docs, chunks = docs2, chunks2

      # we do this here because we don't want to trim  ner items that are considered empty.
      # we should probably fix trim_to_prefer_person to not do any trimming if all ner's are empty
      for doc in docs.values():
        doc[f'{src_lang}_signal_ner'] = doc.get(f'{src_lang}_signal_ner', {})

      if not do_skip_src_lang_processing:
        #do ner processing in src_lang with potential anonymization
        docs2, chunks2 = self.process_ner_chunks_with_trans(
                          src_lang,
                          docs,
                          chunks,
                          target_lang=src_lang,
                          do_spacy = do_spacy,
                          do_hf_ner = do_hf_ner,
                          do_ontology = do_ontology,
                          do_backtrans=False,
                          do_augment=False,
                          do_anonymization=do_anonymization if target_lang == src_lang else False,
                          do_kenlm = do_kenlm,
                          do_marian_mt=do_marian_mt,
                          do_regex = do_regex,
                          do_cleanup=do_cleanup,
                          batch_size = batch_size,
                          ontology_weight=ontology_weight,
                          spacy_weight=spacy_weight,
                          hf_ner_weight=hf_ner_weight,
                          regex_weight=regex_weight,
                          backtrans_weight=backtrans_weight,
                          do_qg_rel=do_qg_rel and src_lang == 'en',
                          do_docs_trim_for_person=do_docs_trim_for_person)
        if do_docs_trim_for_person:
          do_docs_trim_for_person = len(docs2) == len(docs)
        docs, chunks = docs2, chunks2

      if target_lang != src_lang:
        #do ner processing in target language with optional backtrans and anonymization
        docs2, chunks2 = self.process_ner_chunks_with_trans(
                            src_lang,
                            docs,
                            chunks,
                            target_lang = target_lang,
                            do_spacy = do_spacy,
                            do_hf_ner = do_hf_ner,
                            do_ontology = do_ontology,
                            do_backtrans=do_backtrans,
                            do_augment=False,
                            do_anonymization=do_anonymization,
                            do_regex = do_regex,
                            do_cleanup = do_cleanup,
                            do_qg_rel=do_qg_rel and target_lang == 'en',
                            do_kenlm = do_kenlm,
                            do_marian_mt=do_marian_mt,
                            batch_size = batch_size,
                            ontology_weight=ontology_weight,
                            spacy_weight=spacy_weight,
                            hf_ner_weight=hf_ner_weight,
                            regex_weight=regex_weight,
                            backtrans_weight=backtrans_weight,
                            do_docs_trim_for_person=do_docs_trim_for_person)
        docs, chunks = docs2, chunks2

      #TODO: do the case where we only anonymize
      #if do_skip_src_lang_processing and target_lang == src_lang and do_anonymization:
      #   docs, chunks = self.anonymize(docs, chunks, src_lang, faker_src_lang, faker_en, anon_scope=anon_scope)

      if copy_anon_to_text and do_anonymization:
        for doc in docs.values():
          if f'{src_lang}_text_anon' in doc:
            doc['text'] = doc[f'{src_lang}_text_anon']

      assert not do_augment or augment_lang not in (src_lang, target_lang), "augmented langauge should be different than src_lang and target_lang"

      if do_augment:
        #do data augmentation by adding a new text field with fake names, etc. in augment_lang
        docs2, chunks2 = self.process_ner_chunks_with_trans(
                            src_lang,
                            docs,
                            chunks,
                            target_lang = augment_lang,
                            do_spacy = do_spacy,
                            do_hf_ner = do_hf_ner,
                            do_ontology = do_ontology,
                            do_backtrans=False,
                            do_augment=do_augment,
                            do_marian_mt=do_marian_mt,
                            do_anonymization=False,
                            do_regex = do_regex,
                            do_cleanup=do_cleanup,
                            do_qg_rel=do_qg_rel and augment_lang == 'en',
                            do_kenlm = do_kenlm,
                            batch_size = batch_size,
                            ontology_weight=ontology_weight,
                            spacy_weight=spacy_weight,
                            hf_ner_weight=hf_ner_weight,
                            regex_weight=regex_weight,
                            backtrans_weight=backtrans_weight,
                            do_docs_trim_for_person=do_docs_trim_for_person)
        docs, chunks = docs2, chunks2

      return list(docs.values()) #this is not guaranteed to be ordered

  @staticmethod
  def get_docs(src_langs=None, docs=None, max_chunk_size=25, num_workers=1, cutoff=-1, hfdataset="", filter_out_no_registry=True):
      """
      NOTE: We filter the TurkuNLP registry docs if there are no registry label for a doc. This might not be the ideal behaviour.
      """
      #print("Intialize Documents")

      def get_chunk_size(cutoff, len_docs, num_workers, max_chunk_size):
        suggested_chunk_size = int(len_docs/num_workers)
        if cutoff is not None and cutoff > 0:
          suggested_chunk_size = int(num_workers/cutoff)
          if suggested_chunk_size <= 1:
              suggested_chunk_size = cutoff
        return min(max_chunk_size, suggested_chunk_size)
      
      def load_py_from_str(s, default=None):
        if not s.strip(): return default
        ret = {'__ret': None}
        exec("__ret= "+s, ret)
        return ret['__ret']

      if hfdataset:
          d = load_dataset(*hfdataset.split(",")) 
          len_docs = len(d)
          chunk_size = get_chunk_size(cutoff, len_docs, num_workers, max_chunk_size)
          curr_recs = 0
          for i in range(0, len_docs, chunk_size):
              j = min(i + chunk_size, len_docs)
              curr_recs += j - i
              if cutoff > 0 and curr_recs >= cutoff:
                j -= curr_recs - cutoff
              if i <= j: yield [d['train'][k] for k in range(i,j)]
              if cutoff > 0 and curr_recs >= cutoff: break
      elif not docs and src_langs is not None:
        if type(src_langs) is str: src_langs = [src_langs]
        
        # we will load the data from turkunlp_data
        for src_lang in src_langs:
          use_load_py_from_str=False
          _file = os.path.abspath(os.path.dirname(__file__))+f"/turkunlp_data/{src_lang}_data.jsonl.gz"
          if os.path.exists(_file):
                chunk_size = get_chunk_size(cutoff, 1000000, num_workers, max_chunk_size)
                cnt = 0
                ret = []
                with gzip.open(_file, "rb") as f:
                  for t in f:
                    t = t.decode()
                    dat = None
                    if not use_load_py_from_str: 
                      try:
                        dat = json.loads(t)
                      except:
                        use_load_py_from_str = True
                    if use_load_py_from_str:
                      dat = load_py_from_str(t, {})
                    if dat:
                      if cutoff is not None and cutoff > 0 and cnt >= cutoff:
                        yield ret
                        ret = []
                        break
                      else:
                        if cnt  % chunk_size == 0 and ret:
                          yield ret
                          ret = []
                        ret.append(dat)
                        cnt += 1
                if ret:
                  yield ret
                  ret = []
          else:
            raise RuntimeError("can't load dataset")
      elif isinstance(docs, str):
          yield [{'text': docs}]
      elif isinstance(docs, list):
          if isinstance(docs[0], dict):
            len_docs=len(docs)
            chunk_size = get_chunk_size(cutoff, len_docs, num_workers, max_chunk_size)
            for i in range(0, len_docs, chunk_size):
                j = min(i + chunk_size, len_docs)
                yield docs[i:j]
          else:
            len_docs=len(docs)
            chunk_size = get_chunk_size(cutoff, len_docs, num_workers, max_chunk_size)
            for i in range(0, len_docs, chunk_size):
                j = min(i + chunk_size, len_docs)
                yield [{'text': t} for t in docs[i:j]]
      elif not docs:
        yield []
      else:  
        yield docs

  @staticmethod
  def preload_cache(src_langs=["en"], target_langs=["en"], domain=None):
    #print ("preload_cache")
    SentenceTransformer("sentence-transformers/LaBSE", cache_folder=os.path.expanduser ('~')+"/.cache")
    en_spacy_nlp = spacy.load('en_core_web_sm')
    try:
      coref = neuralcoref.NeuralCoref(en_spacy_nlp.vocab)
    except:
      logger.info("neuralcoref not loaded!")
      pass
    arr2 = []
    AutoTokenizer.from_pretrained("google/mt5-small")
    for arr in TextAugment.hf_ner_model_map.values():
      for model_name, _, _ in arr:
        arr2.append(model_name)
    for model_name in list(set(arr2)):
        AutoModel.from_pretrained(model_name)
        AutoTokenizer.from_pretrained(model_name, model_max_len=512)
        AutoConfig.from_pretrained(model_name)
    for model_name in TextAugment.m2m100_lang.values():
        AutoModel.from_pretrained(model_name, model_max_len=512)
        AutoTokenizer.from_pretrained(model_name)
        AutoConfig.from_pretrained(model_name)
    for aHash in qg_pipeline.SUPPORTED_TASKS.values():
      for model_name in aHash["default"].values():
        AutoModel.from_pretrained(model_name, model_max_len=512)
        AutoTokenizer.from_pretrained(model_name)
        AutoConfig.from_pretrained(model_name)
    seen = {}
    for src_lang, target_lang in zip(src_langs, target_langs):
        if (src_lang, target_lang) not in seen: 
          model_name = marian_mt.get((src_lang, target_lang))
          seen[(src_lang, target_lang)] = 1
          if model_name is not None:
            AutoModel.from_pretrained(model_name)
            AutoTokenizer.from_pretrained(model_name, model_max_len=512)
            AutoConfig.from_pretrained(model_name)
        if (target_lang, src_lang) not in seen: 
          model_name = marian_mt.get((target_lang, src_lang))
          seen[(target_lang, src_lang)] = 1
          if model_name is not None:
            AutoModel.from_pretrained(model_name)
            AutoTokenizer.from_pretrained(model_name, model_max_len=512)
            AutoConfig.from_pretrained(model_name)                
    TextAugment.load_kenlm_model(store_model=False)
               
  @staticmethod
  def multiprocess_ner(docs,
                    outfile,
                    src_langs,
                    target_langs,
                    hfdataset=None,
                    do_spacy = True,
                    do_hf_ner = True,
                    do_ontology = True,
                    do_skip_src_lang_processing=False,
                    do_backtrans=False,
                    do_augment=False,
                    do_anonymization=False,
                    augment_lang=None,
                    do_cleanup=True,
                    do_regex = True,
                    do_marian_mt = True,
                    batch_size = 5,
                    num_words_per_chunk=70,
                    ontology_weight=0.85,
                    spacy_weight=1.00,
                    hf_ner_weight=1.25,
                    regex_weight=1.5,
                    backtrans_weight=0.9,
                    do_docs_trim_for_person=False,
                    do_docs_filter=False,
                    do_qg_rel=False,
                    do_kenlm = True,
                    cutoff=None,
                    num_workers=2):

    logger.info( ("multiprocess_ner ", outfile, src_langs))
    assert num_workers >= 2, "Can't do multiprocessing with less than 2 workers"
    multiprocessing.set_start_method('spawn', force=True)
    if type(src_langs) is str: src_langs = [src_langs]
    if type(target_langs) is str: target_langs = [target_langs]
    if src_langs is None: src_langs = ["en"]
    if target_langs is None: target_langs = ["en"]*len(src_langs)
    start = time.time()
    TextAugmentDeviceModel.initializer_all(src_langs=src_langs, target_langs=target_langs)
    processor = TextAugment(single_process=False)
    # processor.initializer()
    logger.info(("creating multiprocessing pool for num_workers ", num_workers, TextAugmentDeviceModel.available_device_models))
    pool = multiprocessing.Pool(processes=num_workers, initializer= partial(processor.initializer, all_available_device_model=TextAugmentDeviceModel.available_device_models  ))      
    if outfile is not None:
      _file =  open(outfile, 'w', encoding='utf-8')
    else:
      _file = None
    for src_lang, target_lang in zip(src_langs, target_langs):
      if outfile is None:
        if _file is not None: _file.close()
        _file = open(f"{src_lang}_out.jsonl", 'w', encoding='utf-8')
      docs = TextAugment.get_docs(src_lang, hfdataset=hfdataset, cutoff=cutoff, num_workers=num_workers)
      processed_docs = pool.imap_unordered(partial(processor.process_ner,
                                                      src_lang=src_lang,
                                                      target_lang=target_lang,
                                                      do_spacy = do_spacy ,
                                                      do_hf_ner = do_hf_ner ,
                                                      do_ontology = do_ontology,
                                                      do_skip_src_lang_processing=do_skip_src_lang_processing,
                                                      do_backtrans=do_backtrans,
                                                      do_augment=do_augment,
                                                      do_anonymization=do_anonymization,
                                                      augment_lang=augment_lang,
                                                      do_cleanup=do_cleanup,
                                                      do_regex = do_regex ,
                                                      do_marian_mt = do_marian_mt,
                                                      num_words_per_chunk=num_words_per_chunk,
                                                      ontology_weight=ontology_weight,
                                                      spacy_weight=spacy_weight,
                                                      hf_ner_weight=hf_ner_weight,
                                                      regex_weight=regex_weight,
                                                      backtrans_weight=backtrans_weight,
                                                      do_docs_trim_for_person=do_docs_trim_for_person,
                                                      do_docs_filter=do_docs_filter,
                                                      do_qg_rel=do_qg_rel,
                                                      do_kenlm = do_kenlm,
                                                      cutoff=cutoff,
                                                      batch_size=batch_size,
                                                       ),
                                              docs)
      i = 0
      for  docs in tqdm(processed_docs):
          i += 1
          for doc in processor.serialize_ner_items(docs):
            doc = json.dumps(doc)
            _file.write(f'{doc}\n')
      if _file is not None: _file.close()

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
    parser.add_argument('-do_backtrans', dest='do_backtrans', type=int, help='Wether or not to do back translation', default = 1)
    parser.add_argument('-do_augment', dest='do_augment', type=int, help='Wether or not to do translation augmentation', default = 0)
    parser.add_argument('-do_anonymization', dest='do_anonymization', type=int, help='Wether or not to anonymize the src_lang', default = 0)
    parser.add_argument('-do_regex', dest='do_regex', type=int, help='Wether or not to apply regex models', default = 1)
    parser.add_argument('-do_cleanup', dest='do_cleanup', type=int, help='Wether or not to cleanup NERs that are just stopwords or small number', default = 1)
    parser.add_argument('-do_marian_mt', dest='do_marian_mt', type=int, help='Wether or not to use marianMT for translation instead of M2M100', default = 1)
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
      args.target_lang = args.src_lang
      args.do_anonymization = False
      args.do_augmentation = False
    elif args.do_qg_rel_only:
      args.do_spacy = False
      args.do_hf_ner = False
      args.do_regex = False
      args.do_backtrans = False
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
      src_lang = []
    if not args.target_lang:
        target_lang =["en"]
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
