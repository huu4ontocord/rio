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
from edugp_kenlm_model import *
from huggingface_hub import hf_hub_url, cached_download
import sys
import argparse
from torch import multiprocessing
import time
from functools import partial

import sys
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

from faker import Faker
from faker.providers import person, company, geo, address, ssn, internet
import qg_pipeline

from mariam_mt import mariam_mt
try:
  import neuralcoref
except:
  neuralcoref = None
  pass

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

def _get_oscar_urls(language, shuffled="unshuffled", deduplicated="deduplicated"):
  _BASE_DATA_URL_FORMAT_STR = ("https://s3.amazonaws.com/datasets.huggingface.co/oscar/1.0/{shuffled}/{deduplicated}/{language}/")
  _BASE_CHECKSUM_FILE_NAME = "{language}_sha256.txt"
  base_data_url = _BASE_DATA_URL_FORMAT_STR.format(
            shuffled=shuffled, language=language, deduplicated=deduplicated
        )
  checksum_url = base_data_url + _BASE_CHECKSUM_FILE_NAME.format(language=language)
  with fsspec.open(checksum_url, encoding="utf-8") as f:
    data_filenames = [line.decode().split("\t")[0] for line in f if line]
    return [base_data_url + data_filename for data_filename in data_filenames]

def _download_urls(urls):
  for url in urls:
    if not os.path.exists(url.split("/")[-1]):
      os.system(f"wget {url}")



trannum = str.maketrans("0123456789", "1111111111")

#use english firstnames for now
bantu_surnames = ["Dlamini", "Gumede", "Hadebe", "Ilunga", "Kamau", "Khoza", "Lubega", "M'Bala", "Mabaso", "Mabika",
                  "Mabizela", "Mabunda", "Mabuza", "Macharia", "Madima", "Madondo", "Mahlangu", "Maidza", "Makhanya",
                  "Malewezi", "Mamba", "Mandanda", "Mandlate", "Mangwana", "Manjate", "Maponyane", "Mapunda", "Maraire",
                  "Masango", "Maseko", "Masemola", "Masengo", "Mashabane", "Masire", "Masondo", "Masuku", "Mataka",
                  "Matovu", "Mbala", "Mbatha", "Mbugua", "Mchunu", "Mkhize", "Mofokeng", "Mokonyane", "Mutombo",
                  "Ncube", "Ndagire", "Ndhlovu", "Ndikumana", "Ndiritu", "Ndlovu", "Ndzinisa", "Ngcobo", "Nkomo",
                  "Nkosi", "Nkurunziza", "Radebe", "Tshabalala", "Tshivhumbe", "Vila"]
#male and female
vietnamese_firstnames = ["Anh", "Dung", "Hanh", "Hoa", "Hong", "Khanh", "Lan", "Liem", "Nhung", "Duy", "Xuan"]
vietnamese_surnames = ["Nguyễn","Trần","Lê","Phạm","Hoàng","Huỳnh","Phan","Vũ","Võ", "Đặng","Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý"]
""
#use english firstnames for now
bengali_surnames  = ["Banerjee", "Bagchi", "Bhaduri", "Bhattacharjee", "Chakraborty", "Chatterjee", "Ganguly", "Goswami", "Ghoshal", "Lahiri", "Maitra", "Mukherjee", "Sanyal", "Chakraborty", "Bhattacharya", "Baidya", "Sengupta", "Dasgupta", "Duttagupta", "Gupta", "Sen-Sharma", "Basu", "Bose", "Dutta", "Ghosh", "Choudhury", "Guha", "Gain", "Mitra", "Singh", "Sinha", "Sen", "Pal", "De", "Dey", "Deb", "Dev", "Jana", "Palit", "Chanda", "Chandra", "Das", "Dam", "Kar", "Nandi", "Sarkar", "Nag", "Som"]

#male and female
urdu_firstnames = ["Azhar", "Benazir", "Farahnaz", "Maliha", "Minoo", "Romana", "Sania", "Azhar", "Burhan", "Changezi", "Faizan", "Fasih", "Fuad", "Hassim", "Jan", "Shoaib"]
urdu_surnames = ["Abid", "Ahmad", "Akbar", "Akmal", "Alam", "Ayaz", "Bohra", "Bucha", "Bukhari", "Buksh", "Bux", "Chandpuri", "Changezi", "Emani", "Farrukhabadi", "Farrukhi", "Fazail", "Hassim", "Hilaly", "Hussaini ", "Brahmin", "Lucknawi", "Ludhianvi", "Matloob", "Omar", "Vaishya", "Rahimtoola", "Shafiq", "Shoaib", "Siddiqui", "Siddiqui", "Tikka", "Yasser", ]

#basque and catalan - use Spanish names

available_gpus = [torch.cuda.device(i).idx for i in range(torch.cuda.device_count())]
available_global_models = [None]* len(available_gpus)

class TextAugmentGlobalModel:
  def __init__(self, device_id=None, device=None):
    if device_id is not None:
      self.device_id = int(device_id)
      self.device = "cuda:"+str(device_id)
    elif device is not None:
      self.device=device
      self.device_id = int(device.split(":")[-1])
    else:
      self.device = None
      self.device_id = None
    self.labse = None
    self.qg  = None
    self.translation_pipelines = None
    self.ner_model_name2pipelines = None
    self.mariam_mt = None

  @staticmethod
  def initializer_all(self, src_langs=["en"], target_langs=["en"], aug_langs=["en"]):
    global available_global_models
    for i, available_global_model in enumerate(available_global_models):
      if available_global_model is None:  
        available_global_model = TextAugmentGlobalModel(device_id=i)
      available_global_model.initializer(src_langs=src_langs, target_langs=target_langs, aug_langs=aug_langs)
      available_global_models[available_global_model.device_id] = available_global_model

  def initializer(self, device_id=None, device=None, src_langs=["en"], target_langs=["en"], aug_langs=["en"]):
    if device_id is not None:
      self.device_id = int(device_id)
      self.device = "cuda:"+str(device_id)
    elif device is not None:
      self.device=device
      self.device_id = int(device.split(":")[-1])
    else:
      self.device = self.device
      self.device_id = self.device_id
    if not hasattr(self, 'qg') or self.qg is None: self.qg = qg_pipeline.pipeline("multitask-qa-qg", device=self.device) # TODO make sure it's running in half mode
    if not hasattr(self, 'labse') or self.labse is None: self.labse =  SentenceTransformer("sentence-transformers/LaBSE", cache_folder="~/.cache").half().eval().to(self.device)
    if not hasattr(self, 'ner_model_name2pipelines') or self.ner_model_name2pipelines is None: self.ner_model_name2pipelines = {}
    if not hasattr(self, 'translation_pipelines') or self.translation_pipelines is None: 
      self.translation_pipelines  = {}
      self.translation_pipelines["facebook/m2m100_418M"] =  M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").eval().half().to(self.device)
      #TODO - do MariamMT model load, other m2m models
    for target_lang in list(set(target_langs + src_langs + aug_langs)):
      for model_name, model_cls, hf_ner_weight2 in TextAugment.hf_ner_model_map.get(target_lang, []):
          if model_name not in self.ner_model_name2pipelines:
            try:
              model = model_cls.from_pretrained(model_name).half().eval().to(self.device)
              tokenizer = AutoTokenizer.from_pretrained(model_name)
              ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=self.device_id)
              self.ner_model_name2pipelines[model_name] = ner_pipeline
            except:
              try:
                ner_pipeline = pipeline("ner",  model=model_name, tokenizer=(model_name, {"use_fast": True},), device=self.device_id)
                self.ner_model_name2pipelines[model_name] = ner_pipeline
              except:
                ner_pipeline = pipeline("ner",  model=model_name, tokenizer=(model_name, {"use_fast": True},), framework="tf",  device=self.device_id)
                self.ner_model_name2pipelines[model_name] = ner_pipeline

class TextAugment:

  m2m_model_name = ""
  m2m_tokenizer = None
  en_spacy_nlp = None
  faker_en_list  = None
  kenlm_model = None
  
  #from https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py which is under the MIT License
  # see also for ICD https://stackoverflow.com/questions/5590862/icd9-regex-pattern - but this could be totally wrong!
  # we do regex in this order in order to not capture ner inside domain names and email addresses.
  regex_rulebase = {
    "NORP": {
      "en": [(re.compile(r"upper class|middle class|working class|lower class", re.IGNORECASE), None),],
    },
    "AGE": {
      "en": [
          (
              re.compile(
                  r"\S+ years old|\S+\-years\-old|\S+ year old|\S+\-year\-old|born [ ][\d][\d]+[\\ /.][\d][\d][\\ /.][\d][\d]+|died [ ][\d][\d]+[\\ /.][\d][\d][\\ /.][\d][\d]+", re.IGNORECASE
              ),
              None,
          )
      ],
      "zh": [(regex.compile(r"\d{1,3}歲|岁"), None)],
    },
    # Some of this code from https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/anonymization.py which is under the Apache 2 license
    "ADDRESS": {
      "en": [
              #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
              (
                  re.compile(
                      r"\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$).*\b\d{5}(?:[-\s]\d{4})?\b|\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)", re.IGNORECASE
                  ),
                  None,
              ),
              (
                  re.compile(r"P\.? ?O\.? Box \d+"), None
              )
      ],

      "zh": [
          (
              regex.compile(
                  r"((\p{Han}{1,3}(自治区|省))?\p{Han}{1,4}((?<!集)市|县|州)\p{Han}{1,10}[路|街|道|巷](\d{1,3}[弄|街|巷])?\d{1,4}号)"
              ),
              None,
          ),
          (
              regex.compile(
                  r"(?<zipcode>(^\d{5}|^\d{3})?)(?<city>\D+[縣市])(?<district>\D+?(市區|鎮區|鎮市|[鄉鎮市區]))(?<others>.+)"
              ),
              None,
          ),
      ],
    },
    "DISEASE": {
      "en": [(re.compile("diabetes|cancer|HIV|AIDS|Alzheimer's|Alzheimer|heart disease", re.IGNORECASE), None)],
    },
    # many of the id_regex are from Presidio which is licensed under the MIT License
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/aba_routing_recognizer.py
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/au_abn_recognizer.py
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/us_passport_recognizer.py
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/medical_license_recognizer.py
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/es_nif_recognizer.py
    "ID": {
      "en": [
              (
                re.compile(r"\b[0123678]\d{3}-\d{4}-\d\b"),
                (
                    "aba",
                    "routing",
                    "abarouting",
                    "association",
                    "bankrouting",
                ),
              ),
              (
                  re.compile(r"(\b[0-9]{9}\b)"),
                  (
                      "us",
                      "united",
                      "states",
                      "passport",
                      "passport#",
                      "travel",
                      "document",
                  ),
              ),
              (
                  re.compile(r"[a-zA-Z]{2}\d{7}|[a-zA-Z]{1}9\d{7}"),
                  ("medical", "certificate", "DEA"),
              ),
              (re.compile(r"\d{3}\s\d{3}\s\d{3}"), None),
              (
                  re.compile(
                      r"GB\s?\d{6}\s?\w|GB\d{3}\s\d{3}\s\d{2}\s\d{3}|GBGD\d{3}|GBHA\d{3}}|GB\d{3} \d{4} \d{2}(?: \d{3})?|GB(?:GD|HA)\d{3}"
                  ),
                  None,
              ),
              (re.compile(r"IE\d[1-9]\d{5}\d[1-9]|IE\d{7}[1-9][1-9]?"), None),
              (re.compile(r"[1-9]\d{10}"), None),
              (
                  re.compile(
                      r"\d{2}-\d{7}-\d|\d{11}|\d{2}-\d{9}-\d|\d{4}-\d{4}-\d{4}|\d{4}-\d{7}-\d"
                  ),
                  None,
              ),
              (
                  re.compile(r"\b\d{2}\s\d{3}\s\d{3}\s\d{3}\b|\b\d{11}\b"),
                  ("australian business number", "abn"),
              ),
      ],
      "id":[
              (
                  re.compile(
                      r"\d{6}([04][1-9]|[1256][0-9]|[37][01])(0[1-9]|1[0-2])\d{6}"
                  ),
                  None,
              )
      ],
      "es": [
              (re.compile(r"(?:ES)?\d{6-8}-?[A-Z]"), None),
              (
                  re.compile(r"\b[0-9]?[0-9]{7}[-]?[A-Z]\b"),
                  ("documento nacional de identidad", "DNI", "NIF", "identificación"),
              ),
              (re.compile(r"[1-9]\d?\d{6}|8\d{8}|9\d{8}|10\d{8}|11\d{8}|12\d{8}|"), None)
      ],
      "pt": [(re.compile(r"\d{3}\.d{3}\.d{3}-\d{2}|\d{11}"), None),
             (re.compile(r"PT\d{9}"), None),
      ],
      "zh": [
          (
              regex.compile(
                  r"(?:[16][1-5]|2[1-3]|3[1-7]|4[1-6]|5[0-4])\d{4}(?:19|20)\d{2}(?:(?:0[469]|11)(?:0[1-9]|[12][0-9]|30)|(?:0[13578]|1[02])(?:0[1-9]|[12][0-9]|3[01])|02(?:0[1-9]|[12][0-9]))\d{3}[\dXx]"
              ),
              None,
          ),
          (
              regex.compile(
                  r"(^[EeKkGgDdSsPpHh]\d{8}$)|(^(([Ee][a-fA-F])|([DdSsPp][Ee])|([Kk][Jj])|([Mm][Aa])|(1[45]))\d{7}$)"
              ),
              None,
          ),
          (
              regex.compile(
                  r"((\d{4}(| )\d{4}(| )\d{4}$)|([a-zA-Z][1-2]{1}[0-9]{8})|([0-3]{1}\d{8}))"
              ),
              None,
          ),
          (
              regex.compile('^(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领 A-Z]{1}[A-HJ-NP-Z]{1}(?:(?:[0-9]{5}[DF])|(?:[DF](?:[A-HJ-NP-Z0-9])[0-9]{4})))|(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领 A-Z]{1}[A-Z]{1}[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9 挂学警港澳]{1})$'),
              None
          ),
          (
              regex.compile('\b[A-Z]{3}-\d{4}\b'),
              None,
          ),
          (
              regex.compile(
                  r"(0?\d{2,4}-[1-9]\d{6,7})|({\+86|086}-| ?1[3-9]\d{9} , ([\+0]?86)?[\-\s]?1[3-9]\d{9})"
              ),
              None,
          ),
          (
              regex.compile(
                  r"((\d{4}(| )\d{4}(| )\d{4}$)|([a-zA-Z][1-2]{1}[0-9]{8})|([0-3]{1}\d{8}))((02|03|037|04|049|05|06|07|08|089|082|0826|0836|886 2|886 3|886 37|886 4|886 49|886 5|886 6|886 7|886 8|886 89|886 82|886 826|886 836|886 9|886-2|886-3|886-37|886-4|886-49|886-5|886-6|886-7|886-8|886-89|886-82|886-826|886-836)(| |-)\d{4}(| |-)\d{4}$)|((09|886 9|886-9)(| |-)\d{2}(|-)\d{2}(|-)\d{1}(|-)\d{3})"
              ),
              None,
          ),
      ],
      "default": [
              #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py ssn
              (
                  re.compile(
                     '(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[- ](?!00)[0-9]{2}[- ](?!0000)[0-9]{4}'
                  ),
                  None,
              ),
              # https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py phone with exts
              (
                  re.compile('((?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?(?:[2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?(?:[0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(?:\d+)?))', re.IGNORECASE),
                  None
              ),
              # phone
              (
                  re.compile('''((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))'''),
                  None,
              ),
              #email
              (re.compile("([a-z0-9!#$%&'*+\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)", re.IGNORECASE), None),
              #credit card
              (re.compile('((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])'), None),
              #ip
              (re.compile('(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', re.IGNORECASE), None),
              #ipv6
              (re.compile('((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])'), None),
              #icd code - see https://stackoverflow.com/questions/5590862/icd9-regex-pattern
              (re.compile('[A-TV-Z][0-9][A-Z0-9](\.[A-Z0-9]{1,4})'), None),
              # generic government id. consider a more complicated string with \w+ at the beginning or end
              (re.compile(r"\d{8}|\d{9}|\d{10}|\d{11}"), None),
      ],
    },
  }


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
      #"zh": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0], ["zh_model", XLMRobertaForTokenClassification, 0.9 ]],
      "zh": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ]],
      'vi': [["lhkhiem28/COVID-19-Named-Entity-Recognition-for-Vietnamese", RobertaForTokenClassification, 1.0]],#["jplu/tf-xlm-r-ner-40-lang", None ],
      'hi': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]], #["jplu/tf-xlm-r-ner-40-lang", None, 1.0 ]],
      'ur': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]], #["jplu/tf-xlm-r-ner-40-lang", None, 1.0 ]],
      'id': [["cahya/bert-base-indonesian-NER", BertForTokenClassification, 1.0]],
      'bn': [["sagorsarker/mbert-bengali-ner", BertForTokenClassification, 1.0]],

      # NOT PART OF OUR LANGUAGE SET. EXPERIMENTAL
      'he': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]], #["jplu/tf-xlm-r-ner-40-lang", None, 1.0 ]],
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
  #TODO figure out actual numbers. Also, add languge specific models
  public_figure_kenlm_cutoff_map = {('en', 'en'): 450,
                                    ('yo', 'en'): 450,
                                    ('zu', 'en'): 450,
                                    ('sn', 'en'): 450,
                                    ('st', 'en'): 450,
                                    ('ny', 'en'): 450,
                                    ('xh', 'en'): 450,
                                    ('sw', 'en'): 450,
                                    ('ig', 'en'): 450,
                                    ('ar', 'en'): 450,
                                    ('en', 'en'): 450,
                                    ('es', 'en'): 450,
                                    ('eu', 'en'): 450,
                                    ('ca', 'en'): 450,
                                    ('pt', 'en'): 450,
                                    ('fr', 'en'): 450,
                                    ('zh', 'en'): 450,
                                    ('vi', 'en'): 450,
                                    ('hi', 'en'): 450,
                                    ('ur', 'en'): 450,
                                    ('id', 'en'): 450,
                                    ('bn', 'en'): 450,
                                    }
  m2m100_lang = {
    ('en', 'yo'): "Davlan/m2m100_418M-eng-yor-mt",
    ('yo', 'en'): "Davlan/m2m100_418M-yor-eng-mt",
    ('en', 'zu'): "masakhane/m2m100_418M-en-zu-mt",
    ('zu', 'en'): "masakhane/m2m100_418M-zu-en-mt",
    ('*', '*') : "facebook/m2m100_418M"
    }

  strip_chars = " ,،、{}[]|()\"'“”《》«»?!:;?。…．"
  punc_char = ".?!:;?。…．"
  special_char = " ,{}[]()|\\\"'“”《》«»~!@#$%^&*{}[]()_+=-0987654321`<>,、،./?':;“”\"\t\n\\πه☆●¦″．۩۱（☛₨➩°・■↑☻、๑º‹€σ٪’Ø·−♥ıॽ،٥《‘©。¨﴿！★×✱´٬→±x：¹？£―▷ф¡Г♫∟™ª₪®▬「—¯；¼❖․ø•�」٣，٢◦‑←§١ー٤）˚›٩▼٠«¢¸٨³½˜٭ˈ¿¬ι۞⌐¥►†ƒ∙²»¤…﴾⠀》′ا✓→¶'"
  junk = set(",{}[]()|\\\"'“”《》«»~!@#$%^&*{}[]()_+=-0987654321`<>,、،./?':;“”\"\t\n\\πه☆●¦″．۩۱（☛₨➩°・■↑☻、๑º‹€σ٪’Ø·−♥ıॽ،٥《‘©。¨﴿！★×✱´٬→±x：¹？£―▷ф¡Г♫∟™ª₪®▬「—¯；¼❖․ø•�」٣，٢◦‑←§١ー٤）˚›٩▼٠«¢¸٨³½˜٭ˈ¿¬ι۞⌐¥►†ƒ∙²»¤…﴾⠀》′ا✓→¶'")
  #don't add a space for junk chars
  ontology_manager = None
  max_stoword_len_zh = max([0]+[len(a) for a in stopwords.get('zh', [])])
  max_stoword_len_ko = max([0]+[len(a) for a in stopwords.get('ko', [])])
  max_stoword_len_ja = max([0]+[len(a) for a in stopwords.get('ja', [])])
  stopwords_en = set(stopwords.get('en',[]))


  def __init__(self, device=None, single_process=1, available_global_model=None, labse=None, ontology_manager=None, translation_pipelines=None, ner_model_name2pipelines=None, en_spacy_nlp=None, faker_en_list=None, qg=None, kenlm_model=None):

    if device is not None:
      self.device = device
      if device == "cpu": 
        self.device_id = -1
      else:
        device_id = int(device.split(":")[-1])
    else:
      if available_gpus:
        self.device_id = random.choice(available_gpus)
        self.device = "cuda:"+str(self.device_id) 
      else:
        self.device_id = -1
        self.device = "cpu"  
    if single_process:
      self.initializer(available_global_model=available_global_model, device=self.device, labse=labse, ontology_manager=ontology_manager, translation_pipelines=translation_pipelines, ner_model_name2pipelines=ner_model_name2pipelines, en_spacy_nlp=en_spacy_nlp, faker_en_list=faker_en_list, qg=qg, kenlm_model=kenlm_model)

  def initializer(self, all_available_global_model=None, available_global_model=None, device=None,  labse=None, ontology_manager=None, translation_pipelines=None, ner_model_name2pipelines=None, en_spacy_nlp=None, faker_en_list=None, qg=None, kenlm_model=None):
    global available_global_models
    if all_available_global_model is not None:
      available_global_models = all_available_global_model
    if device is not None:
      self.device = device
    device = self.device
    if available_global_model is not None:
      available_global_models[available_global_model.device_id] = available_global_model
      device_id = available_global_model.device_id
      self.device = device = available_global_model.device
      labse = available_global_model.labse
      qg = available_global_model.qg
      translation_pipelines = available_global_model.translation_pipelines
      ner_model_name2pipelines = available_global_model.ner_model_name2pipelines
    else:
      if device is None:
        if available_gpus:
          self.device_id = random.choice(available_gpus)
          device = self.device = "cuda:"+str(self.device_id) 
        else:
          self.device_id = -1
          device = self.device = "cpu"  
      print (device)
      if device != "cpu":
        device_id = int(device.split(":")[-1])
        available_global_model = available_global_models[device_id]
        if available_global_model is None: 
          available_global_models[device_id] = available_global_model = TextAugmentGlobalModel(device=self.device)
        labse = available_global_model.labse
        qg = available_global_model.qg
        translation_pipelines = available_global_model.translation_pipelines
        ner_model_name2pipelines = available_global_model.ner_model_name2pipelines

    if labse is not None: self.labse = labse 
    if translation_pipelines is not None: self.translation_pipelines = translation_pipelines
    if ner_model_name2pipelines is not None: self.ner_model_name2pipelines = ner_model_name2pipelines
    if qg is not None: self.qg = qg
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
        pass
    print (self.device)
    if not hasattr(self, 'qg') or self.qg is None: self.qg = qg_pipeline.pipeline("multitask-qa-qg", device=self.device) # TODO make sure it's running in half mode
    if not hasattr(self, 'labse') or self.labse is None: self.labse =  SentenceTransformer("sentence-transformers/LaBSE", cache_folder="~/.cache").half().eval().to(self.device)
    if not hasattr(self, 'ner_model_name2pipelines') or self.ner_model_name2pipelines is None: self.ner_model_name2pipelines = {}
    if not hasattr(self, 'translation_pipelines') or self.translation_pipelines is None: 
      self.translation_pipelines  = {}
      self.translation_pipelines["facebook/m2m100_418M"] =  M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").eval().half().to(self.device)
    #TODO MariamMT in global context
    if available_global_model is not None:
        available_global_model.labse = self.labse 
        available_global_model.qg = self.qg
        available_global_model.translation_pipelines = self.translation_pipelines 
        available_global_model.ner_model_name2pipeline = self.ner_model_name2pipelines
        available_global_models[available_global_model.device_id] = available_global_model 
        
    if TextAugment.ontology_manager is None: TextAugment.ontology_manager = None # OntologyManager(src_lang='en') #src_lang=src_lang
    if TextAugment.kenlm_model is None: 
      #TODO - save to temp dir
      os.system("mkdir -p ./wikipedia")
      if not os.path.exists("./wikipedia/en.arpa.bin"): 
        file_url= hf_hub_url(repo_id="edugp/kenlm", filename="wikipedia/en.arpa.bin")
        file = cached_download(file_url)
        os.system(f"ln -s {file} ./wikipedia/en.arpa.bin")
      if not os.path.exists("./wikipedia/en.sp.model"): 
        file_url= hf_hub_url(repo_id="edugp/kenlm", filename="wikipedia/en.sp.model")
        file = cached_download(file_url)
        os.system(f"ln -s {file} ./wikipedia/en.sp.model")
      if not os.path.exists("./wikipedia/en.sp.vocab"):
        file_url= hf_hub_url(repo_id="edugp/kenlm", filename="wikipedia/en.sp.vocab")
        file = cached_download(file_url)
        os.system(f"ln -s {file} ./wikipedia/en.sp.vocab")
      TextAugment.kenlm_model = KenlmModel("wikipedia", "en")
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

  def check_good_sentence(self, s, src_lang, stopwords, stopword_ratio_cutoff=0.06, bannedwords=None, flagged_words=None, badword_ratio_cutoff=0.15,  junk_ratio=0.16, max_badword_len=5):
    #basic dejunk
    # for flagged_words, only filter out if the ratio is exceeded AND there exists one banned word
    if bannedwords is None:
      bannedwords = self.banned_words.get(src_lang, self.banned_words['default'])
    default_bannedwords = self.banned_words['default']
    s = s.lower().strip()
    if not s: return False
    #print ([s2 for s2 in s if s2 in self.junk])
    #print (len([s2 for s2 in s if s2 in self.junk]), len(s))
    jr = len([s2 for s2 in s if s2 in self.junk])/len(s)
    if jr >= junk_ratio:
      return False
    if src_lang in ("ja", "ko", "zh"):
      sArr = s
    else:
      sArr = [s2.strip(self.special_char) for s2 in s.lower().split() if s2.strip(self.special_char)]
    if len(sArr) == 0:
      return False
    #stopword check
    if stopwords is not None:
      #TODO: catch multi word with spaces
      #print ('sw', len([s2 for s2 in sArr if s2 in stopwords])/len(sArr))
      if src_lang not in ("ja", "ko", "zh") and len([s2 for s2 in sArr if s2 in stopwords])/len(sArr) < stopword_ratio_cutoff:
        return False
      if src_lang in ("ja", "ko", "zh"):
        if src_lang == "zh":
          max_stoword = self.max_stoword_len_zh
        elif src_lang == "ko":
          max_stoword = self.max_stoword_len_ko
        elif src_lang == "ja":
          max_stoword = self.max_stoword_len_ja
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
    if flagged_words is not None:
      #print ('bw', len([s2 for s2 in sArr if s2 in flagged_words])/len(sArr))
      if src_lang not in ("ja", "ko", "zh") and len([s2 for s2 in sArr if s2 in flagged_words])/len(sArr) > badword_ratio_cutoff:
        if any(s2 for s2 in sArr if s2 in bannedwords) or any(s2 for s2 in sArr if s2 in default_bannedwords):
          return False
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
        if (bad_cnt/total_cnt) > badword_ratio_cutoff:
          for bword in bannedwords:
            if bword in s:
              return False
          for bword in default_bannedwords:
            if bword in s:
              return False
    #langid check
    try:
        lang =  langid.classify(s)[0]
    except:
        return True
    return lang == src_lang

  #WIP - we can use this q/a q/g method to extract people, place and thing, and potentially age/date AND to get a relationship between a person and a PII info
  def generate_questions_answers_rel(self, docs, chunks, src_lang, default_answers=[], text_key=None, ner_key=None, rel_key=None, signal='qg_rel', weight=1.0):
    answers = {}

    if ner_key is None:
      ner_key = f'{src_lang}_ner'
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
      text = text.replace("U.S.","US").replace("\n", " ").replace(",", " , ").replace("  ", " ").strip().replace(" , ", ", ") # replace(" He ", " Lincoln ").replace(" he ", " Lincoln ").replace(" him ", " Lincoln ")
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

  def do_translations(self, texts, src_lang='en', target_lang='hi', batch_size=16, do_mariam_mt=False):
    if not do_mariam_mt:
      m2m_model_name = self.m2m100_lang.get((src_lang, target_lang), self.m2m100_lang[('*', '*')])
      if m2m_model_name != self.m2m_model_name or self.m2m_tokenizer is None:
        self.m2m_tokenizer = M2M100Tokenizer.from_pretrained(m2m_model_name)
      try:
        self.m2m_tokenizer.src_lang = src_lang
        target_lang_bos_token = self.m2m_tokenizer.get_lang_id(target_lang)
      except:
        do_mariam_mt = True
        pass
      if not do_mariam_mt:
        if m2m_model_name != self.m2m_model_name or self.m2m_model is None:
          if m2m_model_name in  self.translation_pipelines:
            self.m2m_model =  self.translation_pipelines[m2m_model_name]
          else:
            self.translation_pipelines[m2m_model_name] = self.m2m_model = M2M100ForConditionalGeneration.from_pretrained(m2m_model_name).eval().half().to(self.device)
        self.m2m_model_name = m2m_model_name
        translations = []
        for src_text_list in self.batch(texts, batch_size):
          try:
            batch = self.m2m_tokenizer(src_text_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
          except:
            do_mariam_mt = True
            break

          gen = self.m2m_model.generate(**batch, forced_bos_token_id=target_lang_bos_token, no_repeat_ngram_size=4, ) #
          outputs = self.m2m_tokenizer.batch_decode(gen, skip_special_tokens=True)
          translations.extend(outputs)
        if not do_mariam_mt:
          return translations

    translations = []
    #mariam_mt = self.mariam_mt
    model_name = mariam_mt.get((src_lang, target_lang))
    mt_pipeline = None
    if model_name is not None and model_name not in self.translation_pipelines:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).half().eval().to(self.device)
        if self.device == 'cpu':
          mt_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)
        else:
          mt_pipeline = pipeline("translation", model=model, tokenizer=tokenizer, device=self.device_id)
        self.translation_pipelines[model_name] = mt_pipeline
    if not mt_pipeline:
        raise RuntimeError("no translation pipeline") # we could do multi-step translation where there are no pairs
    mt_pipeline = self.translation_pipelines[model_name]
    for src_text_list in self.batch(texts, batch_size):
        outputs = [t['translation_text'] for t in mt_pipeline(src_text_list, batch_size=batch_size)]
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
    if ner_key is None:
      ner_key = f'{src_lang}_ner'
    if text_key is None:
      text_key = f'{src_lang}_text'
    for doc in docs.values():
      ner = doc[ner_key] = doc.get(ner_key, {})
      sentence = doc[text_key]
      if ner is None: ner = {}
      if src_lang in ("zh", "ko", "ja"):
          sentence_set = set(sentence.lower())
      else:
          sentence_set = set(sentence.lower().split(" "))
      idx = 0
      all_ner = []
      original_sentence = sentence
      for tag, regex_group in self.regex_rulebase.items():
          for regex_context in regex_group.get(src_lang, []) + regex_group.get("default", []):
              if True:
                  regex, context = regex_context
                  found_context = False
                  if context:
                      for c1 in context:
                        c1 = c1.lower()
                        for c2 in c1.split():
                          if c2 in sentence_set:
                              found_context = True
                              break
                        if found_context: break
                      if not found_context:
                          continue
                  for ent in regex.findall(sentence):
                      if not isinstance(ent, str) or not ent:
                          continue
                      sentence2 = original_sentence
                      delta = 0
                      while True:
                        if ent not in sentence2:
                          break
                        else:
                          i = sentence2.index(ent)
                          j = i + len(ent)
                          if found_context:
                              len_sentence = len(sentence2)
                              left = sentence2[max(0, i - context_window) : i].lower()
                              right = sentence2[j : min(len_sentence, j + context_window)].lower()
                              found_context = False
                              for c in context:
                                c = c.lower()
                                if c in left or c in right:
                                    found_context = True
                                    break
                              if not found_context:
                                sentence2 = sentence2[i+len(ent):]
                                continue
                              #ner[ent.strip()] = tag
                          sentence2 = sentence2[i+len(ent):]
                          all_ner.append((ent, delta+i, delta+j, tag))
                          delta += j
                          idx += 1
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
    if stopwords is None:
      stopwords = set(stopwords.get(src_lang, []))
    if offset_key is None:
      offset_key = f'{src_lang}_offset'
    if ner_key is None:
      ner_key = f'{src_lang}_ner'
    if text_key is None:
      text_key = f'{src_lang}_text'
    print (chunks)
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
        while text[start] in self.strip_chars:
          start += 1
          if start >= end: break
        end = start + len(text[start:end].strip(self.strip_chars))
        ner_result['word'] = text[start:end]
        ner_result['start'] = start+offset
        ner_result['end'] = end+offset
        if results2 and results2[-1]['end'] > ner_result['start']:
          continue
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
            print ('offset mismatch', text[start:end], ner_result['word'])
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
    if not nlp:
      return
    if stopwords is None:
      stopwords = set(stopwords.get(src_lang, []))
    offset_key=f'{src_lang}_offset'
    if ner_key is None:
      ner_key = f'{src_lang}_ner'
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
            if mention not in ref2mention[coref]:
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
          cluster_text_list = set([m.text.lower() for m in cl.mentions])
          # I don't use "us" because that is sometimes the U.S.
          # TODO: fix to check for upper case only US as an exception
          if "you" in cluster_text_list or "your"  in cluster_text_list  or "yours"  in cluster_text_list  or  "we" in cluster_text_list  or 'i' in cluster_text_list  or 'my' in cluster_text_list  or 'mine' in cluster_text_list or 'me' in cluster_text_list or 'he' in cluster_text_list or "she" in cluster_text_list or "his" in cluster_text_list or "her" in cluster_text_list or "him" in cluster_text_list or "hers" in cluster_text_list:
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
          if label in ('WORK_OF_ART', 'NOUN', None): continue
          if label in ('GPE', 'FAC'): label = 'LOC'

          ner_word = mention[0]
          if ner_word and ner_word.lower() not in stopwords:
              if not self.cjk_detect(ner_word):
                if ner_word not in text: continue
                i += text[i:].index(ner_word)
                ner_word = text[i:].split(" ", 1)[0]
              ner_word = ner_word.strip(self.strip_chars)
              if ner_word.lower() not in stopwords:
                mention2 = (ner_word, i, i+len(ner_word))
                aHash = ner.get(mention2, {})
                aHash[(label, signal)] = aHash.get((label, signal), 0) + spacy_weight * (1.0 + len(ner_word)/100) * extra_weight
                ner[mention2] = aHash
                if label in ('PERSON', 'PUBLIC_FIGURE'):
                  coref = list(set([a[0] for a in ref2mention.get(mention2ref.get(mention), [])]))
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
      if neuralcoref is not None:
        return self.spacy_ner_coref(docs, nlp, stopwords, spacy_weight, src_lang, extra_weight=extra_weight, text_key=text_key, ner_key=ner_key, signal=signal)
      else:
        if not nlp:
          return
        if stopwords is None:
          stopwords = set(stopwords.get(src_lang, []))
        offset_key=f'{src_lang}_offset'
        if ner_key is None:
          ner_key = f'{src_lang}_ner'
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
          #do hypo collapse so that loc->address, person->public_figure when there are overlaps, date/id->id
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
            labelsHash2['ID'] = labelsHash2['ID'] + labelsHash2['DATE']
            del labelsHash2['DATE']
          if 'CARDINAL' in labelsHash2 and 'ID' in labelsHash2:
            labelsHash2['ID'] = labelsHash2['ID'] + labelsHash2['CARDINAL']
            del labelsHash2['CARDINAL']

          ner[mention] = labelsHash2
      doc[collapse_ner_key] = ner


    return docs

  def create_augment_anon_context(self, docs, chunks, src_lang, faker_target_lang, faker_en, aug_scope={'ID', 'PERSON'}, target_lang=None, \
                                 items_key=None, context_key=None, ner_key=None):
        if target_lang is None: target_lang = src_lang
        if ner_key is None: ner_key = f'{src_lang}_ner'
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
        if target_lang is None: target_lang = src_lang
        if ner_key is None: ner_key = f'{src_lang}_collapse_ner'
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
    #TODO: create a {src_lang}_ner_anon field.
    #TODO: another  way to do anonymimzation is to pass the anonymized text through backtrans. TBD?
    if target_lang is None: target_lang = src_lang

    self.create_augment_anon_context(docs, chunks, src_lang, faker_src_lang, faker_en, aug_scope=anon_scope, target_lang=src_lang, \
                                          items_key=f'{src_lang}_items', context_key=f'{src_lang}_anon_context', ner_key=f'{src_lang}_collapse_ner')

    docs, chunks = self.replace_items_in_chunks(docs, chunks, src_lang, replace_with_bracket=False, do_augment=True, \
                                                                         context_key=f'{src_lang}_non_context', \
                                                                         ner_key=f'{src_lang}_ner', items_key=f'{src_lang}_items', \
                                                                         text_key=f'{src_lang}_text', replace_text_key=f'{src_lang}_text_anon', \
                                                                         offset_key=f'{src_lang}_offset')
    for doc in docs.values():
      doc[f'{src_lang}_text_anon']  = " ".join([chunk[f'{src_lang}_text_anon'] for chunk in doc['chunks']])

    return docs, chunks

  #TODO: we also need a deserialize
  @staticmethod
  def serialize_ner_items(docs, ner_keys=None, outfile=""):
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
                    ner_value = [{'label': label, "score": score} for label, score in ner_value.items()]
                    ner_dict = {'text': text, 'start': start, 'end': end, 'ner_values': ner_value}
                    serialize_items.append(ner_dict)
                doc[ner_key] = serialize_items
        if outfile:       
          with open(outfile, 'w', encoding='utf-8') as file:
            for doc in serialize_docs:
              file.write(f'{doc}\n')
        return serialize_docs

  @staticmethod
  def deserialize_ner_items(docs=None, infile="", return_dict=False):
    def load_py_from_str(s, default=None):
      if not s.strip(): return default
      ret = {'__ret': None}
      #print (s)
      exec("__ret= "+s, ret)
      return ret['__ret']
      
    def deserialize_doc(doc):
       for ner_key in [key for key in doc if key.endswith('_ner')]:
                ner_items = doc[ner_key]
                if ner_items and type(ner_items) is list and 'ner_values' in ner_items[0]:
                  deserialize_items = {}
                  for item in ner_items:
                    mention = (item['text'], item['start'], item['end'])
                    aHash = dict([(tuple(a['label']) if type(a['label']) is list else a['label'], float(a['score'])) for a in item['ner_values']])
                    deserialize_items[mention] = aHash
                  doc[ner_key] = deserialize_items

    if infile:
      docs= [load_py_from_str(s, {}) for s in open(infile, "rb").read().decode().split("\n")]
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
                          batch_size = 5,
                          batch_window=70,
                          ontology_weight=0.85,
                          spacy_weight=1.00,
                          hf_ner_weight=1.25,
                          regex_weight=1.5,
                          backtrans_weight=0.9,
                          do_docs_trim=False,
                          do_kenlm = True,
                          do_qa_qg_rel=False,
                          aug_scope={'ADDRESS', 'ORG', 'PERSON', 'LOC', 'ID'}, #TODO, public figure, age, norp and disease
                          anon_scope={'PERSON', 'ID'},):
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

    stopwords1 = set(stopwords[src_lang])
    stopwords2 = set(stopwords[target_lang])

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

    # init hf ner pipelines
    if do_hf_ner:
      for model_name, model_cls, hf_ner_weight2 in self.hf_ner_model_map.get(target_lang, []):
        if model_name not in self.ner_model_name2pipelines:
          #print ("setting")
          try:
            model = model_cls.from_pretrained(model_name).half().eval().to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.device == 'cpu':
              ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
            else:
              ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=self.device_id)
            self.ner_model_name2pipelines[model_name] = ner_pipeline
          except:
            try:
              if self.device == 'cpu':
                ner_pipeline = pipeline("ner",  model=model_name, tokenizer=(model_name, {"use_fast": True},))
              else:
                ner_pipeline = pipeline("ner",  model=model_name, tokenizer=(model_name, {"use_fast": True},), device=self.device_id)
              self.ner_model_name2pipelines[model_name] = ner_pipeline
            except:
              if self.device == 'cpu':
                ner_pipeline = pipeline("ner",  model=model_name, tokenizer=(model_name, {"use_fast": True},), framework="tf")
              else:
                ner_pipeline = pipeline("ner",  model=model_name, tokenizer=(model_name, {"use_fast": True},), framework="tf", device=self.device_id)
              self.ner_model_name2pipelines[model_name] = ner_pipeline
        ner_pipelines.append((model_name, self.ner_model_name2pipelines[model_name], hf_ner_weight2))
    target_is_cjk = target_lang in ('zh', 'ko', 'ja')
    src_is_cjk = src_lang in ('zh', 'ko', 'ja')
    lbracket = "[["
    rbracket = "]]"

    if do_augment:
      target_text_key = f'{target_lang}_text_aug'
      target_ner_key =  f'{target_lang}_ner_aug'
      target_collapse_ner_key =  f'{target_lang}_collapse_ner_aug'
      target_offset_key = f'{target_lang}_offset_aug'
      target_src_sim_key = f'{src_lang}_2_{target_lang}_sim_aug'
    else:
      target_text_key = f'{target_lang}_text'
      target_ner_key = f'{target_lang}_ner'
      target_collapse_ner_key = f'{target_lang}_collapse_ner'
      target_offset_key = f'{target_lang}_offset'
      target_src_sim_key = f'{src_lang}_2_{target_lang}_sim'

    public_figure_kenlm_cutoff = self.public_figure_kenlm_cutoff_map.get((src_lang, target_lang), 1500)

    docs = self.collapse_ner(docs, ner_key = f'{src_lang}_ner', collapse_ner_key = f'{src_lang}_collapse_ner',  text_key = f'{src_lang}_text', stopwords=stopwords1)

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
                                          items_key=f'{src_lang}_items', context_key=f'{src_lang}_aug_context', ner_key=f'{src_lang}_collapse_ner')
        docs, chunks = self.replace_items_in_chunks(docs, chunks,  src_lang, lbracket=lbracket, rbracket=rbracket, \
                                                                         replace_with_bracket=True, do_augment=do_augment, \
                                                                         context_key=f'{src_lang}_aug_context', \
                                                                         ner_key=f'{src_lang}_collapse_ner',  items_key=f'{src_lang}_items', \
                                                                         text_key=f'{src_lang}_text', replace_text_key=f'{src_lang}_tmp_text', \
                                                                         offset_key=f'{src_lang}_offset')
        chunks2 = [chunk[f'{src_lang}_tmp_text'] for chunk in chunks]
        text2 = self.do_translations(chunks2, src_lang=src_lang, target_lang=target_lang, batch_size=batch_size)
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
                  if mention in doc[f'{src_lang}_ner']:
                    aHash = doc[f'{src_lang}_ner'][mention]
                    for key2 in list(aHash.keys()):
                      aHash[key2] /= 2.0
                else:
                  #vals = list(doc[f'{src_lang}_ner'][mention].keys())
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

    if False: # do_ontology and self.ontology_manager is not None:
        # ontology - context independent - there are some bugs in disease detection which needs to be fixed so we will skip for now
        for doc in docs.values():
          doc[target_ner_key] = ner = doc.get(target_ner_key, {})
          if target_lang == 'en':
            chunk2ner = self.ontology_manager.tokenize(doc[target_text_key])['chunk2ner']
            onto_items = []
            for c, label in chunk2ner.items():
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

    if False: #do_qa_qg_rel and target_lang == 'en':
      docs, chunks= self.generate_questions_answers_rel(docs, chunks, target_lang, ner_key=target_ner_key)

    docs = self.collapse_ner(docs, target_ner_key, target_collapse_ner_key, target_text_key, stopwords2, do_cleanup_only=True)

    if do_docs_trim:
      docs, chunks = self.trim_to_prefer_person(docs, chunks)

    if do_kenlm and target_lang == 'en':
      for doc in docs.values():
        ner = doc[target_ner_key]
        persons = []
        for ent, aHash in ner.items():
          if any(key[0] == 'PERSON' for key in aHash.keys()):
            persons.append(ent[0])
        public_figures = []
        for ent in list(set(persons)):
          if self.kenlm_model.get_perplexity(f"{ent} (born") <= public_figure_kenlm_cutoff:
            public_figures.append(ent)
        public_figures = set(public_figures)
        for ent, aHash in ner.items():
          if ent in public_figures:
            aHash[('PUBLIC_FIGURE', 'kenlm')] = aHash.get(('PUBLIC_FIGURE', 'kenlm'), 0) + 1.0

    # this will mess up the items array and the other arrays that depends on mentions unles we do_cleanup_only
    docs = self.collapse_ner(docs, target_ner_key, target_collapse_ner_key, target_text_key, stopwords2, do_cleanup_only=True)

    if target_lang != src_lang and not do_augment:
          for doc in docs.values():
            ner =  doc[f'{target_lang}_ner']
            src_ner = doc[f'{src_lang}_ner']
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
        doc[f'{target_lang}_2_{src_lang}_tmp'] = doc.get(f'{target_lang}_2_{src_lang}_tmp', {})
        aHash = doc[f'{target_lang}_2_{src_lang}_tmp']
        items = doc[f'{src_lang}_items']
        doc[f'{target_lang}_2_{src_lang}_context'] = dict([(a, items[b][0]) for a, b in aHash.items()])
        docs, chunks = self.replace_items_in_chunks(docs, chunks,  src_lang, lbracket=lbracket, rbracket=rbracket, \
                                                                         replace_with_bracket=True, do_augment=True, \
                                                                         ner_key=f'{target_lang}_ner', items_key=f'{target_lang}_items', \
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

        all_embed = self.labse.encode(backtrans_text, convert_to_tensor=True)
        all_trans_embed = self.labse.encode([chunk[f'{src_lang}_text'] for chunk in chunks], convert_to_tensor=True)
        similarity = cosine_similarity(all_embed, all_trans_embed, dim=1)
        for chunk, trans_text, sim_score in zip(chunks, backtrans_text, similarity):
          _id = chunk['id']
          doc = docs[_id]
          offset = chunk[f'{src_lang}_offset']
          orig_text = chunk[f'{src_lang}_text']
          trans_text = chunk[f'{src_lang}_text_backtrans_from_{target_lang}']
          items = doc[f'{target_lang}_items']
          len_items = len(items)
          ner = doc[f'{target_lang}_ner']
          bner = doc[f'{src_lang}_2_{target_lang}_backtrans_ner_tmp'] = doc.get(f'{src_lang}_2_{target_lang}_backtrans_ner_tmp', {})
          pos = 0
          sim_score = sim_score.item()
          #print ('backtrans match', sim_score, orig_text, '**', trans_text)
          if sim_score < 0.70:
            chunk[f'{src_lang}_text_backtrans_from_{target_lang}'] = " . . . "
            #TODO - save away sim_score ?
            continue
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
              print ('err', idx, t, o)
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
                      ner_word = ner_word.strip(self.strip_chars+".")
                      #print (ner_word, ner_word in orig_text, orig_text)
                      if ner_word not in orig_text[pos:]:
                        print ('cant find in orig_text ', ner_word, '**', orig_text[pos:], '**', orig_text)
                      else:
                        i = orig_text[pos:].index(ner_word)
                        start = pos + i 
                        len_nerword = len(ner_word)
                        pos = start + len_nerword
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
          ner = doc[f'{src_lang}_ner']
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
            ner =  doc[f'{target_lang}_ner']
            src_ner = doc[f'{src_lang}_ner']
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
      docs = self.collapse_ner(docs, ner_key = f'{src_lang}_ner', collapse_ner_key = f'{src_lang}_collapse_ner',  text_key = f'{src_lang}_text', stopwords=stopwords1)

    if do_anonymization and faker_src_lang is not None and faker_en is not None:
      docs, chunks = self.anonymize(docs, chunks, src_lang, faker_src_lang, faker_en, anon_scope=anon_scope)

    #TODO: remove all _tmp fields

    return docs, chunks

  def process_ner(self,
              docs,
              src_lang,
              do_spacy = True,
              do_hf_ner = True,
              do_ontology = True,
              do_skip_src_lang_processing=False,
              do_backtrans=False,
              do_augment=False,
              do_anonymization=False,
              copy_anon_to_text=True, # if we do_anonymize, we will copy {src_lang}_text_anon -> text
              augment_lang="es",
              do_cleanup=True,
              do_regex = True,
              batch_size = 5,
              batch_window=70,
              ontology_weight=0.85,
              spacy_weight=1.00,
              hf_ner_weight=1.25,
              regex_weight=1.5,
              backtrans_weight=0.9,
              do_docs_trim=True,
              do_qa_qg_rel=False,
              do_kenlm = True,
              cutoff=None,
              target_lang='en',
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
      src_is_cjk = src_lang in ('zh', 'ko', 'ja')
      if src_is_cjk:
        sep = ""
      else:
        sep = " "
      if type(docs) is dict:
        docs = list(docs.values())      
      #for testing only
      if cutoff is not None and len(docs) > cutoff:
        docs = docs[:cutoff]
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
      if do_docs_trim:
        docs = [doc for doc in docs if self.check_good_sentence(doc[f'{src_lang}_text'], src_lang, stopwords=stopwords1, flagged_words=flagged_words1)]
        #print ('trimmed junk', (len_docs-len(docs))/len_docs)
      len_d = len(docs)
      
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
        while len_text > batch_window:
            for j in range(batch_window-1, len_text):
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
      if do_docs_trim:
        docs2, chunks2 = self.trim_to_prefer_person(docs, chunks)
        do_docs_trim = len(docs2) == len(docs)
        docs, chunks = docs2, chunks2

      # we do this here because we don't want to trim  ner items that are considered empty.
      # we should probably fix trim_to_prefer_person to not do any trimming if all ner's are empty
      for doc in docs.values():
        doc[f'{src_lang}_ner'] = doc.get(f'{src_lang}_ner', {})

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
                          do_regex = do_regex,
                          do_cleanup=do_cleanup,
                          batch_size = batch_size,
                          ontology_weight=ontology_weight,
                          spacy_weight=spacy_weight,
                          hf_ner_weight=hf_ner_weight,
                          regex_weight=regex_weight,
                          backtrans_weight=backtrans_weight,
                          do_qa_qg_rel=do_qa_qg_rel and src_lang == 'en',
                          do_docs_trim=do_docs_trim)
        if do_docs_trim:
          do_docs_trim = len(docs2) == len(docs)
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
                            do_qa_qg_rel=do_qa_qg_rel and target_lang == 'en',
                            do_kenlm = do_kenlm,
                            batch_size = batch_size,
                            ontology_weight=ontology_weight,
                            spacy_weight=spacy_weight,
                            hf_ner_weight=hf_ner_weight,
                            regex_weight=regex_weight,
                            backtrans_weight=backtrans_weight,
                            do_docs_trim=do_docs_trim)
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
                            do_anonymization=False,
                            do_regex = do_regex,
                            do_cleanup=do_cleanup,
                            do_qa_qg_rel=do_qa_qg_rel and augment_lang == 'en',
                            do_kenlm = do_kenlm,
                            batch_size = batch_size,
                            ontology_weight=ontology_weight,
                            spacy_weight=spacy_weight,
                            hf_ner_weight=hf_ner_weight,
                            regex_weight=regex_weight,
                            backtrans_weight=backtrans_weight,
                            do_docs_trim=do_docs_trim)
        docs, chunks = docs2, chunks2

      return docs

  #given a dataset, file or any out of core random access text/json data source (by lines), we choose several shards of the data
  #and we begin to send in chunks of it at a time to the process.
  @staticmethod
  def _multiprocess_ner_helper(docs_chunk):
      print('docs_chunk length ', len(docs_chunk))
      return processor.process_ner(docs=docs_chunk, src_lang='vi', target_lang='en', do_regex=True,
                                   do_spacy=True, do_backtrans=True, cutoff=None, batch_size=5)


  @staticmethod
  def intialize_docs(docs=None, src_lang=None):
      print("Intialize Documents")
      domain = ""
      if docs is None:
        try:
          domain = 'oscar_registry'
          d = load_dataset("TurkuNLP/register_oscar", data_files=f"{src_lang}/{src_lang}_00000*")
          docs = [doc for doc in d['train'] if 'labels' not in doc or doc['labels'] !=[]]
        except:
          try:
            domain = 'mc4_registry'
            d = load_dataset("TurkuNLP/register_mc4", data_files=f"{src_lang}/{src_lang}_00000*")
            docs = [doc for doc in d['train'] if 'labels' not in doc or doc['labels'] !=[]]
          except:
            domain = 'oscar'
            url = _get_oscar_urls(src_lang)[0]
            _download_urls([url])
            docs = [{'text':text}  for text in [try_decode(t) for t in gzip.open(url.split("/")[-1], "rb").readlines()] if text]
            try:
              domain = 'mc4'
              d = load_dataset("mc4", src_lang)
              d = list(d['train'])
            except:
              pass
      elif isinstance(docs, str):
          docs = [{'text': docs}]
      elif isinstance(docs, list):
          if isinstance(docs[0], dict):
            return docs
          else:
            return [{'text': t} for t in docs]
      print(f"Finished intialize Documents from {domain}")
      return docs

  @staticmethod
  def preload_cache(src_lang, target_lang):
    AutoConfig.from_pretrained("sentence-transformers/LaBSE")
    AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    AutoModel.from_pretrained("sentence-transformers/LaBSE")
    SentenceTransformer("sentence-transformers/LaBSE", cache_folder="~/.cache")
    en_spacy_nlp = spacy.load('en_core_web_sm')
    try:
      coref = neuralcoref.NeuralCoref(en_spacy_nlp.vocab)
    except:
      pass
    if src_lang:
      try:
        load_dataset("TurkuNLP/register_oscar", data_files=f"{src_lang}/{src_lang}_00000*")
      except:
          try:
            load_dataset("TurkuNLP/register_mc4", data_files=f"{src_lang}/{src_lang}_00000*")
          except:
            url = _get_oscar_urls(src_lang)[0]
            _download_urls([url])
            try:
              load_dataset("mc4", src_lang)
            except:
              pass
    arr2 = []
    for arr in TextAugment.hf_ner_model_map.values():
      for model_name, _, _ in arr:
        arr2.append(model_name)
    for model_name in list(set(arr2)):
        AutoModel.from_pretrained(model_name)
        AutoTokenizer.from_pretrained(model_name)
        AutoConfig.from_pretrained(model_name)
    for model_name in TextAugment.m2m100_lang.values():
        AutoModel.from_pretrained(model_name)
        AutoTokenizer.from_pretrained(model_name)
        AutoConfig.from_pretrained(model_name)
    for aHash in qg_pipeline.SUPPORTED_TASKS.values():
      for model_name in aHash["default"].values():
        AutoModel.from_pretrained(model_name)
        AutoTokenizer.from_pretrained(model_name)
        AutoConfig.from_pretrained(model_name)

    #TODO - get temp dir and move this into the temp dir
    os.system("mkdir -p ./wikipedia")
    if not os.path.exists("./wikipedia/en.arpa.bin"): 
      file_url= hf_hub_url(repo_id="edugp/kenlm", filename="wikipedia/en.arpa.bin")
      file = cached_download(file_url)
      os.system(f"ln -s {file} ./wikipedia/en.arpa.bin")
    if not os.path.exists("./wikipedia/en.sp.model"): 
      file_url= hf_hub_url(repo_id="edugp/kenlm", filename="wikipedia/en.sp.model")
      file = cached_download(file_url)
      os.system(f"ln -s {file} ./wikipedia/en.sp.model")
    if not os.path.exists("./wikipedia/en.sp.vocab"):
      file_url= hf_hub_url(repo_id="edugp/kenlm", filename="wikipedia/en.sp.vocab")
      file = cached_download(file_url)
      os.system(f"ln -s {file} ./wikipedia/en.sp.vocab")
    #kenlm_model = KenlmModel("./wikipedia", "en")
  

  @staticmethod
  def multiprocess_ner(docs,
                    outfile,
                    src_lang=None,
                    target_lang=None,
                    do_regex=True,
                    do_spacy=True,
                    do_backtrans=True,
                    cutoff=None,
                    batch_size=5,
                    num_workers=2):
    multiprocessing.set_start_method('spawn', force=True)
    if target_lang is None:
      target_lang = "en"
    #TODO create a generator function to return docs_chunks
    if num_workers > 1:
        chunk_size = int(len(docs) / num_workers)
        docs_chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]
    else:
      docs_chunks = [docs]
    start = time.time()
    TextAugmentGlobalModel.initializer_all(src_langs=[src_lang], target_langs=[target_lang])
    processor = TextAugment(single_process=False)
    # processor.initializer()
    print(len(docs_chunks))
    with open(outfile, 'w', encoding='utf-8') as file:
        # for i in range(0, num_workers):
          pool = multiprocessing.Pool(processes=num_workers, initializer=partial(processor.initializer, all_available_global_model=available_global_models))
          processed_docs = pool.imap_unordered(partial(processor.process_ner,
                                                      src_lang=src_lang,
                                                      target_lang=target_lang,
                                                      do_regex=do_regex,
                                                      do_spacy=do_spacy,
                                                      do_backtrans=do_backtrans,
                                                      cutoff=cutoff,
                                                      batch_size=batch_size,
                                                       ),
                                              docs_chunks)
          i = 0
          for  docs in tqdm(processed_docs):
            #print(f"processed {i}: (Time elapsed: {(int(time.time() - start))}s)")
            i += 1
            for doc in processor.serialize_ner_items(docs):
            # for doc in docs.values():
              file.write(f'{doc}\n')

in_notebook = 'google.colab' in sys.modules
if not in_notebook:
  try:
      get_ipython()
  except:
    in_notebook = False

if not in_notebook:
  parser = argparse.ArgumentParser(description='Text Annotation, Augmentation and Anonymization')
  parser.add_argument('-src_lang', dest='src_lang', type=str, help='Source Language', default=None)
  parser.add_argument('-target_lang', dest='target_lang', type=str, help='Target Language', default="en")
  parser.add_argument('-cutoff', dest='cutoff', type=int, help='Cutoff documents, -1 is none', default=30)
  parser.add_argument('-batch_size', dest='batch_size', type=int, help='batch size', default=5)
  parser.add_argument('-infile', dest='infile', type=str, help='file to load', default=None)
  parser.add_argument('-outfile', dest='outfile', type=str, help='file to save', default=None)
  parser.add_argument('-num_workers', dest='num_workers', type=int, help='Num of Workers', default = 1)
  parser.add_argument('-preload_cache', dest='preload_cache', action='store_true', help='Preload the cache of models and data', default = False)
  args = parser.parse_args()
else:
  args = None


if __name__ == "__main__":
  if args is not None:
    src_lang = args.src_lang
    target_lang = args.target_lang
    cutoff = args.cutoff
    batch_size = args.batch_size
    infile = args.infile
    outfile = args.outfile
    num_workers = args.num_workers
    if cutoff <= 0:
      cutoff = None
    if outfile is None:
      if src_lang is not None:
        outfile = f"{src_lang}_out.jsonl"
      else:
        outfile = "out.jsonl"
    docs = TextAugment.deserialize_ner_items(infile=infile) if infile else None
    if args.preload_cache: 
      TextAugment.preload_cache(src_lang, target_lang)
    #TODO - do multiprocessing
    elif src_lang is not None:
      if num_workers > 1:
        if not docs:
          docs = TextAugment.intialize_docs(docs, src_lang)
        TextAugment.multiprocess_ner(docs,
                    outfile,
                    src_lang=src_lang,
                    target_lang=target_lang,
                    do_regex=True,
                    do_spacy=True,
                    do_backtrans=True,
                    cutoff=cutoff,
                    batch_size=batch_size,
                    num_workers=num_workers)
      else:
        processor = TextAugment(single_process=True)
        if not docs:
          docs = processor.intialize_docs(docs, src_lang)
        docs = processor.process_ner(docs=docs, src_lang=src_lang, target_lang=target_lang, do_regex=True, do_spacy=True,
                                          do_backtrans=True, cutoff=cutoff, batch_size=batch_size)
        docs = processor.serialize_ner_items(docs, outfile=outfile)
