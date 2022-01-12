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
from collections import Counter
from data_tooling.pii_processing.ontology.ontology_manager import OntologyManager
from data_tooling.ac_dc.stopwords import stopwords as stopwords_ac_dc
from data_tooling.ac_dc.badwords import badwords as badwords_ac_dc
from  datasets import load_dataset
from transformers import AutoTokenizer, RobertaForTokenClassification, M2M100ForConditionalGeneration, M2M100Tokenizer, pipelines
import spacy
from tqdm import tqdm
import difflib
from transformers import pipeline, MarianMTModel, XLMRobertaForTokenClassification, BertForTokenClassification, ElectraForTokenClassification
import random
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import langid
from nltk.corpus import stopwords
import json
import os
import torch
torch.cuda.empty_cache()
mariam_mt = {('aav', 'en'): 'Helsinki-NLP/opus-mt-aav-en', ('aed', 'es'): 'Helsinki-NLP/opus-mt-aed-es', ('af', 'de'): 'Helsinki-NLP/opus-mt-af-de', ('af', 'en'): 'Helsinki-NLP/opus-mt-af-en', ('af', 'eo'): 'Helsinki-NLP/opus-mt-af-eo', ('af', 'es'): 'Helsinki-NLP/opus-mt-af-es', ('af', 'fi'): 'Helsinki-NLP/opus-mt-af-fi', ('af', 'fr'): 'Helsinki-NLP/opus-mt-af-fr', ('af', 'nl'): 'Helsinki-NLP/opus-mt-af-nl', ('af', 'ru'): 'Helsinki-NLP/opus-mt-af-ru', ('af', 'sv'): 'Helsinki-NLP/opus-mt-af-sv', ('afa', 'afa'): 'Helsinki-NLP/opus-mt-afa-afa', ('afa', 'en'): 'Helsinki-NLP/opus-mt-afa-en', ('alv', 'en'): 'Helsinki-NLP/opus-mt-alv-en', ('am', 'sv'): 'Helsinki-NLP/opus-mt-am-sv', ('ar', 'de'): 'Helsinki-NLP/opus-mt-ar-de', ('ar', 'el'): 'Helsinki-NLP/opus-mt-ar-el', ('ar', 'en'): 'Helsinki-NLP/opus-mt-ar-en', ('ar', 'eo'): 'Helsinki-NLP/opus-mt-ar-eo', ('ar', 'es'): 'Helsinki-NLP/opus-mt-ar-es', ('ar', 'fr'): 'Helsinki-NLP/opus-mt-ar-fr', ('ar', 'he'): 'Helsinki-NLP/opus-mt-ar-he', ('ar', 'it'): 'Helsinki-NLP/opus-mt-ar-it', ('ar', 'pl'): 'Helsinki-NLP/opus-mt-ar-pl', ('ar', 'ru'): 'Helsinki-NLP/opus-mt-ar-ru', ('ar', 'tr'): 'Helsinki-NLP/opus-mt-ar-tr', ('art', 'en'): 'Helsinki-NLP/opus-mt-art-en', ('ase', 'de'): 'Helsinki-NLP/opus-mt-ase-de', ('ase', 'en'): 'Helsinki-NLP/opus-mt-ase-en', ('ase', 'es'): 'Helsinki-NLP/opus-mt-ase-es', ('ase', 'fr'): 'Helsinki-NLP/opus-mt-ase-fr', ('ase', 'sv'): 'Helsinki-NLP/opus-mt-ase-sv', ('az', 'en'): 'Helsinki-NLP/opus-mt-az-en', ('az', 'es'): 'Helsinki-NLP/opus-mt-az-es', ('az', 'tr'): 'Helsinki-NLP/opus-mt-az-tr', ('bat', 'en'): 'Helsinki-NLP/opus-mt-bat-en', ('bcl', 'de'): 'Helsinki-NLP/opus-mt-bcl-de', ('bcl', 'en'): 'Helsinki-NLP/opus-mt-bcl-en', ('bcl', 'es'): 'Helsinki-NLP/opus-mt-bcl-es', ('bcl', 'fi'): 'Helsinki-NLP/opus-mt-bcl-fi', ('bcl', 'fr'): 'Helsinki-NLP/opus-mt-bcl-fr', ('bcl', 'sv'): 'Helsinki-NLP/opus-mt-bcl-sv', ('be', 'es'): 'Helsinki-NLP/opus-mt-be-es', ('bem', 'en'): 'Helsinki-NLP/opus-mt-bem-en', ('bem', 'es'): 'Helsinki-NLP/opus-mt-bem-es', ('bem', 'fi'): 'Helsinki-NLP/opus-mt-bem-fi', ('bem', 'fr'): 'Helsinki-NLP/opus-mt-bem-fr', ('bem', 'sv'): 'Helsinki-NLP/opus-mt-bem-sv', ('ber', 'en'): 'Helsinki-NLP/opus-mt-ber-en', ('ber', 'es'): 'Helsinki-NLP/opus-mt-ber-es', ('ber', 'fr'): 'Helsinki-NLP/opus-mt-ber-fr', ('bg', 'de'): 'Helsinki-NLP/opus-mt-bg-de', ('bg', 'en'): 'Helsinki-NLP/opus-mt-bg-en', ('bg', 'eo'): 'Helsinki-NLP/opus-mt-bg-eo', ('bg', 'es'): 'Helsinki-NLP/opus-mt-bg-es', ('bg', 'fi'): 'Helsinki-NLP/opus-mt-bg-fi', ('bg', 'fr'): 'Helsinki-NLP/opus-mt-bg-fr', ('bg', 'it'): 'Helsinki-NLP/opus-mt-bg-it', ('bg', 'ru'): 'Helsinki-NLP/opus-mt-bg-ru', ('bg', 'sv'): 'Helsinki-NLP/opus-mt-bg-sv', ('bg', 'tr'): 'Helsinki-NLP/opus-mt-bg-tr', ('bg', 'uk'): 'Helsinki-NLP/opus-mt-bg-uk', ('bi', 'en'): 'Helsinki-NLP/opus-mt-bi-en', ('bi', 'es'): 'Helsinki-NLP/opus-mt-bi-es', ('bi', 'fr'): 'Helsinki-NLP/opus-mt-bi-fr', ('bi', 'sv'): 'Helsinki-NLP/opus-mt-bi-sv', ('bn', 'en'): 'Helsinki-NLP/opus-mt-bn-en', ('bnt', 'en'): 'Helsinki-NLP/opus-mt-bnt-en', ('bzs', 'en'): 'Helsinki-NLP/opus-mt-bzs-en', ('bzs', 'es'): 'Helsinki-NLP/opus-mt-bzs-es', ('bzs', 'fi'): 'Helsinki-NLP/opus-mt-bzs-fi', ('bzs', 'fr'): 'Helsinki-NLP/opus-mt-bzs-fr', ('bzs', 'sv'): 'Helsinki-NLP/opus-mt-bzs-sv', ('ca', 'de'): 'Helsinki-NLP/opus-mt-ca-de', ('ca', 'en'): 'Helsinki-NLP/opus-mt-ca-en', ('ca', 'es'): 'Helsinki-NLP/opus-mt-ca-es', ('ca', 'fr'): 'Helsinki-NLP/opus-mt-ca-fr', ('ca', 'it'): 'Helsinki-NLP/opus-mt-ca-it', ('ca', 'nl'): 'Helsinki-NLP/opus-mt-ca-nl', ('ca', 'pt'): 'Helsinki-NLP/opus-mt-ca-pt', ('ca', 'uk'): 'Helsinki-NLP/opus-mt-ca-uk', ('cau', 'en'): 'Helsinki-NLP/opus-mt-cau-en', ('ccs', 'en'): 'Helsinki-NLP/opus-mt-ccs-en', ('ceb', 'en'): 'Helsinki-NLP/opus-mt-ceb-en', ('ceb', 'es'): 'Helsinki-NLP/opus-mt-ceb-es', ('ceb', 'fi'): 'Helsinki-NLP/opus-mt-ceb-fi', ('ceb', 'fr'): 'Helsinki-NLP/opus-mt-ceb-fr', ('ceb', 'sv'): 'Helsinki-NLP/opus-mt-ceb-sv', ('cel', 'en'): 'Helsinki-NLP/opus-mt-cel-en', ('chk', 'en'): 'Helsinki-NLP/opus-mt-chk-en', ('chk', 'es'): 'Helsinki-NLP/opus-mt-chk-es', ('chk', 'fr'): 'Helsinki-NLP/opus-mt-chk-fr', ('chk', 'sv'): 'Helsinki-NLP/opus-mt-chk-sv', ('cpf', 'en'): 'Helsinki-NLP/opus-mt-cpf-en', ('cpp', 'cpp'): 'Helsinki-NLP/opus-mt-cpp-cpp', ('cpp', 'en'): 'Helsinki-NLP/opus-mt-cpp-en', ('crs', 'de'): 'Helsinki-NLP/opus-mt-crs-de', ('crs', 'en'): 'Helsinki-NLP/opus-mt-crs-en', ('crs', 'es'): 'Helsinki-NLP/opus-mt-crs-es', ('crs', 'fi'): 'Helsinki-NLP/opus-mt-crs-fi', ('crs', 'fr'): 'Helsinki-NLP/opus-mt-crs-fr', ('crs', 'sv'): 'Helsinki-NLP/opus-mt-crs-sv', ('cs', 'de'): 'Helsinki-NLP/opus-mt-cs-de', ('cs', 'en'): 'Helsinki-NLP/opus-mt-cs-en', ('cs', 'eo'): 'Helsinki-NLP/opus-mt-cs-eo', ('cs', 'fi'): 'Helsinki-NLP/opus-mt-cs-fi', ('cs', 'fr'): 'Helsinki-NLP/opus-mt-cs-fr', ('cs', 'sv'): 'Helsinki-NLP/opus-mt-cs-sv', ('cs', 'uk'): 'Helsinki-NLP/opus-mt-cs-uk', ('csg', 'es'): 'Helsinki-NLP/opus-mt-csg-es', ('csn', 'es'): 'Helsinki-NLP/opus-mt-csn-es', ('cus', 'en'): 'Helsinki-NLP/opus-mt-cus-en', ('cy', 'en'): 'Helsinki-NLP/opus-mt-cy-en', ('da', 'de'): 'Helsinki-NLP/opus-mt-da-de', ('da', 'en'): 'Helsinki-NLP/opus-mt-da-en', ('da', 'eo'): 'Helsinki-NLP/opus-mt-da-eo', ('da', 'es'): 'Helsinki-NLP/opus-mt-da-es', ('da', 'fi'): 'Helsinki-NLP/opus-mt-da-fi', ('da', 'fr'): 'Helsinki-NLP/opus-mt-da-fr', ('da', 'no'): 'Helsinki-NLP/opus-mt-da-no', ('da', 'ru'): 'Helsinki-NLP/opus-mt-da-ru', ('de', 'ZH'): 'Helsinki-NLP/opus-mt-de-ZH', ('de', 'af'): 'Helsinki-NLP/opus-mt-de-af', ('de', 'ar'): 'Helsinki-NLP/opus-mt-de-ar', ('de', 'ase'): 'Helsinki-NLP/opus-mt-de-ase', ('de', 'bcl'): 'Helsinki-NLP/opus-mt-de-bcl', ('de', 'bg'): 'Helsinki-NLP/opus-mt-de-bg', ('de', 'bi'): 'Helsinki-NLP/opus-mt-de-bi', ('de', 'bzs'): 'Helsinki-NLP/opus-mt-de-bzs', ('de', 'ca'): 'Helsinki-NLP/opus-mt-de-ca', ('de', 'crs'): 'Helsinki-NLP/opus-mt-de-crs', ('de', 'cs'): 'Helsinki-NLP/opus-mt-de-cs', ('de', 'da'): 'Helsinki-NLP/opus-mt-de-da', ('de', 'de'): 'Helsinki-NLP/opus-mt-de-de', ('de', 'ee'): 'Helsinki-NLP/opus-mt-de-ee', ('de', 'efi'): 'Helsinki-NLP/opus-mt-de-efi', ('de', 'el'): 'Helsinki-NLP/opus-mt-de-el', ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en', ('de', 'eo'): 'Helsinki-NLP/opus-mt-de-eo', ('de', 'es'): 'Helsinki-NLP/opus-mt-de-es', ('de', 'et'): 'Helsinki-NLP/opus-mt-de-et', ('de', 'eu'): 'Helsinki-NLP/opus-mt-de-eu', ('de', 'fi'): 'Helsinki-NLP/opus-mt-de-fi', ('de', 'fj'): 'Helsinki-NLP/opus-mt-de-fj', ('de', 'fr'): 'Helsinki-NLP/opus-mt-de-fr', ('de', 'gaa'): 'Helsinki-NLP/opus-mt-de-gaa', ('de', 'gil'): 'Helsinki-NLP/opus-mt-de-gil', ('de', 'guw'): 'Helsinki-NLP/opus-mt-de-guw', ('de', 'ha'): 'Helsinki-NLP/opus-mt-de-ha', ('de', 'he'): 'Helsinki-NLP/opus-mt-de-he', ('de', 'hil'): 'Helsinki-NLP/opus-mt-de-hil', ('de', 'ho'): 'Helsinki-NLP/opus-mt-de-ho', ('de', 'hr'): 'Helsinki-NLP/opus-mt-de-hr', ('de', 'ht'): 'Helsinki-NLP/opus-mt-de-ht', ('de', 'hu'): 'Helsinki-NLP/opus-mt-de-hu', ('de', 'ig'): 'Helsinki-NLP/opus-mt-de-ig', ('de', 'ilo'): 'Helsinki-NLP/opus-mt-de-ilo', ('de', 'is'): 'Helsinki-NLP/opus-mt-de-is', ('de', 'iso'): 'Helsinki-NLP/opus-mt-de-iso', ('de', 'it'): 'Helsinki-NLP/opus-mt-de-it', ('de', 'kg'): 'Helsinki-NLP/opus-mt-de-kg', ('de', 'ln'): 'Helsinki-NLP/opus-mt-de-ln', ('de', 'loz'): 'Helsinki-NLP/opus-mt-de-loz', ('de', 'lt'): 'Helsinki-NLP/opus-mt-de-lt', ('de', 'lua'): 'Helsinki-NLP/opus-mt-de-lua', ('de', 'ms'): 'Helsinki-NLP/opus-mt-de-ms', ('de', 'mt'): 'Helsinki-NLP/opus-mt-de-mt', ('de', 'niu'): 'Helsinki-NLP/opus-mt-de-niu', ('de', 'nl'): 'Helsinki-NLP/opus-mt-de-nl', ('de', 'no'): 'Helsinki-NLP/opus-mt-de-no', ('de', 'nso'): 'Helsinki-NLP/opus-mt-de-nso', ('de', 'ny'): 'Helsinki-NLP/opus-mt-de-ny', ('de', 'pag'): 'Helsinki-NLP/opus-mt-de-pag', ('de', 'pap'): 'Helsinki-NLP/opus-mt-de-pap', ('de', 'pis'): 'Helsinki-NLP/opus-mt-de-pis', ('de', 'pl'): 'Helsinki-NLP/opus-mt-de-pl', ('de', 'pon'): 'Helsinki-NLP/opus-mt-de-pon', ('de', 'tl'): 'Helsinki-NLP/opus-mt-de-tl', ('de', 'uk'): 'Helsinki-NLP/opus-mt-de-uk', ('de', 'vi'): 'Helsinki-NLP/opus-mt-de-vi', ('dra', 'en'): 'Helsinki-NLP/opus-mt-dra-en', ('ee', 'de'): 'Helsinki-NLP/opus-mt-ee-de', ('ee', 'en'): 'Helsinki-NLP/opus-mt-ee-en', ('ee', 'es'): 'Helsinki-NLP/opus-mt-ee-es', ('ee', 'fi'): 'Helsinki-NLP/opus-mt-ee-fi', ('ee', 'fr'): 'Helsinki-NLP/opus-mt-ee-fr', ('ee', 'sv'): 'Helsinki-NLP/opus-mt-ee-sv', ('efi', 'de'): 'Helsinki-NLP/opus-mt-efi-de', ('efi', 'en'): 'Helsinki-NLP/opus-mt-efi-en', ('efi', 'fi'): 'Helsinki-NLP/opus-mt-efi-fi', ('efi', 'fr'): 'Helsinki-NLP/opus-mt-efi-fr', ('efi', 'sv'): 'Helsinki-NLP/opus-mt-efi-sv', ('el', 'ar'): 'Helsinki-NLP/opus-mt-el-ar', ('el', 'eo'): 'Helsinki-NLP/opus-mt-el-eo', ('el', 'fi'): 'Helsinki-NLP/opus-mt-el-fi', ('el', 'fr'): 'Helsinki-NLP/opus-mt-el-fr', ('el', 'sv'): 'Helsinki-NLP/opus-mt-el-sv', ('en', 'aav'): 'Helsinki-NLP/opus-mt-en-aav', ('en', 'af'): 'Helsinki-NLP/opus-mt-en-af', ('en', 'afa'): 'Helsinki-NLP/opus-mt-en-afa', ('en', 'alv'): 'Helsinki-NLP/opus-mt-en-alv', ('en', 'ar'): 'Helsinki-NLP/opus-mt-en-ar', ('en', 'az'): 'Helsinki-NLP/opus-mt-en-az', ('en', 'bat'): 'Helsinki-NLP/opus-mt-en-bat', ('en', 'bcl'): 'Helsinki-NLP/opus-mt-en-bcl', ('en', 'bem'): 'Helsinki-NLP/opus-mt-en-bem', ('en', 'ber'): 'Helsinki-NLP/opus-mt-en-ber', ('en', 'bg'): 'Helsinki-NLP/opus-mt-en-bg', ('en', 'bi'): 'Helsinki-NLP/opus-mt-en-bi', ('en', 'bnt'): 'Helsinki-NLP/opus-mt-en-bnt', ('en', 'bzs'): 'Helsinki-NLP/opus-mt-en-bzs', ('en', 'ca'): 'Helsinki-NLP/opus-mt-en-ca', ('en', 'ceb'): 'Helsinki-NLP/opus-mt-en-ceb', ('en', 'cel'): 'Helsinki-NLP/opus-mt-en-cel', ('en', 'chk'): 'Helsinki-NLP/opus-mt-en-chk', ('en', 'cpf'): 'Helsinki-NLP/opus-mt-en-cpf', ('en', 'cpp'): 'Helsinki-NLP/opus-mt-en-cpp', ('en', 'crs'): 'Helsinki-NLP/opus-mt-en-crs', ('en', 'cs'): 'Helsinki-NLP/opus-mt-en-cs', ('en', 'cus'): 'Helsinki-NLP/opus-mt-en-cus', ('en', 'cy'): 'Helsinki-NLP/opus-mt-en-cy', ('en', 'da'): 'Helsinki-NLP/opus-mt-en-da', ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de', ('en', 'dra'): 'Helsinki-NLP/opus-mt-en-dra', ('en', 'ee'): 'Helsinki-NLP/opus-mt-en-ee', ('en', 'efi'): 'Helsinki-NLP/opus-mt-en-efi', ('en', 'el'): 'Helsinki-NLP/opus-mt-en-el', ('en', 'eo'): 'Helsinki-NLP/opus-mt-en-eo', ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es', ('en', 'et'): 'Helsinki-NLP/opus-mt-en-et', ('en', 'eu'): 'Helsinki-NLP/opus-mt-en-eu', ('en', 'euq'): 'Helsinki-NLP/opus-mt-en-euq', ('en', 'fi'): 'Helsinki-NLP/opus-mt-en-fi', ('en', 'fiu'): 'Helsinki-NLP/opus-mt-en-fiu', ('en', 'fj'): 'Helsinki-NLP/opus-mt-en-fj', ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr', ('en', 'ga'): 'Helsinki-NLP/opus-mt-en-ga', ('en', 'gaa'): 'Helsinki-NLP/opus-mt-en-gaa', ('en', 'gem'): 'Helsinki-NLP/opus-mt-en-gem', ('en', 'gil'): 'Helsinki-NLP/opus-mt-en-gil', ('en', 'gl'): 'Helsinki-NLP/opus-mt-en-gl', ('en', 'gmq'): 'Helsinki-NLP/opus-mt-en-gmq', ('en', 'gmw'): 'Helsinki-NLP/opus-mt-en-gmw', ('en', 'grk'): 'Helsinki-NLP/opus-mt-en-grk', ('en', 'guw'): 'Helsinki-NLP/opus-mt-en-guw', ('en', 'gv'): 'Helsinki-NLP/opus-mt-en-gv', ('en', 'ha'): 'Helsinki-NLP/opus-mt-en-ha', ('en', 'he'): 'Helsinki-NLP/opus-mt-en-he', ('en', 'hi'): 'Helsinki-NLP/opus-mt-en-hi', ('en', 'hil'): 'Helsinki-NLP/opus-mt-en-hil', ('en', 'ho'): 'Helsinki-NLP/opus-mt-en-ho', ('en', 'ht'): 'Helsinki-NLP/opus-mt-en-ht', ('en', 'hu'): 'Helsinki-NLP/opus-mt-en-hu', ('en', 'hy'): 'Helsinki-NLP/opus-mt-en-hy', ('en', 'id'): 'Helsinki-NLP/opus-mt-en-id', ('en', 'ig'): 'Helsinki-NLP/opus-mt-en-ig', ('en', 'iir'): 'Helsinki-NLP/opus-mt-en-iir', ('en', 'ilo'): 'Helsinki-NLP/opus-mt-en-ilo', ('en', 'inc'): 'Helsinki-NLP/opus-mt-en-inc', ('en', 'ine'): 'Helsinki-NLP/opus-mt-en-ine', ('en', 'is'): 'Helsinki-NLP/opus-mt-en-is', ('en', 'iso'): 'Helsinki-NLP/opus-mt-en-iso', ('en', 'it'): 'Helsinki-NLP/opus-mt-en-it', ('en', 'itc'): 'Helsinki-NLP/opus-mt-en-itc', ('en', 'jap'): 'Helsinki-NLP/opus-mt-en-jap', ('en', 'kg'): 'Helsinki-NLP/opus-mt-en-kg', ('en', 'kj'): 'Helsinki-NLP/opus-mt-en-kj', ('en', 'kqn'): 'Helsinki-NLP/opus-mt-en-kqn', ('en', 'kwn'): 'Helsinki-NLP/opus-mt-en-kwn', ('en', 'kwy'): 'Helsinki-NLP/opus-mt-en-kwy', ('en', 'lg'): 'Helsinki-NLP/opus-mt-en-lg', ('en', 'ln'): 'Helsinki-NLP/opus-mt-en-ln', ('en', 'loz'): 'Helsinki-NLP/opus-mt-en-loz', ('en', 'lu'): 'Helsinki-NLP/opus-mt-en-lu', ('en', 'lua'): 'Helsinki-NLP/opus-mt-en-lua', ('en', 'lue'): 'Helsinki-NLP/opus-mt-en-lue', ('en', 'lun'): 'Helsinki-NLP/opus-mt-en-lun', ('en', 'luo'): 'Helsinki-NLP/opus-mt-en-luo', ('en', 'lus'): 'Helsinki-NLP/opus-mt-en-lus', ('en', 'map'): 'Helsinki-NLP/opus-mt-en-map', ('en', 'mfe'): 'Helsinki-NLP/opus-mt-en-mfe', ('en', 'mg'): 'Helsinki-NLP/opus-mt-en-mg', ('en', 'mh'): 'Helsinki-NLP/opus-mt-en-mh', ('en', 'mk'): 'Helsinki-NLP/opus-mt-en-mk', ('en', 'mkh'): 'Helsinki-NLP/opus-mt-en-mkh', ('en', 'ml'): 'Helsinki-NLP/opus-mt-en-ml', ('en', 'mos'): 'Helsinki-NLP/opus-mt-en-mos', ('en', 'mr'): 'Helsinki-NLP/opus-mt-en-mr', ('en', 'mt'): 'Helsinki-NLP/opus-mt-en-mt', ('en', 'mul'): 'Helsinki-NLP/opus-mt-en-mul', ('en', 'ng'): 'Helsinki-NLP/opus-mt-en-ng', ('en', 'nic'): 'Helsinki-NLP/opus-mt-en-nic', ('en', 'niu'): 'Helsinki-NLP/opus-mt-en-niu', ('en', 'nl'): 'Helsinki-NLP/opus-mt-en-nl', ('en', 'nso'): 'Helsinki-NLP/opus-mt-en-nso', ('en', 'ny'): 'Helsinki-NLP/opus-mt-en-ny', ('en', 'nyk'): 'Helsinki-NLP/opus-mt-en-nyk', ('en', 'om'): 'Helsinki-NLP/opus-mt-en-om', ('en', 'pag'): 'Helsinki-NLP/opus-mt-en-pag', ('en', 'pap'): 'Helsinki-NLP/opus-mt-en-pap', ('en', 'phi'): 'Helsinki-NLP/opus-mt-en-phi', ('en', 'pis'): 'Helsinki-NLP/opus-mt-en-pis', ('en', 'pon'): 'Helsinki-NLP/opus-mt-en-pon', ('en', 'poz'): 'Helsinki-NLP/opus-mt-en-poz', ('en', 'pqe'): 'Helsinki-NLP/opus-mt-en-pqe', ('en', 'pqw'): 'Helsinki-NLP/opus-mt-en-pqw', ('en', 'rn'): 'Helsinki-NLP/opus-mt-en-rn', ('en', 'rnd'): 'Helsinki-NLP/opus-mt-en-rnd', ('en', 'ro'): 'Helsinki-NLP/opus-mt-en-ro', ('en', 'roa'): 'Helsinki-NLP/opus-mt-en-roa', ('en', 'ru'): 'Helsinki-NLP/opus-mt-en-ru', ('en', 'run'): 'Helsinki-NLP/opus-mt-en-run', ('en', 'rw'): 'Helsinki-NLP/opus-mt-en-rw', ('en', 'sal'): 'Helsinki-NLP/opus-mt-en-sal', ('en', 'sem'): 'Helsinki-NLP/opus-mt-en-sem', ('en', 'sg'): 'Helsinki-NLP/opus-mt-en-sg', ('en', 'sit'): 'Helsinki-NLP/opus-mt-en-sit', ('en', 'sk'): 'Helsinki-NLP/opus-mt-en-sk', ('en', 'sla'): 'Helsinki-NLP/opus-mt-en-sla', ('en', 'sm'): 'Helsinki-NLP/opus-mt-en-sm', ('en', 'sn'): 'Helsinki-NLP/opus-mt-en-sn', ('en', 'sq'): 'Helsinki-NLP/opus-mt-en-sq', ('en', 'ss'): 'Helsinki-NLP/opus-mt-en-ss', ('en', 'st'): 'Helsinki-NLP/opus-mt-en-st', ('en', 'sv'): 'Helsinki-NLP/opus-mt-en-sv', ('en', 'sw'): 'Helsinki-NLP/opus-mt-en-sw', ('en', 'swc'): 'Helsinki-NLP/opus-mt-en-swc', ('en', 'tdt'): 'Helsinki-NLP/opus-mt-en-tdt', ('en', 'ti'): 'Helsinki-NLP/opus-mt-en-ti', ('en', 'tiv'): 'Helsinki-NLP/opus-mt-en-tiv', ('en', 'tl'): 'Helsinki-NLP/opus-mt-en-tl', ('en', 'tll'): 'Helsinki-NLP/opus-mt-en-tll', ('en', 'tn'): 'Helsinki-NLP/opus-mt-en-tn', ('en', 'to'): 'Helsinki-NLP/opus-mt-en-to', ('en', 'toi'): 'Helsinki-NLP/opus-mt-en-toi', ('en', 'tpi'): 'Helsinki-NLP/opus-mt-en-tpi', ('en', 'trk'): 'Helsinki-NLP/opus-mt-en-trk', ('en', 'ts'): 'Helsinki-NLP/opus-mt-en-ts', ('en', 'tut'): 'Helsinki-NLP/opus-mt-en-tut', ('en', 'tvl'): 'Helsinki-NLP/opus-mt-en-tvl', ('en', 'tw'): 'Helsinki-NLP/opus-mt-en-tw', ('en', 'ty'): 'Helsinki-NLP/opus-mt-en-ty', ('en', 'uk'): 'Helsinki-NLP/opus-mt-en-uk', ('en', 'umb'): 'Helsinki-NLP/opus-mt-en-umb', ('en', 'ur'): 'Helsinki-NLP/opus-mt-en-ur', ('en', 'urj'): 'Helsinki-NLP/opus-mt-en-urj', ('en', 'vi'): 'Helsinki-NLP/opus-mt-en-vi', ('en', 'xh'): 'Helsinki-NLP/opus-mt-en-xh', ('en', 'zh'): 'Helsinki-NLP/opus-mt-en-zh', ('en', 'zle'): 'Helsinki-NLP/opus-mt-en-zle', ('en', 'zls'): 'Helsinki-NLP/opus-mt-en-zls', ('en', 'zlw'): 'Helsinki-NLP/opus-mt-en-zlw', ('en_el_es_fi', 'en_el_es_fi'): 'Helsinki-NLP/opus-mt-en_el_es_fi-en_el_es_fi', ('eo', 'af'): 'Helsinki-NLP/opus-mt-eo-af', ('eo', 'bg'): 'Helsinki-NLP/opus-mt-eo-bg', ('eo', 'cs'): 'Helsinki-NLP/opus-mt-eo-cs', ('eo', 'da'): 'Helsinki-NLP/opus-mt-eo-da', ('eo', 'de'): 'Helsinki-NLP/opus-mt-eo-de', ('eo', 'el'): 'Helsinki-NLP/opus-mt-eo-el', ('eo', 'en'): 'Helsinki-NLP/opus-mt-eo-en', ('eo', 'es'): 'Helsinki-NLP/opus-mt-eo-es', ('eo', 'fi'): 'Helsinki-NLP/opus-mt-eo-fi', ('eo', 'fr'): 'Helsinki-NLP/opus-mt-eo-fr', ('eo', 'he'): 'Helsinki-NLP/opus-mt-eo-he', ('eo', 'hu'): 'Helsinki-NLP/opus-mt-eo-hu', ('eo', 'it'): 'Helsinki-NLP/opus-mt-eo-it', ('eo', 'nl'): 'Helsinki-NLP/opus-mt-eo-nl', ('eo', 'pl'): 'Helsinki-NLP/opus-mt-eo-pl', ('eo', 'pt'): 'Helsinki-NLP/opus-mt-eo-pt', ('eo', 'ro'): 'Helsinki-NLP/opus-mt-eo-ro', ('eo', 'ru'): 'Helsinki-NLP/opus-mt-eo-ru', ('eo', 'sh'): 'Helsinki-NLP/opus-mt-eo-sh', ('eo', 'sv'): 'Helsinki-NLP/opus-mt-eo-sv', ('es', 'NORWAY'): 'Helsinki-NLP/opus-mt-es-NORWAY', ('es', 'aed'): 'Helsinki-NLP/opus-mt-es-aed', ('es', 'af'): 'Helsinki-NLP/opus-mt-es-af', ('es', 'ar'): 'Helsinki-NLP/opus-mt-es-ar', ('es', 'ase'): 'Helsinki-NLP/opus-mt-es-ase', ('es', 'bcl'): 'Helsinki-NLP/opus-mt-es-bcl', ('es', 'ber'): 'Helsinki-NLP/opus-mt-es-ber', ('es', 'bg'): 'Helsinki-NLP/opus-mt-es-bg', ('es', 'bi'): 'Helsinki-NLP/opus-mt-es-bi', ('es', 'bzs'): 'Helsinki-NLP/opus-mt-es-bzs', ('es', 'ca'): 'Helsinki-NLP/opus-mt-es-ca', ('es', 'ceb'): 'Helsinki-NLP/opus-mt-es-ceb', ('es', 'crs'): 'Helsinki-NLP/opus-mt-es-crs', ('es', 'cs'): 'Helsinki-NLP/opus-mt-es-cs', ('es', 'csg'): 'Helsinki-NLP/opus-mt-es-csg', ('es', 'csn'): 'Helsinki-NLP/opus-mt-es-csn', ('es', 'da'): 'Helsinki-NLP/opus-mt-es-da', ('es', 'de'): 'Helsinki-NLP/opus-mt-es-de', ('es', 'ee'): 'Helsinki-NLP/opus-mt-es-ee', ('es', 'efi'): 'Helsinki-NLP/opus-mt-es-efi', ('es', 'el'): 'Helsinki-NLP/opus-mt-es-el', ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en', ('es', 'eo'): 'Helsinki-NLP/opus-mt-es-eo', ('es', 'es'): 'Helsinki-NLP/opus-mt-es-es', ('es', 'et'): 'Helsinki-NLP/opus-mt-es-et', ('es', 'eu'): 'Helsinki-NLP/opus-mt-es-eu', ('es', 'fi'): 'Helsinki-NLP/opus-mt-es-fi', ('es', 'fj'): 'Helsinki-NLP/opus-mt-es-fj', ('es', 'fr'): 'Helsinki-NLP/opus-mt-es-fr', ('es', 'gaa'): 'Helsinki-NLP/opus-mt-es-gaa', ('es', 'gil'): 'Helsinki-NLP/opus-mt-es-gil', ('es', 'gl'): 'Helsinki-NLP/opus-mt-es-gl', ('es', 'guw'): 'Helsinki-NLP/opus-mt-es-guw', ('es', 'ha'): 'Helsinki-NLP/opus-mt-es-ha', ('es', 'he'): 'Helsinki-NLP/opus-mt-es-he', ('es', 'hil'): 'Helsinki-NLP/opus-mt-es-hil', ('es', 'ho'): 'Helsinki-NLP/opus-mt-es-ho', ('es', 'hr'): 'Helsinki-NLP/opus-mt-es-hr', ('es', 'ht'): 'Helsinki-NLP/opus-mt-es-ht', ('es', 'id'): 'Helsinki-NLP/opus-mt-es-id', ('es', 'ig'): 'Helsinki-NLP/opus-mt-es-ig', ('es', 'ilo'): 'Helsinki-NLP/opus-mt-es-ilo', ('es', 'is'): 'Helsinki-NLP/opus-mt-es-is', ('es', 'iso'): 'Helsinki-NLP/opus-mt-es-iso', ('es', 'it'): 'Helsinki-NLP/opus-mt-es-it', ('es', 'kg'): 'Helsinki-NLP/opus-mt-es-kg', ('es', 'ln'): 'Helsinki-NLP/opus-mt-es-ln', ('es', 'loz'): 'Helsinki-NLP/opus-mt-es-loz', ('es', 'lt'): 'Helsinki-NLP/opus-mt-es-lt', ('es', 'lua'): 'Helsinki-NLP/opus-mt-es-lua', ('es', 'lus'): 'Helsinki-NLP/opus-mt-es-lus', ('es', 'mfs'): 'Helsinki-NLP/opus-mt-es-mfs', ('es', 'mk'): 'Helsinki-NLP/opus-mt-es-mk', ('es', 'mt'): 'Helsinki-NLP/opus-mt-es-mt', ('es', 'niu'): 'Helsinki-NLP/opus-mt-es-niu', ('es', 'nl'): 'Helsinki-NLP/opus-mt-es-nl', ('es', 'no'): 'Helsinki-NLP/opus-mt-es-no', ('es', 'nso'): 'Helsinki-NLP/opus-mt-es-nso', ('es', 'ny'): 'Helsinki-NLP/opus-mt-es-ny', ('es', 'pag'): 'Helsinki-NLP/opus-mt-es-pag', ('es', 'pap'): 'Helsinki-NLP/opus-mt-es-pap', ('es', 'pis'): 'Helsinki-NLP/opus-mt-es-pis', ('es', 'pl'): 'Helsinki-NLP/opus-mt-es-pl', ('es', 'pon'): 'Helsinki-NLP/opus-mt-es-pon', ('es', 'prl'): 'Helsinki-NLP/opus-mt-es-prl', ('es', 'rn'): 'Helsinki-NLP/opus-mt-es-rn', ('es', 'ro'): 'Helsinki-NLP/opus-mt-es-ro', ('es', 'ru'): 'Helsinki-NLP/opus-mt-es-ru', ('es', 'rw'): 'Helsinki-NLP/opus-mt-es-rw', ('es', 'sg'): 'Helsinki-NLP/opus-mt-es-sg', ('es', 'sl'): 'Helsinki-NLP/opus-mt-es-sl', ('es', 'sm'): 'Helsinki-NLP/opus-mt-es-sm', ('es', 'sn'): 'Helsinki-NLP/opus-mt-es-sn', ('es', 'srn'): 'Helsinki-NLP/opus-mt-es-srn', ('es', 'st'): 'Helsinki-NLP/opus-mt-es-st', ('es', 'swc'): 'Helsinki-NLP/opus-mt-es-swc', ('es', 'tl'): 'Helsinki-NLP/opus-mt-es-tl', ('es', 'tll'): 'Helsinki-NLP/opus-mt-es-tll', ('es', 'tn'): 'Helsinki-NLP/opus-mt-es-tn', ('es', 'to'): 'Helsinki-NLP/opus-mt-es-to', ('es', 'tpi'): 'Helsinki-NLP/opus-mt-es-tpi', ('es', 'tvl'): 'Helsinki-NLP/opus-mt-es-tvl', ('es', 'tw'): 'Helsinki-NLP/opus-mt-es-tw', ('es', 'ty'): 'Helsinki-NLP/opus-mt-es-ty', ('es', 'tzo'): 'Helsinki-NLP/opus-mt-es-tzo', ('es', 'uk'): 'Helsinki-NLP/opus-mt-es-uk', ('es', 've'): 'Helsinki-NLP/opus-mt-es-ve', ('es', 'vi'): 'Helsinki-NLP/opus-mt-es-vi', ('es', 'war'): 'Helsinki-NLP/opus-mt-es-war', ('es', 'wls'): 'Helsinki-NLP/opus-mt-es-wls', ('es', 'xh'): 'Helsinki-NLP/opus-mt-es-xh', ('es', 'yo'): 'Helsinki-NLP/opus-mt-es-yo', ('es', 'yua'): 'Helsinki-NLP/opus-mt-es-yua', ('es', 'zai'): 'Helsinki-NLP/opus-mt-es-zai', ('et', 'de'): 'Helsinki-NLP/opus-mt-et-de', ('et', 'en'): 'Helsinki-NLP/opus-mt-et-en', ('et', 'es'): 'Helsinki-NLP/opus-mt-et-es', ('et', 'fi'): 'Helsinki-NLP/opus-mt-et-fi', ('et', 'fr'): 'Helsinki-NLP/opus-mt-et-fr', ('et', 'ru'): 'Helsinki-NLP/opus-mt-et-ru', ('et', 'sv'): 'Helsinki-NLP/opus-mt-et-sv', ('eu', 'de'): 'Helsinki-NLP/opus-mt-eu-de', ('eu', 'en'): 'Helsinki-NLP/opus-mt-eu-en', ('eu', 'es'): 'Helsinki-NLP/opus-mt-eu-es', ('eu', 'ru'): 'Helsinki-NLP/opus-mt-eu-ru', ('euq', 'en'): 'Helsinki-NLP/opus-mt-euq-en', ('fi', 'NORWAY'): 'Helsinki-NLP/opus-mt-fi-NORWAY', ('fi', 'ZH'): 'Helsinki-NLP/opus-mt-fi-ZH', ('fi', 'af'): 'Helsinki-NLP/opus-mt-fi-af', ('fi', 'bcl'): 'Helsinki-NLP/opus-mt-fi-bcl', ('fi', 'bem'): 'Helsinki-NLP/opus-mt-fi-bem', ('fi', 'bg'): 'Helsinki-NLP/opus-mt-fi-bg', ('fi', 'bzs'): 'Helsinki-NLP/opus-mt-fi-bzs', ('fi', 'ceb'): 'Helsinki-NLP/opus-mt-fi-ceb', ('fi', 'crs'): 'Helsinki-NLP/opus-mt-fi-crs', ('fi', 'cs'): 'Helsinki-NLP/opus-mt-fi-cs', ('fi', 'de'): 'Helsinki-NLP/opus-mt-fi-de', ('fi', 'ee'): 'Helsinki-NLP/opus-mt-fi-ee', ('fi', 'efi'): 'Helsinki-NLP/opus-mt-fi-efi', ('fi', 'el'): 'Helsinki-NLP/opus-mt-fi-el', ('fi', 'en'): 'Helsinki-NLP/opus-mt-fi-en', ('fi', 'eo'): 'Helsinki-NLP/opus-mt-fi-eo', ('fi', 'es'): 'Helsinki-NLP/opus-mt-fi-es', ('fi', 'et'): 'Helsinki-NLP/opus-mt-fi-et', ('fi', 'fi'): 'Helsinki-NLP/opus-mt-fi-fi', ('fi', 'fj'): 'Helsinki-NLP/opus-mt-fi-fj', ('fi', 'fr'): 'Helsinki-NLP/opus-mt-fi-fr', ('fi', 'fse'): 'Helsinki-NLP/opus-mt-fi-fse', ('fi', 'gaa'): 'Helsinki-NLP/opus-mt-fi-gaa', ('fi', 'gil'): 'Helsinki-NLP/opus-mt-fi-gil', ('fi', 'guw'): 'Helsinki-NLP/opus-mt-fi-guw', ('fi', 'ha'): 'Helsinki-NLP/opus-mt-fi-ha', ('fi', 'he'): 'Helsinki-NLP/opus-mt-fi-he', ('fi', 'hil'): 'Helsinki-NLP/opus-mt-fi-hil', ('fi', 'ho'): 'Helsinki-NLP/opus-mt-fi-ho', ('fi', 'hr'): 'Helsinki-NLP/opus-mt-fi-hr', ('fi', 'ht'): 'Helsinki-NLP/opus-mt-fi-ht', ('fi', 'hu'): 'Helsinki-NLP/opus-mt-fi-hu', ('fi', 'id'): 'Helsinki-NLP/opus-mt-fi-id', ('fi', 'ig'): 'Helsinki-NLP/opus-mt-fi-ig', ('fi', 'ilo'): 'Helsinki-NLP/opus-mt-fi-ilo', ('fi', 'is'): 'Helsinki-NLP/opus-mt-fi-is', ('fi', 'iso'): 'Helsinki-NLP/opus-mt-fi-iso', ('fi', 'it'): 'Helsinki-NLP/opus-mt-fi-it', ('fi', 'kg'): 'Helsinki-NLP/opus-mt-fi-kg', ('fi', 'kqn'): 'Helsinki-NLP/opus-mt-fi-kqn', ('fi', 'lg'): 'Helsinki-NLP/opus-mt-fi-lg', ('fi', 'ln'): 'Helsinki-NLP/opus-mt-fi-ln', ('fi', 'lu'): 'Helsinki-NLP/opus-mt-fi-lu', ('fi', 'lua'): 'Helsinki-NLP/opus-mt-fi-lua', ('fi', 'lue'): 'Helsinki-NLP/opus-mt-fi-lue', ('fi', 'lus'): 'Helsinki-NLP/opus-mt-fi-lus', ('fi', 'lv'): 'Helsinki-NLP/opus-mt-fi-lv', ('fi', 'mfe'): 'Helsinki-NLP/opus-mt-fi-mfe', ('fi', 'mg'): 'Helsinki-NLP/opus-mt-fi-mg', ('fi', 'mh'): 'Helsinki-NLP/opus-mt-fi-mh', ('fi', 'mk'): 'Helsinki-NLP/opus-mt-fi-mk', ('fi', 'mos'): 'Helsinki-NLP/opus-mt-fi-mos', ('fi', 'mt'): 'Helsinki-NLP/opus-mt-fi-mt', ('fi', 'niu'): 'Helsinki-NLP/opus-mt-fi-niu', ('fi', 'nl'): 'Helsinki-NLP/opus-mt-fi-nl', ('fi', 'no'): 'Helsinki-NLP/opus-mt-fi-no', ('fi', 'nso'): 'Helsinki-NLP/opus-mt-fi-nso', ('fi', 'ny'): 'Helsinki-NLP/opus-mt-fi-ny', ('fi', 'pag'): 'Helsinki-NLP/opus-mt-fi-pag', ('fi', 'pap'): 'Helsinki-NLP/opus-mt-fi-pap', ('fi', 'pis'): 'Helsinki-NLP/opus-mt-fi-pis', ('fi', 'pon'): 'Helsinki-NLP/opus-mt-fi-pon', ('fi', 'ro'): 'Helsinki-NLP/opus-mt-fi-ro', ('fi', 'ru'): 'Helsinki-NLP/opus-mt-fi-ru', ('fi', 'run'): 'Helsinki-NLP/opus-mt-fi-run', ('fi', 'rw'): 'Helsinki-NLP/opus-mt-fi-rw', ('fi', 'sg'): 'Helsinki-NLP/opus-mt-fi-sg', ('fi', 'sk'): 'Helsinki-NLP/opus-mt-fi-sk', ('fi', 'sl'): 'Helsinki-NLP/opus-mt-fi-sl', ('fi', 'sm'): 'Helsinki-NLP/opus-mt-fi-sm', ('fi', 'sn'): 'Helsinki-NLP/opus-mt-fi-sn', ('fi', 'sq'): 'Helsinki-NLP/opus-mt-fi-sq', ('fi', 'srn'): 'Helsinki-NLP/opus-mt-fi-srn', ('fi', 'st'): 'Helsinki-NLP/opus-mt-fi-st', ('fi', 'sv'): 'Helsinki-NLP/opus-mt-fi-sv', ('fi', 'sw'): 'Helsinki-NLP/opus-mt-fi-sw', ('fi', 'swc'): 'Helsinki-NLP/opus-mt-fi-swc', ('fi', 'tiv'): 'Helsinki-NLP/opus-mt-fi-tiv', ('fi', 'tll'): 'Helsinki-NLP/opus-mt-fi-tll', ('fi', 'tn'): 'Helsinki-NLP/opus-mt-fi-tn', ('fi', 'to'): 'Helsinki-NLP/opus-mt-fi-to', ('fi', 'toi'): 'Helsinki-NLP/opus-mt-fi-toi', ('fi', 'tpi'): 'Helsinki-NLP/opus-mt-fi-tpi', ('fi', 'tr'): 'Helsinki-NLP/opus-mt-fi-tr', ('fi', 'ts'): 'Helsinki-NLP/opus-mt-fi-ts', ('fi', 'tvl'): 'Helsinki-NLP/opus-mt-fi-tvl', ('fi', 'tw'): 'Helsinki-NLP/opus-mt-fi-tw', ('fi', 'ty'): 'Helsinki-NLP/opus-mt-fi-ty', ('fi', 'uk'): 'Helsinki-NLP/opus-mt-fi-uk', ('fi', 've'): 'Helsinki-NLP/opus-mt-fi-ve', ('fi', 'war'): 'Helsinki-NLP/opus-mt-fi-war', ('fi', 'wls'): 'Helsinki-NLP/opus-mt-fi-wls', ('fi', 'xh'): 'Helsinki-NLP/opus-mt-fi-xh', ('fi', 'yap'): 'Helsinki-NLP/opus-mt-fi-yap', ('fi', 'yo'): 'Helsinki-NLP/opus-mt-fi-yo', ('fi', 'zne'): 'Helsinki-NLP/opus-mt-fi-zne', ('fi_nb_no_nn_ru_sv_en', 'SAMI'): 'Helsinki-NLP/opus-mt-fi_nb_no_nn_ru_sv_en-SAMI', ('fiu', 'en'): 'Helsinki-NLP/opus-mt-fiu-en', ('fiu', 'fiu'): 'Helsinki-NLP/opus-mt-fiu-fiu', ('fj', 'en'): 'Helsinki-NLP/opus-mt-fj-en', ('fj', 'fr'): 'Helsinki-NLP/opus-mt-fj-fr', ('fr', 'af'): 'Helsinki-NLP/opus-mt-fr-af', ('fr', 'ar'): 'Helsinki-NLP/opus-mt-fr-ar', ('fr', 'ase'): 'Helsinki-NLP/opus-mt-fr-ase', ('fr', 'bcl'): 'Helsinki-NLP/opus-mt-fr-bcl', ('fr', 'bem'): 'Helsinki-NLP/opus-mt-fr-bem', ('fr', 'ber'): 'Helsinki-NLP/opus-mt-fr-ber', ('fr', 'bg'): 'Helsinki-NLP/opus-mt-fr-bg', ('fr', 'bi'): 'Helsinki-NLP/opus-mt-fr-bi', ('fr', 'bzs'): 'Helsinki-NLP/opus-mt-fr-bzs', ('fr', 'ca'): 'Helsinki-NLP/opus-mt-fr-ca', ('fr', 'ceb'): 'Helsinki-NLP/opus-mt-fr-ceb', ('fr', 'crs'): 'Helsinki-NLP/opus-mt-fr-crs', ('fr', 'de'): 'Helsinki-NLP/opus-mt-fr-de', ('fr', 'ee'): 'Helsinki-NLP/opus-mt-fr-ee', ('fr', 'efi'): 'Helsinki-NLP/opus-mt-fr-efi', ('fr', 'el'): 'Helsinki-NLP/opus-mt-fr-el', ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en', ('fr', 'eo'): 'Helsinki-NLP/opus-mt-fr-eo', ('fr', 'es'): 'Helsinki-NLP/opus-mt-fr-es', ('fr', 'fj'): 'Helsinki-NLP/opus-mt-fr-fj', ('fr', 'gaa'): 'Helsinki-NLP/opus-mt-fr-gaa', ('fr', 'gil'): 'Helsinki-NLP/opus-mt-fr-gil', ('fr', 'guw'): 'Helsinki-NLP/opus-mt-fr-guw', ('fr', 'ha'): 'Helsinki-NLP/opus-mt-fr-ha', ('fr', 'he'): 'Helsinki-NLP/opus-mt-fr-he', ('fr', 'hil'): 'Helsinki-NLP/opus-mt-fr-hil', ('fr', 'ho'): 'Helsinki-NLP/opus-mt-fr-ho', ('fr', 'hr'): 'Helsinki-NLP/opus-mt-fr-hr', ('fr', 'ht'): 'Helsinki-NLP/opus-mt-fr-ht', ('fr', 'hu'): 'Helsinki-NLP/opus-mt-fr-hu', ('fr', 'id'): 'Helsinki-NLP/opus-mt-fr-id', ('fr', 'ig'): 'Helsinki-NLP/opus-mt-fr-ig', ('fr', 'ilo'): 'Helsinki-NLP/opus-mt-fr-ilo', ('fr', 'iso'): 'Helsinki-NLP/opus-mt-fr-iso', ('fr', 'kg'): 'Helsinki-NLP/opus-mt-fr-kg', ('fr', 'kqn'): 'Helsinki-NLP/opus-mt-fr-kqn', ('fr', 'kwy'): 'Helsinki-NLP/opus-mt-fr-kwy', ('fr', 'lg'): 'Helsinki-NLP/opus-mt-fr-lg', ('fr', 'ln'): 'Helsinki-NLP/opus-mt-fr-ln', ('fr', 'loz'): 'Helsinki-NLP/opus-mt-fr-loz', ('fr', 'lu'): 'Helsinki-NLP/opus-mt-fr-lu', ('fr', 'lua'): 'Helsinki-NLP/opus-mt-fr-lua', ('fr', 'lue'): 'Helsinki-NLP/opus-mt-fr-lue', ('fr', 'lus'): 'Helsinki-NLP/opus-mt-fr-lus', ('fr', 'mfe'): 'Helsinki-NLP/opus-mt-fr-mfe', ('fr', 'mh'): 'Helsinki-NLP/opus-mt-fr-mh', ('fr', 'mos'): 'Helsinki-NLP/opus-mt-fr-mos', ('fr', 'ms'): 'Helsinki-NLP/opus-mt-fr-ms', ('fr', 'mt'): 'Helsinki-NLP/opus-mt-fr-mt', ('fr', 'niu'): 'Helsinki-NLP/opus-mt-fr-niu', ('fr', 'no'): 'Helsinki-NLP/opus-mt-fr-no', ('fr', 'nso'): 'Helsinki-NLP/opus-mt-fr-nso', ('fr', 'ny'): 'Helsinki-NLP/opus-mt-fr-ny', ('fr', 'pag'): 'Helsinki-NLP/opus-mt-fr-pag', ('fr', 'pap'): 'Helsinki-NLP/opus-mt-fr-pap', ('fr', 'pis'): 'Helsinki-NLP/opus-mt-fr-pis', ('fr', 'pl'): 'Helsinki-NLP/opus-mt-fr-pl', ('fr', 'pon'): 'Helsinki-NLP/opus-mt-fr-pon', ('fr', 'rnd'): 'Helsinki-NLP/opus-mt-fr-rnd', ('fr', 'ro'): 'Helsinki-NLP/opus-mt-fr-ro', ('fr', 'ru'): 'Helsinki-NLP/opus-mt-fr-ru', ('fr', 'run'): 'Helsinki-NLP/opus-mt-fr-run', ('fr', 'rw'): 'Helsinki-NLP/opus-mt-fr-rw', ('fr', 'sg'): 'Helsinki-NLP/opus-mt-fr-sg', ('fr', 'sk'): 'Helsinki-NLP/opus-mt-fr-sk', ('fr', 'sl'): 'Helsinki-NLP/opus-mt-fr-sl', ('fr', 'sm'): 'Helsinki-NLP/opus-mt-fr-sm', ('fr', 'sn'): 'Helsinki-NLP/opus-mt-fr-sn', ('fr', 'srn'): 'Helsinki-NLP/opus-mt-fr-srn', ('fr', 'st'): 'Helsinki-NLP/opus-mt-fr-st', ('fr', 'sv'): 'Helsinki-NLP/opus-mt-fr-sv', ('fr', 'swc'): 'Helsinki-NLP/opus-mt-fr-swc', ('fr', 'tiv'): 'Helsinki-NLP/opus-mt-fr-tiv', ('fr', 'tl'): 'Helsinki-NLP/opus-mt-fr-tl', ('fr', 'tll'): 'Helsinki-NLP/opus-mt-fr-tll', ('fr', 'tn'): 'Helsinki-NLP/opus-mt-fr-tn', ('fr', 'to'): 'Helsinki-NLP/opus-mt-fr-to', ('fr', 'tpi'): 'Helsinki-NLP/opus-mt-fr-tpi', ('fr', 'ts'): 'Helsinki-NLP/opus-mt-fr-ts', ('fr', 'tum'): 'Helsinki-NLP/opus-mt-fr-tum', ('fr', 'tvl'): 'Helsinki-NLP/opus-mt-fr-tvl', ('fr', 'tw'): 'Helsinki-NLP/opus-mt-fr-tw', ('fr', 'ty'): 'Helsinki-NLP/opus-mt-fr-ty', ('fr', 'uk'): 'Helsinki-NLP/opus-mt-fr-uk', ('fr', 've'): 'Helsinki-NLP/opus-mt-fr-ve', ('fr', 'vi'): 'Helsinki-NLP/opus-mt-fr-vi', ('fr', 'war'): 'Helsinki-NLP/opus-mt-fr-war', ('fr', 'wls'): 'Helsinki-NLP/opus-mt-fr-wls', ('fr', 'xh'): 'Helsinki-NLP/opus-mt-fr-xh', ('fr', 'yap'): 'Helsinki-NLP/opus-mt-fr-yap', ('fr', 'yo'): 'Helsinki-NLP/opus-mt-fr-yo', ('fr', 'zne'): 'Helsinki-NLP/opus-mt-fr-zne', ('fse', 'fi'): 'Helsinki-NLP/opus-mt-fse-fi', ('ga', 'en'): 'Helsinki-NLP/opus-mt-ga-en', ('gaa', 'de'): 'Helsinki-NLP/opus-mt-gaa-de', ('gaa', 'en'): 'Helsinki-NLP/opus-mt-gaa-en', ('gaa', 'es'): 'Helsinki-NLP/opus-mt-gaa-es', ('gaa', 'fi'): 'Helsinki-NLP/opus-mt-gaa-fi', ('gaa', 'fr'): 'Helsinki-NLP/opus-mt-gaa-fr', ('gaa', 'sv'): 'Helsinki-NLP/opus-mt-gaa-sv', ('gem', 'en'): 'Helsinki-NLP/opus-mt-gem-en', ('gem', 'gem'): 'Helsinki-NLP/opus-mt-gem-gem', ('gil', 'en'): 'Helsinki-NLP/opus-mt-gil-en', ('gil', 'es'): 'Helsinki-NLP/opus-mt-gil-es', ('gil', 'fi'): 'Helsinki-NLP/opus-mt-gil-fi', ('gil', 'fr'): 'Helsinki-NLP/opus-mt-gil-fr', ('gil', 'sv'): 'Helsinki-NLP/opus-mt-gil-sv', ('gl', 'en'): 'Helsinki-NLP/opus-mt-gl-en', ('gl', 'es'): 'Helsinki-NLP/opus-mt-gl-es', ('gl', 'pt'): 'Helsinki-NLP/opus-mt-gl-pt', ('gmq', 'en'): 'Helsinki-NLP/opus-mt-gmq-en', ('gmq', 'gmq'): 'Helsinki-NLP/opus-mt-gmq-gmq', ('gmw', 'en'): 'Helsinki-NLP/opus-mt-gmw-en', ('gmw', 'gmw'): 'Helsinki-NLP/opus-mt-gmw-gmw', ('grk', 'en'): 'Helsinki-NLP/opus-mt-grk-en', ('guw', 'de'): 'Helsinki-NLP/opus-mt-guw-de', ('guw', 'en'): 'Helsinki-NLP/opus-mt-guw-en', ('guw', 'es'): 'Helsinki-NLP/opus-mt-guw-es', ('guw', 'fi'): 'Helsinki-NLP/opus-mt-guw-fi', ('guw', 'fr'): 'Helsinki-NLP/opus-mt-guw-fr', ('guw', 'sv'): 'Helsinki-NLP/opus-mt-guw-sv', ('gv', 'en'): 'Helsinki-NLP/opus-mt-gv-en', ('ha', 'en'): 'Helsinki-NLP/opus-mt-ha-en', ('ha', 'es'): 'Helsinki-NLP/opus-mt-ha-es', ('ha', 'fi'): 'Helsinki-NLP/opus-mt-ha-fi', ('ha', 'fr'): 'Helsinki-NLP/opus-mt-ha-fr', ('ha', 'sv'): 'Helsinki-NLP/opus-mt-ha-sv', ('he', 'ar'): 'Helsinki-NLP/opus-mt-he-ar', ('he', 'de'): 'Helsinki-NLP/opus-mt-he-de', ('he', 'eo'): 'Helsinki-NLP/opus-mt-he-eo', ('he', 'es'): 'Helsinki-NLP/opus-mt-he-es', ('he', 'fi'): 'Helsinki-NLP/opus-mt-he-fi', ('he', 'fr'): 'Helsinki-NLP/opus-mt-he-fr', ('he', 'it'): 'Helsinki-NLP/opus-mt-he-it', ('he', 'ru'): 'Helsinki-NLP/opus-mt-he-ru', ('he', 'sv'): 'Helsinki-NLP/opus-mt-he-sv', ('he', 'uk'): 'Helsinki-NLP/opus-mt-he-uk', ('hi', 'en'): 'Helsinki-NLP/opus-mt-hi-en', ('hi', 'ur'): 'Helsinki-NLP/opus-mt-hi-ur', ('hil', 'de'): 'Helsinki-NLP/opus-mt-hil-de', ('hil', 'en'): 'Helsinki-NLP/opus-mt-hil-en', ('hil', 'fi'): 'Helsinki-NLP/opus-mt-hil-fi', ('ho', 'en'): 'Helsinki-NLP/opus-mt-ho-en', ('hr', 'es'): 'Helsinki-NLP/opus-mt-hr-es', ('hr', 'fi'): 'Helsinki-NLP/opus-mt-hr-fi', ('hr', 'fr'): 'Helsinki-NLP/opus-mt-hr-fr', ('hr', 'sv'): 'Helsinki-NLP/opus-mt-hr-sv', ('ht', 'en'): 'Helsinki-NLP/opus-mt-ht-en', ('ht', 'es'): 'Helsinki-NLP/opus-mt-ht-es', ('ht', 'fi'): 'Helsinki-NLP/opus-mt-ht-fi', ('ht', 'fr'): 'Helsinki-NLP/opus-mt-ht-fr', ('ht', 'sv'): 'Helsinki-NLP/opus-mt-ht-sv', ('hu', 'de'): 'Helsinki-NLP/opus-mt-hu-de', ('hu', 'en'): 'Helsinki-NLP/opus-mt-hu-en', ('hu', 'eo'): 'Helsinki-NLP/opus-mt-hu-eo', ('hu', 'fi'): 'Helsinki-NLP/opus-mt-hu-fi', ('hu', 'fr'): 'Helsinki-NLP/opus-mt-hu-fr', ('hu', 'sv'): 'Helsinki-NLP/opus-mt-hu-sv', ('hu', 'uk'): 'Helsinki-NLP/opus-mt-hu-uk', ('hy', 'en'): 'Helsinki-NLP/opus-mt-hy-en', ('hy', 'ru'): 'Helsinki-NLP/opus-mt-hy-ru', ('id', 'en'): 'Helsinki-NLP/opus-mt-id-en', ('id', 'es'): 'Helsinki-NLP/opus-mt-id-es', ('id', 'fi'): 'Helsinki-NLP/opus-mt-id-fi', ('id', 'fr'): 'Helsinki-NLP/opus-mt-id-fr', ('id', 'sv'): 'Helsinki-NLP/opus-mt-id-sv', ('ig', 'de'): 'Helsinki-NLP/opus-mt-ig-de', ('ig', 'en'): 'Helsinki-NLP/opus-mt-ig-en', ('ig', 'es'): 'Helsinki-NLP/opus-mt-ig-es', ('ig', 'fi'): 'Helsinki-NLP/opus-mt-ig-fi', ('ig', 'fr'): 'Helsinki-NLP/opus-mt-ig-fr', ('ig', 'sv'): 'Helsinki-NLP/opus-mt-ig-sv', ('iir', 'en'): 'Helsinki-NLP/opus-mt-iir-en', ('iir', 'iir'): 'Helsinki-NLP/opus-mt-iir-iir', ('ilo', 'de'): 'Helsinki-NLP/opus-mt-ilo-de', ('ilo', 'en'): 'Helsinki-NLP/opus-mt-ilo-en', ('ilo', 'es'): 'Helsinki-NLP/opus-mt-ilo-es', ('ilo', 'fi'): 'Helsinki-NLP/opus-mt-ilo-fi', ('ilo', 'sv'): 'Helsinki-NLP/opus-mt-ilo-sv', ('inc', 'en'): 'Helsinki-NLP/opus-mt-inc-en', ('inc', 'inc'): 'Helsinki-NLP/opus-mt-inc-inc', ('ine', 'en'): 'Helsinki-NLP/opus-mt-ine-en', ('ine', 'ine'): 'Helsinki-NLP/opus-mt-ine-ine', ('is', 'de'): 'Helsinki-NLP/opus-mt-is-de', ('is', 'en'): 'Helsinki-NLP/opus-mt-is-en', ('is', 'eo'): 'Helsinki-NLP/opus-mt-is-eo', ('is', 'es'): 'Helsinki-NLP/opus-mt-is-es', ('is', 'fi'): 'Helsinki-NLP/opus-mt-is-fi', ('is', 'fr'): 'Helsinki-NLP/opus-mt-is-fr', ('is', 'it'): 'Helsinki-NLP/opus-mt-is-it', ('is', 'sv'): 'Helsinki-NLP/opus-mt-is-sv', ('iso', 'en'): 'Helsinki-NLP/opus-mt-iso-en', ('iso', 'es'): 'Helsinki-NLP/opus-mt-iso-es', ('iso', 'fi'): 'Helsinki-NLP/opus-mt-iso-fi', ('iso', 'fr'): 'Helsinki-NLP/opus-mt-iso-fr', ('iso', 'sv'): 'Helsinki-NLP/opus-mt-iso-sv', ('it', 'ar'): 'Helsinki-NLP/opus-mt-it-ar', ('it', 'bg'): 'Helsinki-NLP/opus-mt-it-bg', ('it', 'ca'): 'Helsinki-NLP/opus-mt-it-ca', ('it', 'de'): 'Helsinki-NLP/opus-mt-it-de', ('it', 'en'): 'Helsinki-NLP/opus-mt-it-en', ('it', 'eo'): 'Helsinki-NLP/opus-mt-it-eo', ('it', 'es'): 'Helsinki-NLP/opus-mt-it-es', ('it', 'fr'): 'Helsinki-NLP/opus-mt-it-fr', ('it', 'is'): 'Helsinki-NLP/opus-mt-it-is', ('it', 'lt'): 'Helsinki-NLP/opus-mt-it-lt', ('it', 'ms'): 'Helsinki-NLP/opus-mt-it-ms', ('it', 'sv'): 'Helsinki-NLP/opus-mt-it-sv', ('it', 'uk'): 'Helsinki-NLP/opus-mt-it-uk', ('it', 'vi'): 'Helsinki-NLP/opus-mt-it-vi', ('itc', 'en'): 'Helsinki-NLP/opus-mt-itc-en', ('itc', 'itc'): 'Helsinki-NLP/opus-mt-itc-itc', ('ja', 'ar'): 'Helsinki-NLP/opus-mt-ja-ar', ('ja', 'bg'): 'Helsinki-NLP/opus-mt-ja-bg', ('ja', 'da'): 'Helsinki-NLP/opus-mt-ja-da', ('ja', 'de'): 'Helsinki-NLP/opus-mt-ja-de', ('ja', 'en'): 'Helsinki-NLP/opus-mt-ja-en', ('ja', 'es'): 'Helsinki-NLP/opus-mt-ja-es', ('ja', 'fi'): 'Helsinki-NLP/opus-mt-ja-fi', ('ja', 'fr'): 'Helsinki-NLP/opus-mt-ja-fr', ('ja', 'he'): 'Helsinki-NLP/opus-mt-ja-he', ('ja', 'hu'): 'Helsinki-NLP/opus-mt-ja-hu', ('ja', 'it'): 'Helsinki-NLP/opus-mt-ja-it', ('ja', 'ms'): 'Helsinki-NLP/opus-mt-ja-ms', ('ja', 'nl'): 'Helsinki-NLP/opus-mt-ja-nl', ('ja', 'pl'): 'Helsinki-NLP/opus-mt-ja-pl', ('ja', 'pt'): 'Helsinki-NLP/opus-mt-ja-pt', ('ja', 'ru'): 'Helsinki-NLP/opus-mt-ja-ru', ('ja', 'sh'): 'Helsinki-NLP/opus-mt-ja-sh', ('ja', 'sv'): 'Helsinki-NLP/opus-mt-ja-sv', ('ja', 'tr'): 'Helsinki-NLP/opus-mt-ja-tr', ('ja', 'vi'): 'Helsinki-NLP/opus-mt-ja-vi', ('jap', 'en'): 'Helsinki-NLP/opus-mt-jap-en', ('ka', 'en'): 'Helsinki-NLP/opus-mt-ka-en', ('ka', 'ru'): 'Helsinki-NLP/opus-mt-ka-ru', ('kab', 'en'): 'Helsinki-NLP/opus-mt-kab-en', ('kg', 'en'): 'Helsinki-NLP/opus-mt-kg-en', ('kg', 'es'): 'Helsinki-NLP/opus-mt-kg-es', ('kg', 'fr'): 'Helsinki-NLP/opus-mt-kg-fr', ('kg', 'sv'): 'Helsinki-NLP/opus-mt-kg-sv', ('kj', 'en'): 'Helsinki-NLP/opus-mt-kj-en', ('kl', 'en'): 'Helsinki-NLP/opus-mt-kl-en', ('ko', 'de'): 'Helsinki-NLP/opus-mt-ko-de', ('ko', 'en'): 'Helsinki-NLP/opus-mt-ko-en', ('ko', 'es'): 'Helsinki-NLP/opus-mt-ko-es', ('ko', 'fi'): 'Helsinki-NLP/opus-mt-ko-fi', ('ko', 'fr'): 'Helsinki-NLP/opus-mt-ko-fr', ('ko', 'hu'): 'Helsinki-NLP/opus-mt-ko-hu', ('ko', 'ru'): 'Helsinki-NLP/opus-mt-ko-ru', ('ko', 'sv'): 'Helsinki-NLP/opus-mt-ko-sv', ('kqn', 'en'): 'Helsinki-NLP/opus-mt-kqn-en', ('kqn', 'es'): 'Helsinki-NLP/opus-mt-kqn-es', ('kqn', 'fr'): 'Helsinki-NLP/opus-mt-kqn-fr', ('kqn', 'sv'): 'Helsinki-NLP/opus-mt-kqn-sv', ('kwn', 'en'): 'Helsinki-NLP/opus-mt-kwn-en', ('kwy', 'en'): 'Helsinki-NLP/opus-mt-kwy-en', ('kwy', 'fr'): 'Helsinki-NLP/opus-mt-kwy-fr', ('kwy', 'sv'): 'Helsinki-NLP/opus-mt-kwy-sv', ('lg', 'en'): 'Helsinki-NLP/opus-mt-lg-en', ('lg', 'es'): 'Helsinki-NLP/opus-mt-lg-es', ('lg', 'fi'): 'Helsinki-NLP/opus-mt-lg-fi', ('lg', 'fr'): 'Helsinki-NLP/opus-mt-lg-fr', ('lg', 'sv'): 'Helsinki-NLP/opus-mt-lg-sv', ('ln', 'de'): 'Helsinki-NLP/opus-mt-ln-de', ('ln', 'en'): 'Helsinki-NLP/opus-mt-ln-en', ('ln', 'es'): 'Helsinki-NLP/opus-mt-ln-es', ('ln', 'fr'): 'Helsinki-NLP/opus-mt-ln-fr', ('loz', 'de'): 'Helsinki-NLP/opus-mt-loz-de', ('loz', 'en'): 'Helsinki-NLP/opus-mt-loz-en', ('loz', 'es'): 'Helsinki-NLP/opus-mt-loz-es', ('loz', 'fi'): 'Helsinki-NLP/opus-mt-loz-fi', ('loz', 'fr'): 'Helsinki-NLP/opus-mt-loz-fr', ('loz', 'sv'): 'Helsinki-NLP/opus-mt-loz-sv', ('lt', 'de'): 'Helsinki-NLP/opus-mt-lt-de', ('lt', 'eo'): 'Helsinki-NLP/opus-mt-lt-eo', ('lt', 'es'): 'Helsinki-NLP/opus-mt-lt-es', ('lt', 'fr'): 'Helsinki-NLP/opus-mt-lt-fr', ('lt', 'it'): 'Helsinki-NLP/opus-mt-lt-it', ('lt', 'pl'): 'Helsinki-NLP/opus-mt-lt-pl', ('lt', 'ru'): 'Helsinki-NLP/opus-mt-lt-ru', ('lt', 'sv'): 'Helsinki-NLP/opus-mt-lt-sv', ('lt', 'tr'): 'Helsinki-NLP/opus-mt-lt-tr', ('lu', 'en'): 'Helsinki-NLP/opus-mt-lu-en', ('lu', 'es'): 'Helsinki-NLP/opus-mt-lu-es', ('lu', 'fi'): 'Helsinki-NLP/opus-mt-lu-fi', ('lu', 'fr'): 'Helsinki-NLP/opus-mt-lu-fr', ('lu', 'sv'): 'Helsinki-NLP/opus-mt-lu-sv', ('lua', 'en'): 'Helsinki-NLP/opus-mt-lua-en', ('lua', 'es'): 'Helsinki-NLP/opus-mt-lua-es', ('lua', 'fi'): 'Helsinki-NLP/opus-mt-lua-fi', ('lua', 'fr'): 'Helsinki-NLP/opus-mt-lua-fr', ('lua', 'sv'): 'Helsinki-NLP/opus-mt-lua-sv', ('lue', 'en'): 'Helsinki-NLP/opus-mt-lue-en', ('lue', 'es'): 'Helsinki-NLP/opus-mt-lue-es', ('lue', 'fi'): 'Helsinki-NLP/opus-mt-lue-fi', ('lue', 'fr'): 'Helsinki-NLP/opus-mt-lue-fr', ('lue', 'sv'): 'Helsinki-NLP/opus-mt-lue-sv', ('lun', 'en'): 'Helsinki-NLP/opus-mt-lun-en', ('luo', 'en'): 'Helsinki-NLP/opus-mt-luo-en', ('lus', 'en'): 'Helsinki-NLP/opus-mt-lus-en', ('lus', 'es'): 'Helsinki-NLP/opus-mt-lus-es', ('lus', 'fi'): 'Helsinki-NLP/opus-mt-lus-fi', ('lus', 'fr'): 'Helsinki-NLP/opus-mt-lus-fr', ('lus', 'sv'): 'Helsinki-NLP/opus-mt-lus-sv', ('lv', 'en'): 'Helsinki-NLP/opus-mt-lv-en', ('lv', 'es'): 'Helsinki-NLP/opus-mt-lv-es', ('lv', 'fi'): 'Helsinki-NLP/opus-mt-lv-fi', ('lv', 'fr'): 'Helsinki-NLP/opus-mt-lv-fr', ('lv', 'ru'): 'Helsinki-NLP/opus-mt-lv-ru', ('lv', 'sv'): 'Helsinki-NLP/opus-mt-lv-sv', ('mfe', 'en'): 'Helsinki-NLP/opus-mt-mfe-en', ('mfe', 'es'): 'Helsinki-NLP/opus-mt-mfe-es', ('mfs', 'es'): 'Helsinki-NLP/opus-mt-mfs-es', ('mg', 'en'): 'Helsinki-NLP/opus-mt-mg-en', ('mg', 'es'): 'Helsinki-NLP/opus-mt-mg-es', ('mh', 'en'): 'Helsinki-NLP/opus-mt-mh-en', ('mh', 'es'): 'Helsinki-NLP/opus-mt-mh-es', ('mh', 'fi'): 'Helsinki-NLP/opus-mt-mh-fi', ('mk', 'en'): 'Helsinki-NLP/opus-mt-mk-en', ('mk', 'es'): 'Helsinki-NLP/opus-mt-mk-es', ('mk', 'fi'): 'Helsinki-NLP/opus-mt-mk-fi', ('mk', 'fr'): 'Helsinki-NLP/opus-mt-mk-fr', ('mkh', 'en'): 'Helsinki-NLP/opus-mt-mkh-en', ('ml', 'en'): 'Helsinki-NLP/opus-mt-ml-en', ('mos', 'en'): 'Helsinki-NLP/opus-mt-mos-en', ('mr', 'en'): 'Helsinki-NLP/opus-mt-mr-en', ('ms', 'de'): 'Helsinki-NLP/opus-mt-ms-de', ('ms', 'fr'): 'Helsinki-NLP/opus-mt-ms-fr', ('ms', 'it'): 'Helsinki-NLP/opus-mt-ms-it', ('ms', 'ms'): 'Helsinki-NLP/opus-mt-ms-ms', ('mt', 'en'): 'Helsinki-NLP/opus-mt-mt-en', ('mt', 'es'): 'Helsinki-NLP/opus-mt-mt-es', ('mt', 'fi'): 'Helsinki-NLP/opus-mt-mt-fi', ('mt', 'fr'): 'Helsinki-NLP/opus-mt-mt-fr', ('mt', 'sv'): 'Helsinki-NLP/opus-mt-mt-sv', ('mul', 'en'): 'Helsinki-NLP/opus-mt-mul-en', ('ng', 'en'): 'Helsinki-NLP/opus-mt-ng-en', ('nic', 'en'): 'Helsinki-NLP/opus-mt-nic-en', ('niu', 'de'): 'Helsinki-NLP/opus-mt-niu-de', ('niu', 'en'): 'Helsinki-NLP/opus-mt-niu-en', ('niu', 'es'): 'Helsinki-NLP/opus-mt-niu-es', ('niu', 'fi'): 'Helsinki-NLP/opus-mt-niu-fi', ('niu', 'fr'): 'Helsinki-NLP/opus-mt-niu-fr', ('niu', 'sv'): 'Helsinki-NLP/opus-mt-niu-sv', ('nl', 'af'): 'Helsinki-NLP/opus-mt-nl-af', ('nl', 'ca'): 'Helsinki-NLP/opus-mt-nl-ca', ('nl', 'en'): 'Helsinki-NLP/opus-mt-nl-en', ('nl', 'eo'): 'Helsinki-NLP/opus-mt-nl-eo', ('nl', 'es'): 'Helsinki-NLP/opus-mt-nl-es', ('nl', 'fi'): 'Helsinki-NLP/opus-mt-nl-fi', ('nl', 'fr'): 'Helsinki-NLP/opus-mt-nl-fr', ('nl', 'no'): 'Helsinki-NLP/opus-mt-nl-no', ('nl', 'sv'): 'Helsinki-NLP/opus-mt-nl-sv', ('nl', 'uk'): 'Helsinki-NLP/opus-mt-nl-uk', ('no', 'da'): 'Helsinki-NLP/opus-mt-no-da', ('no', 'de'): 'Helsinki-NLP/opus-mt-no-de', ('no', 'es'): 'Helsinki-NLP/opus-mt-no-es', ('no', 'fi'): 'Helsinki-NLP/opus-mt-no-fi', ('no', 'fr'): 'Helsinki-NLP/opus-mt-no-fr', ('no', 'nl'): 'Helsinki-NLP/opus-mt-no-nl', ('no', 'no'): 'Helsinki-NLP/opus-mt-no-no', ('no', 'pl'): 'Helsinki-NLP/opus-mt-no-pl', ('no', 'ru'): 'Helsinki-NLP/opus-mt-no-ru', ('no', 'sv'): 'Helsinki-NLP/opus-mt-no-sv', ('no', 'uk'): 'Helsinki-NLP/opus-mt-no-uk', ('nso', 'de'): 'Helsinki-NLP/opus-mt-nso-de', ('nso', 'en'): 'Helsinki-NLP/opus-mt-nso-en', ('nso', 'es'): 'Helsinki-NLP/opus-mt-nso-es', ('nso', 'fi'): 'Helsinki-NLP/opus-mt-nso-fi', ('nso', 'fr'): 'Helsinki-NLP/opus-mt-nso-fr', ('nso', 'sv'): 'Helsinki-NLP/opus-mt-nso-sv', ('ny', 'de'): 'Helsinki-NLP/opus-mt-ny-de', ('ny', 'en'): 'Helsinki-NLP/opus-mt-ny-en', ('ny', 'es'): 'Helsinki-NLP/opus-mt-ny-es', ('nyk', 'en'): 'Helsinki-NLP/opus-mt-nyk-en', ('om', 'en'): 'Helsinki-NLP/opus-mt-om-en', ('pa', 'en'): 'Helsinki-NLP/opus-mt-pa-en', ('pag', 'de'): 'Helsinki-NLP/opus-mt-pag-de', ('pag', 'en'): 'Helsinki-NLP/opus-mt-pag-en', ('pag', 'es'): 'Helsinki-NLP/opus-mt-pag-es', ('pag', 'fi'): 'Helsinki-NLP/opus-mt-pag-fi', ('pag', 'sv'): 'Helsinki-NLP/opus-mt-pag-sv', ('pap', 'de'): 'Helsinki-NLP/opus-mt-pap-de', ('pap', 'en'): 'Helsinki-NLP/opus-mt-pap-en', ('pap', 'es'): 'Helsinki-NLP/opus-mt-pap-es', ('pap', 'fi'): 'Helsinki-NLP/opus-mt-pap-fi', ('pap', 'fr'): 'Helsinki-NLP/opus-mt-pap-fr', ('phi', 'en'): 'Helsinki-NLP/opus-mt-phi-en', ('pis', 'en'): 'Helsinki-NLP/opus-mt-pis-en', ('pis', 'es'): 'Helsinki-NLP/opus-mt-pis-es', ('pis', 'fi'): 'Helsinki-NLP/opus-mt-pis-fi', ('pis', 'fr'): 'Helsinki-NLP/opus-mt-pis-fr', ('pis', 'sv'): 'Helsinki-NLP/opus-mt-pis-sv', ('pl', 'ar'): 'Helsinki-NLP/opus-mt-pl-ar', ('pl', 'de'): 'Helsinki-NLP/opus-mt-pl-de', ('pl', 'en'): 'Helsinki-NLP/opus-mt-pl-en', ('pl', 'eo'): 'Helsinki-NLP/opus-mt-pl-eo', ('pl', 'es'): 'Helsinki-NLP/opus-mt-pl-es', ('pl', 'fr'): 'Helsinki-NLP/opus-mt-pl-fr', ('pl', 'lt'): 'Helsinki-NLP/opus-mt-pl-lt', ('pl', 'no'): 'Helsinki-NLP/opus-mt-pl-no', ('pl', 'sv'): 'Helsinki-NLP/opus-mt-pl-sv', ('pl', 'uk'): 'Helsinki-NLP/opus-mt-pl-uk', ('pon', 'en'): 'Helsinki-NLP/opus-mt-pon-en', ('pon', 'es'): 'Helsinki-NLP/opus-mt-pon-es', ('pon', 'fi'): 'Helsinki-NLP/opus-mt-pon-fi', ('pon', 'fr'): 'Helsinki-NLP/opus-mt-pon-fr', ('pon', 'sv'): 'Helsinki-NLP/opus-mt-pon-sv', ('pqe', 'en'): 'Helsinki-NLP/opus-mt-pqe-en', ('prl', 'es'): 'Helsinki-NLP/opus-mt-prl-es', ('pt', 'ca'): 'Helsinki-NLP/opus-mt-pt-ca', ('pt', 'eo'): 'Helsinki-NLP/opus-mt-pt-eo', ('pt', 'gl'): 'Helsinki-NLP/opus-mt-pt-gl', ('pt', 'tl'): 'Helsinki-NLP/opus-mt-pt-tl', ('pt', 'uk'): 'Helsinki-NLP/opus-mt-pt-uk', ('rn', 'de'): 'Helsinki-NLP/opus-mt-rn-de', ('rn', 'en'): 'Helsinki-NLP/opus-mt-rn-en', ('rn', 'es'): 'Helsinki-NLP/opus-mt-rn-es', ('rn', 'fr'): 'Helsinki-NLP/opus-mt-rn-fr', ('rn', 'ru'): 'Helsinki-NLP/opus-mt-rn-ru', ('rnd', 'en'): 'Helsinki-NLP/opus-mt-rnd-en', ('rnd', 'fr'): 'Helsinki-NLP/opus-mt-rnd-fr', ('rnd', 'sv'): 'Helsinki-NLP/opus-mt-rnd-sv', ('ro', 'eo'): 'Helsinki-NLP/opus-mt-ro-eo', ('ro', 'fi'): 'Helsinki-NLP/opus-mt-ro-fi', ('ro', 'fr'): 'Helsinki-NLP/opus-mt-ro-fr', ('ro', 'sv'): 'Helsinki-NLP/opus-mt-ro-sv', ('roa', 'en'): 'Helsinki-NLP/opus-mt-roa-en', ('ru', 'af'): 'Helsinki-NLP/opus-mt-ru-af', ('ru', 'ar'): 'Helsinki-NLP/opus-mt-ru-ar', ('ru', 'bg'): 'Helsinki-NLP/opus-mt-ru-bg', ('ru', 'da'): 'Helsinki-NLP/opus-mt-ru-da', ('ru', 'en'): 'Helsinki-NLP/opus-mt-ru-en', ('ru', 'eo'): 'Helsinki-NLP/opus-mt-ru-eo', ('ru', 'es'): 'Helsinki-NLP/opus-mt-ru-es', ('ru', 'et'): 'Helsinki-NLP/opus-mt-ru-et', ('ru', 'eu'): 'Helsinki-NLP/opus-mt-ru-eu', ('ru', 'fi'): 'Helsinki-NLP/opus-mt-ru-fi', ('ru', 'fr'): 'Helsinki-NLP/opus-mt-ru-fr', ('ru', 'he'): 'Helsinki-NLP/opus-mt-ru-he', ('ru', 'hy'): 'Helsinki-NLP/opus-mt-ru-hy', ('ru', 'lt'): 'Helsinki-NLP/opus-mt-ru-lt', ('ru', 'lv'): 'Helsinki-NLP/opus-mt-ru-lv', ('ru', 'no'): 'Helsinki-NLP/opus-mt-ru-no', ('ru', 'sl'): 'Helsinki-NLP/opus-mt-ru-sl', ('ru', 'sv'): 'Helsinki-NLP/opus-mt-ru-sv', ('ru', 'uk'): 'Helsinki-NLP/opus-mt-ru-uk', ('ru', 'vi'): 'Helsinki-NLP/opus-mt-ru-vi', ('run', 'en'): 'Helsinki-NLP/opus-mt-run-en', ('run', 'es'): 'Helsinki-NLP/opus-mt-run-es', ('run', 'sv'): 'Helsinki-NLP/opus-mt-run-sv', ('rw', 'en'): 'Helsinki-NLP/opus-mt-rw-en', ('rw', 'es'): 'Helsinki-NLP/opus-mt-rw-es', ('rw', 'fr'): 'Helsinki-NLP/opus-mt-rw-fr', ('rw', 'sv'): 'Helsinki-NLP/opus-mt-rw-sv', ('sal', 'en'): 'Helsinki-NLP/opus-mt-sal-en', ('sem', 'en'): 'Helsinki-NLP/opus-mt-sem-en', ('sem', 'sem'): 'Helsinki-NLP/opus-mt-sem-sem', ('sg', 'en'): 'Helsinki-NLP/opus-mt-sg-en', ('sg', 'es'): 'Helsinki-NLP/opus-mt-sg-es', ('sg', 'fi'): 'Helsinki-NLP/opus-mt-sg-fi', ('sg', 'fr'): 'Helsinki-NLP/opus-mt-sg-fr', ('sg', 'sv'): 'Helsinki-NLP/opus-mt-sg-sv', ('sh', 'eo'): 'Helsinki-NLP/opus-mt-sh-eo', ('sh', 'uk'): 'Helsinki-NLP/opus-mt-sh-uk', ('sk', 'en'): 'Helsinki-NLP/opus-mt-sk-en', ('sk', 'es'): 'Helsinki-NLP/opus-mt-sk-es', ('sk', 'fi'): 'Helsinki-NLP/opus-mt-sk-fi', ('sk', 'fr'): 'Helsinki-NLP/opus-mt-sk-fr', ('sk', 'sv'): 'Helsinki-NLP/opus-mt-sk-sv', ('sl', 'es'): 'Helsinki-NLP/opus-mt-sl-es', ('sl', 'fi'): 'Helsinki-NLP/opus-mt-sl-fi', ('sl', 'fr'): 'Helsinki-NLP/opus-mt-sl-fr', ('sl', 'ru'): 'Helsinki-NLP/opus-mt-sl-ru', ('sl', 'sv'): 'Helsinki-NLP/opus-mt-sl-sv', ('sl', 'uk'): 'Helsinki-NLP/opus-mt-sl-uk', ('sla', 'en'): 'Helsinki-NLP/opus-mt-sla-en', ('sla', 'sla'): 'Helsinki-NLP/opus-mt-sla-sla', ('sm', 'en'): 'Helsinki-NLP/opus-mt-sm-en', ('sm', 'es'): 'Helsinki-NLP/opus-mt-sm-es', ('sm', 'fr'): 'Helsinki-NLP/opus-mt-sm-fr', ('sn', 'en'): 'Helsinki-NLP/opus-mt-sn-en', ('sn', 'es'): 'Helsinki-NLP/opus-mt-sn-es', ('sn', 'fr'): 'Helsinki-NLP/opus-mt-sn-fr', ('sn', 'sv'): 'Helsinki-NLP/opus-mt-sn-sv', ('sq', 'en'): 'Helsinki-NLP/opus-mt-sq-en', ('sq', 'es'): 'Helsinki-NLP/opus-mt-sq-es', ('sq', 'sv'): 'Helsinki-NLP/opus-mt-sq-sv', ('srn', 'en'): 'Helsinki-NLP/opus-mt-srn-en', ('srn', 'es'): 'Helsinki-NLP/opus-mt-srn-es', ('srn', 'fr'): 'Helsinki-NLP/opus-mt-srn-fr', ('srn', 'sv'): 'Helsinki-NLP/opus-mt-srn-sv', ('ss', 'en'): 'Helsinki-NLP/opus-mt-ss-en', ('ssp', 'es'): 'Helsinki-NLP/opus-mt-ssp-es', ('st', 'en'): 'Helsinki-NLP/opus-mt-st-en', ('st', 'es'): 'Helsinki-NLP/opus-mt-st-es', ('st', 'fi'): 'Helsinki-NLP/opus-mt-st-fi', ('st', 'fr'): 'Helsinki-NLP/opus-mt-st-fr', ('st', 'sv'): 'Helsinki-NLP/opus-mt-st-sv', ('sv', 'NORWAY'): 'Helsinki-NLP/opus-mt-sv-NORWAY', ('sv', 'ZH'): 'Helsinki-NLP/opus-mt-sv-ZH', ('sv', 'af'): 'Helsinki-NLP/opus-mt-sv-af', ('sv', 'ase'): 'Helsinki-NLP/opus-mt-sv-ase', ('sv', 'bcl'): 'Helsinki-NLP/opus-mt-sv-bcl', ('sv', 'bem'): 'Helsinki-NLP/opus-mt-sv-bem', ('sv', 'bg'): 'Helsinki-NLP/opus-mt-sv-bg', ('sv', 'bi'): 'Helsinki-NLP/opus-mt-sv-bi', ('sv', 'bzs'): 'Helsinki-NLP/opus-mt-sv-bzs', ('sv', 'ceb'): 'Helsinki-NLP/opus-mt-sv-ceb', ('sv', 'chk'): 'Helsinki-NLP/opus-mt-sv-chk', ('sv', 'crs'): 'Helsinki-NLP/opus-mt-sv-crs', ('sv', 'cs'): 'Helsinki-NLP/opus-mt-sv-cs', ('sv', 'ee'): 'Helsinki-NLP/opus-mt-sv-ee', ('sv', 'efi'): 'Helsinki-NLP/opus-mt-sv-efi', ('sv', 'el'): 'Helsinki-NLP/opus-mt-sv-el', ('sv', 'en'): 'Helsinki-NLP/opus-mt-sv-en', ('sv', 'eo'): 'Helsinki-NLP/opus-mt-sv-eo', ('sv', 'es'): 'Helsinki-NLP/opus-mt-sv-es', ('sv', 'et'): 'Helsinki-NLP/opus-mt-sv-et', ('sv', 'fi'): 'Helsinki-NLP/opus-mt-sv-fi', ('sv', 'fj'): 'Helsinki-NLP/opus-mt-sv-fj', ('sv', 'fr'): 'Helsinki-NLP/opus-mt-sv-fr', ('sv', 'gaa'): 'Helsinki-NLP/opus-mt-sv-gaa', ('sv', 'gil'): 'Helsinki-NLP/opus-mt-sv-gil', ('sv', 'guw'): 'Helsinki-NLP/opus-mt-sv-guw', ('sv', 'ha'): 'Helsinki-NLP/opus-mt-sv-ha', ('sv', 'he'): 'Helsinki-NLP/opus-mt-sv-he', ('sv', 'hil'): 'Helsinki-NLP/opus-mt-sv-hil', ('sv', 'ho'): 'Helsinki-NLP/opus-mt-sv-ho', ('sv', 'hr'): 'Helsinki-NLP/opus-mt-sv-hr', ('sv', 'ht'): 'Helsinki-NLP/opus-mt-sv-ht', ('sv', 'hu'): 'Helsinki-NLP/opus-mt-sv-hu', ('sv', 'id'): 'Helsinki-NLP/opus-mt-sv-id', ('sv', 'ig'): 'Helsinki-NLP/opus-mt-sv-ig', ('sv', 'ilo'): 'Helsinki-NLP/opus-mt-sv-ilo', ('sv', 'is'): 'Helsinki-NLP/opus-mt-sv-is', ('sv', 'iso'): 'Helsinki-NLP/opus-mt-sv-iso', ('sv', 'kg'): 'Helsinki-NLP/opus-mt-sv-kg', ('sv', 'kqn'): 'Helsinki-NLP/opus-mt-sv-kqn', ('sv', 'kwy'): 'Helsinki-NLP/opus-mt-sv-kwy', ('sv', 'lg'): 'Helsinki-NLP/opus-mt-sv-lg', ('sv', 'ln'): 'Helsinki-NLP/opus-mt-sv-ln', ('sv', 'lu'): 'Helsinki-NLP/opus-mt-sv-lu', ('sv', 'lua'): 'Helsinki-NLP/opus-mt-sv-lua', ('sv', 'lue'): 'Helsinki-NLP/opus-mt-sv-lue', ('sv', 'lus'): 'Helsinki-NLP/opus-mt-sv-lus', ('sv', 'lv'): 'Helsinki-NLP/opus-mt-sv-lv', ('sv', 'mfe'): 'Helsinki-NLP/opus-mt-sv-mfe', ('sv', 'mh'): 'Helsinki-NLP/opus-mt-sv-mh', ('sv', 'mos'): 'Helsinki-NLP/opus-mt-sv-mos', ('sv', 'mt'): 'Helsinki-NLP/opus-mt-sv-mt', ('sv', 'niu'): 'Helsinki-NLP/opus-mt-sv-niu', ('sv', 'nl'): 'Helsinki-NLP/opus-mt-sv-nl', ('sv', 'no'): 'Helsinki-NLP/opus-mt-sv-no', ('sv', 'nso'): 'Helsinki-NLP/opus-mt-sv-nso', ('sv', 'ny'): 'Helsinki-NLP/opus-mt-sv-ny', ('sv', 'pag'): 'Helsinki-NLP/opus-mt-sv-pag', ('sv', 'pap'): 'Helsinki-NLP/opus-mt-sv-pap', ('sv', 'pis'): 'Helsinki-NLP/opus-mt-sv-pis', ('sv', 'pon'): 'Helsinki-NLP/opus-mt-sv-pon', ('sv', 'rnd'): 'Helsinki-NLP/opus-mt-sv-rnd', ('sv', 'ro'): 'Helsinki-NLP/opus-mt-sv-ro', ('sv', 'ru'): 'Helsinki-NLP/opus-mt-sv-ru', ('sv', 'run'): 'Helsinki-NLP/opus-mt-sv-run', ('sv', 'rw'): 'Helsinki-NLP/opus-mt-sv-rw', ('sv', 'sg'): 'Helsinki-NLP/opus-mt-sv-sg', ('sv', 'sk'): 'Helsinki-NLP/opus-mt-sv-sk', ('sv', 'sl'): 'Helsinki-NLP/opus-mt-sv-sl', ('sv', 'sm'): 'Helsinki-NLP/opus-mt-sv-sm', ('sv', 'sn'): 'Helsinki-NLP/opus-mt-sv-sn', ('sv', 'sq'): 'Helsinki-NLP/opus-mt-sv-sq', ('sv', 'srn'): 'Helsinki-NLP/opus-mt-sv-srn', ('sv', 'st'): 'Helsinki-NLP/opus-mt-sv-st', ('sv', 'sv'): 'Helsinki-NLP/opus-mt-sv-sv', ('sv', 'swc'): 'Helsinki-NLP/opus-mt-sv-swc', ('sv', 'th'): 'Helsinki-NLP/opus-mt-sv-th', ('sv', 'tiv'): 'Helsinki-NLP/opus-mt-sv-tiv', ('sv', 'tll'): 'Helsinki-NLP/opus-mt-sv-tll', ('sv', 'tn'): 'Helsinki-NLP/opus-mt-sv-tn', ('sv', 'to'): 'Helsinki-NLP/opus-mt-sv-to', ('sv', 'toi'): 'Helsinki-NLP/opus-mt-sv-toi', ('sv', 'tpi'): 'Helsinki-NLP/opus-mt-sv-tpi', ('sv', 'ts'): 'Helsinki-NLP/opus-mt-sv-ts', ('sv', 'tum'): 'Helsinki-NLP/opus-mt-sv-tum', ('sv', 'tvl'): 'Helsinki-NLP/opus-mt-sv-tvl', ('sv', 'tw'): 'Helsinki-NLP/opus-mt-sv-tw', ('sv', 'ty'): 'Helsinki-NLP/opus-mt-sv-ty', ('sv', 'uk'): 'Helsinki-NLP/opus-mt-sv-uk', ('sv', 'umb'): 'Helsinki-NLP/opus-mt-sv-umb', ('sv', 've'): 'Helsinki-NLP/opus-mt-sv-ve', ('sv', 'war'): 'Helsinki-NLP/opus-mt-sv-war', ('sv', 'wls'): 'Helsinki-NLP/opus-mt-sv-wls', ('sv', 'xh'): 'Helsinki-NLP/opus-mt-sv-xh', ('sv', 'yap'): 'Helsinki-NLP/opus-mt-sv-yap', ('sv', 'yo'): 'Helsinki-NLP/opus-mt-sv-yo', ('sv', 'zne'): 'Helsinki-NLP/opus-mt-sv-zne', ('swc', 'en'): 'Helsinki-NLP/opus-mt-swc-en', ('swc', 'es'): 'Helsinki-NLP/opus-mt-swc-es', ('swc', 'fi'): 'Helsinki-NLP/opus-mt-swc-fi', ('swc', 'fr'): 'Helsinki-NLP/opus-mt-swc-fr', ('swc', 'sv'): 'Helsinki-NLP/opus-mt-swc-sv', ('taw', 'en'): 'Helsinki-NLP/opus-mt-taw-en', ('th', 'en'): 'Helsinki-NLP/opus-mt-th-en', ('th', 'fr'): 'Helsinki-NLP/opus-mt-th-fr', ('ti', 'en'): 'Helsinki-NLP/opus-mt-ti-en', ('tiv', 'en'): 'Helsinki-NLP/opus-mt-tiv-en', ('tiv', 'fr'): 'Helsinki-NLP/opus-mt-tiv-fr', ('tiv', 'sv'): 'Helsinki-NLP/opus-mt-tiv-sv', ('tl', 'de'): 'Helsinki-NLP/opus-mt-tl-de', ('tl', 'en'): 'Helsinki-NLP/opus-mt-tl-en', ('tl', 'es'): 'Helsinki-NLP/opus-mt-tl-es', ('tl', 'pt'): 'Helsinki-NLP/opus-mt-tl-pt', ('tll', 'en'): 'Helsinki-NLP/opus-mt-tll-en', ('tll', 'es'): 'Helsinki-NLP/opus-mt-tll-es', ('tll', 'fi'): 'Helsinki-NLP/opus-mt-tll-fi', ('tll', 'fr'): 'Helsinki-NLP/opus-mt-tll-fr', ('tll', 'sv'): 'Helsinki-NLP/opus-mt-tll-sv', ('tn', 'en'): 'Helsinki-NLP/opus-mt-tn-en', ('tn', 'es'): 'Helsinki-NLP/opus-mt-tn-es', ('tn', 'fr'): 'Helsinki-NLP/opus-mt-tn-fr', ('tn', 'sv'): 'Helsinki-NLP/opus-mt-tn-sv', ('to', 'en'): 'Helsinki-NLP/opus-mt-to-en', ('to', 'es'): 'Helsinki-NLP/opus-mt-to-es', ('to', 'fr'): 'Helsinki-NLP/opus-mt-to-fr', ('to', 'sv'): 'Helsinki-NLP/opus-mt-to-sv', ('toi', 'en'): 'Helsinki-NLP/opus-mt-toi-en', ('toi', 'es'): 'Helsinki-NLP/opus-mt-toi-es', ('toi', 'fi'): 'Helsinki-NLP/opus-mt-toi-fi', ('toi', 'fr'): 'Helsinki-NLP/opus-mt-toi-fr', ('toi', 'sv'): 'Helsinki-NLP/opus-mt-toi-sv', ('tpi', 'en'): 'Helsinki-NLP/opus-mt-tpi-en', ('tpi', 'sv'): 'Helsinki-NLP/opus-mt-tpi-sv', ('tr', 'ar'): 'Helsinki-NLP/opus-mt-tr-ar', ('tr', 'az'): 'Helsinki-NLP/opus-mt-tr-az', ('tr', 'en'): 'Helsinki-NLP/opus-mt-tr-en', ('tr', 'eo'): 'Helsinki-NLP/opus-mt-tr-eo', ('tr', 'es'): 'Helsinki-NLP/opus-mt-tr-es', ('tr', 'fr'): 'Helsinki-NLP/opus-mt-tr-fr', ('tr', 'lt'): 'Helsinki-NLP/opus-mt-tr-lt', ('tr', 'sv'): 'Helsinki-NLP/opus-mt-tr-sv', ('tr', 'uk'): 'Helsinki-NLP/opus-mt-tr-uk', ('trk', 'en'): 'Helsinki-NLP/opus-mt-trk-en', ('ts', 'en'): 'Helsinki-NLP/opus-mt-ts-en', ('ts', 'es'): 'Helsinki-NLP/opus-mt-ts-es', ('ts', 'fi'): 'Helsinki-NLP/opus-mt-ts-fi', ('ts', 'fr'): 'Helsinki-NLP/opus-mt-ts-fr', ('ts', 'sv'): 'Helsinki-NLP/opus-mt-ts-sv', ('tum', 'en'): 'Helsinki-NLP/opus-mt-tum-en', ('tum', 'es'): 'Helsinki-NLP/opus-mt-tum-es', ('tum', 'fr'): 'Helsinki-NLP/opus-mt-tum-fr', ('tum', 'sv'): 'Helsinki-NLP/opus-mt-tum-sv', ('tvl', 'en'): 'Helsinki-NLP/opus-mt-tvl-en', ('tvl', 'es'): 'Helsinki-NLP/opus-mt-tvl-es', ('tvl', 'fi'): 'Helsinki-NLP/opus-mt-tvl-fi', ('tvl', 'fr'): 'Helsinki-NLP/opus-mt-tvl-fr', ('tvl', 'sv'): 'Helsinki-NLP/opus-mt-tvl-sv', ('tw', 'es'): 'Helsinki-NLP/opus-mt-tw-es', ('tw', 'fi'): 'Helsinki-NLP/opus-mt-tw-fi', ('tw', 'fr'): 'Helsinki-NLP/opus-mt-tw-fr', ('tw', 'sv'): 'Helsinki-NLP/opus-mt-tw-sv', ('ty', 'es'): 'Helsinki-NLP/opus-mt-ty-es', ('ty', 'fi'): 'Helsinki-NLP/opus-mt-ty-fi', ('ty', 'fr'): 'Helsinki-NLP/opus-mt-ty-fr', ('ty', 'sv'): 'Helsinki-NLP/opus-mt-ty-sv', ('tzo', 'es'): 'Helsinki-NLP/opus-mt-tzo-es', ('uk', 'bg'): 'Helsinki-NLP/opus-mt-uk-bg', ('uk', 'ca'): 'Helsinki-NLP/opus-mt-uk-ca', ('uk', 'cs'): 'Helsinki-NLP/opus-mt-uk-cs', ('uk', 'de'): 'Helsinki-NLP/opus-mt-uk-de', ('uk', 'en'): 'Helsinki-NLP/opus-mt-uk-en', ('uk', 'es'): 'Helsinki-NLP/opus-mt-uk-es', ('uk', 'fi'): 'Helsinki-NLP/opus-mt-uk-fi', ('uk', 'fr'): 'Helsinki-NLP/opus-mt-uk-fr', ('uk', 'he'): 'Helsinki-NLP/opus-mt-uk-he', ('uk', 'hu'): 'Helsinki-NLP/opus-mt-uk-hu', ('uk', 'it'): 'Helsinki-NLP/opus-mt-uk-it', ('uk', 'nl'): 'Helsinki-NLP/opus-mt-uk-nl', ('uk', 'no'): 'Helsinki-NLP/opus-mt-uk-no', ('uk', 'pl'): 'Helsinki-NLP/opus-mt-uk-pl', ('uk', 'pt'): 'Helsinki-NLP/opus-mt-uk-pt', ('uk', 'ru'): 'Helsinki-NLP/opus-mt-uk-ru', ('uk', 'sh'): 'Helsinki-NLP/opus-mt-uk-sh', ('uk', 'sl'): 'Helsinki-NLP/opus-mt-uk-sl', ('uk', 'sv'): 'Helsinki-NLP/opus-mt-uk-sv', ('uk', 'tr'): 'Helsinki-NLP/opus-mt-uk-tr', ('umb', 'en'): 'Helsinki-NLP/opus-mt-umb-en', ('ur', 'en'): 'Helsinki-NLP/opus-mt-ur-en', ('urj', 'en'): 'Helsinki-NLP/opus-mt-urj-en', ('urj', 'urj'): 'Helsinki-NLP/opus-mt-urj-urj', ('ve', 'en'): 'Helsinki-NLP/opus-mt-ve-en', ('ve', 'es'): 'Helsinki-NLP/opus-mt-ve-es', ('vi', 'de'): 'Helsinki-NLP/opus-mt-vi-de', ('vi', 'en'): 'Helsinki-NLP/opus-mt-vi-en', ('vi', 'eo'): 'Helsinki-NLP/opus-mt-vi-eo', ('vi', 'es'): 'Helsinki-NLP/opus-mt-vi-es', ('vi', 'fr'): 'Helsinki-NLP/opus-mt-vi-fr', ('vi', 'it'): 'Helsinki-NLP/opus-mt-vi-it', ('vi', 'ru'): 'Helsinki-NLP/opus-mt-vi-ru', ('vsl', 'es'): 'Helsinki-NLP/opus-mt-vsl-es', ('wa', 'en'): 'Helsinki-NLP/opus-mt-wa-en', ('wal', 'en'): 'Helsinki-NLP/opus-mt-wal-en', ('war', 'en'): 'Helsinki-NLP/opus-mt-war-en', ('war', 'es'): 'Helsinki-NLP/opus-mt-war-es', ('war', 'fi'): 'Helsinki-NLP/opus-mt-war-fi', ('war', 'fr'): 'Helsinki-NLP/opus-mt-war-fr', ('war', 'sv'): 'Helsinki-NLP/opus-mt-war-sv', ('wls', 'en'): 'Helsinki-NLP/opus-mt-wls-en', ('wls', 'fr'): 'Helsinki-NLP/opus-mt-wls-fr', ('wls', 'sv'): 'Helsinki-NLP/opus-mt-wls-sv', ('xh', 'en'): 'Helsinki-NLP/opus-mt-xh-en', ('xh', 'es'): 'Helsinki-NLP/opus-mt-xh-es', ('xh', 'fr'): 'Helsinki-NLP/opus-mt-xh-fr', ('xh', 'sv'): 'Helsinki-NLP/opus-mt-xh-sv', ('yap', 'en'): 'Helsinki-NLP/opus-mt-yap-en', ('yap', 'fr'): 'Helsinki-NLP/opus-mt-yap-fr', ('yap', 'sv'): 'Helsinki-NLP/opus-mt-yap-sv', ('yo', 'en'): 'Helsinki-NLP/opus-mt-yo-en', ('yo', 'es'): 'Helsinki-NLP/opus-mt-yo-es', ('yo', 'fi'): 'Helsinki-NLP/opus-mt-yo-fi', ('yo', 'fr'): 'Helsinki-NLP/opus-mt-yo-fr', ('yo', 'sv'): 'Helsinki-NLP/opus-mt-yo-sv', ('zai', 'es'): 'Helsinki-NLP/opus-mt-zai-es', ('zh', 'bg'): 'Helsinki-NLP/opus-mt-zh-bg', ('zh', 'de'): 'Helsinki-NLP/opus-mt-zh-de', ('zh', 'en'): 'Helsinki-NLP/opus-mt-zh-en', ('zh', 'fi'): 'Helsinki-NLP/opus-mt-zh-fi', ('zh', 'he'): 'Helsinki-NLP/opus-mt-zh-he', ('zh', 'it'): 'Helsinki-NLP/opus-mt-zh-it', ('zh', 'ms'): 'Helsinki-NLP/opus-mt-zh-ms', ('zh', 'nl'): 'Helsinki-NLP/opus-mt-zh-nl', ('zh', 'sv'): 'Helsinki-NLP/opus-mt-zh-sv', ('zh', 'uk'): 'Helsinki-NLP/opus-mt-zh-uk', ('zh', 'vi'): 'Helsinki-NLP/opus-mt-zh-vi', ('zle', 'en'): 'Helsinki-NLP/opus-mt-zle-en', ('zle', 'zle'): 'Helsinki-NLP/opus-mt-zle-zle', ('zls', 'en'): 'Helsinki-NLP/opus-mt-zls-en', ('zls', 'zls'): 'Helsinki-NLP/opus-mt-zls-zls', ('zlw', 'en'): 'Helsinki-NLP/opus-mt-zlw-en', ('zlw', 'fiu'): 'Helsinki-NLP/opus-mt-zlw-fiu', ('zlw', 'zlw'): 'Helsinki-NLP/opus-mt-zlw-zlw', ('zne', 'es'): 'Helsinki-NLP/opus-mt-zne-es', ('zne', 'fi'): 'Helsinki-NLP/opus-mt-zne-fi', ('zne', 'fr'): 'Helsinki-NLP/opus-mt-zne-fr', ('zne', 'sv'): 'Helsinki-NLP/opus-mt-zne-sv'}
import spacy
from sentence_transformers import SentenceTransformer
from data_tooling.pii_processing.ontology.ontology_manager import OntologyManager
 
import qg_pipeline
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
labse =  SentenceTransformer("sentence-transformers/LaBSE").half().eval().cuda()
qg = qg_pipeline.pipeline("multitask-qa-qg")

ontology_manager = None#OntologyManager(target_lang='en') #target_lang=target_lang
translation_pipelines = {}
ner_model_name2pipelines = {} 

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

class TextAugment:

  m2m_model = None
  m2m_tokenizer = None 
  en_spacy_nlp = None 
  en_stopwords = set(stopwords_ac_dc['en'])
  labse = None

  #TODO, copy in the code from https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/anonymization.py
  rulebase = {"en": [([
      ("AGE", re.compile("\S+ years old|\S+\-years\-old|\S+ year old|\S+\-year\-old"), None, None, None),
      ("STREET_ADDRESS", re.compile(
          '\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)'), None, None, None),
      ("STREET_ADDRESS", re.compile('P\.? ?O\.? Box \d+'), None, None, None),
      ("GOVT_ID", re.compile(
          '(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[- ](?!00)[0-9]{2}[- ](?!0000)[0-9]{4}'), None, None, None),
      ("DISEASE", re.compile("diabetes|cancer|HIV|AIDS|Alzheimer's|Alzheimer|heart disease"), None, None, None),
      ("NORP", re.compile("upper class|middle class|working class|lower class"), None, None, None),
      ], 1),
    ],
   "vi": []
  }

  # see https://www.researchgate.net/publication/259179064_Comparing_Methods_for_Detecting_Child_Exploitation_Content_Online for common CSAM words
  # http://antipaedo.lip6.fr/T12/keywords_cnrs.pdf - for top 20 from table 7 and 8 of the paper, plus other , ignoring stopwords like "tuesday"
  # WARNING: Translations are probably not accurate. TODO to fix.
  # we are only doing our current covered langauges. TODO, do for all languages.
  banned_words = {
      'ar': {"فتاة صغيرة" , "فتاة صغيرة" , "فتى صغير", "مراهق" , "لولي" , "لولا" , "لوليتا" , "لولي" , "لولي" , "لوليتاغوي" , "فتاة" , "صبي" ,"طفل" , "حمام" },
      'bn': {'ছোট মেয়ে', 'ছোট মেয়ে', 'ছোট ছেলে', 'কিশোর', 'ললি', 'লোলা', 'ললিতা', 'ললি', 'লোলি', 'ললিটাগুই', 'মেয়ে', 'ছেলে' , 'শিশু', 'স্নান',},
      'hi': {'युवा लड़की', 'छोटी लड़की', 'छोटा लड़का', 'किशोर', 'लॉली', 'लोला', 'लोलिता', 'लल्ली', 'लोली', 'लोलितागुय', 'लड़की', 'लड़का' , 'बच्चा', 'स्नान',},
      'eu': {'neska gaztea', 'neska txikia', 'mutil txikia', 'nerabea', 'neska', 'mutil' , 'haurra', 'bainua',},
      'ca': {'noia', 'nena', 'nen petit', 'nen' , 'nen', 'banyer',},
      'vi': {'thiếu niên', 'cậu nhỏ', 'cậu bé', 'cô gái trẻ', 'cô bé', 'cậu bé', 'trẻ em', 'tắm', },
      'zh': {'儿童','女孩','孩子', '小女孩', '小姑娘','小男孩', '年輕女孩','年轻女孩','年轻的女孩','洗澡','洛丽塔','洛麗塔','浴','男孩','萝拉','萝莉','萝莉塔吉','蘿拉','蘿莉','蘿莉塔','青少年'},
      'fr': {'jeune fille','petite fille','petit garçon','ado',  'fille', 'garçon' , 'enfant', 'bain',},
      'id': {'gadis muda','gadis kecil','anak laki-laki kecil','remaja',  'perempuan', 'laki-laki' , 'anak', 'mandi',},
      'fa': {'دختر جوان',  'دختر کوچولو',  'پسر کوچولو',  'نوجوان',  'لولی',  'لولا',  'لولیتا',  'لولی',  'لولی',  'لولیتاگو',  'دختر',  'پسر' ,'کودک',  'حمام', },
      'ur': {'نوجوان لڑکی',  'چھوٹی لڑکی',  'چھوٹا لڑکا',  'نوعمر',  'لولی',  'لولا',  'لولیتا',  'لولی',  'لولی',  'لولیتاگوئے',  'لڑکی',  'لڑکا' ,'بچہ',  'غسل', },
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
  # note that we do not have a transformer model for catalan, but spacy covers catalan and we use transfer learning from Davlan/xlm-roberta-base-ner-hrl
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
      "pt": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ]], #there is a
      "fr": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ]],
      "zh": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ]],
      'vi': [["lhkhiem28/COVID-19-Named-Entity-Recognition-for-Vietnamese", RobertaForTokenClassification, 1.0]],#["jplu/tf-xlm-r-ner-40-lang", None ], 
      'hi': [["jplu/tf-xlm-r-ner-40-lang", None, 1.0 ]],
      'ur': [["jplu/tf-xlm-r-ner-40-lang", None, 1.0 ]],
      'id': [["cahya/bert-base-indonesian-NER", BertForTokenClassification, 1.0]], 
      'bn': [["sagorsarker/mbert-bengali-ner", BertForTokenClassification, 1.0]],
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
  translation_pipelines = {}
  ner_model_name2pipelines = {}  
  strip_chars = " ,،、{}[]|()\"'“”《》«»"
  punc_char = ".?!:;?。…"
  special_char = " ,{}[]()|\\\"'“”《》«»~!@#$%^&*{}[]()_+=-0987654321`<>,、،./?':;“”\"\t\n\\πه☆●¦″．۩۱（☛₨➩°・■↑☻、๑º‹€σ٪’Ø·−♥ıॽ،٥《‘©。¨﴿！★×✱´٬→±x：¹？£―▷ф¡Г♫∟™ª₪®▬「—¯；¼❖․ø•�」٣，٢◦‑←§١ー٤）˚›٩▼٠«¢¸٨³½˜٭ˈ¿¬ι۞⌐¥►†ƒ∙²»¤…﴾⠀》′ا✓→¶'"
  junk = set(",{}[]()|\\\"'“”《》«»~!@#$%^&*{}[]()_+=-0987654321`<>,、،./?':;“”\"\t\n\\πه☆●¦″．۩۱（☛₨➩°・■↑☻、๑º‹€σ٪’Ø·−♥ıॽ،٥《‘©。¨﴿！★×✱´٬→±x：¹？£―▷ф¡Г♫∟™ª₪®▬「—¯；¼❖․ø•�」٣，٢◦‑←§١ー٤）˚›٩▼٠«¢¸٨³½˜٭ˈ¿¬ι۞⌐¥►†ƒ∙²»¤…﴾⠀》′ا✓→¶'")
  #don't add a space for junk chars
  ontology_manager = None
  max_stoword_len_zh = max([len(a) for a in stopwords_ac_dc.get('zh')])
  max_stoword_len_ko = max([len(a) for a in stopwords_ac_dc.get('ko')])
  max_stoword_len_ja = max([len(a) for a in stopwords_ac_dc.get('ja')])
  qg = None
    
  def __init__(self):
    TextAugment.labse = labse 
    TextAugment.ontology_manager = ontology_manager
    TextAugment.translation_pipelines = translation_pipelines
    TextAugment.ner_model_name2pipelines = ner_model_name2pipelines
    TextAugment.qg = qg
    if False: # use the below for production usage. the above is for testing. 
      if TextAugment.en_spacy_nlp is None: TextAugment.en_spacy_nlp = spacy.load('en_core_web_sm')
      if TextAugment.labse is None: TextAugment.labse =  SentenceTransformer("sentence-transformers/LaBSE").half().eval().cuda()
      if TextAugment.ontology_manager is None: TextAugment.ontology_manager = OntologyManager(src_lang='en') #src_lang=src_lang
    print ("finished load")

  def check_good_sentence(self, s, src_lang, stopwords, stopword_ratio_cutoff=0.06, bannedwords=None, badwords=None, badword_ratio_cutoff=0.15,  junk_ratio=0.16, max_badword_len=5):
    #basic dejunk
    # for badwords, only filter out if the ratio is exceeded AND there exists one banned word
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
    if badwords is not None:
      #print ('bw', len([s2 for s2 in sArr if s2 in badwords])/len(sArr))
      if src_lang not in ("ja", "ko", "zh") and len([s2 for s2 in sArr if s2 in badwords])/len(sArr) > badword_ratio_cutoff:
        if any(s2 for s2 in sArr if s2 in bannedwords) or any(s2 for s2 in sArr if s2 in default_bannedwords):
          return False
      if src_lang in ("ja", "ko", "zh"):
        badword_ratio_cutoff /= 100
        len_s = len(s)
        bad_cnt = 0
        total_cnt = 0
        for i in range(len_s):
          for j in range(i+1,min(len_s, i+max_badword_len)):
            if s[i:j] in badwords:
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

def generate_questions(self, batch, default_answers=[]):
    answers = {}

    i= 0
    allqa = []
    for chunk in batch:
      text = chunk['text']
      answers1={}
      #ti = time.time()
      text = text.replace("U.S.","US").replace("\n", " ").replace(",", " , ").replace("  ", " ").strip().replace(" He ", " Lincoln ").replace(" he ", " Lincoln ").replace(" him ", " Lincoln ").replace(" , ", ", ")
      aHash = nlp(text) # , default_answers=default_answers)
      allqa.append(aHash)
      default_answers = list(set([a['answer'] for a in aHash]+default_answers))
      print (aHash)
      #for aHash1 in aHash:
      #  extraction = vis.parse(list(dep_parser(aHash1['question']).sents)[0], aHash1['answer'])
      #  print (extraction.arg1, '*', extraction.rel, '*', extraction.arg2)

      for aHash1 in aHash:
        if answers.get(aHash1['answer'].lower()) or answers1.get(aHash1['answer'].lower()):
          continue
        if len(aHash1['answer'].split()) > 10:
          aHash1['answer'] = " ".join(aHash1['answer'].split()[:10])
        i+=1
        quest=aHash1['question'].lower().strip("?").replace("'s",  " 's").replace("  ", " ").split()
        label=""
        if quest[0] == "who" and aHash1['answer'][-1] =='s':
          label="organization_"+str(i)
          if "'s" in quest:
            for j in range(len(quest)):
              if j > 0 and quest[j-1]=="'s":
                label = quest[j]+"_"+str(i)
                break
          for a in aHash1['answer'].lower().split():
            if a not in stopwords_hash:
              answers[a] = label
        elif quest[0] == "who":
          label="person_"+str(i)
          if "'s" in quest:
            for j in range(len(quest)):
              if j > 0 and quest[j-1]=="'s":
                label = quest[j]+"_"+str(i)
                break
          for a in aHash1['answer'].lower().split():
            if a not in stopwords_hash:
              answers[a] = label
        elif quest[0] == "where":
          label="location_"+str(i)
        elif quest[0] == "when":
          label="date_or_time_"+str(i)
        elif quest[0] == "why":
          label="reason_"+str(i)
        elif quest[0] == "how" and quest[1] in ("much", "many"):
          label="quantity_"+str(i)
        elif quest[0] == "how":
          label="method_"+str(i)
        elif quest[0] in ("which", "what") and quest[1] not in stopwords_hash:
          label=quest[1]+"_"+str(i)
        elif "'s" in quest:
          for j in range(len(quest)):
            if j > 0 and quest[j-1]=="'s":
              label = quest[j]+"_"+str(i)
              break
        if label:
          answers[aHash1['answer'].lower()] = label


        #for b in a['answer'].lower().split():
        #  answers[b] = label
      print (answers)

    for aHash in allqa:
      answers1={}
      for aHash1 in aHash:
        if answers1.get(aHash1['answer'].lower()):
          continue
        quest = " "+aHash1['question'].lower().strip("?").replace("'s",  " 's").replace("  ", " ")+" "
        q_type =  quest[0]
        agent = []
        answer_keys = list(answers.keys())
        answer_keys.sort(key=lambda k: len(k), reverse=True)
        for a in answer_keys:
          if " "+a+" " in quest:
              quest = quest.replace(" "+a+" ", " "+answers[a]+" ")
          elif " "+a+", " in quest:
              quest = quest.replace(" "+a+", ", " "+answers[a]+", ")
        quest = quest.split()
        #print (quest)
        qtype = []
        if answers.get(aHash1['answer'].lower()):
          if answers.get(aHash1['answer'].lower()).split("_")[0] == "person":
            qtype = ["is", "who"]
        if not qtype and quest[0] in ("when", "where", "why", "how"): #, "which"
          qtype=[quest[0]]
          if quest[0]=="how" and quest[1] in ("much", "many"):
            qtype = qtype + [quest[1]]

        #label=[q for q in quest if (q not in ("much", "many",) and not stopwords_hash.get(q) and q not in answers)]#+qtype
        label=[q for q in quest if (q[0] not in "0123456789") and (q not in ("the", "a", "an"))]
        if len(label) > 10:
            label=label[:10]
        answers1[aHash1['answer'].lower()] = " ".join(label)
      print (answers1)

  @staticmethod
  def get_aligned_text(sent1, sent2, src_lang):
    """
    Given two sentences, find blocks of text that match and that don't match.
    return the blocks, and a matching score.
    Used to extract NER from original language sentence.
    """
    if src_lang in ("ja", "ko", "zh"):
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
        blocks2.append([sep.join(sent1[prevEndA:blocks[blockI][0]]), sep.join(sent2[prevEndB:blocks[blockI][1]]), 0])
        nonMatchLen += max(blocks[blockI][0] - prevEndA, blocks[blockI][1] - prevEndB)
      if blocks[blockI][2] != 0:
        blocks2.append([sep.join(sent1[blocks[blockI][0]:blocks[blockI][0]+blocks[blockI][2]]), sep.join(sent2[blocks[blockI][1]:blocks[blockI][1]+blocks[blockI][2]]), 1])
        prevEndA = blocks[blockI][0]+blocks[blockI][2]
        prevEndB = blocks[blockI][1]+blocks[blockI][2]
        matchLen += blocks[blockI][2]
    #score = float(matchLen+1)/float(nonMatchLen+1)
    return (blocks2, score)

  def do_translations(self, texts, src_lang='en', target_lang='hi', batch_size=16, do_mariam_mt=False):
    if not do_mariam_mt:
      if self.m2m_tokenizer is None: 
        self.m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
      try:
        self.m2m_tokenizer.src_lang = src_lang
        target_lang_bos_token = self.m2m_tokenizer.get_lang_id(target_lang)
        translations = []
        for src_text_list in tqdm(self.batch(texts, batch_size)):
          try:
            batch = self.m2m_tokenizer(src_text_list, return_tensors="pt", padding=True, truncation=True).to('cuda')
          except:
            break
          if self.m2m_model is None:
            self.m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").eval().half().cuda()
          gen = self.m2m_model.generate(**batch, forced_bos_token_id=target_lang_bos_token, no_repeat_ngram_size=4, ) #
          outputs = self.m2m_tokenizer.batch_decode(gen, skip_special_tokens=True)
          translations.extend(outputs)
        return translations
      except:
       pass
    translations = []
    #mariam_mt = self.mariam_mt
    model_name = mariam_mt.get((src_lang, target_lang))
    mt_pipeline = None
    if model_name is not None and model_name not in self.translation_pipelines:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).half().eval().cuda()
        mt_pipeline = pipeline("translation", model=model, tokenizer=tokenizer, device=0)
        self.translation_pipelines[model_name] = mt_pipeline
        if not mt_pipeline:
          raise RuntimeError("no translation pipeline") # we could do multi-step translation where there are no pairs
    mt_pipeline = self.translation_pipelines[model_name]
    for src_text_list in tqdm(self.batch(texts, batch_size)):
        outputs = [t['translation_text'] for t in mt_pipeline(src_text_list)]
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

  def hf_ner(self, hf_pipeline, src_lang, docs, chunks, stopwords=None, weight=1.5):
    """
    run the text through a Huggingface ner pipeline. 
    any tags found by this method will be weighted by the weight param
    TODO: use the predicted value of the logits to further weight prediction
    """
    if stopwords is None:
      stopwords = set(ac_dc_stopwords.get(src_lang, []))
    offset_key=f'{src_lang}_offset'
    text_key=f'{src_lang}_text'
    ner_key=f'{src_lang}_ner'
    results_arr = hf_pipeline([chunk[text_key] for chunk in chunks])
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
      if results[0]['start'] is not None:
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
          if label in ('STREET_ADDRESS',): label = 'STREET_ADDRESS'
          elif label in ('PUBLIC_FIGURE',): label = 'PUBLIC_FIGURE'
          elif label in ('NAME', 'PER', 'PERSON'): label = 'PERSON'
          elif label in ('LOCATION', 'LOC', 'GPE'): label = 'GPE'
          elif label in ('ORGANIZATION', 'ORG'): label = 'ORG'
          elif label in ('AGE',): label = 'AGE'
          elif label in ('NORP',): label = 'NORP'
          elif label in ('BIO', 'SYMPTOM_AND_DISEASE', 'DISEASE' ): label = 'DISEASE'
          elif label in ('PATIENT_ID', 'GOVT_ID' ): label = 'GOVT_ID'
          elif label in ('USER_ID', ): label = 'USER_ID'
          elif label in ('MISC', ) and '@' in ner_result['word']: label = 'USER_ID'
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

  def spacy_ner(self, docs, nlp, stopwords, spacy_weight, src_lang, extra_weight=1.0):
        """ 
        Use the spacy models to create mentions w/ NER
        """
        if not nlp:
          return
        if stopwords is None:
          stopwords = set(ac_dc_stopwords.get(src_lang, []))
        offset_key=f'{src_lang}_offset'
        text_key=f'{src_lang}_text'
        ner_key=f'{src_lang}_ner'
        for doc in docs.values():
          ner =  doc[ner_key] =  doc.get(ner_key,{})
          text = doc[text_key]
          doc = nlp(text)
          entities = list(doc.ents)
          ents = [(entity.text, entity.label_ if (entity.label_ in ('PERSON', 'GPE', 'ORG', 'NORP') and 'http:' not in entity.text) else 'MISC') for entity in entities]
          i = 0
          for ner_word, label in ents: 
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
                aHash[label] = aHash.get(label, 0) + spacy_weight * (1.0 + len(ner_word)/100) * extra_weight
                ner[mention] = aHash

  def trim_to_prefer_person(self, docs, chunks, prob=100):
      # downsample to mostly docs with mentions of people, govt_id and email
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
          if key.endswith('_ner'):
            if not found_ner:
              found_ner = doc[key] != {}
            ner =  doc[key] 
            for aHash in ner.values():
              if 'PUBLIC_FIGURE' in aHash or 'PERSON' in aHash or 'GOVT_ID' in aHash or 'USER_ID' in aHash: 
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
      print ('trim_to_prefer_person', (len_docs-len(docs2))/len_docs)
      return docs2, chunks2


  def process_ner_chunks_with_trans(self, 
                          src_lang, 
                          docs, 
                          chunks, 
                          target_lang=None,
                          do_spacy = True,
                          do_hf_ner = True,
                          do_ontology = True,
                          do_backtrans=False,
                          do_regex = True,
                          do_cleanup=True,
                          batch_size = 5, 
                          batch_window=70,
                          ontology_weight=0.9,
                          spacy_weight=1.25,
                          hf_ner_weight=1.0,
                          backtrans_weight=0.9,
                          do_postprocessing_after_backtrans=False,
                          do_docs_trim=False):
    if target_lang is None:
        target_lang = src_lang


    stopwords1 = set(stopwords_ac_dc[src_lang])
    stopwords2 = set(stopwords_ac_dc[target_lang])

    #init spacy pipeline
    spacy_nlp = None
    if do_spacy:
      if target_lang == 'en':
        spacy_nlp = spacy.load('en_core_web_sm')
      elif target_lang == 'zh':
        try:
          spacy_nlp = spacy.load('zh_core_web_sm')
        except:
          pass
      elif target_lang == 'pt':
        spacy_nlp = spacy.load('pt_core_news_sm')
      elif target_lang == 'fr':
        spacy_nlp = spacy.load('fr_core_news_sm')
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
          print ("setting") 
          try:
            model = model_cls.from_pretrained(model_name).half().eval().cuda()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0)
            self.ner_model_name2pipelines[model_name] = ner_pipeline
          except:
            try:
              ner_pipeline = pipeline("ner",  model=model_name, tokenizer=(model_name, {"use_fast": True},), device=0)
              self.ner_model_name2pipelines[model_name] = ner_pipeline
            except:
              ner_pipeline = pipeline("ner",  model=model_name, tokenizer=(model_name, {"use_fast": True},), framework="tf", device=0)
              self.ner_model_name2pipelines[model_name] = ner_pipeline
        ner_pipelines.append((self.ner_model_name2pipelines[model_name], hf_ner_weight2))
    target_is_cjk = target_lang in ('zh', 'ko', 'ja')
    src_is_cjk = src_lang in ('zh', 'ko', 'ja')
    if target_lang == src_lang:
      backtrans_weight = 1.0
      do_backtrans = False
      
    elif target_lang != src_lang:
        #translate from src_lang to target_lang and do ner in target_lang.  translation also acts as an error check and additional ner. 
        # we check to see if the already tagged items in src lang should have scores for tags increased or are common words in target lang and should not be tagged. 
        # we also add new labels for items that are already tagged in src_lang.

        
        if src_is_cjk:
            sep = ""
        else:
            sep = " "

        if src_is_cjk:
            lbracket = "[["
            rbracket = "]]"
        else:
            lbracket = "["
            rbracket = "]"

        for chunk in chunks:
          text = chunk[f'{src_lang}_text'].replace("(", "{").replace(")", "}")
          _id = chunk['id']
          offset = chunk[f'{src_lang}_offset']
          doc = docs[_id]
          offset_end = offset + len(text)
          if f'{src_lang}_items' not in doc:
            doc[f'{src_lang}_items'] = list(doc.get(f'{src_lang}_ner', {}).keys())
            doc[f'{src_lang}_items'].sort(key=lambda a: a[1])
          i = 0
          for idx, key in enumerate(doc[f'{src_lang}_items']):
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
          chunk[f'{src_lang}_tmpl_text'] = text

        src_items_sorted = list(enumerate(doc[f'{src_lang}_items']))
        src_items_sorted.sort(key=lambda a: len(a[1][0]))
        for chunk in chunks:
          text = chunk[f'{src_lang}_tmpl_text']
          _id = chunk['id']
          doc = docs[_id]
          for idx, key in src_items_sorted:
            if len(key[0]) < 5 and not self.cjk_detect(key[0]):
              text = text.replace(" "+key[0]+" ", f"  **{idx}**  ")
            else:
              text = text.replace(key[0], f" **{idx}** ")
          chunk[f'{src_lang}_tmpl_text'] = text

        for chunk in chunks:
          text = chunk[f'{src_lang}_tmpl_text']
          _id = chunk['id']
          doc = docs[_id]
          for idx, key in enumerate(doc[f'{src_lang}_items']):
            text = text.replace(f" **{idx}** ", f" {idx} {lbracket} {key[0]} {rbracket}")
          chunk[f'{src_lang}_tmpl_text'] = text.replace("  ", " ")
   
        #print ('*****', chunks2)
        chunks2 = [chunk[f'{src_lang}_tmpl_text'] for chunk in chunks]
        text2 = self.do_translations(chunks2, src_lang=src_lang, target_lang=target_lang, batch_size=batch_size)
        for chunk, trans_text in zip(chunks, text2):
          #langid check
          try:
            lang =  langid.classify(trans_text)[0]
          except:
            lang = target_lang
          if lang == target_lang:
            chunk[f'{target_lang}_text'] = trans_text.lstrip(" .").replace(rbracket, "]").replace(lbracket, "[").replace("}", ")").replace("{", "(")
          else:
            chunk[f'{target_lang}_text'] = " . . . "

        all_embed = self.labse.encode(chunks2, convert_to_tensor=True)
        all_trans_embed = self.labse.encode([chunk[f'{target_lang}_text'] for chunk in chunks], convert_to_tensor=True)
        similarity = cosine_similarity(all_embed, all_trans_embed, dim=1)
        for chunk, sim_score in zip(chunks, similarity):
          trans_text = chunk[f'{target_lang}_text']
          sim_score = sim_score.item()
          print (sim_score, '**', trans_text, '**', chunk[f'{src_lang}_tmpl_text'])
          _id = chunk['id']
          doc = docs[_id]
          if sim_score < 0.75:
            trans_text = chunk[f'{target_lang}_text'] = " . . . "
            if doc.get(f'{target_lang}_text', ""):
              chunk[f'{target_lang}_offset'] = len(doc.get(f'{target_lang}_text', "")) + 1
            else:
              chunk[f'{target_lang}_offset'] = 0
            doc[f'{target_lang}_text'] = (doc.get(f'{target_lang}_text', "") + " " + trans_text).strip()
            chunk[f'{src_lang}_2_{target_lang}_sim'] = 0.0
            continue
          chunk[f'{src_lang}_2_{target_lang}_sim'] = sim_score
          len_items = len(doc[f'{src_lang}_items'])
          doc[f'{target_lang}_2_{src_lang}_ner'] = doc.get(f'{target_lang}_2_{src_lang}_ner', {})
          while "[" in trans_text:
            before, after = trans_text.split("[",1)
            before = before.strip()
            after = after.strip()
            before_arr = before.split()
            if "]" not in after or not before_arr:
              trans_text = before + sep + after
              continue
            idx = before_arr[-1]
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
                ent_lower = ent.lower()
                if ent_lower in stopwords2: 
                  #reduce weight of target labels if this is translated into an en stopword
                  if key in doc[f'{src_lang}_ner']: 
                    aHash = doc[f'{src_lang}_ner'][key]
                    for key in list(aHash.keys()):
                      aHash[key] /= 2.0
                else:
                  vals = list(doc[f'{src_lang}_ner'][key].keys())
                  ent = ent.strip(self.strip_chars)
                  doc[f'{target_lang}_2_{src_lang}_ner'][ent] = idx
            else: # except:
              pass
            trans_text = before + " " + ent + " " + after
          trans_text = chunk[f'{target_lang}_text'] = trans_text.replace("  ", "").strip() 
          if doc.get(f'{target_lang}_text', ""):
            chunk[f'{target_lang}_offset'] = len(doc.get(f'{target_lang}_text', "")) + 1
          else:
            chunk[f'{target_lang}_offset'] = 0
          doc[f'{target_lang}_text'] = (doc.get(f'{target_lang}_text', "") + " " + trans_text).strip()
    
    if do_regex:
      pass #TBD

    if do_ontology:
        # ontology - context independent - there are some bugs in disease detection which needs to be fixed
        for doc in docs.values():
          doc[f'{target_lang}_ner'] = ner = doc.get(f'{target_lang}_ner', {})
          if target_lang == 'en':
            chunk2ner = self.ontology_manager.tokenize(doc[f'{target_lang}_text'])['chunk2ner']
            onto_items = []
            for c, label in chunk2ner.items():
              ner_word  = c[0].replace(" ", "").replace("_", "").replace("_", "") if self.cjk_detect(c[0]) else c[0].replace("_", " ").replace("_", " ").rstrip(self.strip_chars)
              if ner_word.lower() not in stopwords2:
                if not self.cjk_detect(ner_word) and label in ('PERSON', 'PUBLIC_FIGURE', 'ORG') and " " not in ner_word: continue
                onto_items.append(((ner_word, c[1], c[1] + len(ner_word)), label))
            for ner_mention, label in list(set(onto_items)):
                aHash = ner.get(ner_mention, {})
                aHash[label] = aHash.get(label, 0) + ontology_weight * (1.0 + len(ner_mention[0])/100) * backtrans_weight
                ner[ner_mention] = aHash

    if do_spacy:
        if spacy_nlp:
          # spacy
          self.spacy_ner(docs, spacy_nlp, stopwords2, spacy_weight, target_lang, extra_weight=backtrans_weight)

    if do_hf_ner:
        # transformer
        for ner_pipeline, hf_ner_weight2 in ner_pipelines:
          for a_batch in self.batch(chunks, batch_size):
            self.hf_ner(ner_pipeline, target_lang, docs, a_batch, stopwords=stopwords2, weight=hf_ner_weight*backtrans_weight*hf_ner_weight2)
    
    if do_cleanup:
        #do some cleanups. we don't want any ner that are just short numbers, stopwords or single characters.
        for _id, doc in docs.items():
          ner =  doc[f'{target_lang}_ner'] 
          for key in list(doc[f'{target_lang}_ner'].keys()):
            ner_word = key[0]
            try:
              if len(ner_word) < 4 and float(ner_word):
                print ("deleting ", ner_word)
                del doc[f'{target_lang}_ner'][key]
                continue
            except:
              pass
            if ner_word.lower() in stopwords2 or (not self.cjk_detect(ner_word) and len(ner_word) <= 1):
              print ("deleting ", ner_word)
              del doc[f'{target_lang}_ner'][key]

    #increase weight of src ner items if the target translations indicate it's an NER
    if target_lang != src_lang:
          for doc in docs.values():
            ner =  doc[f'{target_lang}_ner'] 
            target2src_ner = doc.get(f'{target_lang}_2_{src_lang}_ner', {})
            for ent, idx in target2src_ner.items():
              key = doc[f'{src_lang}_items'][idx]
              #NOTE that this is an unordered match
              ner_match = [key2 for key2 in ner if ent == key2[0]]
              if not ner_match and len(ent) > 3:
                ner_match = [key2 for key2 in ner if (ent in key2[0] or (len(key2[0]) > 3 and key2[0] in ent))]
              if ner_match:
                if key in doc[f'{src_lang}_ner']: 
                  aHash = doc[f'{src_lang}_ner'][key]
                  all_labels = []
                  for key2 in ner_match:
                    all_labels.extend(list(ner[key2].keys()))
                  all_labels = set(all_labels)
                  found = False
                  for label in list(aHash.keys()):
                    if label in all_labels or 'MISC' in all_labels:
                      aHash[label] *= 1.1
                      print ('increasing ', key, label, aHash[label])
                      found = True
                  if not found:
                    print ('not found', key, all_labels)

    if do_docs_trim:
      docs, chunks = self.trim_to_prefer_person(docs, chunks)

    if do_backtrans and target_lang != src_lang:
        #backtrans from src_lang to target_lang back to src_lang allows us to catch more NER using target lang NER tools.
        #then we tag in target_lang those items we haven't already found, and tranlsate back to match the original text.
        #NOTE: We do not modify the original text, but only use backtrans to do NER tagging and other analysis. 
        if target_is_cjk:
              sep = ""
        else:
              sep = " "
        if target_is_cjk: 
              lbracket = "[["
              rbracket = "]]"
        else:
              lbracket = "["
              rbracket = "]"
        for chunk in chunks:
          _id = chunk['id']
          text = chunk[f'{target_lang}_text'].replace("[", "{").replace("(", "{").replace(")", "}").replace("]", "}")
          offset = chunk[f'{target_lang}_offset']
          doc = docs[_id]
          offset_end = offset + len(text)
          if f'{target_lang}_items' not in doc:
            doc[f'{target_lang}_items'] = list(doc.get(f'{target_lang}_ner', {}).keys())
            doc[f'{target_lang}_items'].sort(key=lambda a: a[1])
          i = 0
          for idx, key in enumerate(doc[f'{target_lang}_items']):
            if key[1] < offset:
              continue
            if key[2] > offset_end:
              break
            if len(key[0]) < 5 and not self.cjk_detect(key[0]):
              if " "+key[0]+" " in text[i:]:
                j = text.index(" "+key[0]+" ", i)
                text = text[:j]+(text[j:].replace(" "+key[0]+" ", f"  **{idx}**  ", 1))
                i = j
            else:
              if key[0] in text[i:]:
                j = text.index(key[0], i)
                text = text[:j]+(text[j:].replace(key[0], f"  **{idx}**  ", 1))
                i = j
          chunk[f'{target_lang}_tmpl_text'] = text

        target_items_sorted = list(enumerate(doc[f'{target_lang}_items']))
        target_items_sorted.sort(key=lambda a: len(a[1][0]))
        for chunk in chunks:
          text = chunk[f'{target_lang}_tmpl_text']
          _id = chunk['id']
          doc = docs[_id]
          for idx, key in target_items_sorted:
            if len(key[0]) < 5 and not self.cjk_detect(key[0]):
              text = text.replace(" "+key[0]+" ", f"  **{idx}**  ")
            else:
              text = text.replace(key[0], f" **{idx}** ")
          chunk[f'{target_lang}_tmpl_text'] = text

        for chunk in chunks:
          text = chunk[f'{target_lang}_text']
          _id = chunk['id']
          doc = docs[_id]
          for idx, key in enumerate(doc[f'{target_lang}_items']):
            text = text.replace(f" **{idx}** ", f" {idx} {lbracket} {key[0]} {rbracket}")
          chunk[f'{target_lang}_tmpl_text'] = text.replace("  ", " ")

        backtrans_text = self.do_translations([chunk[f'{target_lang}_tmpl_text'] for chunk in chunks], src_lang=target_lang, target_lang=src_lang, batch_size=batch_size)
        for chunk, trans_text in zip(chunks, backtrans_text):
          #langid check
          try:
            lang =  langid.classify(trans_text)[0]
          except:
            lang = target_lang
          if lang == target_lang:
            chunk[f'{src_lang}_text_backtrans_from_{target_lang}'] = trans_text.lstrip(" .").replace(rbracket, "]").replace(lbracket, "[").replace("}", ")").replace("{", "(")
          else:
            chunk[f'{src_lang}_text_backtrans_from_{target_lang}'] = " . . . "
        #TODO: do similiarty test?
        for chunk, trans_text in zip(chunks, backtrans_text):
          _id = chunk['id']
          doc = docs[_id]
          orig_text = chunk[f'{src_lang}_text']
          trans_text = chunk[f'{src_lang}_text_backtrans_from_{target_lang}'] 
          items = doc[f'{target_lang}_items']
          len_items = len(items)
          doc[f'{src_lang}_2_{target_lang}_backtrans_ner'] = ner = doc.get(f'{src_lang}_2_{target_lang}_backtrans_ner', {})
          pos = 0
          blocks, score =  self.get_aligned_text(orig_text, trans_text, src_lang)
          prev_t = None
          prev_o = None
          ner_word = ""
          ent2 = ""
          idx = None
          for o, t, _ in blocks:
            before = after = ""
            if "]" in t:
              ner_word = ""
              ent2 = ""
              t_arr = t.split("]")
              before = sep.join(t_arr[-1:])
              after = t_arr[-1]
              before = before.strip()
              if not before: 
                continue
              idx = before.split()[-1]
              try:
                idx = int(idx)
              except:
                idx = None
                if prev_t and prev_t.strip():
                  idx = prev_t.strip().split()[-1]
                  try:
                    idx = int(idx)
                  except:
                    idx = None
                    pass  
            if idx is not None and idx < len_items:
              ner_word += o
              if after:
                ent2 = after.split("[", 1)[0]
              else:
                ent2 += t.split("[", 1)[0]
              if "[" in t:
                key = items[idx]
                if key in ner:
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
                    found=False
                    if  len_ent2arr > 3:
                      ent3 = sep.join(ent2arr[:3])
                      if ent3 in new_word:
                        new_word = ner_word[ner_word.index(ent3):]
                        found=True
                    if not found:
                      if len_ent2arr < len(new_wordarr):
                        new_word = sep.join(new_wordarr[-len_ent2arr:])
                  if ner_word and ner_word.lower() not in stopwords1: 
                    i = orig_text[pos:].index(ner_word)
                    start = pos + i 
                    len_nerword = len(ner_word)
                    pos = start + len_nerword
                    mention = (ner_word, offset + start, offset + start + len_nerword)
                    aHash = ner.get(mention, {})
                    for label in ner[key]:
                      print (f'found new mention from {target_lang}', mention, label)
                      aHash[label] = aHash.get(label, 0) + ner[key][label]
                    ner[mention] = aHash
                idx = None
                ner_word = ""
                ent2 = ""
            prev_o, prev_t = o, t

        # increase the src_lang ner score if we already matched this ner in src_lang or there was a partial match
        for doc in docs.values():
          bner = doc[f'{src_lang}_2_{target_lang}_backtrans_ner']
          ner = doc[f'{src_lang}_ner']
          for key, aHash in bner.items():
            if key in ner: continue
            ent = key[0]
            ner_match = [key2 for key2 in ner if ent == key2[0]]
            if not ner_match and len(ent) > 3:
              ner_match = [key2 for key2 in ner if (ent in key2[0] or (len(key2[0]) > 3 and key2[0] in ent))]
            all_keys = []
            for key2 in ner_match:
              all_keys.extend(list(ner[key2].keys()))
            all_keys = set(all_keys)
            for label in list(aHash.keys()):
              if label in all_keys or 'MISC' in all_keys:
                    aHash[label] *= 1.1
                    print ('increasing in backtrans ', key, label, aHash[label])
          for key, aHash1 in bner.items():
            ner[key] = aHash2 = ner.get(key, {})
            for key2 in aHash1:
              aHash2[key2] = aHash2.get(key2, 0.0) + aHash1[key2]

        if do_postprocessing_after_backtrans:
          pass
    
        if do_cleanup:
          #do some cleanups. we don't want any ner that are just short numbers, stopwords or single characters.
          for _id, doc in docs.items():
            ner =  doc[f'{src_lang}_ner'] 
            for key in list(doc[f'{src_lang}_ner'].keys()):
              ner_word = key[0]
              try:
                if len(ner_word) < 4 and float(ner_word):
                  print ("deleting ", ner_word)
                  del doc[f'{src_lang}_ner'][key]
                  continue
              except:
                pass
              if ner_word.lower()in stopwords1 or (not self.cjk_detect(ner_word) and len(ner_word) <= 1):
                print ("deleting ", ner_word)
                del doc[f'{src_lang}_ner'][key]

    return docs, chunks

  def process_ner(self, 
              src_lang,
              text=None,
              docs=None,
              do_spacy = True,
              do_hf_ner = True,
              do_ontology = True,
              do_backtrans=False,
              do_cleanup=True,
              do_regex = True,
              batch_size = 5, 
              batch_window=70,
              ontology_weight=0.9,
              spacy_weight=1.25,
              hf_ner_weight=1.5,
              backtrans_weight=0.9,
              do_docs_trim=True,
              do_postprocessing_after_backtrans=False,
              cutoff=None,
              target_lang='en'):
      src_is_cjk = src_lang in ('zh', 'ko', 'ja')
      if src_is_cjk:
        sep = ""
      else:
        sep = " "

      if text is None and docs is None:
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
            docs = [{f'{src_lang}_text': text.decode()} for text in open(url.split("/")[-1], "rb").readlines()]
      elif docs is None:
        if isinstance(text, str):
          docs = [{f'{src_lang}_text': text}]
        elif isinstance(text, list):
          if isinstance(text[0], dict):
            docs = text
          else:
            docs = [{f'{src_lang}_text': t} for t in text]
      #for testing only
      if cutoff is not None and len(docs) > cutoff:
        docs = docs[:cutoff]
      #print (docs)
      len_docs = len(docs)
      for doc in docs:
        doc[f'{src_lang}_text'] = doc['text']
        del doc['text']
      badwords1 = set([s for s in badwords_ac_dc.get(src_lang, []) if len(s) < 5])
      stopwords1 = set(stopwords_ac_dc.get(src_lang, []))
      docs = [doc for doc in docs if self.check_good_sentence(doc[f'{src_lang}_text'], src_lang, stopwords=stopwords1, badwords=badwords1)]
      print ('trimmed junk', (len_docs-len(docs))/len_docs)
      len_d = len(docs)
      #badwords2 = set([s for s in badwords_ac_dc.get(target_lang, []) if len(s) < 5])
      
      counter = {}
      chunks = []
      _id = -1
      for doc in docs:
        _id += 1
        if 'id' not in doc:
          doc['id'] = str(_id)
        doc[f'{src_lang}_text'] = doc[f'{src_lang}_text'].replace("[", "(").replace("]", ")") # we use [] as special chars
        doc['lang'] = src_lang
        doc['domain'] = domain
        doc['chunks'] = [] 
        offset = 0
        if src_is_cjk:
          textarr = doc[f'{src_lang}_text']
        else:
          textarr = doc[f'{src_lang}_text'].split()
        if True:
          text = []
          for t in textarr:
            punc_found = [punc for punc in t if punc in self.punc_char] 
            if punc_found and t[-1] not in self.punc_char and t[0] not in "0123456789" and t[0] == t[0].lower():
              w = t[t.index(punc_found[0])+1]
              if w == w.upper():
                t, t1 = t.split(punc_found[0],1)
                t = t+punc_found[0]+(" " if src_is_cjk else "") 
                text.append(t)
                text.append(t1)
                continue
            text.append(t)
          text[0] = text[0].lstrip()
          text[-1] = text[-1].rstrip()
          doc[f'{src_lang}_text'] = sep.join(text)
          len_text = len(text)
          while len_text > batch_window:
            for j in range(batch_window-1, len_text):
              if (src_is_cjk and text[j] in self.punc_char) or (not src_is_cjk and text[j][-1] in self.punc_char):
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
    
      docs = dict([(doc['id'], doc) for doc in docs])
      if do_docs_trim:
        docs2, chunks2 = self.trim_to_prefer_person(docs, chunks)
        do_docs_trim = len(docs2) == len(docs)
        docs, chunks = docs2, chunks2

      # we do this here because we don't want to trim  ner items that are considered empty.
      # we should probably fix trim_to_prefer_person to not do any trimming if all ner's are empty
      for doc in docs.values():
        doc[f'{src_lang}_ner'] = doc.get(f'{src_lang}_ner', {})
        
      #do ner processing in src_lang
      docs2, chunks2 = self.process_ner_chunks_with_trans(
                          src_lang, 
                          docs, 
                          chunks, 
                          do_spacy = do_spacy,
                          do_hf_ner = do_hf_ner,
                          do_ontology = do_ontology,
                          do_backtrans=False,
                          do_regex = do_regex,
                          do_cleanup=do_cleanup,
                          batch_size = batch_size, 
                          ontology_weight=ontology_weight,
                          spacy_weight=spacy_weight,
                          hf_ner_weight=hf_ner_weight,
                          backtrans_weight=backtrans_weight,
                          do_postprocessing_after_backtrans=False,
                          do_docs_trim=do_docs_trim)
      if do_docs_trim:
        do_docs_trim = len(docs2) == len(docs)
      docs, chunks = docs2, chunks2
      
      if target_lang != src_lang:
        #do ner processing in target language with optional backtrans
        docs2, chunks2 = self.process_ner_chunks_with_trans(
                            src_lang, 
                            docs, 
                            chunks, 
                            target_lang = target_lang,
                            do_spacy = do_spacy,
                            do_hf_ner = do_hf_ner,
                            do_ontology = do_ontology,
                            do_backtrans=do_backtrans,
                            do_regex = do_regex,
                            do_cleanup=do_cleanup,
                            batch_size = batch_size, 
                            ontology_weight=ontology_weight,
                            spacy_weight=spacy_weight,
                            hf_ner_weight=hf_ner_weight,
                            backtrans_weight=backtrans_weight,
                            do_postprocessing_after_backtrans=True,
                            do_docs_trim=do_docs_trim)
        docs, chunks = docs2, chunks2
      return docs, chunks
