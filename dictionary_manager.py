# coding=utf-8
# Copyright, 2021-2022 Ontocord, LLC, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Un3less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#from random import sample
import glob, os, re
import gzip
import os, argparse
import itertools
from collections import Counter, OrderedDict
import os
import json
import threading
import numpy as np
import os
import time
import json
import copy

from time import time
import numpy as np
from collections import Counter
from itertools import chain
import glob
import json
import math, os
import random
import transformers
import sys, os
import json
import gzip
from tqdm import tqdm
from collections import Counter
import re
import gzip
import urllib
import re
from transformers import AutoTokenizer
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s : %(processName)s : %(levelname)s : %(message)s',
    level=logging.INFO)

try:
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           os.path.pardir)))
except:
  sys.path.append(os.path.abspath(os.path.join("./",
                                           os.path.pardir)))
from cjk import *
from char_manager import *
from stopwords import stopwords as all_stopwords
default_data_dir = os.path.abspath(os.path.join(onto_dir, "data"))
    
mt5_tokenizer = None
lexicon = None

def cjk_tokenize_text(text, connector="_"):
        """ tokenize using mt5. meant for cjk languages"""
        global mt5_tokenizer
        if mt5_tokenizer is None:
            mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        words = mt5_tokenizer.tokenize(text.replace("_", " ").replace("  ", " ").strip())

        words2 = []
        for word in words:
            if not words2:
                words2.append(word)
                continue
            if not cjk_detect(word):
                if not cjk_detect(words2[-1]):
                    if words2[-1] in strip_chars_set:
                        words2[-1] += " " + word
                    else:
                        words2[-1] += word
                    continue
            words2.append(word)
        text = connector.join(words2).replace(mt5_underscore, " ").replace("  ", " ").replace("  ", " ").strip()
        return text

      
def detect_in_dictionary(text, src_lang="en", stopwords=None, tag_type={'PERSON', 'PUBLIC_FIGURE'}, dictionary=None, connector="_", word2ngram=None, supress_cjk_tokenize=False, check_person_org_loc_caps=True, collapse_consecutive_ner=False, fix_abbreviations=True, label2label=None):
        """
        :text: the text to detect NER in. 
        :dictionary: word->label
        :word2ngram: startword->(min_ngram, max_ngram)
        :src_lang: the language of this text if known
        :stopwords: the stopwrods for src_lang
        :connector: the connector to use between words in a compound word
        :tag_type: the type of NER labels to collect
        :supress_cjk_tokenize: whether to supress tokenizing using mt5
        :check_person_org_loc_caps: check to see if a PERSON, ORG or LOC starts with a capitalized word and ends with a capitalized word
        :collapse_consecutive_ner: whether to collapse consecutive words tagged with the same lable to collapse into one lable.  John/PERSON Smith/PERSON -> John_Smith/PERSON
        :fix_abbreviations: whether to fix this error in tokenizing:  U.S.A . => U.S.A.
        :label2label: a dict that maps from underlying ontology labels to labels used by the caller of detect.
        
        Returns: a  list of 4 tuples of [(entity, start, end, label)...
        
        This function detect NER in a text using a simple dictionary lookup. 
        For compound words, transform into single word sequence, with a word potentially
        having a connector seperator.  Optionally, use the mt5 tokenizer
        to separate the words into subtokens first, and then do multi-word
        parsing.  Used for mapping a word back to an item in an ontology.
        Returns the tokenized text along with word to ner label mapping
        for words in this text.
        """
        global lexicon, default_data_dir
        
        if dictionary is None:
          if lexicon is None:
             if os.path.exists(default_data_dir+"/lexicon.json.gz"):
                 with gzip.open(default_data_dir+"/lexicon.json.gz", 'r') as fin:
                    lexicon = json.loads(fin.read().decode('utf-8'))
                else:
                    lexicon = json.load(open(default_data_dir+"/lexicon.json", "rb"))
          dictionary = lexicon
        if stopwords is None:
          stopwords = all_stopwords.get(src_lang, {})
        labels = []
        if not supress_cjk_tokenize and cjk_detect(text):
            text = cjk_tokenize_text(text, connector)
        sent = text.strip().split()
        len_sent = len(sent)
        pos = 0
        for i in range(len_sent - 1):
            #print (i, sent[i], sent)
            if sent[i] is None: continue
            start_word = sent[i].lower().lstrip(strip_chars)
            if start_word in stopwords:
                pos += len(sent[i]) + 1
                continue
            start_word = start_word.translate(trannum).split(connector)[0]
            start_word = start_word if len(start_word) <= word_shingle_cutoff else start_word[:word_shingle_cutoff]
            ngram_start, ngram_end = (1,5) if not word2ngram else word2ngram.get(start_word, (1,5))
            if ngram_start > 0:
                for j in range(ngram_start - 1, ngram_end - 2, -1):
                    if len_sent - i > j:
                        wordArr = sent[i:i + 1 + j]
                        new_word = connector.join(wordArr).strip(strip_chars)
                        if not has_nonstopword(wordArr): break
                        # we don't match sequences that start and end with stopwords
                        if wordArr[-1].lower() in stopwords: continue
                        label = dictionary.get(new_word)  

                        if label is not None:
                          #check_person_org_gpe_caps
                          # fix abbreviations - this is very general
                          if fix_abbreviations:
                            len_last_word =  len(sent[i + j])
                            if sent[i + j][-1] == '.' and len_last_word > 1 and len_last_word <=3:
                              new_word = new_word + "."
                          label = label if not label2label else label2label.get(label, label)
                          if new_word in stopwords: continue
                                
                          if (tag_type is None or label in tag_type) :
                            new_word = new_word.replace(" ", connector)
                            is_caps = wordArr[0][0] == wordArr[0][0].upper() and wordArr[-1][0] == wordArr[-1][0].upper()
                            if check_person_org_loc_caps and not is_caps and (
                                  "PUBLIC_FIGURE" in label or "PERSON" in label or "ORG" in label or "GPE" in label):
                              continue
                              
                            if new_word not in stopwords:
                                #print ('found word', new_word)
                                sent[i] = new_word
                                ners.append([new_word, pos, pos + len(new_word), label])
                                for k in range(i + 1, i + j + 1):
                                    sent[k] = None
                                #print (sent)
                                break
                          else:
                            #print ('****', len(new_word))
                            if len(new_word) <  20 and new_word.count(' ') < 3: #make this a param
                              #if this is a very long word we found, give the tokenizer a chance to find embedded NERs
                              if new_word not in stopwords:
                                  #print ('found word, but not labeling', label, new_word)
                                  sent[i] = new_word
                                  for k in range(i + 1, i + j + 1):
                                      sent[k] = None
                                  break
            pos += len(sent[i]) + 1

        #collapse NER types if consecutive predictions are the same
        if collapse_consecutive_ner is not None:
          #print ("collapsing")
          prev_ner = None
          ners2 = []
          for a_ner in ners:
            #print (prev_ner, ner)
            if prev_ner and a_ner[-1] == prev_ner[-1] and  prev_ner[-1] in collapse_consecutive_ner and ((prev_ner[2] == a_ner[1]) or (prev_ner[2] == a_ner[1]-1)):
              if  (prev_ner[2] == ner[1]-1): 
                ners2[-1][0] += (connector if text[a_ner[1]-1]==' ' else  text[a_ner[1]-1])+ a_ner[0]
              else:
                ners2[-1][0] += a_ner[0]
              ners2[-1][2]= a_ner[0][2
              prev_ner = a_ner
              continue
            prev_ner = a_ner
            ners2.append(a_ner)
          ners = ners2

        return ners
