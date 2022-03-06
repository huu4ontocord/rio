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
import multiprocessing

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
import faker
import gzip
from faker.providers import person, job
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
import default_onto_tags
from stopwords import stopwords

mt5_underscore = "▁"
trannum = str.maketrans("0123456789", "1111111111")

try:
  onto_dir = os.path.dirname(__file__)
except:
  onto_dir = "./"

class OntologyManager:
    """
  Basic ontology manager. Stores the upper ontology and lexicon that
  maps to the leaves of the ontology.  Has functions to determine
  whether a word is in the ontology, and to tokenize a sentence with
  words from the ontology.
  """

    default_strip_chars = "-,~`.?!@#$%^&*(){}[]|\\/-_+=<>;'\" ,،、“”《》«»!:;?。…．"
    stopwords_all= set(itertools.chain(*[list(s) for s in stopwords.values()]))
    base_onto_name = "yago_cn_wn"
    default_data_dir = os.path.abspath(os.path.join(onto_dir, "data"))

    ontology = None
    upper_ontology = None
    mt5_tokenizer = None
    word_shingle_cutoff = 3
    _max_lexicon = 0

    def __init__(self, target_lang="", data_dir=None, tmp_dir=None, compound_word_step=3,
                 strip_chars=None, \
                 upper_ontology=None, ontology_file="ontology.json.gz",
                 target_lang_data_file=None, word2ner_file=None, \
                 connector="_", label2label=None,  \
                tag_type={'PERSON', 'PUBLIC_FIGURE', }, ontology=None):
        """
         OntologyManager manages an ontology or dictionary of words, and tags and tokenizes a sentences based on the dictionary.
        """
        if OntologyManager.mt5_tokenizer  is None:
           OntologyManager.mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=True)
        self.is_cjk = -1 if target_lang == "" else 1 if target_lang in ("zh", "ja", "ko") else 0
        self.tag_type = tag_type
        self.target_lang_lexicon = {}
        self.target_lang = target_lang
        self.stopwords = set(stopwords.get(target_lang, [])) if target_lang else OntologyManager.stopwords_all
        if data_dir is None: data_dir = OntologyManager.default_data_dir
        if tmp_dir is None: tmp_dir = "/tmp/ontology/"
        os.system(f"mkdir -p {data_dir}")
        os.system(f"mkdir -p {tmp_dir}")
        OntologyManager.tmp_dir = tmp_dir
        OntologyManager.data_dir = data_dir
        if strip_chars is None:
            strip_chars = self.default_strip_chars
        self.strip_chars_set = set(strip_chars)
        self.strip_chars = strip_chars
        self.connector = connector
        self.compound_word_step = compound_word_step
        if label2label is None:
            label2label = default_onto_tags.default_label2label
        self.label2label = label2label
        if upper_ontology is None:
            upper_ontology = default_onto_tags.default_upper_ontology
        self.target_lang_dat = None
        if OntologyManager.upper_ontology is None:
          OntologyManager.load_upper_ontology(upper_ontology)
        if not OntologyManager.ontology: OntologyManager.ontology = OrderedDict()
        if ontology:  
          for onto_name, onto in ontology.items():
            OntologyManager.ontology[onto_name] = onto
        elif ontology_file:
          self.load_ontology_file(ontology_file)
        if word2ner_file is not None:
            self.load_word2ner_file(word2ner_file)
        if target_lang_data_file is None and target_lang:
            target_lang_data_file = f"{data_dir}/{target_lang}.json"
        if target_lang_data_file is not None:
            #print ("load", target_lang_data_file)
            self.load_target_lang_data(target_lang_data_file, target_lang=target_lang)
        # used for cjk processing

    @staticmethod
    def load_upper_ontology(upper_ontology):
        # TODO: load and save from json file
        if upper_ontology is None: upper_ontology = {}

        OntologyManager.upper_ontology = {}

        for key, val in upper_ontology.items():
            key = key.upper()
            if key not in OntologyManager.upper_ontology:
                OntologyManager.upper_ontology[key] = [val, len(OntologyManager.upper_ontology)]
            else:
                OntologyManager.upper_ontology[key] = [val, OntologyManager.upper_ontology[key][1]]

    def load_word2ner_file(self, word2ner_file, onto_name=None):
        if onto_name is None: 
          onto_name = OntologyManager.base_onto_name
        data_dir = OntologyManager.data_dir
        tmp_dir = OntologyManager.tmp_dir
        if word2ner_file is None: return
        if os.path.exists(word2ner_file):
            word2ner = json.load(open(word2ner_file, "rb"))
            self.add_to_ontology(word2ner, onto_name=onto_name)
        elif os.path.exists(os.path.join(data_dir, word2ner_file)):
            word2ner = json.load(open(os.path.join(data_dir, word2ner_file), "rb"))
            self.add_to_ontology(word2ner, onto_name=self.base_onto_name)
        else:
            logger.info(f"warning: could not find {word2ner_file}")

    def load_ontology_file(self, ontology_file="ontology.json.gz", tmp_dir=None, data_dir=None):
        """ Load the prefix based json file representing the base ontology (lexicon) """
        if data_dir is None: data_dir = OntologyManager.data_dir
        if tmp_dir is None: tmp_dir = OntologyManager.tmp_dir
        if ontology_file is not None:
            if not OntologyManager.ontology:
              if not os.path.exists(ontology_file):
                  ontology_file = f"{data_dir}/{ontology_file}"
              #if not os.path.exists(ontology_file):
              #    OntologyManager.ontology = {}
              if not os.path.exists(ontology_file):
                  logger.info(f"{ontology_file} does not exist")
              else:
                if ontology_file.endswith(".gz"):
                    with gzip.open(ontology_file, 'r') as fin:
                        OntologyManager.ontology = json.loads(fin.read().decode('utf-8'))
                else:
                    OntologyManager.ontology = json.load(open(ontology_file, "rb"))
              if not OntologyManager.ontology: 
                OntologyManager.ontology = OrderedDict()              
              if len(OntologyManager.ontology) > 20: 
                #for backwards compatability
                #this is probably a dictionary of ontologies and not a lexicon itself
                OntologyManager.ontology = OrderedDict({OntologyManager.base_onto_name+"0":OntologyManager.ontology })
              #print (len(OntologyManager.ontology))
              for ontology in OntologyManager.ontology.values():
                 for lexicons in ontology.values():
                   #print (lexicon)
                   for lex in lexicons[2:]:
                     for val in lex.values():
                        label = val[0][0]
                        if label in OntologyManager.upper_ontology:
                            val[0] = OntologyManager.upper_ontology[label][0] # shrink the memory down by reusing the same string
                        if len(val) > 1:
                          OntologyManager._max_lexicon = max(OntologyManager._max_lexicon, max(val[1]))
                        else:
                          OntologyManager._max_lexicon += 1
        else:
              if not OntologyManager.ontology: 
                OntologyManager.ontology = OrderedDict()


    def save_ontology_file(self, ontology_file="ontology.json.gz"):
        """ Saves the base cross lingual ontology/leixcon in the prefix format."""
        data_dir = self.data_dir
        tmp_dir = self.tmp_dir
        # print (data_dir, ontology_file)
        ontology_file = ontology_file.replace(".gz", "")
        if not ontology_file.startswith(data_dir):
            ontology_file = f"{data_dir}/{ontology_file}"
        json.dump(OntologyManager.ontology, open(ontology_file, "w", encoding="utf8"),
                  indent=1)
        os.system(f"gzip -f {ontology_file}")
        os.system(f"rm {ontology_file}")

    def load_target_lang_data(self, target_lang_data_file=None, target_lang=None):
        """ loads a langauage specific json file.  """

        data_dir = self.data_dir
        tmp_dir = self.tmp_dir
        if target_lang_data_file is None:
            if os.path.exists(os.path.join(data_dir, f'{target_lang}.json')):
                target_lang_data_file = os.path.join(data_dir, f'{target_lang}.json')
        if target_lang_data_file is None: return
        if os.path.exists(target_lang_data_file):
            self.target_lang_data = json.load(open(target_lang_data_file, "rb"))
        else:
            self.target_lang_data = {}

    def save_target_lang_data(self, target_lang_data_file):
        if target_lang_data_file is None: return
        data_dir = self.data_dir
        tmp_dir = self.tmp_dir
        json.dump(self.target_lang_data, open(f"{data_dir}/{target_lang_data_file}", "w", encoding="utf8"), indent=1)
        # os.system(f"gzip -f {data_dir}/{target_lang_data_file}")

    def _has_nonstopword(self, wordArr):
        for word in wordArr:
            if word.strip(self.strip_chars) not in self.stopwords:
                return True
        return False

    def _get_all_word_shingles(self, wordArr, word_shingle_cutoff=None, more_shingles=True):
        """  create patterned variations (prefix and suffix based shingles). will lower case everything. """
        lenWordArr = len(wordArr)
        wordArr = [w.lower() for w in wordArr]
        if word_shingle_cutoff is None: word_shingle_cutoff = self.word_shingle_cutoff
        compound_word_step = self.compound_word_step
        wordArr1 = wordArr2 = wordArr3 = wordArr4 = None
        ret = OrderedDict()
        if lenWordArr > compound_word_step:
            #TODO: we can add some randomness in how we create patterns
            wordArr1 = wordArr[:compound_word_step - 1] + [wordArr[-1]]
            wordArr2 = [wordArr[0]] + wordArr[1 - compound_word_step:]
            wordArr1 = [w if len(w) <= word_shingle_cutoff else w[:word_shingle_cutoff] for w in wordArr1]
            wordArr2 = [w if len(w) <= word_shingle_cutoff else w[:word_shingle_cutoff] for w in wordArr2]
            ret[tuple(wordArr1)] = 1 
            ret[tuple(wordArr2)] = 1
            if more_shingles:
                wordArr3 = copy.copy(wordArr1)
                wordArr3[-1] = wordArr3[-1] if len(wordArr3[-1]) <= word_shingle_cutoff else '*' + wordArr3[-1][len(
                    wordArr3[-1]) - word_shingle_cutoff + 1:]
                wordArr4 = copy.copy(wordArr2)
                wordArr4[-1] = wordArr4[-1] if len(wordArr4[-1]) <= word_shingle_cutoff else '*' + wordArr4[-1][len(
                    wordArr4[-1]) - word_shingle_cutoff + 1:]
                wordArr3 = [w if len(w) <= word_shingle_cutoff else w[:word_shingle_cutoff] for w in wordArr3]
                wordArr4 = [w if len(w) <= word_shingle_cutoff else w[:word_shingle_cutoff] for w in wordArr4]
                ret[tuple(wordArr3)] = 1 
                ret[tuple(wordArr4)] = 1
        else:  # lenWordArr <= compound_word_step
            wordArr1 = [w if len(w) <= word_shingle_cutoff else w[:word_shingle_cutoff] for w in wordArr]
            ret[tuple(wordArr1)] = 1
            if lenWordArr > 1 and more_shingles:
                wordArr2 = copy.copy(wordArr)
                wordArr2[-1] = wordArr2[-1] if len(wordArr2[-1]) <= word_shingle_cutoff else '*' +  wordArr2[-1][len(
                    wordArr2[-1]) - word_shingle_cutoff + 1:]
                wordArr2 = [w if len(w) <= word_shingle_cutoff else w[:word_shingle_cutoff] for w in wordArr2]
                ret[tuple(wordArr2)]= 1
        return [list(a) for a in ret.keys()]

    def get_word2ner_stats(self, word2ner, weight_factors=None):
        if weight_factors is None:
          weight_factors = default_onto_tags.label_weight_factors
        connector = self.connector
        label_cnt = {}
        for word_ner in word2ner:
            word = word_ner[0]
            label = word_ner[1]
            if len(word_ner) > 2:
              weight = word_ner[2]
            else: 
              weight = 1.0
            if len(word_ner) > 3:
              _idx = word_ner[3]
            else: 
              _idx = -1
            label = label.upper()
            word, wordArr = self.canonical_word(word)
            orig_lens = len(word) + len(wordArr)
            while wordArr:
                if wordArr[0] in self.stopwords:
                    wordArr = wordArr[1:]
                else:
                    break
            if not wordArr:
                continue
            # we don't have an actual count of the word in the corpus, so we create a weight based
            # on the length, assuming shorter words with less compound parts are more frequent
            weight = weight + 1 / (1.0 + math.sqrt(orig_lens))
            weight *= weight_factors.get(label, 1.0)
            label_cnt[label] = label_cnt.get(label, 0) + weight
        return Counter(label_cnt).most_common()

    @staticmethod
    def onto_level_2_word_shingle_cutoff(level):
      return OntologyManager.word_shingle_cutoff * (1+level**2)

    def canonical_word(self, word, connector=None, supress_cjk_tokenize=False, do_lower=False, do_trannum=False):
      """ Does not do trannum or lower case automatically. """
      if connector is None:
        connector = self.connector
      if self.is_cjk < 0:
        is_cjk = self.cjk_detect(word)
      else:
        is_cjk = self.is_cjk   
      if not supress_cjk_tokenize and is_cjk:
        word = self.cjk_tokenize_word(word, connector)

      orig_word = word = word.replace(" ", connector).replace(connector+connector, connector).strip(self.strip_chars + connector).replace('__', connector)
      if do_lower: word = word.lower()
      if do_trannum: word = word.translate(trannum)
      wordArr = word.split(connector)      
      # some proper nouns start with stop words like determiners. let's strip those.
      while wordArr:
        if wordArr[0] in self.stopwords:
          wordArr = wordArr[1:]
        else:
          break
        if not wordArr:
          continue
      word = connector.join(wordArr).replace('__', connector).replace(connector+connector, connector)
      if not word:
        return orig_word, orig_word.split(connector)
      return word, wordArr

    def add_to_ontology(self, word2ner, word_shingle_cutoff=None, onto_name=None, keep_idx=False, add_to_full_word2ner=False, full_word2ner=None, depth=4, max_depth=4, weight_factors={'PERSON': 5, }):
        """
        Add words to the ontology. The ontology is stored in prefixed 
        based patterns, using an upper ontology and a sequence of lexicon
        mappings that are more specific. 

        OrderedDict({'lex0': lexcion0, 'lex1': lexcion1, 'lex2': lexcion2... })

        Lex0 may use 3-grams, whereas lex1 may use 7-grams, etc.
        The last lexicon is simply a dictionary of the word => the NER info.
        
        We use the prefix based patterns to generalize the lexicon by using subsequences of
        the words and compound words.  Each word is shortened to
        word_shingle_cutoff. Compound words are connected by a connector.
        Compound words longer than compound_word_step are shortened to
        that length for storage purposes.  All words except upper ontology
        labels are lower cased.  

        Each lexicon corrects the previous lexicon's mistakes by make the prefix based patterns more precise.
        e.g., each lexicon will use a longer word_shingle_cutoff than the previous for higher precision (lower recall).

        """
        
        if weight_factors is None:
          weight_factors = {}
        if onto_name is None:
            onto_name = self.base_onto_name
        if word_shingle_cutoff is None: word_shingle_cutoff = self.onto_level_2_word_shingle_cutoff(len(OntologyManager.ontology))
        ontology = OntologyManager.ontology[onto_name+str(max_depth-depth)] = OntologyManager.ontology.get(onto_name+str(max_depth-depth), {})
        compound_word_step = self.compound_word_step
        #print ('word_shingle_cutoff', word_shingle_cutoff)
        connector = self.connector
        prefix_set = []
        #sync up the max_lexicon. make sure the word2ner list has 4 items: [word, ner, weight, idx]
        for word_ner in word2ner:
          if len(word_ner) >= 4:
             OntologyManager._max_lexicon = max(OntologyManager._max_lexicon, word_ner[3])
        #canonicolaize the word2ner
        for word_ner in word2ner:
          word = word_ner[0]
          word, wordArr = self.canonical_word(word, connector, supress_cjk_tokenize=False, do_lower=True, do_trannum=False)
          word_ner[0] = word
          if len(word_ner) >= 4:
             continue
          elif len(word_ner) == 3:
             word_ner.extend([OntologyManager._max_lexicon])
             OntologyManager._max_lexicon+= 1
            
          elif len(word_ner) == 2:
             word_ner.extend([0.0, OntologyManager._max_lexicon])
             OntologyManager._max_lexicon+= 1

        if full_word2ner is  None: 
        	full_word2ner = word2ner
        elif add_to_full_word2ner:
            full_word2ner.extend(word2ner)

        lexicon = {}
        for word_ner in tqdm(word2ner):
            word, label, orig_weight, _idx = word_ner
            label = label.upper()
            word = word.translate(trannum) #only do this for the lookup. don't change the original.
            wordArr = word.split(connector)
            #print (word, wordArr)
            orig_lens = len(word) + len(wordArr)
            # we don't have an actual count of the word in the corpus, so we create a weight based
            # on the length, assuming shorter words with less compound parts are more frequent
            if orig_weight:
              weight = orig_weight + 1 / (1.0 + math.sqrt(orig_lens))
            else:
              weight = 1 + 1 / (1.0 + math.sqrt(orig_lens))
              weight *= weight_factors.get(label, 1.0)
            lenWordArr = len(wordArr)
            bucket = lenWordArr // (compound_word_step + 1)
            if lenWordArr == 0:
                continue
            # add some randomness and only do suffix ends in some cases. TODO: we can use a config var.
            # this decreases space.
            for shingle in self._get_all_word_shingles(wordArr, word_shingle_cutoff=word_shingle_cutoff,
                                                       more_shingles=_idx % 5 == 0):
                #print ('shingle', shingle)
                if not shingle: continue
                word_shingle = connector.join(shingle)
                key = (word_shingle, bucket)
                # print (word0, word, weight)
                if type(label) in (list, tuple, set):
                    if type(label) != list:
                        label = list(label)
                    _label, idxs, _cnt = lexicon.get(key, [[_label[0]], {}, {}])
                    _idxs[label[0]] = _idxs.get(label[0], []) + [_idx]
                    if _cnt is None: _cnt = {}
                    _cnt[label[0]] = _cnt.get(label[0], 0.0) + weight
                    lexicon[key] = [_label, _idxs, _cnt, _idxs]
                else:
                    _label, _idxs, _cnt = lexicon.get(key, [[label], {}, {}])
                    _idxs[label] = _idxs.get(label, []) + [_idx]
                    if _cnt is None: _cnt = {}
                    _cnt[label] = _cnt.get(label, 0.0) + weight
                    lexicon[key] = [_label, _idxs, _cnt]
                prev_val = ontology.get(shingle[0], [1, 100])
                ontology[shingle[0]] = [max(lenWordArr, prev_val[0]),
                                        2 if lenWordArr == 2 else min(max(lenWordArr - 1, 1), prev_val[1])]
        new_word2ner = []
        del_keys = []
        unmatched = []
        #print (lexicon)
        for key in tqdm(lexicon):
            _cnt = lexicon[key][2]
            if _cnt:
                #if len(lexicon[key][1]) > 1: print (lexicon[key][1])
                label = Counter(_cnt).most_common(1)[0][0]
                lexicon[key][0] = lexicon.get(label, [[label]])[0]
                right_idx = lexicon[key][1][label]
                wrong_idx = list(itertools.chain(*[values for key, values in lexicon[key][1].items() if key != label]))
                len_wrong_idx= len(wrong_idx)
                if len_wrong_idx > len(right_idx): 
                    new_word2ner.extend(right_idx + wrong_idx)
                    unmatched.extend(right_idx + wrong_idx)
                    del_keys.append(key) 
                elif wrong_idx:
                    new_word2ner.extend(wrong_idx + random.sample(right_idx, len_wrong_idx))
                    if keep_idx:
                      lexicon[key] = lexicon[key][:2]
                    else:
                      lexicon[key] = lexicon[key][:1]
                    unmatched.extend(wrong_idx)
                else:    
                    if keep_idx:
                      lexicon[key] = lexicon[key][:2]
                    else:
                      lexicon[key] = lexicon[key][:1]

        for key in del_keys: del lexicon[key]

        for key in lexicon.keys():
            word, slot = key
            prefix = word.split(connector, 1)[0]
            if prefix in ontology:
                rec = ontology[prefix]
                if len(rec) == 2:
                    rec.append({})
                    rec.append({})
                    rec.append({})
                    rec.append({})
                lexicon2 = rec[2 + min(3, slot)]
                if connector in word:
                    word2 = '*' + connector + word.split(connector, 1)[1]
                else:
                    word2 = '*'
                lexicon2[word2] = lexicon[(word, slot)]

        del_keys = []
        for key, val in ontology.items():
          if len(val) <= 2:
            del_keys.append(key)
        for key in del_keys: del ontology[key]

        new_word2ner = (set(new_word2ner))
        if depth > 1:
            if len(new_word2ner) >  len(word2ner)/2:
                self.ontology[onto_name+str(max_depth-depth)] = {}
                new_word2ner = word2ner
            else:
                
                new_word2ner = [full_word2ner[i] for i in new_word2ner]
            logger.info(("creating lexicon of depth", depth-1, len(new_word2ner)))
            self.add_to_ontology(new_word2ner, onto_name=onto_name, full_word2ner=full_word2ner, depth=depth-1, max_depth=max_depth)
        if depth == 1:
            wrong = []
            wrong_none =  []
            for  word_ner in tqdm(full_word2ner): 
              word = word_ner[0]
              results = self.in_ontology(word, check_person_org_gpe_caps=False)
              if word_ner[1] != results[1]:
                if not results[1]:
                  wrong_none.append(word_ner)
                else:
                  wrong.append(word_ner)
                #print (word_ner, results)
            #print (len(wrong+wrong_none))             
            if wrong+wrong_none: 
              if keep_idx:
                self.ontology[onto_name+str(max_depth)] = dict([(name_ner[0], [0, 0, {'*': [[name_ner[1]], name_ner[-1]]}])  for name_ner in  wrong + wrong_none] +\
                                                               [(name_ner[0].translate(trannum), [0, 0, {'*': [[name_ner[1]], name_ner[-1]]}])   for name_ner in wrong + wrong_none])
              else:
                self.ontology[onto_name+str(max_depth)] = dict([(name_ner[0], [0, 0, {'*': [[name_ner[1],]]}]) for name_ner in  wrong + wrong_none] + \
                                                    [(name_ner[0].translate(trannum), [0, 0, {'*': [[name_ner[1]],]}]) for name_ner in  wrong + wrong_none])

        
    def in_ontology(self, word, connector=None, supress_cjk_tokenize=False, check_person_org_gpe_caps=True):
        """ find whether a word is in the ontology. 
        First we see if the word is in the target_lang_lexicon. 
        If not then we test by each of the base ontology lexicon.
        """
        orig_word = word
        
        word_shingle_cutoff = self.word_shingle_cutoff
        compound_word_step = self.compound_word_step
        if connector is None:
            connector = self.connector
        if self.is_cjk < 0:
          is_cjk = self.cjk_detect(word)
        else:
          is_cjk = self.is_cjk   
        word, wordArr = self.canonical_word(word, connector, supress_cjk_tokenize, do_lower=False, do_trannum=False)
        if not wordArr or not wordArr[0] or not wordArr[-1]:
            return word, None
        is_caps = wordArr[0][0] == wordArr[0][0].upper() and wordArr[-1][0] == wordArr[-1][
                                0].upper()
        word = word.lower()
        if word in self.target_lang_lexicon:
            return orig_word, self.target_lang_lexicon[word]
        word0 = word.translate(trannum)
        if word0 in self.target_lang_lexicon:
            return orig_word, self.target_lang_lexicon[word0]
        if is_cjk:
          word1 = word.replace(connector, "")
          if word1 in self.target_lang_lexicon:
              return orig_word, self.target_lang_lexicon[word1]
          word2 = word1.translate(trannum)
          if word2 in self.target_lang_lexicon:
              return orig_word, self.target_lang_lexicon[word2]    

        len_ontology = len(OntologyManager.ontology)
        lookup_len = len(wordArr) // (compound_word_step + 1)
        for level, ontology in reversed(list(enumerate(OntologyManager.ontology.values()))):
            if not ontology: continue
            if level == len_ontology-1:
              all_shingles=[[word], [word0]] + [] if not is_cjk else [[word1], [word2]]
              for shingleArr in all_shingles:
                if shingleArr and shingleArr[0] in ontology:
                  shingle = '*'
                  lexicon2 = ontology[shingleArr[0]][2]
                  dat = lexicon2.get(shingle, (None, None))
                  label = dat[0]
                  if label is not None:
                    if check_person_org_gpe_caps and not is_caps and (
                                  "PUBLIC_FIGURE" in label or "PERSON" in label or "ORG" in label or "GPE" in label):
                              # ideally we would keep patterns like AaA as part of the shingle to match. This is a hack.
                      continue
                    return word, label[0]  
            all_shingles = self._get_all_word_shingles(wordArr, word_shingle_cutoff=self.onto_level_2_word_shingle_cutoff(level), more_shingles=not is_cjk) 
            #print (all_shingles)
            # find patterned variations (shingles)
            for shingleArr in all_shingles:  # we can probably dedup to make it faster
                if shingleArr and shingleArr[0] in ontology:
                    if len(ontology[shingleArr[0]]) <2 + min(3, lookup_len)+1:
                      continue
                    lexicon2 = ontology[shingleArr[0]][2 + min(3, lookup_len)]
                    if len(shingleArr) > 1:
                        shingle = '*' + connector + connector.join((shingleArr[1:]))
                    else:
                        shingle = '*'
                    dat = lexicon2.get(shingle, (None, None))
                    label = dat[0]
                    if label is not None:
                      if check_person_org_gpe_caps and not is_caps and (
                                    "PUBLIC_FIGURE" in label or "PERSON" in label or "ORG" in label or "GPE" in label):
                              # ideally we would keep patterns like AaA as part of the shingle to match. This is a hack.
                        continue
                      return word, label[0]                     
        return orig_word, None

    def cjk_tokenize_word(self, word, connector=None):
        if connector is None:
            connector = self.connector
        return "_".join(self.mt5_tokenizer.tokenize(word)).replace(mt5_underscore, "_").\
                replace("__","_").replace("__", "_").strip("_")


    def cjk_tokenize_text(self, text, connector=None):
        """ tokenize using mt5. meant for cjk languages"""
        if connector is None:
            connector = self.connector
        if OntologyManager.mt5_tokenizer is None:
            OntologyManager.mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        words = OntologyManager.mt5_tokenizer.tokenize(text.replace("_", " ").replace("  ", " ").strip())

        words2 = []
        for word in words:
            if not words2:
                words2.append(word)
                continue
            if not self.cjk_detect(word):
                if not self.cjk_detect(words2[-1]):
                    if words2[-1] in self.strip_chars_set:
                        words2[-1] += " " + word
                    else:
                        words2[-1] += word
                    continue
            words2.append(word)
        text = " ".join(words2).replace(mt5_underscore, " ").replace("  ", " ").replace("  ", " ").strip()
        return text


    def _get_ngram_start_end(self, start_word):
        """ find the possible range of a compound word that starts with start_word """
        ngram_start = -1
        ngram_end = 100000
        for ontology in OntologyManager.ontology.values():
            rec = ontology.get(start_word, [ngram_start, ngram_end])
            ngram_start, ngram_end = max(ngram_start, rec[0]), min(ngram_end, rec[1])
        return ngram_start, ngram_end

    def detect(self, text, connector=None, supress_cjk_tokenize=False, check_person_org_gpe_caps=True, collapse_consecutive_ner=None):
        """
        Detect NER in a text. For compound words,
        transform into single word sequence, with a word potentially
        having a connector seperator.  Optionally, use the mt5 tokenizer
        to separate the words into subtokens first, and then do multi-word
        parsing.  Used for mapping a word back to an item in an ontology.
        Returns the tokenized text along with word to ner label mapping
        for words in this text.
        """
        word_shingle_cutoff = self.word_shingle_cutoff
        compound_word_step = self.compound_word_step
        labels = []
        if connector is None:
            connector = self.connector
        if not supress_cjk_tokenize and self.cjk_detect(text):
            text = self.cjk_tokenize_text(text, connector)
        sent = text.strip().split()
        len_sent = len(sent)
        pos = 0
        for i in range(len_sent - 1):
            #print (i, sent[i], sent)
            if sent[i] is None: continue
            start_word = sent[i].lower().lstrip(self.strip_chars)
            if start_word in self.stopwords:
                pos += len(sent[i]) + 1
                continue
            start_word = start_word.translate(trannum).split(connector)[0]
            start_word = start_word if len(start_word) <= word_shingle_cutoff else start_word[:word_shingle_cutoff]
            ngram_start, ngram_end = self._get_ngram_start_end(start_word)

            #print (start_word, ngram_start, ngram_end)
            if ngram_start > 0:
                for j in range(ngram_start - 1, ngram_end - 2, -1):
                    if len_sent - i > j:
                        wordArr = sent[i:i + 1 + j]
                        new_word = " ".join(wordArr).strip(self.strip_chars)
                        if not self._has_nonstopword(wordArr): break
                        # we don't match sequences that start and end with stopwords
                        if wordArr[-1].lower() in self.stopwords: continue
                        _, label = self.in_ontology(new_word, connector=connector, supress_cjk_tokenize=True, check_person_org_gpe_caps=check_person_org_gpe_caps)  

                        if label is not None:
                          # fix abbreviations - this is very general
                          if True:
                            len_last_word =  len(sent[i + j])
                            if sent[i + j][-1] == '.' and len_last_word > 1 and len_last_word <=3:
                              new_word = new_word + "."
                          label =self.label2label.get(label, label)
                          #print (new_word, label, len(new_word))
                          if (self.tag_type is None or label in self.tag_type) and (label in self.upper_ontology):
                            new_word = new_word.replace(" ", connector)
                            if new_word not in self.stopwords:
                                #print ('found word', new_word)
                                sent[i] = new_word
                                labels.append([[new_word, pos, pos + len(new_word)], label])
                                for k in range(i + 1, i + j + 1):
                                    sent[k] = None
                                #print (sent)
                                break
                          else:
                            #print ('****', len(new_word))
                            if len(new_word) <  20 and new_word.count(' ') < 3: #make this a param
                              #if this is a very long word we found, give the tokenizer a chance to find embedded NERs
                              if new_word not in self.stopwords:
                                  #print ('found word, but not labeling', label, new_word)
                                  sent[i] = new_word
                                  for k in range(i + 1, i + j + 1):
                                      sent[k] = None
                                  break
            pos += len(sent[i]) + 1

        #collapse NER types if consecutive predictions are the same
        if collapse_consecutive_ner is not None:
          #print ("collapsing")
          prev_label = None
          labels2 = []
          for label in labels:
            #print (prev_label, label)
            if prev_label and label[1] == prev_label[1] and  prev_label[1] in collapse_consecutive_ner and ((prev_label[0][2] == label[0][1]) or (prev_label[0][2] == label[0][1]-1)):
              if  (prev_label[0][2] == label[0][1]-1): 
                labels2[-1][0][0] += (connector if text[label[0][1]-1]==' ' else  text[label[0][1]-1])+ label[0][0]
              else:
                labels2[-1][0][0] += label[0][0]
              labels2[-1][0][-1] = label[0][-1]
              prev_label = label
              continue
            prev_label = label
            labels2.append(label)
          labels = labels2

        return dict([(tuple(a), b) for a, b in labels])

    def tokenize(self, text, connector=None, supress_cjk_tokenize=False, return_dict=True, check_person_org_gpe_caps=True, collapse_consecutive_ner=None):
      """
      Parse text for words in the ontology, with detected words separated by underscores (connectors).
      """
      ner = self.detect(text, connector=connector, supress_cjk_tokenize=supress_cjk_tokenize,  check_person_org_gpe_caps=check_person_org_gpe_caps, collapse_consecutive_ner=collapse_consecutive_ner)
      if connector is None:
            connector = self.connector
      text2 = ""
      prev_pos = 0
      for span, label in ner.items():
        if span[1] >0:
          text2 += text[prev_pos:span[1]]+span[0].replace(" ", connector)
        else:
          text2 += span[0].replace(" ", connector)
        prev_pos = span[2]
        continue
      text2 += text[prev_pos:]
      text = text2
      if return_dict:
        return {'text': text, 'chunk2ner': ner}
      else:
        return text

    def cjk_detect(self, texts):
        # chinese
        if re.search("[\u4e00-\u9FFF]", texts):
            return "zh"
        # korean
        if re.search("[\uac00-\ud7a3]", texts):
            return "ko"
        # japanese
        if re.search("[\u3040-\u30ff]", texts):
            return "ja"

        return None


if __name__ == "__main__":
    data_dir = tmp_dir = None
    if "-s" in sys.argv:
        tmp_dir = sys.argv[sys.argv.index("-s") + 1]
    if "-t" in sys.argv:
        sentence = sys.argv[sys.argv.index("-t") + 1]
        manager = OntologyManager(data_dir=data_dir, tmp_dir=tmp_dir)
        txt = manager.tokenize(sentence)
        print(txt)
