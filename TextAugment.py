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
import os
import re
import fsspec
from tqdm import tqdm
import difflib
import langid
import json
import random
import logging
from typing import List, Dict, Optional

import torch
from torch.nn.functional import cosine_similarity
import spacy
from nltk.corpus import stopwords
from ontology.ontology_manager import OntologyManager

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    MarianMTModel,
    pipeline,
)
from sentence_transformers import SentenceTransformer

import qg_pipeline
from utils import (
    rulebase,
    banned_words,
    hf_ner_model_map,
    mariam_mt,
    LoggingHandler,
    get_oscar_urls,
    download_urls,
    CharManager,
    get_docs,
    badwords as badwords_ac_dc,
    stopwords as stopwords_ac_dc,
)

# torch.cuda.empty_cache()
# labse = SentenceTransformer("sentence-transformers/LaBSE").half().eval().cuda()
# qg = qg_pipeline.pipeline("multitask-qa-qg")
#
# ontology_manager = None  # OntologyManager(target_lang='en') #target_lang=target_lang
# translation_pipelines = {}
# ner_model_name2pipelines = {}

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


class TextAugment:
    # don't add a space for junk chars
    max_stoword_len_zh = max([len(a) for a in stopwords_ac_dc.get('zh', [''])])
    max_stoword_len_ko = max([len(a) for a in stopwords_ac_dc.get('ko', [''])])
    max_stoword_len_ja = max([len(a) for a in stopwords_ac_dc.get('ja', [''])])

    def __init__(self,
                 ontology_manager: OntologyManager = None,
                 translation_pipelines: Dict = {},
                 ner_model_name2pipelines: Dict = {},
                 qg=None,
                 production_mode: bool = False):

        self.ontology_manager = ontology_manager
        self.translation_pipelines = translation_pipelines
        self.ner_model_name2pipelines = ner_model_name2pipelines
        self.qg = qg
        self.banned_words = banned_words
        self.hf_ner_model_map = hf_ner_model_map

        self.strip_chars = CharManager.strip_chars
        self.punc_char = CharManager.punc_char
        self.special_char = CharManager.special_char
        self.junk = CharManager.junk

        # models
        # labse model from sentence transformers
        self.labse = SentenceTransformer("sentence-transformers/LaBSE").half().eval().cuda()

        # spacy
        self.en_spacy_nlp = spacy.load('en_core_web_sm')

        # m2m models for translation
        self.m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").eval().half().cuda()
        self.m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

        # TODO: OntologyManager init has changed
        if production_mode:  # use the below for production usage. the above is for testing.
            if not self.ontology_manager: self.ontology_manager = OntologyManager()  # src_lang=src_lang
        logging.info("Finished Loading TextAugment")

    def check_good_sentence(self, s, src_lang, stopwords, stopword_ratio_cutoff=0.06, bannedwords=None, badwords=None,
                            badword_ratio_cutoff=0.15, junk_ratio=0.16, max_badword_len=5):
        # basic dejunk
        # for badwords, only filter out if the ratio is exceeded AND there exists one banned word
        if bannedwords is None:
            bannedwords = self.banned_words.get(src_lang, self.banned_words['default'])
        default_bannedwords = self.banned_words['default']
        s = s.lower().strip()
        if not s: return False
        # print ([s2 for s2 in s if s2 in self.junk])
        # print (len([s2 for s2 in s if s2 in self.junk]), len(s))
        jr = len([s2 for s2 in s if s2 in self.junk]) / len(s)
        if jr >= junk_ratio:
            return False
        if src_lang in ("ja", "ko", "zh"):
            sArr = s
        else:
            sArr = [s2.strip(self.special_char) for s2 in s.lower().split() if s2.strip(self.special_char)]
        if len(sArr) == 0:
            return False
        # stopword check
        if stopwords is not None:
            # TODO: catch multi word with spaces
            # print ('sw', len([s2 for s2 in sArr if s2 in stopwords])/len(sArr))
            if src_lang not in ("ja", "ko", "zh") and len([s2 for s2 in sArr if s2 in stopwords]) / len(
                    sArr) < stopword_ratio_cutoff:
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
                    for j in range(i + 1, min(len_s, i + max_stoword)):
                        if s[i:j] in stopwords:
                            stop_cnt += 1
                        total_cnt += 1
                # print ('stopword', (stop_cnt/total_cnt) )
                if (stop_cnt / total_cnt) < stopword_ratio_cutoff:
                    return False
        if badwords is not None:
            # print ('bw', len([s2 for s2 in sArr if s2 in badwords])/len(sArr))
            if src_lang not in ("ja", "ko", "zh") and len([s2 for s2 in sArr if s2 in badwords]) / len(
                    sArr) > badword_ratio_cutoff:
                if any(s2 for s2 in sArr if s2 in bannedwords) or any(s2 for s2 in sArr if s2 in default_bannedwords):
                    return False
            if src_lang in ("ja", "ko", "zh"):
                badword_ratio_cutoff /= 100
                len_s = len(s)
                bad_cnt = 0
                total_cnt = 0
                for i in range(len_s):
                    for j in range(i + 1, min(len_s, i + max_badword_len)):
                        if s[i:j] in badwords:
                            bad_cnt += 1
                        total_cnt += 1
                if (bad_cnt / total_cnt) > badword_ratio_cutoff:
                    for bword in bannedwords:
                        if bword in s:
                            return False
                    for bword in default_bannedwords:
                        if bword in s:
                            return False
        # langid check
        try:
            lang = langid.classify(s)[0]
        except:
            return True
        return lang == src_lang

    def generate_questions(self, batch, default_answers=[]):
        answers = {}

        i = 0
        allqa = []
        for chunk in batch:
            text = chunk['text']
            answers1 = {}
            # ti = time.time()
            text = text.replace("U.S.", "US").replace("\n", " ").replace(",", " , ").replace("  ", " ").strip().replace(
                " , ", ", ")  # replace(" He ", " Lincoln ").replace(" he ", " Lincoln ").replace(" him ", " Lincoln ")
            aHash = self.qg(text)  # , default_answers=default_answers)
            allqa.append(aHash)
            default_answers = list(set([a['answer'] for a in aHash] + default_answers))
            print(aHash)
            # for aHash1 in aHash:
            #  extraction = vis.parse(list(dep_parser(aHash1['question']).sents)[0], aHash1['answer'])
            #  print (extraction.arg1, '*', extraction.rel, '*', extraction.arg2)

            for aHash1 in aHash:
                if answers.get(aHash1['answer'].lower()) or answers1.get(aHash1['answer'].lower()):
                    continue
                if len(aHash1['answer'].split()) > 10:
                    aHash1['answer'] = " ".join(aHash1['answer'].split()[:10])
                i += 1
                quest = aHash1['question'].lower().strip("?").replace("'s", " 's").replace("  ", " ").split()
                label = ""
                # TODO, use spacy_en to get NER and only fall back to "who", "when", "where" to determine ner if we find nothing
                if quest[0] == "who" and aHash1['answer'][-1] == 's':
                    label = "organization_" + str(i)
                    if "'s" in quest:
                        for j in range(len(quest)):
                            if j > 0 and quest[j - 1] == "'s":
                                label = quest[j] + "_" + str(i)
                                break
                    for a in aHash1['answer'].lower().split():
                        if a not in stopwords_hash:
                            answers[a] = label
                elif quest[0] == "who":
                    label = "person_" + str(i)
                    if "'s" in quest:
                        for j in range(len(quest)):
                            if j > 0 and quest[j - 1] == "'s":
                                label = quest[j] + "_" + str(i)
                                break
                    for a in aHash1['answer'].lower().split():
                        if a not in stopwords_hash:
                            answers[a] = label
                elif quest[0] == "where":
                    label = "location_" + str(i)
                elif quest[0] == "when":
                    label = "date_or_time_" + str(i)
                elif quest[0] == "why":
                    label = "reason_" + str(i)
                elif quest[0] == "how" and quest[1] in ("much", "many"):
                    label = "quantity_" + str(i)
                elif quest[0] == "how":
                    label = "method_" + str(i)
                elif quest[0] in ("which", "what") and quest[1] not in stopwords_hash:
                    label = quest[1] + "_" + str(i)
                elif "'s" in quest:
                    for j in range(len(quest)):
                        if j > 0 and quest[j - 1] == "'s":
                            label = quest[j] + "_" + str(i)
                            break
                if label:
                    answers[aHash1['answer'].lower()] = label

                # for b in a['answer'].lower().split():
                #  answers[b] = label
            print(answers)

        for aHash in allqa:
            answers1 = {}
            for aHash1 in aHash:
                if answers1.get(aHash1['answer'].lower()):
                    continue
                quest = " " + aHash1['question'].lower().strip("?").replace("'s", " 's").replace("  ", " ") + " "
                q_type = quest[0]
                agent = []
                answer_keys = list(answers.keys())
                answer_keys.sort(key=lambda k: len(k), reverse=True)
                for a in answer_keys:
                    if " " + a + " " in quest:
                        quest = quest.replace(" " + a + " ", " " + answers[a] + " ")
                    elif " " + a + ", " in quest:
                        quest = quest.replace(" " + a + ", ", " " + answers[a] + ", ")
                quest = quest.split()
                # print (quest)
                qtype = []
                if answers.get(aHash1['answer'].lower()):
                    if answers.get(aHash1['answer'].lower()).split("_")[0] == "person":
                        qtype = ["is", "who"]
                if not qtype and quest[0] in ("when", "where", "why", "how"):  # , "which"
                    qtype = [quest[0]]
                    if quest[0] == "how" and quest[1] in ("much", "many"):
                        qtype = qtype + [quest[1]]

                # label=[q for q in quest if (q not in ("much", "many",) and not stopwords_hash.get(q) and q not in answers)]#+qtype
                label = [q for q in quest if (q[0] not in "0123456789") and (q not in ("the", "a", "an"))]
                if len(label) > 10:
                    label = label[:10]
                answers1[aHash1['answer'].lower()] = " ".join(label)
            print(answers1)

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
        aMatch = difflib.SequenceMatcher(None, sent1, sent2)
        score = aMatch.ratio()
        blocks = aMatch.get_matching_blocks()
        blocks2 = []
        prevEndA = 0
        prevEndB = 0
        matchLen = 0
        nonMatchLen = 0
        # print (blocks)
        for blockI in range(len(blocks)):
            if blockI > 0 or (blockI == 0 and (blocks[blockI][0] != 0 or blocks[blockI][1] != 0)):
                blocks2.append(
                    [sep.join(sent1[prevEndA:blocks[blockI][0]]), sep.join(sent2[prevEndB:blocks[blockI][1]]), 0])
                nonMatchLen += max(blocks[blockI][0] - prevEndA, blocks[blockI][1] - prevEndB)
            if blocks[blockI][2] != 0:
                blocks2.append([sep.join(sent1[blocks[blockI][0]:blocks[blockI][0] + blocks[blockI][2]]),
                                sep.join(sent2[blocks[blockI][1]:blocks[blockI][1] + blocks[blockI][2]]), 1])
                prevEndA = blocks[blockI][0] + blocks[blockI][2]
                prevEndB = blocks[blockI][1] + blocks[blockI][2]
                matchLen += blocks[blockI][2]
        # score = float(matchLen+1)/float(nonMatchLen+1)
        return (blocks2, score)

    def do_translations(self, texts, src_lang='en', target_lang='hi', batch_size=16, do_mariam_mt=False):
        if not do_mariam_mt:
            try:
                self.m2m_tokenizer.src_lang = src_lang
                target_lang_bos_token = self.m2m_tokenizer.get_lang_id(target_lang)
                translations = []
                for src_text_list in tqdm(self.batch(texts, batch_size)):
                    try:
                        batch = self.m2m_tokenizer(src_text_list, return_tensors="pt", padding=True,
                                                   truncation=True).to('cuda')
                    except:
                        break
                    gen = self.m2m_model.generate(**batch, forced_bos_token_id=target_lang_bos_token,
                                                  no_repeat_ngram_size=4, )  #
                    outputs = self.m2m_tokenizer.batch_decode(gen, skip_special_tokens=True)
                    translations.extend(outputs)
                return translations
            except:
                pass
        translations = []
        # mariam_mt = self.mariam_mt
        model_name = mariam_mt.get((src_lang, target_lang))
        mt_pipeline = None
        if model_name is not None and model_name not in self.translation_pipelines:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).half().eval().cuda()
            mt_pipeline = pipeline("translation", model=model, tokenizer=tokenizer, device=0)
            self.translation_pipelines[model_name] = mt_pipeline
            if not mt_pipeline:
                raise RuntimeError(
                    "no translation pipeline")  # we could do multi-step translation where there are no pairs
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

        offset_key = f'{src_lang}_offset'
        text_key = f'{src_lang}_text'
        ner_key = f'{src_lang}_ner'
        results_arr = hf_pipeline([chunk[text_key] for chunk in chunks])
        results_arr2 = []
        offset = 0
        for chunk, results in zip(chunks, results_arr):
            text = chunk[text_key]
            _id = chunk['id']
            ner = docs[_id][ner_key] = docs[_id].get(ner_key, {})
            offset = chunk[offset_key]
            len_text = len(text)
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
                            if start - j == -1 or text[start - j] in self.strip_chars:
                                start = max(start - j, 0)
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
                ner_result['start'] = start + offset
                ner_result['end'] = end + offset
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
            prev_word = [0, 0]
            prev_label = None
            prev_word2 = ""
            for ner_result in results:
                start = ner_result['start']
                if start is None:
                    prev_word2 = ""
                    continue
                end = ner_result['end']
                if text[start:end] != ner_result['word']:
                    print('offset mismatch', text[start:end], ner_result['word'])
                if "-" in ner_result['entity']:
                    _, label = ner_result['entity'].split('-')
                else:
                    label = ner_result['entity']
                if label in ('STREET_ADDRESS',):
                    label = 'STREET_ADDRESS'
                elif label in ('PUBLIC_FIGURE',):
                    label = 'PUBLIC_FIGURE'
                elif label in ('NAME', 'PER', 'PERSON'):
                    label = 'PERSON'
                elif label in ('LOCATION', 'LOC', 'GPE'):
                    label = 'GPE'
                elif label in ('ORGANIZATION', 'ORG'):
                    label = 'ORG'
                elif label in ('AGE',):
                    label = 'AGE'
                elif label in ('NORP',):
                    label = 'NORP'
                elif label in ('BIO', 'SYMPTOM_AND_DISEASE', 'DISEASE'):
                    label = 'DISEASE'
                elif label in ('PATIENT_ID', 'GOVT_ID'):
                    label = 'GOVT_ID'
                elif label in ('USER_ID',):
                    label = 'USER_ID'
                elif label in ('MISC',) and '@' in ner_result['word']:
                    label = 'USER_ID'
                else:
                    label = 'MISC'
                if prev_label is not None:
                    if not ner_result['entity'].startswith('B-') and label == prev_label and (
                            prev_word[1] >= start - 5):
                        prev_word[1] = max(prev_word[1], end)
                        prev_word2 = prev_word2 + " " + ner_result['word']
                    else:
                        if ner_result['entity'].startswith('B-'):
                            if prev_word[1] > start:
                                prev_word[1] = start
                        if prev_word[0] != prev_word[1]:
                            ner_word = text[prev_word[0]:prev_word[1]]
                            # if ner_word != prev_word2:
                            #  print (ner_word, '**', prev_word2)
                            # ner_word.strip(self.strip_chars)
                            mention = (ner_word, prev_word[0], prev_word[1])
                            if ner_word and ner_word.lower() not in stopwords:
                                aHash = ner.get(mention, {})
                                aHash[prev_label] = aHash.get(prev_label, 0) + weight * (1.0 + len(ner_word) / 100)
                                ner[mention] = aHash
                            prev_word = [start, end]
                            prev_word2 = ner_result['word']
                elif prev_label is None:
                    prev_word = [start, end]
                    prev_word2 = ner_result['word']
                prev_label = label

            if prev_label is not None and prev_word[0] != prev_word[1]:
                ner_word = text[prev_word[0]:prev_word[1]]
                # if ner_word != prev_word2:
                #  print (ner_word, '**', prev_word2)
                mention = (ner_word, prev_word[0], prev_word[1])
                if ner_word and ner_word.lower() not in stopwords:
                    aHash = ner.get(mention, {})
                    aHash[prev_label] = aHash.get(prev_label, 0) + weight * (1.0 + len(ner_word) / 100)
                    ner[mention] = aHash

    def _spacy_pipeline(self, lang: str=None):
        spacy_nlp = None
        try:
            spacy_nlp = spacy.load(f"{lang}_core_web_sm")
        except:
            logging.error(f"Failed to load spacy pipeline for {lang} at {lang}_core_web_sm")
        finally:
            return spacy_nlp

    def spacy_ner(self, docs, nlp, stopwords, spacy_weight, src_lang, extra_weight=1.0):
        """ 
        Use the spacy models to create mentions w/ NER
        """
        if not nlp:
            return
        if stopwords is None:
            stopwords = set(ac_dc_stopwords.get(src_lang, []))
        offset_key = f'{src_lang}_offset'
        text_key = f'{src_lang}_text'
        ner_key = f'{src_lang}_ner'
        for doc in docs.values():
            ner = doc[ner_key] = doc.get(ner_key, {})
            text = doc[text_key]
            doc = nlp(text)
            entities = list(doc.ents)
            ents = [(entity.text, entity.label_ if (
                    entity.label_ in ('PERSON', 'GPE', 'ORG', 'NORP') and 'http:' not in entity.text) else 'MISC')
                    for entity in entities]
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
                        mention = (ner_word, i, i + len(ner_word))
                        aHash = ner.get(mention, {})
                        aHash[label] = aHash.get(label, 0) + spacy_weight * (1.0 + len(ner_word) / 100) * extra_weight
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
                    ner = doc[key]
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
        logging.info(f'trim_to_prefer_person {str((len_docs - len(docs2)) / len_docs)}')
        return docs2, chunks2

    def _split_text_into_chunks(self,
                                src_lang: str,
                                src_is_cjk: bool,
                                doc: Dict,
                                batch_window: int,
                                sep: str,
                                chunks: [Dict],
                                ) -> None:
        offset = 0
        text = []

        textarr = doc[f'{src_lang}_text'] if src_is_cjk else doc[f'{src_lang}_text'].split()

        for t in textarr:
            punc_found = [punc for punc in t if punc in self.punc_char]
            if punc_found and t[-1] not in self.punc_char and t[0] not in "0123456789" and t[0] == t[0].lower():
                w = t[t.index(punc_found[0]) + 1]
                if w == w.upper():
                    t, t1 = t.split(punc_found[0], 1)
                    t = t + punc_found[0] + (" " if src_is_cjk else "")
                    text.append(t)
                    text.append(t1)
                    continue
            text.append(t)
        text[0] = text[0].lstrip()
        text[-1] = text[-1].rstrip()
        doc[f'{src_lang}_text'] = sep.join(text)
        len_text = len(text)
        while len_text > batch_window:
            for j in range(batch_window - 1, len_text):
                if (src_is_cjk and text[j] in self.punc_char) or (
                        not src_is_cjk and text[j][-1] in self.punc_char):
                    break
            text_str = sep.join(text[:j + 1])
            chunks.append({f'{src_lang}_text': text_str, 'id': doc['id'], f'{src_lang}_offset': offset})
            doc['chunks'].append(chunks[-1])
            offset += len(text_str) + (0 if src_is_cjk else 1)
            text = text[j + 1:]
            len_text = len(text)
        if text:
            text_str = sep.join(text)
            chunks.append({f'{src_lang}_text': text_str, 'id': doc['id'], f'{src_lang}_offset': offset})
            doc['chunks'].append(chunks[-1])

    def process_ner_chunks_with_trans(self,
                                      src_lang,
                                      docs,
                                      chunks,
                                      target_lang=None,
                                      do_spacy=True,
                                      do_hf_ner=True,
                                      do_ontology=True,
                                      do_backtrans=False,
                                      do_regex=True,
                                      do_cleanup=True,
                                      batch_size=5,
                                      batch_window=70,
                                      ontology_weight=0.9,
                                      spacy_weight=1.25,
                                      hf_ner_weight=1.0,
                                      backtrans_weight=0.9,
                                      do_postprocessing_after_backtrans=False,
                                      do_docs_trim=False):
        if target_lang is None:
            target_lang = src_lang
            do_backtrans = False
            backtrans_weight = 1.0

        if target_lang == src_lang and do_backtrans:
            do_backtrans = False
            logging.warning(f"Warning src_lang==target_lang={src_lang} but do_backtrans=={True}")
            logging.info(f"Set do_backtrans=={False}")

        stopwords1 = set(stopwords_ac_dc[src_lang])
        stopwords2 = set(stopwords_ac_dc[target_lang])
        ner_pipelines = []

        # init spacy pipeline
        spacy_nlp = self._spacy_pipeline(target_lang) if do_spacy else None



        # init hf ner pipelines
        if do_hf_ner:
            for model_name, model_cls, hf_ner_weight2 in self.hf_ner_model_map.get(target_lang, []):
                if model_name not in self.ner_model_name2pipelines:
                    logging.info(f"adding {model_name} into ner_model_name2pipelines")
                    # TODO: specific tf or pytroch in hf_ner_model_map to avoid error
                    ner_pipeline = None
                    try:
                        ner_pipeline = pipeline("ner", model=model_name,
                                                tokenizer=(model_name, {"use_fast": True},), device=0)
                    except:
                        ner_pipeline = pipeline("ner", model=model_name,
                                                tokenizer=(model_name, {"use_fast": True},), framework="tf",
                                                device=0)
                    finally:
                        self.ner_model_name2pipelines[model_name] = ner_pipeline
                ner_pipelines.append((self.ner_model_name2pipelines[model_name], hf_ner_weight2))

        target_is_cjk = target_lang in ('zh', 'ko', 'ja')
        src_is_cjk = src_lang in ('zh', 'ko', 'ja')

        if do_backtrans:
            # translate from src_lang to target_lang and do ner in target_lang.  translation also acts as an error check and additional ner.
            # we check to see if the already tagged items in src lang should have scores for tags increased or are common words in target lang and should not be tagged.
            # we also add new labels for items that are already tagged in src_lang.

            sep = "" if src_is_cjk else " "

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
                        if " " + key[0] + " " in text[i:]:
                            j = text.index(" " + key[0] + " ", i)
                            text = text[:j] + (text[j:].replace(" " + key[0] + " ", f"  **{idx}**  ", 1))
                            i = j
                    else:
                        if key[0] in text[i:]:
                            j = text.index(key[0], i)
                            text = text[:j] + (text[j:].replace(key[0], f"  **{idx}**  ", 1))
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
                        text = text.replace(" " + key[0] + " ", f"  **{idx}**  ")
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

            # print ('*****', chunks2)
            chunks2 = [chunk[f'{src_lang}_tmpl_text'] for chunk in chunks]
            text2 = self.do_translations(chunks2, src_lang=src_lang, target_lang=target_lang, batch_size=batch_size)
            for chunk, trans_text in zip(chunks, text2):
                # langid check
                try:
                    lang = langid.classify(trans_text)[0]
                except:
                    lang = target_lang
                if lang == target_lang:
                    chunk[f'{target_lang}_text'] = trans_text.lstrip(" .").replace(rbracket, "]").replace(lbracket,
                                                                                                          "[").replace(
                        "}", ")").replace("{", "(")
                else:
                    chunk[f'{target_lang}_text'] = " . . . "

            all_embed = self.labse.encode(chunks2, convert_to_tensor=True)
            all_trans_embed = self.labse.encode([chunk[f'{target_lang}_text'] for chunk in chunks],
                                                convert_to_tensor=True)
            similarity = cosine_similarity(all_embed, all_trans_embed, dim=1)
            for chunk, sim_score in zip(chunks, similarity):
                trans_text = chunk[f'{target_lang}_text']
                sim_score = sim_score.item()
                print(sim_score, '**', trans_text, '**', chunk[f'{src_lang}_tmpl_text'])
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
                    before, after = trans_text.split("[", 1)
                    before = before.strip()
                    after = after.strip()
                    before_arr = before.split()
                    if "]" not in after or not before_arr:
                        trans_text = before + sep + after
                        continue
                    idx = before_arr[-1]
                    ent, after = after.split("]", 1)
                    ent = ent.strip()

                    try:
                        idx = int(idx)
                    except:
                        idx = None
                    if idx is not None and idx < len_items:
                        before = " ".join(before_arr[:-1])
                        key = doc[f'{src_lang}_items'][idx]
                        ent_lower = ent.lower()
                        if ent_lower in stopwords2:
                            # reduce weight of target labels if this is translated into an en stopword
                            if key in doc[f'{src_lang}_ner']:
                                aHash = doc[f'{src_lang}_ner'][key]
                                for key in list(aHash.keys()):
                                    aHash[key] /= 2.0
                        else:
                            vals = list(doc[f'{src_lang}_ner'][key].keys())
                            ent = ent.strip(self.strip_chars)
                            doc[f'{target_lang}_2_{src_lang}_ner'][ent] = idx
                    trans_text = before + " " + ent + " " + after
                trans_text = chunk[f'{target_lang}_text'] = trans_text.replace("  ", "").strip()
                if doc.get(f'{target_lang}_text', ""):
                    chunk[f'{target_lang}_offset'] = len(doc.get(f'{target_lang}_text', "")) + 1
                else:
                    chunk[f'{target_lang}_offset'] = 0
                doc[f'{target_lang}_text'] = (doc.get(f'{target_lang}_text', "") + " " + trans_text).strip()

        if do_regex:
            pass  # TBD

        if do_ontology:
            # ontology - context independent - there are some bugs in disease detection which needs to be fixed
            for doc in docs.values():
                doc[f'{target_lang}_ner'] = ner = doc.get(f'{target_lang}_ner', {})
                if target_lang == 'en':
                    chunk2ner = self.ontology_manager.tokenize(doc[f'{target_lang}_text'])['chunk2ner']
                    onto_items = []
                    for c, label in chunk2ner.items():
                        ner_word = c[0].replace(" ", "").replace("_", "").replace("_", "") if self.cjk_detect(c[0]) else \
                            c[0].replace("_", " ").replace("_", " ").rstrip(self.strip_chars)
                        if ner_word.lower() not in stopwords2:
                            if not self.cjk_detect(ner_word) and label in (
                                    'PERSON', 'PUBLIC_FIGURE', 'ORG') and " " not in ner_word: continue
                            onto_items.append(((ner_word, c[1], c[1] + len(ner_word)), label))
                    for ner_mention, label in list(set(onto_items)):
                        aHash = ner.get(ner_mention, {})
                        aHash[label] = aHash.get(label, 0) + ontology_weight * (
                                1.0 + len(ner_mention[0]) / 100) * backtrans_weight
                        ner[ner_mention] = aHash

        if do_spacy:
            if spacy_nlp:
                # spacy
                self.spacy_ner(docs, spacy_nlp, stopwords2, spacy_weight, target_lang, extra_weight=backtrans_weight)

        if do_hf_ner:
            # transformer
            for ner_pipeline, hf_ner_weight2 in ner_pipelines:
                for a_batch in self.batch(chunks, batch_size):
                    self.hf_ner(ner_pipeline, target_lang, docs, a_batch, stopwords=stopwords2,
                                weight=hf_ner_weight * backtrans_weight * hf_ner_weight2)

        if do_cleanup:
            # do some cleanups. we don't want any ner that are just short numbers, stopwords or single characters.
            for _id, doc in docs.items():
                ner = doc[f'{target_lang}_ner']
                for key in list(doc[f'{target_lang}_ner'].keys()):
                    ner_word = key[0]
                    try:
                        if len(ner_word) < 4 and float(ner_word):
                            print("deleting ", ner_word)
                            del doc[f'{target_lang}_ner'][key]
                            continue
                    except:
                        pass
                    if ner_word.lower() in stopwords2 or (not self.cjk_detect(ner_word) and len(ner_word) <= 1):
                        print("deleting ", ner_word)
                        del doc[f'{target_lang}_ner'][key]

        # increase weight of src ner items if the target translations indicate it's an NER
        if target_lang != src_lang:
            for doc in docs.values():
                ner = doc[f'{target_lang}_ner']
                target2src_ner = doc.get(f'{target_lang}_2_{src_lang}_ner', {})
                for ent, idx in target2src_ner.items():
                    key = doc[f'{src_lang}_items'][idx]
                    # NOTE that this is an unordered match
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
                                    print('increasing ', key, label, aHash[label])
                                    found = True
                            if not found:
                                print('not found', key, all_labels)

        if do_docs_trim:
            docs, chunks = self.trim_to_prefer_person(docs, chunks)

        if do_backtrans and target_lang != src_lang:
            # backtrans from src_lang to target_lang back to src_lang allows us to catch more NER using target lang NER tools.
            # then we tag in target_lang those items we haven't already found, and tranlsate back to match the original text.
            # NOTE: We do not modify the original text, but only use backtrans to do NER tagging and other analysis.

            sep = "" if target_is_cjk else " "

            if target_is_cjk:
                lbracket = "[["
                rbracket = "]]"
            else:
                lbracket = "["
                rbracket = "]"
            for chunk in chunks:
                _id = chunk['id']
                text = chunk[f'{target_lang}_text'].replace("[", "{").replace("(", "{").replace(")", "}").replace("]",
                                                                                                                  "}")
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
                        if " " + key[0] + " " in text[i:]:
                            j = text.index(" " + key[0] + " ", i)
                            text = text[:j] + (text[j:].replace(" " + key[0] + " ", f"  **{idx}**  ", 1))
                            i = j
                    else:
                        if key[0] in text[i:]:
                            j = text.index(key[0], i)
                            text = text[:j] + (text[j:].replace(key[0], f"  **{idx}**  ", 1))
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
                        text = text.replace(" " + key[0] + " ", f"  **{idx}**  ")
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

            backtrans_text = self.do_translations([chunk[f'{target_lang}_tmpl_text'] for chunk in chunks],
                                                  src_lang=target_lang, target_lang=src_lang, batch_size=batch_size)
            for chunk, trans_text in zip(chunks, backtrans_text):
                # langid check
                try:
                    lang = langid.classify(trans_text)[0]
                except:
                    lang = target_lang
                if lang == target_lang:
                    chunk[f'{src_lang}_text_backtrans_from_{target_lang}'] = trans_text.lstrip(" .").replace(rbracket,
                                                                                                             "]").replace(
                        lbracket, "[").replace("}", ")").replace("{", "(")
                else:
                    chunk[f'{src_lang}_text_backtrans_from_{target_lang}'] = " . . . "
            # TODO: do similiarty test?
            for chunk, trans_text in zip(chunks, backtrans_text):
                _id = chunk['id']
                doc = docs[_id]
                orig_text = chunk[f'{src_lang}_text']
                trans_text = chunk[f'{src_lang}_text_backtrans_from_{target_lang}']
                items = doc[f'{target_lang}_items']
                len_items = len(items)
                doc[f'{src_lang}_2_{target_lang}_backtrans_ner'] = ner = doc.get(
                    f'{src_lang}_2_{target_lang}_backtrans_ner', {})
                pos = 0
                blocks, score = self.get_aligned_text(orig_text, trans_text, src_lang)
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
                                    found = False
                                    if len_ent2arr > 3:
                                        ent3 = sep.join(ent2arr[:3])
                                        if ent3 in new_word:
                                            new_word = ner_word[ner_word.index(ent3):]
                                            found = True
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
                                        print(f'found new mention from {target_lang}', mention, label)
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
                            print('increasing in backtrans ', key, label, aHash[label])
                for key, aHash1 in bner.items():
                    ner[key] = aHash2 = ner.get(key, {})
                    for key2 in aHash1:
                        aHash2[key2] = aHash2.get(key2, 0.0) + aHash1[key2]

            if do_postprocessing_after_backtrans:
                pass

            if do_cleanup:
                # do some cleanups. we don't want any ner that are just short numbers, stopwords or single characters.
                for _id, doc in docs.items():
                    ner = doc[f'{src_lang}_ner']
                    for key in list(doc[f'{src_lang}_ner'].keys()):
                        ner_word = key[0]
                        try:
                            if len(ner_word) < 4 and float(ner_word):
                                print("deleting ", ner_word)
                                del doc[f'{src_lang}_ner'][key]
                                continue
                        except:
                            pass
                        if ner_word.lower() in stopwords1 or (not self.cjk_detect(ner_word) and len(ner_word) <= 1):
                            print("deleting ", ner_word)
                            del doc[f'{src_lang}_ner'][key]

        return docs, chunks

    def process_ner(self,
                    src_lang: str = None,
                    docs=None,
                    do_spacy=True,
                    do_hf_ner=True,
                    do_ontology=True,
                    do_backtrans=False,
                    do_cleanup=True,
                    do_regex=True,
                    batch_size=5,
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
        sep = "" if src_is_cjk else " "

        if docs is None:
            docs, domain = get_docs(src_lang=src_lang)
        elif isinstance(docs, str):
            docs = [{f'{src_lang}_text': docs}]
        elif isinstance(docs, list):
            if isinstance(docs[0], dict):
                docs = docs
            else:
                docs = [{f'{src_lang}_text': t} for t in docs]

        # for testing only
        if cutoff is not None and len(docs) > cutoff:
            docs = docs[:cutoff]

        len_docs = len(docs)
        for doc in docs:
            doc[f'{src_lang}_text'] = doc['text']
            del doc['text']

        badwords1 = set([s for s in badwords_ac_dc.get(src_lang, []) if len(s) < 5])
        stopwords1 = set(stopwords_ac_dc.get(src_lang, []))
        docs = [doc for doc in docs if
                self.check_good_sentence(doc[f'{src_lang}_text'], src_lang, stopwords=stopwords1, badwords=badwords1)]
        logging.info(f'trimmed junk {str((len_docs - len(docs)) / len_docs)}')

        chunks = []

        for _id, doc in enumerate(docs):
            if 'id' not in doc:
                doc['id'] = str(_id)
            doc.setdefault('id', str(_id))
            doc[f'{src_lang}_text'] = doc[f'{src_lang}_text'].replace("[", "(").replace("]",")")  # we use [] as special chars
            doc['lang'] = src_lang
            doc['domain'] = domain
            doc['chunks'] = []

            self._split_text_into_chunks(src_lang=src_lang, src_is_cjk=src_is_cjk, doc=doc, batch_window=batch_window, sep=sep, chunks=chunks)

        docs = dict([(doc['id'], doc) for doc in docs])
        if do_docs_trim:
            docs2, chunks2 = self.trim_to_prefer_person(docs, chunks)
            do_docs_trim = len(docs2) == len(docs)
            docs, chunks = docs2, chunks2

        # we do this here because we don't want to trim  ner items that are considered empty.
        # we should probably fix trim_to_prefer_person to not do any trimming if all ner's are empty
        for doc in docs.values():
            doc[f'{src_lang}_ner'] = doc.get(f'{src_lang}_ner', {})

        # do ner processing in src_lang
        docs2, chunks2 = self.process_ner_chunks_with_trans(
            src_lang,
            docs,
            chunks,
            do_spacy=do_spacy,
            do_hf_ner=do_hf_ner,
            do_ontology=do_ontology,
            do_backtrans=False,
            do_regex=do_regex,
            do_cleanup=do_cleanup,
            batch_size=batch_size,
            ontology_weight=ontology_weight,
            spacy_weight=spacy_weight,
            hf_ner_weight=hf_ner_weight,
            backtrans_weight=backtrans_weight,
            do_postprocessing_after_backtrans=False,
            do_docs_trim=do_docs_trim)
        if do_docs_trim:
            do_docs_trim = len(docs2) == len(docs)
        docs, chunks = docs2, chunks2

        if do_backtrans:
            logging.info(f"Doing backtrans with src_lang={src_lang} to target_lang={target_lang}")
            # do ner processing in target language with optional backtrans
            docs2, chunks2 = self.process_ner_chunks_with_trans(
                src_lang,
                docs,
                chunks,
                target_lang=target_lang,
                do_spacy=do_spacy,
                do_hf_ner=do_hf_ner,
                do_ontology=do_ontology,
                do_backtrans=do_backtrans,
                do_regex=do_regex,
                do_cleanup=do_cleanup,
                batch_size=batch_size,
                ontology_weight=ontology_weight,
                spacy_weight=spacy_weight,
                hf_ner_weight=hf_ner_weight,
                backtrans_weight=backtrans_weight,
                do_postprocessing_after_backtrans=True,
                do_docs_trim=do_docs_trim)
            docs, chunks = docs2, chunks2
        return docs, chunks
