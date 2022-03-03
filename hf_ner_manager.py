"""
Copyright, 2021-2022 Ontocord, LLC, and other authors of Muliwai, All rights reserved.
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
import torch
from transformers import pipeline, AutoTokenizer, XLMRobertaForTokenClassification, BertForTokenClassification, ElectraForTokenClassification, RobertaForTokenClassification
from cjk import cjk_detect

hf_ner_model_map = {
      "sn": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 1.0]], 
      "st": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 1.0]], 
      "ny": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 1.0]], 
      "xh": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 1.0]], 
      "zu": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 1.0]], 
      "sw": [["Davlan/xlm-roberta-base-masakhaner", XLMRobertaForTokenClassification, 1.0]], 
      "yo": [["Davlan/xlm-roberta-base-masakhaner", XLMRobertaForTokenClassification, 1.0 ]],
      "ig": [["Davlan/xlm-roberta-base-masakhaner", XLMRobertaForTokenClassification, 1.0 ]],
      "ar": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0],],
      "en": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0],], 
      "es": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ], ],
      "eu": [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0], ],
      "ca": [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0], ],
      "pt": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ], ],
      "fr": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ], ],
      "zh": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ], ],
      'vi': [["lhkhiem28/COVID-19-Named-Entity-Recognition-for-Vietnamese", RobertaForTokenClassification, 1.0], ],
      'hi': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0]],
      'bn': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0]], 
      'ur': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0]], 
      'id': [["cahya/bert-base-indonesian-NER", BertForTokenClassification, 1.0],],

      # NOT PART OF OUR ORIGINAL LANGUAGE SET. 
      "fon": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 0.8]], 
      "lg": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 0.8]], 
      "rw": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 0.8]], 
      "wo": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 0.8]], 
      'gu': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      'as': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      'mr': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      'ml': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      'kn': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      'ne': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      'pa': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      'or': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      'ta': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      'te': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      
      }

_id = 0
ner_model_name2pipelines = {}

def load_hf_ner_pipelines(target_lang, device="cpu", device_id=-1):
    """ Loads and stores a set of NER pipelines in a cache"""
    if device != "cpu" and device_id == -1:
      device_id = 0
    pipelines = []
    for model_name, model_cls, hf_ner_weight2 in hf_ner_model_map.get(target_lang, []):
          if model_name not in ner_model_name2pipelines:
            try:
              model = model_cls.from_pretrained(model_name).half().eval().to(device)
              tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512, truncation=True)
              if device == "cpu":
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
              else:
                ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=device_id)
              ner_model_name2pipelines[model_name] = ner_pipeline
              pipelines.append({'pipeline': ner_pipeline, 'weight': hf_ner_weight2, 'name': model_name})
            except:
              #logger.info("problems loading model and tokenizer for pipeline. attempting to load without passing in model")
              tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512, truncation=True)
              if device == "cpu":
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                ner_pipeline = pipeline("ner", model=model, tokenizer=(tokenizer, {"use_fast": True},), )
              else:
                ner_pipeline = pipeline("ner",  model=model_name, tokenizer=(tokenizer, {"use_fast": True},), device=device_id)
              ner_model_name2pipelines[model_name] = ner_pipeline
              pipelines.append({'pipeline': ner_pipeline, 'weight': hf_ner_weight2, 'name': model_name})
    return pipelines

def chunkify(doc, src_lang, sep, num_words_per_chunk, strip_chars, punc_char,  text_key=None, ):
      """
      Do basic sentence splitting and limiting each chunk's number of words to prevent overflow. We assume the docs are long. 
      """
      global _id
      if text_key is None:
        if f'{src_lang}_text' in doc:
          text_key = f'{src_lang}_text'
        elif 'text' in doc:
          text_key = 'text'
      src_is_cjk = src_lang in ("zh", "ja", "ko", "th")
      if type(doc) is str:
        doc = {'text': doc}
      chunks = doc['chunks'] = doc.get('chunks', [])
      if 'id' not in doc or int(doc['id']) < 0:
            doc['id'] = str(_id)
            _id += 1
      if text_key in doc: 
        #simple multi-lingual tokenizer and sentence splitter
        offset = 0
        if src_is_cjk:
          text = list(doc[text_key].replace("。", "。 ").replace("  ", " "))
        else:
          textarr = doc[text_key].replace("  ", " ").split()
          text = []
          for t in textarr:
            len_t = len(t)
            if len_t == 1:
              text.append(t)
              continue
            punc_found = [punc for punc in t if punc in punc_char]
            word1, word2 = "", ""
            if punc_found:
              tarr = t.split(punc_found[0])
              word1 = tarr[-2]
              word2 = tarr[-1]
            if punc_found and t[-1] not in punc_char and \
                              ((punc_found[0] not in ".。") or \
                               (t[0] not in "0123456789" and t[0] == t[0].lower()) or \
                               (word1 and word1[-1] in strip_chars) or \
                               (word2 and word2[0] in strip_chars)):
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
        doc[text_key] = sep.join(text)
        len_text = len(text)
        src_text = ""
        while len_text > num_words_per_chunk:
            for j in range(num_words_per_chunk-1, len_text):
              if j > num_words_per_chunk * 2: break
              if (src_is_cjk and text[j] in punc_char+' ') or \
                  (not src_is_cjk and text[j][-1] in punc_char):
                break
            text_str = sep.join(text[:j+1])
            chunks.append({text_key: text_str, 'id': doc['id'], f'{src_lang}_offset': offset})
            doc['chunks'].append(chunks[-1])
            offset += len(text_str) + (0 if src_is_cjk else 1)
            text = text[j+1:]
            len_text = len(text)
        if text:
            text_str = sep.join(text)
            chunks.append({text_key: text_str, 'id': doc['id'], f'{src_lang}_offset': offset})

def detect(sentence, src_lang,  tag_type={'PERSON', 'PUBLIC_FIGURE'},  chunks=None, hf_pipelines=None, sep=" ", strip_chars="", punc_char=".", num_words_per_chunk=150, stopwords=None, device="cpu", device_id=-1, weight=1., text_key=None, ner_key=None, offset_key=None, batch_size=20,):
    """
    Output:
       - This function returns a list of 4 tuples, representing an NER detection for [(entity, start, end, tag), ...]
    Input:
       :sentence: the sentence to tag
       :src_lang: the language of the sentence
       :tag_type: the type of NER tags we are detecting. If None, then detect everything.
    Algortithm:
    run the sentence through a Huggingface ner pipeline.
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
    doc = {'text': sentence}
    if chunks is None:
      chunks = chunkify(doc, src_lang, sep, num_words_per_chunk, strip_chars, punc_char)
    if hf_pipelines is None:
      hf_pipelines = load_hf_ner_pipelines(src_lang, device=device, device_id=device_id)
    for hf_pipeline in hf_pipelines:    
      results_arr = hf_pipeline['pipeline']([chunk[text_key] for chunk in chunks], batch_size=min(batch_size, len(chunks)))
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
          if not cjk_detect(text[ner_result['start']:ner_result['end']]) and src_lang not in ("zh", "ja", "ko", "th"):
                #strip the strip_chars
                if text[start] not in strip_chars:
                  for j in range(1, start):
                    if start - j == -1 or text[start-j] in strip_chars:
                      start = max(start -j, 0)
                      break
                end = ner_result['end']
                if end < len_text and text[end] != ' ':
                  end += len(text[end:].split(' ', 1)[0])
          else:
                start = ner_result['start']
                end = ner_result['end']
          #strip the strip_chars
          while text[start] in strip_chars and start < len_text:
            start += 1
            if start >= end: break
          #save away the ner result with the proper offset for this chunk
          if start < len_text and start < end:
              end = start + len(text[start:end].strip(strip_chars))
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
          ner = doc[ner_key]
          text = doc[text_key]
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
            elif label in ('PATIENT_ID', 'GOVT_ID', 'ID' ): label = 'ID'
            elif label in ('USER', ): label = 'USER'
            elif label in ('EMAIL', ): label = 'EMAIL'
            else: label = 'MISC'
            if tag_type and label not in tag_type: continue
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
                    mention = (ner_word, prev_word[0], prev_word[1])
                    if ner_word and ner_word.lower() not in stopwords:
                      aHash = ner.get(mention, {})
                      aHash[prev_label] = aHash.get(prev_label, 0) + weight * hf_pipeline['weight']
                      ner[mention] = aHash
                    prev_word = [start, end]
                    prev_word2 = ner_result['word']
            elif prev_label is None:
              prev_word = [start, end]
              prev_word2 = ner_result['word']
            prev_label = label

          if prev_label is not None and prev_word[0] != prev_word[1]:
              ner_word = text[prev_word[0]:prev_word[1]]
              mention = (ner_word, prev_word[0], prev_word[1])
              if ner_word and ner_word.lower() not in stopwords:
                  aHash = ner.get(mention, {})
                  aHash[prev_label] = aHash.get(prev_label, 0) + weight * hf_pipeline['weight']
                  ner[mention] = aHash
    return doc[ner_key]

