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
from collections import Counter
from transformers import pipeline, AutoTokenizer, XLMRobertaForTokenClassification, BertForTokenClassification, ElectraForTokenClassification, RobertaForTokenClassification
from cjk import cjk_detect
from kenlm_manager import *
from char_manager import *
import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s : %(processName)s : %(levelname)s : %(message)s',
    level=logging.INFO)

try:
  if not stopwords:
    from stopwords import stopwords
except:
  try:
    from stopwords import stopwords
  except:
    stopwords = {}
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
      'vi': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0],], # ["lhkhiem28/COVID-19-Named-Entity-Recognition-for-Vietnamese", RobertaForTokenClassification, 1.0], ],
      'hi': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0]],
      'bn': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0]], 
      'ur': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0]], 
      'id': [["cahya/bert-base-indonesian-NER", BertForTokenClassification, 1.0],],

      # NOT PART OF OUR ORIGINAL LANGUAGE SET. 
      "fon": [["Davlan/xlm-roberta-base-masakhaner", XLMRobertaForTokenClassification, 0.8]], 
      "lg": [["Davlan/xlm-roberta-base-masakhaner", XLMRobertaForTokenClassification, 1.0]], 
      "rw": [["Davlan/xlm-roberta-base-masakhaner", XLMRobertaForTokenClassification, 1.0]], 
      "wo": [["Davlan/xlm-roberta-base-masakhaner", XLMRobertaForTokenClassification, 1.0]], 
      'gu': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 0.8 ]],
      'as': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 0.8 ]],
      'mr': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 0.8 ]],
      'ml': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 0.8 ]],
      'kn': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 0.8 ]],
      'ne': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 0.8 ]],
      'pa': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 0.8 ]],
      'or': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 0.8 ]],
      'ta': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 0.8 ]],
      'te': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 0.8 ]],
      }

_id = 0
ner_model2pipelines = {}

def load_hf_ner_pipelines(target_lang, device="cpu", device_id=-1):
    """ Loads and stores a set of NER pipelines in a cache"""
    if device != "cpu" and device_id == -1 and ":" in device:
      device_id = int(device.split(":")[-1])
    pipelines = []
    for model_name, model_cls, hf_ner_weight2 in hf_ner_model_map.get(target_lang, []):
          if (model_name, device) not in ner_model2pipelines:
              model = model_cls.from_pretrained(model_name).eval().to(device)
              tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512, truncation=True)
              if device == "cpu":
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
              else:
                model = model.half()
                ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=device_id)
              ner_model2pipelines[(model_name, device)] = ner_pipeline
          else:
              ner_pipeline = ner_model2pipelines[(model_name, device)]
          pipelines.append({'pipeline': ner_pipeline, 'weight': hf_ner_weight2, 'name': model_name})
    return pipelines

def chunkify(doc, src_lang,  num_words_per_chunk=150,  text_key=None, offset_key=None):
      """
      Do basic sentence splitting and limiting each chunk's number of words to prevent overflow. We assume the docs are long. 
      """
      global _id
      if text_key is None:
        if f'{src_lang}_text' in doc:
          text_key = f'{src_lang}_text'
        else:
          text_key = 'text'
      if text_key is None:
        if f'{src_lang}_offset' in doc:
          offset_key = f'{src_lang}_offset'
        else:
          offset_key = 'offset'
      src_is_cjk = src_lang in ("zh", "ja", "ko", "th")
      if src_is_cjk: 
        sep = ""
      else:
        sep = " "
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
            chunks.append({text_key: text_str, 'id': doc['id'], 'offset': offset})
            doc['chunks'].append(chunks[-1])
            offset += len(text_str) + (0 if src_is_cjk else 1)
            text = text[j+1:]
            len_text = len(text)
        if text:
            text_str = sep.join(text)
            chunks.append({text_key: text_str, 'id': doc['id'], 'offset': offset})
            doc['chunks'].append(chunks[-1])
      return chunks

def detect_ner_with_hf_model(sentence, src_lang,  tag_type={'PERSON', 'PUBLIC_FIGURE'},  chunks=None, hf_pipelines=None, num_words_per_chunk=150,  device="cpu", device_id=-1, weight=1.0, text_key=None, ner_key=None, offset_key=None, batch_size=20,):
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
      NOTE: we don't use results_arr = hf_pipeline([chunk[text_key] for chunk in chunks], grouped_entities=True)
      because grouped_entities does not properly group all entities as we do it below.
    """
    if device_id < 0 and device != "cpu" and ":" in device:
      device_id = int(device.split(":")[-1])
    sw = set(stopwords.get(src_lang, []))
    if offset_key is None:
      offset_key = 'offset'
    if ner_key is None:
      ner_key = 'ner'
    if text_key is None:
      text_key = 'text'
    doc = {'text': sentence}
    src_is_cjk = src_lang in ("zh", "ja", "ko", "th")
    if src_is_cjk: 
        sep = ""
    else:
        sep = " "
    if chunks is None:
      chunks = chunkify(doc, src_lang, num_words_per_chunk)
    if hf_pipelines is None:
      hf_pipelines = load_hf_ner_pipelines(src_lang, device=device, device_id=device_id)
    for hf_pipeline in hf_pipelines: 
      results_arr = hf_pipeline['pipeline']([chunk[text_key] for chunk in chunks], batch_size=min(batch_size, len(chunks)))
      results_arr2 = []
      offset = 0
      for chunk, results in zip(chunks, results_arr):
        text = chunk[text_key]
        _id = chunk['id']
        ner = doc[ner_key] = doc.get(ner_key,{})
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
          prev_span = [0,0]
          prev_tag_and_score = None
          prev_word = ""
          for ner_result in results:
            start = ner_result['start']
            if start is None:
              prev_word = ""
              continue
            end = ner_result['end']
            if text[start:end] != ner_result['word']:
              logger.info ('offset mismatch', text[start:end], ner_result['word'])
            if "-" in ner_result['entity']:
              _, tag = ner_result['entity'].split('-')
            else:
              tag = ner_result['entity']
            tag = tag.upper()
            if tag in ('ADDRESS', 'STREET_ADDRESS'): tag = 'ADDRESS'
            elif tag in ('PUBLIC_FIGURE',): tag = 'PUBLIC_FIGURE'
            elif tag in ('NAME', 'PER', 'PERSON'): tag = 'PERSON'
            elif tag in ('LOCATION', 'LOC', 'GPE'): tag = 'LOC'
            elif tag in ('ORGANIZATION', 'ORG'): tag = 'ORG'
            elif tag in ('AGE',): tag = 'AGE'
            elif tag in ('NORP',): tag = 'NORP'
            elif tag in ('BIO', 'SYMPTOM_AND_DISEASE', 'DISEASE' ): tag = 'DISEASE'
            elif tag in ('PATIENT_ID', 'GOVT_ID', 'ID' ): tag = 'ID'
            elif tag in ('USER', ): tag = 'USER'
            elif tag in ('EMAIL', ): tag = 'EMAIL'
            else: tag = 'MISC'
            if tag_type and tag not in tag_type: continue
            if prev_tag_and_score is not None:
                if not ner_result['entity'].startswith('B-') and tag == prev_tag_and_score[0] and (prev_span[1] >= start - 5):
                  #keep expanding this span and matched word and create a min score
                  prev_span[1] =  max(prev_span[1], end)
                  prev_word = prev_word + " " + ner_result['word']
                  prev_tag_and_score[1] = min(prev_tag_and_score[1], ner_result['score']) 
                  continue
                else:
                  if ner_result['entity'].startswith('B-'):
                    if prev_span[1] > start:
                      prev_span[1] = start
                  if prev_span[0] != prev_span[1]:
                    ner_word = text[prev_span[0]:prev_span[1]]
                    mention = (ner_word, prev_span[0], prev_span[1])
                    if ner_word and ner_word.lower() not in sw:
                      aHash = ner.get(mention, {})
                      aHash[prev_tag_and_score[0]] = aHash.get(prev_tag_and_score[0], 0) + weight * hf_pipeline['weight'] * prev_tag_and_score[1]
                      ner[mention] = aHash
                    prev_span = [start, end]
                    prev_word = ner_result['word']
            elif prev_tag_and_score is None:
              prev_span = [start, end]
              prev_word = ner_result['word']
              
            prev_tag_and_score = [tag, ner_result['score']]

          if prev_tag_and_score is not None and prev_span[0] != prev_span[1]:
              ner_word = text[prev_span[0]:prev_span[1]]
              mention = (ner_word, prev_span[0], prev_span[1])
              if ner_word and ner_word.lower() not in sw:
                  aHash = ner.get(mention, {})
                  aHash[prev_tag_and_score[0]] = aHash.get(prev_tag_and_score[0], 0) + weight * hf_pipeline['weight'] * prev_tag_and_score[1]
                  ner[mention] = aHash

    # now let's flatten into a 4 tuple which is expected by other functions. For the tag, we take the winning tag.
    ners = [tuple(list(a) +  [max(Counter(b))]) for a, b in doc[ner_key].items()]
    models = load_kenlm_model(src_lang, pretrained_models=["wikipedia"] if src_lang not in  ('ig', 'zu', 'ny', 'sn', "st") else ["mc4"])
    for i, a_ner in enumerate(ners):
      ent = a_ner[0]
      match, score, cutoff = check_for_common_name(src_lang, pretrained_models=["wikipedia"] if src_lang not in  ('ig', 'zu', 'ny', 'sn', "st") else ["mc4"], name=ent, kenlm_models=models, return_score=True)
      if match:
        #single word or short names may require an even lower cutoff
        if src_is_cjk and len(ent) <= 3:
          if score > cutoff/2: continue
        elif sep not in ent:
          if score > cutoff/2: continue
        a_ner = list(a_ner)
        a_ner[-1] = 'PUBLIC_FIGURE'
        ners[i] = tuple(a_ner)
    return ners


if __name__ == "__main__":
  sentence = """ '\ufeff Nje ipase Kristi wa ninu Bibeli? Nje ipase Kristi wa ninu Bibeli?Ibeere: "Nje ipase Kristi wa ninu Bibeli?"Idahun: Nipa ohun ti Jesu so nipa ara, awon omo leyin re si gba gbo nipa ipase re. Won ni agbara lati dari ji ese wa- nkan ti o je pe Oluwa nikan lo le se, eyi ti o fi je wipe a dese si (ise awon Aposteli 5:13; Kolosse 3:13; Orin Dafidi 130:4; Jeremiah 31:14). Pelu ohun ti a n jinyan re yi, Jesu naa ni won ni yio dajo awon ti o wa “laye tabi ti o ti ku” (2 Timoteu 4:1). Tomasi ki gbe pe Jesu, “Oluwa mi ati Olorun mi! ” (Johannu 20;28). Paulu pe Jesu “Oba nla Olugbala” ( Titus 2: 13), gege bi o ti wa si aye, o si “da bi Olorun” (Filemoni 2:5-8). Eni ti o ko si Heberu nipa Jesu wipe, Ite re, Olorun, lai ati lailai ni” (Heberu 1:8). Johannu wipe, Li atetekose li Oro wa, Oro si wa pelu Olorun, Olorun si li Oro (Jesu) na” (Johannu 1;1). Eyi ti o wa ninu iwe mimo ti o ko wa nipa ipase Kristi le di pipo (wo Ifihan 1: 17; 2: 8;22;13 1 Korinti 10:4; 1 Peteru 2:6-8. Orin Dafidi 18:2; 95:1; 1 Peteru 5:4; Heberu 13: 20), sugbon e yi ye ki o je ki awon enia fi mon nipa ipase re pelu awon omo leyin re. Jesu si awon Oruro miran bi Yahweh (Oruro ti Olorun n je tele) ninu iwe majemu lailai. Majemu lailai wipe “Oludande” (Orin Dafidi 130; 7, Hosea 13: 14) ni won lo fun Jesu ninu majemu Titán ( Titu 2: 13; Ifihan 5;9). Wo pe Jesu ni Emmanueli (“Oluwa wa pelu wa” ninu Matteu 1). Ninu iwe Sekariah 12: 10, yahweh ni o so be, “ Nwon o ma wo eniti a gun li oko. ” Sugbon iwe majemu titán li o so eyi nipa kikan mo agbelebu re (Johannu 19;37; Ifihan 1:7). Ti o b a je wipe yahweh ni won gun ni oko, ti won si wo, ti o si je wipe Jesu ni won gun ni oko ti won si wo, Jesu si ni Yahweh naa. Paulu so ninu Isaiah 45: 22-23 nipa Jesu ninu Filippi 2;10-11. Leyin naa, a lo oruko Jesu pelu yahweh ninu adura “Ore-ofe si nyin ati alafia lati odo Olorun Baba wa, ati Jesu Kristi Oluwa wa’ (Galatia 1; 3, Efesu 1:2). Eyi a je oro odi ti ipase Kristi o ba wa. Oruko Jesu je gege bi Yahweh ti Jesu ti pase wipe ki a se irubomi ni oruko (le kan soso) ti Baba ati Omo ati Emi Mimo” ( Matteu 28: 19 wo 2 Korinti 13: 14).Ohun ti Oluwa le se ni amo si ise re. Jesu ko ji oku dide nikan (Johannu 5: 21; 11: 38-44), o si dariji ese wa (Ise Awon Aposteli 5;31; 13: 38), od gbogbo aye ati ohun kohun (Johannu 1:2; Kolosse 1:16-17)! Eyi je ki amo gidi gan wipe ti a ba wo oro Yahweh wipe o si ti wa latetekose (Isaiah 44:24). Kristi ni awa ohun iwa ti o je wipe eni ti o ni iru ipase yi ni o le ni. Ayeraye (Johannu 8: 58), o wa pelu wa ni igbagbogbo (Matteu 18: 20, 28: 20), o mo ohun gbogbo (Matteu 16: 21), O le se ohun gbogbo (Johannu 11: 38-44).Ni isin yi, ki awon kan so wipe Olorun ni awon tabi ki a paro fun awon kan wipe otito ni, ki a si wa ohun ti a fi ma jiyan re. Kristi fihan wa wipe ohun ni ipase oro gege bi ise iyanu re, ati ajinde re. O si se ise iyanu bi o si ti yi omi si oti wini (Johannu 2: 7), o rin lori omi (Matteu 2: 3), ati awon alaisan (Matteu 9:35; Maku 1: 40-42), ti o si ji oku dide (Johannu 11:43-44; Luku 7:11-15; Maku 5:35). Jesu si jinde. Eyi ti a mon ti o si yao si awon olorun miran ti won le jinde- ko si si wipe a gbo nipa re. Bi Dokito Gary Haberlas se so, awon bi ohun mejila lo fi han si awon keferi nipa ajinde re:1. Jesu ku nitori a kan mo agbelebu2. A si sin3. Iku re je ki awon omo leyin re wa ninu ironu ati irota.4. Isa oku Jesu wo si ri wipe o sofo lyin ojo die.5. Awon omo leyin re si mo wipe awon ti ri ajinde Jesu.6. Leyin eyi, awon omo leyin re si ni okun ati agbara lati sise.7. Eyi ni iwasu larin awon ijo nigbati won bere8. won si wasu oro yi ni Jerusalemu.9. Nitori iwasu yi, a si kede ibere ile ijosin ti o si gboro.10. Ojo Ajinde, Sunday, ti o yato si Sabati (satiday) ti a mo si ojo ijosin11. Jakobu, ti ko ni igbagbo, si di atunbi ni igba ti o ri ajinde Jesu.12. Paulu, ota Kristiani, si gbagbo nipa ohun ti o mon nigbati o pade ajinde Jesu.Nje ti enikeni ba wi pe iro ni gbogbo ohun ti mo ti so yi, awon die ni a le fihan lati le je ki won mon bi, ajinde re ati iroyin ayo: Iku Jesu, sisin re, ajinde ati riri re (1 Korinti 15:1-5). A si mon wipe orisirisi ni a le fi han nipa ohun ti mo ko yi, sugbon ajinde re ni o dahun gbogbo ibere naa. Awon enia so wipe awon omo leyin re wipe won ri ajinde Jesu. Bi o ba je iro tabi a n ro, ko si ohun ti o le yi oj okan pada gege bi ajinde re. Ikini, ki ni ere won? Kristianiti o je ki won ni ola ati owo. Ekeji, awon oniro o le jinyan fun ohun ti a ba ti wa tele. Ko si ohun miran sugbon ajinde re ti awon omo leyin Jesu gbagbo ti won si le ku sibe. Beni, orisirisi awon enia ni o ti ku nipa iro ti won pa ti won si ro wipe otito ni sugbon ko si eni ti yio ku fun ohun ti won gbagbo. Ni soki, Kristi so wipe Yahweh ni ohun, ohun ni ipase (lai se “olorun” nikan- sugbon Olorun otito), awon omo leyin re (awon Ju ti o ye ki won beru irubo) gbagbo, won si tele. Kristi fi han wa wipe ohun ni ipase ise iyanu nipaajinde re. Ko si enikeni ti o le fi eyi han.
  """
  src_lang = "yo"

  print (detect_ner_with_hf_model(sentence, src_lang))
