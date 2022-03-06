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

# This file extends faker to more languages and includes tests based on kenlm statistics to lessen the liklihood a name is a real person's name.
# This file also provides basic anonymization and context management so that names are swapped appropraitely.

from faker import Faker
from faker.providers import person, company, geo, address, ssn, internet
from fake_names import *
from kenlm_manager import *
from typing import List
import random
import time
import copy

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

class FakerExtensions:
  def __init__(
      self,
      lang: str = "vi",
      trials: int = 1000,
      faker=None,
      
  ):
      self.lang = lang
      self.trials = trials
      self.num_genders = 2
      if faker is None:
        if lang in ("vi",  "mr",  "yo", "sw","sn", "st", "ig", "ny", "xh", "zu", "st"):
          faker = self.faker = Faker("en_GB")
        else:
          faker = self.faker = Faker(random.choice(faker_map["es" if lang in ("eu", "ca") else "hi" if lang in ("as", "gu", "pa","bn", "ur", ) else lang]))
        faker.add_provider(person)
        faker.add_provider(ssn)
        faker.add_provider(address)
        faker.add_provider(geo)
        faker.add_provider(internet)
        faker.add_provider(company)
      else:
        self.faker = faker
      self.kenlm_models = load_kenlm_model(self.lang, pretrained_models=["wikipedia"] if lang not in ('ig', 'zu', 'ny', 'sn', "st") else ["mc4"])
      self.patterns = public_figure_kenlm_cutoff_map.get(self.lang, [{'cutoff': 500, 'pattern': "{} (born"}])
      if self.lang == "vi":
          surname_list_of_lists: List[List[str]] = [vietnamese_surnames]
          first_middle_name_list_of_lists: List[List[str]] = [vietnamese_first_middlenames_male, vietnamese_first_middlenames_female]
          second_middle_name_list_of_lists: List[List[str]] = [vietnamese_second_middlenames_male, vietnamese_second_middlenames_female]
          first_name_list_of_lists: List[List[str]] = [vietnamese_firstnames_male, vietnamese_firstnames_female]
          self.name_lists: List[List[List[str]]] = [surname_list_of_lists, first_middle_name_list_of_lists, second_middle_name_list_of_lists, first_name_list_of_lists]
          self.name_lists_probabilities = [1.0, 0.5, 0.5, 1.0]
          assert len(self.name_lists) == len(self.name_lists_probabilities)
      elif self.lang == "bn":
          surname_list_of_lists: List[List[str]] = [bengali_surnames]
          first_name_list_of_lists: List[List[str]] = [bengali_firstnames_male, bengali_firstnames_female]
          self.name_lists = [first_name_list_of_lists, surname_list_of_lists]
          self.name_lists_probabilities = [1.0, 1.0]
          assert len(self.name_lists) == len(self.name_lists_probabilities)
      elif self.lang == "pa":
          surname_list_of_lists: List[List[str]] = [punjabi_surnames]
          first_name_list_of_lists: List[List[str]] = [punjabi_firstnames_male, punjabi_firstnames_female]
          self.name_lists = [first_name_list_of_lists, surname_list_of_lists]
          self.name_lists_probabilities = [1.0, 1.0]
          assert len(self.name_lists) == len(self.name_lists_probabilities)
      elif self.lang == "gu":
          surname_list_of_lists: List[List[str]] = [gujurati_surnames]
          first_name_list_of_lists: List[List[str]] = [gujurati_firstnames_male, gujurati_firstnames_female]
          self.name_lists = [first_name_list_of_lists, surname_list_of_lists]
          self.name_lists_probabilities = [1.0, 1.0]
          assert len(self.name_lists) == len(self.name_lists_probabilities)
      elif self.lang == "ur":
          self.num_genders = 1
          surname_list_of_lists: List[List[str]] = [urdu_surnames]
          first_name_list_of_lists: List[List[str]] = [urdu_firstnames]
          self.name_lists = [first_name_list_of_lists, surname_list_of_lists]
          self.name_lists_probabilities = [1.0, 1.0]
          assert len(self.name_lists) == len(self.name_lists_probabilities)
      elif self.lang == "ca":
          surname_list_of_lists: List[List[str]] = [catalan_surnames]
          first_name_list_of_lists: List[List[str]] = [catalan_firstnames_male, catalan_firstnames_female]
          self.name_lists = [first_name_list_of_lists, surname_list_of_lists]
          self.name_lists_probabilities = [1.0, 1.0]
          assert len(self.name_lists) == len(self.name_lists_probabilities)
      elif self.lang == "yo":
          surname_list_of_lists: List[List[str]] = [yoruba_surnames]
          first_name_list_of_lists: List[List[str]] = [yoruba_firstnames_male, yoruba_firstnames_female]
          self.name_lists = [first_name_list_of_lists, surname_list_of_lists]
          self.name_lists_probabilities = [1.0, 1.0]
          assert len(self.name_lists) == len(self.name_lists_probabilities)
      elif self.lang in ("mr", "sw", "sn", "st", "ig", "ny", "xh", "zu"):
          first_name_list_of_lists: List[List[str]] = [bantu_firstnames_male, bantu_firstnames_female]
          surname_list_of_lists: List[List[str]] =  [bantu_surnames]
          self.name_lists = [first_name_list_of_lists, surname_list_of_lists]
          self.name_lists_probabilities = [1.0, 1.0]
          assert len(self.name_lists) == len(self.name_lists_probabilities)
      else:
          self.name_lists = [[]]
          self.name_lists_probabilities = [1.0]
	
  def generate_fakename(self, one_name=False, gender: int = None):
      """ Generate fake name.  Use gender to generate a gender-specific name. Use 0 for male and 1 for female   """
      if gender is None:
          gender = random.choice(range(self.num_genders))
      elif gender < 0 or gender >= self.num_genders:
          raise Exception(f"Unknown gender type {gender}")
      output_name = []
      for i, name_list_of_lists in enumerate(self.name_lists):
          # Sometimes, we might have a single list for all genders,
          # thus we take the minimun to avoid out of index
          if not name_list_of_lists:
            if gender==1:
              output_name.append(self.faker.first_name_female())
            else:
              output_name.append(self.faker.first_name_male())
          else:
            name_list = name_list_of_lists[min(len(name_list_of_lists) - 1, gender)]
            if random.random() <= self.name_lists_probabilities[i]:
              output_name.append(random.choice(name_list))
          if one_name and output_name:
            return " ".join(output_name)
      return " ".join(output_name)

  def check_like_known_name(self, fake_name, verbose=False):
      """ Check fake name close to real common name."""
      if not self.kenlm_models: return False
      return check_for_common_name(
        src_lang = self.lang,
        pretrained_models = ['wikipedia'],
        name = fake_name,
        verbose = verbose, 
        kenlm_models = self.kenlm_models)
        

  def create_name(self, one_name=False, verbose=False):
      """ Create fake name and varify by kelnm models """
      success = False
      for _ in range(self.trials):
        if self.lang in ("pa", "gu","as", "mr", "vi", "bn", "ur", "ca", "yo", "sw", "sn", "st", "ig", "ny", "xh", "ca", "zu"): 
          fake_name = self.generate_fakename(one_name=one_name)
        else:
          if one_name:
            fake_name = self.faker.first_name()
          else:
            if self.lang in ("zh", "ja", "ko", "th"):
              fake_name = self.faker.first_name()+self.faker.last_name()
            else:
              fake_name = self.faker.first_name()+" "+self.faker.last_name()
        # we want our fake names to not be too close to a famous name
        if not self.check_like_known_name(fake_name, verbose):
            success = True
            return fake_name
      if not success and verbose:
          print('Could not find any fake name. Try reducing perplexity_cutoff')
      if one_name:
        fake_anme = self.faker.firstname()
      else:
        fake_name = self.faker.name()
  
  #TODO - create male and female versions of firstname and name similar to faker

  def first_name(self, ent=None, context=None, verbose=False,):
    return self.name(one_name=True, ent=ent, context=context, verbose=verbose)

  def name(self, one_name=False, ent=None, context=None, match_first_name=True, match_last_name=True, verbose=False,):
    """ Provides an extension for faker's name method. Also manages the context when anonyimzing ent. 
    Can match first and last names in the context.
    Sentence: John Doe went to the store. John bought milk. => Jack Smith went to the store. Jack bought milk. 
    """
    is_cjk = self.lang in ("zh", "ko", "ja", "th")
    if ent is None or context is None:   
      return self.create_name(one_name=one_name, verbose=verbose)
    if ent in context: return context[ent]
    if not is_cjk and " " not in ent: one_name = True
    na = self.create_name(one_name=one_name, verbose=verbose)
    na  = context[ent] = context.get(ent, na)
    if " " in ent:
        if is_cjk:
          ent_arr = ent
        else:
          #strip out prefixes and suffixes
          orig_ent_arr = ent_arr = ent.split(" ")
          if ent_arr[0][-1] == ".":
            ent_arr = ent_arr[1:]
          if ent_arr and ent_arr[-1][-1] == ".":
            ent_arr = ent_arr[:-1]
	  if not ent_arr: ent_arr = orig_ent_arr
        if match_first_name:
          ent1 = ent_arr[0] if not is_cjk else (ent[:2] if len(ent) > 2 else ent[:1])
          if ent1 in context:
            na1 = context[ent1]
            if is_cjk:
              na = "".join([na1]+na.split()[1:])
            elif " " in na:
              na = " ".join([na1]+na.split()[1:])
            else:
              na = na1 + " " + na
            context[ent] = na
          elif is_cjk:
            val = context[ent][:2] if len(context[ent]) > 2 else context[ent][:1] 
            context[ent1] = context.get(ent1, val) 
          elif " " in context[ent]:
            val_arr = context[ent].split()
            if val_arr[0][-1] == ".":
              val_arr = val_arr[1:]
            val = val_arr[0]
            context[ent1] = context.get(ent1, val) 

        if match_last_name:
          ent2 = ent_arr[-1] if not is_cjk else (ent[-2:] if len(ent) > 3 else ent[-1:])
          if ent2 in context:
            na2 = context[ent2]
            if is_cjk:
              na = "".join(na.split()[:-1] + [na2])
            elif " " in na:
              na = " ".join(na.split()[:-1] + [na2])
            else:
              na = na + " " + na2
            context[ent] = na
          elif is_cjk:
            val = context[ent][-2:] if len(context[ent]) > 3 else context[ent][-1:] 
            context[ent2] = context.get(ent2, val) 
          elif " " in context[ent]:
            val_arr = context[ent].split()
            if val_arr[-1][-1] == ".":
              val_arr = val_arr[:-1]
            val = val_arr[-1]
            context[ent2] = context.get(ent2, val) 
    return context[ent] 

  def company(self, ent=None, context=None):
    if ent is None or context is None: 
      try: 
        return self.faker.company()
      except:
        return "COMPANY"
    try:
      co = self.faker.company()
    except:
      co = "COMPANY"
    co = context[ent] = context.get(ent, co)
    if " " in ent:
        ent2 = ent.split(" ")[0]
        if len(ent2) > 4:
          if ent2 in context:
            co2 = context[ent2]
            if " " in co:
              co = " ".join([co2]+co.split()[1:])
            else:
              co = co2 + " " + co
            context[ent] = co
          elif " " in context[ent]:
            val = context[ent].split()[0]
          else:
            val = context[ent]
          context[ent2] = context.get(ent2, val) 
    return context[ent] 

  #TODO - call faker's phone, ssn (ID), user, email, url etc. to create psuedo data as opposed to just labels

  def ssn(self, ent=None, context=None):
    if ent is None or context is None: 
      return self.faker.ssn()
    context[ent] =  context.get(ent, self.faker.ssn())
    return context[ent]

  def email(self, ent=None, context=None):
    if ent is None or context is None: 
      return self.faker.email()
    context[ent] =  context.get(ent, self.faker.email())
    return context[ent]

  def address(self, ent=None, context=None):
    if ent is None or context is None: 
      return self.faker.address()
    context[ent] =  context.get(ent, self.faker.address())
    return context[ent]

  def country(self, ent=None, context=None):
    if ent is None or context is None: 
      return self.faker.country()
    context[ent] =  context.get(ent, self.faker.country())
    return context[ent]

  def state(self, ent=None, context=None):
    if ent is None or context is None: 
      if self.lang == 'zh': 
          return self.faker.province()
      else: 
          return self.faker.state()
    if self.lang == 'zh': 
        context[ent] =  context.get(ent, self.faker.province())
    else: 
        context[ent] =  context.get(ent, self.faker.state())
    return context[ent]

trannum = str.maketrans("0123456789", "1111111111")
from collections import Counter

def augment_anonymize(sentence, lang_id, ner, tag_type={'IP_ADDRESS', 'KEY', 'ID', 'PHONE', 'USER', 'EMAIL', 'LICENSE_PLATE', 'PERSON'}, faker=None, context=None, do_augment=False):
  if faker is None:
    faker = FakerExtensions(lang_id)
  if context is None:
    context = {}
  is_cjk = lang_id in ("zh", "ja", "ko", "th")
  if True:
    # we want to match the longest spans first for anonymization
    # replace entities with anchors
    new_ner = copy.deepcopy(ner)
    if type(new_ner) is dict:
      new_ner = [list(a) +  [max(Counter(b))] for a, b in new_ner.items()]
      #for key, val in ner.items():
      #  new_ner.append(Counter())
    new_ner.sort(key=lambda a: len(a[0]), reverse=True)     
    for idx, a_ner in enumerate(new_ner):
      ent = a_ner[0]
      if a_ner[-1] == 'PERSON' and not is_cjk:
          #strip out prefixes and suffixes
          ent_arr = ent.split(" ")
          if ent_arr[0][-1] == ".":
            ent_arr = ent_arr[1:]
          if ent_arr and ent_arr[-1][-1] == ".":
            ent_arr = ent_arr[:-1]
          if ent_arr:
            ent = " ".join(ent_arr)
      tag = a_ner[-1]
      if tag not in tag_type: continue
      sentence = sentence.replace(ent+" ", f"<{idx}> ")
      sentence = sentence.replace(" "+ent, f" <{idx}>")
      if len(ent) > 5:
        sentence = sentence.replace(ent, f"<{idx}>") 
    #now actually do the aug/anon
    new_ner2 = []
    for idx, a_ner in enumerate(new_ner):
      ent = a_ner[0]
      tag = a_ner[-1]
      if tag not in tag_type:
        new_ner2.append((ent, tag))
        continue
      if tag == 'PERSON': #TODO, tied to URL and USER 
        ent2 = faker.name(ent=ent, context=context)
        if (ent2, tag) not in new_ner2:
          new_ner2.append((ent2, tag))
        sentence = sentence.replace(f"<{idx}>", f' {ent2} ' if not is_cjk else ent2)
      elif tag == 'ORG':
        ent2 = faker.company(ent=ent, context=context)
        if (ent2, tag) not in new_ner2:
          new_ner2.append((ent2, tag))
        sentence = sentence.replace(f"<{idx}>", f' {ent2} ' if not is_cjk else ent2)
      elif tag == 'LOC':
        ent2 = faker.state(ent=ent, context=context)
        if (ent2, tag) not in new_ner2:
          new_ner2.append((ent2, tag))
        sentence = sentence.replace(f"<{idx}>", f' {ent2} ' if not is_cjk else ent2)
      elif tag == 'ADDRESS':
        ent2 = faker.address(ent=ent, context=context)
        if (ent2, tag) not in new_ner2:
          new_ner2.append((ent2, tag))
        sentence = sentence.replace(f"<{idx}>", f' {ent2} ' if not is_cjk else ent2)
      elif tag != 'PUBLIC_FIGURE':
        if (f' <{tag}> ', tag) not in new_ner2:
          new_ner2.append((f' <{tag}> ', tag))
        sentence = sentence.replace(f"<{idx}>", f' <{tag}> ')
      elif do_augment:
        if tag in ('IP_ADDRESS', 'ID', 'PHONE',):
          ent2 = ent.translate(trannum)
        elif tag in ('KEY',):
          ent2 = 'KEY-11111'
        elif tag in ('LP',):
          ent2 = 'LP-11111' 
        elif tag in ('USER',):
          ent2 = "@"+faker.email().split("@")[0]
        elif tag in ('EMAIL',):
          ent2 = faker.email()
      else:
        new_ner2.append((ent, tag))
  
      #TODO: NORP, AGE, DISEASE, URL, GENDER, JOB, MEDICAL_THERAPY
        
    sentence = sentence.replace("  ", " ").strip()
    new_ner3 = []
    sentence2 = copy.copy(sentence)
    len_text = len(sentence2)
    new_ner2.sort(key=lambda a: len(a[0]), reverse=True)
    for a_ner in new_ner2:
        ent, tag = a_ner
        pos = 0
        while pos < len_text and ent in sentence2[pos:]:
          i = sentence2[pos:].index(ent)
          start = pos + i
          end = start + len(ent)
          pos = end+1
          mention2 = [ent, start, end, tag]
          new_ner3.append(mention2)
        sentence2 = sentence2.replace(ent, " "*len(ent))
          
  new_ner3.sort(key=lambda a: a[1])
  return sentence, new_ner3, context

if __name__ == "__main__":
  if True:
    print (augment_anonymize('Mr. John Smith Esq. is nice. John says hi.', 'en', [['Mr. John Smith Esq.', 0, 19, 'PERSON'], ['John', 25, 28, 'PERSON']], ))
    print (augment_anonymize('John is nice. John Smith says hi.', 'en', [['John', 0, 4, 'PERSON'], ['John Smith', 14, 24, 'PERSON']], ))

  if False:
    # TODO: do "as"
    generator = FakerExtensions(lang='zh')
    fake_name = generator.name(ent="周淑", context=context)
    print ('found name', fake_name)
    fake_name = generator.name(ent="周淑华", context=context)
    print ('found name', fake_name)
  if False:
    for lang in ["zh", "pa", "gu","as", "zh", "en", "yo","mr", "ny", "sn", "st", "xh", "zu", "ar", "bn", "ca",  "es", "eu", "fr", "hi", "id", "ig", "pt",  "sw", "ur","vi",  ]:
      print (f'*** {lang}')
      generator = FakerExtensions(lang=lang)
      start_time=time.time()
      for i in range(100):
          fake_name = generator.name()
          print ('found name', fake_name)

      """
      context = {}
      for i in range(100):
          fake_name = generator.name(ent="周淑", context=context)
          print ('found name', fake_name)
          fake_name = generator.name(ent="周淑华", context=context)
          print ('found name', fake_name)
      """
      print(f"Running time {time.time() - start_time}")
