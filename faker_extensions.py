from faker import Faker
from faker.providers import person, company, geo, address, ssn, internet
from fake_names import *
from kenlm_model_extensions import *
import random
import time

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

class FakeNameGenerator:
  def __init__(
      self,
      lang: str = "vi",
      trials: int = 1000
  ):
      self.lang = lang
      self.trials = trials
      self.kenlm_models = load_kenlm_model(lang)
      self.patterns = public_figure_kenlm_cutoff_map.get(lang, [{'cutoff': 500, 'pattern': "{} (born"}])
      if self.lang == "vi":
          self.surname_list = [vietnamese_surnames]
          self.first_name_list = [vietnamese_firstnames_male, vietnamese_firstnames_female]
          self.middle_name_list = [[vietnamese_first_middlenames_male, vietnamese_second_middlenames_male], [vietnamese_first_middlenames_female,vietnamese_second_middlenames_female]]
      elif self.lang == "bn":
          self.surname_list = [bengali_surnames]
          self.first_name_list = [bengali_firstnames_male, bengali_firstnames_female]
          self.middle_name_list = []
      elif self.lang == "ur":
          self.surname_list = [urdu_surnames]
          self.first_name_list = [urdu_firstnames]
          self.middle_name_list = []

  def generate(self):
      """ Generate fake name """
      surname = None
      middlename = None
      firstname = None

      surname = random.choice(self.surname_list[0])
      # check if target language has different name for gender
      if len(self.first_name_list) > 1:
          gender = random.randint(0,1) # 0: male, 1: female
          firstname = random.choice(self.first_name_list[gender])
          while firstname == surname:
              firstname = random.choice(self.first_name_list[gender])
      else:
          firstname = random.choice(self.first_name_list)
          while firstname == surname:
              firstname = random.choice(self.first_name_list)

      # check target language has middle name
      if len(self.middle_name_list) > 0:
          if random.random() > 0.5: # problitiity for fake name including middle name
              middlename = random.choice(self.middle_name_list[gender][0])
      # concatenate fake name
      if middlename is not None:
          if self.lang == "vi":
              fake_name = f"{surname} {middlename} {firstname}"
          else:
              fake_name = f"{firstname} {middlename} {surname}"
      else:
          if self.lang == "vi":
              fake_name = f"{surname} {firstname}"
          else:
              fake_name = f"{firstname} {surname}"
      return fake_name

  def check_fakename(self, fake_name, verbose=False):
      """ Check fake name close to real name"""
      for model in self.kenlm_models:
          for pattern in self.patterns:
              test_name = pattern['pattern'].format(fake_name)
              if model.get_perplexity(test_name) < pattern['cutoff']:
                  if verbose:
                      print(fake_name, model.get_perplexity(test_name))
                  return True
      return False

  def create_fakename(self, verbose=False):
      """ Create fake name and varify by kelnm models """
      success = False
      for _ in range(self.trials):
          fake_name = self.generate()
          if self.check_fakename(fake_name, verbose):
              success = True
              return fake_name
      if not success:
          print('Could not find any fake name. Try reducing perplexity_cutoff')

if __name__ == "__main__":
    generator = FakeNameGenerator(lang="vi")
    start_time=time.time()
    for i in range(100):
        fake_name = generator.create_fakename(verbose=True)
    print(f"Running time {time.time() - start_time}")
