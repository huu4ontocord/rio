from fake_names import *
from text_augment import TextAugment
from typing import List
import random
import time

class FakeNameGenerator:
  def __init__(
      self,
      lang: str = "vi",
      trials: int = 1000
  ):
      self.lang = lang
      self.trials = trials
      self.kenlm_models = TextAugment.load_kenlm_model(lang)
      self.patterns = TextAugment.public_figure_kenlm_cutoff_map.get(lang, [{'cutoff': 500, 'pattern': "{} (born"}])
      self.num_genders = 1
      if self.lang == "vi":
          self.num_genders = 2
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
      elif self.lang == "ur":
        surname_list_of_lists: List[List[str]] = [urdu_surnames]
        first_name_list_of_lists: List[List[str]] = [bengali_firstnames_male, bengali_firstnames_female]
        self.name_lists = [first_name_list_of_lists, surname_list_of_lists]
        self.name_lists_probabilities = [1.0, 1.0]
        assert len(self.name_lists) == len(self.name_lists_probabilities)

  def generate(self):
      """ Generate fake name """
      gender = random.choice(range(self.num_genders))
      output_name = []
      for i, name_list_of_lists in enumerate(self.name_lists):
          # Sometimes, we might have a single list for all genders,
          # thus we take the minimun to avoid out of index
          name_list = name_list_of_lists[min(len(name_list_of_lists) - 1, gender)]
          if random.random() <= self.name_lists_probabilities[i]:
              output_name.append(random.choice(name_list))
      return " ".join(output_name)

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
