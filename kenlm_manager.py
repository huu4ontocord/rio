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
# from Eduardo Gonzalez Ponferrada/edugp's repo: https://huggingface.co/edugp/kenlm/blob/main/model.py which is under the Apache 2 License
# thank you edugp!!
import os
import re
import unicodedata
from typing import Dict
import warnings
import kenlm
import sentencepiece
from huggingface_hub import cached_download, hf_hub_url

## additional code to support kenlm entity querying
kenlm_models = {
    'wikipedia': {},
    'oscar': {},
    'mc4': {},
}

# TODO figure out actual numbers. Also, add languge specific kenlm models. Check if there are variations b/c of
#  gender, so we would have at least two patterns.
public_figure_kenlm_cutoff_map = {
    'en': {'wikipedia': [{'cutoff': 500, 'pattern': "{} (born"}],  # in wikipedia, you often have: Lincoln (born .... )
           'oscar': [{'cutoff': 500, 'pattern': "{} was born"}],
           },
    'yo': {'wikipedia': [{'cutoff': 400, 'pattern': "{} ni a bi lori"}],
           'oscar': [{'cutoff': 400, 'pattern': "{} ni a bi lori"}],
           },
    'zu': {'wikipedia': [{'cutoff': 400, 'pattern': "{} wazalwa ngo"}],
           'oscar': [{'cutoff': 400, 'pattern': "{} wazalwa ngo"}],
           'mc4': [{'cutoff': 400, 'pattern': "{} wazalwa ngo"}],  # for now, we are using the mc4 model for zu and ig
           },
    'sn': {'wikipedia': [{'cutoff': 500, 'pattern': "{} akazvarwa"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} akazvarwa"}],
           },
    'st': {'wikipedia': [{'cutoff': 500, 'pattern': "{} o hlahile ka"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} o hlahile ka"}],
           },
    'ny': {'wikipedia': [{'cutoff': 500, 'pattern': "{} anabadwa pa"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} anabadwa pa"}],
           },
    'xh': {'wikipedia': [{'cutoff': 500, 'pattern': "{} wazalwa ngo"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} wazalwa ngo"}],
           },
    'sw': {'wikipedia': [{'cutoff': 500, 'pattern': "{} alizaliwa tarehe"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} alizaliwa tarehe"}],
           },
    'ig': {'wikipedia': [{'cutoff': 300, 'pattern': "{} amụrụ"}],
           'oscar': [{'cutoff': 300, 'pattern': "{} amụrụ"}],
           'mc4': [{'cutoff': 300, 'pattern': "{} amụrụ"}],
           },
    'ar': {'wikipedia': [{'cutoff': 600, 'pattern': "ولد {} من"}],
           'oscar': [{'cutoff': 600, 'pattern': "ولد {} من"}]
           },
    'zh': {'wikipedia': [{'cutoff': 500, 'pattern': "{}生於"}],
           'oscar': [{'cutoff': 500, 'pattern': "{}生於"}]
           },
    'vi': {'wikipedia': [{'cutoff': 500, 'pattern': "{} sinh ra"},
                         {'cutoff': 500, 'pattern': "{} sáng lập"}],
           'oscar': [{'cutoff': 450, 'pattern': "{} sinh ra"},
                     {'cutoff': 450, 'pattern': "{} sáng lập"}],
           },
    'hi': {'wikipedia': [{'cutoff': 500, 'pattern': "{} का जन्म ए"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} का जन्म ए"}],
           },
    'ur': {'wikipedia': [{'cutoff': 500, 'pattern': "{} پیدا ہوا"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} پیدا ہوا"}],
           },
    'id': {'wikipedia': [{'cutoff': 500, 'pattern': "{} lahir"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} lahir"}],
           },
    'bn': {'wikipedia': [{'cutoff': 500, 'pattern': "{} জন্ম"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} জন্ম"}],
           }
}


def load_kenlm_model(
        src_lang: str = "en",
        pretrained_models: list = ['wikipedia'],
        store_model: bool = True,
        cache_dir: str = None,
) -> dict:
    """
    Load all supported kenlm model for source language. Consider if we want to use an LRU.
    TODO: Incorporate OSCAR kenlm models. They are quite big, and we still need patterns and cutoffs.
    """
    assert len(pretrained_models) <= len(
        kenlm_models), 'Total of number kenlm models loads larger than supported kenlm models'
    src_lang = src_lang if src_lang in public_figure_kenlm_cutoff_map else "en"
    all_models = {}
    model_files = ["arpa.bin", "sp.model", "sp.vocab"]
    # cache to dir
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + "/.cache"

    # check if pretrain model exist
    for model_type in pretrained_models:
        if model_type not in kenlm_models.keys():
            warnings.warn(f"{model_type} pretrained model is not supported!")
        else:
            if src_lang in kenlm_models[model_type]:
                all_models[model_type] = kenlm_models[model_type][src_lang]
            else:
                os.system(f"mkdir -p {cache_dir}/{model_type}")
                found = True
                for model_file in model_files:
                    if not os.path.exists(f"{cache_dir}/{model_type}/{src_lang}.{model_file}"):
                        try:
                            file_url = hf_hub_url(repo_id="edugp/kenlm",
                                                  filename=f"{model_type}/{src_lang}.{model_file}")
                            file = cached_download(file_url)
                            os.system(f"ln -s {file} {cache_dir}/{model_type}/{src_lang}.{model_file}")
                        except:
                            warnings.warn(f'could not find model {src_lang}.{model_file}. will stop searching...')
                            found = False
                            break
                if found:
                    model = KenlmModel(f"{cache_dir}/{model_type}", src_lang)
                    all_models[model_type] = model
                    if store_model:
                        kenlm_models[model_type][src_lang] = model
    return all_models


# TODO: refactor code in the faker_extensions with this code
def check_for_common_name(
        src_lang: str = "en",
        pretrained_models: list = ['wikipedia'],
        name: str = None,
        verbose: bool = False,
        kenlm_models=None,
        return_score=False,
):
    """
    Check if a name is a public figure or a very common name
    """
    # load all kenlm models and cutoff patterns
    if kenlm_models is None:
        kenlm_models = load_kenlm_model(src_lang, pretrained_models)
    public_patterns = public_figure_kenlm_cutoff_map.get(src_lang, public_figure_kenlm_cutoff_map.get('en'))
    for model_type, model in kenlm_models.items():
        for pattern in public_patterns.get(model_type, public_patterns.get('wikipedia')):
            test_name = pattern['pattern'].format(name)
            score = model.get_perplexity(test_name)
            if score < pattern['cutoff']:
                #if verbose:
                #    print(name, score)
                if return_score:
                    return True, score, pattern['cutoff']
                return True
    if return_score:
        return False, 0.0, 0.0
    return False


### Edugp code

class SentencePiece:
    def __init__(
            self,
            model: str,
    ):
        super().__init__()
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(model))

    def do(self, text: dict) -> dict:
        tokenized = self.sp.encode_as_pieces(text)
        return " ".join(tokenized)


class KenlmModel:
    digit_re: re.Pattern = re.compile(r"\d")
    unicode_punct: Dict[str, str] = {
        "，": ",",
        "。": ".",
        "、": ",",
        "„": '"',
        "”": '"',
        "“": '"',
        "«": '"',
        "»": '"',
        "１": '"',
        "」": '"',
        "「": '"',
        "《": '"',
        "》": '"',
        "´": "'",
        "∶": ":",
        "：": ":",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "；": ";",
        "–": "-",
        "—": " - ",
        "．": ". ",
        "～": "~",
        "’": "'",
        "…": "...",
        "━": "-",
        "〈": "<",
        "〉": ">",
        "【": "[",
        "】": "]",
        "％": "%",
        "►": "-",
    }
    unicode_punct_re = re.compile(f"[{''.join(unicode_punct.keys())}]")
    non_printing_chars_re = re.compile(
        f"[{''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))}]"
    )
    kenlm_model_dir = None
    sentence_piece_model_dir = None

    def __init__(
            self,
            model_dataset: str,
            language: str,
            lower_case: bool = False,
            remove_accents: bool = False,
            normalize_numbers: bool = True,
            punctuation: int = 1,
    ):
        self.model_dataset = model_dataset
        self.model = kenlm.Model(os.path.join(self.model_dataset, f"{language}.arpa.bin"))
        self.tokenizer = SentencePiece(os.path.join(self.model_dataset, f"{language}.sp.model"))
        self.accent = remove_accents
        self.case = lower_case
        self.numbers = normalize_numbers
        self.punct = punctuation
        self.language = language

    @classmethod
    def from_pretrained(
            cls,
            model_dataset: str,
            language: str,
    ):
        return cls(
            model_dataset,
            language,
            False,
            False,
            True,
            1,
        )

    def pp(self, log_score, length):
        return 10.0 ** (-log_score / length)

    def get_perplexity(self, doc: str, normalize_cc_net: bool = True):
        if normalize_cc_net:
            doc = self.normalize(
                doc,
                accent=self.accent,
                case=self.case,
                numbers=self.numbers,
                punct=self.punct,
            )
        # Tokenize (after normalizing): See https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/mine.py#L352 for full pipeline
        doc = self.tokenizer.do(doc)
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = self.model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return round(self.pp(doc_log_score, doc_length), 1)

    def normalize(
            self,
            line: str,
            accent: bool = True,
            case: bool = True,
            numbers: bool = True,
            punct: int = 1,
    ) -> str:
        line = line.strip()
        if not line:
            return line
        if case:
            line = line.lower()
        if accent:
            line = self.strip_accents(line)
        if numbers:
            line = self.digit_re.sub("0", line)
        if punct == 1:
            line = self.replace_unicode_punct(line)
        elif punct == 2:
            line = self.remove_unicode_punct(line)
        line = self.remove_non_printing_char(line)
        return line

    @staticmethod
    def strip_accents(line: str) -> str:
        """Strips accents from a piece of text."""
        nfd = unicodedata.normalize("NFD", line)
        output = [c for c in nfd if unicodedata.category(c) != "Mn"]
        if len(output) == line:
            return line
        return "".join(output)

    def replace_unicode_punct(self, text: str) -> str:
        return "".join(self.unicode_punct.get(c, c) for c in text)

    def remove_unicode_punct(self, text: str) -> str:
        """More aggressive version of replace_unicode_punct but also faster."""
        return self.unicode_punct_re.sub("", text)

    def remove_non_printing_char(self, text: str) -> str:
        return self.non_printing_chars_re.sub("", text)

    def check_common_name(self, name: str, return_score: bool = False):
        """
        Check if a name is a common name.

        :param name: Name to check.
        :param return_score: If True, return the score of the name and cutoff threshold of the pattern.
        :return: True if name is a common name, False otherwise.
        """
        public_patterns = public_figure_kenlm_cutoff_map.get(self.language, public_figure_kenlm_cutoff_map.get('en'))
        model_type = self.model_dataset.split("/")[-1]
        for pattern in public_patterns.get(model_type, public_patterns.get('wikipedia')):
            test_name = pattern['pattern'].format(name)
            score = self.get_perplexity(test_name)
            if score < pattern['cutoff']:
                if return_score:
                    return True, score, pattern['cutoff']
                return True
        if return_score:
            return False, 0.0, 0.0
        return False
