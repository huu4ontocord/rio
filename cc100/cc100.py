# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
import datasets


_DESCRIPTION = """\
This corpus is an attempt to recreate the dataset used for training XLM-R. This corpus comprises of monolingual data for 100+ languages and also includes data for romanized languages (indicated by *_rom). This was constructed using the urls and paragraph indices provided by the CC-Net repository by processing January-December 2018 Commoncrawl snapshots. Each file comprises of documents separated by double-newlines and paragraphs within the same document separated by a newline. The data is generated using the open source CC-Net repository. No claims of intellectual property are made on the work of preparation of the corpus.
"""
_HOMEPAGE_URL = "https://data.statmt.org/cc-100/"
_CITATION = """\
@inproceedings{conneau-etal-2020-unsupervised,
    title = "Unsupervised Cross-lingual Representation Learning at Scale",
    author = "Conneau, Alexis  and
      Khandelwal, Kartikay  and
      Goyal, Naman  and
      Chaudhary, Vishrav  and
      Wenzek, Guillaume  and
      Guzm{'a}n, Francisco  and
      Grave, Edouard  and
      Ott, Myle  and
      Zettlemoyer, Luke  and
      Stoyanov, Veselin",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.747",
    doi = "10.18653/v1/2020.acl-main.747",
    pages = "8440--8451",
    abstract = "This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +14.6{%} average accuracy on XNLI, +13{%} average F1 score on MLQA, and +2.4{%} F1 score on NER. XLM-R performs particularly well on low-resource languages, improving 15.7{%} in XNLI accuracy for Swahili and 11.4{%} for Urdu over previous XLM models. We also present a detailed empirical analysis of the key factors that are required to achieve these gains, including the trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing per-language performance; XLM-R is very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We will make our code and models publicly available.",
}
@inproceedings{wenzek-etal-2020-ccnet,
    title = "{CCN}et: Extracting High Quality Monolingual Datasets from Web Crawl Data",
    author = "Wenzek, Guillaume  and
      Lachaux, Marie-Anne  and
      Conneau, Alexis  and
      Chaudhary, Vishrav  and
      Guzm{'a}n, Francisco  and
      Joulin, Armand  and
      Grave, Edouard",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.494",
    pages = "4003--4012",
    abstract = "Pre-training text representations have led to significant improvements in many areas of natural language processing. The quality of these models benefits greatly from the size of the pretraining corpora as long as its quality is preserved. In this paper, we describe an automatic pipeline to extract massive high-quality monolingual datasets from Common Crawl for a variety of languages. Our pipeline follows the data processing introduced in fastText (Mikolov et al., 2017; Grave et al., 2018), that deduplicates documents and identifies their language. We augment this pipeline with a filtering step to select documents that are close to high quality corpora like Wikipedia.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
"""

_VERSION = "1.0.0"
_BASE_URL = "https://data.statmt.org/cc-100/{}.txt.xz"

"""
From : https://data.statmt.org/cc-100/

af Afrikaans (305M)
am Amharic (133M)
ar Arabic (5.4G)
as Assamese (7.6M)
az Azerbaijani (1.3G)
be Belarusian (692M)
bg Bulgarian (9.3G)
bn Bengali (860M)
bn_rom Bengali Romanized (164M)
br Breton (21M)
bs Bosnian (18M)
ca Catalan (2.4G)
cs Czech (4.4G)
cy Welsh (179M)
da Danish (12G)
de German (18G)
el Greek (7.4G)
en English (82G)
eo Esperanto (250M)
es Spanish (14G)
et Estonian (1.7G)
eu Basque (488M)
fa Persian (20G)
ff Fulah (3.1M)
fi Finnish (15G)
fr French (14G)
fy Frisian (38M)
ga Irish (108M)
gd Scottish Gaelic (22M)
gl Galician (708M)
gn Guarani (1.5M)
gu Gujarati (242M)
ha Hausa (61M)
he Hebrew (6.1G)
hi Hindi (2.5G)
hi_rom Hindi Romanized (129M)
hr Croatian (5.7G)
ht Haitian (9.1M)
hu Hungarian (15G)
hy Armenian (776M)
id Indonesian (36G)
ig Igbo (6.6M)
is Icelandic (779M)
it Italian (7.8G)
ja Japanese (15G)
jv Javanese (37M)
ka Georgian (1.1G)
kk Kazakh (889M)
km Khmer (153M)
kn Kannada (360M)
ko Korean (14G)
ku Kurdish (90M)
ky Kyrgyz (173M)
la Latin (609M)
lg Ganda (7.3M)
li Limburgish (2.2M)
ln Lingala (2.3M)
lo Lao (63M)
lt Lithuanian (3.4G)
lv Latvian (2.1G)
mg Malagasy (29M)
mk Macedonian (706M)
ml Malayalam (831M)
mn Mongolian (397M)
mr Marathi (334M)
ms Malay (2.1G)
my Burmese (46M)
my_zaw Burmese (Zawgyi) (178M)
ne Nepali (393M)
nl Dutch (7.9G)
no Norwegian (13G)
ns Northern Sotho (1.8M)
om Oromo (11M)
or Oriya (56M)
pa Punjabi (90M)
pl Polish (12G)
ps Pashto (107M)
pt Portuguese (13G)
qu Quechua (1.5M)
rm Romansh (4.8M)
ro Romanian (16G)
ru Russian (46G)
sa Sanskrit (44M)
si Sinhala (452M)
sc Sardinian (143K)
sd Sindhi (67M)
sk Slovak (6.1G)
sl Slovenian (2.8G)
so Somali (78M)
sq Albanian (1.3G)
sr Serbian (1.5G)
ss Swati (86K)
su Sundanese (15M)
sv Swedish (21G)
sw Swahili (332M)
ta Tamil (1.3G)
ta_rom Tamil Romanized (68M)
te Telugu (536M)
te_rom Telugu Romanized (79M)
th Thai (8.7G)
tl Tagalog (701M)
tn Tswana (8.0M)
tr Turkish (5.4G)
ug Uyghur (46M)
uk Ukrainian (14G)
ur Urdu (884M)
ur_rom Urdu Romanized (141M)
uz Uzbek (155M)
vi Vietnamese (28G)
wo Wolof (3.6M)
xh Xhosa (25M)
yi Yiddish (51M)
yo Yoruba (1.1M)
zh-Hans Chinese (Simplified) (14G)
zh-Hant Chinese (Traditional) (5.3G)
zu Zulu (4.3M)
"""

cc100_langs = {'af': 'Afrikaans',
 'am': 'Amharic',
 'ar': 'Arabic',
 'as': 'Assamese',
 'az': 'Azerbaijani',
 'be': 'Belarusian',
 'bg': 'Bulgarian',
 'bn': 'Bengali',
 'bn_rom': 'Bengali Romanized',
 'br': 'Breton',
 'bs': 'Bosnian',
 'ca': 'Catalan',
 'cs': 'Czech',
 'cy': 'Welsh',
 'da': 'Danish',
 'de': 'German',
 'el': 'Greek',
 'en': 'English',
 'eo': 'Esperanto',
 'es': 'Spanish',
 'et': 'Estonian',
 'eu': 'Basque',
 'fa': 'Persian',
 'ff': 'Fulah',
 'fi': 'Finnish',
 'fr': 'French',
 'fy': 'Frisian',
 'ga': 'Irish',
 'gd': 'Scottish Gaelic',
 'gl': 'Galician',
 'gn': 'Guarani',
 'gu': 'Gujarati',
 'ha': 'Hausa',
 'he': 'Hebrew',
 'hi': 'Hindi',
 'hi_rom': 'Hindi Romanized',
 'hr': 'Croatian',
 'ht': 'Haitian',
 'hu': 'Hungarian',
 'hy': 'Armenian',
 'id': 'Indonesian',
 'ig': 'Igbo',
 'is': 'Icelandic',
 'it': 'Italian',
 'ja': 'Japanese',
 'jv': 'Javanese',
 'ka': 'Georgian',
 'kk': 'Kazakh',
 'km': 'Khmer',
 'kn': 'Kannada',
 'ko': 'Korean',
 'ku': 'Kurdish',
 'ky': 'Kyrgyz',
 'la': 'Latin',
 'lg': 'Ganda',
 'li': 'Limburgish',
 'ln': 'Lingala',
 'lo': 'Lao',
 'lt': 'Lithuanian',
 'lv': 'Latvian',
 'mg': 'Malagasy',
 'mk': 'Macedonian',
 'ml': 'Malayalam',
 'mn': 'Mongolian',
 'mr': 'Marathi',
 'ms': 'Malay',
 'my': 'Burmese',
 'my_zaw': 'Burmese',
 'ne': 'Nepali',
 'nl': 'Dutch',
 'no': 'Norwegian',
 'ns': 'Northern Sotho',
 'om': 'Oromo',
 'or': 'Oriya',
 'pa': 'Punjabi',
 'pl': 'Polish',
 'ps': 'Pashto',
 'pt': 'Portuguese',
 'qu': 'Quechua',
 'rm': 'Romansh',
 'ro': 'Romanian',
 'ru': 'Russian',
 'sa': 'Sanskrit',
 'sc': 'Sardinian',
 'sd': 'Sindhi',
 'si': 'Sinhala',
 'sk': 'Slovak',
 'sl': 'Slovenian',
 'so': 'Somali',
 'sq': 'Albanian',
 'sr': 'Serbian',
 'ss': 'Swati',
 'su': 'Sundanese',
 'sv': 'Swedish',
 'sw': 'Swahili',
 'ta': 'Tamil',
 'ta_rom': 'Tamil Romanized',
 'te': 'Telugu',
 'te_rom': 'Telugu Romanized',
 'th': 'Thai',
 'tl': 'Tagalog',
 'tn': 'Tswana',
 'tr': 'Turkish',
 'ug': 'Uyghur',
 'uk': 'Ukrainian',
 'ur': 'Urdu',
 'ur_rom': 'Urdu Romanized',
 'uz': 'Uzbek',
 'vi': 'Vietnamese',
 'wo': 'Wolof',
 'xh': 'Xhosa',
 'yi': 'Yiddish',
 'yo': 'Yoruba',
 'zh-Hans': 'Chinese',
 'zh-Hant': 'Chinese',
 'zu': 'Zulu'}
_LANGUAGES = list(cc100_langs.keys())


class Cc100Config(datasets.BuilderConfig):
    def __init__(self, *args, lang=None, **kwargs):
        super().__init__(
            *args,
            name=f"{lang}",
            **kwargs,
        )
        self.lang = lang


class Cc100(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        Cc100Config(
            lang=lang,
            description=f"Language: {lang}",
            version=datasets.Version(_VERSION),
        )
        for lang in _LANGUAGES
    ]
    BUILDER_CONFIG_CLASS = Cc100Config

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                },
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        def _base_url(lang):
            return _BASE_URL.format(lang)

        download_url = _base_url(self.config.lang)
        path = dl_manager.download_and_extract(download_url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"datapath": path},
            )
        ]

    def _generate_examples(self, datapath):
        with open(datapath, encoding="utf-8") as f:
            for sentence_counter, row in enumerate(f):
                result = (
                    sentence_counter,
                    {
                        "id": str(sentence_counter),
                        "text": row,
                    },
                )
                yield result
