import os
import fsspec
import glob, os
from datasets import load_dataset

langs = {
        "af": "Afrikaans",
        "als": "Tosk Albanian",
        "am": "Amharic",
        "an": "Aragonese",
        "ar": "Arabic",
        "arz": "Egyptian Arabic",
        "ast": "Asturian",
        "as": "Assamese",
        "av": "Avaric",
        "azb": "South Azerbaijani",
        "az": "Azerbaijani",
        "bar": "Bavarian",
        "ba": "Bashkir",
        "bcl": "Central Bikol",
        "be": "Belarusian",
        "bg": "Bulgarian",
        "bh": "Bihari",
        "bn": "Bengali",
        "bo": "Tibetan",
        "bpy": "Bishnupriya",
        "br": "Breton",
        "bs": "Bosnian",
        "bxr": "Russia Buriat",
        "ca": "Catalan",
        "cbk": "Chavacano",
        "ceb": "Cebuano",
        "ce": "Chechen",
        "ckb": "Central Kurdish",
        "cs": "Czech",
        "cv": "Chuvash",
        "cy": "Welsh",
        "da": "Danish",
        "de": "German",
        "diq": "Dimli",
        "dsb": "Lower Sorbian",
        "dv": "Dhivehi",
        "el": "Modern Greek",
        "eml": "Emilian-Romagnol",
        "en": "English",
        "eo": "Esperanto",
        "es": "Spanish",
        "et": "Estonian",
        "eu": "Basque",
        "fa": "Persian",
        "fi": "Finnish",
        "frr": "Northern Frisian",
        "fr": "French",
        "fy": "Western Frisian",
        "ga": "Irish",
        "gd": "Scottish Gaelic",
        "gl": "Galician",
        "gn": "Guarani",
        "gom": "Goan Konkani",
        "gu": "Gujarati",
        "he": "Hebrew",
        "hi": "Hindi",
        "hr": "Croatian",
        "hsb": "Upper Sorbian",
        "ht": "Haitian",
        "hu": "Hungarian",
        "hy": "Armenian",
        "ia": "Interlingua",
        "id": "Indonesian",
        "ie": "Interlingue",
        "ilo": "Iloko",
        "io": "Ido",
        "is": "Icelandic",
        "it": "Italian",
        "ja": "Japanese",
        "jbo": "Lojban",
        "jv": "Javanese",
        "ka": "Georgian",
        "kk": "Kazakh",
        "km": "Central Khmer",
        "kn": "Kannada",
        "ko": "Korean",
        "krc": "Karachay-Balkar",
        "ku": "Kurdish",
        "kv": "Komi",
        "kw": "Cornish",
        "ky": "Kirghiz",
        "la": "Latin",
        "lb": "Luxembourgish",
        "lez": "Lezghian",
        "li": "Limburgan",
        "lmo": "Lombard",
        "lo": "Lao",
        "lrc": "Northern Luri",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "mai": "Maithili",
        "mg": "Malagasy",
        "mhr": "Eastern Mari",
        "min": "Minangkabau",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mn": "Mongolian",
        "mrj": "Western Mari",
        "mr": "Marathi",
        "ms": "Malay",
        "mt": "Maltese",
        "mwl": "Mirandese",
        "my": "Burmese",
        "myv": "Erzya",
        "mzn": "Mazanderani",
        "nah": "Nahuatl", # languages
        "nap": "Neapolitan",
        "nds": "Low German",
        "ne": "Nepali",
        "new": "Newari",
        "nl": "Dutch",
        "nn": "Norwegian Nynorsk",
        "no": "Norwegian",
        "oc": "Occitan",
        "or": "Oriya",
        "os": "Ossetian",
        "pam": "Pampanga",
        "pa": "Panjabi",
        "pl": "Polish",
        "pms": "Piemontese",
        "pnb": "Western Panjabi",
        "ps": "Pushto",
        "pt": "Portuguese",
        "qu": "Quechua",
        "rm": "Romansh",
        "ro": "Romanian",
        "ru": "Russian",
        "sah": "Yakut",
        "sa": "Sanskrit",
        "scn": "Sicilian",
        "sd": "Sindhi",
        "sh": "Serbo-Croatian",
        "si": "Sinhala",
        "sk": "Slovak",
        "sl": "Slovenian",
        "so": "Somali",
        "sq": "Albanian",
        "sr": "Serbian",
        "su": "Sundanese",
        "sv": "Swedish",
        "sw": "Swahili",
        "ta": "Tamil",
        "te": "Telugu",
        "tg": "Tajik",
        "th": "Thai",
        "tk": "Turkmen",
        "tl": "Tagalog",
        "tr": "Turkish",
        "tt": "Tatar",
        "tyv": "Tuvinian",
        "ug": "Uighur",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "uz": "Uzbek",
        "vec": "Venetian",
        "vi": "Vietnamese",
        "vo": "Volapük",
        "war": "Waray",
        "wa": "Walloon",
        "wuu": "Wu Chinese",
        "xal": "Kalmyk",
        "xmf": "Mingrelian",
        "yi": "Yiddish",
        "yo": "Yoruba",
        "yue": "Yue Chinese",
        "zh": "Chinese",
    }
  

def get_oscar_urls(language, shuffled="unshuffled", deduplicated="deduplicated"):
    _BASE_DATA_URL_FORMAT_STR = (
        "https://s3.amazonaws.com/datasets.huggingface.co/oscar/1.0/{shuffled}/{deduplicated}/{language}/")
    _BASE_CHECKSUM_FILE_NAME = "{language}_sha256.txt"
    base_data_url = _BASE_DATA_URL_FORMAT_STR.format(
        shuffled=shuffled, language=language, deduplicated=deduplicated
    )
    checksum_url = base_data_url + _BASE_CHECKSUM_FILE_NAME.format(language=language)
    with fsspec.open(checksum_url, encoding="utf-8") as f:
        data_filenames = [line.decode().split("\t")[0] for line in f if line]
        return [base_data_url + data_filename for data_filename in data_filenames]


def download_urls(urls):
    for url in urls:
        if not os.path.exists(url.split("/")[-1]):
            os.system(f"wget {url}")

def create_mdd_dataset(dst_data="./dst_data/", min_size1=200_000_000, min_size2=100_000_000):
  os.system(f"mkdir -p {dst_data}")
  for lang in langs:
    url = get_oscar_urls(lang)[:1]
    download_urls(url)
    file_path = url.split("/")[-1]
    file_size = os.path.getsize(file_path)
    os.system(f"mv {file_path} {dst_data}")
    if file_size < min_size1:
      #print(file_path, file_size)
      lang = file_path.split("/")[-1].replace("_dedup.txt.gz", "")
      #if lang == "mt": continue
      os.system(f"wget https://data.statmt.org/cc-100/{lang}.txt.xz")
      file_path2 = f"{lang}.txt.xz"
      if os.path.exists(file_path2):
        file_size2 = os.path.getsize(file_path2)
        print(file_path2, file_size2)
        if file_size2 < min_size2 and not os.path.exists(f"{lang}.mc4.txt.gz"):
          try:
            ds = load_dataset("mc4", lang)
            with open(f"{lang}.mc4.txt", "w", encoding="utf8") as f:
              for t in ds['train']['text']:
                f.write(t+"\n")
            os.system(f"gzip {lang}.mc4.txt")
            os.system(f"mv {lang}.mc4.txt.gz {dst_data}")
          except:
            pass
        os.system(f"mv {lang}.txt.xz {dst_data}")
      elif  not os.path.exists(f"{lang}.mc4.txt.gz"):
        try:
          ds = load_dataset("mc4", lang)
          with open(f"{lang}.mc4.txt", "w", encoding="utf8") as f:
              for t in ds['train']['text']:
                f.write(t+"\n")
          os.system(f"gzip {lang}.mc4.txt")
          os.system(f"mv {lang}.mc4.txt.gz {dst_data}")
        except:
          pass  
