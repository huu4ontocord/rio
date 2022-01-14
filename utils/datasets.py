import os
from typing import List

import fsspec
from datasets import load_dataset
import logging

from .LogingHandler import LoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


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


def get_docs(src_lang: str = None) -> List[str]:
    logging.info("Docs is None so trying to load dataset")

    docs = None
    domain = None
    try:
        domain = 'oscar_registry'
        d = load_dataset("TurkuNLP/register_oscar", data_files=f"{src_lang}/{src_lang}_00000*")
        docs = [doc for doc in d['train'] if 'labels' not in doc or doc['labels'] != []]
    except:
        try:
            logging.info("Failed to load oscar_registry")
            domain = 'mc4_registry'
            d = load_dataset("TurkuNLP/register_mc4", data_files=f"{src_lang}/{src_lang}_00000*")
            docs = [doc for doc in d['train'] if 'labels' not in doc or doc['labels'] != []]
        except:
            logging.info("Failed to load mc4_registry")
            domain = 'oscar'
            url = get_oscar_urls(src_lang)[0]
            download_urls([url])
            docs = [{f'{src_lang}_text': line.decode()} for line in open(url.split("/")[-1], "rb").readlines()]
    finally:
        logging.info(f"Loaded Documents in domain: {domain}")
        return docs, domain
