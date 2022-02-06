from process import TextAugment
import sys
import argparse
from torch import multiprocessing
import time
from functools import partial

def multiprocess_ner(docs,
                   outputfile,
                   src_lang=None,
                   target_lang=None,
                   do_regex=True,
                   do_spacy=True,
                   do_backtrans=True,
                   cutoff=None,
                   batch_size=5,
                   num_workers=2):
  multiprocessing.set_start_method('spawn', force=True)

  if num_workers != 0:
      chunk_size = int(len(docs) / num_workers)
      docs_chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]
  else:
    docs_chunks = [docs]
  start = time.time()
  processor = TextAugment(single_process=False)
  # processor.initializer()
  print(len(docs_chunks))
  with open(outputfile, 'w', encoding='utf-8') as file:
      # for i in range(0, num_workers):
        pool = multiprocessing.Pool(processes=num_workers, initializer=processor.initializer)

        # processed_docs = pool.imap_unordered(TextAugment._multiprocess_ner_helper,
        #                                      docs_chunks)

        processed_docs = pool.imap_unordered(partial(processor.process_ner,
                                                    # docs=docs_chunks[i],
                                                    src_lang=src_lang,
                                                    target_lang=target_lang,
                                                    do_regex=do_regex,
                                                    do_spacy=do_spacy,
                                                    do_backtrans=do_backtrans,
                                                    cutoff=cutoff,
                                                    batch_size=batch_size),
                                            docs_chunks)

        for i, docs in enumerate(processed_docs):
          print(f"processed {i}: (Time elapsed: {(int(time.time() - start))}s)")
          for doc in docs:
          # for doc in docs.values():
            file.write(f'{doc}\n')



if __name__ == "__main__":
    def load_py_from_str(s, default=None):
        if not s.strip(): return default
        ret = {'__ret': None}
        # print (s)
        exec("__ret= " + s, ret)
        return ret['__ret']


    def load_all_pii(infile="./zh_pii.jsonl"):
        return [load_py_from_str(s, {}) for s in open(infile, "rb").read().decode().split("\n")]
    parser = argparse.ArgumentParser(description='Text Annotation, Augmentation and Anonymization')
    parser.add_argument('-src_lang', dest='src_lang', type=str, help='Source Language', default=None)
    parser.add_argument('-target_lang', dest='target_lang', type=str, help='Target Language', default="en")
    parser.add_argument('-cutoff', dest='cutoff', type=int, help='Cutoff documents, -1 is none', default=30)
    parser.add_argument('-batch_size', dest='batch_size', type=int, help='batch size', default=5)
    parser.add_argument('-infile', dest='infile', type=str, help='file to load', default=None)
    parser.add_argument('-outfile', dest='outfile', type=str, help='file to save', default="out.jsonl")
    parser.add_argument('-num_workers', dest='num_workers', type=int, help='Num of Workers', default=1)
    parser.add_argument('-preload_cache', dest='preload_cache', action='store_true',
                      help='Preload the cache of models and data', default=False)
    parser.add_argument('-multi_process', dest='multi_process', help='Multi Processing NER', action='store_true',
                      default=False)
    args = parser.parse_args()
    src_lang = args.src_lang
    target_lang = args.target_lang
    cutoff = args.cutoff
    batch_size = args.batch_size
    multi_process = args.multi_process
    num_workers = args.num_workers
    infile = args.infile
    outfile = args.outfile
    if args.preload_cache: TextAugment.preload_cache(src_lang, target_lang)
    docs = load_all_pii(infile) if infile else TextAugment.intialize_docs(src_lang=src_lang)

    #TODO - do multiprocessing
    if not multi_process:
      processor = TextAugment(single_process=True)

      docs, chunks = processor.process_ner(docs=docs,
                                           src_lang=src_lang,
                                           target_lang=target_lang,
                                           do_regex=True,
                                           do_spacy=True,
                                           do_backtrans=True,
                                           cutoff=cutoff,
                                           batch_size=batch_size)
      print('total out docs ', len(docs))
      docs = processor.serialize_ner_items(docs, ner_keys=[src_lang, target_lang])
      with open(outfile, 'w', encoding='utf-8') as file:
        for doc in docs:
          file.write(f'{doc}\n')
    else:
        print(f"Multi Processing with {num_workers} workers")
        multiprocess_ner(docs=docs,
                                     src_lang=src_lang,
                                     target_lang=target_lang,
                                     do_regex=True,
                                     do_spacy=True,
                                     do_backtrans=True,
                                     cutoff=cutoff,
                                     batch_size=batch_size,
                                     outputfile=outfile,
                                     num_workers=num_workers)