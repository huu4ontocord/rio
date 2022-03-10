from muliwai.utils.download_mdd import *
from muliwai.utils.dedup_mdd import *
from muliwai.preprocess_manager import *
from char_manager import *
from stopwords import stopwords as all_stopwords
from langid_manager import *
from banned_words import *
from flagged_words import *
from cjk import *
import glob, os, tqdm
import multiprocessing


strip_chars = " ,،、{}[]|()\"'“”《》«»!:;?.。…．"
     
def filter_mdd(langs, src_folder="./src_data/", dst_folder="./dst_data/", min_file_size=1_000_000, max_file_size=2_500_000_000):
  prev_lang = ""
  dup_hash = {}
  o = None
  os.system(f"mkdir -p {dst_folder}")
  for file_path in tqdm.tqdm(glob.glob(f"{src_folder}/*")):
    file_size = os.path.getsize(file_path)
    if file_size > min_file_size:
      file_name = file_path.split("/")[-1]
      #print(file_path, file_size)
      lang = file_path.split("/")[-1].replace("_mdd.txt.gz", "")
      if lang not in langs: continue
      if os.path.exists(f"{dst_folder}/{lang}_mdd_filtered.txt.gz"): continue
      if lang != prev_lang:
        dup_hash = {}
        stopwords = all_stopwords.get(lang, {})
        flaggedwords = flagged_words.get(lang, {})
        bannedwords = banned_words.get(lang, {})
        lang_groups = get_lang_groups(lang)
        print (lang, lang_groups)
        if o is not None: 
          o.close()
          os.system(f"gzip {prev_lang}_mdd_filtered.txt; mv {prev_lang}_mdd_filtered.txt.gz {dest_folder}")
        prev_lang = lang
        o = open(f"{lang}_mdd_filtered.txt", "w", encoding="utf8")
      txt_filename = file_name.replace(".gz", "").replace(".xz", "")
      if not os.path.exists(txt_filename):
        os.system(f"cp {file_path} ./")
        if file_path.endswith("xz"):
          os.system(f"xz -d {file_name}")
        else:
          os.system(f"gunzip {file_name}")
      i = 0
      with open(txt_filename, "rb") as f:
        for line in f:
          line = line.decode().strip()
          if not check_good_sentence(line, lang, stopwords, flaggedwords=flaggedwords, bannedwords=bannedwords,\
                                     show_err=True, lang_groups=lang_groups, banned_word_ratio_cutoff=0.005, flagged_word_ratio_cutoff=0.01,  ):
            #print ('skipping', line)
            continue
          o.write(line+"\n")
        i+=1
        if i+1%1000==0:
          file_size = os.path.getsize(f"{lang}_mdd_filtered.txt")
          if file_size >max_file_size:
            break
      os.system(f"rm {txt_filename}")

  if o is not None: 
    o.close()
    os.system(f"gzip {prev_lang}_mdd_filtered.txt; mv {prev_lang}_mdd_filtered.txt.gz {dest_folder}")   
