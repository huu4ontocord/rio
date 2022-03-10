#deduplicate text data at the sentence level. 
import glob, os, tqdm
import multiprocessing


strip_chars = " ,،、{}[]|()\"'“”《》«»!:;?.。…．"
     
def dedup(langs, src_folder="./src_data/" dst_folder="./dst_data/", min_file_size=1_000_000, max_file_size=2_000_000_000):
  prev_lang = ""
  dup_hash = {}
  o = None
  os.system(f"mkdir -p {dest_folder}")
  for file_path in tqdm.tqdm(glob.glob(f"{src_data}/*")):
    file_size = os.path.getsize(file_path)
    if file_size > min_file_size:
      file_name = file_path.split("/")[-1]
      #print(file_path, file_size)
      lang = file_path.split("/")[-1].replace("_part_1.txt.gz", "").replace("_dedup.txt.gz", "").replace(".txt.xz", "").replace(".txt.gz", "")
      if lang not in langs: continue
      if os.path.exists(f"{dest_folder}/{lang}_mdd.txt.gz"): continue
      if lang != prev_lang:
        dup_hash = {}
        if o is not None: 
          o.close()
          os.system(f"gzip {prev_lang}_mdd.txt; mv {prev_lang}_mdd.txt.gz {dest_folder}")
        prev_lang = lang
        o = open(f"{lang}_mdd.txt", "w", encoding="utf8")
      txt_filename = file_name.replace(".gz", "").replace(".xz", "")
      if not os.path.exists(txt_filename):
        os.system(f"cp {file_path} /content/")
        if file_path.endswith("xz"):
          os.system(f"xz -d {file_name}")
        else:
          os.system(f"gunzip {file_name}")
      i = 0
      with open(txt_filename, "rb") as f:
        for line in f:
          line = line.decode().strip()
          if len(line) < 150: 
            #print ('skipping', line)
            continue
          all_lines = []
          line = line.replace("。", ". ").replace(".\"", "\".").replace(".”", "”.").replace(".》", "》.").replace(".»", "».")
          for line2 in line.split(". "):
            line3 = line2.lower().replace(" ", "").strip()
            if not line3: continue
            code = hash(line3)
            if code in dup_hash:
              if len(line3) < 15:
                #print ('found dup but short line', line2)
                all_lines.append(line2+".")
                continue
              #print ('dup', line2)
            else:
              dup_hash[code] = 1
              all_lines.append(line2 +".")
          if all_lines: 
            line4 = " ".join(all_lines)
            if line4[-2] in strip_chars:
              line4 = line4[:-1]
            if len(line4) < 150: continue
            o.write(line4+"\n")
        i+=1
        if i+1%1000==0:
          file_size = os.path.getsize(f"{lang}_mdd.txt")
          if file_size >max_file_size:
            break
      os.system(f"rm {txt_filename}")

  if o is not None: 
    o.close()
    os.system(f"gzip {prev_lang}_mdd.txt; mv {prev_lang}_mdd.txt.gz {dest_folder}")   
