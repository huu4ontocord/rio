import glob, json, itertools

def load_py_from_str(s, default=None):
  if not s.strip(): return default
  ret = {'__ret': None}
  #print (s)
  exec("__ret= "+s, ret)
  return ret['__ret']

def load_all_pii(infile="./all_pii.jsonl"):
  return [load_py_from_str(s, {}) for s in open(infile, "rb").read().decode().split("\n")]

def create_all_pii(src_lang=None, infiles = ["/content/drive/Shareddrives/BigScience/pii_annotated/reviewed/*", "/content/drive/Shareddrives/BigScience/pii_annotated/annotated/pii_*"], outfile="all_pii.jsonl"):
  seen = {}
  with open(outfile, "w", encoding="utf8") as f:
    for file in list(itertools.chain(*[glob.glob(d) for d in infiles])):
      dat = json.load(open(file))
      lang = file.split("pii_")[-1].split("_")[0]
      if lang == 'hindi': lang = 'hi'
      if src_lang is None or src_lang  == lang:
        #if lang == 'fa': continue
        #print (dat)
        dlist = dat.get('examples',[]) + dat.get('data', [])
        if len(dlist) < 100: continue
        #if not dat['examples'][0]['metadata']: continue
        # dat['examples'][-1]['content'],
        #'classname': 'No PII'
        for i in range(len(dlist)):
          annotations = dlist[i].get('annotations', [])
          if annotations or dlist[i]['metadata'].get('ner'):
            ner = dlist[i]['metadata'].get('ner', [])
            domain = dlist[i]['metadata'].get('domain', '')
            _id = dlist[i]['metadata'].get('id', '')
            ner2 = {}
            labels3 = []
            ner3 = {}
            text = dlist[i]['content']
            dat2 = {}
            if text.replace(" ", "").lower() in seen:
              dat2 = seen[text.replace(" ", "").lower()]
              labels3 = dat2['labels']
              ner3 = dat2.get(f'{lang}_ner', {})
              #print ("already seen", lang, text)
            for entity_tag in ner:
              if not entity_tag: continue
              entity, tag = entity_tag
              if entity in text:
                start = text.index(entity)
                end = start + len(entity)
                idx = (entity, start, end)
                aHash = ner2.get(idx, {})
                aHash[tag] = aHash.get(tag, 0) +  2.0
                ner2[idx]  = aHash
            for a in annotations:
              tag = a['tag']
              idx = (a['value'], a['start'], a['end'])
              aHash = ner2.get(idx, {})
              aHash[tag] = aHash.get(tag, 0) +  1.0
              ner2[idx]  = aHash
            for idx in ner3.keys():
              aHash0 = ner3.get(idx, {})
              aHash = ner2.get(idx, {})
              for tag in aHash0.keys():
                aHash[tag] = max(aHash.get(tag, 0), aHash0.get(tag,0))
              ner2[idx]  = aHash
            labels = [a.get('classname') for a in dlist[i].get('classifications', []) if a.get('classname')] + labels3
            dat2[f'{lang}_text'] = text
            dat2[f'{lang}_ner'] = ner2
            dat2['domain']= domain
            dat2['id'] = -1 if not domain else _id
            dat2['labels'] = labels
            seen[text.replace(" ", "").lower()] = dat2
    keys = sorted(seen.keys())
    for key in keys:
      dat2 = seen[key]
      #s = json.dumps(dat2)
      f.write(str(dat2)+"\n")
      #print (dat2)
#!cp /content/drive/Shareddrives/BigScience/all_pii.jsonl.gz ./
#!gunzip all_pii.jsonl.gz
#data = load_all_pii()
