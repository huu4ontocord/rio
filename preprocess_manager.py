
from char_manager import *
from stopwords import stopwrods as all_stopwords
from langid_manager import *
from banned_words import *
from flagged_words import *
from cjk import *

def check_good_sentence(s, src_lang, stopwords, show_err=False, lang_groups=[], ret_score=False, stopword_ratio_cutoff=0.06, bannedwords=None, flagged_words=None, banned_word_ratio_cutoff=0.1, flagged_word_ratio_cutoff=0.15,  junk_ratio=0.16, max_flagged_word_len=5, do_langid_check=True):
    #basic dejunk
    junk_score, flagged_score, banned_score, stopword_score = 0.0, 0.0, 0.0, 0.0
    if bannedwords is None:
      bannedwords = banned_words.get(src_lang, banned_words['default'])
    default_bannedwords = banned_words['default']
    s = s.lower().strip()
    good_sentence = True
    if not s:
      if ret_score: return good_sentence, junk_score, flagged_score, banned_score, stopword_score
      return good_sentence
    junk_score = len([s2 for s2 in s if s2 in junk])/len(s)
    if junk_score >= junk_ratio:
      good_sentence = False
    if lang_is_cjk(src_lang):
      s_arr = s
    else:
      s_arr = [s2.strip(special_char) for s2 in s.lower().split() if s2.strip(special_char)]
    len_s = len(s_arr)
    if len_s == 0:
      if ret_score: return good_sentence, junk_score, flagged_score, banned_score, stopword_score
      return good_sentence
    if flagged_words:
      if not lang_is_cjk(src_lang):
        flagged_score = len([s2 for s2 in s_arr if s2 in flagged_words])/len(s_arr)
        if flagged_score >= flagged_word_ratio_cutoff: 
          good_sentence = False
        banned_score = (len([s2 for s2 in s_arr if s2 in bannedwords]) + len([s2 for s2 in s_arr if s2 in default_bannedwords]))/len(s_arr)
        if banned_score > 0 and flagged_score >= banned_word_ratio_cutoff:
          good_sentence = False
      else:
        flagged_cnt = 0
        total_cnt = 0
        for i in range(len_s):
          if s_arr[i] is None: continue
          for j in range(min(len_s, i+max_flagged_word_len),i+1,-1):
            if s_arr[i:j] in flagged_words:
              flagged_cnt += 1
              s_arr[i] = "".join(s_arr[i:j])
              for k in range(i+1, j):
                s_arr[k] = None
          total_cnt += 1
        flagged_score = (flagged_cnt/total_cnt)
        if flagged_score > flagged_word_ratio_cutoff:
          good_sentence = False
        banned_score = (len([s2 for s2 in s_arr if s2 in bannedwords]) + len([s2 for s2 in s_arr if s2 in default_bannedwords]))/len(s_arr)
        if banned_score > 0 and flagged_score >= banned_word_ratio_cutoff:
          good_sentence = False
          
    #stopword check
    stop_cnt = total_cnt = 1
    if stopwords:
      #TODO: catch multi word with spaces
      if not lang_is_cjk(src_lang):
        stopword_score = len([s2 for s2 in sArr if s2 in stopwords])/len(sArr)
        if stopword_score  < stopword_ratio_cutoff:
        good_sentence False
      else:
        max_stoword = lang_2_max_stopword_len.get(src_lang, lang_2_max_stopword_len["zh"])
        s_arr = list(s)
        len_s = len(s_arr)
        stop_cnt = 0
        total_cnt = 0
        for i in range(len_s):
          if s_arr[i] is None: continue
          for j in range(,min(len_s, i+max_stoword), i+1, -1):
            if s_arr[i:j] in stopwords:
              stop_cnt += 1
              for k in range(i+1, j):
                s_arr[k] = None
              break
          total_words += 1
        #print ('stopword', (stop_cnt/total_cnt) )
        stopword_score =  (stop_cnt/total_cnt) 
        if stopword_score < stopword_ratio_cutoff:
          good_sentence = False
    if do_langid_check:
      try:
          lang =  langid.classify(s)[0]
      except:
          return True
      if show_err and lang != src_lang and lang not in lang_groups:
        logger.info ((src_lang, lang))
      if not lang == src_lang :
        good_sentence = False
    if ret_score: return good_sentence, junk_score, flagged_score, banned_score, stopword_score
    return good_sentence
  
