import re, regex
regex_rulebase = {

    "AGE": {
      "en": [
          (
              re.compile(
                  r"\S+ years old|\S+\-years\-old|\S+ year old|\S+\-year\-old", re.IGNORECASE
              ),
              None, None
          )
      ],
       "zh": [(regex.compile(r"\d{1,3}歲|\d{1,3}岁"), None, None)],
    },
    "DATE": {
    },
    #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
    "TIME": {
      "default": [(re.compile('\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?', re.IGNORECASE), None, None),],
    },
    "URL": {
      "default": [(re.compile('https?:\/\/[^\s\"\']{8,50}|www[^\s\"\']{8,50}', re.IGNORECASE), None, None)],
    },
    "ADDRESS": {
      "en": [
              #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
              (
                  re.compile(
                      r"\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$).*\b\d{5}(?:[-\s]\d{4})?\b|\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)", re.IGNORECASE
                  ),
                  None, None
              ),
      ],
      #from https://github.com/Aggregate-Intellect/bigscience_aisc_pii_detection/blob/main/language/zh/rules.py which is under Apache 2
      "zh": [
          (
              regex.compile(
                  r"((\p{Han}{1,3}(自治区|省))?\p{Han}{1,4}((?<!集)市|县|州)\p{Han}{1,10}[路|街|道|巷](\d{1,3}[弄|街|巷])?\d{1,4}号)"
              ),
              None, None
          ),
          (
              regex.compile(
                  r"(?<zipcode>(^\d{5}|^\d{3})?)(?<city>\D+[縣市])(?<district>\D+?(市區|鎮區|鎮市|[鄉鎮市區]))(?<others>.+)"
              ),
              None, None
          ),
      ],
    },
    "PHONE": {
      "zh" : [(re.compile(r"\d{4}-\d{8}"), None, None),],
      # we can probably remove one of the below
      "default": [
              # https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py phone with exts
              (
                  re.compile('((?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?(?:[2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?(?:[0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(?:\d+)?))', re.IGNORECASE),
                  None, None
              ),
              # common regex phone
              (
                  re.compile('((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))'),
                  None, None
              ), 
              ( re.compile('[\+\d]?(\d{2,3}[-\.\s]??\d{2,3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'), None, None)     
      ]      
    },
    "IP_ADDRESS": {
        "default": [(re.compile('(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', re.IGNORECASE), None, None),]
              
        },
    "USER": {
      "default": [
              # generic user id
              (re.compile(r"\s@[a-z][0-9a-z]{4-8}", re.IGNORECASE), None, None),
              #email
              (re.compile("(\w+[a-z0-9!#$%&'*+\/=?^_`{|.}~-]*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)", re.IGNORECASE), None, None),
      ]    
    },
    "ID": {
      #"en": [
              #en license plate
              #(re.compile("LICENSE_PLATE", regex.compile('[A-Z]{3}-\d{4}|[A-Z]{1,3}-[A-Z]{1,2}-\d{1,4}'), None, None)
      #],
      #"zh": [
              #LICENSE_PLATE             
      #        (regex.compile('^(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领 A-Z]{1}[A-HJ-NP-Z]{1}(?:(?:[0-9]{5}[DF])|(?:[DF](?:[A-HJ-NP-Z0-9])[0-9]{4})))|(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领 A-Z]{1}[A-Z]{1}[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9 挂学警港澳]{1})$'), None, None),
      #],
      "default": [
              #credit card from common regex
              (re.compile('((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])'), None, None),
              #icd code - see https://stackoverflow.com/questions/5590862/icd9-regex-pattern
              (re.compile('[A-TV-Z][0-9][A-Z0-9](\.[A-Z0-9]{1,4})'), None, None),
              # generic id with dashes
              (re.compile('[A-Z#]{0,3}(?:[-\./ ]*\d){6,13}'), None, ('PP ', 'PP. ', 'P.O. BOX', 'PO BOX', 'pp ', 'pp. ', 'p.o. box', 'po box',)),
              # IBAN
              (re.compile('[A-Z]{2}\d+\d+[A-Z]{0,4}(?:[- ]*\d){10,32}[A-Z]{0,3}'), None, None),
      ],
    },
 }
