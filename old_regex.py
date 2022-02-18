# TO BE REVISED 

#from https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py which is under the MIT License
# see also for ICD https://stackoverflow.com/questions/5590862/icd9-regex-pattern - but this could be totally wrong!
# we do regex in this order in order to not capture ner inside domain names and email addresses.
#NORP, AGE, ADDRESS and DISEASE regexes are just test cases. We will use transformers and rules to detect these.
regex_rulebase_extended = {
    #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
    "DATE": {
        "default": [
            #year
            (re.compile('\d{4}'), None),
            #date
            (re.compile('(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}', re.IGNORECASE), None),
        ],
    },
    #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
    "TIME": {
        "default": [(re.compile('\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?', re.IGNORECASE), None),],
    },
    "NORP": {
      "en": [(re.compile(r"upper class|middle class|working class|lower class", re.IGNORECASE), None),],
    },
    "AGE": {
      "en": [
          (
              re.compile(
                  r"\S+ years old|\S+\-years\-old|\S+ year old|\S+\-year\-old|born [ ][\d][\d]+[\\ /.][\d][\d][\\ /.][\d][\d]+|died [ ][\d][\d]+[\\ /.][\d][\d][\\ /.][\d][\d]+", re.IGNORECASE
              ),
              None,
          )
      ],
       "zh": [(regex.compile(r"\d{1,3}歲|\d{1,3}岁|[一二三四五六七八九十百]{1,3}岁|[一二三四五六七八九十百]{1,3}歲"), None)],
    },
    # Some of this code from https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/anonymization.py which is under the Apache 2 license
    "ADDRESS": {
      "en": [
              #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
              (
                  re.compile(
                      r"\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$).*\b\d{5}(?:[-\s]\d{4})?\b|\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)", re.IGNORECASE
                  ),
                  None,
              ),
             #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py
              (
                  re.compile(r"P\.? ?O\.? Box \d+"), None
              )
      ],
      #from https://github.com/Aggregate-Intellect/bigscience_aisc_pii_detection/blob/main/language/zh/rules.py which is under Apache 2
      "zh": [
          (
              regex.compile(
                  r"((\p{Han}{1,3}(自治区|省))?\p{Han}{1,4}((?<!集)市|县|州)\p{Han}{1,10}[路|街|道|巷](\d{1,3}[弄|街|巷])?\d{1,4}号)"
              ),
              None,
          ),
          (
              regex.compile(
                  r"(?<zipcode>(^\d{5}|^\d{3})?)(?<city>\D+[縣市])(?<district>\D+?(市區|鎮區|鎮市|[鄉鎮市區]))(?<others>.+)"
              ),
              None,
          ),
      ],
    },
    "DISEASE": {
      "en": [(re.compile("diabetes|cancer|HIV|AIDS|Alzheimer's|Alzheimer|heart disease", re.IGNORECASE), None)],
      "zh": [(regex.compile(r"(糖尿|癌症|抗癌|爱滋|艾滋|愛滋|阿茲海默|老人痴呆|老人癡呆|心臟病|心脏病)", re.IGNORECASE), None)]
    },
    # many of the id_regex are from Presidio which is licensed under the MIT License
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/aba_routing_recognizer.py
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/au_abn_recognizer.py
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/us_passport_recognizer.py
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/medical_license_recognizer.py
    # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/es_nif_recognizer.py
    "ID": {
      "en": [
              (
                re.compile(r"\b[0123678]\d{3}-\d{4}-\d\b"),
                (
                    "aba",
                    "routing",
                    "abarouting",
                    "association",
                    "bankrouting",
                ),
              ),
              (
                  re.compile(r"(\b[0-9]{9}\b)"),
                  (
                      "us",
                      "united",
                      "states",
                      "passport",
                      "passport#",
                      "travel",
                      "document",
                  ),
              ),
              (
                  re.compile(r"[a-zA-Z]{2}\d{7}|[a-zA-Z]{1}9\d{7}"),
                  ("medical", "certificate", "DEA"),
              ),
              (re.compile(r"\d{3}\s\d{3}\s\d{3}"), None),
              (
                  re.compile(
                      r"GB\s?\d{6}\s?\w|GB\d{3}\s\d{3}\s\d{2}\s\d{3}|GBGD\d{3}|GBHA\d{3}}|GB\d{3} \d{4} \d{2}(?: \d{3})?|GB(?:GD|HA)\d{3}"
                  ),
                  None,
              ),
              (re.compile(r"IE\d[1-9]\d{5}\d[1-9]|IE\d{7}[1-9][1-9]?"), None),
              (re.compile(r"[1-9]\d{10}"), None),
              (
                  re.compile(r"\b\d{2}\s\d{3}\s\d{3}\s\d{3}\b|\b\d{11}\b"),
                  ("australian business number", "abn"),
              ),
      ],
      "id":[
              (
                  re.compile(
                      r"\d{6}([04][1-9]|[1256][0-9]|[37][01])(0[1-9]|1[0-2])\d{6}"
                  ),
                  None,
              )
      ],
      "es": [
              (re.compile(r"(?:ES)?\d{6-8}-?[A-Z]"), None),
              (
                  re.compile(r"\b[0-9]?[0-9]{7}[-]?[A-Z]\b"),
                  ("documento nacional de identidad", "DNI", "NIF", "identificación"),
              ),
              (re.compile(r"[1-9]\d?\d{6}|8\d{8}|9\d{8}|10\d{8}|11\d{8}|12\d{8}|"), None)
      ],
      "pt": [(re.compile(r"\d{3}\.d{3}\.d{3}-\d{2}|\d{11}"), None),
             (re.compile(r"PT\d{9}"), None),
      ],
      #from https://github.com/Aggregate-Intellect/bigscience_aisc_pii_detection/blob/main/language/zh/rules.py which is under Apache 2
      "zh": [
          (
              regex.compile(
                  r"(?:[16][1-5]|2[1-3]|3[1-7]|4[1-6]|5[0-4])\d{4}(?:19|20)\d{2}(?:(?:0[469]|11)(?:0[1-9]|[12][0-9]|30)|(?:0[13578]|1[02])(?:0[1-9]|[12][0-9]|3[01])|02(?:0[1-9]|[12][0-9]))\d{3}[\dXx]"
              ),
              None,
          ),
          (
              regex.compile(
                  r"(^[EeKkGgDdSsPpHh]\d{8}$)|(^(([Ee][a-fA-F])|([DdSsPp][Ee])|([Kk][Jj])|([Mm][Aa])|(1[45]))\d{7}$)"
              ),
              None,
          ),
          (
              regex.compile(
                  r"((\d{4}(| )\d{4}(| )\d{4}$)|([a-zA-Z][1-2]{1}[0-9]{8})|([0-3]{1}\d{8}))"
              ),
              None,
          ),
          (
              regex.compile('^(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领 A-Z]{1}[A-HJ-NP-Z]{1}(?:(?:[0-9]{5}[DF])|(?:[DF](?:[A-HJ-NP-Z0-9])[0-9]{4})))|(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领 A-Z]{1}[A-Z]{1}[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9 挂学警港澳]{1})$'),
              None
          ),
          (
              regex.compile('\b[A-Z]{3}-\d{4}\b'),
              None,
          ),
          (
              regex.compile(
                  r"(0?\d{2,4}-[1-9]\d{6,7})|({\+86|086}-| ?1[3-9]\d{9} , ([\+0]?86)?[\-\s]?1[3-9]\d{9})"
              ),
              None,
          ),
          (
              regex.compile(
                  r"((\d{4}(| )\d{4}(| )\d{4}$)|([a-zA-Z][1-2]{1}[0-9]{8})|([0-3]{1}\d{8}))((02|03|037|04|049|05|06|07|08|089|082|0826|0836|886 2|886 3|886 37|886 4|886 49|886 5|886 6|886 7|886 8|886 89|886 82|886 826|886 836|886 9|886-2|886-3|886-37|886-4|886-49|886-5|886-6|886-7|886-8|886-89|886-82|886-826|886-836)(| |-)\d{4}(| |-)\d{4}$)|((09|886 9|886-9)(| |-)\d{2}(|-)\d{2}(|-)\d{1}(|-)\d{3})"
              ),
              None,
          ),
          #consider whether we want to make PHONE a separate tag, that collapses to ID
          #phone
          (re.compile(r"\d{4}-\d{8}"), None),
      ],
      "default": [
              #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py ssn
              (
                  re.compile(
                     '(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[- ](?!00)[0-9]{2}[- ](?!0000)[0-9]{4}'
                  ),
                  None,
              ),
              # https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py phone with exts
              (
                  re.compile('((?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?(?:[2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?(?:[0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(?:\d+)?))', re.IGNORECASE),
                  None
              ),
              # phone
              (
                  re.compile('''((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))'''),
                  None,
              ),
              #email
              (re.compile("([a-z0-9!#$%&'*+\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)", re.IGNORECASE), None),
              #credit card
              (re.compile('((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])'), None),
              #ip
              (re.compile('(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', re.IGNORECASE), None),
              #ipv6
              (re.compile('((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])'), None),
              #icd code - see https://stackoverflow.com/questions/5590862/icd9-regex-pattern
              (re.compile('[A-TV-Z][0-9][A-Z0-9](\.[A-Z0-9]{1,4})'), None),
              # generic government id. consider a more complicated string with \w+ at the beginning or end
              (re.compile(r"\d{6,13}|[૦-૯]{6,13}|[೦-೯]{6,13}|[൦-൯]{6,13}|[୦-୯]{6,13}|[௦-௯]{6,13}|[۰-۹]{6,13}|[০-৯]{6,13}|[٠-٩]{6,13}|[壹-玖〡-〩零〇-九十廿卅卌百千万亿兆]{6,13}"), None),
              #more generic ids
              (
                  re.compile(
                      r"\d{2}-\d{7}-\d|\d{11}|\d{2}-\d{9}-\d|\d{4}-\d{4}-\d{4}|\d{4}-\d{7}-\d"
                  ),
                  None,
              ),
              # generic id with dashes
              (re.compile('[A-Z]{0,3}(?:[- ]*\d){6,13}'), None),
              # generic user id
              (re.compile(r"\S*@[a-zA-Z]+\S*"), None),
              # bitcoin
              (re.compile('(?<![a-km-zA-HJ-NP-Z0-9])[13][a-km-zA-HJ-NP-Z0-9]{26,33}(?![a-km-zA-HJ-NP-Z0-9])'), None),
      ],
    },
 }
