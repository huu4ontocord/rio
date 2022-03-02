from transformers import XLMRobertaForTokenClassification, BertForTokenClassification, ElectraForTokenClassification, RobertaForTokenClassification

hf_ner_model_map = {
      "sn": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 1.0]], 
      "st": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 1.0]], 
      "ny": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 1.0]], 
      "xh": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 1.0]], 
      "zu": [["Davlan/xlm-roberta-base-sadilar-ner", XLMRobertaForTokenClassification, 1.0]], 
      "sw": [["Davlan/xlm-roberta-base-masakhaner", XLMRobertaForTokenClassification, 1.0]], 
      "yo": [["Davlan/xlm-roberta-base-masakhaner", XLMRobertaForTokenClassification, 1.0 ]],
      "ig": [["Davlan/xlm-roberta-base-masakhaner", XLMRobertaForTokenClassification, 1.0 ]],
      "ar": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0],],
      "en": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0],], # ["bioformers/bioformer-cased-v1.0-ncbi-disease", BertForTokenClassification, 1.0] ["jplu/tf-xlm-r-ner-40-lang", None ],
      "es": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ], ],
      "eu": [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0], ],
      "ca": [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0], ],
      "pt": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ], ],
      "fr": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ], ],
      "zh": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0 ], ],
      'vi': [["lhkhiem28/COVID-19-Named-Entity-Recognition-for-Vietnamese", RobertaForTokenClassification, 1.0], ],
      'hi': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0]],
      'bn': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0]], 
      'ur': [["Davlan/xlm-roberta-base-wikiann-ner", XLMRobertaForTokenClassification, 1.0]], 
      'id': [["cahya/bert-base-indonesian-NER", BertForTokenClassification, 1.0],],

      # NOT PART OF OUR LANGUAGE SET. EXPERIMENTAL
      'he': [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8 ]],
      'hr': [["classla/bcms-bertic-ner", ElectraForTokenClassification, 1.0]],
      'bs': [["classla/bcms-bertic-ner", ElectraForTokenClassification, 1.0]],
      'sr': [["classla/bcms-bertic-ner", ElectraForTokenClassification, 1.0]],
      'cnr': [["classla/bcms-bertic-ner", ElectraForTokenClassification, 1.0]],
      'hbs': [["classla/bcms-bertic-ner", ElectraForTokenClassification, 1.0]],
      'da': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      'no': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      'nb': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      'nn': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      'sv': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      'fo': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      'is': [["saattrupdan/nbailab-base-ner-scandi", BertForTokenClassification, 1.0]],
      }

m2m100_lang = {
    ('en', 'yo'): "Davlan/m2m100_418M-eng-yor-mt",
    ('yo', 'en'): "Davlan/m2m100_418M-yor-eng-mt",
    ('en', 'zu'): "masakhane/m2m100_418M-en-zu-mt",
    ('zu', 'en'): "masakhane/m2m100_418M-zu-en-mt",
    ('*', '*') : "facebook/m2m100_418M"
    }

