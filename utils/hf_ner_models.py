from transformers import XLMRobertaForTokenClassification, BertForTokenClassification, ElectraForTokenClassification, RobertaForTokenClassification

# note that we do not have a transformer model for catalan, but spacy covers catalan and we use transfer learning from Davlan/xlm-roberta-base-ner-hrl


hf_ner_model_map = {
    "sn": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]],
    # consider using one of the smaller models
    "st": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]],
    # consider using one of the smaller models
    "ny": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]],
    # consider using one of the smaller models
    "xh": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]],
    # consider using one of the smaller models
    "zu": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]],
    # consider using one of the smaller models
    "sw": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]],
    # consider using one of the smaller models
    "yo": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]],
    "ig": [["Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification, 1.0]],
    "ar": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0]],
    "en": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0],
           ["bioformers/bioformer-cased-v1.0-ncbi-disease", BertForTokenClassification, 1.0]],
    # ["jplu/tf-xlm-r-ner-40-lang", None ],
    "es": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0]],
    "eu": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8]],
    "ca": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 0.8]],
    "pt": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0]],  # there is a
    "fr": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0]],
    "zh": [["Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification, 1.0]],
    'vi': [["lhkhiem28/COVID-19-Named-Entity-Recognition-for-Vietnamese", RobertaForTokenClassification, 1.0]],
    # ["jplu/tf-xlm-r-ner-40-lang", None ],
    'hi': [["jplu/tf-xlm-r-ner-40-lang", None, 1.0]],
    'ur': [["jplu/tf-xlm-r-ner-40-lang", None, 1.0]],
    'id': [["cahya/bert-base-indonesian-NER", BertForTokenClassification, 1.0]],
    'bn': [["sagorsarker/mbert-bengali-ner", BertForTokenClassification, 1.0]],
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