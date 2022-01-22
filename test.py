import nltk
from TextAugment import TextAugment

processor = TextAugment()
docs, chunks = processor.process_ner(src_lang="vi", do_ontology=False, do_backtrans=True, cutoff=1)
print(docs)