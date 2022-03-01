results = dict()
split = "test"
for lang in tqdm(["en"]): #ontology_langs
    onto = OntologyManager(lang)
    do_remove_whitespace = (lang.startswith("zh")) or (lang in ('ja'))

    try:
        dataset = load_dataset("wikiann", lang)
    except:
        print(f"{lang} not found in wikiann. skipping...")
        continue
        
    # can iterate through splits for coverage - here we use train

    dataset = dataset[split]  
    results[lang] = dict(TP=[], FN=[], FP=[], ents=[])
    
    data = dataset if num_instances is None else dataset.select(range(num_instances))
    
    for x in data:
        gold_entities = defaultdict(list)
        for span in x["spans"]:
            m = re.match("(\w+):(.+)", span)
            ent_type, ent = m.group(1), m.group(2)
            if ent_type not in relevant_tags:
                continue
            if do_remove_whitespace:
                ent = ent.replace(" ", "")
            ent = ent.strip().lower()
            gold_entities[ent_type].append(ent)

        # de-tokenize to feed into ontology pipeline
        onto_entities = defaultdict(list)
        text = " ".join(x["tokens"])
        # get ontology module output
        onto_output = onto.tokenize(text)
        for ent, ent_type in onto_output['chunk2ner'].items():
            ent_span = ent[0].strip().lower()
            ent_span = ent_span.replace("_", " ")
            if do_remove_whitespace:
                ent_span = ent_span.replace(" ", "")
            onto_entities[ent_type].append(ent_span)
        
        # compare expected matches
        TP_i, FN_i, FP_i = [], [], []
        gold_entities_list = list(set([_ for v in gold_entities.values() for _ in v]))
        onto_entities_list = list(set([_ for v in onto_entities.values() for _ in v]))

        for entity_type, entities in gold_entities.items():
            for entity in set(entities):
                # we DONT enforce entity type requirements;
                # we also DONT check for partial matches at this stage
                if entity in onto_entities_list:
                    TP_i.append({entity_type:entity})
                else:
                    FN_i.append({entity_type:entity})
        for entity_type, entities in onto_entities.items():
            for entity in set(entities):
                if entity not in gold_entities_list:
                    FP_i.append({entity_type:entity})
        
        results[lang]["ents"].append(
            {
                "expected": gold_entities_list,
                "predicted": onto_entities_list
            }
        )

        results[lang]["TP"].append(TP_i)
        results[lang]["FN"].append(FN_i)
        results[lang]["FP"].append(FP_i)
