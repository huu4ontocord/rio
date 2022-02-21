## What is this
- This is a library for accessing a large multilingual ontology.
- What is an ontology? It is simply a dictionary that ties into a hiearchy (a PUBLIC_FIGURE is a type of PERSON).
- The dictionary is based on Conceptnet5 and Yago entities. We have also created constraints on various entities, and hand crafted rules for conflicts between various types. 
- See ontology_builder_data.py and ontology_builder.py for details on the constraints.

## How to rebuild the ontology
```
python ontology_builder.py -c
```

## How to use the API
```
from ontology_manager import OntologyManager
onto = OntologyManager()
onto.tokenize("George Washington was the 1st president of the United States")
```
