## What is this
- This is a library for accessing a large multilingual ontology.
- What is an ontology? It is simply a dictionary that ties into a hiearchy (a PUBLIC_FIGURE is a type of PERSON).
- The dictionary based on Conceptnet5 and Yago entities. We have also created constraints on various entities, and hand crafted rules for conflicts between various types. 
- See ontology_builder_data.py and ontology_builder.py for details on the constraints.

## How to rebuild the ontonolgy
```
python ontology_builder.py -c
```
