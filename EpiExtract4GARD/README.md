# EpiExtract4GARD
This notebook contains the code to train BioBERT for named entity recognition of 
This is based on the PyTorch implementation of 
DMIS Lab

This is the bidirectional NER system (multi-type token classification)
## Bi-Directional Transformer-based NER
### Key notebooks
- *gather_pubs_per_disease.ipynb*: Generates *whole_abstract_set* and *positive_abstract_set.csv*. *whole_abstract_set* is a dataset created by sampling 500 rare disease names and their synonyms from *GARD.csv* until &ge;50 abstracts had been returned or the search results were exhausted. Although ~25,000 abstracts were expected, 7699 unique abstracts were returned due to the limited research on rare diseases. After running each of these through the LSTM RNN classifier, the *positive_abstract_set.csv* was created from the abstracts which had an epidemiological probability >50%. *positive_abstract_set.csv* will be passed to *create_labeled_dataset_V2.ipynb*
- *create_labeled_dataset_V2.ipynb*: Generates 
- *compile_datasets.ipynb*: Combines the [CoNLL++ dataset](https://github.com/huggingface/datasets/tree/master/datasets/conllpp) with the EpiCustom
- *Analyze_dz_num_sample.ipynb*: Combines

### Data files
- *whole_abstract_set.csv*: Contains 7699 unique abstracts (9284 total) that 
- *positive_abstract_set.csv*: Contains 620 unique abstracts (755 total) that were classified as epidemiological

### Folders
- *datasets*: contains the 
- *NER*: Contains the model and its training
- *UnusedCode*: Contains many of the 

### Other
- *Epidemiology extraction general.ipynb*: Contains preliminary code for rule-based information extraction from the abstracts of epidemiology studies. Includes functions to identify location, statistics, and prevalence type. Has not been rigorously tested and more development is definitely needed, but could be useful as a starting point.
- *Rule-based approach.ipynb*: Old code for a rule-based approach to epidemiology classification. Relies on keyword and phrase matching.
- *ML sentence classification.ipynb*: Similar process to *ML document classification.ipynb*, but predictions are generated on sentences. Training and validation sets have been deleted and must be re-created.
