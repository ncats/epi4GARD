# EpiExtract4GARD
# DOCUMENTATION PENDING
This notebook contains the code for a pipeline that can extract epidemiological information from rare disease literature. The pipeline includes disease identification via dictionary look-up and identification of locations, epidemiological identifiers (e.g. "prevalence", "annual incidence", "estimated occurrence") and epidemiological rates (e.g. "1.7 per 1,000,000 live births" or 2.1:34,492) via BioBERT fine-tuned for named entity recognition (multi-type token classification).

To see how it integrates with the entire epi4GARD alert system 

## Bi-Directional Transformer-based NER
### Key notebooks
- *gather_pubs_per_disease.ipynb*: Generates *whole_abstract_set* and *positive_abstract_set.csv*. *whole_abstract_set* is a dataset created by sampling 500 rare disease names and their synonyms from *GARD.csv* until &ge;50 abstracts had been returned or the search results were exhausted. Although ~25,000 abstracts were expected, 7699 unique abstracts were returned due to the limited research on rare diseases. After running each of these through the LSTM RNN classifier, the *positive_abstract_set.csv* was created from the abstracts which had an epidemiological probability >50%. *positive_abstract_set.csv* will be passed to *create_labeled_dataset_V2.ipynb*
- *create_labeled_dataset_V2.ipynb*: Uses [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities) and rules I created iteratively to auto-label the dataset. Generates *epi_{train,val,test}_setV2.tsv* files
- *modify_existing_labels.ipynb*: Generated new rules for labeling that improved the V2 set to create V3.2 set. Input: *epi_{train, val, test}_setV2.tsv* files Output: *epi_{train, val, test}_setV3.tsv* files
- *compile_datasets.ipynb*: Combines the [CoNLL++ dataset](https://github.com/huggingface/datasets/tree/master/datasets/conllpp) with *epi_train_setV3.tsv* to generate *training_setV3.tsv*. Also contains code to combine other datasets and train the models in notebook (did not work effectively) which were ultimately not used in this study.
-*Case Study.ipynb*:
- *Find efficacy of test predictions.ipynb*: 

Orphanet_Comparison_Final.ipynb


### Data files
- *whole_abstract_set.csv*: Contains 7699 unique abstracts (9284 total) that were returned from the EBI API call.
- *positive_abstract_set.csv*: Contains 620 unique abstracts (755 total) that were classified as epidemiological from the *whole_abstract_set.csv*
- *train.tsv*: Contains 620 unique abstracts (755 total) that were classified as epidemiological from the *whole_abstract_set.csv*
- *en_product9_prev.xml*: Contains the [Orphanet Data](http://www.orphadata.org/cgi-bin/epidemio.html) for the Case Study Comparison. This document was downloaded on August 31, 2021. 
- Use ```curl "http://www.orphadata.org/data/xml/en_product9_prev.xml" -o en_product9_prev.xml``` to download the file in the correct repository
### Folders
- *datasets*: Contains EpiCustomV2 dataset and Large_DatasetV2 which was the combined 
- *NER*: Contains the model and its training
- *UnusedCode*: Contains many of the my practice files, code, and options for training the model that were abandoned.

### Other
- *Epidemiology extraction general.ipynb*: Contains preliminary code for rule-based information extraction from the abstracts of epidemiology studies. Includes functions to identify location, statistics, and prevalence type. Has not been rigorously tested and more development is definitely needed, but could be useful as a starting point.
- *Rule-based approach.ipynb*: Old code for a rule-based approach to epidemiology classification. Relies on keyword and phrase matching.
- *ML sentence classification.ipynb*: Similar process to *ML document classification.ipynb*, but predictions are generated on sentences. Training and validation sets have been deleted and must be re-created.
