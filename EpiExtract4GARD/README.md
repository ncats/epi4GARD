# EpiExtract4GARD
# DOCUMENTATION PENDING
This notebook contains the code for a pipeline that can extract epidemiological information from rare disease literature. The pipeline includes disease identification via dictionary look-up and identification of locations, epidemiological identifiers (e.g. "prevalence", "annual incidence", "estimated occurrence") and epidemiological rates (e.g. "1.7 per 1,000,000 live births" or 2.1:34,492) via BioBERT fine-tuned for named entity recognition (multi-type token classification). 

To see how it integrates with the entire epi4GARD alert system [click here](https://github.com/ncats/epi4GARD#epi4gard).

## Bi-Directional Transformer-based NER
### Key notebooks
- *gather_pubs_per_disease.ipynb*: Generates *whole_abstract_set* and *positive_abstract_set.csv*. *whole_abstract_set* is a dataset created by sampling 500 rare disease names and their synonyms from *GARD.csv* until &ge;50 abstracts had been returned or the search results were exhausted. Although ~25,000 abstracts were expected, 7699 unique abstracts were returned due to the limited research on rare diseases. After running each of these through the LSTM RNN classifier, the *positive_abstract_set.csv* was created from the abstracts which had an epidemiological probability >50%. *positive_abstract_set.csv* will be passed to *create_labeled_dataset_V2.ipynb*
- *create_labeled_dataset_V2.ipynb*: Uses [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities) and rules I created iteratively to auto-label the dataset. Generates *epi_{train,val,test}_setV2.tsv* files
- *modify_existing_labels.ipynb*: Generated new rules for labeling that improved the V2 set to create V3.2 set. Input: *epi_{train, val, test}_setV2.tsv* files Output: *epi_{train, val, test}_setV3.tsv* files
- *compile_datasets.ipynb*: Combines the [CoNLL++ dataset](https://github.com/huggingface/datasets/tree/master/datasets/conllpp) with *epi_train_setV3.tsv* to generate *training_setV3.tsv*. Also contains code to combine other datasets and train the models in notebook (did not work effectively) which were ultimately not used in this study.
-*Case Study.ipynb*:
- *Find efficacy of test predictions.ipynb*: Contains the code to compare two datasets at the token- and entity-levels. Use to compare the unmodified 
- *Orphanet_Comparison_Final.ipynb*:

### Key Python Files
- *extract_abs.py*:

- *classify_abs.py*: 

### Data files
- *GARD.csv*: Contains the names and synonyms of all GARD diseases generated from a [neo4j knowledge graph](https://pubmed.ncbi.nlm.nih.gov/33183351/). *NOTE*: Contains errors due to the substitution of semicolons for commas to separate synonym names. Was utilized in *gather_pubs_per_disease.ipynb* and originally in *extract_abs.py* for disease identification, but that function is deprecated. Utilize *gard-id-name-synonyms.json* in future.
- *whole_abstract_set.csv*: Contains 7699 unique abstracts (9284 total) that were returned from the EBI API call.
- *positive_abstract_set.csv*: Contains 620 unique abstracts (755 total) that were classified as epidemiological from the *whole_abstract_set.csv*
- *epi_{train,val,test}_setV2.tsv* files: 
- *epi_test_setV2-corrected.tsv*: The corrected dataset
- *en_product9_prev.xml*: Contains the [Orphanet Data](http://www.orphadata.org/cgi-bin/epidemio.html) for the Case Study Comparison. This document was downloaded on August 31, 2021. Use ```curl "http://www.orphadata.org/data/xml/en_product9_prev.xml" -o en_product9_prev.xml``` to download the file 
- *gard-id-name-synonyms.json*: Contains the names and synonyms of all GARD diseases generated from a [neo4j knowledge graph](https://pubmed.ncbi.nlm.nih.gov/33183351/). Utilized in *extract_abs.py* for disease identification. 
- *Orphanet-Comparison-FINAL.csv*: Contains the output of a large scale comparison to the Orphanet rare disease epidemiology database.
### Folders
- *datasets*: Contains EpiCustomV2, Large_DatasetV2, EpiCustomV3, Large_DatasetV3 datasets for training the model. 
- *NER*: Contains the code to fine-tune BioBERT for NER. See internal README for details. 
- *UnusedCode*: Contains many of the my practice files, code, and options for training the model that were abandoned.
### Other
- *Analyze_dz_num_sample.ipynb*: Analyzes the distribution of epidemiological articles returned from *gather_pubs_per_disease.ipynb*. Finds that 32.6  percent of diseases have 0 epidemiological studies and 96.6 percent of diseases have less than 5 epidemiological studies in this study. Generates *DiseaseSampleEpi_HistFINAL.png*. Previously generated *ArticlesPerDisease_Hist.2.png* which is the distribution of all articles returned from the search (many rare diseases have fewer than 50 articles returned when querying EBI API).
