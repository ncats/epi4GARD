# EpiExtract4GARD
This notebook contains the code for a pipeline that can extract epidemiological information from rare disease literature. The pipeline includes disease identification via dictionary look-up and identification of locations, epidemiological identifiers (e.g. "prevalence", "annual incidence", "estimated occurrence") and epidemiological rates (e.g. "1.7 per 1,000,000 live births", "2.1:1.000.000", "one in five million", "0.03%") via BioBERT fine-tuned for named entity recognition (multi-type token classification). 

The final model is freely available to use on [Hugging Face](https://huggingface.co/ncats/EpiExtract4GARD). To see how it integrates with the entire epi4GARD alert system [click here](https://github.com/ncats/epi4GARD#epi4gard). 

## Bi-Directional Transformer-based NER
### Key notebooks
- *gather_pubs_per_disease.ipynb*: Generates *whole_abstract_set.csv* and *positive_abstract_set.csv*. *whole_abstract_set.csv* is a dataset created by sampling 500 rare disease names and their synonyms from *GARD.csv* until &ge;50 abstracts had been returned or the search results were exhausted. Although ~25,000 abstracts were expected, 7699 unique abstracts were returned due to the limited research on rare diseases. After running each of these through the LSTM RNN classifier, the *positive_abstract_set.csv* was created from the abstracts which had an epidemiological probability >50%. *positive_abstract_set.csv* will be passed to *create_labeled_dataset_V2.ipynb*
- *create_labeled_dataset_V2.ipynb*: Uses [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities) and rules I created iteratively to auto-label the dataset. Generates *epi_{train,val,test}_setV2.tsv* files
- *modify_existing_labels.ipynb*: Generated new rules for labeling that improved the V2 set to create V3.2 set. Input: *epi_{train, val, test}_setV2.tsv* files Output: *epi_{train, val, test}_setV3.tsv* files
- *compile_datasets.ipynb*: Combines the [CoNLL++ dataset](https://github.com/huggingface/datasets/tree/master/datasets/conllpp) with *epi_train_setV3.tsv* to generate *training_setV3.tsv*. Also contains code to combine other datasets and train the models in notebook (did not work effectively) which were ultimately not used in this study. 
- *Case Study.ipynb*: Demonstrates the ability of the pipeline to search by disease or GARD ID and extract epidemiological information by utilizing *extract_abs.py* and *classify_abs.py*. Generates *proof_of_concept* folder. 
- *Find efficacy of test predictions.ipynb*: Contains the code to compare two datasets at the token- and entity-levels. Used to compare the unmodified test set to the manually validated test set (finds efficacy of programmatic labeling) and the model's test predictions to the manually validated test set (finds out precision, recall, and F1 score for each entity class).
- *Orphanet_Comparison_Final.ipynb*: Compares the output of my model to Orphanet's data (*en_product9_prev.xml*) _en masse_. Generates *Orphanet-Comparison-FINAL.csv*. Due to differences in how data is curated, a quantitative analysis is not possible. A qualitative analysis is found here. **UPDATE ONCE UPLOADING**

### Key Python Files
- *extract_abs.py*: This is the workhorse of the pipeline. It can be imported into another notebook, python file, or run from the command line. Some key functions:
  - _load_GARD_diseases()_: Outputs *GARD_dict*, a dictionary of form {disease name/synonym : GARD ID} and *max_length*, int that has the length of the longest series of words that comprises a rare disease name/synonym. Utilizes *gard-id-name-synonyms.json* to get this information.
  - _init_NER_pipeline()_: Outputs a [huggingface tranformers pipeline variable](https://huggingface.co/transformers/main_classes/pipelines.html)
  - _get_diseases(sentence, GARD_dict, max_length)_: Checks every possible string combination in a sentence against the *GARD_dict* to find matches. Runs with time complexity O(*n*)
  - _autosearch(searchterm, GARD_dict)_: Allows searching by GARD ID (i.e. lemma) or any form of a disease name. Matches form to lemma and outputs a list of all other disease forms (synonyms)
  - _search_term_extraction(search_term, model_variables)_: Input search term and the variables of both the transformer and LSTM RNN models and outputs a list of all related epidemiological abstracts with relevant information extracted from the text. See *Case Study.ipynb* for demo.
- *classify_abs.py*: Optimized and added to [the original](https://github.com/ncats/epi4GARD#python-files). Some key functions:
  - _init_classify_model()_: Returns all of the variables needed for the LSTM RNN classification model. 
  - _search_getAbs(searchterm_list, maxResults)_: Allows one to search EBI and PubMed APIs for a search term(type = str) or a list of search terms. Returns a dictionary of {PMID : title+abstract}.

### Data files
- *GARD.csv*: Contains the names and synonyms of all GARD diseases generated from a [neo4j knowledge graph](https://pubmed.ncbi.nlm.nih.gov/33183351/). *NOTE*: Contains errors due to the substitution of semicolons for commas to separate synonym names. Was utilized in *gather_pubs_per_disease.ipynb* and originally in *extract_abs.py* for disease identification, but that function is deprecated. Utilize *gard-id-name-synonyms.json* in future.
- *positive_abstract_set.csv*: Contains 620 unique abstracts (755 total) that were classified as epidemiological from the *whole_abstract_set.csv*
- *epi_{train,val,test}_setV3.tsv* files: These are the V3 training, validation, and programmatically generated test sets. The training set was copied to ```datasets/EpiCustomV3``` and renamed *train.tsv*. The validation set was copied to ```datasets/EpiCustomV3``` and ```datasets/Large_DatasetV3``` and renamed *val.tsv*. The V3 test set (uncorrected) is important as it is used by *Find efficacy of test predictions.ipynb* to find the efficacy of the programmatic labeling, but was otherwise not used with the model. 
- *epi_test_setV2-corrected.tsv*: The manually validated test set (i.e. text with ground truth labels). Validation was completed by all four authors, which includes a rare disease expert, after V2 so there is no *epi_test_setV3-corrected*. This set was copied to ```datasets/EpiCustomV3``` and ```datasets/Large_DatasetV3``` and renamed *test.tsv*
- *training_setV3.tsv*: Generated from *compile_datasets.ipynb*. It was copied to ```datasets/Large_DatasetV3``` and renamed *train.tsv*
- *en_product9_prev.xml*: Contains the [Orphanet Data](http://www.orphadata.org/cgi-bin/epidemio.html) for the Case Study Comparison. This document was downloaded on August 31, 2021. ValMoy is epidemiologic rate per 100,000 persons. See [Orphanet's documentation](https://www.orpha.net/orphacom/cahiers/docs/GB/Epidemiology_in_Orphanet_R1_Ann_Epi_EP_05.pdf).
  - Use ```curl "http://www.orphadata.org/data/xml/en_product9_prev.xml" -o en_product9_prev.xml``` to download the file 
- *gard-id-name-synonyms.json*: Contains the names and synonyms of all GARD diseases generated from a [neo4j knowledge graph](https://pubmed.ncbi.nlm.nih.gov/33183351/). Utilized in *extract_abs.py* for disease identification. 
- *Orphanet-Comparison-FINAL.csv*: Contains the output of a large scale comparison to the Orphanet rare disease epidemiology database.
### Key Folders
- *datasets*: Contains EpiCustomV2, Large_DatasetV2, EpiCustomV3, Large_DatasetV3 folders of data for training the model. 
- *NER*: Contains the code to fine-tune BioBERT for NER. See internal README for details. 
- *proof_of_concept*: Contains the output of *Case Study.ipynb*
### Other
- *whole_abstract_set.csv*: Contains 7699 unique abstracts (9284 total) that were returned from the EBI API call.
- *Analyze_dz_num_sample.ipynb*: Analyzes the distribution of epidemiological articles returned from *gather_pubs_per_disease.ipynb*. Finds that 32.6  percent of diseases have 0 epidemiological studies and 96.6 percent of diseases have less than 5 epidemiological studies in this study. Generates *DiseaseSampleEpi_HistFINAL.png*. Previously generated *ArticlesPerDisease_Hist.2.png* which is the distribution of all articles returned from the search (many rare diseases have fewer than 50 articles returned when querying EBI API).
- *Epi4GARD_test_set[CORRECTED].xlsx*: We input the programmatically labeled test set into a shared Google Sheet and validated it by manually correcting labels. The downloaded this Excel sheet for re-input into *.tsv* file
- *xlsx2tsv.ipynb*: Converted *Epi4GARD_test_set[CORRECTED].xlsx* into *epi_test_setV2-corrected.tsv* because Excel cannot do it in the correct format.
- *API-Analysis.ipynb*: Analyzes the EBI RESTfuland NCBI (PubMed) APIs using the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index). Also re-outputs an unmodified, but easier to manuallly search Orphanet data in csv. Generates *API-Analysis* folder and *All Orphanet Data.csv*. 
- *API-Analysis* Folder: Contains the output of *API-Analysis.ipynb* which is just a set of \*.csv files named by GARD disease that contain the PMIDs returned from each API for each _autosearch()_.
- *All Orphanet Data.csv*: Same as *en_product9_prev.xml*, but easier to search and see comparisons.
- *UnusedCode*: Contains my practice files, code, and options for training the model that were abandoned.
