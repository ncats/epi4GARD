# EpiExtract4GARD
This notebook contains the code to train

This is the bidirectional NER system (multi-type token classification)
## Bi-Directional Transformer-based NER
### Key notebooks
- *gather_pubs_per_disease.ipynb*: Generates *whole_abstract_set* and *positive_abstract_set.csv*. *whole_abstract_set* is a dataset created by sampling 500 rare disease names and their synonyms from *GARD.csv*  until at least 50 abstracts had been returned or the search results were exhausted. Although ~25,000 abstracts were expected 9284 FIX!!!! unique abstracts were returned due to the limited research on rare diseases. After running each of these through the LSTM RNN classifier, the *positive_abstract_set.csv* was created from the abstracts which had an epidemiological probability >50%. *positive_abstract_set.csv* will be passed to *create_labeled_dataset_V2.ipynb*
- *create_labeled_dataset_V2.ipynb*: Generates 

### Python files
- *classify_abs.py*: Script to apply the model on one article given its PubMed ID. Can be run at the command line, or imported into a notebook (see Alert system proof of concept.ipynb for an example).

### Data files
- *en_product9_prev.xml*: Orphanet epidemiology file, downloaded directly from their website in July 2020. For each rare disease, it includes a prevalence estimate, type, and source (most often PubMed ID).
- *negative_dataset.csv*: Negative dataset assembled by Prepare negative dataset.ipynb. Columns: PubMed ID, abstract text. 25,015 rows.
- *orphanet_epi_mesh.csv*: Positive dataset assembled by Prepare positive dataset.ipynb. Columns: PubMed ID, abstract text. 1,145 rows.
- *records.json*: Results of a GARD Neo4j query for all indexed rare diseases. Includes GARD ID and GARD name.
- *train_abstracts.npy*, *validation_abstracts.npy*: Original abstracts from training and validation sets. (All such .npy files have the same consistent index.)
- *train_padded.npy, validation_padded.npy*: Preprocessed abstracts from training and validation sets
- *training_label_seq.npy, validation_label_seq.npy*: Sequential labels for the training and validation sets.
- *train_pmids.npy, validation_pmids.npy*: PubMed IDs for the training and validation sets.
- *curator_labeled_dataset.xlsx*: Evaluation set labeled by a GARD curator. Columns: Label (from a previous model, no longer relevant), PubMed ID, article title, Abs (abstract text), PMID_Links (link to PubMed article), Validation (curator label), Unnamed (curator comments).
- *proof_of_concept/disease_name.csv*: Results of Alert system proof of concept.ipynb notebook, saved to spreadsheets for each rare disease. Columns: Pubmed ID, abstract text, predicted probability of epidemiology, binary epidemiology prediction (thresholded at 0.5).
- *curator_results_comparison_orphanet.csv*: Comparison of the curator labels and the classifier on the evaluation set. Columns: PubMed ID, article title, abstract text, curator label, curator notes, epidemiology prediction, epidemiology probability.

### Models
- *tokenizer.pickle*: Pickle file with weights and parameters for tokenizer (from tensorflow.keras.preprocessing.text), fit on the datasets. Should be reloaded and applied to texts before generating new predictions.
- *saved_model/model_name*: Folders with model parameters and weights, which can be reloaded in the tensorflow.keras framework. *my_model_orphanet_final* is the folder which contains the most up-to-date model. All other model folders are from previous iterations and can be ignored.

### Other
- *Epidemiology extraction general.ipynb*: Contains preliminary code for rule-based information extraction from the abstracts of epidemiology studies. Includes functions to identify location, statistics, and prevalence type. Has not been rigorously tested and more development is definitely needed, but could be useful as a starting point.
- *Rule-based approach.ipynb*: Old code for a rule-based approach to epidemiology classification. Relies on keyword and phrase matching.
- *ML sentence classification.ipynb*: Similar process to *ML document classification.ipynb*, but predictions are generated on sentences. Training and validation sets have been deleted and must be re-created.
