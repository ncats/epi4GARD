# epi4GARD

This project is an alert system for the [NIH Genetic and Rare Diseases Information Center (GARD)](https://rarediseases.info.nih.gov/). The purpose of this system is to automatically identify rare disease articles as epidemiological, and if so, extract the epidemiological information from them. It is currently composed of two components:
 - A long short-term memory recurrent neural network that classifies rare disease publications as epidemiological or not (created by Jennifer John, [see more](https://knowledge.amia.org/73035-amia-1.4612663/t002-1.4614253/t002-1.4614254/3475589-1.4614363/3475589-1.4614364?qr=1))
 - A bidirectional transformer-based model that performs named entity recognition to identify epidemiological information from abstracts classified as epidemiological. (created by William Kariampuzha)

### Key notebooks
- *Prepare positive dataset.ipynb*: Generates *orphanet_epi_mesh.csv*, the final positive dataset (articles that are all epidemiology studies). First, PubMed IDs are extracted from a collection of epidemiology sources provided by Orphanet. The final positive set consists of the PubMed IDs that have epidemiology, incidence, or prevalence MeSH terms. The notebook includes code to optionally expand the dataset by including articles with epidemiology-related MeSH terms beyond those included in the Orphanet file, although this was shown to have worse performance.
- *Prepare negative dataset.ipynb*: Generates *negative_dataset.csv*, the final negative dataset (articles that are not epidemiology studies). Using the EBI API, the top 5 PubMed search results for each of the 6,000+ rare diseases included in the GARD database are retrieved. Articles that have epidemiology MeSH terms or keywords in the abstract or that are also in the Orphanet file are removed.
- *ML document classification.ipynb*: Trains recurrent neural network (RNN) with the training data. The positive and negative datasets are combined, shuffled, and preprocessed. The final training and validation sets are saved to .npy files for future reference. The model is defined and trained (with weights saved to a file), and preliminary evaluation results are given.
- *Model evaluation.ipynb*: Provides an evaluation of the model on the holdout test set. Evaluation includes precision, recall, F1 score, AUC, ROC curve, and example results.
- *Curator set evaluation.ipynb*: Evaluates the model on the evaluation set, consisting of 100 articles labeled by a GARD curator. 
- *Alert system proof of concept.ipynb*: Demonstration of the model results in practice. For 5 rare diseases, the top PubMed results are returned. The model predicts on these abstracts to generate rankings of the probability of epidemiology by article.

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
