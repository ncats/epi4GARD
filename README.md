# epi4GARD

This project is an in-progress alert system for the [NIH Genetic and Rare Diseases Information Center (GARD)](https://rarediseases.info.nih.gov/). The purpose of this system is to automatically identify rare disease articles as epidemiological, and if so, extract the epidemiological information from them. It is currently composed of two components:
 - A long short-term memory recurrent neural network that classifies rare disease publications as epidemiological or not (created by Jennifer John, [see more](https://knowledge.amia.org/73035-amia-1.4612663/t002-1.4614253/t002-1.4614254/3475589-1.4614363/3475589-1.4614364?qr=1))
 - A bidirectional transformer-based model that performs named entity recognition to identify epidemiological information from abstracts classified as epidemiological. (created by William Kariampuzha)

This external folder contains the LSTM RNN classifier. The [EpiExtract4GARD folder](https://github.com/ncats/epi4GARD/tree/master/EpiExtract4GARD#epiextract4gard) contains the entirety of the transformer-based NER model.
