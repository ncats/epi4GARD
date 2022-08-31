![LOGO]()
# epi4GARD
[![ORGANIZATION](https://img.shields.io/badge/NIH-2F5486)](https://nih.gov/)
[![ORGANIZATION](https://img.shields.io/badge/NCATS-5F3168)](https://ncats.nih.gov/)
[![LICENSE](https://img.shields.io/badge/license-National%20Center%20for%20Advancing%20Translational%20Sciences-lightgrey?style=black)](LICENSE)
[![CONTACT](https://img.shields.io/badge/contact-William.Kariampuzha%40nih.gov-blue)](mailto:William.Kariampuzha@nih.gov)

The aim of this project for the [NIH Genetic and Rare Diseases Information Center (GARD)](https://rarediseases.info.nih.gov/) is to automatically identify rare disease articles as epidemiological, and if so, extract the epidemiological information from them. It is currently composed of two components:
 - A long short-term memory recurrent neural network that classifies rare disease publications as epidemiological or not (created by Jennifer John, [see more](https://knowledge.amia.org/73035-amia-1.4612663/t002-1.4614253/t002-1.4614254/3475589-1.4614363/3475589-1.4614364?qr=1))
 - A bidirectional transformer-based model that performs named entity recognition to identify epidemiological information from abstracts classified as epidemiological. (created by William Kariampuzha)

The [EpiExtract4GARD folder](/EpiExtract4GARD#epiextract4gard) contains the entirety of the transformer-based NER model.
