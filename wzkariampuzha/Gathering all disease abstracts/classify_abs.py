import argparse
import requests
import xml.etree.ElementTree as ET
import pickle
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from nltk.corpus import stopwords
import spacy
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
STOPWORDS = set(stopwords.words('english'))

max_length = 300
trunc_type = 'post'
padding_type = 'post'

# Standardize the abstract by replacing all named entities with their entity label.
# Eg. 3 patients reported at a clinic in England --> CARDINAL patients reported at a clinic in GPE
# expects the spaCy model en_core_web_lg as input
def standardizeAbstract(abstract, nlp):
    doc = nlp(abstract)
    newAbstract = abstract
    for e in reversed(doc.ents):
        if e.label_ in {'PERCENT','CARDINAL','GPE','LOC','DATE','TIME','QUANTITY','ORDINAL'}:
            start = e.start_char
            end = start + len(e.text)
            newAbstract = newAbstract[:start] + e.label_ + newAbstract[end:]
    return newAbstract

# Same as above but replaces biomedical named entities from scispaCy models
# Expects as input en_ner_bc5cdr_md and en_ner_bionlp13cg_md
def standardizeSciTerms(abstract, nlpSci, nlpSci2):
    doc = nlpSci(abstract)
    newAbstract = abstract
    for e in reversed(doc.ents):
        start = e.start_char
        end = start + len(e.text)
        newAbstract = newAbstract[:start] + e.label_ + newAbstract[end:]
        
    doc = nlpSci2(newAbstract)
    for e in reversed(doc.ents):
        start = e.start_char
        end = start + len(e.text)
        newAbstract = newAbstract[:start] + e.label_ + newAbstract[end:]
    return newAbstract

# Generate predictions for a PubMed Id
# nlp: en_core_web_lg
# nlpSci: en_ner_bc5cdr_md
# nlpSci2: en_ner_bionlp13cg_md
# Defaults to load my_model_orphanet_final, the most up-to-date version of the classification model,
# but can also be run on any other tf.keras model


def get_abstract(pmid):
# retrieve abstract from EBI API
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:'+str(pmid)+'&resulttype=core'
    r = requests.get(url)
    root = ET.fromstring(r.content)
    
    abstract = ''
    for child in root.iter('*'):
        if child.tag == 'abstractText':
            abstract = child.text
    return abstract


def getPredictions(pmid, nlp, nlpSci, nlpSci2, model='my_model_orphanet_final'):
    abstract = get_abstract(pmid)
    
    # load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        new_tokenizer = pickle.load(handle)
    
    new_model = tf.keras.models.load_model('saved_model/'+model) # load the model
    
    # remove stopwords
    for word in STOPWORDS:
        token = ' ' + word + ' '
        abstract = abstract.replace(token, ' ')
        abstract = abstract.replace(' ', ' ')
        
    # preprocess abstract
    abstract_standard = [standardizeAbstract(standardizeSciTerms(abstract, nlpSci, nlpSci2), nlp)]
    sequence = new_tokenizer.texts_to_sequences(abstract_standard)
    padded = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    y_pred1 = new_model.predict(padded) # generate prediction
    y_pred = np.argmax(y_pred1, axis=1) # get binary prediction
    
    prob = y_pred1[0][1]
    if y_pred == 1:
        isEpi = True
    else:
        isEpi = False

    return prob, isEpi
   

def getAbstractPredictions(abstract, nlp, nlpSci, nlpSci2, model='my_model_orphanet_final'):
    
    if(len(abstract)<5):
        abstract = get_abstract(pmid)
        
    # load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        new_tokenizer = pickle.load(handle)
    
    new_model = tf.keras.models.load_model('saved_model/'+model) # load the model
    
    # remove stopwords
    for word in STOPWORDS:
        token = ' ' + word + ' '
        abstract = abstract.replace(token, ' ')
        abstract = abstract.replace(' ', ' ')
        
    # preprocess abstract
    abstract_standard = [standardizeAbstract(standardizeSciTerms(abstract, nlpSci, nlpSci2), nlp)]
    sequence = new_tokenizer.texts_to_sequences(abstract_standard)
    padded = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    y_pred1 = new_model.predict(padded) # generate prediction
    y_pred = np.argmax(y_pred1, axis=1) # get binary prediction
    
    prob = y_pred1[0][1]
    if y_pred == 1:
        isEpi = True
    else:
        isEpi = False

    return prob, isEpi

if __name__ == '__main__':
    print('Loading 3 NLP models...')
    nlp = spacy.load('en_core_web_lg')
    print('Core model loaded.')
    nlpSci = spacy.load("en_ner_bc5cdr_md")
    print('Disease and chemical model loaded.')
    nlpSci2 = spacy.load('en_ner_bionlp13cg_md')
    print('All models loaded.')
    pmid = input('\nEnter PubMed PMID (or DONE): ')
    while pmid != 'DONE':
        prob, isEpi = getPredictions(pmid, nlp, nlpSci, nlpSci2)
        print(prob, isEpi)
        pmid = input('\nEnter PubMed PMID (or DONE): ')
