import argparse
import requests
import xml.etree.ElementTree as ET
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
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

# Prepare model
#nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer= init_classify_model()
def init_classify_model(model='my_model_orphanet_final'):
    #Load spaCy models
    nlp = spacy.load('en_core_web_lg')
    nlpSci = spacy.load("en_ner_bc5cdr_md")
    nlpSci2 = spacy.load('en_ner_bionlp13cg_md')
    
    # load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        classify_tokenizer = pickle.load(handle)
    
    # load the model
    classify_model = tf.keras.models.load_model(model) 
    
    return nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer

#Gets abstract and title (concatenated) from EBI API
def PMID_getAb(PMID): 
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:'+str(PMID)+'&resulttype=core'
    r = requests.get(url)
    root = ET.fromstring(r.content)
    titles = [title.text for title in root.iter('title')]
    abstracts = [abstract.text for abstract in root.iter('abstractText')]
    if len(abstracts) > 0 and len(abstracts[0])>5:
        return titles[0]+' '+abstracts[0]
    else:
        return ''

## DEPRECATED, use search_getAbs for more comprehensive results
def search_Pubmed_API(searchterm, maxResults): #returns a dictionary of {pmids:abstracts} 
    # get results from searching for disease name through PubMed API
    term = ''
    dz_words = searchterm.split()
    for word in dz_words:
        term += word + '%20'
    query = term[:-3]
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term='+query
    r = requests.get(url)
    root = ET.fromstring(r.content)

    pmids = []
    i = 0

    # loop over resulting articles
    for result in root.iter('IdList'):
        pmids = [pmid.text for pmid in result.iter('Id')]
        if i >= maxResults:
            break
        pmid_to_abs = {}
        for pmid in pmids:
            abstract = PMID_getAb(pmid)
            if len(abstract)>5:
                pmid_to_abs[pmid]=abstract
                i+=1
    return pmid_to_abs

## DEPRECATED, use search_getAbs for more comprehensive results
# get results from searching for disease name through EBI API
def search_EBI_API(search_term,maxResults): #returns a dictionary of {pmids:abstracts}    
    term = ''
    dz_words = search_term.split()
    for word in dz_words:
        term += word + '%20'
    query = term[:-3]
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query='+query+'&resulttype=core'
    r = requests.get(url)
    root = ET.fromstring(r.content)

    pmids_abs = {}
    i = 0

    # loop over resulting articles
    for result in root.iter('result'):
        if i >= maxResults:
            break
        pmids = [pmid.text for pmid in result.iter('id')]
        if len(pmids) > 0:
            pmid = pmids[0]
            if pmid[0].isdigit():
                abstracts = [abstract.text for abstract in result.iter('abstractText')]
                titles = [title.text for title in result.iter('title')]
                if len(abstracts) > 0:# and len(abstracts[0])>5:
                    pmids_abs[pmid] = titles[0]+' '+abstracts[0]
                    i+=1
     
    return pmids_abs

## This is the main, most comprehensive search_term function, it can take in a search term or a list of search terms and output a dictionary of {pmids:abstracts}
## Gets results from searching through both PubMed and EBI search term APIs, also makes use of the EBI API for PMIDs. 
## EBI API and PubMed API give different results
# This makes n+2 API calls where n<=maxResults, which is slow 
# There is a way to optimize by gathering abstracts from the EBI API when also getting pmids but did not pursue due to time constraints
def search_getAbs(searchterm_list, maxResults):
    #Keep track of total results
    i = 0
    
    #set of all pmids
    pmids = set()
    
    #dictionary {pmid:abstract}
    pmid_abs = {}
    
    #type validation, allows string or list input
    if type(searchterm_list)!=list:
        if type(searchterm_list)==str:
            searchterm_list = [searchterm_list]
        else:
            searchterm_list = list(searchterm_list)
    
    #gathers pmids into a set first
    for dz in searchterm_list:
        term = ''
        dz_words = dz.split()
        for word in dz_words:
            term += word + '%20'
        query = term[:-3]

        ## get pmid results from searching for disease name through PubMed API
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term='+query
        r = requests.get(url)
        root = ET.fromstring(r.content)

        # loop over resulting articles
        for result in root.iter('IdList'):
            if i >= maxResults:
                break
            pmidlist = [pmid.text for pmid in result.iter('Id')]
            pmids.update(pmidlist)
            i+=len(pmidlist)

        ## get results from searching for disease name through EBI API
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query='+query+'&resulttype=core'
        r = requests.get(url)
        root = ET.fromstring(r.content)

        # loop over resulting articles
        for result in root.iter('result'):
            if i >= maxResults:
                break
            pmidlist = [pmid.text for pmid in result.iter('id')]
            #can also gather abstract and title here but for some reason did not work as intended the first time
            if len(pmidlist) > 0:
                pmid = pmidlist[0]
                if pmid[0].isdigit():
                    pmids.add(pmid)
                    i += 1

    ## get abstracts from EBI PMID API and output a dictionary
    for pmid in pmids:
        abstract = PMID_getAb(pmid)
        if len(abstract)>5:
            pmid_abs[pmid] = abstract
    
    return pmid_abs

# Generate predictions for a PubMed Id
# nlp: en_core_web_lg
# nlpSci: en_ner_bc5cdr_md
# nlpSci2: en_ner_bionlp13cg_md
# Defaults to load my_model_orphanet_final, the most up-to-date version of the classification model,
# but can also be run on any other tf.keras model
#This was originally getPredictions
def getPMIDPredictions(pmid, nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer):
    
    abstract = PMID_getAb(pmid)
    
    # remove stopwords
    for word in STOPWORDS:
        token = ' ' + word + ' '
        abstract = abstract.replace(token, ' ')
        abstract = abstract.replace(' ', ' ')
        
    # preprocess abstract
    abstract_standard = [standardizeAbstract(standardizeSciTerms(abstract, nlpSci, nlpSci2), nlp)]
    sequence = classify_tokenizer.texts_to_sequences(abstract_standard)
    padded = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    y_pred1 = classify_model.predict(padded) # generate prediction
    y_pred = np.argmax(y_pred1, axis=1) # get binary prediction
    
    prob = y_pred1[0][1]
    if y_pred == 1:
        isEpi = True
    else:
        isEpi = False

    return abstract, prob, isEpi


def getTextPredictions(abstract, nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer):
    
    # remove stopwords
    for word in STOPWORDS:
        token = ' ' + word + ' '
        abstract = abstract.replace(token, ' ')
        abstract = abstract.replace(' ', ' ')
        
    # preprocess abstract
    abstract_standard = [standardizeAbstract(standardizeSciTerms(abstract, nlpSci, nlpSci2), nlp)]
    sequence = classify_tokenizer.texts_to_sequences(abstract_standard)
    padded = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    y_pred1 = classify_model.predict(padded) # generate prediction
    y_pred = np.argmax(y_pred1, axis=1) # get binary prediction
    
    prob = y_pred1[0][1]
    if y_pred == 1:
        isEpi = True
    else:
        isEpi = False

    return prob, isEpi

if __name__ == '__main__':
    print('Loading 5 NLP models...')
    nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer= init_classify_model()
    print('All models loaded.')
    pmid = input('\nEnter PubMed PMID (or DONE): ')
    while pmid != 'DONE':
        abstract, prob, isEpi = getPredictions(pmid, nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer)
        print(abstract, prob, isEpi)
        pmid = input('\nEnter PubMed PMID (or DONE): ')