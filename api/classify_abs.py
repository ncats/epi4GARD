import argparse
import requests
import xml.etree.ElementTree as ET
import pickle
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
STOPWORDS = set(stopwords.words('english'))
max_length = 300
trunc_type = 'post'
padding_type = 'post'

from typing import (
    Dict,
    List,
    Tuple,
    Set,
    Optional,
    Any,
    Union,
)

# Standardize the abstract by replacing all named entities with their entity label.
# Eg. 3 patients reported at a clinic in England --> CARDINAL patients reported at a clinic in GPE
# expects the spaCy model en_core_web_lg as input
def standardizeAbstract(abstract:str, nlp:Any) -> str:
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
def standardizeSciTerms(abstract:str, nlpSci:Any, nlpSci2:Any) -> str:
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
def init_classify_model(model:str='my_model_orphanet_final') -> Tuple[Any,Any,Any,Any,Any]:
    #Load spaCy models
    nlp = spacy.load('en_core_web_lg')
    nlpSci = spacy.load("en_ner_bc5cdr_md")
    nlpSci2 = spacy.load('en_ner_bionlp13cg_md')
    
    # load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        classify_tokenizer = pickle.load(handle)
    
    # load the model
    classify_model = tf.keras.models.load_model(model) 
    
    return (nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer)

#Gets abstract and title (concatenated) from EBI API
def PMID_getAb(PMID:Union[int,str]) -> str: 
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:'+str(PMID)+'&resulttype=core'
    r = requests.get(url)
    root = ET.fromstring(r.content)
    titles = [title.text for title in root.iter('title')]
    abstracts = [abstract.text for abstract in root.iter('abstractText')]
    if len(abstracts) > 0 and len(abstracts[0])>5:
        return titles[0]+' '+abstracts[0]
    else:
        return ''

def search_Pubmed_API(searchterm_list:Union[List[str],str], maxResults:int) -> Dict[str,str]: #returns a dictionary of {pmids:abstracts} 
    print('search_Pubmed_API is DEPRECATED. UTILIZE search_NCBI_API for NCBI ENTREZ API results. Utilize search_getAbs for most comprehensive results.')
    return search_NCBI_API(searchterm_list, maxResults)
    
## DEPRECATED, use search_getAbs for more comprehensive results
def search_NCBI_API(searchterm_list:Union[List[str],str], maxResults:int) -> Dict[str,str]: #returns a dictionary of {pmids:abstracts} 
    print('search_NCBI_API is DEPRECATED. Utilize search_getAbs for most comprehensive results.')
    pmid_to_abs = {}
    i = 0
    
    #type validation, allows string or list input
    if type(searchterm_list)!=list:
        if type(searchterm_list)==str:
            searchterm_list = [searchterm_list]
        else:
            searchterm_list = list(searchterm_list)
    
    #gathers pmids into a set first
    for dz in searchterm_list:
        # get results from searching for disease name through PubMed API
        term = ''
        dz_words = dz.split()
        for word in dz_words:
            term += word + '%20'
        query = term[:-3]
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term='+query
        r = requests.get(url)
        root = ET.fromstring(r.content)    

        # loop over resulting articles
        for result in root.iter('IdList'):
            pmids = [pmid.text for pmid in result.iter('Id')]
            if i >= maxResults:
                break
            for pmid in pmids:
                if pmid not in pmid_to_abs.keys():
                    abstract = PMID_getAb(pmid)
                    if len(abstract)>5:
                        pmid_to_abs[pmid]=abstract
                        i+=1
                    
    return pmid_to_abs

## DEPRECATED, use search_getAbs for more comprehensive results
# get results from searching for disease name through EBI API
def search_EBI_API(searchterm_list:Union[List[str],str], maxResults:int) -> Dict[str,str]: #returns a dictionary of {pmids:abstracts}    
    print('DEPRECATED. Utilize search_getAbs for most comprehensive results.')
    pmids_abs = {}
    i = 0
    
    #type validation, allows string or list input
    if type(searchterm_list)!=list:
        if type(searchterm_list)==str:
            searchterm_list = [searchterm_list]
        else:
            searchterm_list = list(searchterm_list)
    
    #gathers pmids into a set first
    for dz in searchterm_list:
        if i >= maxResults:
            break
        term = ''
        dz_words = dz.split()
        for word in dz_words:
            term += word + '%20'
        query = term[:-3]
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query='+query+'&resulttype=core'
        r = requests.get(url)
        root = ET.fromstring(r.content)

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
# Filtering can be 
#   'strict' - must have some exact match to at leastone of search terms/phrases in text)
#   'lenient' - part of the abstract must match at least one word in the search term phrases.
#   'none'
def search_getAbs(searchterm_list:Union[List[str],List[int],str], maxResults:int, filtering:str) -> Dict[str,str]:
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
            if len(pmids) >= maxResults:
                break
            pmidlist = [pmid.text for pmid in result.iter('Id')]
            pmids.update(pmidlist)

        ## get results from searching for disease name through EBI API
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query='+query+'&resulttype=core'
        r = requests.get(url)
        root = ET.fromstring(r.content)

        # loop over resulting articles
        for result in root.iter('result'):
            if len(pmids) >= maxResults:
                break
            pmidlist = [pmid.text for pmid in result.iter('id')]
            #can also gather abstract and title here but for some reason did not work as intended the first time. Optimize in future versions to reduce latency.
            if len(pmidlist) > 0:
                pmid = pmidlist[0]
                if pmid[0].isdigit():
                    pmids.add(pmid)
    
    #Construct sets for filtering (right before adding abstract to pmid_abs
    # The purpose of this is to do a second check of the abstracts, filters out any abstracts unrelated to the search terms
    #if filtering is 'lenient' or default
    if filtering !='none' or filtering !='strict':
        filter_terms = set(searchterm_list).union(set(str(re.sub(',','',' '.join(searchterm_list))).split()).difference(STOPWORDS))
        '''
        # The above is equivalent to this but uses less memory and may be faster:
        #create a single string of the terms within the searchterm_list
        joined = ' '.join(searchterm_list)
        #remove commas
        comma_gone = re.sub(',','',joined)
        #split the string into list of words and convert list into a Pythonic set
        split = set(comma_gone.split())
        #remove the STOPWORDS from the set of key words
        key_words = split.difference(STOPWORDS)
        #create a new set of the list members in searchterm_list
        search_set = set(searchterm_list)
        #join the two sets
        terms = search_set.union(key_words)
        #if any word(s) in the abstract intersect with any of these terms then the abstract is good to go.
        '''
    
    ## get abstracts from EBI PMID API and output a dictionary
    for pmid in pmids:
        abstract = PMID_getAb(pmid)
        if len(abstract)>5:
            #do filtering here
            if filtering == 'strict':
                uncased_ab = abstract.lower()
                for term in searchterm_list:
                    if term.lower() in uncased_ab:
                        pmid_abs[pmid] = abstract
                        break
            elif filtering =='none':
                pmid_abs[pmid] = abstract
            
            #Default filtering is 'lenient'.
            else:
                #Else and if are separated for readability and to better understand logical flow.
                if set(filter_terms).intersection(set(word_tokenize(abstract))):
                    pmid_abs[pmid] = abstract
                
                    
    print('Found',len(pmids),'PMIDs. Gathered',len(pmid_abs),'Relevant Abstracts.')
    
    return pmid_abs

# Generate predictions for a PubMed Id
# nlp: en_core_web_lg
# nlpSci: en_ner_bc5cdr_md
# nlpSci2: en_ner_bionlp13cg_md
# Defaults to load my_model_orphanet_final, the most up-to-date version of the classification model,
# but can also be run on any other tf.keras model
#This was originally getPredictions
def getPMIDPredictions(pmid:Union[str,int], classify_model_vars:Tuple[Any,Any,Any,Any,Any]) -> Tuple[str,float,bool]:
    nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer = classify_model_vars
    abstract = PMID_getAb(pmid)
    
    if len(abstract)>5:
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
    
    else:
        return abstract, 0.0, False


def getTextPredictions(abstract:str, classify_model_vars:Tuple[Any,Any,Any,Any,Any]) -> Tuple[float,bool]:
    
    nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer = classify_model_vars
    
    if len(abstract)>5:
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
    
    else:
        return 0.0, False

if __name__ == '__main__':
    print('Loading 5 NLP models...')
    classify_model_vars= init_classify_model()
    print('All models loaded.')
    pmid = input('\nEnter PubMed PMID (or DONE): ')
    while pmid != 'DONE':
        abstract, prob, isEpi = getPredictions(pmid, classify_model_vars)
        print(abstract, prob, isEpi)
        pmid = input('\nEnter PubMed PMID (or DONE): ')