import sys
import nltk
from nltk.corpus import stopwords
from nltk import tokenize
STOPWORDS = set(stopwords.words('english'))
import string
PUNCTUATION = set(char for char in string.punctuation)
import csv
import spacy
import re
from transformers import BertConfig, AutoModelForTokenClassification, BertTokenizer, pipeline
import numpy as np
import pandas as pd
import torch
import requests
import xml.etree.ElementTree as ET
import classify_abs
import json
import codecs


## Section: Dictionary Look-up for Disease Labeling
# This generates a dictionary of all GARD disease names. It is a dependency for get_diseases, autosearch, and all higher level functions that utilize those functions.
def load_GARD_diseases():
    diseases = json.load(codecs.open('gard-id-name-synonyms.json', 'r', 'utf-8-sig'))

    #keys are going to be disease names, values are going to be the GARD ID, set up this way bc dictionaries are faster lookup than lists
    GARD_dict = {}
    #It is possible to use this to speed up get_diseases f(x), but decided it was not needed
    GARD_firstwds = set()

    #Find out what the length of the longest disease name sequence is, of all names and synonyms. This is used by get_diseases
    max_length = -1
    for entry in diseases:
        if entry['name'] not in GARD_dict.keys():
            s = entry['name'].lower().strip()
            if s not in STOPWORDS and len(s)>4:
                GARD_dict[s] = entry['gard_id']
                if s.split()[0] not in STOPWORDS:
                    GARD_firstwds.add(s.split()[0])
                #compare length
                l = len(s.split())
                if l>max_length:
                    max_length = l
        if entry['synonyms']:
            for synonym in entry['synonyms']:
                if synonym not in GARD_dict.keys():
                    s = synonym.lower().strip()
                    if s not in STOPWORDS and len(s)>4:
                        GARD_dict[s] = entry['gard_id']
                        if s.split()[0] not in STOPWORDS:
                            GARD_firstwds.add(s.split()[0])
                        #compare length
                        l = len(s.split())
                        if l>max_length:
                            max_length = l
    return GARD_dict, max_length

#Works much faster if broken down into sentences. Resulted in poorer testing when incorporating GARD_firstwd_dict.
#compares every phrase in a sentence to see if it matches anything in the GARD dictionary of diseases.
def get_diseases(sentence, GARD_dict, max_length):   
    tokens = [s.lower().strip() for s in nltk.word_tokenize(sentence)]
    diseases = []
    ids = []
    i=0
    #Iterates through every word, builds string that is max_length or less to compare.
    while i <len(tokens):
        #Find out maximum comparison length and 
        if (len(tokens)-i) < max_length:
            compare_length=len(tokens)-i
        else:
            compare_length = max_length
        #Compares longest sequences first and goes down until there is a match
        #print('(start compare_length)',compare_length)
        while compare_length>0:
            s = ' '.join(tokens[i:i+compare_length])
            if s.lower() in GARD_dict.keys():
                diseases.append(s)
                ids.append(GARD_dict[s.lower()])
                #Need to skip over the next few indexes
                i+=compare_length-1
                break
            else:
                compare_length-=1
        i+=1
    return diseases,ids

## Section: Prepare ML/DL Models
# This fuction prepares the model. Should call before running in notebook.
def init_NER_pipeline(name_or_path_to_model_folder = "ncats/EpiExtract4GARD"): #NER_pipeline = init_NER_pipeline()
    tokenizer = BertTokenizer.from_pretrained(name_or_path_to_model_folder)
    custommodel = AutoModelForTokenClassification.from_pretrained(name_or_path_to_model_folder)
    customNER = pipeline('ner', custommodel, tokenizer=tokenizer, aggregation_strategy='simple')
    return customNER


## Section: Information Acquisition
#moved PMID_getAb and search_getAbs to classify_abs.py

## Section: Information Extraction
#Preprocessing function, turns abstracts into sentences
def str2sents(string):
    string = re.sub('<.{1,4}>', ' ', string)
    string = re.sub("  *", " " , string)
    string = re.sub("^ ", "" , string)
    string = re.sub(" $", "" , string)
    string = re.sub("  ", " " , string)
    string=string.strip()
    sentences = tokenize.sent_tokenize(string)
    return sentences

#Input: Sentences, Output: Sets with GARD Disease ID, isease Name, Location, Epidemiologic Identifier, Epidemiologic Statistic Info
def parse_info(sentences, model_outputs, GARD_dict, max_length):
    ab_ids=set()
    ab_dis=set()
    ab_locs=set()
    ab_epis=set()
    #Stats are going to be a list because I want them to remain ordered and they can have repeated information
    ab_stats=[]
    
    for output in model_outputs:
        stats = [entity['word'] for entity in output if entity['entity_group'] =='STAT']
        epis = {entity['word'] for entity in output if entity['entity_group'] =='EPI'}
        locs = {entity['word'] for entity in output if entity['entity_group'] =='LOC'}
        ab_stats+=stats
        ab_epis.update(epis)
        ab_locs.update(locs)

    for sentence in sentences:
        diseases,ids = get_diseases(sentence, GARD_dict, max_length)
        ab_dis.update(diseases)
        ab_ids.update(ids)
    
    return ab_ids, ab_dis, ab_locs, ab_epis, ab_stats

#These are the main three main functions that can be called in a noteboook.
#Extracts Disease GARD ID, Disease Name, Location, Epidemiologic Identifier, Epidemiologic Statistic given a PubMed ID
def PMID_extraction(pmid, NER_pipeline, GARD_dict, max_length):
    text = classify_abs.PMID_getAb(pmid)
    if len(text)>5:
        sentences = str2sents(text)
        model_outputs = [NER_pipeline(sent) for sent in sentences]
        ab_ids, ab_dis, ab_locs, ab_epis, ab_stats = parse_info(sentences, model_outputs, GARD_dict, max_length)
        return text, ab_ids, ab_dis, ab_locs, ab_epis, ab_stats
    else:
        return '*ABSTRACT NOT FOUND*',{"N/A"},{"N/A"},{"N/A"},{"N/A"},["N/A"]

#Can search by 7-digit GARD_ID, 12-digit "GARD:{GARD_ID}", matched search term, or arbitrary search term
#Returns list of terms to search by
def autosearch(searchterm, GARD_dict, matching=2):
    searchterm = searchterm.lower()
    while matching>=1:
        if 'gard:' in searchterm and len(searchterm)==12:
            searchterm = searchterm.replace('gard:','GARD:')
            l = [k for k,v in GARD_dict.items() if v==searchterm]
            print("SEARCH TERM MATCHED TO GARD DICTIONARY. SEARCHING FOR: ",l)
            return l
        
        elif len(searchterm)==7 and searchterm[0].isdigit() and searchterm[-1].isdigit():
            searchterm = 'GARD:'+searchterm
            l = [k for k,v in GARD_dict.items() if v==searchterm]
            print("SEARCH TERM MATCHED TO GARD DICTIONARY. SEARCHING FOR: ",l)
            return l
        
        elif searchterm in GARD_dict.keys():
            l = [k for k,v in GARD_dict.items() if v==GARD_dict[searchterm]]
            print("SEARCH TERM MATCHED TO GARD DICTIONARY. SEARCHING FOR: ",l)
            return l
        
        else:
            #This can be replaced with some other common error in user input that is easily fixed
            searchterm = searchterm.replace(' ','-')
            return autosearch(searchterm, GARD_dict, matching-1)
    print("SEARCH TERM DID NOT MATCH TO GARD DICTIONARY. SEARCHING BY USER INPUT")
    return [searchterm]
    
#Given a search term and max results to return, this will acquire PubMed IDs and Title+Abstracts and Classify them as epidemiological.
#It then extracts Epidemiologic Information[Disease GARD ID, Disease Name, Location, Epidemiologic Identifier, Epidemiologic Statistic] for each abstract
def search_term_extraction(search_term, maxResults, #for abstract search
                           NER_pipeline, GARD_dict, max_length, #for extraction 
                           nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer): #for classification
    #Format of Output
    results = pd.DataFrame(columns=['pmid', 'abstract','epi_prob','isEpi','ab_ids','ab_dis','ab_locs','ab_epis','ab_stats'])
    
    ##Check to see if search term maps to anything in the GARD dictionary, if so it pulls up all synonyms for the search
    search_term_list = autosearch(search_term, GARD_dict)
    
    #Gather title+abstracts into a dictionary {pmid:abstract}
    pmid_abs = classify_abs.search_getAbs(search_term_list,maxResults)
    
    for pmid, abstract in pmid_abs.items():
        #Preprocessing Functions for Extraction
        sentences = str2sents(abstract)
        model_outputs = [NER_pipeline(sent) for sent in sentences]
        
        epi_prob, isEpi = classify_abs.getTextPredictions(abstract, nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer)
        if isEpi:
            ab_ids, ab_dis, ab_locs, ab_epis, ab_stats = parse_info(sentences, model_outputs, GARD_dict, max_length)
            results = results.append({'pmid':pmid, 'abstract':abstract, 'epi_prob':epi_prob, 'isEpi':isEpi,
                                      'ab_ids':ab_ids, 'ab_dis':ab_dis, 'ab_locs':ab_locs, 
                                      'ab_epis':ab_epis, 'ab_stats':ab_stats}, ignore_index=True)
    return results.sort_values('epi_prob', ascending=False)

#Extract if you already have the text and you do not want epi_predictions (this makes things much faster)
#ab_ids, ab_dis, ab_locs, ab_epis, ab_stats = abstract_extraction(abstract, NER_pipeline, GARD_dict, max_length)
def abstract_extraction(text, NER_pipeline, GARD_dict, max_length): 
    if len(text)>5:
        sentences = str2sents(text)
        model_outputs = [NER_pipeline(sent) for sent in sentences]
        ab_ids, ab_dis, ab_locs, ab_epis, ab_stats = parse_info(sentences, model_outputs, GARD_dict, max_length)
        return ab_ids, ab_dis, ab_locs, ab_epis, ab_stats
    else:
        return '*ABSTRACT NOT FOUND*',{"N/A"},{"N/A"},{"N/A"},{"N/A"},["N/A"]


if __name__ == '__main__':
    print('Loading NER Pipeline and GARD Disease Dictionary....')
    path_to_model_folder = input('Input path_to_model_folder. Input "d" to use default model.')
    if path_to_model_folder == 'd':
        NER_pipeline = init_NER_pipeline()
    else:
        NER_pipeline = init_NER_pipeline(path_to_model_folder)
    
    GARD_dict, max_length = load_GARD_diseases()
    print('NER Pipeline and GARD Diseases Loaded')
    search_type = input('Input "pmid" to search by PubMed ID. Input "name" to search by disease name.')
    
    if search_type=='pmid':
        pmid = input('\nEnter PubMed ID (or DONE): ')
        while pmid != 'DONE':
            text, ab_ids, ab_dis, ab_locs, ab_epis, ab_stats = PMID_extraction(pmid, NER_pipeline, GARD_dict, max_length)
            print(text,
              '\nGARD Disease ID: ',ab_ids, 
              '\nGARD Disease: ',ab_dis, 
              '\nLocations: ',ab_locs, 
              '\nEpi Identifier: ',ab_epis, 
              '\nEpi Statistics: ',ab_stats)
            pmid = input('\nEnter PubMed PMID (or DONE): ')
    else:
        print('Searching by term.')
        print('Loading classification model....')
        nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer= classify_abs.init_classify_model()
        print('Classification Model Loaded')
    
        search_term = input('\nEnter search term (or DONE): ')
        while search_term != 'DONE':
            #Input validation for maxResults so it does not automatically break the code
            while True:
                try:
                    maxResults = int(input('How many results to return? [integer]'))
                except ValueError:
                    print("Please enter an integer in digits.")
                    continue
            else:
                break
            
            results = search_term_extraction(search_term, maxResults, 
                                             NER_pipeline, GARD_dict, max_length,
                                             nlp, nlpSci, nlpSci2, classify_model, classify_tokenizer)
            print(result)
            search_term = input('\nEnter search term (or DONE): ')