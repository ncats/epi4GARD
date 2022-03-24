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
from unidecode import unidecode
from collections import OrderedDict
from typing import (
    Dict,
    List,
    Tuple,
    Set,
    Optional,
    Any,
    Union,
)

## Section: Dictionary Look-up for Disease Labeling
# This generates a dictionary of all GARD disease names. It is a dependency for get_diseases, autosearch, and all higher level functions that utilize those functions.
# GARD_dict, max_length = load_GARD_diseases()
def load_GARD_diseases() -> Tuple[Dict[str,str], int]:
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
def get_diseases(sentence:str, GARD_dict:Dict[str,str], max_length:int) -> Tuple[List[str], List[str]]:   
    tokens = [s.lower().strip() for s in nltk.word_tokenize(sentence)]
    diseases = []
    ids = []
    i=0
    #Iterates through every word, builds string that is max_length or less to compare.
    while i <len(tokens):
        #Find out the length of the comparison string, either max_length or less. This brings algorithm from O(n^2) to O(n) time
        compare_length = min(len(tokens)-i, max_length)
        
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
# This fuction prepares the model. Should call before running in notebook. -- The [Any] Type is a Huggingface Pipeline variable
# Default with typing from here: https://stackoverflow.com/questions/38727520/how-do-i-add-default-parameters-to-functions-when-using-type-hinting
def init_NER_pipeline(name_or_path_to_model_folder:str = "ncats/EpiExtract4GARD-v2") -> Tuple[Any, Set[str]]: #NER_pipeline, entities = init_NER_pipeline()
    tokenizer = BertTokenizer.from_pretrained(name_or_path_to_model_folder)
    custommodel = AutoModelForTokenClassification.from_pretrained(name_or_path_to_model_folder)
    customNER = pipeline('ner', custommodel, tokenizer=tokenizer, aggregation_strategy='simple')
    
    config = BertConfig.from_pretrained(name_or_path_to_model_folder)
    labels = {re.sub(".-","",label) for label in config.label2id.keys() if label != "O"}
    return customNER, labels

## Section: Information Acquisition
#moved PMID_getAb and search_getAbs to classify_abs.py

## Section: Information Extraction
#Preprocessing function, turns abstracts into sentences
def str2sents(string:str) -> List[str]:
    superscripts = re.findall('<sup>.</sup>', string)
    for i in range(len(superscripts)):
        string = re.sub('<sup>.</sup>', '^'+superscripts[i][5], string)
    string = re.sub('<.{1,4}>', ' ', string)
    string = re.sub("  *", " " , string)
    string = re.sub("^ ", "" , string)
    string = re.sub("$", "" , string)
    string = re.sub("  ", " " , string)
    string = re.sub("™", "" , string)
    string = re.sub("®", "" , string)
    string = re.sub("•", "" , string)
    string = re.sub("…", "" , string)
    string = re.sub("♀", "female" , string)
    string = re.sub("♂", "male" , string)
    string = unidecode(string)
    string=string.strip()
    sentences = tokenize.sent_tokenize(string)
    return sentences

# Input: Sentences & Model Outputs 
# Output: Dictionary with all entity types (dynamic to fit multiple models)
# model_outputs is list of NER_pipeline outputs
# labels are a set of all the possible entities (not including "O"). This is a misnomer. Was originally named "entities" but changed to not get confused with other code
def parse_info(sentences:List[str], model_outputs:List[List[Union[Dict[str,str],None]]], labels:Set, extract_diseases:bool, GARD_dict:Dict[str,str], max_length:int) -> Dict[str,Union[List[str],None]]:
    #do not use dict.fromkeys(labels,set()) as the value is a single instance which all keys point to. 
    #The value is therefore effectively immutable. 
    
    #See: https://docs.python.org/3/library/stdtypes.html?highlight=dict%20fromkeys#dict.fromkeys
    output_dict = {label:[] for label in labels}
    for output in model_outputs:
        #This abstracts the labels so that models with different types and numbers of labels can be used.
        for label in labels:
            output_dict[label]+=[entity_dict['word'] for entity_dict in output if entity_dict['entity_group'] ==label]
                
    if 'DIS' not in output_dict.keys() and extract_diseases:
        output_dict['DIS'] = []
        output_dict['IDS'] = []
        for sentence in sentences:
            diseases,ids = get_diseases(sentence, GARD_dict, max_length)
            output_dict['DIS']+=diseases
            output_dict['IDS']+=ids
        
    #Clean up Output Dict
    for entity, output in output_dict.items():
        if not output:
            output_dict[entity] = None
        else:
            #remove duplicates from list but keep ordering instead of using sets
            output = list(OrderedDict.fromkeys(output)) 
            output_dict[entity] = output
            
    if output_dict['EPI'] and (output_dict['STAT'] or output_dict['LOC'] or output_dict['DATE']):
        return output_dict

#These are the main three main functions that can be called in a noteboook.
#Extracts Disease GARD ID, Disease Name, Location, Epidemiologic Identifier, Epidemiologic Statistic, etc. given a PubMed ID
#Dynamic dictionary output to fit multiple models
def PMID_extraction(pmid:Union[str,int], NER_pipeline:Any, labels:Union[Set[str],List[str]], GARD_dict:Dict[str,str], max_length:int) -> Dict[str,Union[str,List[str],None]]: #extraction = PMID_extraction(pmid, NER_pipeline, labels, GARD_dict, max_length)
    text = classify_abs.PMID_getAb(pmid)
    if len(text)>5:
        sentences = str2sents(text)
        model_outputs = [NER_pipeline(sent) for sent in sentences]
        output_dict = parse_info(sentences, model_outputs, labels, GARD_dict, max_length)
        output_dict['ABSTRACT'] = text
        return output_dict
    else:
        out = ['ABSTRACT']
        out+=list(labels)
        output_dict =dict.fromkeys(out,"N/A")
        output_dict['ABSTRACT'] = '*ABSTRACT NOT FOUND*'
        return output_dict

#Can search by 7-digit GARD_ID, 12-digit "GARD:{GARD_ID}", matched search term, or arbitrary search term
#Returns list of terms to search by
# search_term_list = autosearch(search_term, GARD_dict)
def autosearch(searchterm:Union[str,int], GARD_dict:Dict[str,str], matching=2) -> List[str]:
    
    #comparisons below only handly strings, allows int input
    if type(searchterm) is not str:
        searchterm = str(searchterm)
    
    #for the disease names to match
    searchterm = searchterm.lower()
    
    while matching>=1:
        #search in form of 'GARD:0000001'
        if 'gard:' in searchterm and len(searchterm)==12:
            searchterm = searchterm.replace('gard:','GARD:')
            l = [k for k,v in GARD_dict.items() if v==searchterm]
            if len(l)>0:
                print("SEARCH TERM MATCHED TO GARD DICTIONARY. SEARCHING FOR: ",l)
                return l
        
        #can take int or str of digits of variable input
        #search in form of 777 or '777' or '00777' or '0000777'
        elif searchterm[0].isdigit() and searchterm[-1].isdigit():
            if len(searchterm)>7:
                raise ValueError('GARD ID IS NOT VALID. RE-ENTER SEARCH TERM')
            searchterm = 'GARD:'+'0'*(7-len(str(searchterm)))+str(searchterm)
            l = [k for k,v in GARD_dict.items() if v==searchterm]
            if len(l)>0:
                print("SEARCH TERM MATCHED TO GARD DICTIONARY. SEARCHING FOR: ",l)
                return l
        
        #search in form of 'mackay shek carr syndrome' and returns all synonyms ('retinal degeneration with nanophthalmos, cystic macular degeneration, and angle closure glaucoma', 'retinal degeneration, nanophthalmos, glaucoma', 'mackay shek carr syndrome')
        #considers the GARD ID as the lemma, and the search term as one form. maps the form to the lemma and then uses that lemma to find all related forms in the GARD dict. 
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
   
#This ensures that there is a standardized ordering of df columns while ensuring dynamics with multiple models. This is used by search_term_extraction.
def order_labels(entity_classes:Union[Set[str],List[str]]) -> List[str]:
    ordered_labels = []
    label_order = ['DIS','ABRV','EPI','STAT','LOC','DATE','SEX','ETHN']
    ordered_labels = [label for label in label_order if label in entity_classes]
    #This adds any extra entities (from yet-to-be-created models) to the end of the ordered list of labels 
    for entity in entity_classes:
        if entity not in label_order:
            ordered_labels.append(entity)
    return ordered_labels

#Given a search term and max results to return, this will acquire PubMed IDs and Title+Abstracts and Classify them as epidemiological.
#It then extracts Epidemiologic Information[Disease GARD ID, Disease Name, Location, Epidemiologic Identifier, Epidemiologic Statistic] for each abstract
# results = search_term_extraction(search_term, maxResults, filering, NER_pipeline, labels, extract_diseases, GARD_dict, max_length, classify_model_vars)
#Returns a Pandas dataframe                                                                                                          
def search_term_extraction(search_term:Union[int,str], maxResults:int, filtering:str, #for abstract search
                           NER_pipeline:Any, entity_classes:Union[Set[str],List[str]], #for biobert extraction 
                           extract_diseases:bool, GARD_dict:Dict[str,str], max_length:int, #for disease extraction
                           classify_model_vars:Tuple[Any,Any,Any,Any,Any]) -> Any: #for classification
                                                                                                                                             
    #Format of Output
    ordered_labels = order_labels(entity_classes)
    if extract_diseases:
        columns = ['PMID', 'ABSTRACT','EPI_PROB','IsEpi','IDS','DIS']+ordered_labels
    else:
        columns = ['PMID', 'ABSTRACT','EPI_PROB','IsEpi']+ordered_labels
    
    results = pd.DataFrame(columns=columns)
    
    ##Check to see if search term maps to anything in the GARD dictionary, if so it pulls up all synonyms for the search
    search_term_list = autosearch(search_term, GARD_dict)
    
    #Gather title+abstracts into a dictionary {pmid:abstract}
    pmid_abs = classify_abs.search_getAbs(search_term_list, maxResults,filtering)
    
    for pmid, abstract in pmid_abs.items():
        epi_prob, isEpi = classify_abs.getTextPredictions(abstract, classify_model_vars)
        if isEpi:
            #Preprocessing Functions for Extraction
            sentences = str2sents(abstract)
            model_outputs = [NER_pipeline(sent) for sent in sentences]
            extraction = parse_info(sentences, model_outputs, entity_classes, extract_diseases, GARD_dict, max_length)
            if extraction:
                extraction.update({'PMID':pmid, 'ABSTRACT':abstract, 'EPI_PROB':epi_prob, 'IsEpi':isEpi})
                #Slow dataframe update
                results = results.append(extraction, ignore_index=True)
    
    print(len(results),'abstracts classified as epidemiological.')
    return results.sort_values('EPI_PROB', ascending=False)

#Identical to search_term_extraction, except it returns a JSON object instead of a df
def API_extraction(search_term:Union[int,str], maxResults:int, filtering:str, #for abstract search
                   NER_pipeline:Any, entity_classes:Union[Set[str],List[str]], #for biobert extraction 
                   extract_diseases:bool, GARD_dict:Dict[str,str], max_length:int, #for disease extraction
                   classify_model_vars:Tuple[Any,Any,Any,Any,Any]) -> Any: #for classification
                                                                                                                                             
    #Format of Output
    ordered_labels = order_labels(entity_classes)
    if extract_diseases:
        json_output = ['PMID', 'ABSTRACT','EPI_PROB','IsEpi','IDS','DIS']+ordered_labels
    else:
        json_output = ['PMID', 'ABSTRACT','EPI_PROB','IsEpi']+ordered_labels
    
    results = {'entries':[]} 
    
    ##Check to see if search term maps to anything in the GARD dictionary, if so it pulls up all synonyms for the search
    search_term_list = autosearch(search_term, GARD_dict)
    
    #Gather title+abstracts into a dictionary {pmid:abstract}
    pmid_abs = classify_abs.search_getAbs(search_term_list, maxResults,filtering)
    
    for pmid, abstract in pmid_abs.items():
        epi_prob, isEpi = classify_abs.getTextPredictions(abstract, classify_model_vars)
        if isEpi:
            #Preprocessing Functions for Extraction
            sentences = str2sents(abstract)
            model_outputs = [NER_pipeline(sent) for sent in sentences]
            extraction = parse_info(sentences, model_outputs, entity_classes, extract_diseases, GARD_dict, max_length)
            if extraction:
                extraction.update({'PMID':pmid, 'ABSTRACT':abstract, 'EPI_PROB':epi_prob, 'IsEpi':isEpi})
                extraction = OrderedDict([(term, extraction[term]) for term in json_output])
                results['entries'].append(extraction)
    
    #sort 
    results['entries'].sort(reverse=True, key=lambda x:x['EPI_PROB'])
    
    #float is not JSON serializable, so must convert all epi_probs to str
    # This returns a map object, which is not JSON serializable
    #results['entries'] = map(lambda entry:str(entry['EPI_PROB']),results['entries'])
    
    for entry in results['entries']:
        entry['EPI_PROB'] = str(entry['EPI_PROB'])
    
    return json.dumps(results)

#Extract if you already have the text and you do not want epi_predictions (this makes things much faster)
#extraction = abstract_extraction(text, NER_pipeline, labels, GARD_dict, max_length)
def abstract_extraction(text:str, NER_pipeline:Any, entity_classes:Union[Set[str],List[str]], GARD_dict:Dict[str,str], max_length:int) -> Dict[str,Union[str,List[str],None]]: 
    if len(text)>5:
        sentences = str2sents(text)
        model_outputs = [NER_pipeline(sent) for sent in sentences]
        output_dict = parse_info(sentences, model_outputs, entity_classes, GARD_dict, max_length)
        output_dict['ABSTRACT'] = text
        return output_dict
    else:
        out = ['ABSTRACT']
        out+=list(entity_classes)
        output_dict =dict.fromkeys(out,"N/A")
        output_dict['ABSTRACT'] = '*ABSTRACT NOT FOUND*'
        return output_dict
