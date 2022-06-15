from typing import List, Dict, Union, Optional, Set, Tuple

# coding=utf-8
##             PUBLIC DOMAIN NOTICE
##     National Center for Advancing Translational Sciences

## This software/database is a "United States Government Work" under the terms of the United States Copyright Act. It was written as part of the author's official duties as United States Government employee and thus cannot be copyrighted. This software is freely available to the public for use. The National Center for Advancing Translational Science (NCATS) and the U.S. Government have not placed any restriction on its use or reproduction.  Although all reasonable efforts have been taken to ensure the accuracy and reliability of the software and data, the NCATS and the U.S. Government do not and cannot warrant the performance or results that may be obtained by using this software or data. The NCATS and the U.S.  Government disclaim all warranties, express or implied, including warranties of performance, merchantability or fitness for any particular purpose.  Please cite the authors in any work or product based on this material.

# Written by William Kariampuzha @ NIH/NCATS. Adapted from code written by Jennifer John, et al. 
# The transformer-based pipeline code has its own copyright notice under the Apache License. 
# The code was compiled into a single python file to make adding additional features and importing into other modules easy.
# Each section has its own import statements to facilitate clean code reuse, except for typing which applies to all.

## Section: GATHER ABSTRACTS FROM APIs
import requests
import xml.etree.ElementTree as ET
import nltk
nltk.data.path.extend(["/home/user/app/nltk_data","./nltk_data"])
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from nltk import tokenize as nltk_tokenize

#Retreives abstract and title (concatenated) from EBI API based on PubMed ID
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
    
## This is the main, most comprehensive search_term function, it can take in a search term or a list of search terms and output a dictionary of {pmids:abstracts}
## Gets results from searching through both PubMed and EBI search term APIs, also makes use of the EBI API for PMIDs. 
## EBI API and PubMed API give different results
# This makes n+2 API calls where n<=maxResults, which is slow 
# There is a way to optimize by gathering abstracts from the EBI API when also getting pmids but did not pursue due to time constraints
# Filtering can be 
#   'strict' - must have some exact match to at least one of search terms/phrases in text)
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
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term='+query#+"&retmax="+str(int(maxResults/len(searchterm_list)))
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
                #Reversing the list hopefully cuts down on the number of if statements bc the search terms are ordered longest to shortest and shorter terms are more likely to be in the abstract
                for term in reversed(searchterm_list):
                    if term.lower() in uncased_ab:
                        pmid_abs[pmid] = abstract
                        break
            elif filtering =='none':
                pmid_abs[pmid] = abstract
            
            #Default filtering is 'lenient'.
            else:
                #Else and if are separated for readability and to better understand logical flow.
                if set(filter_terms).intersection(set(nltk_tokenize.word_tokenize(abstract))):
                    pmid_abs[pmid] = abstract
                
                    
    print('Found',len(pmids),'PMIDs. Gathered',len(pmid_abs),'Relevant Abstracts.')
    
    return pmid_abs

#This is a streamlit version of search_getAbs. Refer to search_getAbs for documentation
import streamlit as st
def streamlit_getAbs(searchterm_list:Union[List[str],List[int],str], maxResults:int, filtering:str) -> Dict[str,str]:
    pmids = set()
    
    pmid_abs = {}
    
    if type(searchterm_list)!=list:
        if type(searchterm_list)==str:
            searchterm_list = [searchterm_list]
        else:
            searchterm_list = list(searchterm_list)
    #maxResults is multiplied by a little bit because sometimes the results returned is more than maxResults
    percent_by_step = 1/maxResults
    with st.spinner("Gathering PubMed IDs..."):
        PMIDs_bar = st.progress(0)
        for dz in searchterm_list:
            term = ''
            dz_words = dz.split()
            for word in dz_words:
                term += word + '%20'
            query = term[:-3]
            #dividing by the len( of the search_ter
            url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term='+query#+"&retmax="+str(int(maxResults/len(searchterm_list)))
            r = requests.get(url)
            root = ET.fromstring(r.content)
    
            for result in root.iter('IdList'):
                for pmid in result.iter('Id'):
                    if len(pmids) >= maxResults:
                        break
                    pmids.add(pmid.text)
                    PMIDs_bar.progress(min(round(len(pmids)*percent_by_step,1),1.0))
    
            url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query='+query+'&resulttype=core'
            r = requests.get(url)
            root = ET.fromstring(r.content)
    
            for result in root.iter('result'):
                if len(pmids) >= maxResults:
                    break
                pmidlist = [pmid.text for pmid in result.iter('id')]
                if len(pmidlist) > 0:
                    pmid = pmidlist[0]
                    if pmid[0].isdigit():
                        pmids.add(pmid)
                        PMIDs_bar.progress(min(round(len(pmids)*percent_by_step,1),1.0))
        PMIDs_bar.empty()
    
    with st.spinner("Found "+str(len(pmids))+" PMIDs. Gathering Abstracts and Filtering..."):
        abstracts_bar = st.progress(0)
        percent_by_step = 1/maxResults
        if filtering !='none' or filtering !='strict':
            filter_terms = set(searchterm_list).union(set(str(re.sub(',','',' '.join(searchterm_list))).split()).difference(STOPWORDS))
    
        for i, pmid in enumerate(pmids):
            abstract = PMID_getAb(pmid)
            if len(abstract)>5:
                #do filtering here
                if filtering == 'strict':
                    uncased_ab = abstract.lower()
                    #Reversing the list hopefully cuts down on the number of if statements bc the search terms are ordered longest to shortest and shorter terms are more likely to be in the abstract
                    for term in reversed(searchterm_list):
                        if term.lower() in uncased_ab:
                            pmid_abs[pmid] = abstract                            
                            break
                elif filtering =='none':
                    pmid_abs[pmid] = abstract
                #Default filtering is 'lenient'.
                else:
                    #Else and if are separated for readability and to better understand logical flow.
                    if set(filter_terms).intersection(set(nltk_tokenize.word_tokenize(abstract))):
                        pmid_abs[pmid] = abstract
            abstracts_bar.progress(min(round(i*percent_by_step,1),1.0))
        abstracts_bar.empty()
    found = len(pmids)
    relevant = len(pmid_abs)
    st.success('Found '+str(found)+' PMIDs. Gathered '+str(relevant)+' Relevant Abstracts. Classifying and extracting epidemiology information...')
    
    return pmid_abs, (found, relevant)

## Section: LSTM RNN Epi Classification Model (EpiClassify4GARD)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import spacy

class Classify_Pipeline:
    def __init__(self,model:str='LSTM_RNN'):
        #Load spaCy models
        self.nlp = spacy.load('en_core_web_lg')
        self.nlpSci = spacy.load("en_ner_bc5cdr_md")
        self.nlpSci2 = spacy.load('en_ner_bionlp13cg_md')
        # load the tokenizer
        with open(model+'/tokenizer.pickle', 'rb') as handle:
            import pickle
            self.classify_tokenizer = pickle.load(handle)
        # Defaults to load my_model_orphanet_final, the most up-to-date version of the classification model,
        # but can also be run on any other tf.keras model
    
        # load the model
        self.classify_model = tf.keras.models.load_model(model)
        # for preprocessing
        from nltk.corpus import stopwords
        self.STOPWORDS = set(stopwords.words('english'))
        # Modes 
        self.max_length = 300
        self.trunc_type = 'post'
        self.padding_type = 'post'

    def __str__(self) -> str:
        return "Instantiation: epi_classify = Classify_Pipeline(path_to_model_folder)" +"\n Calling: prob, isEpi = epi_classify(text) \n PubMed ID Predictions: abstracts, prob, isEpi = epi_classify.getPMIDPredictions(pmid)"
    
    def __call__(self, abstract:str) -> Tuple[float,bool]:
        return self.getTextPredictions(abstract)
    
    def getTextPredictions(self, abstract:str) -> Tuple[float,bool]:
        if len(abstract)>5:
            # remove stopwords
            for word in self.STOPWORDS:
                token = ' ' + word + ' '
                abstract = abstract.replace(token, ' ')
                abstract = abstract.replace('  ', ' ')

            # preprocess abstract
            abstract_standard = [self.standardizeAbstract(self.standardizeSciTerms(abstract))]
            sequence = self.classify_tokenizer.texts_to_sequences(abstract_standard)
            padded = pad_sequences(sequence, maxlen=self.max_length, padding=self.padding_type, truncating=self.trunc_type)

            y_pred1 = self.classify_model.predict(padded) # generate prediction
            y_pred = np.argmax(y_pred1, axis=1) # get binary prediction

            prob = y_pred1[0][1]
            if y_pred == 1:
                isEpi = True
            else:
                isEpi = False

            return prob, isEpi
        else:
            return 0.0, False
    
    def getPMIDPredictions(self, pmid:Union[str,int]) -> Tuple[str,float,bool]:
        abstract = PMID_getAb(pmid)
        prob, isEpi = self.getTextPredictions(abstract)
        return abstract, prob, isEpi
    
    # Standardize the abstract by replacing all named entities with their entity label.
    # Eg. 3 patients reported at a clinic in England --> CARDINAL patients reported at a clinic in GPE
    # expects the spaCy model en_core_web_lg as input
    def standardizeAbstract(self, abstract:str) -> str:
        doc = self.nlp(abstract)
        newAbstract = abstract
        for e in reversed(doc.ents):
            if e.label_ in {'PERCENT','CARDINAL','GPE','LOC','DATE','TIME','QUANTITY','ORDINAL'}:
                start = e.start_char
                end = start + len(e.text)
                newAbstract = newAbstract[:start] + e.label_ + newAbstract[end:]
        return newAbstract

    # Same as above but replaces biomedical named entities from scispaCy models
    # Expects as input en_ner_bc5cdr_md and en_ner_bionlp13cg_md
    def standardizeSciTerms(self, abstract:str) -> str:
        doc = self.nlpSci(abstract)
        newAbstract = abstract
        for e in reversed(doc.ents):
            start = e.start_char
            end = start + len(e.text)
            newAbstract = newAbstract[:start] + e.label_ + newAbstract[end:]

        doc = self.nlpSci2(newAbstract)
        for e in reversed(doc.ents):
            start = e.start_char
            end = start + len(e.text)
            newAbstract = newAbstract[:start] + e.label_ + newAbstract[end:]
        return newAbstract
        
## Section: GARD SEARCH
# can identify rare diseases in text using the GARD dictionary from neo4j
# and map a GARD ID, name, or synonym to all of the related synonyms for searching APIs
from nltk import tokenize as nltk_tokenize
class GARD_Search:
    def __init__(self):
        import json, codecs
        #These are opened locally so that garbage collection removes them from memory
        with codecs.open('gard-id-name-synonyms.json', 'r', 'utf-8-sig') as f:
            diseases = json.load(f)
        from nltk.corpus import stopwords
        STOPWORDS = set(stopwords.words('english'))
        
        #keys are going to be disease names, values are going to be the GARD ID, set up this way bc dictionaries are faster lookup than lists
        GARD_dict = {}
        #Find out what the length of the longest disease name sequence is, of all names and synonyms. This is used by get_diseases
        max_length = -1
        for entry in diseases:
            if entry['name'] not in GARD_dict.keys():
                s = entry['name'].lower().strip()
                if s not in STOPWORDS and len(s)>5:
                    GARD_dict[s] = entry['gard_id']
                    #compare length
                    max_length = max(max_length,len(s.split()))

            if entry['synonyms']:
                for synonym in entry['synonyms']:
                    if synonym not in GARD_dict.keys():
                        s = synonym.lower().strip()
                        if s not in STOPWORDS and len(s)>5:
                            GARD_dict[s] = entry['gard_id']
                            max_length = max(max_length,len(s.split()))
                            
        self.GARD_dict = GARD_dict
        self.max_length = max_length
    
    def __str__(self) -> str:
        return '''Instantiation: rd_identify = GARD_Search()
                  Calling: diseases, ids = rd_identify(text) 
                  Autosearch: search_terms = rd_identify.autosearch(searchterm)
               '''
    
    def __call__(self, sentence:str) -> Tuple[List[str], List[str]]:
        return self.get_diseases(sentence)
    
    #Works much faster if broken down into sentences.
    #compares every phrase in a sentence to see if it matches anything in the GARD dictionary of diseases.
    def get_diseases(self, sentence:str) -> Tuple[List[str], List[str]]:   
        tokens = [s.lower().strip() for s in nltk_tokenize.word_tokenize(sentence)]
        diseases = []
        ids = []
        i=0
        #Iterates through every word, builds string that is max_length or less to compare.
        while i <len(tokens):
            #Find out the length of the comparison string, either max_length or less. This brings algorithm from O(n^2) to O(n) time
            compare_length = min(len(tokens)-i, self.max_length)

            #Compares longest sequences first and goes down until there is a match
            #print('(start compare_length)',compare_length)
            while compare_length>0:
                s = ' '.join(tokens[i:i+compare_length])
                if s.lower() in self.GARD_dict.keys():
                    diseases.append(s)
                    ids.append(self.GARD_dict[s.lower()])
                    #Need to skip over the next few indexes
                    i+=compare_length-1
                    break
                else:
                    compare_length-=1
            i+=1
        return diseases,ids
    
    #Can search by 7-digit GARD_ID, 12-digit "GARD:{GARD_ID}", matched search term, or arbitrary search term
    #Returns list of terms to search by
    # search_term_list = autosearch(search_term, GARD_dict)
    def autosearch(self, searchterm:Union[str,int], matching=2) -> List[str]:
        #comparisons below only handly strings, allows int input
        if type(searchterm) is not str:
            searchterm = str(searchterm)

        #for the disease names to match
        searchterm = searchterm.lower()

        while matching>=1:
            #search in form of 'GARD:0000001'
            if 'gard:' in searchterm and len(searchterm)==12:
                searchterm = searchterm.replace('gard:','GARD:')
                l = [k for k,v in self.GARD_dict.items() if v==searchterm]
                l.sort(reverse=True, key=lambda x:len(x))
                if len(l)>0:
                    print("SEARCH TERM MATCHED TO GARD DICTIONARY. SEARCHING FOR: ",l)
                    return l

            #can take int or str of digits of variable input
            #search in form of 777 or '777' or '00777' or '0000777'
            elif searchterm[0].isdigit() and searchterm[-1].isdigit():
                if len(searchterm)>7:
                    raise ValueError('GARD ID IS NOT VALID. RE-ENTER SEARCH TERM')
                searchterm = 'GARD:'+'0'*(7-len(str(searchterm)))+str(searchterm)
                l = [k for k,v in self.GARD_dict.items() if v==searchterm]
                l.sort(reverse=True, key=lambda x:len(x))
                if len(l)>0:
                    print("SEARCH TERM MATCHED TO GARD DICTIONARY. SEARCHING FOR: ",l)
                    return l

            #search in form of 'mackay shek carr syndrome' and returns all synonyms ('retinal degeneration with nanophthalmos, cystic macular degeneration, and angle closure glaucoma', 'retinal degeneration, nanophthalmos, glaucoma', 'mackay shek carr syndrome')
            #considers the GARD ID as the lemma, and the search term as one form. maps the form to the lemma and then uses that lemma to find all related forms in the GARD dict. 
            elif searchterm in self.GARD_dict.keys():
                l = [k for k,v in self.GARD_dict.items() if v==self.GARD_dict[searchterm]]
                l.sort(reverse=True, key=lambda x:len(x))
                print("SEARCH TERM MATCHED TO GARD DICTIONARY. SEARCHING FOR: ",l)
                return l

            else:
                #This can be replaced with some other common error in user input that is easily fixed
                searchterm = searchterm.replace('-',' ')
                searchterm = searchterm.replace("'s","")
                return self.autosearch(searchterm, matching-1)
        print("SEARCH TERM DID NOT MATCH TO GARD DICTIONARY. SEARCHING BY USER INPUT")
        return [searchterm]

## Section: BioBERT-based epidemiology NER Model (EpiExtract4GARD)
from nltk import tokenize as nltk_tokenize
from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from torch import nn
import numpy as np
from unidecode import unidecode
import re
from transformers import BertConfig, AutoModelForTokenClassification, BertTokenizer, Trainer
from unidecode import unidecode
from collections import OrderedDict
import json
import pandas as pd
from more_itertools import pairwise

# Subsection: Processing the abstracts into the correct data format

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

@dataclass
class NERInput:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """
    guid: str
    words: List[str]
    labels: List[str]
        
@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
        
class NerDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        abstract: str,
        tokenizer: BertTokenizer,
        config: BertConfig,
    ):
        # TODO clean up all this to leverage built-in features of tokenizers
        ner_inputs = self.abstract2NERinputs(abstract)
        
        self.features = self.convert_NERinputs_to_features(
            ner_inputs,
            config,
            tokenizer,
            cls_token_at_end=bool(config.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if config.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
        )
        self.ner_inputs = ner_inputs
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]
    
    #Preprocessing function, turns abstracts into sentences
    def str2sents(self, string:str) -> List[str]:
        superscripts = re.findall('<sup>.</sup>', string)
        for i in range(len(superscripts)):
            string = re.sub('<sup>.</sup>', '^'+superscripts[i][5], string)
        string = re.sub("<.{1,4}>|  *|  ", " ", string)
        string = re.sub("^ |$|™|®|•|…", "" , string)
        string = re.sub("♀", "female" , string)
        string = re.sub("♂", "male" , string)
        string = unidecode(string)
        string = string.strip()
        sentences = nltk_tokenize.sent_tokenize(string)
        return sentences
    
    
    def abstract2NERinputs(self, abstract:str) -> List[NERInput]:
        guid_index = 0
        sentences = self.str2sents(abstract)
        ner_inputs = [NERInput(str(guid), 
                      nltk_tokenize.word_tokenize(sent), 
                      ["O" for i in range(len(nltk_tokenize.word_tokenize(sent)))]) 
                          for guid, sent in enumerate(sentences)]
        return ner_inputs
    
    def convert_NERinputs_to_features(self, 
        ner_inputs: List[NERInput],
        model_config: BertConfig,
        bert_tokenizer: BertTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ) -> List[InputFeatures]:

        label2id = model_config.label2id
        pad_token = model_config.pad_token_id
        max_seq_length = model_config.max_position_embeddings

        features = []

        for (input_index, ner_input) in enumerate(ner_inputs):
            tokens = []
            label_ids = []
            for word, label in zip(ner_input.words, ner_input.labels):
                word_tokens = bert_tokenizer.tokenize(word)

                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    label_ids.extend([label2id[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = bert_tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            if "token_type_ids" not in bert_tokenizer.model_input_names:
                segment_ids = None

            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids
                )
            )
        return features

# Subsection: Actual NER Pipeline
class NER_Pipeline:
    def __init__(self, name_or_path_to_model_folder:str = "ncats/EpiExtract4GARD-v2"):
        self.bert_tokenizer = BertTokenizer.from_pretrained(name_or_path_to_model_folder)
        #no need for model variable because trainer wraps model and has more functions
        #model = AutoModelForTokenClassification.from_pretrained(name_or_path_to_model_folder)
        self.config = BertConfig.from_pretrained(name_or_path_to_model_folder)
        self.labels = {re.sub(".-","",label) for label in self.config.label2id.keys() if label != "O"}
        self.trainer = Trainer(model=AutoModelForTokenClassification.from_pretrained(name_or_path_to_model_folder))
    
    def __str__(self):
        return "Instantiation: pipe = NER_Pipeline(name_or_path_to_model_folder)"+"\n Calling: output_dict = pipe(text)"
    
    #Custom pipeline by WKariampuzha @NCATS (not Huggingface/Google/NVIDIA copyright)
    def __call__(self, text:str, rd_identify:Union[GARD_Search,None] = None):
        output_dict = {label:[] for label in self.labels}
        
        dataset = NerDataset(text, self.bert_tokenizer, self.config)
        predictions, label_ids, _ = self.trainer.predict(dataset)
        preds_list, _ = self.align_predictions(predictions, label_ids)
        #dataset.ner_inputs.labels = preds_list
        for ner_input, sent_pred_list in zip(dataset.ner_inputs, preds_list):
            ner_input.labels = sent_pred_list
        
        for sentence in dataset.ner_inputs:
            entity = []
            for idx, (current, nxt) in enumerate(pairwise(sentence.labels)):    
                #Main concatenation algorithm
                '''
                Accounts for all variations of 
                current = ['O','B-Tag`','I-Tag`']
                nxt = ["O","B-Tag`","I-Tag`","B-Tag``","I-Tag``"]
                and accounts for the final case
                '''
                if current != "O":
                    current_ib, current_tag = self.get_tag(current)
                    if nxt =="O":
                        #add word at idx
                        entity.append(sentence.words[idx])
                        output_dict[current_tag].append(' '.join(entity))
                        entity.clear()
                    else:
                        nxt_ib, nxt_tag = self.get_tag(nxt)
                        if nxt_tag == current_tag:
                            if nxt_ib =="B":
                                entity.append(sentence.words[idx])
                                output_dict[current_tag].append(' '.join(entity))
                                entity.clear()
                            #Continued "I"
                            else:
                                entity.append(sentence.words[idx])
                        else:
                            entity.append(sentence.words[idx])
                            output_dict[current_tag].append(' '.join(entity))
                            entity.clear()
                            
                #last case
                if idx==len(sentence.labels)-2 and nxt!="O":
                    _, nxt_tag = self.get_tag(nxt)
                    entity.append(sentence.words[idx+1])
                    output_dict[nxt_tag].append(' '.join(entity))
                    entity.clear()
        
        if 'DIS' not in output_dict.keys() and rd_identify:
            output_dict['DIS'] = []
            output_dict['IDS'] = []
            for sentence in dataset.ner_inputs:
                diseases,ids = rd_identify(' '.join(sentence.words))
                output_dict['DIS']+=diseases
                output_dict['IDS']+=ids

        #Clean up Output Dict
        for entity, output in output_dict.items():
            if not output:
                output_dict[entity] = None
            elif entity !='STAT':
                #remove duplicates from list but keep ordering instead of using sets
                output = list(OrderedDict.fromkeys(output)) 
                output_dict[entity] = output

        if output_dict['EPI'] and output_dict['STAT']:
            return output_dict
    
    def align_predictions(self, predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(self.config.id2label[label_ids[i][j]])
                    preds_list[i].append(self.config.id2label[preds[i][j]])

        return preds_list, out_label_list
    
    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # It's not in B-, I- format
            # Default to I- for continuation.
            bi = "I"
            tag = entity_name
        return bi, tag

    
# Unattached function -- not a method
# move this to the NER_pipeline as a method??
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

## SECTION: PIPELINES
## This section combines all of the previous code into pipelines so that usage of these models and search functions are easy to implement in apps.

# Given a search term and max results to return, this will acquire PubMed IDs and Title+Abstracts and Classify them as epidemiological.
# results = search_term_extraction(search_term, maxResults, filering, GARD_dict, classify_model_vars)
#Returns a Pandas dataframe   
def search_term_classification(search_term:Union[int,str], maxResults:int, 
                               filtering:str, rd_identify:GARD_Search, #for abstract search & filtering 
                               epi_classify:Classify_Pipeline) -> pd.DataFrame: #for classification
    
    results = pd.DataFrame(columns=['PMID', 'ABSTRACT','EPI_PROB','IsEpi'])
    
    ##Check to see if search term maps to anything in the GARD dictionary, if so it pulls up all synonyms for the search
    search_term_list = rd_identify.autosearch(search_term)
    
    #Gather title+abstracts into a dictionary {pmid:abstract}
    pmid_abs = search_getAbs(search_term_list, maxResults, filtering)
    
    for pmid, abstract in pmid_abs.items():
        epi_prob, isEpi = epi_classify(abstract)
        result = {'PMID':pmid, 'ABSTRACT':abstract, 'EPI_PROB':epi_prob, 'IsEpi':isEpi}
        #Slow dataframe update
        results = results.append(result, ignore_index=True)
    
    return results.sort_values('EPI_PROB', ascending=False)

#Identical to search_term_classification, except it returns a JSON-compatible dictionary instead of a df
def API_search_classification(search_term:Union[int,str], maxResults:int, 
                              filtering:str, GARD_Search:GARD_Search, #for abstract search & filtering
                              epi_classify:Classify_Pipeline) -> Dict[str,str]: #for classification

    #Format of Output
    results = {'entries':[]} 
    
    ##Check to see if search term maps to anything in the GARD dictionary, if so it pulls up all synonyms for the search
    print('Inside `API_search_classification`. this is `search_term`:',search_term,type(search_term))
    search_term_list = GARD_Search.autosearch(search_term)
    
    #Gather title+abstracts into a dictionary {pmid:abstract}
    pmid_abs = search_getAbs(search_term_list, maxResults, filtering)
    
    for pmid, abstract in pmid_abs.items():
        epi_prob, isEpi = epi_classify(abstract)
        result = {'PMID':pmid, 'ABSTRACT':abstract, 'EPI_PROB':epi_prob, 'IsEpi':isEpi}
        results['entries'].append(result)
    
    #sort 
    results['entries'].sort(reverse=True, key=lambda x:x['EPI_PROB'])
    
    # float is not JSON serializable, so must convert all epi_probs to str
    # This returns a map object, which is not JSON serializable
    # results['entries'] = map(lambda entry:str(entry['EPI_PROB']),results['entries'])
    # so must convert floats to str the boring and slow way
    
    for entry in results['entries']:
        entry['EPI_PROB'] = str(entry['EPI_PROB'])
        
    return results

def API_PMID_classification(pmid:Union[int,str], epi_classify:Classify_Pipeline) ->  Dict[str,str]: 
    text = PMID_getAb(pmid)
    epi_prob, isEpi = epi_classify(text)
    return {'PMID':pmid,'ABSTRACT':text, 'EPI_PROB':str(epi_prob), 'IsEpi':isEpi}

def API_text_classification(text:str,epi_classify:Classify_Pipeline) ->  Dict[str,str]: 
    epi_prob, isEpi = epi_classify(text)
    return {'ABSTRACT':text, 'EPI_PROB':str(epi_prob), 'IsEpi':isEpi}

# Given a search term and max results to return, this will acquire PubMed IDs and Title+Abstracts and Classify them as epidemiological.
# It then extracts Epidemiologic Information[Disease GARD ID, Disease Name, Location, Epidemiologic Identifier, Epidemiologic Statistic] for each abstract
# results = search_term_extraction(search_term, maxResults, filering, NER_pipeline, extract_diseases, GARD_Search, Classify_Pipeline)
#Returns a Pandas dataframe                                                                                                          
def search_term_extraction(search_term:Union[int,str], maxResults:int, filtering:str, #for abstract search
                           epi_ner:NER_Pipeline, #for biobert extraction 
                           GARD_Search:GARD_Search, extract_diseases:bool, #for disease extraction
                           epi_classify:Classify_Pipeline) -> pd.DataFrame: #for classification
    
    
    #Format of Output
    ordered_labels = order_labels(epi_ner.labels)
    if extract_diseases:
        columns = ['PMID', 'ABSTRACT','EPI_PROB','IsEpi','IDS','DIS']+ordered_labels
    else:
        columns = ['PMID', 'ABSTRACT','EPI_PROB','IsEpi']+ordered_labels
    
    results = pd.DataFrame(columns=columns)
    
    ##Check to see if search term maps to anything in the GARD dictionary, if so it pulls up all synonyms for the search
    search_term_list = GARD_Search.autosearch(search_term)
    
    #Gather title+abstracts into a dictionary {pmid:abstract}
    pmid_abs = search_getAbs(search_term_list, maxResults, filtering)
    
    for pmid, abstract in pmid_abs.items():
        epi_prob, isEpi = epi_classify(abstract)
        if isEpi:
            if extract_diseases:
                extraction = epi_ner(abstract, GARD_Search)
            else:
                extraction = epi_ner(abstract)
                
            if extraction:
                extraction.update({'PMID':pmid, 'ABSTRACT':abstract, 'EPI_PROB':epi_prob, 'IsEpi':isEpi})
                #Slow dataframe update
                results = results.append(extraction, ignore_index=True)
    
    print(len(results),'abstracts classified as epidemiological.')
    return results.sort_values('EPI_PROB', ascending=False)
    
#Returns a Pandas dataframe 
def streamlit_extraction(search_term:Union[int,str], maxResults:int, filtering:str, #for abstract search
                           epi_ner:NER_Pipeline, #for biobert extraction 
                           GARD_Search:GARD_Search, extract_diseases:bool, #for disease extraction
                           epi_classify:Classify_Pipeline) -> pd.DataFrame: #for classification
   
    #Format of Output
    ordered_labels = order_labels(epi_ner.labels)
    if extract_diseases:
        columns = ['PMID', 'ABSTRACT','PROB_OF_EPI','IsEpi','IDS','DIS']+ordered_labels
    else:
        columns = ['PMID', 'ABSTRACT','PROB_OF_EPI','IsEpi']+ordered_labels
    
    results = pd.DataFrame(columns=columns)
    
    ##Check to see if search term maps to anything in the GARD dictionary, if so it pulls up all synonyms for the search
    search_term_list = GARD_Search.autosearch(search_term)
    if len(search_term_list)>1:
        st.write("SEARCH TERM MATCHED TO GARD DICTIONARY. SEARCHING FOR: "+ str(search_term_list))
    else:
        st.write("SEARCHING FOR: "+ str(search_term_list))
        
    #Gather title+abstracts into a dictionary {pmid:abstract}
    pmid_abs, sankey_initial = streamlit_getAbs(search_term_list, maxResults, filtering)
    
    if len(pmid_abs)==0:
        st.error('No results were gathered. Enter a new search term.')
        return None, None, None
    else:
        found, relevant = sankey_initial
        epidemiologic = 0
        i = 0
        my_bar = st.progress(i)
        percent_at_step = 100/len(pmid_abs)
        for pmid, abstract in pmid_abs.items():
            epi_prob, isEpi = epi_classify(abstract)
            if isEpi:
                if extract_diseases:
                    extraction = epi_ner(abstract, GARD_Search)
                else:
                    extraction = epi_ner(abstract)
                
                if extraction:
                    extraction.update({'PMID':pmid, 'ABSTRACT':abstract, 'PROB_OF_EPI':epi_prob, 'IsEpi':isEpi})
                    #Slow dataframe update
                    results = results.append(extraction, ignore_index=True)
                    epidemiologic+=1
            i+=1
            my_bar.progress(min(round(i*percent_at_step/100,1),1.0))
        
        st.write(len(results),'abstracts classified as epidemiological.')
        
        sankey_data = (found, relevant, epidemiologic)
        #Export the name and GARD ID to the ap for better integration on page.
        name = search_term_list[-1].capitalize()
        
        if search_term_list[-1] in GARD_Search.GARD_dict.keys():
            disease_gardID = (name, GARD_Search.GARD_dict[search_term_list[-1]])
        else:
            disease_gardID = (name, None)
        
        return results.sort_values('PROB_OF_EPI', ascending=False), sankey_data, disease_gardID

#Identical to search_term_extraction, except it returns a JSON-compatible dictionary instead of a df
def API_search_extraction(search_term:Union[int,str], maxResults:int, filtering:str, #for abstract search
                   epi_ner:NER_Pipeline, #for biobert extraction 
                   GARD_Search:GARD_Search, extract_diseases:bool, #for disease extraction
                   epi_classify:Classify_Pipeline) ->  Dict[str,str]: #for classification
                                                                                                                                             
    #Format of Output
    ordered_labels = order_labels(epi_ner.labels)
    if extract_diseases:
        json_output = ['PMID', 'ABSTRACT','EPI_PROB','IsEpi','IDS','DIS']+ordered_labels
    else:
        json_output = ['PMID', 'ABSTRACT','EPI_PROB','IsEpi']+ordered_labels
    
    results = {'entries':[]} 
    
    ##Check to see if search term maps to anything in the GARD dictionary, if so it pulls up all synonyms for the search
    search_term_list = GARD_Search.autosearch(search_term)
    
    #Gather title+abstracts into a dictionary {pmid:abstract}
    pmid_abs = search_getAbs(search_term_list, maxResults, filtering)
    
    for pmid, abstract in pmid_abs.items():
        epi_prob, isEpi = epi_classify(abstract)
        if isEpi:
            if extract_diseases:
                extraction = epi_ner(abstract, GARD_Search)
            else:
                extraction = epi_ner(abstract)
            if extraction:
                extraction.update({'PMID':pmid, 'ABSTRACT':abstract, 'EPI_PROB':epi_prob})
                extraction = OrderedDict([(term, extraction[term]) for term in json_output if term in extraction.keys()])
                results['entries'].append(extraction)
    
    #sort 
    results['entries'].sort(reverse=True, key=lambda x:x['EPI_PROB'])
    
    # float is not JSON serializable, so must convert all epi_probs to str
    # This returns a map object, which is not JSON serializable
    # results['entries'] = map(lambda entry:str(entry['EPI_PROB']),results['entries'])
    
    for entry in results['entries']:
        entry['EPI_PROB'] = str(entry['EPI_PROB'])
        
    return results

#Identical to search_term_extraction, except it returns a JSON-compatible dictionary instead of a df
def API_text_extraction(text:str, #Text to be extracted
                   epi_ner:NER_Pipeline, #for biobert extraction 
                   GARD_Search:GARD_Search, extract_diseases:bool, #for disease extraction
                   ) ->  Dict[str,str]:
                                                                                                           
    #Format of Output
    ordered_labels = order_labels(epi_ner.labels)
    if extract_diseases:
        json_output = ['ABSTRACT','IDS','DIS']+ordered_labels
    else:
        json_output = ['ABSTRACT']+ordered_labels
    
    extraction = dict()
    #Do the extraction
    if extract_diseases:
        extraction = epi_ner(text, GARD_Search)
    else:
        extraction = epi_ner(text)
    
    if extraction:
        #Re-order the dictionary into desired JSON output
        extraction = OrderedDict([(term, extraction[term]) for term in json_output if term in extraction.keys()])
    else:
        #This may return JSONs of different length than above
        extraction = OrderedDict([(term, []) for term in json_output])
        
    return extraction

def API_text_classification_extraction(text:str, #Text to be extracted
                           epi_ner:NER_Pipeline, #for biobert extraction 
                           GARD_Search:GARD_Search, extract_diseases:bool, #for disease extraction
                           epi_classify:Classify_Pipeline) ->  Dict[str,str]:

    #Format of Output
    ordered_labels = order_labels(epi_ner.labels)
    if extract_diseases:
        json_output = ['ABSTRACT','IsEpi','EPI_PROB','IDS','DIS']+ordered_labels
    else:
        json_output = ['ABSTRACT','IsEpi','EPI_PROB']+ordered_labels
    
    #Do the extraction
    if extract_diseases:
        extraction = epi_ner(text, GARD_Search)
    else:
        extraction = epi_ner(text)
    
    if extraction:
        #Add the epidemiology probability and result
        #Does not matter which order these are done in but doing classification after may save some time if there is no valid extraction
        epi_prob, isEpi = epi_classify(text)
        extraction.update({'EPI_PROB':str(epi_prob),'IsEpi':isEpi})
        
        #Re-order the dictionary into desired JSON output
        output = OrderedDict([(term, extraction[term]) for term in json_output if term in extraction.keys()])
    else:
        #This may return JSONs of different length than above
        output = OrderedDict([(term, []) for term in json_output])
        
    return output

## Section: Deprecated Functions
import requests
import xml.etree.ElementTree as ET

def search_Pubmed_API(searchterm_list:Union[List[str],str], maxResults:int) -> Dict[str,str]: #returns a dictionary of {pmids:abstracts} 
    print('search_Pubmed_API is DEPRECATED. UTILIZE search_NCBI_API for NCBI ENTREZ API results. Utilize search_getAbs for most comprehensive results.')
    return search_NCBI_API(searchterm_list, maxResults)

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
