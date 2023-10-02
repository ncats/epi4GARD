# coding=utf-8
##             PUBLIC DOMAIN NOTICE
##     National Center for Advancing Translational Sciences

## This software/database is a "United States Government Work" under the terms of the United States Copyright Act. It was written as part of the author's official duties as United States Government employee and thus cannot be copyrighted. This software is freely available to the public for use. The National Center for Advancing Translational Science (NCATS) and the U.S. Government have not placed any restriction on its use or reproduction.  Although all reasonable efforts have been taken to ensure the accuracy and reliability of the software and data, the NCATS and the U.S. Government do not and cannot warrant the performance or results that may be obtained by using this software or data. The NCATS and the U.S.  Government disclaim all warranties, express or implied, including warranties of performance, merchantability or fitness for any particular purpose.  Please cite the authors in any work or product based on this material.

# Written by William Kariampuzha @ NIH/NCATS.


#Allows different inputs/outputs to be hinted in function definitions
from typing import Union, Optional
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--root_path", dest="root_path", help="root path of epi api [DEFAULT = /api/epi]", default="/api/epi")
parser.add_argument("-u", "--openapi_url", dest="openapi_url", help="OpenAPI url [DEFAULT = /api/epi/openapi.json]", default="/api/epi/openapi.json")
args = parser.parse_args()

#Epi Pipelines used
from epi_pipeline import (
    # These are the pipeline objects
    NER_Pipeline,
    GARD_Search,
    Classify_Pipeline,
    #Gathers abstracts from two APIs with a rare disease/GARD ID input term and some autosearching on the backend
    search_getAbs,
    # all of these entitled API return JSON-compatible dictionary
    API_search_classification,
    API_search_extraction,
    API_PMID_classification,
    #These run the extraction and classification models on free text
    API_text_classification,
    API_text_extraction,
    API_text_classification_extraction,
    )
# Path intro & parameters here: https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/
# Query intro & parameters here: https://fastapi.tiangolo.com/tutorial/query-params-str-validations/
# They are basically the same functions but queries are optional, paths are required
from fastapi import FastAPI, Path, Query
#See below for how these are used
from enum import Enum
from pydantic import BaseModel

### CUSTOMIZATION OF SWAGGER UI
## See here for how to customize Swagger UI docs: https://fastapi.tiangolo.com/tutorial/metadata/#metadata-for-api
description="""
<p align="center">
    <img src="https://github.com/ncats/epi4GARD/raw/master/ncats.png" alt="National Center for Advancing Translational Sciences Logo" width=600>
    <br>
    <br>
    <img src="https://github.com/ncats/epi4GARD/raw/master/Logo_GARD_fullres.png" alt="NIH Genetic and Rare Diseases Information Center Logo" width=500>
</p>

This API was developed by the **[National Center for Advancing Translational Sciences (NCATS)](https://ncats.nih.gov/)** for the National Institutes of Health (NIH) **[Genetic and Rare Diseases Information Center (GARD)](https://rarediseases.info.nih.gov/)**.

It allows one to _gather_ abstracts for a rare disease (query any rare disease name, synonym, or GARD ID) from two APIs, [_classify_ those abstracts as epidemiologic](https://pubmed.ncbi.nlm.nih.gov/34457147/), and _extract_ epidemiology information from them.

A **full list of rare diseases** tracked by the NIH Genetic and Rare Diseases Information Center can be found [here](https://rarediseases.info.nih.gov/diseases/browse-by-first-letter).


### [Interactive User Interface](https://huggingface.co/spaces/ncats/EpiPipeline4RD) &nbsp; &nbsp; &nbsp; &nbsp; [Example POST calls](https://colab.research.google.com/drive/1QODYn9g8jbqDZTs3eR60mMkMA9PBv59t?usp=sharing) &nbsp; &nbsp; &nbsp; &nbsp; [GitHub Repository](https://github.com/ncats/epi4GARD)

"""

# These are the descriptions for the Swagger UI docs
tags_metadata = [
    {
        "name": "root",
        "description": "Execute this to test the server.",
    },
    {
        "name": "get_rare_disease_abstracts",
        "description": """Query any rare disease name, synonym, or GARD ID. 
        <br>
        <br>
        Uses the NCATS GARD Knowledge Graph to gather PubMed abstracts from the NCBI and EBI APIs. It contains three filtering options to reduce the number of false positive abstracts returned from the APIs. 
        <br>
        <br>
        Returns JSON of form `{'PubMed ID':'Abstract'}`""",
    },
    {
        "name": "get_rare_disease_epidemiology_abstracts",
        "description": """Query any rare disease name, synonym, or GARD ID. 
        <br>
        <br>
        Gets abstracts from `Get Rare Disease Abstracts` then classifies them as epidemiological or not. 
        <br>
        <br>
        Returns JSON of form `{'entries':[{'PMID':pmid, 
                                           'ABSTRACT':abstract, 
                                           'EPI_PROB':epi_prob, 
                                           'IsEpi':isEpi}, 
                                               ...]}`
        """,
        "externalDocs": {
            "description": "Publication:",
            "url":"https://pubmed.ncbi.nlm.nih.gov/34457147/",
        },
    },
    {
        "name": "get_rare_disease_epidemiology_abstracts_from_PubMed_ID",
        "description": """Query any PubMed ID 
        <br>
        <br>
        Gets abstracts from PubMed then classifies them as epidemiological or not. 
        <br>
        <br>
        Returns JSON of form ```{'PMID':pmid, 
                               'ABSTRACT':abstract, 
                               'EPI_PROB':epi_prob, 
                               'IsEpi':isEpi}```
        """,
        "externalDocs": {
            "description": "Publication:",
            "url":"https://pubmed.ncbi.nlm.nih.gov/34457147/",
        },
    },
    {
        "name": "get_rare_disease_epidemiology_extraction",
        "description": """Query any rare disease name, synonym, or GARD ID. 
        <br>
        <br>
        Examples of rare diseases include [**Fellman syndrome**](https://rarediseases.info.nih.gov/diseases/1/gracile-syndrome), [**Classic Homocystinuria**](https://rarediseases.info.nih.gov/diseases/6667/classic-homocystinuria), [**7383**](https://rarediseases.info.nih.gov/diseases/7383/phenylketonuria), and [**GARD:0009941**](https://rarediseases.info.nih.gov/diseases/9941/fshmd1a).
        <br>
        <br>
        Execute one of the examples to see the format of the output.
        """,
        "externalDocs": {
            "description": "Also see User Interface:",
            "url":"https://huggingface.co/spaces/ncats/EpiPipeline4RD",
        },
    },
    {
        "name": "classify_text_as_epidemiological",
        "description": "Submit a POST request to classify a rare disease abstract as epidemiology or not",
        "externalDocs": {
            "description": "Publication:",
            "url":"https://pubmed.ncbi.nlm.nih.gov/34457147/",
        },
    },
    {
        "name": "extract_epidemiology_from_text",
        "description": "Submit a POST request with text for epidemiology information extraction",
        "externalDocs": {
            "description": "Example POST Request:",
            "url":"https://colab.research.google.com/drive/1QODYn9g8jbqDZTs3eR60mMkMA9PBv59t?usp=sharing",
        },
    },
    {
        "name": "classify_and_extract_epidemiology_from_text",
        "description": "Combines above two",
    },
    {
        "name": "get_batch_rare_disease_abstracts",
        "description": "Batch version of `Get Rare Disease Abstracts`",
    },
    {
        "name": "get_batch_rare_disease_epidemiology_abstracts",
        "description": "Batch version of `Get Rare Disease Epidemiology Abstracts`.",
        "externalDocs": {
            "description": "Publication:",
            "url": "https://pubmed.ncbi.nlm.nih.gov/34457147/",
        },
    },
    {
        "name": "get_batch_rare_disease_epidemiology_extraction",
        "description": "Batch version of `Get Rare Disease Epidemiology Extraction`.",
    },
]

# Change these common descriptions & functions all at once
search_term_desc = "The name of the rare disease or the GARD ID you want epidemiology data for."
max_results_desc = "Maximum Number of Abstracts Returned."
filtering_desc = "Type of Abstract Filtering to reduce false positives."
extract_diseases_desc = "Extract Rare Diseases from Text Using GARD Dictionary."

search_term_fx = Path(description=search_term_desc)
max_results_fx = Query(default = 50, 
                       description= max_results_desc, 
                       gt=0, lt=1000)
filtering_fx = Query(default = 'strict', 
                     description=filtering_desc)
extract_diseases_fx = Query(default = False, 
                            description=extract_diseases_desc)

### LOAD THE APP & PIPELINES
app = FastAPI(
    title="EpiPipeline4RD API",
    version="1.1",
    description=description,
    license_info={"name": "National Center for Advancing Translational Sciences License",
                  "url": "https://github.com/ncats/epi4GARD/blob/master/LICENSE"},
    openapi_tags=tags_metadata,
    root_path=args.root_path,                       # needed for nginx proxy setting
    openapi_url=args.openapi_url         # needed for docs when using a proxy
    )
# Pipelines
rd_identify = GARD_Search()
epi_classify = Classify_Pipeline()
epi_extract = NER_Pipeline()


# Create Filtering Class
## Need to predefine types of filtering that we will accept
## See here: https://fastapi.tiangolo.com/tutorial/path-params/#predefined-values
class FilteringType(str, Enum):
    none = 'none'
    lenient = 'lenient'
    strict = 'strict'

# Create Text Classification/Extraction Object
## See here: https://fastapi.tiangolo.com/tutorial/body/
class Text(BaseModel):
    text: str = Path(description="The text you want to extract or classify")
    extract_diseases: bool = extract_diseases_fx

# Create Batch Object
class Batch(BaseModel):
    rd_list: list = Path(description="The names of the rare disease or the GARD ID you want abstracts for, separated by semicolons.")
    max_results:int = max_results_fx
    filtering:FilteringType = filtering_fx
    #extract_diseases is only used for the batch extraction, it does not need to be specified for other batch queries.
    extract_diseases:bool = extract_diseases_fx

## Unfinished/Unneeded code
# Define the return types ? 
class Classification_JSON(BaseModel):
    PMID: Optional[str]
    ABSTRACT : str
    EPI_PROB : str 
    IsEpi : bool
# These would be these labels for the search extraction return. not sure if it is needed
#['PMID', 'ABSTRACT','PROB_OF_EPI','IsEpi','IDS','DIS']+ordered_labels (['DIS','ABRV','EPI','STAT','LOC','DATE','SEX','ETHN'])


def validate_filtering(filtering:str) -> str:
    if filtering == FilteringType.none:
        filtering = 'none'
    elif filtering == FilteringType.lenient:
        filtering = 'lenient'
    elif filtering == FilteringType.strict:
        filtering = 'strict'
    else:
        print(filtering)
        raise ValueError("Filtering must be either 'strict','lenient', or 'none'.")
    return filtering

## Start the app 
# All of these return JSONs

@app.get("/", tags=["root"])
async def root():
    return {"message": "Epidemiology Information Extraction Pipeline for Rare Diseases. Built by the National Center for Advancing Translational Sciences"}

# Syntax of the functions:
## First term: HTTP method used e.g. GET (URL input), PUT (which requires JSON to be sent in the request)
## Middle terms: What is being gathering
## Last term: Input to the function e.g. Rare Disease term or GARD ID (RD), Raw Text, or Batch

# Uses optional arguments from here: https://fastapi.tiangolo.com/tutorial/query-params/
# Example query:
## rdip2.ncats.io:8000/getAbsRD/term=GARD:0000001?max_results=100&filtering=none
## Where '?' separates the required and optional inputs
## and '&' separates the optional inputs from each other

@app.get("/getAbsRD/term={search_term}", tags=["get_rare_disease_abstracts"])
async def get_rare_disease_abstracts(
    search_term:Union[str, int] = search_term_fx,
    max_results:int = max_results_fx,
    filtering:FilteringType = filtering_fx):
    
    searchterm_list = rd_identify.autosearch(search_term)
    
    filtering = validate_filtering(filtering)
    
    return search_getAbs(searchterm_list, max_results, filtering)

@app.get("/getEpiAbsPMID/term={PubMed_ID}", tags=["get_rare_disease_epidemiology_abstracts_from_PubMed_ID"])
async def get_rare_disease_epidemiology_abstracts(PubMed_ID):
    return API_PMID_classification(PubMed_ID, epi_classify)


@app.get("/getEpiAbsRD/term={search_term}", tags=["get_rare_disease_epidemiology_abstracts"])
async def get_rare_disease_epidemiology_abstracts(
    search_term:Union[str, int] = search_term_fx,
    max_results:int = max_results_fx,
    filtering:FilteringType = filtering_fx):
    
    filtering = validate_filtering(filtering)
    
    return API_search_classification(search_term, max_results, filtering, rd_identify, epi_classify)


@app.get("/getEpiAbsExtractRD/term={search_term}", tags=["get_rare_disease_epidemiology_extraction"])
async def get_rare_disease_epidemiology_extraction(
    search_term:Union[str, int] = search_term_fx,
    max_results:int = max_results_fx,
    filtering:FilteringType = filtering_fx,
    extract_diseases:bool = extract_diseases_fx):
    
    filtering = validate_filtering(filtering)
    
    return API_search_extraction(
                        search_term, max_results, filtering,
                        epi_extract, rd_identify, extract_diseases, epi_classify)

# Text Extraction (POST)
# Example query:
## import requests
## def classify_text(text, url="https://rdip2.ncats.io:8000/postEpiClassifyText"):
##     return requests.post(url, json={'text': text}).json() 
@app.post("/postEpiClassifyText/", tags=["classify_text_as_epidemiological"])
async def classify_text_as_epidemiological(i:Text): 
    return API_text_classification(i.text, epi_classify)

# Example query:
## import requests
## def query_plain(text, url="https://rdip2.ncats.io:8000/postEpiExtractText"):
##     return requests.post(url, json={'text': text,'extract_diseases':True}).json() # 'extract_diseases' does not need to be specified, default is False

@app.post("/postEpiExtractText/", tags=["extract_epidemiology_from_text"])
async def extract_epidemiology_from_text(i:Text): 
    return API_text_extraction(i.text, #Text to be extracted
                   epi_extract, #for biobert extraction 
                   rd_identify, i.extract_diseases, #for disease extraction
                   )

@app.post("/postEpiClassifyExtractText/", tags=["classify_and_extract_epidemiology_from_text"])
async def classify_and_extract_epidemiology_from_text(i:Text): 
    return API_text_classification_extraction(i.text, #Text to be extracted
                   epi_extract, #for biobert extraction 
                   rd_identify, i.extract_diseases, #for disease extraction
                   epi_classify)

#Batch Abstracts
# Example Python query:
## import requests
## def query_plain(text, url="https://rdip2.ncats.io:8000/post_RD_Abs_batch"):
##     return requests.post(url, json={'text': text,}).json()

@app.post("/postAbsBatch/", tags=["get_batch_rare_disease_abstracts"])
async def get_batch_rare_disease_abstracts(i:Batch):
    filtering = validate_filtering(i.filtering)
    output = []
    for rd in i.rd_list:
        searchterm_list = rd_identify.autosearch(rd)
        studies = search_getAbs(searchterm_list, i.max_results, filtering)
        output.append({"Disease": rd, "Studies": studies})
    return output

@app.post("/postEpiAbsBatch/", tags=["get_batch_rare_disease_epidemiology_abstracts"])
async def get_batch_rare_disease_epidemiology_abstracts(i:Batch):
    filtering = validate_filtering(i.filtering)
    output = []
    for rd in i.rd_list:
        studies = API_search_classification(search_term, maxResults, 
                              filtering, rd_identify, #for abstract search & filtering
                              epi_classify)
        output.append({"Disease": rd, "Studies": studies})
    return output

#Batch Epi Extraction
@app.post("/postEpiExtractBatch/", tags=["get_batch_rare_disease_epidemiology_extraction"])
async def get_batch_rare_disease_epidemiology_extraction(i:Batch):
    filtering = validate_filtering(i.filtering)
    output = []
    for rd in i.rd_list:
        extraction = API_search_extraction(
                        rd, i.max_results, filtering,
                        epi_extract, rd_identify, i.extract_diseases, epi_classify)        
        output.append({"Disease": rd, "Extraction": extraction})    
    return output
