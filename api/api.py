from flask import Flask, request, jsonify, Blueprint, render_template
#import nltk
#nltk.download()
#from test import search_getAbs3
from classify_abs import search_getAbs
from extract_abs import PMID_extraction, autosearch, search_term_extraction
#from flask_restplus import Api, Resource, fields, reqparse
#from flask_cors import CORS, os

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route("/")
def index():
	return "This is Epi4Gard API."

""" 
# Testing script:
@app.route("/searchPMid/<pmid>")
def searchPMid(pmid):
	search_getAbs2(pmid)
	return("Testing:" + pmid)
	#return jsonify(search_getAbs)

@app.route("/searchPMid/<pmid>/<result>")
def searchPMid(pmid, result):
	search_getAbs3(pmid, result)
	return("Testing:" + pmid + result)
"""

@app.route("/searchPMid/<searchterm_list>/<maxResults>/<filtering>")
def searchPMid(searchterm_list, maxResults, filtering):
	search_getAbs(searchterm_list=[], maxResults=int, filtering=str)
	return(searchterm_list + ", " + maxResults + ", " + filtering)

@app.route("/extractPMid/<pmid>/<labels>/<GARD_dict>/max_length")
def extractPMid(pmid, NER_pipeline, labels, GARD_dict, max_length):
	PMID_extraction(pmid, NER_pipeline, labels, GARD_dict, max_length)
	return(pmid + "," + labels + "," + GARD_dict)
	#return jsonify(PMID_extraction.output_dict)

@app.route("/search/<searchterm>/<GARD_dict>")
def search(searchterm, GARD_dict):
	autosearch()
	return(searchterm + "," + GARD_dict)

@app.route("/search_extract/<searchterm>/<maxResults>/<filtering>/<NER_pipeline>/<labels>/<extract_diseases>/<GARD_dict>/<max_length>/<classify_model_vars>")
def search_extract(search_term, maxResults, filering, 
				   NER_pipeline, labels, extract_diseases, 
				   GARD_dict, max_length, classify_model_vars):
	search_term_extraction()
	return(search_term + "," + maxResults + "," + filering + "," 
	+ NER_pipeline + "," + labels + "," + extract_diseases + "," 
	+ GARD_dict + "," + max_length + "," + classify_model_vars)
	#return "Results of the search: "+str(search_term_extraction)



app.run()
