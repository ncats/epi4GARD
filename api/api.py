from flask import Flask, request, jsonify, Blueprint, render_template
#from flask_restplus import Api, Resource, fields, reqparse
#from flask_cors import CORS, os

app = Flask(__name__)
app.config["DEBUG"] = True

diseases = [
	{'id': 6,
	 'name': 'Acromesomelic dysplasia',
	 'synonyms': 'Acromesomelic dwarfism'},

	{'id': 7,
	 'name': 'Acromicric dysplasia',
	 'synonyms': 'Acromicric skeletal dysplasia'},

	{'id': 8,
	 'name': 'Agnosia',
	 'synonyms': 'Primary visual agnosia, Monomodal visual amnesia, Visual amnesia'},
]

@app.route("/", methods=["GET"])
def home():
	return render_template('index.html')

@app.route("/epiapi/v1/resources/diseases", methods=["GET"])
def epiapi():
	return jsonify(diseases)

@app.route("/epiapi_id", methods=["GET"])
def epiapi_id():
	
	# check if ID was given; if yes, assign to the id varioable
	# if no, return an error message
	if 'id' in request.args:
		id = int(request.args['id'])
	else:
		return "Error: No valid input provided."

	results = []

	# check and match result that correspond to input id
	for disease in diseases:
		if disease['id']==id:
			results.append(disease)
	
	return jsonify(results)

app.run()




