from flask import Flask, request, jsonify, Blueprint, render_template
from flask_restplus import Api, Resource, fields, reqparse
from flask_cors import CORS, os
from extract_abs import  search_term_extraction as extract

app = Flask(__name__)
api = Api(app, version='1.0', title='EPIforGARD API', validate=False)

app.register_blueprint(basic_endpoint)

model_input = api.model('User input:', {"PMID": fields.Integer(maximum=10)})

@app.route('/epiapi', methods=['GET'])

class EPIAPI(Resource):
	@api.response(model_input)
	@api.expect(model_input)
	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('PMID', type=int)
		args=parser.parse_args()
		inp = int(args["PMID"])
		result = extract(inp)
		return jsonify({"primes": result})


if __name__ == '__main__':
	app.run()
