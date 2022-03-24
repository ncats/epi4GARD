from flask import Blueprint, request

blueprint = Blueprint('api', __name__, url_prefix='/basic_api')

@blueprint.route('/entities', methods=['GET', 'POST'])
def entities():
    if request.method == "GET":
        return {
            'method': request.method
        }
    if request.method == "POST":
        return {
            'message': 'This endpoint should create an entity',
            'method': request.method,
            'body': request.json
        }

@blueprint.route('/entities/<int:entity_id>', methods=['GET', 'PUT', 'DELETE'])
def entity(entity_id):
    if request.method == "GET":
        return {
            'id': entity_id
            'method': request.method
        }
    if request.method == "PUT":
        return {
            'id': entity_id,
            'method': request.method,
            'body': request.json
        }
    if request.method == "DELETE":
        return {
            'id': entity_id,
            'method': request.method
        }