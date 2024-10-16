from flask import request
from werkzeug.exceptions import BadRequest

def get_request_data(title=None):
    params = request.get_json()
    if not params:
        raise BadRequest("No request body")
    if 'content' not in params or not isinstance(params['content'], str) or not params['content'].strip():
        raise BadRequest("Invalid or missing content field")
    if title is None:
        if 'user_id' not in params or not isinstance(params['user_id'], int):
            raise BadRequest("Missing or invalid user_id")
        if 'chat_id' not in params or not isinstance(params['chat_id'], int):
            raise BadRequest("Missing or invalid chat_id")
    return params
