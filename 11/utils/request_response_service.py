from flask import request
from werkzeug.exceptions import BadRequest
from conversation_history import save_conversation, history
from document_retriever import my_retriever
import json

def get_request_data(title=None):
    params = request.get_json()
    if not params:
        raise BadRequest("No request body")
    
    if 'content' not in params or not isinstance(params['content'], str) or not params['content'].strip():
        raise BadRequest("No content field in request body, empty value or invalid value")
    
    if title is None:
        if 'user_id' not in params or not params['user_id'] or not isinstance(params['user_id'], int):
            raise BadRequest("No user_id field in request body, empty value or invalid value")
        if 'chat_id' not in params or not params['chat_id'] or not isinstance(params['chat_id'], int):
            raise BadRequest("No chat_id field in request body, empty value or invalid value")

    return params

def generate_response_stream(user_id, chat_id, user_input):
    my_history = history(user_id, chat_id)
    save_conversation(user_id, chat_id, "user", user_input)
    answer_text = ''
    for chunk in retriever.stream({"question": user_input, "chat_history": my_history}):
        answer_text += chunk
        chunk_json = json.dumps({"text": chunk}, ensure_ascii=False)
        yield f"data: {chunk_json}\n\n"
    
    save_conversation(user_id, chat_id, "assistant", answer_text)
