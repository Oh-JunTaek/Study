from flask import request, Response
from services.conversation_service import stream_chatgpt, text_chatgpt
from utils.request_handler import get_request_data
from werkzeug.exceptions import BadRequest

def handle_else_topic(user_input, user_id, chat_id):
    system_prompt = "You are a helpful assistant. 사용자들은 한국어로 질문할 거고, 너도 한국어로 대답해야 돼."
    return stream_chatgpt(system_prompt, user_input, user_id, chat_id)

def topic_classification(user_input):
    system_prompt = """
            You are a classifier. Your task is to analyze '{user_input}'.
        - If '{user_input}' is a question about the asking weather, return 'WEATHER'.
        - If '{user_input}' is a question about public transportation routes involving a specific origin and destination, return 'TRANS'.
        - If '{user_input}' does not match either of the above cases, return 'ELSE'.
        """
    return text_chatgpt(system_prompt, user_input)

def extract_arrv_dest(user_input):
    system_prompt = """
            Your task is to identify the departure and destination from the user's input.
            Follow these guidelines:
            1. If either the departure or destination is ambiguous or unclear, mark it as unknown.
            2. If the input refers to the user's current location, mark it as current.
            3. If the input suggests the user's home location, mark it as home.
            4. Please return a dictionary formatted like this : {"from":departure, "to":destination}
            """
    return text_chatgpt(system_prompt, user_input)

def get_request_data(title=None):
    params = request.get_json()
    if not params:
        raise BadRequest("No request body")
    if 'content' not in params or not params['content'].strip():
        raise BadRequest("Invalid or missing content field")
    if title is None:
        if 'user_id' not in params or not isinstance(params['user_id'], int):
            raise BadRequest("Missing or invalid user_id")
        if 'chat_id' not in params or not isinstance(params['chat_id'], int):
            raise BadRequest("Missing or invalid chat_id")
    return params
