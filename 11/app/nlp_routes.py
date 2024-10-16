from flask import Blueprint, jsonify, Response, stream_with_context
from utils import get_request_data, generate_response_stream, text_chatgpt

nlp_blueprint = Blueprint('nlp', __name__)

@nlp_blueprint.route("/nlp-api/conv", methods=['POST'])
def llm():
    params = get_request_data()
    user_input, user_id, chat_id = params['content'], params['user_id'], params['chat_id']
    response_generator = generate_response_stream(user_id, chat_id, user_input)
    return Response(stream_with_context(response_generator), mimetype='text/event-stream')

@nlp_blueprint.route("/nlp-api/title", methods=['POST'])
def make_title():
    params = get_request_data(title=True)
    user_input = params['content']
    system_prompt = """넌 대화 타이틀을 만드는 역할이야. 챗봇에서 사용자의 첫 번째 메시지를 기반으로 해당 대화의 제목을 요약해줘."""
    title = text_chatgpt(system_prompt, user_input)
    
    if not title:
        return jsonify({"error": "죄송해요. 챗 지피티가 제목을 제대로 가져오지 못했어요."})
    
    title = title.strip('"')
    return jsonify({"title": title})
