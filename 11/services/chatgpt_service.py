from conversation_history import save_conversation, history
from openai import OpenAIError
from flask import Response
import time, logging
import openai

# 환경 변수에서 API 키를 로드
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def stream_chatgpt(system_prompt, user_prompt, user_id, chat_id):
    messages = [{"role": "system", "content": system_prompt + "\n 정보를 일반 텍스트로 작성해 주세요. 굵게 표시하지 말고, 특수 형식 없이 일반 텍스트로만 작성해 주세요."},
                {"role": "user", "content": user_prompt}]
    
    if user_id and chat_id:
        conv_history = history(user_id, chat_id, limit=2)
        for conv in conv_history:
            messages.append({"role": conv['role'], "content": conv['text']})

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.0,
            max_tokens=500, stream=True
        )
        def event_stream():
            result_txt = ''
            for chunk in response:
                text = chunk.choices[0].delta.content
                if text:
                    result_txt += text
                    yield f"data: {text}\n\n"
            save_conversation(user_id, chat_id, "system", result_txt)

        return Response(event_stream(), mimetype='text/event-stream')
    except OpenAIError as e:
        logging.error(f"Error while calling chatGPT API function: {str(e)}")
        raise e

def text_chatgpt(system_prompt, user_prompt):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt + "\n 정보를 일반 텍스트로 작성해 주세요. 굵게 표시하지 말고, 특수 형식 없이 일반 텍스트로만 작성해 주세요."},
                      {"role": "user", "content": user_prompt}],
            temperature=0.0, max_tokens=500, stream=False
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        logging.error(f"Error while calling chatGPT API function: {str(e)}")
        raise e
