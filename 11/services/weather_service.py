from get_weather import get_weather_info

def handle_weather_topic(user_input, user_id, chat_id):
    weather_info = get_weather_info()
    system_prompt = f"You are a helpful assistant, and you will kindly answer questions about current weather. 한국어로 대답해야 해. 현재 날씨 정보는 다음과 같아. {weather_info}. 이 날씨 정보를 다 출력할 필요는 없고, 주어진 질문에 필요한 답만 해줘."
    return stream_chatgpt(system_prompt, user_input, user_id, chat_id)
