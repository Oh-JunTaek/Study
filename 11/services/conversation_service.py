from datetime import datetime
import pytz

def save_conversation(collection, user_id, chat_id, role, text):
    """
    MongoDB에 대화 내용을 저장하는 함수
    """
    korea_tz = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(korea_tz).strftime('%Y-%m-%d %H:%M:%S')

    conversation = {
        "user_id": user_id,
        "chat_id": chat_id,
        "timestamp": current_time,
        "role": role,
        "text": text
    }
    collection.insert_one(conversation)

def history(collection, user_id, chat_id, limit=4):
    """
    MongoDB에서 대화 기록을 조회하는 함수
    """
    query = {"user_id": user_id, "chat_id": chat_id}
    conversations = collection.find(query).sort("timestamp", -1).limit(limit)
    return list(conversations)[::-1]
