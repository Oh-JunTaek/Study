import logging
import os
from flask import Flask
from flask_cors import CORS
from routes.nlp_routes import nlp_blueprint
from services.error_handler import register_error_handlers
from services.mongo_client import get_mongo_client

# 플라스크 앱 정의
app = Flask(__name__)
CORS(app)

# 에러 핸들러 등록
register_error_handlers(app)

# 라우트 등록
app.register_blueprint(nlp_blueprint)

# 로깅 설정
logging.basicConfig(
    filename='./logging/error_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# MongoDB 클라이언트 설정
client, db, collection = get_mongo_client()

if __name__ == '__main__':
    print("Flask app is running")
    app.run(port=5001, debug=True)
