# 디렉토리 구조 제안

1. app/: Flask 관련 설정 파일과 메인 애플리케이션 코드.

* app.py: Flask 메인 실행 파일
* error_handler.py: Flask의 에러 핸들러를 모아둔 파일
* document_retriever.py: 문서 검색 및 RAG 관련 로직 처리

2. services/: 주요 서비스 로직 (대화 기록, 메시지 처리, 데이터 처리 등).

* conversation_history.py: 대화 기록 저장 및 조회 관련 코드
* mongo_client.py: MongoDB 클라이언트 설정
* stream_handler.py: 스트리밍 처리 관련 코드
* nlp_handler.py: NLP 처리 로직 (OpenAI API 호출 등)

3. retrievers/: 문서 검색 로직 관련 모듈

* faiss_retriever.py: FAISS 관련 코드
* bm25_retriever.py: BM25 관련 코드

4. utils/: 다양한 유틸리티 함수들.

* utils.py: Flask 요청 처리, 공통 기능
* config.py: 환경 변수 및 설정 파일 관리

5. logging/: 로깅 설정 및 로그 파일 저장.

* logger_config.py: 로깅 설정 관련 코드
* error_log.log: 에러 로그 저장 파일

6. data/: MD 파일 및 기타 데이터 파일 관리

* ktb_data_09.md: 데이터를 처리할 파일
* test.md: 테스트용 데이터

7. scripts/: Jenkins나 Docker 관련 스크립트.

* Jenkinsfile: Jenkins 파이프라인 스크립트
* Dockerfile: Docker 이미지 빌드 스크립트

## 트리형식 표현
```
project/
│
├── app/
│   ├── app.py
│   ├── error_handler.py
│   └── document_retriever.py
│
├── services/
│   ├── conversation_history.py
│   ├── mongo_client.py
│   └── stream_handler.py
│
├── retrievers/
│   ├── faiss_retriever.py
│   └── bm25_retriever.py
│
├── utils/
│   ├── utils.py
│   ├── config.py
│   └── logger_config.py
│
├── logging/
│   ├── error_log.log
│   └── logger_config.py
│
├── data/
│   ├── retrievers/
│   │   ├── faiss_index.faiss
│   │   └── faiss_index.pkl
│   ├── ktb_data_09.md
│   └── test.md
│
├── scripts/
│   ├── Jenkinsfile
│   └── Dockerfile
│
├── venv/  # 가상환경
├── .env
├── .gitignore
└── requirements.txt
```
