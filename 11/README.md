# 디렉토리 구조 제안

1. app/: Flask 관련 설정 파일과 메인 애플리케이션 코드.

* app.py: Flask 메인 실행 파일
* nlp_routes.py: NLP 관련 엔드포인트를 처리하는 라우트 파일.
* routes.py: 애플리케이션의 일반적인 라우트 처리.

2. services/: 주요 서비스 로직 (대화 기록, 메시지 처리, 데이터 처리 등).

* chatgpt_service.py: ChatGPT API와 통신을 처리하고 관련 로직을 관리합니다.
* conversation_service.py: 대화 기록을 저장하고 불러오는 기능을 담당합니다.
* error_handler.py: 예외 및 오류에 대한 커스텀 에러 핸들러를 정의합니다.
* logging_service.py: 로그 작성 및 관리 로직을 처리합니다.
* mongo_client.py: MongoDB와의 연결을 관리하며, 데이터베이스 작업을 처리합니다.
* nlp_service.py: 일반적인 NLP 처리를 담당하는 서비스입니다.
* qa_chain_service.py: RAG(문서 검색 및 답변 생성)를 이용한 질문 응답 체인 관리 서비스입니다.
* retriever_service.py: FAISS 및 BM25 검색기를 사용한 문서 검색 로직을 처리합니다.
* weather_service.py: 날씨 API와의 통합을 관리하는 서비스입니다.

3. utils/: 다양한 유틸리티 함수들.

* document_loader.py: Markdown 또는 기타 유형의 문서를 로드하여 검색을 위해 처리하는 기능.
* request_handler.py: 요청 데이터 유효성 검사 및 처리.
* request_response_service.py: 요청-응답 주기를 관리하고, 스트림 처리 및 메시지 포맷팅 기능.

4. data/: MD 파일 및 기타 데이터 파일 관리

* ktb_data_09.md: 데이터를 처리할 파일
* test.md: 테스트용 데이터

5. scripts/: Jenkins나 Docker 관련 스크립트.

* Jenkinsfile: Jenkins 파이프라인 스크립트
* Dockerfile: Docker 이미지 빌드 스크립트

## 트리형식 표현
```
project/
│
├── app/   
│   ├── nlp_routes.py                 # NLP 관련 라우트 처리
│   └── routes.py                     # 일반적인 라우트 처리
│
├── services/                         # 주요 서비스 로직
│   ├── chatgpt_service.py            # ChatGPT API 통신 및 로직
│   ├── conversation_service.py       # 대화 기록 저장 및 불러오기
│   ├── error_handler.py              # 에러 핸들러 정의
│   ├── logging_service.py            # 로깅 설정 및 관리
│   ├── mongo_client.py               # MongoDB 연결 관리
│   ├── nlp_service.py                # NLP 관련 서비스 로직
│   ├── qa_chain_service.py           # RAG 기반 질문 응답 체인
│   ├── retriever_service.py          # FAISS/BM25 기반 문서 검색기
│   └── weather_service.py            # 날씨 API와 통합
│
├── utils/                            # 유틸리티 함수 및 공통 로직
│   ├── document_loader.py            # 문서 로더 및 검색 준비 로직
│   ├── request_handler.py            # 요청 데이터 처리 및 유효성 검사
│   └── request_response_service.py   # 요청-응답 스트림 및 포맷팅 처리
│
├── scripts/                          # 빌드 및 배포 스크립트
│   ├── Dockerfile                    # Docker 이미지 빌드 스크립트
│   └── Jenkinsfile                   # Jenkins 파이프라인 정의
│
├── data/                             # 데이터 파일 및 문서
│   ├── ktb_data_09.md                # MD 형식의 문서 데이터
│   └── test.md                       # 테스트 데이터
│
├── app.py                            # 메인 Flask 애플리케이션
├── .env                              # 환경 변수 파일
├── .gitignore                        # Git에서 무시할 파일 규칙
├── README.md                         # 프로젝트 설명서
└── requirements.txt                  # Python 패키지 의존성 파일

```

## 기타
이 외에 뭐 깃이그노어나 뭐 .env 등등등 그거는 안넣고 일단 메인함수들만 모듈화 해봤습니다. 어제 보여드린거랑은 좀 다른 모습인데 ellen코드스타일 최대한 살리는 방향으로 해봤어요!

## error_handler사용방법
```
from flask import Flask
from services.error_handler import register_error_handlers

app = Flask(__name__)
register_error_handlers(app)

if __name__ == '__main__':
    app.run()
```

