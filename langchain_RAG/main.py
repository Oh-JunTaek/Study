import os
from rag import rag_process, search_similar_documents, load_faiss_index
from models.llama_model import get_llama_model  # llama_model.py에서 함수 가져오기
from dotenv import load_dotenv

load_dotenv()
# 인덱스 파일 경로 설정
index_file = "data/preprocessed/faiss_index.index"

# 메인 함수
if __name__ == "__main__":
    # 1. FAISS 인덱스 파일이 있는지 확인
    if not os.path.exists(index_file):
        print("데이터 전처리작업을 먼저 진행해주세요")
    else:
        # FAISS 인덱스 로드
        index = load_faiss_index(index_file)

        # 2. 지속적인 대화 루프
        while True:
            # 사용자 인풋 받기
            prompt = input("Enter your prompt (or 'exit' to quit): ")

            # 'exit'를 입력하면 대화 종료
            if prompt.lower() == 'exit':
                print("대화를 종료합니다.")
                break

            # 3. 문서에서 RAG 검색 시도
            similar_docs = search_similar_documents(prompt, index, get_llama_model)
            
            # 4. RAG 결과가 있는지 확인
            if similar_docs.size > 0:
                # RAG 기반 응답 생성
                response = get_llama_model(prompt, rag_based=True)
            else:
                # 일반 LLM 기반 응답 생성
                response = get_llama_model(prompt, rag_based=False)
            
            # 결과 출력
            print(response)