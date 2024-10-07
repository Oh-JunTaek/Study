import faiss
from utils.query_vectorizer import vectorize_query
import numpy as np

# FAISS 인덱스 로드
def load_faiss_index(file_name):
    index = faiss.read_index(file_name)
    return index

# FAISS 인덱스 로드 중 로그 출력
def load_index_for_rag(index_file):
    print("FAISS 인덱스 로드 중...")
    index = load_faiss_index(index_file)
    print(f"FAISS 인덱스 로드 완료: {index.ntotal} 개의 벡터가 로드되었습니다.")
    return index

# Query(질문)를 벡터화하고, 유사한 문서 검색
def search_similar_documents(query, index, vectorizer):
    print("질문을 벡터화 중...")
    query_vector = vectorize_query(query)  # 벡터화 함수 호출

    # query_vector가 2차원 배열인지 다시 한 번 확인
    if len(query_vector.shape) != 2:
        raise ValueError("FAISS에 전달할 벡터는 2차원 배열이어야 합니다.")

    print("FAISS 인덱스에서 유사 문서 검색 중...")
    D, I = index.search(query_vector, k=5)  # 상위 5개의 유사 문서 검색
    return I  # 문서 인덱스 반환

# 전체 RAG 프로세스
def rag_process(query, index_file, vectorizer):
    # 1. FAISS 인덱스 로드
    index = load_index_for_rag(index_file)
    
    # 2. 유사 문서 검색
    similar_docs = search_similar_documents(query, index, vectorizer)
    
    # 검색된 문서를 바탕으로 결과 생성 (여기서 LLM 사용 가능)
    print(f"유사한 문서 인덱스: {similar_docs}")
