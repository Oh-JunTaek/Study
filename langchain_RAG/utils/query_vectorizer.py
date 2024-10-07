from langchain_openai import OpenAIEmbeddings
import numpy as np

# 질의(Query)를 벡터로 변환
def vectorize_query(query):
    vectorizer = OpenAIEmbeddings()
    query_vector = vectorizer.embed_query(query)
    
    # query_vector가 리스트로 반환된 경우, numpy 배열로 변환
    query_vector = np.array(query_vector)
    
    # query_vector가 1차원 배열일 경우, 2차원 배열로 변환
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)  # 2차원 배열로 변환
    
    return query_vector  # 2차원 배열 반환
