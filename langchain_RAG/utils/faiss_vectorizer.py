from langchain_openai import OpenAIEmbeddings
import numpy as np

# 텍스트 청크를 벡터로 변환하는 함수
def vectorize_chunks(chunks):
    # OpenAI Embeddings 사용
    embeddings = OpenAIEmbeddings()
    
    # 벡터화 진행
    chunk_vectors = embeddings.embed_documents(chunks)
    
    # 리스트를 numpy 배열로 변환
    chunk_vectors = np.array(chunk_vectors)
    
    return chunk_vectors
