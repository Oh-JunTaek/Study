from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import os
import logging

from langchain_openai import OpenAIEmbeddings

def create_bm25_retriever(splitted_docs): 
    bm25_retriever = BM25Retriever.from_documents(splitted_docs)
    bm25_retriever.k = 1  # 검색 결과 개수를 1로 설정
    logging.info("BM25 retriever created")
    return bm25_retriever

def create_FAISS_retriever(splitted_docs, faiss_index_path="data/retrievers/faiss_index"): 
    embedding_function = OpenAIEmbeddings()
    index_faiss_file = os.path.join(faiss_index_path, "index.faiss")
    index_pkl_file = os.path.join(faiss_index_path, "index.pkl")

    if os.path.exists(index_faiss_file) and os.path.exists(index_pkl_file): 
        logging.info("FAISS index already exists")
        faiss_db = FAISS.load_local(faiss_index_path, embeddings=embedding_function)
    else:
        logging.info("Creating new FAISS index")
        faiss_db = FAISS.from_documents(splitted_docs, embedding=embedding_function)
        faiss_db.save_local(faiss_index_path)  # 인덱스를 로컬에 저장

    faiss_retriever = faiss_db.as_retriever(search_kwargs={"score_threshold": 0.7})
    return faiss_retriever, faiss_db

def create_ensemble_retriever(retrievers): 
    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers,
        weights=[0.7, 0.3],
    )
    logging.info("Ensemble retriever created.")
    return ensemble_retriever
