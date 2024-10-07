import faiss

# FAISS 인덱스 저장
def save_faiss_index(index, file_name):
    faiss.write_index(index, file_name)

# FAISS 인덱스 로드
def load_faiss_index(file_name):
    index = faiss.read_index(file_name)
    return index
