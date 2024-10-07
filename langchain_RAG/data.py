import chardet  # chardet를 추가하여 인코딩을 감지
from utils.preprocessing import sliding_window
from utils.faiss_vectorizer import vectorize_chunks
from utils.indexing import save_faiss_index, load_faiss_index
from utils.pdf_to_text import read_pdf
from utils.clean_text import clean_text
import faiss
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 인코딩 감지 후 파일 읽기
def read_file_with_detected_encoding(file_path):
    if file_path.endswith(".pdf"):  # PDF 파일 처리
        return read_pdf(file_path)
    
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()

# 데이터 로드 및 전처리 (클리닝 + 슬라이딩 윈도우)
def preprocess_text(data_directory, window_size, overlap):
    print("1단계: 데이터 전처리 시작...")
    
    chunks = []
    files = [f for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]
    
    for file in files:
        file_path = os.path.join(data_directory, file)
        
        try:
            # 파일 읽기 (인코딩 감지 사용)
            text = read_file_with_detected_encoding(file_path)
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {file_path}, 오류: {e}")
            continue
        
        # 1. 데이터 클리닝
        cleaned_text = clean_text(text)
        
        # 2. 슬라이딩 윈도우 적용
        file_chunks = sliding_window(cleaned_text, window_size, overlap)
        chunks.extend(file_chunks)
    
    print(f"전체 청크 개수: {len(chunks)}")
    return chunks

# 벡터화 단계
def vectorize_data(chunks):
    print("2단계: 벡터화 작업 시작...")
    # 벡터화 작업
    chunk_vectors = vectorize_chunks(chunks)
    print(f"벡터화 완료: 벡터 개수: {len(chunk_vectors)}")
    return chunk_vectors

# FAISS 인덱스 저장 및 로드 처리
def save_and_load_index(chunk_vectors, index_file):
    print("3단계: FAISS 인덱스 생성 및 저장 중...")
    # FAISS 인덱스 생성
    dimension = len(chunk_vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_vectors)
    
    # 인덱스 파일로 저장
    save_faiss_index(index, index_file)
    print(f"FAISS 인덱스 저장 완료: {index_file}")
    
    # 저장된 인덱스를 로드
    loaded_index = load_faiss_index(index_file)
    print(f"저장된 인덱스에서 벡터 개수: {loaded_index.ntotal}")

# 전체 파이프라인 실행
def process_data(data_directory, window_size, overlap, index_file):
    print("데이터 전처리 파이프라인 시작...")

    # 1. 전처리 (클리닝 + 슬라이딩 윈도우)
    chunks = preprocess_text(data_directory, window_size, overlap)
    
    # 2. 벡터화
    chunk_vectors = vectorize_data(chunks)
    
    # 3. 인덱스 저장 및 로드
    save_and_load_index(chunk_vectors, index_file)
    
    print("데이터 전처리 파이프라인 완료.")

# 파일 경로 및 파라미터 설정
data_directory = './data'
window_size = 200
overlap = 50
index_file = "data/preprocessed/faiss_index.index"

if __name__ == "__main__":
    process_data(data_directory, window_size, overlap, index_file)
