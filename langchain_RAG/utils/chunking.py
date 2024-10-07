import os
from preprocessing import sliding_window

# data 폴더의 모든 파일을 처리
def load_and_process_files(data_directory, window_size, overlap):
    # data 디렉토리에서 파일 목록 불러오기
    files = [f for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]
    
    all_chunks = []  # 모든 파일에서 나온 텍스트 청크 저장
    
    for file in files:
        file_path = os.path.join(data_directory, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 슬라이딩 윈도우 적용
        chunks = sliding_window(text, window_size, overlap)
        all_chunks.extend(chunks)  # 모든 청크를 리스트에 추가
    
    return all_chunks

# 예시: data 폴더 경로와 슬라이딩 윈도우 파라미터 설정
data_directory = './data'  # 루트 폴더의 data 디렉토리
window_size = 200  # 청크의 길이 설정 (문자 수)
overlap = 50  # 청크 간 겹치는 길이 설정

# 파일 처리 및 슬라이딩 윈도우 적용
all_chunks = load_and_process_files(data_directory, window_size, overlap)

# 청크 결과 출력
for i, chunk in enumerate(all_chunks):
    print(f"청크 {i+1}: {chunk}\n")
