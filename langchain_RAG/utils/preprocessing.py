def sliding_window(text, window_size, overlap):
    """
    슬라이딩 윈도우 기법을 사용하여 텍스트를 작은 청크로 나눔.
    :param text: 입력 텍스트
    :param window_size: 한 번에 처리할 텍스트의 길이 (문자 단위)
    :param overlap: 청크 간 겹치는 구간의 길이
    :return: 슬라이딩 윈도우로 나눠진 텍스트 청크 리스트
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + window_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += window_size - overlap  # 다음 윈도우 시작점 계산 (겹침 부분 고려)
    
    return chunks

# 예시 텍스트
text = "이것은 슬라이딩 윈도우 기법을 적용한 텍스트 전처리 예제입니다. 긴 문서를 처리할 때 사용합니다."

# 윈도우 크기와 겹침 구간 설정
window_size = 200  # 한 번에 처리할 텍스트 길이
overlap = 5  # 겹치는 구간

# 슬라이딩 윈도우 적용
chunks = sliding_window(text, window_size, overlap)

# 결과 출력
for i, chunk in enumerate(chunks):
    print(f"청크 {i+1}: {chunk}")
