import re

# 특수문자 제거 및 데이터 클리닝 함수
def clean_text(text):
    # 특수문자 제거 (필요시 추가적인 정규식 적용 가능)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # 여러 공백을 하나의 공백으로 변환
    text = re.sub(r'\s+', ' ', text).strip()
    return text
