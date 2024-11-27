import torch

# 모델 경로
model_path = "C:/Users/dev/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth"

# 모델을 CPU로 매핑하여 로드
model = torch.load(model_path, map_location=torch.device('cpu'))

# 간단한 텍스트 생성 함수 (모델이 어떻게 사용되는지에 따라 수정 필요)
def generate_text(input_text):
    # 입력 텍스트 처리 (모델 구조에 따라 다를 수 있음)
    # 현재는 가정된 예시 코드
    input_tensor = torch.tensor([1])  # 임시 입력 (실제 토큰화를 적용해야 함)
    with torch.no_grad():
        output = model(input_tensor)
    return output

# 텍스트 생성 테스트
response = generate_text("안녕하세요, 모델을 테스트합니다.")
print("모델 응답:", response)
