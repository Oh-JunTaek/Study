import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 및 토크나이저 경로 설정
model_path = "C:/Users/dev/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth"
tokenizer_path = "C:/Users/dev/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model"

# 토크나이저 및 모델 로드 (HuggingFace transformers가 있다고 가정)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 간단한 텍스트 생성 함수
def generate_text(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 대화형 테스트
def chat_with_model():
    print("모델과의 대화가 시작되었습니다. '종료'라고 입력하면 종료됩니다.")
    while True:
        user_input = input("사용자: ")
        if user_input.lower() == "종료":
            print("대화를 종료합니다.")
            break

        # 모델에게 사용자 입력 전달
        response = generate_text(model, tokenizer, user_input)
        print(f"모델: {response}")

# '너는 누구야?'로 테스트
response = generate_text(model, tokenizer, "너는 누구야?")
print("모델 응답:", response)

# 대화형 모드를 원하면 이 함수 호출
# chat_with_model()
