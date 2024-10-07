import requests
import json
from prompt.persona_prompt import generate_final_prompt  # 프롬프트 생성 함수 가져오기

def get_llama_model(prompt_text, rag_based=False, temperature=0.9):
    url = "http://localhost:11434/api/generate"
    
    # 페르소나와 사용자 프롬프트 결합
    final_prompt = generate_final_prompt(prompt_text)
    
    payload = {
        "model": "llama3.1:8b",
        "prompt": final_prompt,
        "temperature": temperature
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers, stream=True)  # 스트리밍 활성화

    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_data = line.decode("utf-8")
                json_obj = json.loads(json_data)
                full_response += json_obj.get("response", "")
            except ValueError as e:
                print(f"Error decoding JSON: {e}")
    
    # 응답에 따라 RAG 기반인지 여부 표시
    if rag_based:
        return f"[RAG 기반 응답]: {full_response}"
    else:
        return f"[일반 LLM 응답]: {full_response}"
