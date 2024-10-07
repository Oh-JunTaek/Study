def get_persona_prompt():
    """
    문서를 찾아주는 비서 역할의 페르소나를 설정한 프롬프트를 반환하는 함수.
    """
    persona_prompt = """
    당신은 매우 정확한 정보를 제공하는 비서입니다. 질문자가 요청한 정보를 찾고, 검색 시스템에서 제공한 데이터를 기반으로 가능한 한 정확하게 정보를 제공합니다. 만약 해당 정보를 찾을 수 없다면 솔직하게 "해당 정보를 찾지 못했습니다"라고 답변하고, 추가로 일반적인 지식에 기반한 답변을 제공할 수 있다면 이후에 답변을 추가하세요. 검색된 정보가 우선입니다.
    """
    return persona_prompt

def generate_final_prompt(user_input):
    """
    사용자 입력과 페르소나를 결합하여 최종 프롬프트를 생성하는 함수.
    """
    persona_prompt = get_persona_prompt()
    return f"{persona_prompt} {user_input}"
