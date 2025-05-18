# app/services/relevance_service.py
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import asyncio
import re # 문장 분리 fallback 및 후처리에 사용

# --- Kiwi 로더 및 가용성 확인 ---
_KIWI_AVAILABLE = False
kiwi_analyzer = None # 타입 힌팅을 위해 미리 선언
try:
    from kiwipiepy import Kiwi
    kiwi_analyzer = Kiwi()
    _KIWI_AVAILABLE = True
    print("Relevance Service: Kiwi 형태소 분석기 로드 성공.")
except ImportError:
    print("Relevance Service 경고: Kiwi 라이브러리를 찾을 수 없습니다. 기본적인 문장 분리만 가능합니다.")
except Exception as e_kiwi_init:
    print(f"Relevance Service 경고: Kiwi 초기화 중 오류 발생 - {e_kiwi_init}. 기본적인 문장 분리만 가능합니다.")

# === 내부 헬퍼 함수: 문장 분리 (기존 text_process_service.py 로직 통합) ===
def _split_text_into_sentences(text_block: str) -> List[str]:
    if not text_block: return []
    if kiwi_analyzer and _KIWI_AVAILABLE:
        try:
            sentences_obj_list = kiwi_analyzer.split_into_sents(text_block)
            sentences = [s.text.strip() for s in sentences_obj_list if s.text.strip()]
            return sentences
        except Exception as e_kiwi_split:
            print(f"Relevance Service 오류: Kiwi 문장 분리 중 예외 - {e_kiwi_split}. 기본 분리로 대체.")
    # Kiwi 실패 시 fallback (기존 로직 개선 여지 있음)
    sentences_fallback = re.split(r'(?<=[.!?])\s+', text_block.strip()) # 정규식 기반 분리
    return [s.strip() for s in sentences_fallback if s.strip()]


# === 서비스 주 함수 ===
async def analyze_sentence_relevance_service(
    openai_client: OpenAI,
    rc_txt: str,
    subj: Optional[str],
    info_n: List[Dict[str, Any]],
    model_name: str,
    num_representative_unnecessary: int = 5 # FeedbackRequest 모델의 기본값과 일치
) -> Dict[str, Any]: # MeetingFeedbackResponseModel에 맞는 딕셔너리 반환
    if not openai_client: raise ValueError("OpenAI 클라이언트가 제공되지 않았습니다.")
    if not rc_txt:
        return { # 빈 입력에 대한 기본 응답 (비율 0, 빈 리스트)
            "necessary_ratio": 0.0,
            "unnecessary_ratio": 0.0,
            "representative_unnecessary": []
        }

    sentences_list = _split_text_into_sentences(rc_txt)
    if not sentences_list:
        return {
            "necessary_ratio": 0.0,
            "unnecessary_ratio": 0.0,
            "representative_unnecessary": []
        }
    
    print(f"Relevance Service: 문장 분리 완료 ({len(sentences_list)}개), 관련성 분석 시작...")

    attendees_str_list = [f"{att.get('name', '이름없음')}({att.get('role', '역할없음')})" for att in info_n]
    attendees_info_for_prompt = ", ".join(attendees_str_list) if attendees_str_list else "참석자 정보 없음"
    topic_info = f"* 회의 주제: \"{subj}\"" if subj else "* 회의 주제: (제공되지 않음)"

    async def analyze_single_sentence(sentence_text: str, sentence_idx: int) -> Dict[str, Any]:
        prompt_for_sentence = f"""
당신은 회의록 문장의 중요도를 판단하는 AI입니다. [회의 정보]와 [판단 지침]에 따라, [분석할 문장]이 "필요"한지 "불필요"한지 분류하고 이유를 JSON으로 답해주세요.

[회의 정보]
{topic_info}
* 주요 참석자: "{attendees_info_for_prompt}"

[판단 지침]
- '필요': 주제 관련 논의, 결정, 질문, 답변, 실행 계획 등. 회의 진행에 필수적인 문장.
- '불필요': 주제 무관 사담, 농담, 안부, 불필요한 반복, 모호한 발언.

[분석할 문장 No.{sentence_idx + 1}]
"{sentence_text}"

[요청 JSON 형식]
{{
  "classification": "필요" 또는 "불필요",
  "reason": "판단 이유 (간결하게)"
}}
"""
        try:
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model=model_name, # 개별 문장 분석에도 동일 모델 사용 (또는 더 작은 모델)
                messages=[
                    {"role": "system", "content": "문장 분석 전문가 JSON 답변기"},
                    {"role": "user", "content": prompt_for_sentence}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            response_content = response.choices[0].message.content
            if response_content is None: return {"classification": "오류", "reason": "LLM 응답 없음"}
            analysis_result = json.loads(response_content)
            return {
                "classification": analysis_result.get("classification", "오류"),
                "reason": analysis_result.get("reason", "분석 실패")
            }
        except Exception as e_ana:
            print(f"Relevance Service 오류: 문장 {sentence_idx+1} 분석 중 - {e_ana}")
            return {"classification": "오류", "reason": f"분석 오류: {str(e_ana)[:50]}"}

    # (주의) 문장 수가 매우 많으면 API 호출 제한 및 비용 문제 발생 가능
    # 실제 운영 시에는 동시 요청 수 제한(Semaphore) 및 예외 처리 강화 필요
    analysis_tasks = [analyze_single_sentence(sentence, i) for i, sentence in enumerate(sentences_list)]
    sentence_analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True) # 개별 작업 예외 처리

    necessary_count = 0
    unnecessary_count = 0
    error_count = 0
    unnecessary_sentence_details = []

    for i, result_or_exc in enumerate(sentence_analysis_results):
        if isinstance(result_or_exc, Exception): # gather에서 예외 발생 시
            print(f"Relevance Service: 문장 {i+1} 분석 작업 실패 - {result_or_exc}")
            error_count +=1
            continue

        classification = result_or_exc.get("classification")
        if classification == "필요":
            necessary_count += 1
        elif classification == "불필요":
            unnecessary_count += 1
            unnecessary_sentence_details.append({
                "sentence": sentences_list[i],
                "reason": result_or_exc.get("reason", "")
            })
        else: # "오류" 또는 예상치 못한 값
            error_count += 1
            print(f"Relevance Service: 문장 {i+1} 분류값 오류 - {classification}")
    
    total_analyzed = len(sentences_list)
    
    # 비율 계산
    necessary_ratio = round((necessary_count / total_analyzed) * 100, 2) if total_analyzed > 0 else 0.0
    unnecessary_ratio = round((unnecessary_count / total_analyzed) * 100, 2) if total_analyzed > 0 else 0.0
    # error_ratio는 응답에 포함하지 않기로 함

    # 대표 불필요 문장 (요청된 개수만큼)
    representative_unnecessary_output = [
        {"sentence": detail["sentence"], "reason": detail["reason"]}
        for detail in unnecessary_sentence_details[:num_representative_unnecessary]
    ]
    
    print(f"Relevance Service: 분석 완료. 총 {total_analyzed}문장 중 필요 {necessary_count}, 불필요 {unnecessary_count}, 오류 {error_count}.")

    return {
        "necessary_ratio": necessary_ratio,
        "unnecessary_ratio": unnecessary_ratio,
        "representative_unnecessary": representative_unnecessary_output
    }