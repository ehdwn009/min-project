# app/services/relevance_service.py
import json
import asyncio
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI

# --- Kiwi 로더 (선택적 의존성) ---
_KIWI_AVAILABLE = False
kiwi_analyzer = None
try:
    from kiwipiepy import Kiwi
    kiwi_analyzer = Kiwi() # 기본 모델로 초기화
    # kiwi_analyzer.add_user_word("팀장님", "NNP") # 사용자 정의 단어 추가 가능
    _KIWI_AVAILABLE = True
    print("Relevance Service: Kiwi 형태소 분석기 로드 성공.")
except ImportError:
    print("Relevance Service 경고: Kiwi 라이브러리를 찾을 수 없습니다. 기본적인 문장 분리(정규식 기반)만 가능합니다.")
except Exception as e_kiwi_init:
    print(f"Relevance Service 경고: Kiwi 초기화 중 오류 발생 - {e_kiwi_init}. 기본적인 문장 분리만 가능합니다.")

# === 내부 헬퍼 함수: 문장 분리 ===
def _split_text_into_sentences(text_block: str) -> List[str]:
    """회의록 텍스트를 문장 단위로 분리합니다."""
    if not text_block or not text_block.strip():
        return []

    sentences = []
    if _KIWI_AVAILABLE and kiwi_analyzer:
        try:
            # Kiwi를 사용하여 문장 분리
            # split_into_sents는 각 문장에 대해 (text, start_pos, end_pos, morphemes) 튜플의 리스트를 반환할 수 있음
            # 여기서는 Kiwi.KiwiSentence 객체의 리스트를 반환한다고 가정 (최신 버전 기준)
            sentences_obj_list = kiwi_analyzer.split_into_sents(text_block.strip())
            sentences = [s.text.strip() for s in sentences_obj_list if s.text.strip()]
        except Exception as e_kiwi_split:
            print(f"Relevance Service 오류: Kiwi 문장 분리 중 예외 발생 - {e_kiwi_split}. 정규식 기반 분리로 대체합니다.")
            # Kiwi 실패 시 아래의 정규식 기반 분리 사용
            sentences = [] # 초기화
    
    if not sentences: # Kiwi 사용 불가 또는 실패 시
        # 기본적인 정규식을 사용하여 문장 분리 (마침표, 물음표, 느낌표 기준)
        # 문장 끝에 공백이 여러 개 올 수 있고, 따옴표 등으로 끝날 수도 있음을 고려
        # Lookbehind assertion `(?<=[.!?])` 사용
        temp_sentences = re.split(r'(?<=[.!?])\s+', text_block.strip())
        # 추가적으로 따옴표 등으로 끝나는 문장 처리 (간단한 예시)
        # final_sentences = []
        # for s in temp_sentences:
        #     if s.endswith('"') or s.endswith("'") or s.endswith('”') or s.endswith('’'):
        #         # 닫는 따옴표 앞의 구두점도 고려해야 함
        #         pass # 더 정교한 로직 필요
        sentences = [s.strip() for s in temp_sentences if s.strip()]

    # 매우 짧은 문장(예: 단답형 대답)이나 의미 없는 문자열 제거 (선택적)
    # sentences = [s for s in sentences if len(s) > 3] # 예시: 3글자 초과

    return sentences


# === 서비스 주 함수 ===
async def analyze_sentence_relevance_service(
    openai_client: OpenAI,
    rc_txt: str,
    subj: Optional[str],
    info_n: List[Dict[str, Any]], # 라우터에서 AttendeeInfo.model_dump() 한 dict 리스트
    model_name: str,
    num_representative_unnecessary: int = 5 # 기본값 (라우터에서 MeetingFeedbackResponseModel 기본값과 일치시킬 수 있음)
) -> Dict[str, Any]: # MeetingFeedbackResponseModel 구조에 맞는 딕셔너리 반환
    """
    회의록의 각 문장 관련성을 분석하여 필요/불필요 비율과 대표 불필요 문장을 반환합니다.
    """
    if not openai_client:
        raise ValueError("OpenAI 클라이언트가 필요합니다.")
    if not rc_txt or not rc_txt.strip():
        print("Relevance Service 경고: 분석할 텍스트(rc_txt)가 비어있습니다. 빈 결과를 반환합니다.")
        return {
            "necessary_ratio": 0.0,
            "unnecessary_ratio": 0.0,
            "representative_unnecessary": []
        }

    # 1. 회의록을 문장 단위로 분리
    sentences_list = _split_text_into_sentences(rc_txt)
    if not sentences_list:
        print("Relevance Service 경고: 텍스트에서 문장을 분리할 수 없었습니다. 빈 결과를 반환합니다.")
        return {
            "necessary_ratio": 0.0,
            "unnecessary_ratio": 0.0,
            "representative_unnecessary": []
        }
    
    print(f"Relevance Service: 문장 분리 완료 ({len(sentences_list)}개 문장), 각 문장 관련성 분석 시작...")

    # 프롬프트용 참석자 정보 문자열 생성
    attendees_str_list = []
    for att in info_n:
        name = att.get('name', '이름없음')
        role = att.get('role')
        attendees_str_list.append(f"{name}({role})" if role else name)
    attendees_info_for_prompt = ", ".join(attendees_str_list) if attendees_str_list else "참석자 정보 없음"
    
    topic_info = f"회의 주제는 '{subj}'입니다." if subj else "회의 주제는 제공되지 않았습니다."

    # 각 문장 분석을 위한 비동기 작업 리스트 생성
    analysis_tasks = []

    # (주의) 문장 수가 매우 많으면 API 호출 제한 및 비용 문제 발생 가능
    # 실제 운영 시에는 동시 요청 수 제한(Semaphore), 배치 처리, 비용 최적화 전략 필요
    # 여기서는 모든 문장을 개별적으로 분석 (기존 코드 방식)
    # 필요시 문장 묶음(chunk) 단위로 분석 요청 고려 가능

    for i, sentence_text in enumerate(sentences_list):
        if not sentence_text.strip(): # 빈 문장은 건너뜀
            continue

        # 각 문장에 대한 프롬프트 구성
        prompt_for_sentence = f"""
당신은 회의록 문장의 중요도와 관련성을 판단하는 AI입니다.
[회의 정보]와 [판단 지침]을 참고하여, 아래 [분석할 문장]이 회의의 목적 달성에 "필요"한지 "불필요"한지 분류하고, 그 이유를 간결하게 설명해주십시오.

[회의 정보]
* {topic_info}
* 주요 참석자: "{attendees_info_for_prompt}"

[판단 지침]
- '필요': 회의 주제와 직접 관련된 논의, 의사 결정, 질문, 답변, 실행 계획, 중요한 정보 공유 등 회의의 목적을 달성하는 데 기여하는 문장.
- '불필요': 회의 주제와 무관한 사담, 농담, 인사치레, 불필요한 반복, 모호하거나 의미 없는 발언, 회의 흐름을 방해하는 문장.

[분석할 문장 No.{i + 1}]
"{sentence_text}"

[요청 JSON 형식]
결과는 반드시 다음 JSON 형식으로 제공해주십시오:
{{
  "classification": "필요" 또는 "불필요",
  "reason": "판단 이유 (20단어 이내로 간결하게)"
}}
"""
        # 동기 함수인 openai_client.chat.completions.create를 비동기적으로 실행
        # 각 문장 분석은 독립적이므로 asyncio.gather로 병렬 처리 가능
        task = asyncio.to_thread(
            openai_client.chat.completions.create,
            model=model_name, # 개별 문장 분석에는 더 작고 빠른 모델 사용 고려 가능 (예: gpt-3.5-turbo)
            messages=[
                {"role": "system", "content": "당신은 문장 분석 전문가이며, 요청된 JSON 형식으로 답변합니다."},
                {"role": "user", "content": prompt_for_sentence}
            ],
            temperature=0.1, # 일관된 분류를 위해 낮은 온도
            max_tokens=100,  # 분류 및 이유 설명에 충분한 토큰
            response_format={"type": "json_object"} # JSON 모드 활용
        )
        analysis_tasks.append(task)

    # 모든 문장 분석 작업 병렬 실행 및 결과 수집
    try:
        llm_responses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
    except Exception as e_gather: # gather 자체에서 예외 발생 시 (거의 없음)
        print(f"Relevance Service 오류: asyncio.gather 중 예외 발생 - {e_gather}")
        raise RuntimeError(f"문장 분석 작업 실행 중 오류 발생: {e_gather}") from e_gather


    necessary_count = 0
    unnecessary_count = 0
    error_count = 0
    unnecessary_sentence_details = [] # {"sentence": str, "reason": str} 형태의 딕셔너리 저장

    for i, response_or_exc in enumerate(llm_responses):
        original_sentence = sentences_list[i] # 해당 응답에 매칭되는 원본 문장 (빈 문장 필터링 고려 시 인덱스 주의)

        if isinstance(response_or_exc, Exception):
            print(f"Relevance Service 오류: 문장 '{original_sentence[:30]}...' 분석 중 API 호출 예외 - {response_or_exc}")
            error_count += 1
            continue # 다음 문장으로

        # 정상 응답인 경우 (OpenAI API 응답 객체)
        try:
            response_content = response_or_exc.choices[0].message.content
            if response_content is None:
                print(f"Relevance Service 오류: 문장 '{original_sentence[:30]}...' 분석 결과 내용이 비어있습니다.")
                error_count += 1
                continue

            analysis_result = json.loads(response_content) # JSON 파싱
            classification = analysis_result.get("classification")
            reason = analysis_result.get("reason", "이유 명시 안됨")

            if classification == "필요":
                necessary_count += 1
            elif classification == "불필요":
                unnecessary_count += 1
                unnecessary_sentence_details.append({
                    "sentence": original_sentence,
                    "reason": reason
                })
            else: # "오류" 또는 예상치 못한 값
                print(f"Relevance Service 경고: 문장 '{original_sentence[:30]}...' 분류값 오류 또는 알 수 없음 - '{classification}'. 원본 응답: {response_content}")
                error_count += 1
        
        except json.JSONDecodeError:
            print(f"Relevance Service 오류: 문장 '{original_sentence[:30]}...' 분석 결과 JSON 파싱 실패. 원본 응답: {response_content}")
            error_count +=1
        except (KeyError, AttributeError, IndexError) as e_struct: # 응답 구조 문제
            print(f"Relevance Service 오류: 문장 '{original_sentence[:30]}...' 분석 결과 구조 문제 - {e_struct}. 원본 응답: {response_content}")
            error_count += 1
        except Exception as e_proc: # 기타 처리 중 예외
            print(f"Relevance Service 오류: 문장 '{original_sentence[:30]}...' 분석 결과 처리 중 예외 - {e_proc}")
            error_count += 1
    
    total_analyzed_sentences = len(sentences_list) # 실제 분석 시도된 문장 수 (오류 포함)
    # 또는 total_classified = necessary_count + unnecessary_count

    # 비율 계산 (0으로 나누기 방지)
    # 전체 문장 수에서 오류가 난 문장을 제외하고 비율을 계산할지, 포함할지 정책 결정 필요
    # 여기서는 전체 분리된 문장 수를 기준으로 계산
    valid_classifications = necessary_count + unnecessary_count
    
    necessary_ratio = round((necessary_count / total_analyzed_sentences) * 100, 2) if total_analyzed_sentences > 0 else 0.0
    unnecessary_ratio = round((unnecessary_count / total_analyzed_sentences) * 100, 2) if total_analyzed_sentences > 0 else 0.0
    # 오류 비율도 필요하다면 계산 가능: error_ratio = round((error_count / total_analyzed_sentences) * 100, 2)

    # 대표 불필요 문장 선정 (요청된 개수만큼, 순서대로)
    representative_unnecessary_output = unnecessary_sentence_details[:num_representative_unnecessary]
    
    print(f"Relevance Service: 분석 완료. 총 {total_analyzed_sentences}문장. 필요: {necessary_count}, 불필요: {unnecessary_count}, 분석 오류: {error_count}.")

    # MeetingFeedbackResponseModel의 필드명에 맞춰 딕셔너리 반환
    return {
        "necessary_ratio": necessary_ratio,
        "unnecessary_ratio": unnecessary_ratio,
        "representative_unnecessary": representative_unnecessary_output # 이미 {"sentence": ..., "reason": ...} 형태
    }