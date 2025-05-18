# app/services/action_item_service.py
import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
import asyncio

# Pydantic 모델 임포트 (서비스 내부에서는 dict를 사용하거나, 여기서도 모델을 직접 사용할 수 있음)
# from app.models.meeting import AttendeeInfo # 만약 서비스 함수가 Pydantic 모델을 직접 받는다면

# --- 기존 action_item_extractor.py의 헬퍼 함수 ---
# LLM 응답을 구조화된 리스트로 변환하는 로직
def _transform_llm_response_to_action_items(
    grouped_tasks_data: Dict[str, List[str]], # LLM이 반환한 "담당자 (역할)": ["할일1", "할일2"] 형태의 딕셔너리
    attendees_list_for_role_extraction: List[Dict[str, Any]] # 원본 참석자 정보 (info_n을 dict 리스트로 변환한 것)
) -> List[Dict[str, Any]]: # ActionItemByAssignee 모델에 맞는 딕셔너리 리스트 반환
    """
    LLM이 반환한 grouped_tasks 딕셔너리를
    각 담당자별로 {"name": "이름", "role": "역할", "tasks": ["할일1", "할일2"]} 형태의
    딕셔너리 리스트로 변환합니다.
    기존 _transform_grouped_tasks_to_structured_list 함수의 로직을 따릅니다.
    """
    structured_list = []
    if not grouped_tasks_data:
        return structured_list

    # 원본 참석자 이름-역할 매핑 정보 준비 (더 정확한 역할 매칭을 위해)
    attendee_role_map = {att.get("name", "").strip().lower(): att.get("role", "").strip() for att in attendees_list_for_role_extraction}

    for assignee_key, tasks_list in grouped_tasks_data.items():
        assignee_name = assignee_key.strip()
        assignee_role = ""

        # "이름 (역할)" 형태에서 이름과 역할 분리 시도 (기존 정규식)
        match = re.match(r"^(.*)\s*\((.*)\)$", assignee_key)
        if match:
            assignee_name = match.group(1).strip()
            assignee_role = match.group(2).strip()
        else:
            # LLM이 역할 없이 이름만 반환한 경우, 원본 참석자 정보에서 역할 찾아보기
            assignee_role = attendee_role_map.get(assignee_name.lower(), "역할 미지정") # 소문자로 비교


        # "전체 (팀)" 등과 같은 특수 키 처리 (기존 로직)
        if assignee_name.lower() in ["전체 (팀)", "팀 전체", "전체", "팀"]:
            assignee_name = "팀 전체" # 일관된 이름 사용
            assignee_role = "팀"

        # tasks가 문자열 리스트인지 확인, 아니면 빈 리스트로 (기존 로직)
        actual_tasks = tasks_list if isinstance(tasks_list, list) and all(isinstance(t, str) for t in tasks_list) else []

        structured_list.append({
            "name": assignee_name,
            "role": assignee_role,
            "tasks": actual_tasks
        })

    return structured_list


async def extract_and_assign_action_items(
    openai_client: OpenAI,
    rc_txt: str,
    subj: Optional[str],
    info_n: List[Dict[str, Any]], # Pydantic 모델 리스트가 아닌 dict 리스트로 받음 (라우터에서 변환)
    model_name: str
) -> List[Dict[str, Any]]: # ActionItemByAssignee 모델에 맞는 딕셔너리 리스트 반환
    """
    LLM을 사용하여 회의록에서 할 일을 추출하고 담당자별로 그룹화하여 반환합니다.
    기존 action_item_extractor.py의 assign_action_items 함수 로직을 기반으로 합니다.
    """
    if not openai_client:
        raise ValueError("OpenAI 클라이언트가 제공되지 않았습니다.")
    if not rc_txt:
        print("ActionItem Service: 할 일을 추출할 텍스트(rc_txt)가 비어있습니다.")
        return []
    if not info_n: # 참석자 정보는 역할 기반 할당에 중요
        print("ActionItem Service 경고: 참석자 정보(info_n)가 제공되지 않았습니다. 할 일 할당 정확도가 낮을 수 있습니다.")
        # raise ValueError("할 일 분배를 위해 참석자 정보(info_n)가 필요합니다.") # 또는 경고 후 진행

    # 프롬프트용 참석자 정보 문자열 생성 (기존 방식)
    attendees_str_list = [f"{att.get('name', '이름없음')}({att.get('role', '역할없음')})" for att in info_n]
    attendees_info_for_prompt = ", ".join(attendees_str_list) if attendees_str_list else "참석자 정보 없음"
    
    topic_info = f"* 회의 주제: \"{subj}\"" if subj else "* 회의 주제: (제공되지 않음)"

    prompt = f"""
당신은 회의 분석 및 프로젝트 관리 전문가입니다. 주어진 회의록, 회의 정보, 그리고 참석자 목록을 바탕으로, 각 참석자 또는 팀 전체가 수행해야 할 구체적인 '할 일(액션 아이템)'을 식별하고, 이를 담당자별로 그룹화하여 할당해주십시오.

[회의 정보]
{topic_info}
* 참석자 및 역할(직무): "{attendees_info_for_prompt}"

[회의록 전문]
{rc_txt}

[요청 사항]
1.  회의록 전체 내용을 면밀히 검토하여, 실행 가능한 구체적인 작업이나 책임을 나타내는 '할 일'을 모두 찾아주십시오.
2.  각 할 일은 명확하고 간결하게 기술되어야 하며, 가능하다면 누가 그 일을 맡아야 하는지 명시해주십시오.
3.  담당자는 다음 우선순위를 따릅니다:
    a.  회의록 대화에서 이름이 명시적으로 언급된 경우.
    b.  이름이 명시되지 않았다면, [회의 정보]의 '참석자 및 역할(직무)'을 참고하여 할 일 내용과 가장 관련 깊은 역할을 가진 참석자로 지정.
    c.  특정하기 어렵다면 '팀 전체' 또는 '미지정'으로 표시할 수 있습니다. (가능한 구체적인 담당자를 지정하는 것을 목표로 합니다.)
4.  할 일에 마감 기한이 언급되었다면, " (기한: [언급된 내용])" 형식으로 할 일 설명에 포함시켜 주십시오. (예: "UI 디자인 시안 다음 주 수요일까지 완료 (기한: 다음 주 수요일까지)") 기한 언급이 없다면 생략합니다.
5.  최종 결과는 각 담당자(또는 '팀 전체')를 키로 하고, 해당 담당자의 할 일(문자열) 리스트를 값으로 가지는 JSON 객체 형식으로 제공해야 합니다. 담당자 키는 "이름 (역할)" 형식 또는 이름만 사용할 수 있습니다.

[출력 형식]
다음 JSON 형식을 반드시 따라주십시오. 'grouped_tasks' 객체 안에 결과를 담아주십시오:
{{
  "grouped_tasks": {{
    "이지은 (기획자)": [
      "사용자 리뷰 기능 MVP 범위 정의 및 제안",
      "다음 회의 시간 조정 관련 팀원 의견 취합 및 확정 공지 (기한: 내일 오후까지)"
    ],
    "김영희 (백엔드개발자)": [
      "리뷰 기능 관련 API 엔드포인트 설계 (기한: 오늘까지)",
      "Review DB 테이블 설계 및 생성"
    ],
    "박철수 (프론트개발자)": [
      "리뷰 기능 UI 스케치 정리 및 Figma에 공유"
    ],
    "팀 전체": [
      "다음 회의 아이디어 구체화 (기한: 다음 회의 전까지)"
    ]
  }}
}}
"""
    print(f"ActionItem Service: 할 일 추출 요청 시작 (모델: {model_name}, 주제: '{subj or '없음'}')")

    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=model_name,
            messages=[
                {"role": "system", "content": "당신은 회의 내용을 분석하여 담당자별 할 일을 JSON 형식으로 추출하는 전문가입니다. 반드시 요청된 JSON 형식('grouped_tasks' 키 포함)으로 답변해야 합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # 할 일 추출은 일관성이 중요하므로 낮은 온도
            response_format={"type": "json_object"}
        )

        response_content = response.choices[0].message.content
        if response_content is None:
            print("ActionItem Service 오류: LLM으로부터 비어있는 응답(content=None)을 받았습니다.")
            raise ValueError("LLM으로부터 비어있는 응답을 받았습니다.")

        try:
            llm_response_data = json.loads(response_content)
            grouped_tasks_from_llm = llm_response_data.get("grouped_tasks")

            if grouped_tasks_from_llm is None:
                print(f"ActionItem Service 오류: LLM 응답에 'grouped_tasks' 키가 없습니다. 응답: {response_content}")
                # 이 경우, 할 일이 없다고 판단하거나, LLM 응답 형식이 잘못되었다고 판단 가능
                # 할 일이 없는 경우 빈 리스트 반환
                return []
            if not isinstance(grouped_tasks_from_llm, dict):
                print(f"ActionItem Service 오류: LLM 응답의 'grouped_tasks'가 딕셔너리가 아닙니다. 타입: {type(grouped_tasks_from_llm)}, 응답: {response_content}")
                raise ValueError("LLM 응답 형식이 잘못되었습니다 ('grouped_tasks'가 딕셔너리가 아님).")

            # LLM 응답을 우리가 원하는 최종 구조로 변환 (헬퍼 함수 사용)
            action_items_list = _transform_llm_response_to_action_items(grouped_tasks_from_llm, info_n)
            
            print(f"ActionItem Service: 할 일 추출 및 변환 완료 ({len(action_items_list)}개 담당자/팀)")
            return action_items_list

        except json.JSONDecodeError:
            print(f"ActionItem Service 오류: LLM 응답을 JSON으로 파싱할 수 없습니다. 원본 응답: {response_content}")
            raise ValueError(f"LLM 응답을 JSON으로 파싱하는 데 실패했습니다. 응답 내용: {response_content[:200]}...")
        except KeyError:
            print(f"ActionItem Service 오류: LLM JSON 응답에 'grouped_tasks' 키가 없습니다. 원본 응답: {response_content}")
            raise ValueError("LLM JSON 응답에 'grouped_tasks' 키가 누락되었습니다.")

    except Exception as e:
        print(f"ActionItem Service 오류: OpenAI API 호출/처리 중 예외 발생 - {type(e).__name__}: {e}")
        raise RuntimeError(f"할 일 추출 및 분배 중 오류가 발생했습니다.") from e