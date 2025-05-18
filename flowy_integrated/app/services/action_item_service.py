# app/services/action_item_service.py
import json
import re
import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Pydantic 모델은 라우터나 최상위 응답 모델에서 사용되므로,
# 서비스 함수 자체는 Dict[str, Any] 리스트를 반환하는 것이 일반적일 수 있습니다.
# from app.models.meeting import ActionItemByAssignee # 반환 타입을 명시하기 위함이라면 사용 가능

# --- LLM 응답을 구조화된 리스트로 변환하는 헬퍼 함수 ---
# (기존 업로드된 코드에서 가져온 함수, 필요시 로직 검토 및 수정)
def _transform_llm_response_to_action_items(
    grouped_tasks_data: Dict[str, List[str]],
    attendees_list_for_role_extraction: List[Dict[str, Any]]
) -> List[Dict[str, Any]]: # 최종적으로 ActionItemByAssignee 모델에 매핑될 수 있는 딕셔너리 리스트
    """
    LLM이 반환한 담당자별 작업 목록 딕셔너리를
    {"name": "이름", "role": "역할", "tasks": ["할일1", "할일2"]} 형태의
    딕셔너리 리스트로 변환합니다.
    """
    structured_list = []
    if not grouped_tasks_data or not isinstance(grouped_tasks_data, dict):
        return structured_list

    # 참석자 이름과 역할을 매핑 (소문자로 정규화하여 매칭률 향상)
    attendee_role_map = {
        str(att.get("name", "")).strip().lower(): str(att.get("role", "")).strip()
        for att in attendees_list_for_role_extraction
    }

    for assignee_key, tasks_list in grouped_tasks_data.items():
        assignee_name = str(assignee_key).strip() # 키가 숫자로 올 경우 등 대비
        assignee_role = ""

        # "이름 (역할)" 형태에서 이름과 역할 분리 시도 (정규식 개선 가능)
        match = re.match(r"^(.*)\s*\(([^)]+)\)$", assignee_name) # 괄호 안 내용 탐욕적이지 않게
        if match:
            name_part = match.group(1).strip()
            role_part = match.group(2).strip()
            # LLM이 "홍길동 (홍길동)" 과 같이 이름을 역할에도 넣는 경우가 있어,
            # 역할 부분에 이름과 유사한 패턴이 반복되면 역할은 비우고 이름만 사용하거나,
            # 원본 참석자 정보에서 역할을 찾아오는 것을 우선
            if name_part.lower() == role_part.lower() and name_part: # 이름과 역할이 동일하면 역할은 나중에 다시 찾도록
                assignee_name = name_part
                assignee_role = attendee_role_map.get(name_part.lower(), "역할 미지정")
            else:
                assignee_name = name_part
                assignee_role = role_part
        else:
            # LLM이 역할 없이 이름만 반환한 경우, 원본 참석자 정보에서 역할 찾아보기
            # 또는 이름 자체가 '팀 전체' 등일 수 있음
            assignee_role = attendee_role_map.get(assignee_name.lower(), "역할 미지정")


        # "팀 전체", "전체" 등과 같은 특수 키 처리 (일관된 이름 사용)
        if assignee_name.lower() in ["전체", "팀", "모두", "팀 전체", "team", "all"]:
            assignee_name = "팀 전체"
            assignee_role = "팀" # 역할도 명시적으로 '팀'으로 설정

        # tasks가 문자열 리스트인지 확인, 아니면 빈 리스트로
        actual_tasks = []
        if isinstance(tasks_list, list):
            actual_tasks = [str(task).strip() for task in tasks_list if isinstance(task, str) and str(task).strip()]
        elif isinstance(tasks_list, str) and tasks_list.strip(): # 단일 문자열로 올 경우
             actual_tasks = [tasks_list.strip()]


        if not actual_tasks and assignee_name not in ["팀 전체"]: # 할 일이 없으면 (팀 전체 제외) 추가 안함 (선택적)
            # continue # 할 일이 없는 담당자는 결과에서 제외할 수 있음
            pass


        structured_list.append({
            "name": assignee_name,
            "role": assignee_role if assignee_role else "역할 미지정", # 역할이 비었으면 기본값
            "tasks": actual_tasks
        })

    return structured_list


async def extract_and_assign_action_items(
    openai_client: OpenAI,
    rc_txt: str,
    subj: Optional[str],
    info_n: List[Dict[str, Any]], # 라우터에서 AttendeeInfo.model_dump() 한 dict 리스트로 받음
    model_name: str
) -> List[Dict[str, Any]]: # ActionItemByAssignee 모델 구조에 맞는 딕셔너리 리스트 반환
    """
    LLM을 사용하여 회의록에서 할 일을 추출하고 담당자별로 그룹화하여 반환합니다.
    """
    if not openai_client:
        raise ValueError("OpenAI 클라이언트가 필요합니다.")
    if not rc_txt or not rc_txt.strip():
        print("ActionItem Service 경고: 할 일을 추출할 텍스트(rc_txt)가 비어있습니다. 빈 결과를 반환합니다.")
        return []
    if not info_n:
        # 참석자 정보가 없으면 역할 기반 할당이 어려우나, 이름 기반으로는 가능할 수 있음.
        # 여기서는 경고만 하고 진행하거나, 오류를 발생시킬 수 있음.
        print("ActionItem Service 경고: 참석자 정보(info_n)가 제공되지 않았습니다. 할 일 할당 정확도가 낮을 수 있습니다.")
        # return [] # 또는 오류 발생: raise ValueError("참석자 정보(info_n)가 필요합니다.")

    # 프롬프트용 참석자 정보 문자열 생성 (이름과 역할 포함)
    attendees_str_list = []
    for att in info_n:
        name = att.get('name', '이름없음')
        role = att.get('role')
        attendees_str_list.append(f"{name}({role})" if role else name)
    attendees_info_for_prompt = ", ".join(attendees_str_list) if attendees_str_list else "참석자 정보 없음"

    topic_info = f"회의 주제는 '{subj}'입니다." if subj else "회의 주제는 제공되지 않았습니다."

    # JSON 응답 형식을 명확히 지시하는 프롬프트 (기존 프롬프트 개선)
    # 출력 예시를 프롬프트에 명시적으로 포함하여 LLM이 형식을 더 잘 따르도록 유도
    prompt = f"""
당신은 회의록을 분석하여 각 참석자 또는 팀 전체가 수행해야 할 구체적인 '할 일(액션 아이템)'을 식별하고, 이를 담당자별로 그룹화하여 할당하는 전문가입니다.

[회의 정보]
* {topic_info}
* 참석자 (이름 또는 이름(역할)): "{attendees_info_for_prompt}"

[회의록 전문]
{rc_txt}

[요청 사항]
1. 회의록 전체 내용을 면밀히 검토하여, 실행 가능한 구체적인 작업이나 책임을 나타내는 '할 일'을 모두 찾아주십시오.
2. 각 할 일은 명확하고 간결한 문장으로 기술되어야 합니다.
3. 담당자는 다음 우선순위를 따릅니다:
    a. 회의록 대화에서 이름이나 역할이 명시적으로 언급된 경우.
    b. 이름/역할이 명시되지 않았다면, [회의 정보]의 '참석자' 목록을 참고하여 할 일 내용과 가장 관련 깊은 사람으로 지정합니다.
    c. 특정인에게 할당하기 어렵거나 모두에게 해당되는 경우 '팀 전체'로 지정합니다.
4. 할 일에 마감 기한이 언급되었다면, " (기한: [언급된 내용])" 형식으로 할 일 설명에 포함시켜 주십시오. (예: "UI 디자인 시안 다음 주 수요일까지 완료 (기한: 다음 주 수요일까지)") 기한 언급이 없다면 생략합니다.
5. 최종 결과는 반드시 다음 JSON 형식으로 제공해야 합니다. 'grouped_tasks' 객체 안에 결과를 담아주십시오. 키는 "담당자 이름" 또는 "담당자 이름 (역할)" 형식이고, 값은 해당 담당자의 할 일(문자열) 리스트입니다.

[출력 JSON 형식 예시]
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
    "팀 전체": [
      "다음 회의 아이디어 구체화 (기한: 다음 회의 전까지)"
    ]
  }}
}}
"""
    print(f"ActionItem Service: 할 일 추출 요청 시작 (모델: {model_name}, 주제: '{subj or '없음'}', 참석자 수: {len(info_n)})")

    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=model_name,
            messages=[
                {"role": "system", "content": "당신은 회의 내용을 분석하여 담당자별 할 일을 JSON 형식으로 추출하는 전문가입니다. 반드시 요청된 JSON 형식('grouped_tasks' 키 포함)으로 답변해야 합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # 할 일 추출은 일관성이 중요하므로 낮은 온도
            max_tokens=1000, # 응답 길이를 적절히 설정 (할 일 목록이 길어질 수 있음)
            response_format={"type": "json_object"} # OpenAI 최신 모델은 JSON 모드 지원
        )

        response_content = response.choices[0].message.content
        if response_content is None:
            print("ActionItem Service 오류: LLM으로부터 비어있는 응답(content=None)을 받았습니다.")
            # 이 경우 빈 리스트 반환 또는 예외 발생
            return []

        try:
            # LLM 응답이 이미 JSON 문자열이므로 json.loads()로 파싱
            llm_response_data = json.loads(response_content)
            grouped_tasks_from_llm = llm_response_data.get("grouped_tasks")

            if grouped_tasks_from_llm is None: # 'grouped_tasks' 키가 없는 경우
                print(f"ActionItem Service 경고: LLM 응답에 'grouped_tasks' 키가 없습니다. 응답: {response_content}")
                # 할 일이 없는 것으로 간주하거나, LLM 응답 형식 오류로 처리 가능
                return [] # 빈 리스트 반환
            if not isinstance(grouped_tasks_from_llm, dict):
                print(f"ActionItem Service 오류: LLM 응답의 'grouped_tasks'가 딕셔너리가 아닙니다. 타입: {type(grouped_tasks_from_llm)}, 응답: {response_content}")
                # 이 경우도 할 일이 없는 것으로 간주하거나 오류 처리
                return []

            # LLM 응답을 우리가 원하는 최종 구조로 변환 (헬퍼 함수 사용)
            # _transform_llm_response_to_action_items 함수는 List[Dict[str, Any]]를 반환.
            # 이 Dict는 ActionItemByAssignee 모델의 필드와 일치해야 함.
            action_items_list_of_dicts = _transform_llm_response_to_action_items(
                grouped_tasks_from_llm,
                info_n # 원본 참석자 정보 (dict 리스트) 전달
            )
            
            print(f"ActionItem Service: 할 일 추출 및 변환 완료. {len(action_items_list_of_dicts)}개 담당자/팀에 대한 작업 추출.")
            return action_items_list_of_dicts # Dict 리스트 반환

        except json.JSONDecodeError:
            print(f"ActionItem Service 오류: LLM 응답을 JSON으로 파싱할 수 없습니다. 원본 응답: {response_content}")
            # 이 경우 LLM이 JSON 형식을 제대로 따르지 않은 것임. 프롬프트 수정 또는 후처리 강화 필요.
            # 또는, 파싱 가능한 부분만 추출 시도 (더 복잡)
            raise ValueError(f"LLM으로부터 유효한 JSON 형식의 할 일 목록을 받지 못했습니다. 응답 내용: {response_content[:300]}...")
        except KeyError: # "grouped_tasks" 키가 없는 경우 (json.loads는 성공했으나 키가 없을 때)
             print(f"ActionItem Service 오류: LLM JSON 응답에 'grouped_tasks' 키가 없습니다. 원본 응답: {response_content}")
             raise ValueError("LLM JSON 응답에 'grouped_tasks' 키가 누락되었습니다.")


    except Exception as e:
        print(f"ActionItem Service 오류: OpenAI API 호출/처리 중 예외 발생 - {type(e).__name__}: {e}")
        raise RuntimeError(f"할 일 추출 및 분배 중 서버 오류가 발생했습니다: {e}") from e