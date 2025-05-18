# app/routers/action_assignment.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Any, List # List 타입 힌팅 추가
from openai import OpenAI

# 의존성 주입 함수
from app.main import get_openai_client

# 서비스 함수 (추후 생성)
from app.services.action_item_service import extract_and_assign_action_items

# 요청/응답 Pydantic 모델
from app.models.meeting import ActionAssignmentRequest, ActionAssignmentResponse, AttendeeInfo # AttendeeInfo도 임포트

# 설정값
from app.core.config import settings

router = APIRouter()

@router.post("/assign-tasks", response_model=ActionAssignmentResponse)
async def assign_tasks_endpoint(
    request_data: ActionAssignmentRequest, # 요청 본문을 Pydantic 모델로 받음
    openai_client: OpenAI = Depends(get_openai_client)
):
    """
    제공된 회의록 텍스트, 주제, 참석자 정보를 바탕으로 할 일을 추출하고 담당자에게 분배합니다.
    """
    # Pydantic 모델에서 rc_txt와 info_n은 이미 필수로 설정되어 있음
    if not request_data.rc_txt:
        # 이 검사는 Pydantic 유효성 검사로 인해 실제로는 거의 도달하지 않음
        raise HTTPException(status_code=400, detail="할 일을 추출할 텍스트(rc_txt)가 제공되지 않았습니다.")
    if not request_data.info_n:
        # 이 검사도 Pydantic 유효성 검사로 인해 실제로는 거의 도달하지 않음
        raise HTTPException(status_code=400, detail="참석자 정보(info_n)가 제공되지 않았습니다.")

    # info_n의 각 AttendeeInfo 객체를 서비스 함수가 기대하는 List[Dict[str, Any]] 형태로 변환
    # (서비스 함수가 Pydantic 모델을 직접 받도록 수정할 수도 있지만, dict로 변환하는 것이 일반적일 수 있음)
    attendees_list_for_service: List[Dict[str, Any]] = []
    if request_data.info_n: # None이 아닐 경우에만 처리
        for attendee_pydantic_model in request_data.info_n:
            attendees_list_for_service.append(attendee_pydantic_model.model_dump(exclude_none=True))


    print(f"ActionAssignment Router: 할 일 분배 요청 수신 (주제: '{request_data.subj or '없음'}', 참석자 수: {len(attendees_list_for_service)})")

    try:
        # 서비스 함수 호출
        assigned_tasks_list = await extract_and_assign_action_items(
            openai_client=openai_client,
            rc_txt=request_data.rc_txt,
            subj=request_data.subj,
            info_n=attendees_list_for_service, # 변환된 dict 리스트 전달
            model_name=settings.DEFAULT_LLM_MODEL
        )

        # ActionItemByAssignee 모델 구조에 맞는 딕셔너리 리스트를 서비스 함수가 반환한다고 가정
        # (이전 서비스 함수 초안에서 그렇게 설계했음)
        return ActionAssignmentResponse(
            tasks=assigned_tasks_list, # 서비스 함수의 반환값을 그대로 사용
            message="할 일이 성공적으로 추출 및 분배되었습니다." if assigned_tasks_list else "추출된 할 일이 없습니다."
        )

    except ValueError as ve:
        print(f"ActionAssignment Router 오류 (ValueError): {ve}")
        raise HTTPException(status_code=400, detail=f"요청 데이터 또는 LLM 응답 처리 오류: {ve}")
    except RuntimeError as rte:
        print(f"ActionAssignment Router 오류 (RuntimeError): {rte}")
        raise HTTPException(status_code=503, detail=f"외부 서비스(LLM) 처리 중 오류 발생: {rte}")
    except HTTPException as e_http:
        raise e_http
    except Exception as e:
        print(f"ActionAssignment Router 알 수 없는 오류: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail="할 일 분배 처리 중 알 수 없는 서버 오류가 발생했습니다.")