# app/routers/feedback.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Any, List
from openai import OpenAI

from app.main import get_openai_client
from app.services.relevance_service import analyze_sentence_relevance_service
# FeedbackRequest에서 num_representative_unnecessary가 제거되었으므로,
# 서비스 함수 호출 시 해당 값을 명시적으로 전달하지 않아도 서비스 함수의 기본값이 사용됩니다.
from app.models.meeting import FeedbackRequest, MeetingFeedbackResponseModel, AttendeeInfo
from app.core.config import settings

router = APIRouter()

@router.post("/analyze-sentences", response_model=MeetingFeedbackResponseModel)
async def analyze_meeting_feedback_endpoint(
    request_data: FeedbackRequest, # 이제 num_representative_unnecessary 필드가 없는 모델
    openai_client: OpenAI = Depends(get_openai_client)
):
    """
    제공된 회의록 텍스트, 주제, 참석자 정보를 바탕으로 문장별 필요/불필요 분석을 수행하고,
    관련 통계 및 대표 불필요 문장(기본 5개)을 반환합니다.
    """
    print(f"Feedback Router: 문장 분석 요청 수신 (주제: '{request_data.subj or '없음'}', 참석자 수: {len(request_data.info_n)})")

    attendees_info_list_of_dicts = [attendee.model_dump(exclude_none=True) for attendee in request_data.info_n]

    try:
        feedback_result_dict = await analyze_sentence_relevance_service(
            openai_client=openai_client,
            rc_txt=request_data.rc_txt,
            subj=request_data.subj,
            info_n=attendees_info_list_of_dicts,
            model_name=settings.DEFAULT_LLM_MODEL
            # num_representative_unnecessary 인자를 전달하지 않으면,
            # 서비스 함수 analyze_sentence_relevance_service에 정의된 기본값(5)이 사용됩니다.
        )

        return MeetingFeedbackResponseModel(**feedback_result_dict)

    # ... (오류 처리 로직은 이전과 동일) ...
    except ValueError as ve:
        print(f"Feedback Router 오류 (ValueError): {ve}")
        raise HTTPException(status_code=400, detail=f"요청 데이터 또는 LLM 응답 처리 오류: {ve}")
    except RuntimeError as rte:
        print(f"Feedback Router 오류 (RuntimeError): {rte}")
        raise HTTPException(status_code=503, detail=f"외부 서비스(LLM) 처리 중 오류 발생: {rte}")
    except HTTPException as e_http:
        raise e_http
    except Exception as e:
        print(f"Feedback Router 알 수 없는 오류: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail="문장 분석 처리 중 알 수 없는 서버 오류가 발생했습니다.")