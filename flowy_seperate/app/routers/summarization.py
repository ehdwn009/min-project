from fastapi import APIRouter, Depends, HTTPException
from typing import Any, List, Optional
from openai import OpenAI

# 의존성 주입 함수 (main.py 에서 가져옴)
from app.main import get_openai_client

# 서비스 함수 (summarizer_service.py 에서 가져옴)
from app.services.summarizer_service import get_meeting_summary

# 요청/응답 모델 (models/meeting.py 에서 가져옴)
from app.models.meeting import SummarizationRequest, SummarizationResponse

# 설정값 (config.py 에서 가져옴 - 모델명)
from app.core.config import settings


router = APIRouter()

@router.post("/summarize", response_model=SummarizationResponse)
async def summarize_text_endpoint(
    request_data: SummarizationRequest, # 요청 본문은 Pydantic 모델로 받음
    openai_client: OpenAI = Depends(get_openai_client)
):
    """
    제공된 텍스트(회의록)와 주제를 바탕으로 회의 내용을 요약합니다.
    """
    if not request_data.rc_txt: # Pydantic 모델에서 이미 필수로 설정했으므로, 이론상 이 검사는 불필요
        raise HTTPException(status_code=400, detail="요약할 텍스트(rc_txt)가 제공되지 않았습니다.")

    print(f"Summarization Router: 요약 요청 수신 (주제: '{request_data.subj or '없음'}')")

    try:
        summary_points_list = await get_meeting_summary(
            openai_client=openai_client,
            rc_txt=request_data.rc_txt, # 모델의 rc_txt 필드 사용
            subj=request_data.subj,     # 모델의 subj 필드 사용
            model_name=settings.DEFAULT_LLM_MODEL # config에서 기본 모델명 사용
        )

        return SummarizationResponse(
            summary=summary_points_list,
            message="회의 내용이 성공적으로 요약되었습니다."
        )

    except ValueError as ve: # 서비스 계층에서 발생시킨 ValueError 처리
        print(f"Summarization Router 오류 (ValueError): {ve}")
        raise HTTPException(status_code=400, detail=str(ve)) # 클라이언트 입력 오류로 간주
    except RuntimeError as rte: # 서비스 계층에서 발생시킨 RuntimeError 처리
        print(f"Summarization Router 오류 (RuntimeError): {rte}")
        raise HTTPException(status_code=503, detail=str(rte)) # 외부 서비스(OpenAI) 오류로 간주
    except Exception as e:
        print(f"Summarization Router 알 수 없는 오류: {type(e).__name__} - {e}")
        # traceback.print_exc() # 개발 중 상세 오류 확인
        raise HTTPException(status_code=500, detail="요약 처리 중 알 수 없는 서버 오류가 발생했습니다.")