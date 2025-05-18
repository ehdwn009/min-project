# app/routers/analysis.py
import json
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from typing import Optional, List, Any
from openai import OpenAI

from app.dependencies import get_openai_client, get_stt_pipeline # dependencies.py 에서 가져옴
from app.core.config import settings
from app.models.meeting import (
    AttendeeInfo, MeetingMetadata, STTResponse, SummarizationResponse,
    ActionItemByAssignee, ActionAssignmentResponse, MeetingFeedbackResponseModel,
    RepresentativeUnnecessarySentenceModel, FullAnalysisResult
)
from app.services.stt_service import process_uploaded_rc_file_to_text
from app.services.summarizer_service import get_meeting_summary
from app.services.action_item_service import extract_and_assign_action_items
from app.services.relevance_service import analyze_sentence_relevance_service

router = APIRouter()

@router.post("/analyze", response_model=FullAnalysisResult)
async def analyze_meeting_endpoint(
    metadata_json: str = Form(..., description="회의 주제, 일시, 장소, 참석자 정보(배열)를 포함하는 JSON 문자열"), # 프론트와 필드명 일치 (예: "metadata")
    rc_file: Optional[UploadFile] = File(None, description="녹음 파일 (m4a, wav 등)"),
    openai_client: OpenAI = Depends(get_openai_client),
    stt_pipeline: Optional[Any] = Depends(get_stt_pipeline)
):
    # 1. 메타데이터 JSON 문자열 파싱
    try:
        metadata_dict = json.loads(metadata_json)
        meeting_info_data = MeetingMetadata(**metadata_dict) # Pydantic 모델로 변환
        print(f"분석 요청 수신 (메타데이터 파싱 성공): 주제='{meeting_info_data.subj}'")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="제공된 메타데이터가 유효한 JSON 형식이 아닙니다.")
    except Exception as e: # Pydantic ValidationError 등
        raise HTTPException(status_code=400, detail=f"메타데이터 처리 오류: {str(e)}")

    if rc_file:
        print(f"녹음 파일: {rc_file.filename if rc_file else '없음'}")

    current_rc_txt = ""
    stt_response_data: Optional[STTResponse] = None

    if rc_file:
        print("녹음 파일 처리 시작...")
        if not stt_pipeline:
            raise HTTPException(status_code=503, detail="STT 서비스를 현재 사용할 수 없습니다.")
        try:
            transcribed_text = await process_uploaded_rc_file_to_text(
                rc_file=rc_file,
                stt_pipeline_instance=stt_pipeline
            )
            if transcribed_text is None or not transcribed_text.strip():
                message = "음성 파일에서 텍스트를 추출하지 못했거나 내용이 없습니다."
                current_rc_txt = ""
                stt_response_data = STTResponse(rc_txt="", message=message)
            else:
                current_rc_txt = transcribed_text
                stt_response_data = STTResponse(rc_txt=current_rc_txt, message="음성 파일이 성공적으로 텍스트로 변환되었습니다.")
        except Exception as e:
            # stt_service에서 발생한 예외를 그대로 전달하거나, 여기서 새로운 HTTPException 생성
            if isinstance(e, (ValueError, RuntimeError)): # stt_service에서 발생시킨 특정 오류
                 raise HTTPException(status_code=400 if isinstance(e, ValueError) else 500, detail=f"STT 처리 중 오류: {str(e)}")
            raise HTTPException(status_code=500, detail=f"STT 처리 중 예기치 않은 서버 오류 발생: {str(e)}")
    else:
        # rc_file이 없으면 분석할 음성 소스가 없음 (rc_txt는 안 받는다고 가정)
        raise HTTPException(status_code=400, detail="분석할 음성 파일(rc_file)이 제공되지 않았습니다.")

    # STT 결과가 비어있으면 더 이상 진행 불가 (rc_file은 있었으나 변환 실패)
    if not current_rc_txt and rc_file:
        return FullAnalysisResult(
            meeting_info=meeting_info_data,
            stt_result=stt_response_data, # 실패 메시지 포함
            summary_result=SummarizationResponse(summary=[], message="STT 결과가 없어 요약을 생성할 수 없습니다."),
            action_items_result=ActionAssignmentResponse(tasks=[], message="STT 결과가 없어 할 일을 추출할 수 없습니다."),
            feedback_result=MeetingFeedbackResponseModel(
                necessary_ratio=0.0, unnecessary_ratio=0.0, representative_unnecessary=[]
            )
        )

    # --- 요약 ---
    print(f"회의 요약 시작 (텍스트 길이: {len(current_rc_txt)})...")
    try:
        summary_points = await get_meeting_summary(
            openai_client=openai_client, rc_txt=current_rc_txt,
            subj=meeting_info_data.subj, model_name=settings.DEFAULT_LLM_MODEL
        )
        summary_result_data = SummarizationResponse(summary=summary_points, message="회의 내용이 성공적으로 요약되었습니다.")
    except Exception as e:
        print(f"요약 생성 중 오류: {e}")
        summary_result_data = SummarizationResponse(summary=[], message=f"요약 생성 중 오류가 발생했습니다: {str(e)}")

    # --- 할 일 분배 ---
    print("할 일 분배 시작...")
    attendees_list_for_service = [att.model_dump(exclude_none=True) for att in meeting_info_data.info_n]
    try:
        assigned_tasks_structured = await extract_and_assign_action_items(
            openai_client=openai_client, rc_txt=current_rc_txt,
            subj=meeting_info_data.subj, info_n=attendees_list_for_service,
            model_name=settings.DEFAULT_LLM_MODEL
        )
        action_items_result_data = ActionAssignmentResponse(
            tasks=[ActionItemByAssignee(**task_dict) for task_dict in assigned_tasks_structured],
            message="할 일이 성공적으로 추출 및 분배되었습니다." if assigned_tasks_structured else "추출된 할 일이 없습니다."
        )
    except Exception as e:
        print(f"할 일 추출 중 오류: {e}")
        action_items_result_data = ActionAssignmentResponse(tasks=[], message=f"할 일 추출 중 오류가 발생했습니다: {str(e)}")

    # --- 회의 피드백 ---
    print("회의 피드백 분석 시작...")
    try:
        feedback_result_dict = await analyze_sentence_relevance_service(
            openai_client=openai_client, rc_txt=current_rc_txt,
            subj=meeting_info_data.subj, info_n=attendees_list_for_service,
            model_name=settings.DEFAULT_LLM_MODEL
        )
        feedback_result_data = MeetingFeedbackResponseModel(
            necessary_ratio=feedback_result_dict.get("necessary_ratio", 0.0),
            unnecessary_ratio=feedback_result_dict.get("unnecessary_ratio", 0.0),
            representative_unnecessary=[
                RepresentativeUnnecessarySentenceModel(**item)
                for item in feedback_result_dict.get("representative_unnecessary", [])
            ]
        )
    except Exception as e:
        print(f"피드백 분석 중 오류: {e}")
        feedback_result_data = MeetingFeedbackResponseModel(
            necessary_ratio=0.0, unnecessary_ratio=0.0, representative_unnecessary=[]
        )

    print("모든 분석 완료.")
    return FullAnalysisResult(
        meeting_info=meeting_info_data,
        stt_result=stt_response_data, # STT 결과 포함
        summary_result=summary_result_data,
        action_items_result=action_items_result_data,
        feedback_result=feedback_result_data
    )