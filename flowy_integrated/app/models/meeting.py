# app/models/meeting.py
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Any

# --- 공통 모델 ---
class AttendeeInfo(BaseModel):
    name: str = Field(..., description="참여자 이름")
    email: Optional[EmailStr] = Field(None, description="참여자 이메일")
    role: Optional[str] = Field(None, description="참여자 역할")

# --- Form으로 받을 JSON 문자열 내부 구조를 위한 모델 ---
class MeetingMetadata(BaseModel):
    subj: str = Field(..., description="회의 주제")
    dt: str = Field(..., description="회의 일시 (YYYY-MM-DDTHH:MM:SS)")
    loc: str = Field(..., description="회의 장소")
    info_n: List[AttendeeInfo] = Field(..., description="참석자 정보 리스트")

# --- STT 관련 모델 ---
class STTResponse(BaseModel):
    rc_txt: str = Field(..., description="변환된 텍스트")
    message: Optional[str] = Field(None, description="처리 메시지")

# --- 요약 관련 모델 ---
class SummarizationResponse(BaseModel):
    summary: List[str] = Field(default_factory=list, description="요약 결과 리스트")
    message: Optional[str] = Field(None, description="처리 메시지")

# --- 할 일 분배 관련 모델 ---
class ActionItemByAssignee(BaseModel):
    name: str = Field(..., description="담당자 이름")
    role: Optional[str] = Field(None, description="담당자 역할")
    tasks: List[str] = Field(default_factory=list, description="할 일 목록")

class ActionAssignmentResponse(BaseModel):
    tasks: List[ActionItemByAssignee] = Field(default_factory=list, description="담당자별 할 일 목록")
    message: Optional[str] = Field(None, description="처리 메시지")

# --- 피드백 관련 모델 ---
class RepresentativeUnnecessarySentenceModel(BaseModel):
    sentence: str = Field(..., description="대표 불필요 문장")
    reason: str = Field(..., description="불필요 이유")

class MeetingFeedbackResponseModel(BaseModel):
    necessary_ratio: float = Field(0.0, description="필요 문장 비율 (%)")
    unnecessary_ratio: float = Field(0.0, description="불필요 문장 비율 (%)")
    representative_unnecessary: List[RepresentativeUnnecessarySentenceModel] = Field(
        default_factory=list, description="대표 불필요 문장 목록"
    )

# --- 통합 분석 결과 모델 (API 응답용) ---
class FullAnalysisResult(BaseModel):
    meeting_info: MeetingMetadata
    stt_result: Optional[STTResponse] = None # STT 결과를 응답에 포함 (주석 해제)
    summary_result: SummarizationResponse
    action_items_result: ActionAssignmentResponse
    feedback_result: MeetingFeedbackResponseModel