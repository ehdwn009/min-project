from fastapi import FastAPI, Depends, HTTPException
from openai import OpenAI
from contextlib import asynccontextmanager
from typing import Any
import time
# import traceback

from app.core.config import settings
# from app.routers import stt, summarization, action_assignment, feedback # <--- 이 줄을 아래로 이동

app_state: dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... (lifespan 함수 내용은 이전과 동일) ...
    print("애플리케이션 시작: 리소스를 초기화합니다...")
    # ... (OpenAI 클라이언트 및 STT 파이프라인 초기화 로직) ...
    print("  OpenAI 클라이언트 초기화 중...")
    try:
        app_state["openai_client"] = OpenAI(api_key=settings.OPENAI_API_KEY)
        print("  OpenAI 클라이언트 초기화 완료.")
    except Exception as e:
        print(f"  OpenAI 클라이언트 초기화 실패: {e}")
        app_state["openai_client"] = None

    print("  STT (Whisper) 파이프라인 초기화 중...")
    stt_pipeline_instance = None
    try:
        from transformers import pipeline as hf_pipeline
        stt_model_name = "openai/whisper-large-v3"
        device_to_use = "cuda:0" # 사용자 설정 반영
        print(f"    모델: {stt_model_name}, 장치: {device_to_use} 로 STT 파이프라인 생성 시도...")
        stt_pipeline_instance = hf_pipeline(
            "automatic-speech-recognition",
            model=stt_model_name,
            device=device_to_use
        )
        app_state["stt_pipeline"] = stt_pipeline_instance
        print(f"  STT (Whisper) 파이프라인 '{stt_model_name}' 초기화 완료 (장치: {device_to_use}).")
    except ImportError as e_import:
        print(f"  STT 파이프라인 초기화 실패 (필수 패키지 임포트 오류): {e_import}")
        app_state["stt_pipeline"] = None
    except Exception as e_pipeline:
        print(f"  STT 파이프라인 초기화 중 일반 오류 발생: {e_pipeline}")
        app_state["stt_pipeline"] = None
    yield
    print("애플리케이션 종료: 리소스를 정리합니다...")
    app_state.clear()
    print("애플리케이션 리소스 정리 완료.")


app = FastAPI(
    title="회의 분석 API (Meeting Analyzer API)",
    description="...",
    version="0.1.0",
    lifespan=lifespan
)

# --- 의존성 주입 함수들 정의 ---
def get_openai_client() -> OpenAI:
    client = app_state.get("openai_client")
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI 서비스 사용 불가 (초기화 실패)")
    return client

def get_stt_pipeline() -> Any:
    pipeline = app_state.get("stt_pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="STT 서비스 사용 불가 (모델 로드 실패)")
    return pipeline

# --- 라우터 임포트 (모든 주요 객체 및 함수 정의 이후) ---
from app.routers import stt, summarization, action_assignment, feedback # <--- 위치 변경

app.include_router(stt.router, prefix="/api/stt", tags=["1. 음성-텍스트 변환 (STT)"])
app.include_router(summarization.router, prefix="/api/summarization", tags=["2. 텍스트 요약"])
app.include_router(action_assignment.router, prefix="/api/action-assignment", tags=["3. 역할 및 할 일 분배"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["4. 회의 내용 피드백"])

@app.get("/", tags=["Root"])
async def read_root():
    # ... (이전과 동일) ...
    stt_status = "사용 가능" if app_state.get("stt_pipeline") else "사용 불가 (초기화 실패 또는 진행 중)"
    openai_status = "사용 가능" if app_state.get("openai_client") else "사용 불가 (초기화 실패 또는 진행 중)"
    return {
        "message": "회의 분석 API에 오신 것을 환영합니다! /docs 경로에서 API 문서를 확인하세요.",
        "stt_service_status": stt_status,
        "openai_service_status": openai_status,
        "current_time_kst": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time()))
    }

# CORS (필요시)