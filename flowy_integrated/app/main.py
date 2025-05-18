# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import Any # FastAPI 인스턴스 타입 힌팅용

from app.routers import analysis # 라우터 임포트
# 의존성 함수는 lifespan에서만 사용 (dependencies.py에서 가져옴)
from app.dependencies import initialize_stt_pipeline, get_openai_client as dep_get_openai_client

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # 애플리케이션 시작 시 실행될 코드
    print("애플리케이션 시작 (lifespan): 초기화 시도 (main.py)")
    initialize_stt_pipeline()
    dep_get_openai_client()
    yield
    # 애플리케이션 종료 시 실행될 코드 (필요시)
    print("애플리케이션 종료 (lifespan). (main.py)")

app = FastAPI(
    title="Flowy 회의록 분석 및 알림 API",
    description="음성 또는 텍스트로 입력된 회의록을 분석하고 결과를 제공하는 API입니다.",
    version="0.1.0",
    lifespan=lifespan # lifespan 이벤트 핸들러 등록
)

# API 라우터 등록
app.include_router(analysis.router, prefix="/api/v1", tags=["Meeting Analysis"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Flowy 회의록 분석 API에 오신 것을 환영합니다! (v0.1.0)"}