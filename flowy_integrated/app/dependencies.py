# app/dependencies.py
from fastapi import HTTPException
from openai import OpenAI
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSpeechSeq2Seq
import torch
from typing import Any, Optional

from app.core.config import settings # 설정 파일 임포트

# --- OpenAI 클라이언트 의존성 주입 ---
_openai_client: Optional[OpenAI] = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not settings.OPENAI_API_KEY:
            # print("오류: OPENAI_API_KEY가 설정되지 않았습니다. (dependencies.py)") # 로그 위치 명시
            raise HTTPException(status_code=500, detail="OpenAI API 키가 설정되지 않았습니다.")
        try:
            _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            # print("OpenAI 클라이언트가 성공적으로 초기화되었습니다. (dependencies.py)")
        except Exception as e:
            # print(f"OpenAI 클라이언트 초기화 실패: {e} (dependencies.py)")
            raise HTTPException(status_code=500, detail=f"OpenAI 클라이언트 초기화 중 오류 발생: {str(e)}")
    return _openai_client

# --- STT (Speech-To-Text) 파이프라인 의존성 주입 ---
_stt_pipeline_instance: Optional[Any] = None
_stt_model_name = "openai/whisper-large-v3" # config.py 에서 관리하는 것이 더 좋음

def initialize_stt_pipeline():
    global _stt_pipeline_instance
    if _stt_pipeline_instance is not None: # 이미 초기화되었으면 반환
        return _stt_pipeline_instance
    try:
        # print(f"STT 파이프라인 ({_stt_model_name}) 초기화 시도... (dependencies.py)")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # print(f"STT 파이프라인을 위한 장치: {device} (dependencies.py)")
        _stt_pipeline_instance = hf_pipeline(
            "automatic-speech-recognition",
            model=_stt_model_name,
            device=device,
            chunk_length_s=30, # 긴 오디오 처리
            stride_length_s=[5, 5] # 긴 오디오 처리
        )
        # print(f"STT 파이프라인 ({_stt_model_name}) 초기화 성공. (dependencies.py)")
    except ImportError:
        # print("STT 관련 패키지가 설치되지 않았습니다. STT 기능을 사용할 수 없습니다. (dependencies.py)")
        _stt_pipeline_instance = None
    except Exception as e:
        # print(f"STT 파이프라인 ({_stt_model_name}) 초기화 실패: {e} (dependencies.py)")
        _stt_pipeline_instance = None
    return _stt_pipeline_instance

def get_stt_pipeline():
    if _stt_pipeline_instance is None:
        return initialize_stt_pipeline()
    return _stt_pipeline_instance