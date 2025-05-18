# app/routers/stt.py
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException # Form 제거 가능
from typing import Any # Optional 제거 가능

# 의존성 주입 함수
from app.main import get_stt_pipeline

# 서비스 함수
from app.services.stt_service import process_uploaded_rc_file_to_text

# 응답 모델
from app.models.meeting import STTResponse

router = APIRouter()

@router.post("/transcribe", response_model=STTResponse)
async def transcribe_audio_endpoint(
    # 파일 업로드 (필수)
    rc_file: UploadFile = File(..., description="변환할 음성 파일 (m4a, wav, mp3 등)"),

    # target_language, pipeline_chunk_length_s, pipeline_stride_length_s 파라미터 제거
    # 만약 다른 메타데이터 (subj 등)도 Form으로 받고 있었다면,
    # 파일만 받는 엔드포인트에서는 해당 파라미터도 제거하거나,
    # 요청 본문을 JSON으로 받는 다른 엔드포인트를 고려해야 할 수 있습니다.
    # 지금은 파일만 받는 것으로 단순화합니다.

    stt_pipeline: Any = Depends(get_stt_pipeline)
):
    """
    업로드된 음성 파일 (`rc_file`)을 텍스트로 변환합니다.
    STT 처리 설정은 서버 내부 기본값을 사용합니다.
    """
    # MIME 타입 체크는 유지하거나 필요에 따라 조정
    if not rc_file.content_type or not (
        rc_file.content_type.startswith("audio/") or
        rc_file.content_type in ["application/octet-stream", "video/mp4"] # 다양한 오디오 컨테이너 허용
    ):
        print(f"STT 라우터 경고: 예상치 못한 파일 타입 수신 - '{rc_file.filename}' (타입: {rc_file.content_type})")
        # 실제 서비스에서는 여기서 HTTPException을 발생시켜 잘못된 파일 타입 거부 가능
        # raise HTTPException(status_code=400, detail="지원하지 않는 오디오 파일 형식입니다.")

    print(f"STT 라우터: 파일 수신 시작 - '{rc_file.filename}' (내부 기본 설정 사용)")

    try:
        transcribed_text = await process_uploaded_rc_file_to_text(
            rc_file=rc_file,
            stt_pipeline_instance=stt_pipeline
            # target_language, pipeline_chunk_length_s, pipeline_stride_length_s 인자 제거
        )

        if transcribed_text is None or transcribed_text.strip() == "":
            return STTResponse(rc_txt="", message="음성 파일에서 인식된 텍스트가 없습니다.")

        print(f"STT 라우터: '{rc_file.filename}' 변환 성공.")
        return STTResponse(
            rc_txt=transcribed_text,
            message="오디오 파일이 성공적으로 텍스트로 변환되었습니다."
        )
    # ... (오류 처리 로직은 이전과 동일) ...
    except FileNotFoundError as e_fnf:
        print(f"STT 라우터 오류 (FileNotFoundError): {e_fnf}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: 파일 처리 중 문제가 발생했습니다. ({e_fnf})")
    except ValueError as e_val:
        print(f"STT 라우터 오류 (ValueError): {e_val}")
        raise HTTPException(status_code=400, detail=f"입력 데이터 오류: {e_val}")
    except RuntimeError as e_rt:
        print(f"STT 라우터 오류 (RuntimeError): {e_rt}")
        raise HTTPException(status_code=503, detail=f"서비스 처리 중 오류가 발생했습니다: {e_rt}")
    except HTTPException as e_http:
        raise e_http
    except Exception as e:
        print(f"STT 라우터 알 수 없는 오류: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail="알 수 없는 서버 오류가 발생했습니다.")