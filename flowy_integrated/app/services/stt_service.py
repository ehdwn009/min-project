# app/services/stt_service.py
import os
import uuid
import time
import re
import asyncio
from typing import Any, Optional, List # List 추가

from fastapi import UploadFile
from pydub import AudioSegment, exceptions as pydub_exceptions

# Hugging Face Transformers 파이프라인 (main.py에서 주입될 인스턴스 사용)

# --- 임시 저장 폴더 설정 ---
TEMP_UPLOAD_DIR = "temp_audio_uploads"
if not os.path.exists(TEMP_UPLOAD_DIR):
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# === 텍스트 후처리 유틸리티 ===

def _normalize_whitespace_and_punctuation(text: str) -> str:
    """일반적인 공백 및 구두점 정리"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나로
    text = re.sub(r'\s*([.,?!])', r'\1 ', text)  # 구두점 앞 공백 제거, 뒤 공백 추가
    text = re.sub(r'([.,?!])\1+', r'\1', text)  # 연속된 구두점 정리 (예: "!! " -> "! ")
    text = re.sub(r'\s+([.,?!])', r'\1', text)  # 단어와 구두점 사이 공백 제거 (예: "단어 ." -> "단어.")
    return text.strip()

def _remove_basic_repetitions(text: str, min_repeat_len: int = 2, max_repeat_times: int = 2) -> str:
    """
    단순 반복 단어/짧은 구문 제거 (예: "네 네 네", "그래서 그래서")
    """
    if not text:
        return ""
    # 단어 단위 반복 제거 (max_repeat_times 초과 시 첫번째만 남김. 예: 3번 반복 -> 1번)
    # ( \b\w+\b ) : 단어 경계로 둘러싸인 하나 이상의 단어 문자
    # ( [\s,.]+\1 ){N,} : 공백/쉼표/마침표 중 하나 이상이 온 후, 앞 그룹(\1)이 N번 이상 반복
    # 여기서 N은 (max_repeat_times - 1) 이상이 되어야 함. 즉, max_repeat_times 이상 반복되는 경우.
    # 예: max_repeat_times = 2 (두 번 초과, 즉 세 번부터) -> N = 1
    #     max_repeat_times = 3 (세 번 초과, 즉 네 번부터) -> N = 2
    # 좀 더 직관적으로, (단어)(공백+단어){2,} -> 단어가 3번 이상 반복되는 경우.
    # 아래 정규식은 (단어) (공백+단어) (공백+단어) ... 형태에서 첫 단어만 남김.
    # (max_repeat_times -1) 만큼의 반복을 찾아서, 그 전체를 첫번째 단어로 치환.
    if max_repeat_times > 1:
        pattern = r'(\b\w+\b)(?:[\s,.]+\1){' + str(max_repeat_times - 1) + r',}'
        processed_text = re.sub(pattern, r'\1', text)
    else: # max_repeat_times 가 1 이하면 반복 제거 안 함 (또는 다른 로직)
        processed_text = text
    return processed_text

def _format_time(seconds: float) -> str:
    """시간(초)을 HH:MM:SS 형식의 문자열로 변환합니다."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def _post_process_transcription(raw_text: str) -> str:
    """STT 결과를 후처리하여 가독성을 높이고 오류를 줄입니다."""
    if not raw_text or not isinstance(raw_text, str):
        return ""

    text = raw_text
    # 1. 기본적인 공백 및 구두점 정리
    text = _normalize_whitespace_and_punctuation(text)
    # 2. 기본적인 반복 제거 (예: 3번 이상 반복되는 단어는 한 번만 남김)
    text = _remove_basic_repetitions(text, max_repeat_times=3) # 3번 이상 반복 시 하나로

    # 추가적인 후처리 로직 (필요에 따라 구현)
    # - 문장 분리 및 재조합 (예: Kiwi 형태소 분석기 사용 - relevance_service.py 참고)
    # - 사용자 정의 단어 교정
    # - 필러 단어(음, 어 등) 제거 (주의: 의도된 발화일 수 있음)

    return text.strip()

# === STT 핵심 처리 함수 (Hugging Face Pipeline 활용) ===
def _perform_stt_with_pipeline_sync(
    audio_path: str,
    stt_pipeline_instance: Any, # <--- 파라미터 이름을 stt_pipeline_instance 로 통일 (또는 호출부와 일치)
    chunk_length_s: int = 30, # 이전 답변에서 추가한 인자들
    stride_length_s: Optional[List[int]] = None # 이전 답변에서 추가한 인자들
    # return_timestamps: bool = False # 이전 답변에서 추가한 인자
) -> str:
    """
    동기적으로 STT 파이프라인을 실행하여 오디오 파일로부터 텍스트를 추출합니다.
    Hugging Face Whisper 파이프라인 사용을 기준으로 합니다.
    """
    # stt_pipeline_instance 가 None 인지 확인 (이름 변경에 따른 수정)
    if stt_pipeline_instance is None:
        print("STT 서비스 경고: ASR 파이프라인이 제공되지 않았습니다.")
        raise ValueError("STT 파이프라인을 사용할 수 없습니다. 초기화에 실패했을 수 있습니다.")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"STT 서비스 내부 오류: 오디오 파일 경로를 찾을 수 없습니다 - {audio_path}")

    print(f"  Pipeline STT 처리 시작 (동기): {audio_path}")
    try:
        # 파이프라인 호출 시, 인자 이름을 stt_pipeline_instance로 사용했으므로 여기서는 그대로 사용
        transcription_result = stt_pipeline_instance( # <--- stt_pipeline_instance 사용
            audio_path,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s, # Whisper는 보통 (left_stride, right_stride) 형태를 사용
            return_timestamps=False # 또는 True (긴 오디오 처리 문제 해결 위해)
        )

        raw_text = transcription_result["text"] if isinstance(transcription_result, dict) and "text" in transcription_result else ""
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)

    except Exception as e:
        print(f"STT 서비스 오류: Pipeline 처리 중 예외 발생 - {e}")
        if "3000 mel input features" in str(e):
             print("긴 오디오 처리 관련 오류입니다. 파이프라인 호출 옵션을 확인하세요.")
        raise RuntimeError(f"ASR Pipeline 처리 중 오류가 발생했습니다: {e}") from e

    final_text = _post_process_transcription(raw_text)
    print(f"  Pipeline STT 처리 완료. 후처리된 텍스트 길이: {len(final_text)}")
    return final_text

# === FastAPI 서비스 인터페이스 함수 (라우터에서 호출됨) ===
async def process_uploaded_rc_file_to_text(
    rc_file: UploadFile,
    stt_pipeline_instance: Any # 이 이름으로 파이프라인 인스턴스를 받음
) -> str:
    """
    업로드된 오디오 파일 (m4a 등)을 텍스트로 변환합니다.
    - 임시 파일 저장
    - m4a 또는 기타 형식을 wav로 변환 (16kHz, 모노 권장)
    - STT 파이프라인으로 텍스트 추출
    - 텍스트 후처리
    - 임시 파일 삭제
    """
    print(f"STT 서비스 수신: '{rc_file.filename}' (타입: {rc_file.content_type})")
    service_start_time = time.time()

    unique_id = uuid.uuid4()
    original_filename = rc_file.filename if rc_file.filename else f"audio_{unique_id}"
    # 업로드된 파일의 확장자를 가져옴
    original_extension = os.path.splitext(original_filename)[1].lower()
    if not original_extension: # 확장자가 없는 경우 기본값 부여 (예: .bin)
        original_extension = ".bin" # 또는 업로드 Content-Type 기반으로 추론

    temp_uploaded_file_name = f"upload_{unique_id}{original_extension}"
    temp_uploaded_file_path = os.path.join(TEMP_UPLOAD_DIR, temp_uploaded_file_name)

    # 최종적으로 STT에 입력될 WAV 파일 경로 (변환 시 이 경로 사용)
    wav_input_for_stt_path = os.path.join(TEMP_UPLOAD_DIR, f"stt_input_{unique_id}.wav")
    # 변환 전 원본 파일이 이미 wav이고 조건에 맞으면 바로 이 경로에 저장될 수도 있음

    cleanup_paths: List[str] = [] # 처리 후 삭제할 파일 목록

    try:
        # 1. 원본 오디오 파일 임시 저장
        try:
            file_content = await rc_file.read()
            with open(temp_uploaded_file_path, "wb") as buffer:
                buffer.write(file_content)
            cleanup_paths.append(temp_uploaded_file_path)
            print(f"  원본 임시 파일 저장 완료: {temp_uploaded_file_path} (크기: {len(file_content)} bytes)")
        finally:
            if hasattr(rc_file, 'close') and callable(rc_file.close):
                await rc_file.close()
                print(f"  업로드 파일 핸들({original_filename}) 닫기 완료.")

        # 2. 오디오 파일을 STT가 선호하는 WAV 형식으로 변환 (예: 16kHz, 모노)
        # pydub은 ffmpeg/libav가 설치되어 있어야 m4a 등 다양한 포맷 지원
        target_sr = 16000 # Whisper가 선호하는 샘플링 레이트

        try:
            print(f"  오디오 파일 형식 변환 시도: {temp_uploaded_file_path} -> {wav_input_for_stt_path}")
            # pydub.exceptions.CouldntDecodeError 핸들링 추가
            audio_segment = AudioSegment.from_file(temp_uploaded_file_path) # format 명시 안해도 자동 추론 시도
            
            # 샘플링 레이트 및 채널 변경 (모노)
            audio_segment = audio_segment.set_frame_rate(target_sr)
            audio_segment = audio_segment.set_channels(1)
            
            # WAV 파일로 저장
            audio_segment.export(wav_input_for_stt_path, format="wav")
            cleanup_paths.append(wav_input_for_stt_path)
            print(f"  오디오를 WAV로 변환 완료: {wav_input_for_stt_path}")
            stt_ready_audio_path = wav_input_for_stt_path
        except pydub_exceptions.CouldntDecodeError as e_decode:
            print(f"STT 서비스 오류: 오디오 파일 디코딩 실패 ({temp_uploaded_file_path}). FFmpeg/libav 설치 확인 필요. 오류: {e_decode}")
            # 원본 파일을 그대로 사용 시도 (만약 Whisper가 지원한다면) 또는 오류 발생
            # 여기서는 변환 실패 시 오류 발생시킴 (안정적인 처리를 위해)
            raise ValueError(f"오디오 파일({original_filename})을 처리할 수 없습니다. 지원되지 않는 형식이거나 파일이 손상되었을 수 있습니다.") from e_decode
        except Exception as e_conv:
            print(f"STT 서비스 오류: 오디오 파일 변환 중 예외 발생 ({type(e_conv).__name__}: {e_conv}).")
            raise RuntimeError(f"오디오 파일 변환 중 서버 내부 오류 발생: {e_conv}") from e_conv


        # 3. STT 파이프라인으로 텍스트 추출 (동기 함수를 비동기 컨텍스트에서 실행)
        # _perform_stt_with_pipeline_sync는 CPU bound 작업이므로 asyncio.to_thread 사용
        transcribed_text = await asyncio.to_thread(
            _perform_stt_with_pipeline_sync,
            audio_path=stt_ready_audio_path,
            stt_pipeline_instance=stt_pipeline_instance,
        )

        processing_time = time.time() - service_start_time
        print(f"STT 서비스 전체 처리 완료: '{original_filename}'. 소요시간: {_format_time(processing_time)}")
        return transcribed_text

    except Exception as e:
        # 라우터에서 최종적으로 처리할 수 있도록 예외를 다시 발생시킴
        print(f"STT 서비스 처리 중 예외 발생 ({original_filename}): {type(e).__name__} - {e}")
        raise # 발생한 예외를 그대로 다시 raise
    finally:
        # 4. 임시 파일 삭제
        for path_to_delete in cleanup_paths:
            if os.path.exists(path_to_delete):
                try:
                    os.remove(path_to_delete)
                    print(f"  임시 파일 삭제 완료: {path_to_delete}")
                except PermissionError as e_perm: # 파일이 사용 중일 경우   
                    print(f"STT 서비스 경고: 임시 파일 삭제 실패 (PermissionError) '{path_to_delete}' - {e_perm}. 수동 삭제 필요.")
                except Exception as e_remove:
                    print(f"STT 서비스 경고: 임시 파일 삭제 중 일반 오류 '{path_to_delete}' - {e_remove}")