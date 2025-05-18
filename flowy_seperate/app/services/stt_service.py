# app/services/stt_service.py

from fastapi import UploadFile
import shutil
import os
import uuid
import time
import re
import asyncio
from typing import Any, Optional

# --- 패키지 로드 및 가용성 확인 ---
_PACKAGES_AVAILABLE = False
_PIPELINE_AVAILABLE = False
try:
    from transformers import pipeline as hf_pipeline
    from transformers.pipelines.audio_utils import ffmpeg_read
    from pydub import AudioSegment # 오디오 속성 확인 및 간단한 전처리에 사용 가능
    # import torchaudio # 또는 librosa, soundfile 등 오디오 로딩 라이브러리
    # import torch # pipeline 사용 시 명시적 torch 임포트는 필수는 아님
    _PACKAGES_AVAILABLE = True
    _PIPELINE_AVAILABLE = True # 파이프라인 사용 가능 플래그
    print("STT 서비스: 필수 패키지 (transformers, pydub) 로드 성공.")
except ImportError:
    print("STT 서비스 경고: 필수 패키지 (transformers, pydub 등)를 찾을 수 없습니다. STT 기능이 제한됩니다.")
    # 더미 클래스/함수 정의 (AttributeError 방지용)
    class AudioSegment:
        @staticmethod
        def from_file(dummy, format=None): return AudioSegment()
        def duration_seconds(self): return 0
        # 필요한 다른 더미 메소드 추가 가능

    def ffmpeg_read(bpayload: bytes, sampling_rate: int): # 더미 함수
        return b'' # 빈 바이트 반환


# --- 임시 저장 폴더 설정 ---
TEMP_UPLOAD_DIR = "temp_audio_uploads"
if not os.path.exists(TEMP_UPLOAD_DIR):
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


# === 텍스트 후처리 유틸리티 ===

def _normalize_whitespace_and_punctuation(text: str) -> str:
    """일반적인 공백 및 구두점 정리"""
    if not text: return ""
    # 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    # 구두점 앞 공백 제거 및 뒤 공백 추가 (이미 있다면 유지)
    text = re.sub(r'\s*([.,?!])', r'\1 ', text)
    # 연속된 구두점 정리 (예: "!! " -> "! ") - 간단한 처리
    text = re.sub(r'([.,?!])\1+', r'\1', text)
    # 문장 끝이 아닌 곳의 불필요한 공백 제거
    text = re.sub(r'\s+([.,?!])', r'\1', text) # 예: "단어 ." -> "단어."
    return text.strip()

def _remove_basic_repetitions(text: str, min_repeat_len: int = 3, max_repeat_times: int = 2) -> str:
    """
    단순 반복 단어/짧은 구문 제거 (예: "네 네 네", "그래서 그래서")
    min_repeat_len: 반복으로 간주할 최소 단어/토큰 길이
    max_repeat_times: 이 횟수 이상 반복될 경우 첫 번째만 남김
    """
    if not text: return ""
    # 정규식이 복잡해질 수 있으므로, 여기서는 간단한 로직으로 대체하거나,
    # 기존 stt.py의 정규식을 참고하여 적용 가능
    # 예: "네 네 네" -> "네"
    # 기존: re.sub(r'(\b\w+\b)(\s+\1){2,}', r'\1', filtered_text) # 3번 이상 반복 시 하나로
    # 조금 더 유연하게: max_repeat_times 이상 반복되는 경우
    # 이 부분은 성능과 정확도를 고려하여 더 정교한 알고리즘으로 대체 가능
    # 여기서는 아이디어만 제시하고, 기존 stt.py의 정규식을 활용하는 것을 고려
    
    # 기존 stt.py의 정규식 활용 (단어 2회 초과 반복)
    processed_text = re.sub(r'(\b\w+\b)([\s,.]+\1){' + str(max_repeat_times -1) + r',}', r'\1', text)
    
    # 짧은 구문 반복 (예: "알겠습니다 알겠습니다")
    # 이 부분은 더 정교한 방법 필요 (예: N-gram 분석)
    # 현재는 위 정규식이 어느 정도 커버할 수 있다고 가정
    return processed_text


def _format_time(seconds: float) -> str: # <--- 이 함수 정의 확인!
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def _post_process_transcription(raw_text: str) -> str:
    """
    STT 결과를 후처리하여 가독성을 높이고 오류를 줄입니다.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    # 1. 기본적인 공백 및 구두점 정리
    text = _normalize_whitespace_and_punctuation(raw_text)

    # 2. 기본적인 반복 제거
    text = _remove_basic_repetitions(text, max_repeat_times=2) # 2번 초과 반복(즉, 3번 이상)을 줄임

    # (선택적) 추가적인 후처리 로직:
    # - 문맥에 맞지 않는 짧은 단어 제거 (예: "어", "음" 등 필러 단어 제거 - 단, 의도된 발화일 수 있으므로 주의)
    # - 문장 경계 복원 (만약 Whisper가 문장 구분을 잘 못했을 경우)
    # - 사용자 정의 단어 교정 (자주 오인식되는 단어 목록 기반)

    # 기존 stt.py의 remove_duplicates 함수 (문장 단위 중복/포함 제거) 로직 적용 여부 결정
    # 이 로직은 때때로 너무 많은 내용을 제거할 수 있으므로, 신중하게 적용하거나 파라미터화 필요
    # text = _apply_stt_py_sentence_deduplication(text)

    return text

# (선택적) 기존 stt.py의 문장 단위 중복 제거 로직 (필요시 _post_process_transcription 내부에서 호출)
def _apply_stt_py_sentence_deduplication(text: str) -> str:
    # 기존 stt.py의 remove_duplicates 함수 로직 (약간 수정)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences: return ""

    processed_sentences = []
    last_added_sentence_lower = ""

    for sentence_orig in sentences:
        sentence_strip = sentence_orig.strip()
        if not sentence_strip: continue

        current_sentence_lower = sentence_strip.lower()

        if not processed_sentences: # 첫 문장
            processed_sentences.append(sentence_strip)
            last_added_sentence_lower = current_sentence_lower
            continue

        # 완전 중복
        if current_sentence_lower == last_added_sentence_lower:
            continue

        # 부분 중복 (포함 관계) - 기존 stt.py 로직 (길이 10자 이상 조건은 제외하고 테스트)
        # 이 로직은 때로 과도할 수 있음.
        # if last_added_sentence_lower in current_sentence_lower and len(last_added_sentence_lower) > 5: # 이전이 현재에 포함
        #     processed_sentences[-1] = sentence_strip # 현재 것으로 대체 (더 긴 정보 가정)
        #     last_added_sentence_lower = current_sentence_lower
        #     continue
        # elif current_sentence_lower in last_added_sentence_lower and len(current_sentence_lower) > 5: # 현재가 이전에 포함
        #     continue # 현재 것 추가 안 함

        processed_sentences.append(sentence_strip)
        last_added_sentence_lower = current_sentence_lower
        
    return " ".join(processed_sentences)


# === STT 핵심 처리 함수 (Hugging Face Pipeline 활용) ===
def _perform_stt_with_pipeline( # 동기 함수
    audio_path: str,
    stt_pipeline: Any,
    language: Optional[str] = "ko",       # 기본값 "ko"
    chunk_length_s: int = 30,           # 기본값 30
    stride_length_s: int = 5,            # 기본값 5
) -> str:
    # ... (이전과 동일한 STT 처리 로직, 이 함수는 이미 기본값을 가지고 있음) ...
    if not _PIPELINE_AVAILABLE or stt_pipeline is None:
        print("STT 서비스 경고: ASR 파이프라인 사용 불가. 더미 결과를 반환합니다.")
        time.sleep(0.1)
        return f"[더미 STT(Pipeline): {os.path.basename(audio_path)} (파이프라인/패키지 문제)]"

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"STT 서비스 내부 오류: 오디오 파일 없음 - {audio_path}")

    print(f"  Pipeline STT 처리 시작 (동기): {audio_path}, lang={language}, chunk={chunk_length_s}s, stride={stride_length_s}s")
    try:
        transcription_result = stt_pipeline(
            audio_path,
            chunk_length_s=chunk_length_s, # 함수 파라미터 사용
            stride_length_s=stride_length_s, # 함수 파라미터 사용
            return_timestamps=False,
            # language=language # 파이프라인 생성 시 언어 설정 또는 pipeline 호출 시 task/language 인자 사용 가능
        )
        # 파이프라인 호출 시 language 인자를 직접 지원하는지 확인 필요.
        # whisper 파이프라인의 경우, generate_kwargs를 통해 전달하거나,
        # 파이프라인 생성 시 model_kwargs={"language": language} 등으로 설정 가능.
        # 여기서는 pipeline() 호출 시 직접 language 인자를 받는다고 가정.
        # (만약 아니라면, pipeline 객체 생성 시 또는 generate_kwargs를 통해 설정해야 함)

        raw_text = transcription_result["text"] if isinstance(transcription_result, dict) and "text" in transcription_result else str(transcription_result)

    except Exception as e:
        print(f"STT 서비스 오류: Pipeline 처리 중 예외 - {e}")
        raise RuntimeError(f"ASR Pipeline 처리 중 오류 발생: {e}") from e
    
    final_text = _post_process_transcription(raw_text)
    print(f"  Pipeline STT 처리 완료 (후처리 전 길이: {len(raw_text)}, 후 길이: {len(final_text)})")
    return final_text


# === FastAPI 서비스 인터페이스 함수 (라우터에서 호출됨) ===
async def process_uploaded_rc_file_to_text(
    rc_file: UploadFile,
    stt_pipeline_instance: Any
    # target_language, pipeline_chunk_length_s, pipeline_stride_length_s 파라미터 제거
) -> str:
    # 내부에서 사용할 기본값 설정 (또는 _perform_stt_with_pipeline의 기본값 사용)
    default_target_language = "ko"
    default_chunk_length_s = 30
    default_stride_length_s = 5

    print(f"STT 서비스 수신: '{rc_file.filename}' (타입: {rc_file.content_type}), 언어: {default_target_language} (기본값 사용)") # 로그에 기본값 사용 명시
    service_start_time = time.time()

    # ... (임시 파일 저장 로직은 이전과 동일) ...
    unique_id = uuid.uuid4()
    file_extension = os.path.splitext(rc_file.filename)[1] if rc_file.filename else ".m4a"
    temp_file_name = f"upload_{unique_id}{file_extension}"
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, temp_file_name)
    stt_input_audio_path = temp_file_path
    converted_wav_temp_path = None
    transcribed_text = ""

    try:
        try:
            file_content = await rc_file.read()
            with open(temp_file_path, "wb") as buffer:
                buffer.write(file_content)
            print(f"  임시 파일 저장: {temp_file_path} (크기: {len(file_content)} bytes)")
        finally:
            if hasattr(rc_file, 'file') and rc_file.file and not rc_file.file.closed: # type: ignore
                await rc_file.close()
                print(f"  업로드 파일 핸들({rc_file.filename}) 닫기 완료.")

        if _PACKAGES_AVAILABLE and temp_file_path.lower().endswith((".m4a", ".mp4", ".mov", ".avi", ".flv")): # 지원 필요한 확장자 추가
            print(f"  오디오 파일 감지: {temp_file_path}. WAV로 변환 시도.")
            try:
                wav_temp_file_name = f"converted_{unique_id}.wav"
                converted_wav_temp_path = os.path.join(TEMP_UPLOAD_DIR, wav_temp_file_name)
                
                def convert_to_wav_sync(input_path, output_path, target_sr=16000):
                    # ffmpeg-python을 사용하거나 pydub 사용
                    # 여기서는 pydub 사용 예시 (FFmpeg 필요)
                    audio = AudioSegment.from_file(input_path) # format 명시 안해도 pydub이 추론 시도
                    audio = audio.set_channels(1).set_frame_rate(target_sr)
                    audio.export(output_path, format="wav")
                    print(f"  오디오를 WAV로 변환 완료: {input_path} -> {output_path}")

                await asyncio.to_thread(convert_to_wav_sync, temp_file_path, converted_wav_temp_path)
                stt_input_audio_path = converted_wav_temp_path
            except Exception as e_conv:
                print(f"STT 서비스 경고: 파일을 WAV로 변환 중 오류 발생 ({type(e_conv).__name__}: {e_conv}). 원본 파일로 STT 시도.")
                if converted_wav_temp_path and os.path.exists(converted_wav_temp_path):
                    try: os.remove(converted_wav_temp_path)
                    except Exception: pass
                converted_wav_temp_path = None
                # 원본 파일로 계속 진행 (stt_input_audio_path는 이미 temp_file_path)

        transcribed_text = await asyncio.to_thread(
            _perform_stt_with_pipeline,
            audio_path=stt_input_audio_path,
            stt_pipeline=stt_pipeline_instance,
            language=default_target_language, # 내부 기본값 사용
            chunk_length_s=default_chunk_length_s, # 내부 기본값 사용
            stride_length_s=default_stride_length_s  # 내부 기본값 사용
        )
        
        processing_time = time.time() - service_start_time
        print(f"STT 서비스 완료: '{rc_file.filename}'. 소요시간: {_format_time(processing_time)}")
        return transcribed_text
    # ... (오류 처리 및 finally 블록은 이전과 동일) ...
    except Exception as e:
        print(f"STT 서비스 처리 중 오류 발생 ({rc_file.filename}): {type(e).__name__} - {e}")
        raise
    finally:
        if converted_wav_temp_path and os.path.exists(converted_wav_temp_path):
            try:
                os.remove(converted_wav_temp_path)
                print(f"  임시 변환 WAV 파일 삭제 완료: {converted_wav_temp_path}")
            except Exception as e_remove_wav:
                print(f"STT 서비스 경고: 임시 변환 WAV 파일 삭제 중 오류 '{converted_wav_temp_path}' - {e_remove_wav}")
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"  원본 임시 업로드 파일 삭제 완료: {temp_file_path}")
            except PermissionError as e_perm:
                print(f"STT 서비스 경고: 임시 파일 삭제 실패 (PermissionError) '{temp_file_path}' - {e_perm}")
            except Exception as e_remove:
                print(f"STT 서비스 경고: 임시 파일 삭제 중 일반 오류 '{temp_file_path}' - {e_remove}")