from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment
import numpy as np
import torch
import os
import time
import gc
import json
import re
import math

def format_time(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def get_gpu_memory():
    """GPU 메모리 사용량 확인 (GB 단위)"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        return total_memory / (1024 ** 3)
    else:
        return 0

def normalize_audio(audio_segment, target_dBFS=-20.0):
    """오디오 볼륨 정규화"""
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dBFS)

def split_audio_chunks_with_overlap(audio, chunk_length_sec, overlap_sec):
    """
    오디오를 청크로 분할하되, 오버랩을 추가하여 경계 정보 손실 방지
    """
    chunk_ms = int(chunk_length_sec * 1000)  # 밀리초로 변환하고 정수로 캐스팅
    overlap_ms = int(overlap_sec * 1000)     # 밀리초로 변환하고 정수로 캐스팅
    step_ms = chunk_ms - overlap_ms
    chunks = []
    
    # range에 float 대신 int 사용
    for start in range(0, len(audio), step_ms):
        end = start + chunk_ms
        chunk = audio[start:min(end, len(audio))]
        chunks.append(chunk)
        if end >= len(audio):
            break
    
    return chunks

def remove_duplicates(text):
    """중복 문장 및 단어 제거"""
    # 문장 단위 중복 제거
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    
    for i, sentence in enumerate(sentences):
        # 완전 중복 제거
        if i > 0 and sentence.strip().lower() == sentences[i-1].strip().lower():
            continue
            
        # 부분 중복 확인 (한 문장이 다른 문장에 포함되는 경우)
        if i > 0 and len(sentence) > 10 and len(sentences[i-1]) > 10:
            prev = sentences[i-1].strip().lower()
            curr = sentence.strip().lower()
            
            # 이전 문장이 현재 문장에 포함되는지
            if prev in curr:
                # 현재 문장이 더 길면 이전 문장을 대체
                result[-1] = sentence
                continue
            # 현재 문장이 이전 문장에 포함되는지
            elif curr in prev:
                # 이전 문장이 더 길면 현재 문장 생략
                continue
                
        result.append(sentence)
    
    # 결과 결합
    filtered_text = ' '.join(result)
    
    # 반복 단어 제거 (예: "네, 네, " -> "네, ")
    filtered_text = re.sub(r'(\b\w+\b)(\s+\1){2,}', r'\1', filtered_text)
    
    # 반복 구문 제거 추가
    filtered_text = re.sub(r'([^.!?]{10,})\s+\1', r'\1', filtered_text)
    
    return filtered_text

def get_overlap_ratio(str1, str2):
    """두 문자열의 유사도 계산"""
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    
    if not words1 or not words2:
        return 0
        
    common_words = words1.intersection(words2)
    return len(common_words) / max(len(words1), len(words2))

def remove_repeated_phrases(text):
    """연속적으로 반복되는 구절 제거"""
    words = text.split()
    result = []
    i = 0
    
    while i < len(words):
        if i + 4 < len(words):  # 최소 4단어 이상의 구절 체크
            phrase = ' '.join(words[i:i+4])
            # 다음 위치에서 같은 구절이 나오는지 확인
            next_text = ' '.join(words[i+4:])
            if next_text.startswith(phrase):
                # 중복 건너뛰기 (추가하지 않음)
                i += 4
                continue
        
        result.append(words[i])
        i += 1
    
    return ' '.join(result)

def transcribe_audio_file(audio_path, chunk_length=10, batch_size=1, 
                        overlap_sec=2, save_chunks=True):
    """
    오디오 파일을 텍스트로 변환 (Large 모델 최적화 버전)
    
    Args:
        audio_path: 오디오 파일 경로
        chunk_length: 청크 길이(초)
        batch_size: 배치 크기
        overlap_sec: 청크 간 오버랩 시간(초)
        save_chunks: 청크별 결과 저장 여부
    
    Returns:
        변환된 텍스트
    """
    # 파일 존재 확인
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")

    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 중인 장치: {device}")
    
    # 모델 경로 설정 (Large 모델 고정)
    model_path = "openai/whisper-large"
    
    # GPU 메모리 확인
    gpu_memory_gb = get_gpu_memory()
    print(f"배치 크기: {batch_size}, 사용 가능한 GPU 메모리: {gpu_memory_gb:.2f}GB")
    
    # 모델 로딩
    print("모델 로딩 중...")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # 한국어 강제 설정
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    
    model = model.to(device)
    
    # 오디오 파일 불러오기
    print("오디오 파일 로딩 중...")
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    # 볼륨 정규화 적용 (오디오 품질 향상)
    audio = normalize_audio(audio)
    
    # 전체 오디오 길이 계산
    total_duration = len(audio) / 1000  # 밀리초를 초로 변환
    print(f"전체 오디오 길이: {format_time(total_duration)}")
    
    print(f"사용할 청크 길이: {chunk_length}초, 오버랩: {overlap_sec}초")
    
    # 오버랩이 있는 오디오 분할
    chunks = split_audio_chunks_with_overlap(audio, chunk_length, overlap_sec)
    print(f"총 청크 수: {len(chunks)}")
    
    full_transcription = []
    chunk_results = [] if save_chunks else None
    previous_text = ""
    start_time = time.time()
    
    # 배치 처리
    print("\n음성 인식 진행 중...")
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # 재시도 로직 추가
        retry_count = 0
        max_retries = 2
        
        while retry_count <= max_retries:
            try:
                # 배치의 오디오 배열 변환
                batch_arrays = [
                    np.array(chunk.get_array_of_samples()).astype(np.float32) / (1 << 15)
                    for chunk in batch_chunks
                ]
                
                # Whisper 입력 생성
                inputs = processor(
                    batch_arrays,
                    sampling_rate=16000,
                    return_tensors="pt",
                    return_attention_mask=True
                )
                
                # GPU로 데이터 이동
                input_features = inputs.input_features.to(device)
                attention_mask = inputs.attention_mask.to(device)
                
                # Large 모델 최적 생성 파라미터
                gen_kwargs = {
                    "input_features": input_features,
                    "attention_mask": attention_mask,
                    "max_length": 512,
                    "num_beams": 5,
                    "temperature": 0.15,  # 낮은 온도
                    "length_penalty": 1.0,  # 긴 문장 생성 유도
                    "no_repeat_ngram_size": 2,
                    "language": "ko"
                }
                
                # 텍스트 생성
                predicted_ids = model.generate(**gen_kwargs)
                batch_transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
                
                # 개선된 청크 경계 처리
                if previous_text and batch_transcriptions:
                    prev_words = previous_text.split()[-7:]  # 비교 범위 확대
                    curr_words = batch_transcriptions[0].split()[:7]
                    
                    # 공통 단어 시퀀스 찾기
                    max_overlap = 0
                    overlap_idx = 0
                    
                    for j in range(1, min(len(prev_words), len(curr_words)) + 1):
                        if prev_words[-j:] == curr_words[:j]:
                            max_overlap = j
                            overlap_idx = j
                    
                    # 공통 부분 제거
                    if max_overlap > 0:
                        new_text = " ".join(curr_words[overlap_idx:] + (batch_transcriptions[0].split()[7:] 
                                            if len(curr_words) >= 7 else []))
                        
                        # 빈 문자열이 되지 않도록 방지
                        if new_text.strip():
                            batch_transcriptions[0] = new_text
                
                # 디버깅: 각 청크 결과 기록
                if save_chunks:
                    for j, text in enumerate(batch_transcriptions):
                        chunk_idx = i + j
                        # float가 아닌 int 사용하도록 수정
                        step_sec = chunk_length - overlap_sec
                        chunk_start = chunk_idx * step_sec
                        chunk_end = chunk_start + chunk_length
                        
                        chunk_info = {
                            "chunk_idx": chunk_idx,
                            "time": f"{format_time(chunk_start)} - {format_time(min(chunk_end, total_duration))}",
                            "text_length": len(text),
                            "text": text
                        }
                        chunk_results.append(chunk_info)
                
                # 전체 변환 결과에 추가
                full_transcription.extend(batch_transcriptions)
                
                if batch_transcriptions:
                    previous_text = batch_transcriptions[-1]
                
                # 진행률 표시
                progress = (i + len(batch_chunks)) / len(chunks) * 100
                elapsed = time.time() - start_time
                remain = (elapsed / progress * 100 - elapsed) if progress > 0 else 0
                print(f"\r진행률: {progress:.1f}% | 남은 시간: {format_time(remain)}", end="")
                
                # 메모리 정리
                del batch_arrays, inputs, input_features, attention_mask, predicted_ids
                if device == "cuda": 
                    torch.cuda.empty_cache()
                gc.collect()
                
                # 메모리 안정화를 위한 대기
                time.sleep(0.5)
                
                # 성공적으로 처리 완료
                break
                
            except torch.cuda.OutOfMemoryError:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"\n메모리 부족으로 청크 {i+1} 처리 실패")
                    full_transcription.extend([f"[메모리 부족으로 인식 실패]"] * len(batch_chunks))
                    break
                    
                print(f"\nGPU 메모리 부족, 메모리 정리 후 재시도 중 ({retry_count}/{max_retries})")
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(1)
                
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"\n청크 {i+1} 처리 실패: {e}")
                    full_transcription.extend([f"[인식 실패]"] * len(batch_chunks))
                    break
                    
                print(f"\n청크 처리 중 오류, 재시도 {retry_count}/{max_retries}: {e}")
                if device == "cuda": 
                    torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.5)
    
    # 디버깅용 청크별 결과 저장
    if save_chunks and chunk_results:
        debug_file = "chunk_results_large.json"
        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(chunk_results, f, ensure_ascii=False, indent=2)
        print(f"\n청크별 결과가 {debug_file}에 저장되었습니다.")
    
    # 후처리: 이어붙이기, 공백/중복 정리
    raw_text = " ".join(full_transcription).strip()
    
    # 1. 구두점 정리
    txt = raw_text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
    txt = " ".join(txt.split())
    
    # 2. 연속된 중복 구절 먼저 제거
    txt = remove_repeated_phrases(txt)
    
    # 3. 중복 문장 제거
    txt = remove_duplicates(txt)
    
    print("\n처리 완료!")
    return txt

if __name__ == "__main__":
    audio_path = "야야야.m4a"
    
    # Large 모델 최적 설정
    chunk_length = 20   # 청크 길이 (10초)
    batch_size = 1       # 배치 크기 (1)
    overlap_sec = 4  # 오버랩 (2초, 정수로 변경)
    save_chunks = True   # 디버깅용 청크별 결과 저장
    
    print("\n[음성 인식 시작 - 모델: large]")
    try:
        result = transcribe_audio_file(
            audio_path, 
            chunk_length, 
            batch_size,
            overlap_sec,
            save_chunks
        )
        
        # 결과 저장
        out_file = "transcript_large.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\n결과가 {out_file}에 저장되었습니다.")
        
        # 결과 출력
        print("\n[음성 인식 결과]")
        print(result)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")