# app/services/summarizer_service.py
import json
import asyncio
from typing import List, Optional, Dict, Any # Dict, Any 추가
from openai import OpenAI # 직접 OpenAI 사용
import re

# app.core.config 에서 settings (API 키, 모델명 등)를 직접 사용하지 않고,
# 라우터 레벨에서 주입된 openai_client 와 model_name 을 사용합니다.
# 만약 서비스 레벨에서 직접 settings 를 참조하려면 아래와 같이 import
# from app.core.config import settings

# 모델 정의는 app/models/meeting.py 에 있으므로 여기서는 직접 참조하지 않음
# 반환 타입은 List[str] (요약된 문장들의 리스트)

async def get_meeting_summary(
    openai_client: OpenAI,
    rc_txt: str,
    subj: Optional[str],
    model_name: str
) -> List[str]:
    """
    OpenAI API를 사용하여 회의록 텍스트를 요약합니다.

    Args:
        openai_client: 미리 초기화된 OpenAI 클라이언트 인스턴스.
        rc_txt: 요약할 원본 회의록 텍스트.
        subj: 회의 주제 (요약 품질 향상에 도움).
        model_name: 사용할 OpenAI 모델 이름.

    Returns:
        요약된 내용의 불렛 포인트 리스트 (List[str]).
        오류 발생 시 빈 리스트 또는 예외를 발생시킬 수 있음 (현재는 빈 리스트 반환 경향).

    Raises:
        ValueError: 입력 값 (rc_txt)이 유효하지 않은 경우.
        RuntimeError: OpenAI API 호출 중 오류 발생 시.
    """
    if not openai_client:
        # 이 경우는 보통 Depends 로 주입되므로 발생하기 어려움
        print("Summarizer Service 오류: OpenAI 클라이언트가 제공되지 않았습니다.")
        raise ValueError("OpenAI 클라이언트가 필요합니다.")
    if not rc_txt or not rc_txt.strip():
        print("Summarizer Service 경고: 요약할 텍스트(rc_txt)가 비어있습니다. 빈 요약을 반환합니다.")
        return [] # 빈 텍스트에 대해서는 빈 요약 반환

    topic_info = f"회의 주제는 '{subj}'입니다." if subj else "회의 주제는 제공되지 않았습니다."
    max_summary_points = 5 # 원하는 요약 항목 수 (조절 가능)

    prompt = f"""
다음 회의록 내용을 핵심적인 내용만 간추려 {max_summary_points}개 이내의 주요 불렛 포인트로 요약해주십시오.
각 불렛 포인트는 완전한 문장으로 명확하고 간결하게 작성해주세요.
{topic_info}

[회의록 전문]
{rc_txt}

[요약 결과 (불렛 포인트 리스트 형식)]
"""

    print(f"Summarizer Service: 요약 요청 시작 (모델: {model_name}, 주제: '{subj or '없음'}', 텍스트 길이: {len(rc_txt)})")

    try:
        # OpenAI API 호출 (비동기 처리를 위해 asyncio.to_thread 사용)
        # chat.completions.create는 동기 함수이므로, 비동기 컨텍스트에서 적절히 호출
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=model_name,
            messages=[
                {"role": "system", "content": "당신은 회의록을 분석하여 핵심 내용을 불렛 포인트로 요약하는 전문가입니다. 각 요약 항목은 명확하고 간결한 문장으로 제공해야 합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 약간의 창의성을 허용하되 일관성 유지
            max_tokens=500,   # 요약 결과의 최대 토큰 수 (모델별 제한 확인 필요)
            # response_format={"type": "json_object"} # JSON 출력을 강제할 경우, 프롬프트도 JSON 구조를 명시해야 함
                                                    # 여기서는 텍스트 리스트를 기대하므로 일반 텍스트 응답 후 파싱
        )

        raw_summary_text = response.choices[0].message.content
        if raw_summary_text is None:
            print("Summarizer Service 오류: LLM으로부터 비어있는 응답(content=None)을 받았습니다.")
            return [] # 또는 적절한 예외 발생

        # LLM 응답 파싱: 불렛 포인트 형태의 텍스트를 리스트로 변환
        # 예시: "- 항목1\n- 항목2\n* 항목3" 과 같은 응답을 가정
        summary_points = []
        # 응답이 여러 줄로 오고, 각 줄이 요약 항목이라고 가정
        # 또는 LLM이 "1. 내용\n2. 내용" 등으로 반환할 수도 있음. 프롬프트에 명시하는 것이 좋음.
        # 여기서는 간단히 줄바꿈으로 분리하고, 빈 줄이나 마커 제거
        raw_lines = raw_summary_text.strip().split('\n')
        for line in raw_lines:
            cleaned_line = line.strip()
            # 불렛 포인트 마커 제거 (-, *, 숫자. 등)
            cleaned_line = re.sub(r"^[-\*\d]+\.?\s*", "", cleaned_line)
            if cleaned_line: # 내용이 있는 줄만 추가
                summary_points.append(cleaned_line)
        
        # 요청한 최대 항목 수 제한 (선택적)
        if len(summary_points) > max_summary_points:
             summary_points = summary_points[:max_summary_points]

        if not summary_points:
            print("Summarizer Service 경고: LLM 응답에서 유효한 요약 항목을 추출하지 못했습니다.")
            # LLM이 기대와 다른 형식으로 응답했을 수 있음. 원본 응답을 로깅하는 것이 좋음.
            # print(f"LLM 원본 요약 응답: {raw_summary_text}")

        print(f"Summarizer Service: 요약 생성 완료. {len(summary_points)}개 항목.")
        return summary_points

    except Exception as e:
        # API 호출 실패, 응답 파싱 오류 등 모든 예외 포함
        print(f"Summarizer Service 오류: OpenAI API 호출 또는 응답 처리 중 예외 발생 - {type(e).__name__}: {e}")
        # 라우터에서 이 오류를 잡아서 적절한 HTTP 응답을 생성하도록 함
        raise RuntimeError(f"회의 요약 생성 중 오류가 발생했습니다: {e}") from e