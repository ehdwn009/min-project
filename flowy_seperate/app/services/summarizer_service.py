# app/services/summarizer_service.py
import json
from typing import List, Optional
from openai import OpenAI
import asyncio

# (질문) 회의 주제(subj)가 없을 경우 프롬프트 처리:
# 현재는 "* 회의 주제: (제공되지 않음)"으로 되어 있습니다.
# 이대로 괜찮으실까요, 아니면 프롬프트에서 해당 라인을 아예 생략하는 것이 좋을까요?
# (LLM은 "제공되지 않음"이라는 정보 자체를 바탕으로 응답을 생성할 수도 있습니다.)
# ==> 답변: 우선 현재 방식 유지. 추후 필요시 변경.

async def get_meeting_summary(
    openai_client: OpenAI,
    rc_txt: str, # full_transcript_text -> rc_txt
    subj: Optional[str], # meeting_topic -> subj
    model_name: str
) -> List[str]: # 반환 타입은 요약 문자열 리스트 (SummarizationResponse의 'summary' 필드용)
    if not openai_client:
        raise ValueError("OpenAI 클라이언트가 제공되지 않았습니다.")
    if not rc_txt:
        print("Summarizer Service: 요약할 텍스트(rc_txt)가 비어있습니다.")
        return []

    topic_info = f"* 회의 주제: \"{subj}\"" if subj else "* 회의 주제: (제공되지 않음)"

    # 프롬프트는 LLM이 summary_points 라는 키로 JSON을 반환하도록 유도합니다.
    # 서비스 함수는 이 summary_points 리스트를 추출하여 반환합니다.
    # 라우터에서는 이 리스트를 받아 SummarizationResponse 모델의 'summary' 필드에 할당합니다.
    prompt = f"""
당신은 회의록을 분석하고 핵심 내용을 간결하게 요약하는 전문가입니다. 제공된 [회의록 전문]과 [회의 정보]를 바탕으로, 주요 논의 사항, 결정된 내용, 그리고 핵심 결론을 3개에서 8개 사이의 명확한 불렛 포인트로 요약해주십시오.

[회의 정보]
{topic_info}

[회의록 전문]
{rc_txt}

[요청 사항]
- 각 요약 포인트는 완전한 문장 형태를 유지하며, 회의의 핵심적인 정보를 정확히 전달해야 합니다.
- 회의 주제와 직접적으로 관련된 내용 위주로 요약해주십시오. 주제에서 벗어난 사담이나 개인적인 의견은 제외합니다.
- 결과는 반드시 다음 JSON 형식으로만 제공해주십시오:
{{
  "summary_points": [
    "요약 포인트 1: 핵심 내용 A에 대한 결정 사항입니다.",
    "요약 포인트 2: 주요 논의 B에 대한 결론입니다.",
    "요약 포인트 3: 다음 단계 C가 제안되었습니다."
  ]
}}
"""
    print(f"Summarizer Service: 요약 요청 시작 (모델: {model_name}, 주제: '{subj or '없음'}')")

    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=model_name,
            messages=[
                {"role": "system", "content": "당신은 회의 내용을 분석하여 JSON 형식으로 요약 결과를 제공하는 AI 어시스턴트입니다. 반드시 요청된 JSON 형식으로만 답변해야 합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        if response_content is None:
            print("Summarizer Service 오류: LLM으로부터 비어있는 응답(content=None)을 받았습니다.")
            raise ValueError("LLM으로부터 비어있는 응답을 받았습니다.")

        try:
            summary_data = json.loads(response_content)
            # LLM에게는 "summary_points" 키로 받도록 요청했지만,
            # 이 서비스 함수는 최종적으로 문자열 리스트만 반환합니다.
            summary_points_list = summary_data.get("summary_points")

            if summary_points_list is None:
                print(f"Summarizer Service 오류: LLM 응답에 'summary_points' 키가 없습니다. 응답: {response_content}")
                raise ValueError("LLM 응답 형식이 잘못되었습니다 ('summary_points' 키 부재).")
            if not isinstance(summary_points_list, list):
                print(f"Summarizer Service 오류: LLM 응답의 'summary_points'가 리스트가 아닙니다. 타입: {type(summary_points_list)}, 응답: {response_content}")
                raise ValueError("LLM 응답 형식이 잘못되었습니다 ('summary_points'가 리스트가 아님).")
            
            if not all(isinstance(item, str) for item in summary_points_list):
                print(f"Summarizer Service 오류: 'summary_points' 리스트 내에 문자열이 아닌 항목이 포함되어 있습니다. 응답: {summary_points_list}")
                raise ValueError("LLM 응답 형식이 잘못되었습니다 ('summary_points' 리스트 내 타입 오류).")

            print(f"Summarizer Service: 요약 생성 완료 ({len(summary_points_list)}개 포인트)")
            return summary_points_list # 최종적으로 문자열 리스트를 반환

        except json.JSONDecodeError:
            print(f"Summarizer Service 오류: LLM 응답을 JSON으로 파싱할 수 없습니다. 원본 응답: {response_content}")
            raise ValueError(f"LLM 응답을 JSON으로 파싱하는 데 실패했습니다. 응답 내용: {response_content[:200]}...")
        except KeyError:
            print(f"Summarizer Service 오류: LLM JSON 응답에 'summary_points' 키가 없습니다. 원본 응답: {response_content}")
            raise ValueError("LLM JSON 응답에 'summary_points' 키가 누락되었습니다.")

    except Exception as e:
        print(f"Summarizer Service 오류: OpenAI API 호출/처리 중 예외 발생 - {type(e).__name__}: {e}")
        raise RuntimeError(f"회의 내용 요약 중 오류가 발생했습니다.") from e