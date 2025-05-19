# app/services/email_service.py
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from pydantic import EmailStr
from typing import List, Dict, Any
import os
from pathlib import Path

from app.core.config import settings
from app.models.meeting import FullAnalysisResult # 이메일 본문에 필요할 수 있음

conf = ConnectionConfig(
    MAIL_USERNAME=settings.MAIL_USERNAME,
    MAIL_PASSWORD=settings.MAIL_PASSWORD,
    MAIL_FROM=settings.MAIL_FROM,
    MAIL_PORT=settings.MAIL_PORT,
    MAIL_SERVER=settings.MAIL_SERVER,
    MAIL_FROM_NAME=settings.MAIL_FROM_NAME,
    MAIL_STARTTLS=settings.MAIL_STARTTLS,
    MAIL_SSL_TLS=settings.MAIL_SSL_TLS,
    USE_CREDENTIALS=settings.USE_CREDENTIALS,
    VALIDATE_CERTS=settings.VALIDATE_CERTS,
    TEMPLATE_FOLDER=Path(__file__).parent.parent / 'templates' / 'email' # 템플릿 폴더 경로 설정
)

fm = FastMail(conf)

async def send_analysis_report_email(
    recipients: List[EmailStr],
    subject: str,
    analysis_data: FullAnalysisResult
):
    """
    회의 분석 결과를 이메일로 발송합니다.
    HTML 템플릿을 사용하여 이메일 본문을 구성합니다.
    """
    if not all([settings.MAIL_USERNAME, settings.MAIL_PASSWORD, settings.MAIL_FROM, settings.MAIL_SERVER]):
        print("이메일 서비스 경고: 필수 이메일 설정(사용자명, 비밀번호, 발신자, 서버)이 누락되었습니다. 이메일을 발송할 수 없습니다.")
        return

    # FullAnalysisResult 객체를 템플릿에 전달할 수 있는 딕셔너리로 변환
    # Pydantic 모델의 .model_dump() (v2) 또는 .dict() (v1) 사용
    template_body_data = analysis_data.model_dump(exclude_none=True)

    # 회의 참석자 정보에서 이메일 주소만 추출하여 템플릿에 전달 (선택적)
    # template_body_data['meeting_info']['attendee_emails'] = [
    #     att.email for att in analysis_data.meeting_info.info_n if att.email
    # ]

    message = MessageSchema(
        subject=subject,
        recipients=recipients,
        template_body=template_body_data, # 템플릿에 전달할 데이터
        subtype=MessageType.html
    )

    try:
        print(f"이메일 발송 시도: 받는사람={recipients}, 제목='{subject}'")
        await fm.send_message(message, template_name="analysis_report.html")
        print(f"이메일 성공적으로 발송됨: 받는사람={recipients}")
    except Exception as e:
        print(f"이메일 발송 실패: {e}")
        # 실제 운영 환경에서는 여기에 더 강력한 로깅 및 오류 처리 추가
        # 예를 들어, 특정 수신자에게 실패했는지 등의 정보 기록