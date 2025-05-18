# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import EmailStr # EmailStr 타입을 명시적으로 사용하기 위해 import
import os
from typing import Optional

# 프로젝트 루트 경로를 기준으로 .env 파일 경로 설정 (선택적이지만, 경로 문제 방지에 도움)
# 예를 들어, 이 config.py 파일이 my_meeting_app/app/core/ 에 있다면,
# .env 파일은 my_meeting_app/ 에 있어야 합니다.
# 현재 파일의 디렉토리 (app/core) -> 부모 (app) -> 부모 (프로젝트 루트)
# env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
# print(f"Attempting to load .env file from: {env_path}") # 경로 확인용

class Settings(BaseSettings):
    # OpenAI API 설정
    OPENAI_API_KEY: str
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini" # .env 파일에 없으면 이 기본값 사용

    # 이메일 서버 설정 (FastAPI-Mail 기준)
    # 이 값들은 .env 파일에 정의하는 것을 강력히 권장합니다.
    MAIL_USERNAME: Optional[str] = None
    MAIL_PASSWORD: Optional[str] = None
    MAIL_FROM: Optional[EmailStr] = None # 발신자 이메일 주소
    MAIL_PORT: int = 587 # 기본 포트 (TLS)
    MAIL_SERVER: Optional[str] = None # 예: "smtp.gmail.com"
    MAIL_FROM_NAME: str = "회의록 분석 결과 알리미" # 발신자 이름 기본값

    MAIL_STARTTLS: bool = True # TLS 사용 여부
    MAIL_SSL_TLS: bool = False # SSL/TLS 직접 연결 여부 (보통 STARTTLS와 반대)
    USE_CREDENTIALS: bool = True # 사용자 인증 정보 사용 여부
    VALIDATE_CERTS: bool = True # 인증서 유효성 검사 여부

    # .env 파일에서 환경변수를 로드하도록 설정
    # env_file_encoding='utf-8' 추가는 한글 주석 등이 있을 경우를 대비
    # extra="ignore"는 .env 파일에 정의되지 않은 필드가 Settings 클래스에 있어도 오류를 발생시키지 않음
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra="ignore")

settings = Settings()

# # (선택적) 로드된 값 확인 (민감 정보는 주의해서 출력)
# print(f"Loaded OPENAI_API_KEY: {settings.OPENAI_API_KEY[:5]}...") # API 키 일부만 출력
# print(f"Loaded DEFAULT_LLM_MODEL: {settings.DEFAULT_LLM_MODEL}")
# print(f"Loaded MAIL_USERNAME: {settings.MAIL_USERNAME}")
# print(f"Loaded MAIL_FROM: {settings.MAIL_FROM}")
# print(f"Loaded MAIL_SERVER: {settings.MAIL_SERVER}")