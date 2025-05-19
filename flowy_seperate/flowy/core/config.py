from pydantic_settings import BaseSettings, SettingsConfigDict
import os

# 프로젝트 루트 경로를 기준으로 .env 파일 경로 설정 (선택적이지만, 경로 문제 방지에 도움)
# 예를 들어, 이 config.py 파일이 my_meeting_app/app/core/ 에 있다면,
# .env 파일은 my_meeting_app/ 에 있어야 합니다.
# env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
# print(f"Attempting to load .env file from: {env_path}") # 경로 확인용

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini" # .env 파일에 없으면 이 기본값 사용

    # .env 파일에서 환경변수를 로드하도록 설정
    # env_file_encoding='utf-8' 추가는 한글 주석 등이 있을 경우를 대비
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra="ignore")

settings = Settings()

# # (선택적) 로드된 값 확인
# print(f"Loaded OPENAI_API_KEY: {settings.OPENAI_API_KEY[:5]}...") # API 키 일부만 출력
# print(f"Loaded DEFAULT_LLM_MODEL: {settings.DEFAULT_LLM_MODEL}")