
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "AI Document Parser"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    LLM_PROVIDER: str = "groq"
    DEFAULT_MODEL: str = "llama3.2:1b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    MINIO_ENDPOINT: str = "localhost:9022"
    MINIO_ACCESS_KEY: str = "abc"
    MINIO_SECRET_KEY: str = "abc_password"
    MINIO_BUCKET_NAME: str = "doc-parser-bucket"
    MINIO_SECURE: bool = False
    
    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings():
    return Settings()
