from typing import Optional
from pydantic_settings import BaseSettings

class ResearchSettings(BaseSettings):
    """
    Configuration settings for the Research library.
    Values can be overridden by environment variables or an optional .env file.
    """
    OPENROUTER_API_KEY: Optional[str] = ""
    PROJECT_API_KEY: Optional[str] = "dummy_secret_key"
    TOOL_DB_BASE_URL: str = "http://127.0.0.1:8000/api/v1"
    TOOL_DISCOVERY_MODEL: str = "deepseek/deepseek-v4-flash"
    TOOL_DISCOVERY_TEMPERATURE: float = 0.8
    MAIN_WORKFLOW_MODEL: str = "deepseek/deepseek-v4-flash"
    MAIN_WORKFLOW_TEMPERATURE: float = 0.0
    EMBEDDING_MODEL: str = "voyageai/voyage-4-lite"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = ResearchSettings()
