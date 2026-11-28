from pydantic_settings import BaseSettings, SettingsConfigDict


class ResearchSettings(BaseSettings):
    """Optional defaults for the tool database URL and embedding model."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    TOOL_DB_BASE_URL: str = "http://127.0.0.1:8000/api/v1"
    EMBEDDING_MODEL: str = "voyageai/voyage-4-lite"


settings = ResearchSettings()
