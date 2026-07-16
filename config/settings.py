from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration.

    Values are automatically loaded from the .env file or system environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    GROQ_API_KEY: str = Field(
        ...,
        description="Groq API Key",
    )

    GROQ_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Default Groq model",
    )

    TAVILY_API_KEY: str = Field(
        ...,
        description="API key for Tavily web search",
    )

    NEON_VECTOR_DATABASE_URL: str = Field(
        ...,
        description="Neon PostgreSQL connection string",
    )

    HF_TOKEN: Optional[str] = Field(
        default=None,
        description="Optional Hugging Face Inference API Token to prevent rate limiting",
    )


# Singleton instance
settings = Settings()