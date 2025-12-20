"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "legal-rag"

    # Model settings
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1024  # Reduced dimensions for efficiency
    llm_model: str = "gpt-4o"
    reasoning_model: str = "o3-mini"  # Model for deep reasoning (o1, o3-mini, or gpt-4o)

    # Chunking settings
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 50  # tokens

    # Retrieval settings
    top_k: int = 5  # Number of chunks to retrieve


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()

