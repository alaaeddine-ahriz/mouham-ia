"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_category_from_path(path: str) -> str:
    """
    Extract category from file path based on folder structure.
    
    Expected structure: data/law_codes/<category>/file.pdf
    Returns the folder name as category, or 'general' if not found.
    """
    from pathlib import Path
    
    # Normalize and get parts
    path_obj = Path(path.replace("\\", "/"))
    parts = path_obj.parts
    
    # Look for law_codes or contracts folder, category is the next folder
    for i, part in enumerate(parts):
        if part.lower() in ("law_codes", "contracts", "lois", "contrats"):
            if i + 1 < len(parts) and not parts[i + 1].endswith(".pdf"):
                return parts[i + 1].lower()
    
    return "general"


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
    model: str = "gpt-4.1-mini"  # Single model for all tasks (reasoning + structured output)

    # Chunking settings
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 50  # tokens

    # Retrieval settings
    top_k: int = 5  # Number of chunks to retrieve

    # Language settings (hardcoded for now)
    language: str = "fr"  # Response language: "fr" for French
    language_name: str = "français"  # Full language name


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
