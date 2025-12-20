"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


# Legal domain categories for Moroccan law
LEGAL_CATEGORIES = {
    # Core categories
    "civil": "Droit civil et obligations (DOC, Code Civil)",
    "commercial": "Droit commercial et des affaires",
    "travail": "Droit du travail et sécurité sociale",
    "administratif": "Droit public et administratif",
    "numerique": "Droit numérique et protection des données",
    "fiscal": "Droit fiscal et douanier",
    "penal": "Droit pénal",
    "famille": "Droit de la famille (Moudawana)",
    "immobilier": "Droit immobilier et foncier",
    "procedure": "Procédure civile et pénale",
    "bancaire": "Droit bancaire et financier",
    "assurance": "Droit des assurances",
    "environnement": "Droit de l'environnement",
    "propriete_intellectuelle": "Propriété intellectuelle",
    # Alternative folder names (aliases)
    "affaires_et_commerce": "Droit commercial et des affaires",
    "droit_civil_et_famille": "Droit civil et famille",
    "droit_penal_et_procedure": "Droit pénal et procédure",
    "justice_et_procedure_civile": "Justice et procédure civile",
    "numerique_et_donnees": "Droit numérique et données",
    "travail_et_social": "Droit du travail et social",
    "fiscalite_et_finances_publiques": "Fiscalité et finances publiques",
    "administratif_et_collectivites": "Administratif et collectivités",
    "constitution_et_institutions": "Constitution et institutions",
    "reglementation_et_jurisprudence": "Réglementation et jurisprudence",
    # Default
    "general": "Textes généraux / Non classifié",
}


def get_category_from_path(path: str) -> str:
    """
    Extract category from file path.
    
    Expected structure: data/law_codes/<category>/file.pdf
    Returns 'general' if no category folder found.
    """
    import re
    
    # Normalize path separators
    normalized = path.replace("\\", "/").lower()
    
    # Look for category folder in path
    for category in LEGAL_CATEGORIES.keys():
        if f"/{category}/" in normalized or normalized.startswith(f"{category}/"):
            return category
    
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

