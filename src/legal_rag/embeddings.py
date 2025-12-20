"""OpenAI embeddings with batching support."""

from openai import OpenAI

from .config import get_settings


def get_embeddings(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Get embeddings for a list of texts using OpenAI API.

    Args:
        texts: List of texts to embed
        batch_size: Number of texts to embed per API call

    Returns:
        List of embedding vectors
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    all_embeddings: list[list[float]] = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
            dimensions=settings.embedding_dimensions,
        )

        # Extract embeddings in order
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def get_embedding(text: str) -> list[float]:
    """
    Get embedding for a single text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector
    """
    return get_embeddings([text])[0]

