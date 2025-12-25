"""Pinecone vector store operations."""

import hashlib
import re
import unicodedata

from pinecone import Pinecone, ServerlessSpec

from .config import get_settings
from .ingest.chunker import ArticleChunk, Chunk


def _make_ascii_id(text: str) -> str:
    """Convert text to ASCII-safe ID for Pinecone."""
    # Normalize unicode and convert to ASCII
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    # Replace non-alphanumeric with underscores
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", ascii_text)
    # If too short after conversion, use hash
    if len(safe_id) < 3:
        safe_id = hashlib.md5(text.encode()).hexdigest()[:16]
    return safe_id


# Namespace constants
NAMESPACE_LAW_CODES = "law_codes"
NAMESPACE_USER_CONTRACTS = "user_contracts"


def get_pinecone_client() -> Pinecone:
    """Get Pinecone client instance."""
    settings = get_settings()
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index_exists() -> None:
    """Create Pinecone index if it doesn't exist."""
    settings = get_settings()
    pc = get_pinecone_client()

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if settings.pinecone_index_name not in existing_indexes:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index: {settings.pinecone_index_name}")
    else:
        print(f"Using existing Pinecone index: {settings.pinecone_index_name}")


def upsert_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    namespace: str,
    batch_size: int = 100,
) -> int:
    """
    Upsert chunks with embeddings to Pinecone.

    Args:
        chunks: List of text chunks with metadata
        embeddings: Corresponding embedding vectors
        namespace: Pinecone namespace (law_codes or user_contracts)
        batch_size: Number of vectors per upsert call

    Returns:
        Number of vectors upserted
    """
    settings = get_settings()
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)

    # Prepare vectors
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        safe_source = _make_ascii_id(chunk.source)
        vector_id = f"{namespace}_{safe_source}_{chunk.chunk_index}"
        vectors.append(
            {
                "id": vector_id,
                "values": embedding,
                "metadata": chunk.to_metadata(),
            }
        )

    # Upsert in batches
    total_upserted = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        total_upserted += len(batch)

    return total_upserted


def upsert_article_chunks(
    chunks: list[ArticleChunk],
    embeddings: list[list[float]],
    namespace: str,
    batch_size: int = 100,
) -> int:
    """
    Upsert article chunks with embeddings to Pinecone.

    Uses stable article-based IDs for deduplication.

    Args:
        chunks: List of ArticleChunk objects
        embeddings: Corresponding embedding vectors
        namespace: Pinecone namespace
        batch_size: Number of vectors per upsert call

    Returns:
        Number of vectors upserted
    """
    settings = get_settings()
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)

    # Prepare vectors with stable IDs
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vector_id = chunk.get_vector_id(namespace)
        vectors.append(
            {
                "id": vector_id,
                "values": embedding,
                "metadata": chunk.to_metadata(),
            }
        )

    # Upsert in batches
    total_upserted = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        total_upserted += len(batch)

    return total_upserted

def query_similar(
    query_embedding: list[float],
    namespace: str,
    top_k: int = 5,
    category: str | None = None,
) -> list[dict]:
    """
    Query for similar chunks in a namespace.

    Args:
        query_embedding: Query vector
        namespace: Namespace to search
        top_k: Number of results to return
        category: Optional category filter (e.g., "numerique", "civil")

    Returns:
        List of matches with metadata
    """
    settings = get_settings()
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)

    # Build filter if category specified
    query_filter = None
    if category:
        query_filter = {"category": {"$eq": category}}

    results = index.query(
        vector=query_embedding,
        namespace=namespace,
        top_k=top_k,
        include_metadata=True,
        filter=query_filter,
    )

    return [
        {
            "id": match.id,
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "source": match.metadata.get("source", ""),
            "page_number": match.metadata.get("page_number", 0),
            "section_header": match.metadata.get("section_header", ""),
            "category": match.metadata.get("category", "general"),
        }
        for match in results.matches
    ]


def query_multiple_namespaces(
    query_embedding: list[float],
    namespaces: list[str],
    top_k: int = 5,
    category: str | None = None,
) -> list[dict]:
    """
    Query multiple namespaces and merge results.

    Args:
        query_embedding: Query vector
        namespaces: List of namespaces to search
        top_k: Number of results per namespace
        category: Optional category filter

    Returns:
        Merged and sorted list of matches
    """
    all_results = []

    for namespace in namespaces:
        results = query_similar(query_embedding, namespace, top_k, category=category)
        for result in results:
            result["namespace"] = namespace
        all_results.extend(results)

    # Sort by score (descending) and take top_k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


def delete_by_source(source: str, namespace: str) -> None:
    """
    Delete all vectors from a specific source document.

    Args:
        source: Source filename to delete
        namespace: Namespace to delete from
    """
    settings = get_settings()
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)

    # Query for all vectors with this source
    # Note: This is a workaround since Pinecone doesn't support delete by metadata directly
    # In production, you might want to track vector IDs separately
    index.delete(
        filter={"source": {"$eq": source}},
        namespace=namespace,
    )


def clear_namespace(namespace: str) -> int:
    """
    Delete all vectors in a namespace.
    
    Args:
        namespace: Namespace to clear
        
    Returns:
        Number of vectors deleted (approximate)
    """
    settings = get_settings()
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)
    
    # Get count before deletion
    stats = index.describe_index_stats()
    count = 0
    if hasattr(stats, 'namespaces') and stats.namespaces:
        ns_stats = stats.namespaces.get(namespace)
        if ns_stats:
            count = getattr(ns_stats, 'vector_count', 0)
    
    # Only try to delete if namespace has vectors
    if count > 0:
        try:
            index.delete(delete_all=True, namespace=namespace)
        except Exception:
            pass  # Namespace might not exist
    
    return count


def clear_all_namespaces() -> dict:
    """
    Delete all vectors in all namespaces.
    
    Returns:
        Dict with counts per namespace deleted
    """
    settings = get_settings()
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)
    
    # Get stats before deletion
    stats = index.describe_index_stats()
    deleted = {}
    
    # Clear each namespace
    for namespace in [NAMESPACE_LAW_CODES, NAMESPACE_USER_CONTRACTS]:
        count = 0
        if hasattr(stats, 'namespaces') and stats.namespaces:
            ns_stats = stats.namespaces.get(namespace)
            if ns_stats:
                count = getattr(ns_stats, 'vector_count', 0)
        
        # Only try to delete if namespace has vectors
        if count > 0:
            try:
                index.delete(delete_all=True, namespace=namespace)
            except Exception:
                pass  # Namespace might not exist
        
        deleted[namespace] = count
    
    return deleted


def delete_index() -> bool:
    """
    Delete the entire Pinecone index.
    
    Returns:
        True if deleted, False if didn't exist
    """
    settings = get_settings()
    pc = get_pinecone_client()
    
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if settings.pinecone_index_name in existing_indexes:
        pc.delete_index(settings.pinecone_index_name)
        return True
    return False


def get_index_stats() -> dict:
    """Get statistics about the Pinecone index."""
    settings = get_settings()
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)
    return index.describe_index_stats()


def get_available_categories(namespace: str = NAMESPACE_LAW_CODES) -> dict[str, int]:
    """
    Get available categories from the index metadata.
    
    Retrieves unique category values and their document counts
    by sampling vectors from the index.
    
    Args:
        namespace: Namespace to query (default: law_codes)
    
    Returns:
        Dict mapping category names to approximate document counts
    """
    settings = get_settings()
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)
    
    # Get a sample of vectors to extract categories
    # We use a zero vector to get random results
    zero_vector = [0.0] * settings.embedding_dimensions
    
    # Query a large sample to discover categories
    results = index.query(
        vector=zero_vector,
        namespace=namespace,
        top_k=1000,  # Get a good sample
        include_metadata=True,
    )
    
    # Count categories
    categories: dict[str, int] = {}
    for match in results.matches:
        cat = match.metadata.get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1
    
    return categories


def get_all_categories() -> dict[str, int]:
    """
    Get all available categories across all namespaces.
    
    Returns:
        Dict mapping category names to document counts
    """
    all_categories: dict[str, int] = {}
    
    for namespace in [NAMESPACE_LAW_CODES, NAMESPACE_USER_CONTRACTS]:
        try:
            ns_categories = get_available_categories(namespace)
            for cat, count in ns_categories.items():
                all_categories[cat] = all_categories.get(cat, 0) + count
        except Exception:
            pass  # Namespace might not exist
    
    return all_categories

