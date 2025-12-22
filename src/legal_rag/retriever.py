"""Query routing and retrieval with exact text."""

from enum import Enum

from openai import OpenAI

from .config import get_settings
from .embeddings import get_embedding
from .pinecone_store import (
    NAMESPACE_LAW_CODES,
    NAMESPACE_USER_CONTRACTS,
    query_multiple_namespaces,
    query_similar,
)


class QueryIntent(Enum):
    """Detected intent for namespace routing."""

    LAW_CODES = "law_codes"
    CONTRACTS = "contracts"
    BOTH = "both"


def detect_query_intent(query: str) -> QueryIntent:
    """
    Detect which namespace(s) to search based on query.

    Uses GPT to classify the query intent.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use mini for fast/cheap classification
        messages=[
            {
                "role": "system",
                "content": """Vous êtes un classificateur de requêtes pour un système juridique RAG.

Classifiez les requêtes en UNE des trois catégories:

- LAW: Questions sur les lois, codes, textes législatifs, réglementations, articles de loi, 
  droits, obligations légales, fiscalité, droit du travail, procédures judiciaires, etc.
  MÊME si l'utilisateur pose la question dans un contexte personnel ("mon entreprise", "ma situation").
  
- CONTRACT: Questions sur un contrat SPÉCIFIQUE déjà signé ou en cours de rédaction,
  clauses contractuelles particulières, interprétation d'un contrat existant.
  
- BOTH: Questions qui nécessitent à la fois les textes de loi ET un contrat spécifique.

IMPORTANT: La plupart des questions juridiques personnelles (salaire, impôts, droits, 
obligations employeur/employé) nécessitent de chercher dans les CODES DE LOI (LAW), 
pas dans les contrats.

Répondez avec exactement un mot: LAW, CONTRACT, ou BOTH""",
            },
            {"role": "user", "content": query},
        ],
        max_tokens=10,
        temperature=0,
    )

    classification = response.choices[0].message.content.strip().upper()

    if classification == "CONTRACT":
        return QueryIntent.CONTRACTS
    elif classification == "BOTH":
        return QueryIntent.BOTH
    else:
        # Default to LAW_CODES for most legal questions
        return QueryIntent.LAW_CODES


def retrieve(
    query: str,
    intent: QueryIntent | None = None,
    top_k: int | None = None,
    category: str | None = None,
) -> list[dict]:
    """
    Retrieve relevant chunks for a query.

    Args:
        query: The user's question
        intent: Override automatic intent detection
        top_k: Number of results to return
        category: Optional legal category filter (e.g., "numerique", "civil")

    Returns:
        List of relevant chunks with metadata
    """
    settings = get_settings()
    top_k = top_k or settings.top_k

    # Detect intent if not provided
    if intent is None:
        intent = detect_query_intent(query)

    # Get query embedding
    query_embedding = get_embedding(query)

    # Route to appropriate namespace(s)
    if intent == QueryIntent.LAW_CODES:
        results = query_similar(query_embedding, NAMESPACE_LAW_CODES, top_k, category=category)
        for r in results:
            r["namespace"] = NAMESPACE_LAW_CODES
    elif intent == QueryIntent.CONTRACTS:
        results = query_similar(query_embedding, NAMESPACE_USER_CONTRACTS, top_k, category=category)
        for r in results:
            r["namespace"] = NAMESPACE_USER_CONTRACTS
    else:  # BOTH
        results = query_multiple_namespaces(
            query_embedding,
            [NAMESPACE_LAW_CODES, NAMESPACE_USER_CONTRACTS],
            top_k,
            category=category,
        )

    return results


def format_context_for_llm(chunks: list[dict]) -> str:
    """
    Format retrieved chunks as context for the LLM.

    Each chunk is numbered for citation reference.
    """
    if not chunks:
        return "No relevant context found."

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk["source"]
        page = chunk.get("page_number", "")
        section = chunk.get("section_header", "")
        category = chunk.get("category", "")
        text = chunk["text"]

        # Build source label
        source_label = source
        if category and category != "general":
            source_label = f"[{category.upper()}] {source_label}"
        if page:
            source_label += f", Page {page}"
        if section:
            source_label += f", {section}"

        context_parts.append(f"[{i}] Source: {source_label}\n{text}")

    return "\n\n---\n\n".join(context_parts)

