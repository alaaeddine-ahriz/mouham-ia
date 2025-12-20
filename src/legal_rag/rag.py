"""Main RAG orchestration with citation-aware prompting."""

from dataclasses import dataclass
from typing import Generator

from openai import OpenAI

from .config import get_settings
from .reasoning import (
    LegalReasoning,
    ReasoningDepth,
    compare_provisions,
    reason_deep,
    reason_stream,
    reason_with_decomposition,
)
from .retriever import QueryIntent, format_context_for_llm, retrieve

# Re-export reasoning functions for convenient access
__all__ = [
    "RAGResponse",
    "ask",
    "chat_stream",
    "reason",
    "reason_stream",
    "reason_deep",
    "reason_with_decomposition",
    "compare_provisions",
    "LegalReasoning",
    "ReasoningDepth",
]


SYSTEM_PROMPT = """You are a legal research assistant. Your role is to answer questions based ONLY on the provided context documents.

CRITICAL INSTRUCTIONS:
1. Only answer based on the provided context. If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer this question."

2. For EVERY claim or fact you state, you MUST cite the source using bracketed numbers like [1], [2], etc.

3. When citing, use the EXACT verbatim text from the source. Do not paraphrase or summarize when providing citations.

4. Structure your response as follows:

ANSWER:
[Your synthesized answer with inline citations like [1], [2]]

SOURCES:
[List each citation with the exact quote from the source]

Example format for SOURCES section:
[1] contract_name.pdf | Page 5, Section 3.1
    > "The exact verbatim quote from the document that supports your claim."

[2] Code Civil | Article 1234
    > "The exact text from the law code."

5. Keep quotes concise but complete enough to support your claim.

6. If multiple sources support the same point, cite all of them."""


@dataclass
class RAGResponse:
    """Response from the RAG system."""

    answer: str
    sources: list[dict]
    query: str
    intent: QueryIntent


def ask(
    query: str,
    intent: QueryIntent | None = None,
    top_k: int | None = None,
    category: str | None = None,
) -> RAGResponse:
    """
    Ask a question and get an answer with citations.

    Args:
        query: The user's question
        intent: Override automatic intent detection
        top_k: Number of chunks to retrieve
        category: Optional legal category filter (e.g., "numerique", "civil")

    Returns:
        RAGResponse with answer and source citations
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # Retrieve relevant chunks
    chunks = retrieve(query, intent=intent, top_k=top_k, category=category)

    # Format context
    context = format_context_for_llm(chunks)

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Context Documents:

{context}

---

Question: {query}""",
        },
    ]

    # Call LLM
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.1,  # Low temperature for factual responses
    )

    answer = response.choices[0].message.content

    return RAGResponse(
        answer=answer,
        sources=chunks,
        query=query,
        intent=intent or QueryIntent.BOTH,
    )


def chat_stream(
    query: str,
    intent: QueryIntent | None = None,
    top_k: int | None = None,
    category: str | None = None,
):
    """
    Stream a response for interactive chat.

    Yields chunks of the response as they arrive.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # Retrieve relevant chunks
    chunks = retrieve(query, intent=intent, top_k=top_k, category=category)

    # Format context
    context = format_context_for_llm(chunks)

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Context Documents:

{context}

---

Question: {query}""",
        },
    ]

    # Stream response
    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.1,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def reason(
    query: str,
    depth: ReasoningDepth = ReasoningDepth.DEEP,
    intent: QueryIntent | None = None,
    top_k: int | None = None,
    decompose: bool = False,
) -> LegalReasoning:
    """
    Perform deep legal reasoning on a query.

    This is the main entry point for reasoning capabilities.

    Args:
        query: The legal question to analyze
        depth: Level of analysis (QUICK, STANDARD, DEEP)
        intent: Override automatic intent detection
        top_k: Number of chunks to retrieve
        decompose: If True, decompose query into sub-questions

    Returns:
        LegalReasoning with structured analysis and citations

    Example:
        >>> result = reason("Quelles sont les conditions de validité d'un contrat?")
        >>> print(result.final_answer)
        >>> print(f"Confidence: {result.confidence}")
    """
    if decompose:
        return reason_with_decomposition(query, intent=intent, top_k=top_k)
    else:
        return reason_deep(query, intent=intent, top_k=top_k)

