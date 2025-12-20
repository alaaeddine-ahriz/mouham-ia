"""Mouham'IA - Your AI Legal Assistant (محامي + IA)"""

__version__ = "0.1.0"

# Main API exports
from .rag import (
    RAGResponse,
    ask,
    chat_stream,
    reason,
    LegalReasoning,
    ReasoningDepth,
)
from .reasoning import (
    compare_provisions,
    multi_step_retrieve,
    reason_deep,
    reason_multistep,
    reason_stream,
    reason_with_decomposition,
)
from .retriever import QueryIntent

__all__ = [
    # Core functions
    "ask",
    "chat_stream",
    "reason",
    # Reasoning
    "reason_deep",
    "reason_multistep",
    "reason_stream",
    "reason_with_decomposition",
    "compare_provisions",
    "multi_step_retrieve",
    # Types
    "RAGResponse",
    "LegalReasoning",
    "ReasoningDepth",
    "QueryIntent",
]

