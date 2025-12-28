"""Main RAG orchestration - single entry point for legal analysis."""

from .reasoning import (
    # Main function
    analyze,
    analyze_stream,
    format_for_cli,
    # Models for type hints
    LawyerResponse,
    LawyerAnalysis,
    SourceCitation,
    ArgumentCite,
    ConclusionBreve,
    AnalyseApprofondie,
    MetaAnalyse,
    RaisonnementJuridique,
)
from .retriever import QueryIntent
from .router import QuestionType, route_question

# Public API
__all__ = [
    # Main function
    "analyze",
    "analyze_stream",
    "format_for_cli",
    # Response models
    "LawyerResponse",
    "LawyerAnalysis",
    "SourceCitation",
    "ArgumentCite",
    "ConclusionBreve",
    "AnalyseApprofondie",
    "MetaAnalyse",
    "RaisonnementJuridique",
    # Query intent
    "QueryIntent",
    # Router
    "QuestionType",
    "route_question",
]
