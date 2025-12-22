"""Mouham'IA - Your AI Legal Assistant (محامي + IA)"""

__version__ = "0.1.0"

# Main API - single entry point
from .rag import (
    # Functions
    analyze,
    analyze_stream,
    format_for_cli,
    # Models
    LawyerResponse,
    LawyerAnalysis,
    SourceCitation,
    ArgumentCite,
    ConclusionBreve,
    AnalyseApprofondie,
    MetaAnalyse,
    RaisonnementJuridique,
    QueryIntent,
)

__all__ = [
    # Main functions
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
]
