"""Reporting and diagnostic intelligence system for RAGDiag."""

from ragdiag.reporting.aggregator import build_report
from ragdiag.reporting.insights import generate_insights
from ragdiag.reporting.models import (
    EvaluationReport,
    QueryTypeMetrics,
    RetrievalSummary,
    SemanticSummary,
    TopFailure,
)
from ragdiag.reporting.terminal import render_terminal_report

__all__ = [
    "EvaluationReport",
    "QueryTypeMetrics",
    "RetrievalSummary",
    "SemanticSummary",
    "TopFailure",
    "build_report",
    "generate_insights",
    "render_terminal_report",
]
