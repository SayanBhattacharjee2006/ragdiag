"""Reporting and diagnostic intelligence system for RAGDiag."""

from ragdiag.reporting.aggregator import build_report
from ragdiag.reporting.confidence import compute_confidence
from ragdiag.reporting.health import compute_health_profile
from ragdiag.reporting.insights import generate_insights
from ragdiag.reporting.models import (
    ConfidenceLevel,
    EvaluationConfidence,
    EvaluationReport,
    HealthGrade,
    HealthProfile,
    HealthStatus,
    QueryTypeMetrics,
    RetrievalSummary,
    SemanticSummary,
    TopFailure,
)
from ragdiag.reporting.terminal import render_terminal_report

__all__ = [
    "ConfidenceLevel",
    "EvaluationConfidence",
    "EvaluationReport",
    "HealthGrade",
    "HealthProfile",
    "HealthStatus",
    "QueryTypeMetrics",
    "RetrievalSummary",
    "SemanticSummary",
    "TopFailure",
    "build_report",
    "compute_confidence",
    "compute_health_profile",
    "generate_insights",
    "render_terminal_report",
]
