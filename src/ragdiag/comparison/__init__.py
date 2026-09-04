"""Multi-pipeline comparison subsystem for RAGDiag."""

from ragdiag.comparison.comparator import Comparator, compare_reports
from ragdiag.comparison.models import (
    ComparisonReport,
    MetricDeltas,
    QueryOutcomeComparison,
    QueryTypeDeltas,
)
from ragdiag.comparison.terminal import render_comparison_terminal

__all__ = [
    "Comparator",
    "ComparisonReport",
    "MetricDeltas",
    "QueryOutcomeComparison",
    "QueryTypeDeltas",
    "compare_reports",
    "render_comparison_terminal",
]
