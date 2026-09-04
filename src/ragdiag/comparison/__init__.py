"""Multi-pipeline comparison subsystem for RAGDiag."""

from ragdiag.comparison.comparator import Comparator, compare_reports
from ragdiag.comparison.models import (
    ComparisonReport,
    DiagnosisTransition,
    MetricDeltas,
    MetricRegression,
    QueryOutcomeComparison,
    QueryTypeDeltas,
    RegressedQuery,
    RegressionAnalysis,
)
from ragdiag.comparison.regression import analyze_regressions
from ragdiag.comparison.terminal import render_comparison_terminal

__all__ = [
    "Comparator",
    "ComparisonReport",
    "DiagnosisTransition",
    "MetricDeltas",
    "MetricRegression",
    "QueryOutcomeComparison",
    "QueryTypeDeltas",
    "RegressedQuery",
    "RegressionAnalysis",
    "analyze_regressions",
    "compare_reports",
    "render_comparison_terminal",
]
