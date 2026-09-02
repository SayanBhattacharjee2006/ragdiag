"""Metrics package for retrieval quality and latency evaluation."""

from ragdiag.metrics.aggregation import aggregate_metrics, mean_reciprocal_rank
from ragdiag.metrics.latency import calculate_latency_summary, calculate_percentile
from ragdiag.metrics.models import (
    AggregateEvaluationReport,
    LatencySummary,
    RetrievalMetricResult,
)
from ragdiag.metrics.retrieval import (
    compute_retrieval_metrics,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "AggregateEvaluationReport",
    "LatencySummary",
    "RetrievalMetricResult",
    "aggregate_metrics",
    "calculate_latency_summary",
    "calculate_percentile",
    "compute_retrieval_metrics",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
