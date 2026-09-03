"""Root-cause diagnosis package for RAG failure mode classification."""

from ragdiag.diagnosis.classifier import DiagnosisEngine
from ragdiag.diagnosis.models import DiagnosisResult, FailureCategory
from ragdiag.diagnosis.rules import (
    classify_answer_failure,
    classify_context_sufficiency,
    classify_grounding_failure,
    classify_latency_outlier,
    classify_pipeline_failure,
    classify_ranking_failure,
    classify_retrieval_failure,
)

__all__ = [
    "DiagnosisEngine",
    "DiagnosisResult",
    "FailureCategory",
    "classify_answer_failure",
    "classify_context_sufficiency",
    "classify_grounding_failure",
    "classify_latency_outlier",
    "classify_pipeline_failure",
    "classify_ranking_failure",
    "classify_retrieval_failure",
]
