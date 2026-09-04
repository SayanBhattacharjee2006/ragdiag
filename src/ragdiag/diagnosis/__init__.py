"""Root-cause diagnosis package for RAG failure mode classification."""

from ragdiag.diagnosis.classifier import DiagnosisEngine
from ragdiag.diagnosis.models import (
    FAILURE_ACTIONS,
    DiagnosisResult,
    FailureCategory,
    get_action_for_category,
)
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
    "FAILURE_ACTIONS",
    "FailureCategory",
    "classify_answer_failure",
    "classify_context_sufficiency",
    "classify_grounding_failure",
    "classify_latency_outlier",
    "classify_pipeline_failure",
    "classify_ranking_failure",
    "classify_retrieval_failure",
    "get_action_for_category",
]
