"""RAGDiag: RAG evaluation and root-cause diagnosis SDK/CLI."""

from ragdiag.comparison import (
    Comparator,
    ComparisonReport,
    DiagnosisTransition,
    MetricRegression,
    RegressedQuery,
    RegressionAnalysis,
    analyze_regressions,
)
from ragdiag.diagnosis import (
    FAILURE_ACTIONS,
    DiagnosisEngine,
    DiagnosisResult,
    FailureCategory,
    get_action_for_category,
)
from ragdiag.judges import Judge, JudgeResult, OpenAIJudge
from ragdiag.metrics import (
    aggregate_metrics,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.dataset import GoldenDataset
from ragdiag.models.result import EvaluationResult
from ragdiag.models.sample import QuerySample, QueryType
from ragdiag.pipeline.base import Pipeline
from ragdiag.reporting import (
    EvaluationReport,
    HealthGrade,
    HealthProfile,
    HealthStatus,
    TopFailure,
    build_report,
)
from ragdiag.runner.evaluator import Evaluator

__version__ = "0.1.0"

__all__ = [
    "Comparator",
    "ComparisonReport",
    "DiagnosisEngine",
    "DiagnosisResult",
    "DiagnosisTransition",
    "EvaluationReport",
    "EvaluationResult",
    "Evaluator",
    "FAILURE_ACTIONS",
    "FailureCategory",
    "GoldenDataset",
    "HealthGrade",
    "HealthProfile",
    "HealthStatus",
    "Judge",
    "JudgeResult",
    "MetricRegression",
    "OpenAIJudge",
    "Pipeline",
    "QuerySample",
    "QueryType",
    "RegressedQuery",
    "RegressionAnalysis",
    "RetrievedChunk",
    "TopFailure",
    "__version__",
    "aggregate_metrics",
    "analyze_regressions",
    "build_report",
    "get_action_for_category",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
