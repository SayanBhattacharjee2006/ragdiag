"""RAGDiag: RAG evaluation and root-cause diagnosis SDK/CLI."""

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
from ragdiag.runner.evaluator import Evaluator

__version__ = "0.1.0"

__all__ = [
    "EvaluationResult",
    "Evaluator",
    "GoldenDataset",
    "Judge",
    "JudgeResult",
    "OpenAIJudge",
    "Pipeline",
    "QuerySample",
    "QueryType",
    "RetrievedChunk",
    "__version__",
    "aggregate_metrics",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
