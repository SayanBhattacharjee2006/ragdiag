"""RAGDiag: RAG evaluation and root-cause diagnosis SDK/CLI."""

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
    "Pipeline",
    "QuerySample",
    "QueryType",
    "RetrievedChunk",
    "__version__",
]
