"""RAGDiag: RAG evaluation and root-cause diagnosis SDK/CLI."""

from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.result import EvaluationResult
from ragdiag.models.sample import QuerySample
from ragdiag.pipeline.base import Pipeline

__version__ = "0.1.0"

__all__ = [
    "EvaluationResult",
    "Pipeline",
    "QuerySample",
    "RetrievedChunk",
    "__version__",
]
