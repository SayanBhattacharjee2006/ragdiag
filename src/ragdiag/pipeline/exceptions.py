"""Exceptions for pipeline operations and loading."""

from ragdiag.dataset.exceptions import RAGDiagError


class PipelineError(RAGDiagError):
    """Base exception for pipeline-related errors."""


class PipelineLoadError(PipelineError):
    """Raised when a pipeline file cannot be found, loaded, or does not adhere to conventions."""
