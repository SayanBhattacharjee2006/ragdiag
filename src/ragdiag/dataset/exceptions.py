"""Exceptions for dataset operations in RAGDiag."""


class RAGDiagError(Exception):
    """Base exception for all RAGDiag errors."""


class DatasetError(RAGDiagError):
    """Base exception for dataset-related errors."""


class DatasetLoadError(DatasetError):
    """Raised when a dataset file cannot be found, read, or parsed as valid JSON."""


class DatasetValidationError(DatasetError):
    """Raised when a dataset or its query samples fail schema or semantic validation."""
