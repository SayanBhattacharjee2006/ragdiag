"""Dataset management, loading, and validation module for RAGDiag."""

from ragdiag.dataset.exceptions import (
    DatasetError,
    DatasetLoadError,
    DatasetValidationError,
)
from ragdiag.dataset.loader import load_dataset
from ragdiag.dataset.validator import DatasetValidationSummary, validate_dataset
from ragdiag.models.dataset import GoldenDataset
from ragdiag.models.sample import QuerySample, QueryType

__all__ = [
    "DatasetError",
    "DatasetLoadError",
    "DatasetValidationError",
    "DatasetValidationSummary",
    "GoldenDataset",
    "QuerySample",
    "QueryType",
    "load_dataset",
    "validate_dataset",
]
