"""Validation functions and summary reporting for GoldenDataset."""

from collections import Counter
from dataclasses import dataclass

from ragdiag.dataset.exceptions import DatasetValidationError
from ragdiag.models.dataset import GoldenDataset
from ragdiag.models.sample import QueryType


@dataclass(frozen=True)
class DatasetValidationSummary:
    """Summary statistics for a validated GoldenDataset."""

    name: str
    version: str
    total_samples: int
    query_type_counts: dict[str, int]


def validate_dataset(dataset: GoldenDataset) -> DatasetValidationSummary:
    """Validate dataset-level constraints and generate summary statistics.

    Args:
        dataset: The GoldenDataset instance to validate.

    Returns:
        DatasetValidationSummary containing sample counts and query type breakdowns.

    Raises:
        DatasetValidationError: If dataset-level invariants are violated.
    """
    if not dataset.name.strip():
        raise DatasetValidationError("Dataset 'name' must not be empty or whitespace-only.")

    if not dataset.version.strip():
        raise DatasetValidationError("Dataset 'version' must not be empty or whitespace-only.")

    if not dataset.samples:
        raise DatasetValidationError("Dataset must contain at least one sample.")

    seen_ids: set[str] = set()
    for idx, sample in enumerate(dataset.samples):
        if sample.id in seen_ids:
            raise DatasetValidationError(
                f"Duplicate sample ID '{sample.id}' found at sample index {idx}."
            )
        seen_ids.add(sample.id)

        if not sample.relevant_chunk_ids:
            raise DatasetValidationError(
                f"Sample '{sample.id}' must contain at least one relevant chunk ID."
            )

        seen_chunks: set[str] = set()
        for chunk_idx, chunk_id in enumerate(sample.relevant_chunk_ids):
            if not chunk_id or not chunk_id.strip():
                raise DatasetValidationError(
                    f"Sample '{sample.id}' contains empty chunk ID at index {chunk_idx}."
                )
            if chunk_id in seen_chunks:
                raise DatasetValidationError(
                    f"Sample '{sample.id}' contains duplicate chunk ID '{chunk_id}'."
                )
            seen_chunks.add(chunk_id)

    # Compute query type distribution
    type_counts = Counter(sample.query_type.value for sample in dataset.samples)
    sorted_counts = {
        qt.value: type_counts.get(qt.value, 0)
        for qt in QueryType
        if type_counts.get(qt.value, 0) > 0
    }

    return DatasetValidationSummary(
        name=dataset.name,
        version=dataset.version,
        total_samples=len(dataset.samples),
        query_type_counts=sorted_counts,
    )
