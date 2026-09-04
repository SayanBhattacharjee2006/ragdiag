"""Dataset loading utilities for RAGDiag."""

import json
from pathlib import Path

from pydantic import ValidationError

from ragdiag.dataset.exceptions import DatasetLoadError, DatasetValidationError
from ragdiag.dataset.validator import validate_dataset
from ragdiag.models.dataset import GoldenDataset


def load_dataset(path: str | Path) -> GoldenDataset:
    """Load and validate a GoldenDataset from a JSON file path.

    Args:
        path: Path to the golden dataset JSON file.

    Returns:
        A validated `GoldenDataset` instance.

    Raises:
        DatasetLoadError: If the file does not exist, is not a regular file,
            does not have a .json extension, or contains invalid JSON syntax.
        DatasetValidationError: If the dataset content violates schema or semantic rules.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise DatasetLoadError(f"Dataset file does not exist: {file_path}")

    if not file_path.is_file():
        raise DatasetLoadError(f"Dataset path is not a regular file: {file_path}")

    if file_path.suffix.lower() != ".json":
        raise DatasetLoadError(
            f"Dataset file must have a .json extension, got '{file_path.suffix}'."
        )

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetLoadError(f"Failed to read dataset file '{file_path}': {exc}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DatasetLoadError(
            f"Invalid JSON syntax in '{file_path}' (line {exc.lineno}, col {exc.colno}): {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise DatasetValidationError(
            f"Expected dataset root to be a JSON object, got {type(data).__name__}."
        )

    try:
        dataset = GoldenDataset.model_validate(data)
    except ValidationError as exc:
        errors = []
        for error in exc.errors():
            loc = " -> ".join(str(elem) for elem in error.get("loc", []))
            msg = error.get("msg", "Validation error")
            errors.append(f"{loc}: {msg}" if loc else msg)
        raise DatasetValidationError(
            f"Dataset validation failed for '{file_path}':\n  - " + "\n  - ".join(errors)
        ) from exc

    # Run semantic dataset-level validation
    validate_dataset(dataset)

    return dataset
