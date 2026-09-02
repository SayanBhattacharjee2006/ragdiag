"""Tests for the golden dataset loader, validator, and CLI command."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragdiag.cli.main import app
from ragdiag.dataset import (
    DatasetLoadError,
    DatasetValidationError,
    load_dataset,
    validate_dataset,
)
from ragdiag.models.sample import QueryType

runner = CliRunner()


def create_json_dataset(path: Path, data: dict) -> Path:
    """Helper to write a dictionary as JSON to a given path."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def test_load_valid_dataset(tmp_path: Path) -> None:
    """Verify that a valid JSON dataset loads correctly into GoldenDataset."""
    file_path = tmp_path / "valid_dataset.json"
    data = {
        "name": "test_dataset",
        "version": "1.0",
        "samples": [
            {
                "id": "q1",
                "query": "What is the settlement window?",
                "expected_answer": "T+2 business days.",
                "relevant_chunk_ids": ["doc_01"],
                "query_type": "factual",
            },
            {
                "id": "q2",
                "query": "Why was the settlement delayed?",
                "expected_answer": "Bank holiday on Friday.",
                "relevant_chunk_ids": ["doc_02", "doc_03"],
                "query_type": "reasoning",
            },
        ],
    }
    create_json_dataset(file_path, data)

    ds = load_dataset(file_path)
    assert ds.name == "test_dataset"
    assert ds.version == "1.0"
    assert len(ds.samples) == 2
    assert ds.samples[0].id == "q1"
    assert ds.samples[0].query_type == QueryType.FACTUAL
    assert ds.samples[1].query_type == QueryType.REASONING

    summary = validate_dataset(ds)
    assert summary.total_samples == 2
    assert summary.query_type_counts == {"factual": 1, "reasoning": 1}


def test_missing_file(tmp_path: Path) -> None:
    """Verify DatasetLoadError is raised when file does not exist."""
    missing_path = tmp_path / "non_existent.json"
    with pytest.raises(DatasetLoadError, match="does not exist"):
        load_dataset(missing_path)


def test_non_json_extension(tmp_path: Path) -> None:
    """Verify DatasetLoadError is raised for non-json files."""
    csv_file = tmp_path / "dataset.csv"
    csv_file.write_text("id,query\nq1,text", encoding="utf-8")
    with pytest.raises(DatasetLoadError, match="must have a .json extension"):
        load_dataset(csv_file)


def test_invalid_json_syntax(tmp_path: Path) -> None:
    """Verify DatasetLoadError is raised when JSON is malformed."""
    broken_file = tmp_path / "broken.json"
    broken_file.write_text("{ name: 'missing quotes' ", encoding="utf-8")
    with pytest.raises(DatasetLoadError, match="Invalid JSON syntax"):
        load_dataset(broken_file)


def test_json_root_not_object(tmp_path: Path) -> None:
    """Verify DatasetValidationError when JSON root is a list rather than object."""
    list_file = tmp_path / "list.json"
    list_file.write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Expected dataset root to be a JSON object"):
        load_dataset(list_file)


def test_missing_required_dataset_fields(tmp_path: Path) -> None:
    """Verify DatasetValidationError when required fields (name, version) are missing."""
    file_path = tmp_path / "missing_fields.json"
    create_json_dataset(file_path, {"samples": []})
    with pytest.raises(DatasetValidationError):
        load_dataset(file_path)


def test_empty_samples(tmp_path: Path) -> None:
    """Verify empty samples list is rejected."""
    file_path = tmp_path / "empty_samples.json"
    create_json_dataset(file_path, {"name": "empty", "version": "1.0", "samples": []})
    with pytest.raises(DatasetValidationError):
        load_dataset(file_path)


def test_duplicate_sample_ids(tmp_path: Path) -> None:
    """Verify duplicate sample IDs are rejected."""
    file_path = tmp_path / "duplicate_ids.json"
    data = {
        "name": "dups",
        "version": "1.0",
        "samples": [
            {
                "id": "q001",
                "query": "Query 1",
                "expected_answer": "Answer 1",
                "relevant_chunk_ids": ["c1"],
                "query_type": "factual",
            },
            {
                "id": "q001",
                "query": "Query 2",
                "expected_answer": "Answer 2",
                "relevant_chunk_ids": ["c2"],
                "query_type": "reasoning",
            },
        ],
    }
    create_json_dataset(file_path, data)
    with pytest.raises(DatasetValidationError, match="duplicate sample IDs"):
        load_dataset(file_path)


def test_empty_query_rejected(tmp_path: Path) -> None:
    """Verify empty or whitespace-only query is rejected."""
    file_path = tmp_path / "empty_query.json"
    data = {
        "name": "test",
        "version": "1.0",
        "samples": [
            {
                "id": "q1",
                "query": "   ",
                "expected_answer": "Answer",
                "relevant_chunk_ids": ["c1"],
                "query_type": "factual",
            }
        ],
    }
    create_json_dataset(file_path, data)
    with pytest.raises(DatasetValidationError, match="must not be empty"):
        load_dataset(file_path)


def test_empty_expected_answer_rejected(tmp_path: Path) -> None:
    """Verify empty or whitespace-only expected answer is rejected."""
    file_path = tmp_path / "empty_answer.json"
    data = {
        "name": "test",
        "version": "1.0",
        "samples": [
            {
                "id": "q1",
                "query": "Query",
                "expected_answer": "",
                "relevant_chunk_ids": ["c1"],
                "query_type": "factual",
            }
        ],
    }
    create_json_dataset(file_path, data)
    with pytest.raises(DatasetValidationError, match="must not be empty"):
        load_dataset(file_path)


def test_empty_relevant_chunk_ids_rejected(tmp_path: Path) -> None:
    """Verify sample with empty relevant_chunk_ids is rejected."""
    file_path = tmp_path / "empty_chunks.json"
    data = {
        "name": "test",
        "version": "1.0",
        "samples": [
            {
                "id": "q1",
                "query": "Query",
                "expected_answer": "Answer",
                "relevant_chunk_ids": [],
                "query_type": "factual",
            }
        ],
    }
    create_json_dataset(file_path, data)
    with pytest.raises(DatasetValidationError, match="at least one chunk ID"):
        load_dataset(file_path)


def test_duplicate_relevant_chunk_ids_rejected(tmp_path: Path) -> None:
    """Verify duplicate relevant_chunk_ids within the same sample are rejected."""
    file_path = tmp_path / "dup_chunks.json"
    data = {
        "name": "test",
        "version": "1.0",
        "samples": [
            {
                "id": "q1",
                "query": "Query",
                "expected_answer": "Answer",
                "relevant_chunk_ids": ["chunk_a", "chunk_a"],
                "query_type": "factual",
            }
        ],
    }
    create_json_dataset(file_path, data)
    with pytest.raises(DatasetValidationError, match="Duplicate chunk ID"):
        load_dataset(file_path)


def test_empty_string_inside_relevant_chunk_ids_rejected(tmp_path: Path) -> None:
    """Verify whitespace or empty chunk ID inside relevant_chunk_ids is rejected."""
    file_path = tmp_path / "empty_chunk_str.json"
    data = {
        "name": "test",
        "version": "1.0",
        "samples": [
            {
                "id": "q1",
                "query": "Query",
                "expected_answer": "Answer",
                "relevant_chunk_ids": ["valid_chunk", "  "],
                "query_type": "factual",
            }
        ],
    }
    create_json_dataset(file_path, data)
    with pytest.raises(DatasetValidationError, match="must not be empty"):
        load_dataset(file_path)


def test_unsupported_query_type_rejected(tmp_path: Path) -> None:
    """Verify unsupported query type is rejected."""
    file_path = tmp_path / "unsupported_type.json"
    data = {
        "name": "test",
        "version": "1.0",
        "samples": [
            {
                "id": "q1",
                "query": "Query",
                "expected_answer": "Answer",
                "relevant_chunk_ids": ["c1"],
                "query_type": "open-ended-opinion",
            }
        ],
    }
    create_json_dataset(file_path, data)
    with pytest.raises(DatasetValidationError):
        load_dataset(file_path)


def test_query_types_supported_factual_reasoning_multihop(tmp_path: Path) -> None:
    """Verify factual, reasoning, and multi-hop query types are all accepted."""
    file_path = tmp_path / "taxonomy.json"
    data = {
        "name": "taxonomy_test",
        "version": "1.0",
        "samples": [
            {
                "id": "q1",
                "query": "What is A?",
                "expected_answer": "A is alpha.",
                "relevant_chunk_ids": ["c1"],
                "query_type": "factual",
            },
            {
                "id": "q2",
                "query": "Why did B occur?",
                "expected_answer": "Because of C.",
                "relevant_chunk_ids": ["c2"],
                "query_type": "reasoning",
            },
            {
                "id": "q3",
                "query": "How does D connect to E across F?",
                "expected_answer": "Via multi-step route.",
                "relevant_chunk_ids": ["c3", "c4"],
                "query_type": "multi-hop",
            },
        ],
    }
    create_json_dataset(file_path, data)
    ds = load_dataset(file_path)
    assert ds.samples[0].query_type == QueryType.FACTUAL
    assert ds.samples[1].query_type == QueryType.REASONING
    assert ds.samples[2].query_type == QueryType.MULTI_HOP

    summary = validate_dataset(ds)
    assert summary.query_type_counts == {
        "factual": 1,
        "reasoning": 1,
        "multi-hop": 1,
    }


def test_examples_basic_dataset_validates() -> None:
    """Verify the bundled examples/basic_dataset.json loads and validates cleanly."""
    example_path = Path("examples/basic_dataset.json")
    assert example_path.exists()

    ds = load_dataset(example_path)
    assert ds.name == "basic_dataset"
    assert ds.version == "1.0"
    assert len(ds.samples) == 5

    summary = validate_dataset(ds)
    assert summary.total_samples == 5
    assert summary.query_type_counts == {
        "factual": 2,
        "reasoning": 2,
        "multi-hop": 1,
    }


def test_cli_validate_success(tmp_path: Path) -> None:
    """Verify CLI validate command succeeds with exit code 0 on valid dataset."""
    file_path = tmp_path / "cli_valid.json"
    data = {
        "name": "cli_dataset",
        "version": "1.0",
        "samples": [
            {
                "id": "q1",
                "query": "What is UPI?",
                "expected_answer": "Unified Payments Interface.",
                "relevant_chunk_ids": ["doc_upi"],
                "query_type": "factual",
            }
        ],
    }
    create_json_dataset(file_path, data)

    result = runner.invoke(app, ["validate", "--dataset", str(file_path)])
    assert result.exit_code == 0
    assert "Dataset: cli_dataset" in result.output
    assert "Version: 1.0" in result.output
    assert "Samples: 1" in result.output
    assert "factual: 1" in result.output
    assert "Dataset is valid." in result.output


def test_cli_validate_failure_on_invalid_file(tmp_path: Path) -> None:
    """Verify CLI validate command exits with non-zero code on invalid dataset."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{ malformed }", encoding="utf-8")

    result = runner.invoke(app, ["validate", "--dataset", str(invalid_file)])
    assert result.exit_code != 0
    assert "Validation Failed" in result.output


def test_cli_run_success() -> None:
    """Verify CLI run command executes basic_pipeline against basic_dataset with metrics."""
    pipeline_file = "examples/basic_pipeline.py"
    dataset_file = "examples/basic_dataset.json"

    result = runner.invoke(
        app,
        ["run", "--pipeline", pipeline_file, "--dataset", dataset_file],
    )
    assert result.exit_code == 0
    assert "\x1b" not in result.output
    assert "Pipeline: basic_pipeline" in result.output
    assert "Dataset:  basic_dataset" in result.output
    assert "Queries:  5" in result.output
    assert "Evaluation complete." in result.output
    assert "Retrieval Metrics" in result.output
    assert "Precision@5:" in result.output
    assert "Recall@5:" in result.output
    assert "MRR:" in result.output
    assert "Retrieval Latency" in result.output
    assert "Mean:" in result.output
    assert "P50:" in result.output
    assert "P95:" in result.output
    assert "P99:" in result.output
    assert "Completed:  5" in result.output
    assert "Failed:     0" in result.output


def test_cli_run_missing_pipeline() -> None:
    """Verify CLI run command handles missing pipeline file gracefully."""
    result = runner.invoke(
        app,
        ["run", "--pipeline", "non_existent.py", "--dataset", "examples/basic_dataset.json"],
    )
    assert result.exit_code != 0
    assert "Pipeline Load Failed" in result.output


def test_cli_run_without_judge_has_no_semantic_metrics() -> None:
    """Verify default run without --judge does NOT include semantic metrics."""
    result = runner.invoke(
        app,
        [
            "run",
            "--pipeline",
            "examples/basic_pipeline.py",
            "--dataset",
            "examples/basic_dataset.json",
        ],
    )
    assert result.exit_code == 0
    assert "Semantic Metrics" not in result.output


def test_cli_run_openai_judge_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify requesting --judge openai with missing API key exits cleanly with error message."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(
        app,
        [
            "run",
            "--pipeline",
            "examples/basic_pipeline.py",
            "--dataset",
            "examples/basic_dataset.json",
            "--judge",
            "openai",
        ],
    )
    assert result.exit_code != 0
    assert "OpenAI API Key Missing" in result.output


def test_cli_run_unsupported_judge() -> None:
    """Verify requesting unsupported judge provider exits cleanly with error message."""
    result = runner.invoke(
        app,
        [
            "run",
            "--pipeline",
            "examples/basic_pipeline.py",
            "--dataset",
            "examples/basic_dataset.json",
            "--judge",
            "unknown_provider",
        ],
    )
    assert result.exit_code != 0
    assert "Unsupported Judge Provider" in result.output
