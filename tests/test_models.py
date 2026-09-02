"""Tests for core domain models and public package exports."""

import pytest
from pydantic import ValidationError

from ragdiag import (
    EvaluationResult,
    GoldenDataset,
    Judge,
    JudgeResult,
    OpenAIJudge,
    Pipeline,
    QuerySample,
    QueryType,
    RetrievedChunk,
    __version__,
    aggregate_metrics,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_public_imports() -> None:
    """Verify all key domain entities and version are exposed at package root."""
    assert __version__ == "0.1.0"
    assert Pipeline is not None
    assert RetrievedChunk is not None
    assert QuerySample is not None
    assert EvaluationResult is not None
    assert GoldenDataset is not None
    assert QueryType is not None
    assert Judge is not None
    assert JudgeResult is not None
    assert OpenAIJudge is not None
    assert precision_at_k is not None
    assert recall_at_k is not None
    assert reciprocal_rank is not None
    assert mean_reciprocal_rank is not None
    assert aggregate_metrics is not None


class TestRetrievedChunk:
    """Tests for RetrievedChunk model."""

    def test_chunk_creation_minimal(self) -> None:
        chunk = RetrievedChunk(id="chunk-001", text="Sample document text.")
        assert chunk.id == "chunk-001"
        assert chunk.text == "Sample document text."
        assert chunk.score is None
        assert chunk.metadata is None

    def test_chunk_creation_full(self) -> None:
        chunk = RetrievedChunk(
            id="chunk-002",
            text="Another text segment.",
            score=0.88,
            metadata={"source": "manual.pdf", "page": 4},
        )
        assert chunk.id == "chunk-002"
        assert chunk.score == 0.88
        assert chunk.metadata == {"source": "manual.pdf", "page": 4}

    def test_chunk_validation_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            RetrievedChunk(id="chunk-003")  # Missing text


class TestQuerySample:
    """Tests for QuerySample model with strengthened validation."""

    def test_sample_creation_valid(self) -> None:
        sample = QuerySample(
            id="sample-001",
            query="What is the refund policy?",
            expected_answer="Refunds are processed within 5-7 business days.",
            relevant_chunk_ids=["chunk-01", "chunk-02"],
            query_type=QueryType.FACTUAL,
        )
        assert sample.id == "sample-001"
        assert sample.query == "What is the refund policy?"
        assert sample.expected_answer == "Refunds are processed within 5-7 business days."
        assert sample.relevant_chunk_ids == ["chunk-01", "chunk-02"]
        assert sample.query_type == QueryType.FACTUAL
        assert sample.query_type == "factual"

    def test_sample_creation_string_query_type_coercion(self) -> None:
        sample = QuerySample(
            id="sample-002",
            query="Why did the transaction fail?",
            expected_answer="Card expired.",
            relevant_chunk_ids=["chunk-03"],
            query_type="reasoning",
        )
        assert sample.query_type == QueryType.REASONING

    def test_sample_validation_empty_id(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            QuerySample(
                id="   ",
                query="Query",
                expected_answer="Answer",
                relevant_chunk_ids=["c1"],
                query_type=QueryType.FACTUAL,
            )

    def test_sample_validation_empty_query(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            QuerySample(
                id="sample-01",
                query="",
                expected_answer="Answer",
                relevant_chunk_ids=["c1"],
                query_type=QueryType.FACTUAL,
            )

    def test_sample_validation_empty_expected_answer(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            QuerySample(
                id="sample-01",
                query="Query",
                expected_answer="   ",
                relevant_chunk_ids=["c1"],
                query_type=QueryType.FACTUAL,
            )

    def test_sample_validation_empty_chunk_ids(self) -> None:
        with pytest.raises(ValidationError, match="at least one chunk ID"):
            QuerySample(
                id="sample-01",
                query="Query",
                expected_answer="Answer",
                relevant_chunk_ids=[],
                query_type=QueryType.FACTUAL,
            )

    def test_sample_validation_duplicate_chunk_ids(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate chunk ID"):
            QuerySample(
                id="sample-01",
                query="Query",
                expected_answer="Answer",
                relevant_chunk_ids=["c1", "c1"],
                query_type=QueryType.FACTUAL,
            )

    def test_sample_validation_unsupported_query_type(self) -> None:
        with pytest.raises(ValidationError):
            QuerySample(
                id="sample-01",
                query="Query",
                expected_answer="Answer",
                relevant_chunk_ids=["c1"],
                query_type="unsupported-type",
            )


class TestGoldenDatasetModel:
    """Tests for GoldenDataset domain model."""

    def test_dataset_creation_valid(self) -> None:
        sample = QuerySample(
            id="q1",
            query="What is 3DS?",
            expected_answer="3D Secure authentication.",
            relevant_chunk_ids=["doc1"],
            query_type=QueryType.FACTUAL,
        )
        ds = GoldenDataset(name="eval_v1", version="1.0", samples=[sample])
        assert ds.name == "eval_v1"
        assert ds.version == "1.0"
        assert len(ds.samples) == 1

    def test_dataset_empty_samples_fails(self) -> None:
        with pytest.raises(ValidationError):
            GoldenDataset(name="eval_v1", version="1.0", samples=[])

    def test_dataset_duplicate_sample_ids_fails(self) -> None:
        sample1 = QuerySample(
            id="dup_id",
            query="Q1",
            expected_answer="A1",
            relevant_chunk_ids=["doc1"],
            query_type=QueryType.FACTUAL,
        )
        sample2 = QuerySample(
            id="dup_id",
            query="Q2",
            expected_answer="A2",
            relevant_chunk_ids=["doc2"],
            query_type=QueryType.REASONING,
        )
        with pytest.raises(ValidationError, match="duplicate sample IDs"):
            GoldenDataset(name="eval_v1", version="1.0", samples=[sample1, sample2])


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_result_creation_minimal_defaults(self) -> None:
        result = EvaluationResult(
            query_id="sample-001",
            query="What is the refund policy?",
        )
        assert result.query_id == "sample-001"
        assert result.query == "What is the refund policy?"
        assert result.expected_chunk_ids == []
        assert result.retrieved_chunks == []
        assert result.generated_answer is None
        assert result.metrics == {}
        assert result.diagnosis == {}
        assert result.latency == {}
        assert result.status == "completed"
        assert result.error is None

    def test_result_creation_full(self) -> None:
        chunk = RetrievedChunk(id="c1", text="Policy details", score=0.95)
        result = EvaluationResult(
            query_id="sample-002",
            query="How to initiate a refund?",
            expected_chunk_ids=["c1"],
            retrieved_chunks=[chunk],
            generated_answer="Call the refund API endpoint.",
            metrics={"context_precision": 1.0, "faithfulness": 1.0},
            diagnosis={"category": "none", "root_cause": "clean_run"},
            latency={"retrieval": 0.12, "generation": 0.45},
            status="completed",
            error=None,
        )
        assert len(result.retrieved_chunks) == 1
        assert result.retrieved_chunks[0].id == "c1"
        assert result.metrics["faithfulness"] == 1.0
        assert result.latency["generation"] == 0.45
        assert result.status == "completed"

    def test_result_failure_status(self) -> None:
        result = EvaluationResult(
            query_id="sample-003",
            query="Failing query",
            status="failed",
            error="Connection timeout during retrieval",
        )
        assert result.status == "failed"
        assert result.error == "Connection timeout during retrieval"
