"""Tests for core domain models and public package exports."""

import pytest
from pydantic import ValidationError

from ragdiag import (
    EvaluationResult,
    Pipeline,
    QuerySample,
    RetrievedChunk,
    __version__,
)


def test_public_imports() -> None:
    """Verify all key domain entities and version are exposed at package root."""
    assert __version__ == "0.1.0"
    assert Pipeline is not None
    assert RetrievedChunk is not None
    assert QuerySample is not None
    assert EvaluationResult is not None


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
    """Tests for QuerySample model."""

    def test_sample_creation_with_defaults(self) -> None:
        sample = QuerySample(
            id="sample-001",
            query="What is the refund policy?",
            expected_answer="Refunds are processed within 5-7 business days.",
        )
        assert sample.id == "sample-001"
        assert sample.query == "What is the refund policy?"
        assert sample.expected_answer == "Refunds are processed within 5-7 business days."
        assert sample.relevant_chunk_ids == []
        assert sample.query_type == "general"

    def test_sample_creation_explicit(self) -> None:
        sample = QuerySample(
            id="sample-002",
            query="How do I verify a webhook signature?",
            expected_answer="Use the HMAC SHA256 signature verification helper.",
            relevant_chunk_ids=["chunk-10", "chunk-11"],
            query_type="technical",
        )
        assert sample.relevant_chunk_ids == ["chunk-10", "chunk-11"]
        assert sample.query_type == "technical"

    def test_sample_validation_missing_fields(self) -> None:
        with pytest.raises(ValidationError):
            QuerySample(id="sample-003", query="Incomplete sample")


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
