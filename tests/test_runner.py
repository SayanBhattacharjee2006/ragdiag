"""Comprehensive tests for the Evaluator execution engine."""

import pytest

from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.dataset import GoldenDataset
from ragdiag.models.sample import QuerySample, QueryType
from ragdiag.pipeline.base import Pipeline
from ragdiag.runner.evaluator import Evaluator


class MockSuccessfulPipeline(Pipeline):
    """Deterministic pipeline that succeeds on all calls."""

    name = "mock_success"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(id="c1", text=f"Context for {query}", score=0.9),
            RetrievedChunk(id="c2", text="Additional context", score=0.8),
        ]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        return f"Generated answer for {query} with {len(chunks)} chunks."


class MockEmptyRetrievalPipeline(Pipeline):
    """Pipeline that returns an empty chunk list."""

    name = "mock_empty_retrieval"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return []

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        return f"Fallback answer with {len(chunks)} chunks."


class MockRetrievalErrorPipeline(Pipeline):
    """Pipeline whose retrieve() raises a network error."""

    name = "mock_retrieval_error"

    def __init__(self) -> None:
        super().__init__()
        self.generate_called = False

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        raise ConnectionError("Vector database unavailable")

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        self.generate_called = True
        return "Answer"


class MockGenerationErrorPipeline(Pipeline):
    """Pipeline whose generate() raises an inference timeout."""

    name = "mock_generation_error"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return [RetrievedChunk(id="c1", text="Context chunk", score=0.95)]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        raise TimeoutError("LLM generation request timed out")


class MockInvalidRetrievalTypePipeline(Pipeline):
    """Pipeline whose retrieve() returns non-list or non-RetrievedChunk objects."""

    name = "mock_invalid_retrieval"

    def __init__(self, mode: str = "not_a_list") -> None:
        super().__init__()
        self.mode = mode

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if self.mode == "not_a_list":
            return "not a list"  # type: ignore[return-value]
        # list with raw string
        return ["not_a_chunk"]  # type: ignore[list-item]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        return "Answer"


class MockInvalidGenerationTypePipeline(Pipeline):
    """Pipeline whose generate() returns a non-string."""

    name = "mock_invalid_generation"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return [RetrievedChunk(id="c1", text="Chunk")]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        return {"answer": "not a string"}  # type: ignore[return-value]


class MockSelectiveFailurePipeline(Pipeline):
    """Pipeline that fails specifically on query q002."""

    name = "mock_selective_failure"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if "fail_on_retrieval" in query:
            raise RuntimeError("Intentional retrieval failure on query 2")
        return [RetrievedChunk(id="c1", text=f"Context for {query}")]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        return f"Answer for {query}"


@pytest.fixture
def single_sample() -> QuerySample:
    return QuerySample(
        id="sample_01",
        query="What is the refund period?",
        expected_answer="7 business days",
        relevant_chunk_ids=["c1"],
        query_type=QueryType.FACTUAL,
    )


@pytest.fixture
def multi_sample_dataset() -> GoldenDataset:
    return GoldenDataset(
        name="test_dataset",
        version="1.0",
        samples=[
            QuerySample(
                id="q001",
                query="First query",
                expected_answer="First answer",
                relevant_chunk_ids=["c1"],
                query_type=QueryType.FACTUAL,
            ),
            QuerySample(
                id="q002",
                query="Second query fail_on_retrieval",
                expected_answer="Second answer",
                relevant_chunk_ids=["c2"],
                query_type=QueryType.REASONING,
            ),
            QuerySample(
                id="q003",
                query="Third query",
                expected_answer="Third answer",
                relevant_chunk_ids=["c3"],
                query_type=QueryType.MULTI_HOP,
            ),
        ],
    )


def test_successful_execution(single_sample: QuerySample) -> None:
    """Verify that a successful pipeline run captures all raw evidence and latencies."""
    pipeline = MockSuccessfulPipeline()
    evaluator = Evaluator()

    result = evaluator.execute_sample(single_sample, pipeline)

    assert result.query_id == "sample_01"
    assert result.query == "What is the refund period?"
    assert result.expected_chunk_ids == ["c1"]
    assert len(result.retrieved_chunks) == 2
    assert result.retrieved_chunks[0].id == "c1"
    assert "Generated answer" in (result.generated_answer or "")
    assert result.status == "completed"
    assert result.error is None

    # Check latency fields
    assert "retrieval_ms" in result.latency
    assert "generation_ms" in result.latency
    assert "total_ms" in result.latency
    assert result.latency["retrieval_ms"] >= 0.0
    assert result.latency["generation_ms"] >= 0.0
    assert result.latency["total_ms"] >= result.latency["retrieval_ms"]

    # Check retrieval metrics
    assert "precision_at_5" in result.metrics
    assert "recall_at_5" in result.metrics
    assert "reciprocal_rank" in result.metrics
    assert result.metrics["precision_at_5"] == 0.5
    assert result.metrics["recall_at_5"] == 1.0
    assert result.metrics["reciprocal_rank"] == 1.0

    # Verify future metrics and diagnosis are NOT yet populated
    assert "groundedness" not in result.metrics
    assert "answer_correctness" not in result.metrics
    assert result.diagnosis == {}

    # Check query_type preservation
    assert result.query_type == "factual"


def test_empty_retrieval_execution(single_sample: QuerySample) -> None:
    """Verify that retrieve() returning an empty list completes successfully."""
    pipeline = MockEmptyRetrievalPipeline()
    evaluator = Evaluator()

    result = evaluator.execute_sample(single_sample, pipeline)

    assert result.status == "completed"
    assert result.retrieved_chunks == []
    assert result.generated_answer == "Fallback answer with 0 chunks."
    assert result.error is None
    assert result.latency["total_ms"] >= 0.0
    assert result.metrics["precision_at_5"] == 0.0
    assert result.metrics["recall_at_5"] == 0.0
    assert result.metrics["reciprocal_rank"] == 0.0


def test_retrieval_exception_handling(single_sample: QuerySample) -> None:
    """Verify retrieval exception marks query as failed and does not call generate."""
    pipeline = MockRetrievalErrorPipeline()
    evaluator = Evaluator()

    result = evaluator.execute_sample(single_sample, pipeline)

    assert result.status == "failed"
    assert "retrieval failed" in (result.error or "")
    assert "ConnectionError" in (result.error or "")
    assert not pipeline.generate_called
    assert result.generated_answer is None
    assert result.metrics == {}
    assert result.query_type == "factual"


def test_generation_exception_handling(single_sample: QuerySample) -> None:
    """Verify generation exception marks query as failed but preserves retrieved chunks."""
    pipeline = MockGenerationErrorPipeline()
    evaluator = Evaluator()

    result = evaluator.execute_sample(single_sample, pipeline)

    assert result.status == "failed"
    assert "generation failed" in (result.error or "")
    assert "TimeoutError" in (result.error or "")
    assert len(result.retrieved_chunks) == 1
    assert result.generated_answer is None
    assert result.metrics == {}
    assert result.query_type == "factual"


def test_invalid_retrieval_output_not_a_list(single_sample: QuerySample) -> None:
    """Verify error when retrieve() returns a non-list object."""
    pipeline = MockInvalidRetrievalTypePipeline(mode="not_a_list")
    evaluator = Evaluator()

    result = evaluator.execute_sample(single_sample, pipeline)

    assert result.status == "failed"
    assert result.error is not None
    assert "retrieval failed: TypeError" in result.error
    assert "expected list[RetrievedChunk]" in result.error


def test_invalid_retrieval_output_non_chunk_items(single_sample: QuerySample) -> None:
    """Verify error when retrieve() returns a list containing non-RetrievedChunk items."""
    pipeline = MockInvalidRetrievalTypePipeline(mode="non_chunk_items")
    evaluator = Evaluator()

    result = evaluator.execute_sample(single_sample, pipeline)

    assert result.status == "failed"
    assert result.error is not None
    assert "retrieval failed: TypeError" in result.error
    assert "expected RetrievedChunk" in result.error


def test_invalid_generation_output_non_string(single_sample: QuerySample) -> None:
    """Verify error when generate() returns a non-string object."""
    pipeline = MockInvalidGenerationTypePipeline()
    evaluator = Evaluator()

    result = evaluator.execute_sample(single_sample, pipeline)

    assert result.status == "failed"
    assert result.error is not None
    assert "generation failed: TypeError" in result.error
    assert "expected str" in result.error


def test_error_isolation_multi_query(multi_sample_dataset: GoldenDataset) -> None:
    """Verify error isolation: failure on middle query does not abort evaluation."""
    pipeline = MockSelectiveFailurePipeline()
    evaluator = Evaluator()

    results = evaluator.evaluate(pipeline, multi_sample_dataset)

    assert len(results) == 3

    # Query 1: Completed
    assert results[0].query_id == "q001"
    assert results[0].status == "completed"
    assert results[0].error is None

    # Query 2: Failed (Intentional)
    assert results[1].query_id == "q002"
    assert results[1].status == "failed"
    assert "Intentional retrieval failure" in (results[1].error or "")

    # Query 3: Completed (Executed despite query 2 failure!)
    assert results[2].query_id == "q003"
    assert results[2].status == "completed"
    assert results[2].error is None
    assert "Answer for Third query" in (results[2].generated_answer or "")
