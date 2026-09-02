"""Extensive unit tests for retrieval quality metrics, latency calculations, and aggregation."""

import pytest

from ragdiag.metrics.aggregation import aggregate_metrics, mean_reciprocal_rank
from ragdiag.metrics.latency import calculate_latency_summary, calculate_percentile
from ragdiag.metrics.models import RetrievalMetricResult
from ragdiag.metrics.retrieval import (
    compute_retrieval_metrics,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.result import EvaluationResult


def _make_chunks(*ids: str) -> list[RetrievedChunk]:
    """Helper to construct RetrievedChunk objects from a list of IDs."""
    return [RetrievedChunk(id=cid, text=f"Text for {cid}") for cid in ids]


# ==============================================================================
# Precision@K Tests
# ==============================================================================


class TestPrecisionAtK:
    """Test suite for precision_at_k metric calculation."""

    def test_perfect_precision(self) -> None:
        """All retrieved chunks in top-K are relevant."""
        chunks = _make_chunks("c1", "c2", "c3")
        score = precision_at_k(relevant_chunk_ids=["c1", "c2", "c3"], retrieved_chunks=chunks, k=3)
        assert score == 1.0

    def test_zero_precision(self) -> None:
        """No retrieved chunks in top-K are relevant."""
        chunks = _make_chunks("c1", "c2", "c3")
        score = precision_at_k(relevant_chunk_ids=["c9", "c10"], retrieved_chunks=chunks, k=3)
        assert score == 0.0

    def test_partial_precision(self) -> None:
        """2 out of 5 retrieved chunks are relevant."""
        chunks = _make_chunks("c1", "c7", "c3", "c2", "c9")
        score = precision_at_k(relevant_chunk_ids=["c1", "c2"], retrieved_chunks=chunks, k=5)
        assert score == pytest.approx(0.4)

    def test_k_equals_one(self) -> None:
        """Only rank 1 is inspected."""
        chunks = _make_chunks("c1", "c2", "c3")
        assert precision_at_k(["c1"], chunks, k=1) == 1.0
        assert precision_at_k(["c2"], chunks, k=1) == 0.0

    def test_k_greater_than_returned_results(self) -> None:
        """When fewer results than K are returned, avoid artificially penalizing denominator."""
        chunks = _make_chunks("c1", "c2")
        # 2 returned, both relevant, K=5 -> denominator is min(5, 2) = 2 -> 2/2 = 1.0
        score = precision_at_k(["c1", "c2"], chunks, k=5)
        assert score == 1.0

        # 2 returned, 1 relevant, K=5 -> 1/2 = 0.5
        score_partial = precision_at_k(["c1"], chunks, k=5)
        assert score_partial == 0.5

    def test_empty_retrieved_list(self) -> None:
        """Empty retrieval returns 0.0 without division by zero."""
        score = precision_at_k(["c1", "c2"], [], k=5)
        assert score == 0.0

    def test_empty_relevant_ids(self) -> None:
        """Empty ground truth relevant set returns 0.0."""
        chunks = _make_chunks("c1", "c2")
        assert precision_at_k([], chunks, k=5) == 0.0

    def test_k_zero_or_negative(self) -> None:
        """Non-positive K returns 0.0."""
        chunks = _make_chunks("c1", "c2")
        assert precision_at_k(["c1"], chunks, k=0) == 0.0
        assert precision_at_k(["c1"], chunks, k=-1) == 0.0

    def test_multiple_relevant_documents(self) -> None:
        """Multiple relevant chunks in retrieved results are counted."""
        chunks = _make_chunks("c1", "c2", "c3", "c4")
        score = precision_at_k(["c1", "c3", "c4"], chunks, k=4)
        assert score == pytest.approx(3 / 4)

    def test_relevant_document_outside_top_k(self) -> None:
        """Relevant documents beyond top-K are ignored."""
        chunks = _make_chunks("irr1", "irr2", "irr3", "c1")
        # c1 is at rank 4, but K=3 -> not in top 3
        score = precision_at_k(["c1"], chunks, k=3)
        assert score == 0.0

    def test_duplicate_retrieved_chunk_ids_behavior(self) -> None:
        """Duplicate chunk IDs in retrieval count only once toward the numerator."""
        chunks = _make_chunks("c1", "c1", "c2")
        # c1 is duplicated; relevant set is {"c1"}. Top-2 has ["c1", "c1"].
        # Unique relevant in top-2 is {"c1"}, denominator is min(2, 3) = 2.
        score = precision_at_k(["c1"], chunks, k=2)
        assert score == pytest.approx(1 / 2)


# ==============================================================================
# Recall@K Tests
# ==============================================================================


class TestRecallAtK:
    """Test suite for recall_at_k metric calculation."""

    def test_perfect_recall(self) -> None:
        """All ground truth relevant chunks retrieved in top-K."""
        chunks = _make_chunks("c1", "c2", "c3")
        score = recall_at_k(relevant_chunk_ids=["c1", "c2"], retrieved_chunks=chunks, k=3)
        assert score == 1.0

    def test_partial_recall(self) -> None:
        """Only 1 of 2 relevant chunks retrieved in top-K."""
        chunks = _make_chunks("c1", "c7", "c3", "c4", "c9")
        score = recall_at_k(relevant_chunk_ids=["c1", "c2"], retrieved_chunks=chunks, k=5)
        assert score == pytest.approx(0.5)

    def test_zero_recall(self) -> None:
        """None of the relevant chunks retrieved in top-K."""
        chunks = _make_chunks("c1", "c2", "c3")
        score = recall_at_k(relevant_chunk_ids=["c9", "c10"], retrieved_chunks=chunks, k=3)
        assert score == 0.0

    def test_empty_retrieval(self) -> None:
        """Empty retrieval returns 0.0."""
        score = recall_at_k(relevant_chunk_ids=["c1", "c2"], retrieved_chunks=[], k=5)
        assert score == 0.0

    def test_no_relevant_ids(self) -> None:
        """No ground-truth relevant IDs returns 0.0 defensively."""
        chunks = _make_chunks("c1", "c2")
        score = recall_at_k(relevant_chunk_ids=[], retrieved_chunks=chunks, k=5)
        assert score == 0.0

    def test_k_smaller_than_list(self) -> None:
        """Relevant chunk beyond K is not counted toward recall."""
        chunks = _make_chunks("irr1", "irr2", "c1")
        # c1 is at rank 3, K=2 -> not retrieved within top 2
        score = recall_at_k(relevant_chunk_ids=["c1"], retrieved_chunks=chunks, k=2)
        assert score == 0.0

    def test_k_larger_than_list(self) -> None:
        """K larger than number of retrieved results searches all returned chunks."""
        chunks = _make_chunks("c1", "c2")
        score = recall_at_k(relevant_chunk_ids=["c1", "c2"], retrieved_chunks=chunks, k=10)
        assert score == 1.0

    def test_k_zero_or_negative(self) -> None:
        """Non-positive K returns 0.0."""
        chunks = _make_chunks("c1")
        assert recall_at_k(["c1"], chunks, k=0) == 0.0
        assert recall_at_k(["c1"], chunks, k=-5) == 0.0

    def test_multiple_relevant_chunks(self) -> None:
        """Correct recall with 3 relevant chunks, 2 retrieved."""
        chunks = _make_chunks("c1", "c3", "irr")
        score = recall_at_k(relevant_chunk_ids=["c1", "c2", "c3"], retrieved_chunks=chunks, k=5)
        assert score == pytest.approx(2 / 3)

    def test_duplicate_retrieved_chunk_ids(self) -> None:
        """Duplicate retrieved chunks do not inflate recall numerator."""
        chunks = _make_chunks("c1", "c1")
        score = recall_at_k(relevant_chunk_ids=["c1", "c2"], retrieved_chunks=chunks, k=5)
        assert score == pytest.approx(0.5)


# ==============================================================================
# Reciprocal Rank Tests
# ==============================================================================


class TestReciprocalRank:
    """Test suite for single-query reciprocal_rank calculation."""

    def test_relevant_document_at_rank_1(self) -> None:
        """First relevant document at rank 1 -> RR = 1.0."""
        chunks = _make_chunks("doc_a", "doc_b")
        assert reciprocal_rank(["doc_a"], chunks) == 1.0

    def test_relevant_document_at_rank_2(self) -> None:
        """First relevant document at rank 2 -> RR = 0.5."""
        chunks = _make_chunks("irr", "doc_a")
        assert reciprocal_rank(["doc_a"], chunks) == 0.5

    def test_relevant_document_at_rank_5(self) -> None:
        """First relevant document at rank 5 -> RR = 0.2."""
        chunks = _make_chunks("i1", "i2", "i3", "i4", "doc_a")
        assert reciprocal_rank(["doc_a"], chunks) == pytest.approx(0.2)

    def test_no_relevant_document(self) -> None:
        """No relevant document retrieved -> RR = 0.0."""
        chunks = _make_chunks("i1", "i2")
        assert reciprocal_rank(["doc_a"], chunks) == 0.0

    def test_multiple_relevant_documents_uses_first_rank_only(self) -> None:
        """When multiple relevant documents are retrieved, only the FIRST relevant rank is used."""
        chunks = _make_chunks("i1", "doc_a", "doc_b")
        # doc_a is at rank 2, doc_b is at rank 3 -> should use rank 2 (1/2 = 0.5)
        assert reciprocal_rank(["doc_a", "doc_b"], chunks) == 0.5

    def test_empty_retrieved(self) -> None:
        """Empty retrieved list returns 0.0."""
        assert reciprocal_rank(["doc_a"], []) == 0.0

    def test_empty_relevant_ids(self) -> None:
        """Empty relevant set returns 0.0."""
        assert reciprocal_rank([], _make_chunks("doc_a")) == 0.0


# ==============================================================================
# Mean Reciprocal Rank (MRR) Tests
# ==============================================================================


class TestMeanReciprocalRank:
    """Test suite for mean_reciprocal_rank aggregation function."""

    def test_all_rank_1(self) -> None:
        """All queries having rank 1 yields MRR = 1.0."""
        assert mean_reciprocal_rank([1.0, 1.0, 1.0]) == 1.0

    def test_ranks_1_and_2(self) -> None:
        """Queries with rank 1 (1.0) and rank 2 (0.5) yields MRR = 0.75."""
        assert mean_reciprocal_rank([1.0, 0.5]) == pytest.approx(0.75)

    def test_ranks_2_and_4(self) -> None:
        """Queries with rank 2 (0.5) and rank 4 (0.25) yields MRR = 0.375."""
        assert mean_reciprocal_rank([0.5, 0.25]) == pytest.approx(0.375)

    def test_missing_relevance(self) -> None:
        """Queries where none found relevant document yields MRR = 0.0."""
        assert mean_reciprocal_rank([0.0, 0.0]) == 0.0

    def test_mixed_results(self) -> None:
        """Mixed set of queries [1.0, 0.0, 0.5] yields MRR = 0.5."""
        assert mean_reciprocal_rank([1.0, 0.0, 0.5]) == pytest.approx(0.5)

    def test_empty_list(self) -> None:
        """Empty list returns 0.0."""
        assert mean_reciprocal_rank([]) == 0.0


# ==============================================================================
# Latency Tests
# ==============================================================================


class TestLatency:
    """Test suite for percentile calculation and latency summary."""

    def test_one_value(self) -> None:
        """Single latency measurement sets min, max, mean, and percentiles to that value."""
        summary = calculate_latency_summary([42.5])
        assert summary.count == 1
        assert summary.mean_ms == 42.5
        assert summary.min_ms == 42.5
        assert summary.max_ms == 42.5
        assert summary.p50_ms == 42.5
        assert summary.p95_ms == 42.5
        assert summary.p99_ms == 42.5

    def test_multiple_values(self) -> None:
        """Multiple latency values produce expected min, max, and mean."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        summary = calculate_latency_summary(values)
        assert summary.count == 5
        assert summary.mean_ms == 30.0
        assert summary.min_ms == 10.0
        assert summary.max_ms == 50.0
        assert summary.p50_ms == 30.0

    def test_empty_values(self) -> None:
        """Empty input safely returns all zeros."""
        summary = calculate_latency_summary([])
        assert summary.count == 0
        assert summary.mean_ms == 0.0
        assert summary.p50_ms == 0.0
        assert summary.p95_ms == 0.0
        assert summary.p99_ms == 0.0
        assert summary.min_ms == 0.0
        assert summary.max_ms == 0.0

    def test_known_percentile_dataset(self) -> None:
        """Verify linear interpolation matches known mathematical benchmarks."""
        # 0 to 100 in steps of 1 (101 points)
        dataset = list(range(101))
        assert calculate_percentile(dataset, 50.0) == pytest.approx(50.0)
        assert calculate_percentile(dataset, 95.0) == pytest.approx(95.0)
        assert calculate_percentile(dataset, 99.0) == pytest.approx(99.0)
        assert calculate_percentile(dataset, 0.0) == 0.0
        assert calculate_percentile(dataset, 100.0) == 100.0

    def test_percentile_invalid_arguments(self) -> None:
        """Percentile raises ValueError on empty list or out-of-range percentiles."""
        with pytest.raises(ValueError, match="empty sequence"):
            calculate_percentile([], 50.0)
        with pytest.raises(ValueError, match="between 0.0 and 100.0"):
            calculate_percentile([1.0, 2.0], -5.0)
        with pytest.raises(ValueError, match="between 0.0 and 100.0"):
            calculate_percentile([1.0, 2.0], 105.0)

    def test_non_negative_values(self) -> None:
        """Typical non-negative millisecond values."""
        summary = calculate_latency_summary([0.12, 0.45, 1.20, 0.88])
        assert summary.min_ms >= 0.0
        assert summary.mean_ms >= 0.0


# ==============================================================================
# Aggregation & Model Tests
# ==============================================================================


class TestAggregation:
    """Test suite for aggregate_metrics function."""

    def test_aggregation_all_completed(self) -> None:
        """Aggregates correctly when all results are completed."""
        r1 = EvaluationResult(
            query_id="q1",
            query="q1",
            status="completed",
            expected_chunk_ids=["c1"],
            retrieved_chunks=_make_chunks("c1"),
            metrics={"precision_at_5": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 1.0},
            latency={"retrieval_ms": 10.0},
        )
        r2 = EvaluationResult(
            query_id="q2",
            query="q2",
            status="completed",
            expected_chunk_ids=["c2"],
            retrieved_chunks=_make_chunks("irr", "c2"),
            metrics={"precision_at_5": 0.5, "recall_at_5": 1.0, "reciprocal_rank": 0.5},
            latency={"retrieval_ms": 20.0},
        )

        report = aggregate_metrics([r1, r2], k=5)
        assert report.total_queries == 2
        assert report.completed_queries == 2
        assert report.failed_queries == 0
        assert report.mean_precision_at_k == pytest.approx(0.75)
        assert report.mean_recall_at_k == pytest.approx(1.0)
        assert report.mrr == pytest.approx(0.75)
        assert report.retrieval_latency.count == 2
        assert report.retrieval_latency.mean_ms == 15.0

    def test_failed_results_excluded_from_quality_and_latency_metrics(self) -> None:
        """Failed queries must NOT contaminate retrieval quality or latency denominators."""
        r_success = EvaluationResult(
            query_id="q1",
            query="q1",
            status="completed",
            expected_chunk_ids=["c1"],
            retrieved_chunks=_make_chunks("c1"),
            metrics={"precision_at_5": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 1.0},
            latency={"retrieval_ms": 10.0},
        )
        r_fail = EvaluationResult(
            query_id="q2",
            query="q2",
            status="failed",
            error="Connection timeout",
            metrics={},
            latency={"retrieval_ms": 999.0},
        )

        report = aggregate_metrics([r_success, r_fail], k=5)
        assert report.total_queries == 2
        assert report.completed_queries == 1
        assert report.failed_queries == 1
        # Metrics computed ONLY on r_success
        assert report.mean_precision_at_k == 1.0
        assert report.mean_recall_at_k == 1.0
        assert report.mrr == 1.0
        # Latency computed ONLY on r_success
        assert report.retrieval_latency.count == 1
        assert report.retrieval_latency.mean_ms == 10.0

    def test_all_failed_queries(self) -> None:
        """All queries failed produces 0.0 metrics with full failed count."""
        r_fail = EvaluationResult(
            query_id="q1",
            query="q1",
            status="failed",
            error="Server error",
        )
        report = aggregate_metrics([r_fail], k=5)
        assert report.total_queries == 1
        assert report.completed_queries == 0
        assert report.failed_queries == 1
        assert report.mean_precision_at_k == 0.0
        assert report.mean_recall_at_k == 0.0
        assert report.mrr == 0.0

    def test_compute_retrieval_metrics_helper(self) -> None:
        """Test compute_retrieval_metrics helper model creation."""
        res = compute_retrieval_metrics(
            relevant_chunk_ids=["c1"],
            retrieved_chunks=_make_chunks("c1", "c2"),
            k=5,
        )
        assert isinstance(res, RetrievalMetricResult)
        assert res.precision_at_k == 0.5
        assert res.recall_at_k == 1.0
        assert res.reciprocal_rank == 1.0
        assert res.k == 5
