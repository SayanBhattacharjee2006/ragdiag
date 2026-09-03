"""Comprehensive unit and integration tests for the Root-Cause Diagnosis Engine."""

import pytest
from pydantic import ValidationError

from ragdiag.diagnosis.classifier import DiagnosisEngine
from ragdiag.diagnosis.models import DiagnosisResult, FailureCategory
from ragdiag.diagnosis.rules import (
    classify_answer_failure,
    classify_context_sufficiency,
    classify_grounding_failure,
    classify_latency_outlier,
    classify_pipeline_failure,
    classify_ranking_failure,
    classify_retrieval_failure,
)
from ragdiag.metrics.aggregation import aggregate_metrics
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.result import EvaluationResult
from ragdiag.models.sample import QuerySample, QueryType
from ragdiag.pipeline.base import Pipeline
from ragdiag.runner.evaluator import Evaluator

# ==============================================================================
# Model Tests
# ==============================================================================


class TestDiagnosisModels:
    """Tests for DiagnosisResult and FailureCategory models."""

    def test_failure_categories_match_taxonomy(self) -> None:
        """All 8 MVP failure categories exist in the FailureCategory enum."""
        expected = {
            "PASS",
            "WRONG_CHUNK_RETRIEVED",
            "WRONG_CHUNK_RANK",
            "INSUFFICIENT_CONTEXT",
            "RETRIEVED_BUT_NOT_GROUNDED",
            "ANSWER_INCORRECT",
            "LATENCY_OUTLIER",
            "UNKNOWN",
        }
        actual = {cat.value for cat in FailureCategory}
        assert actual == expected

    def test_diagnosis_result_valid(self) -> None:
        """Valid DiagnosisResult creates successfully."""
        diag = DiagnosisResult(
            category=FailureCategory.WRONG_CHUNK_RETRIEVED,
            severity="major",
            confidence=1.0,
            reason="Expected chunk was missing.",
            evidence=["Expected: doc_1", "Retrieved: doc_2"],
        )
        assert diag.category == FailureCategory.WRONG_CHUNK_RETRIEVED
        assert diag.severity == "major"
        assert diag.confidence == 1.0
        assert len(diag.evidence) == 2

    def test_diagnosis_result_invalid_confidence(self) -> None:
        """Confidence outside 0.0 to 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.5,
                reason="Pass",
            )


# ==============================================================================
# Pure Rule Classifier Tests
# ==============================================================================


class TestPureRules:
    """Tests for pure rule classification functions."""

    def test_classify_pipeline_failure(self) -> None:
        """status != 'completed' triggers UNKNOWN category."""
        diag = classify_pipeline_failure(status="failed", error="RuntimeError: DB down")
        assert diag is not None
        assert diag.category == FailureCategory.UNKNOWN
        assert diag.severity == "major"
        assert "RuntimeError: DB down" in diag.reason

        # Completed returns None
        assert classify_pipeline_failure(status="completed", error=None) is None

    def test_classify_retrieval_failure(self) -> None:
        """Zero overlap between expected and retrieved triggers WRONG_CHUNK_RETRIEVED."""
        expected = ["doc_17"]
        retrieved = [RetrievedChunk(id="doc_21", text="some text")]
        diag = classify_retrieval_failure(expected, retrieved, k=5)
        assert diag is not None
        assert diag.category == FailureCategory.WRONG_CHUNK_RETRIEVED
        assert diag.confidence == 1.0
        assert any("Recall@5: 0.0" in ev for ev in diag.evidence)

        # Overlap returns None
        retrieved_matching = [RetrievedChunk(id="doc_17", text="some text")]
        assert classify_retrieval_failure(expected, retrieved_matching, k=5) is None

    def test_classify_ranking_failure(self) -> None:
        """First relevant chunk appearing beyond rank_threshold triggers WRONG_CHUNK_RANK."""
        expected = ["doc_17"]
        # doc_17 is at rank 4 (1-indexed)
        retrieved = [
            RetrievedChunk(id="doc_1", text="t"),
            RetrievedChunk(id="doc_2", text="t"),
            RetrievedChunk(id="doc_3", text="t"),
            RetrievedChunk(id="doc_17", text="t"),
        ]
        diag = classify_ranking_failure(expected, retrieved, rank_threshold=3)
        assert diag is not None
        assert diag.category == FailureCategory.WRONG_CHUNK_RANK
        assert diag.severity == "warning"
        assert "rank 4" in diag.reason

        # At or within threshold returns None
        assert classify_ranking_failure(expected, retrieved, rank_threshold=4) is None

    def test_classify_context_sufficiency(self) -> None:
        """Partial retrieval of expected chunks triggers INSUFFICIENT_CONTEXT."""
        expected = ["doc_a", "doc_b", "doc_c"]
        retrieved = [
            RetrievedChunk(id="doc_a", text="t"),
            RetrievedChunk(id="doc_x", text="t"),
        ]
        diag = classify_context_sufficiency(expected, retrieved, k=5)
        assert diag is not None
        assert diag.category == FailureCategory.INSUFFICIENT_CONTEXT
        assert diag.severity == "warning"
        assert "1 of 3" in diag.reason
        assert any("Missing chunks: doc_b, doc_c" in ev for ev in diag.evidence)

        # Full retrieval returns None
        retrieved_all = [
            RetrievedChunk(id="doc_a", text="t"),
            RetrievedChunk(id="doc_b", text="t"),
            RetrievedChunk(id="doc_c", text="t"),
        ]
        assert classify_context_sufficiency(expected, retrieved_all, k=5) is None

    def test_classify_grounding_failure(self) -> None:
        """grounded=False triggers RETRIEVED_BUT_NOT_GROUNDED."""
        metrics: dict[str, object] = {
            "grounded": False,
            "answer_correct": True,
            "judge_confidence": 0.94,
            "judge_reason": "Answer claims 30 days while context says 7 days.",
        }
        diag = classify_grounding_failure(metrics)
        assert diag is not None
        assert diag.category == FailureCategory.RETRIEVED_BUT_NOT_GROUNDED
        assert diag.severity == "major"
        assert diag.confidence == 0.94
        assert any("Answer claims 30 days" in ev for ev in diag.evidence)

        # grounded=True returns None
        assert classify_grounding_failure({"grounded": True}) is None

    def test_classify_answer_failure(self) -> None:
        """answer_correct=False triggers ANSWER_INCORRECT."""
        metrics: dict[str, object] = {
            "grounded": True,
            "answer_correct": False,
            "judge_confidence": 0.88,
        }
        diag = classify_answer_failure(
            metrics, expected_answer="7 days", generated_answer="No refund"
        )
        assert diag is not None
        assert diag.category == FailureCategory.ANSWER_INCORRECT
        assert diag.severity == "major"
        assert diag.confidence == 0.88

        # answer_correct=True returns None
        assert classify_answer_failure({"answer_correct": True}) is None

    def test_classify_latency_outlier(self) -> None:
        """Retrieval latency exceeding threshold triggers LATENCY_OUTLIER."""
        diag = classify_latency_outlier({"retrieval_ms": 1250.0}, latency_threshold_ms=1000.0)
        assert diag is not None
        assert diag.category == FailureCategory.LATENCY_OUTLIER
        assert diag.severity == "warning"

        # Within threshold returns None
        assert classify_latency_outlier({"retrieval_ms": 500.0}, 1000.0) is None


# ==============================================================================
# Decision Precedence & DiagnosisEngine Tests
# ==============================================================================


class TestDiagnosisEnginePrecedence:
    """Tests evaluating DiagnosisEngine precedence hierarchy and edge cases."""

    @pytest.fixture
    def engine(self) -> DiagnosisEngine:
        return DiagnosisEngine(k=5, rank_threshold=3, latency_threshold_ms=1000.0)

    def test_pipeline_failure_takes_highest_precedence(self, engine: DiagnosisEngine) -> None:
        """Pipeline failure produces UNKNOWN regardless of any other field."""
        result = EvaluationResult(
            query_id="q1",
            query="q",
            status="failed",
            error="Connection refused",
            latency={"retrieval_ms": 2000.0},
        )
        diag = engine.diagnose(result)
        assert diag.category == FailureCategory.UNKNOWN
        assert diag.severity == "major"

    def test_retrieval_failure_precedence_over_latency(self, engine: DiagnosisEngine) -> None:
        """When retrieval fails AND latency is slow, WRONG_CHUNK_RETRIEVED is chosen."""
        result = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            expected_chunk_ids=["doc_1"],
            retrieved_chunks=[RetrievedChunk(id="doc_wrong", text="t")],
            latency={"retrieval_ms": 1500.0},
            metrics={"precision_at_5": 0.0, "recall_at_5": 0.0},
        )
        diag = engine.diagnose(result)
        assert diag.category == FailureCategory.WRONG_CHUNK_RETRIEVED
        # Secondary latency signal is preserved in evidence
        assert any("Secondary signal" in ev for ev in diag.evidence)

    def test_ranking_failure_precedence(self, engine: DiagnosisEngine) -> None:
        """First relevant chunk at rank 4 > 3 triggers WRONG_CHUNK_RANK."""
        result = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            expected_chunk_ids=["doc_1"],
            retrieved_chunks=[
                RetrievedChunk(id="x", text="t"),
                RetrievedChunk(id="y", text="t"),
                RetrievedChunk(id="z", text="t"),
                RetrievedChunk(id="doc_1", text="t"),
            ],
            latency={"retrieval_ms": 20.0},
            metrics={"precision_at_5": 0.25, "recall_at_5": 1.0},
        )
        diag = engine.diagnose(result)
        assert diag.category == FailureCategory.WRONG_CHUNK_RANK
        assert diag.severity == "warning"

    def test_insufficient_context_precedence(self, engine: DiagnosisEngine) -> None:
        """1 of 2 chunks retrieved at rank 1 triggers INSUFFICIENT_CONTEXT."""
        result = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            expected_chunk_ids=["doc_1", "doc_2"],
            retrieved_chunks=[
                RetrievedChunk(id="doc_1", text="t"),
                RetrievedChunk(id="x", text="t"),
            ],
            latency={"retrieval_ms": 20.0},
            metrics={"precision_at_5": 0.5, "recall_at_5": 0.5},
        )
        diag = engine.diagnose(result)
        assert diag.category == FailureCategory.INSUFFICIENT_CONTEXT
        assert diag.severity == "warning"

    def test_grounding_failure_precedence(self, engine: DiagnosisEngine) -> None:
        """Retrieved context present but ungrounded triggers RETRIEVED_BUT_NOT_GROUNDED."""
        result = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            expected_chunk_ids=["doc_1"],
            retrieved_chunks=[RetrievedChunk(id="doc_1", text="t")],
            latency={"retrieval_ms": 20.0},
            metrics={
                "precision_at_5": 1.0,
                "recall_at_5": 1.0,
                "grounded": False,
                "answer_correct": True,
                "judge_confidence": 0.91,
            },
        )
        diag = engine.diagnose(result)
        assert diag.category == FailureCategory.RETRIEVED_BUT_NOT_GROUNDED
        assert diag.confidence == 0.91

    def test_partial_retrieval_at_low_rank_diagnoses_insufficient_context(
        self, engine: DiagnosisEngine
    ) -> None:
        """Expected: [doc_a, doc_b], Retrieved: [doc_x, doc_y, doc_b] -> INSUFFICIENT_CONTEXT."""
        result = EvaluationResult(
            query_id="q_multi",
            query="Multi-hop query",
            status="completed",
            expected_chunk_ids=["doc_a", "doc_b"],
            retrieved_chunks=[
                RetrievedChunk(id="doc_x", text="text x"),
                RetrievedChunk(id="doc_y", text="text y"),
                RetrievedChunk(id="doc_b", text="text b"),
            ],
            latency={"retrieval_ms": 20.0},
            metrics={"precision_at_5": 0.33, "recall_at_5": 0.5},
        )
        diag = engine.diagnose(result)
        assert diag.category == FailureCategory.INSUFFICIENT_CONTEXT
        assert diag.severity == "warning"

    def test_answer_incorrect_precedence(self, engine: DiagnosisEngine) -> None:
        """Grounded=True, but answer_correct=False triggers ANSWER_INCORRECT."""
        result = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            expected_chunk_ids=["doc_1"],
            expected_answer="Refunds take 7 business days.",
            generated_answer="No refund allowed.",
            retrieved_chunks=[RetrievedChunk(id="doc_1", text="t")],
            latency={"retrieval_ms": 20.0},
            metrics={
                "precision_at_5": 1.0,
                "recall_at_5": 1.0,
                "grounded": True,
                "answer_correct": False,
                "judge_confidence": 0.95,
            },
        )
        diag = engine.diagnose(result)
        assert diag.category == FailureCategory.ANSWER_INCORRECT
        assert diag.confidence == 0.95
        assert any("Expected answer: Refunds take 7 business days." in ev for ev in diag.evidence)
        assert any("Generated answer: No refund allowed." in ev for ev in diag.evidence)

    def test_latency_outlier_when_quality_passes(self, engine: DiagnosisEngine) -> None:
        """When quality checks pass but latency > threshold, triggers LATENCY_OUTLIER."""
        result = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            expected_chunk_ids=["doc_1"],
            retrieved_chunks=[RetrievedChunk(id="doc_1", text="t")],
            latency={"retrieval_ms": 1400.0},
            metrics={"precision_at_5": 1.0, "recall_at_5": 1.0},
        )
        diag = engine.diagnose(result)
        assert diag.category == FailureCategory.LATENCY_OUTLIER
        assert diag.severity == "warning"

    def test_pass_with_judge(self, engine: DiagnosisEngine) -> None:
        """Query passing all retrieval, judge, and latency checks diagnoses as PASS."""
        result = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            expected_chunk_ids=["doc_1"],
            retrieved_chunks=[RetrievedChunk(id="doc_1", text="t")],
            latency={"retrieval_ms": 15.0},
            metrics={
                "precision_at_5": 1.0,
                "recall_at_5": 1.0,
                "grounded": True,
                "answer_correct": True,
            },
        )
        diag = engine.diagnose(result)
        assert diag.category == FailureCategory.PASS
        assert diag.severity == "info"
        assert "Query passed all" in diag.reason

    def test_pass_without_judge_states_semantic_not_evaluated(
        self, engine: DiagnosisEngine
    ) -> None:
        """In no-judge mode, PASS explains that semantic quality was not evaluated."""
        result = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            expected_chunk_ids=["doc_1"],
            retrieved_chunks=[RetrievedChunk(id="doc_1", text="t")],
            latency={"retrieval_ms": 15.0},
            metrics={"precision_at_5": 1.0, "recall_at_5": 1.0},
        )
        diag = engine.diagnose(result)
        assert diag.category == FailureCategory.PASS
        assert "semantic answer quality was not evaluated" in diag.reason


# ==============================================================================
# Aggregation & Query Type Breakdown Tests
# ==============================================================================


class TestDiagnosisAggregation:
    """Tests for aggregating diagnosis counts and breakdowns by query_type."""

    def test_aggregate_diagnosis_counts(self) -> None:
        """Verify aggregate_metrics summarizes diagnosis categories and failure count."""
        r1 = EvaluationResult(
            query_id="q1",
            query="q1",
            status="completed",
            expected_chunk_ids=["c1"],
            retrieved_chunks=[RetrievedChunk(id="c1", text="t")],
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
            query_type="factual",
        )
        r2 = EvaluationResult(
            query_id="q2",
            query="q2",
            status="completed",
            expected_chunk_ids=["c2"],
            retrieved_chunks=[RetrievedChunk(id="wrong", text="t")],
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Wrong retrieval",
            ),
            query_type="reasoning",
        )
        r3 = EvaluationResult(
            query_id="q3",
            query="q3",
            status="failed",
            error="Timeout",
            diagnosis=DiagnosisResult(
                category=FailureCategory.UNKNOWN,
                severity="major",
                confidence=1.0,
                reason="Pipeline failed",
            ),
            query_type="multi-hop",
        )

        report = aggregate_metrics([r1, r2, r3], k=5)
        assert report.total_queries == 3
        assert report.completed_queries == 2
        assert report.failed_queries == 1

        # Check diagnosis counts
        assert report.diagnosis_counts["PASS"] == 1
        assert report.diagnosis_counts["WRONG_CHUNK_RETRIEVED"] == 1
        assert report.diagnosis_counts["UNKNOWN"] == 1
        assert report.failure_count == 2  # r2 and r3 are failures

        # Check breakdown by query_type
        assert report.diagnosis_by_query_type["factual"]["PASS"] == 1
        assert report.diagnosis_by_query_type["reasoning"]["WRONG_CHUNK_RETRIEVED"] == 1
        assert report.diagnosis_by_query_type["multi-hop"]["UNKNOWN"] == 1


# ==============================================================================
# Evaluator Integration Tests
# ==============================================================================


class TestEvaluatorDiagnosisIntegration:
    """Tests confirming Evaluator assigns diagnosis to all results."""

    def test_evaluator_assigns_diagnosis_on_success(self) -> None:
        """Completed query receives DiagnosisResult."""

        class GoodPipeline(Pipeline):
            name = "good"

            def retrieve(self, query: str) -> list[RetrievedChunk]:
                return [RetrievedChunk(id="doc_1", text="t")]

            def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
                return "Answer"

        evaluator = Evaluator(k=5)
        sample = QuerySample(
            id="q1",
            query="What is x?",
            expected_answer="Ans",
            relevant_chunk_ids=["doc_1"],
            query_type=QueryType.FACTUAL,
        )
        result = evaluator.execute_sample(sample, GoodPipeline())
        assert isinstance(result.diagnosis, DiagnosisResult)
        assert result.diagnosis.category == FailureCategory.PASS

    def test_evaluator_assigns_diagnosis_on_pipeline_failure(self) -> None:
        """Failed query receives UNKNOWN DiagnosisResult."""

        class FailPipeline(Pipeline):
            name = "fail"

            def retrieve(self, query: str) -> list[RetrievedChunk]:
                raise RuntimeError("Boom")

            def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
                return ""

        evaluator = Evaluator(k=5)
        sample = QuerySample(
            id="q1",
            query="What is x?",
            expected_answer="Ans",
            relevant_chunk_ids=["doc_1"],
            query_type=QueryType.FACTUAL,
        )
        result = evaluator.execute_sample(sample, FailPipeline())
        assert result.status == "failed"
        assert isinstance(result.diagnosis, DiagnosisResult)
        assert result.diagnosis.category == FailureCategory.UNKNOWN
        assert "Boom" in result.diagnosis.reason

    def test_evaluator_preserves_expected_answer_on_success(self) -> None:
        """Evaluator populates expected_answer on successful execution."""

        class GoodPipeline(Pipeline):
            name = "good"

            def retrieve(self, query: str) -> list[RetrievedChunk]:
                return [RetrievedChunk(id="doc_1", text="t")]

            def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
                return "Answer"

        evaluator = Evaluator(k=5)
        sample = QuerySample(
            id="q1",
            query="Query",
            expected_answer="Expected reference answer.",
            relevant_chunk_ids=["doc_1"],
            query_type=QueryType.FACTUAL,
        )
        result = evaluator.execute_sample(sample, GoodPipeline())
        assert result.expected_answer == "Expected reference answer."

    def test_evaluator_preserves_expected_answer_on_retrieval_failure(self) -> None:
        """Evaluator populates expected_answer when retrieval fails."""

        class FailRetrievalPipeline(Pipeline):
            name = "fail_retrieval"

            def retrieve(self, query: str) -> list[RetrievedChunk]:
                raise RuntimeError("Retrieval crash")

            def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
                return ""

        evaluator = Evaluator(k=5)
        sample = QuerySample(
            id="q1",
            query="Query",
            expected_answer="Expected reference answer.",
            relevant_chunk_ids=["doc_1"],
            query_type=QueryType.FACTUAL,
        )
        result = evaluator.execute_sample(sample, FailRetrievalPipeline())
        assert result.status == "failed"
        assert result.expected_answer == "Expected reference answer."

    def test_evaluator_preserves_expected_answer_on_generation_failure(self) -> None:
        """Evaluator populates expected_answer when generation fails."""

        class FailGenerationPipeline(Pipeline):
            name = "fail_generation"

            def retrieve(self, query: str) -> list[RetrievedChunk]:
                return [RetrievedChunk(id="doc_1", text="t")]

            def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
                raise RuntimeError("Generation crash")

        evaluator = Evaluator(k=5)
        sample = QuerySample(
            id="q1",
            query="Query",
            expected_answer="Expected reference answer.",
            relevant_chunk_ids=["doc_1"],
            query_type=QueryType.FACTUAL,
        )
        result = evaluator.execute_sample(sample, FailGenerationPipeline())
        assert result.status == "failed"
        assert result.expected_answer == "Expected reference answer."
