"""Comprehensive tests for Phase 8 Multi-Pipeline Comparison."""

import pytest
from typer.testing import CliRunner

from ragdiag.cli.main import app
from ragdiag.comparison import (
    Comparator,
    ComparisonReport,
    MetricDeltas,
    compare_reports,
)
from ragdiag.dataset import load_dataset
from ragdiag.diagnosis.models import DiagnosisResult, FailureCategory
from ragdiag.judges.base import Judge, JudgeResult
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.result import EvaluationResult
from ragdiag.models.sample import QuerySample
from ragdiag.pipeline.base import Pipeline
from ragdiag.reporting.aggregator import build_report

runner = CliRunner()


class MockJudge(Judge):
    """Deterministic mock judge for testing semantic comparison."""

    def __init__(self, correctness_map: dict[str, bool], groundedness_map: dict[str, bool]):
        self.correctness_map = correctness_map
        self.groundedness_map = groundedness_map

    def judge(
        self, sample: QuerySample, retrieved_chunks: list[RetrievedChunk], generated_answer: str
    ) -> JudgeResult:
        is_correct = self.correctness_map.get(sample.id, True)
        is_grounded = self.groundedness_map.get(sample.id, True)
        return JudgeResult(
            answer_correct=is_correct,
            grounded=is_grounded,
            confidence=0.9,
            reasoning="Mock evaluation",
        )


class FastMockPipeline(Pipeline):
    """Mock pipeline simulating fast retrieval with partial accuracy."""

    name = "fast_pipeline"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if "auto-debit" in query:
            return [RetrievedChunk(id="doc_subscriptions_03", text="sub text", score=0.8)]
        return [
            RetrievedChunk(id="doc_refund_policy_01", text="refund text", score=0.9),
            RetrievedChunk(id="doc_webhooks_04", text="webhook text", score=0.9),
        ]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        return "Fast answer"


class AccurateMockPipeline(Pipeline):
    """Mock pipeline simulating complete retrieval with higher latency."""

    name = "accurate_pipeline"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(id="doc_refund_policy_01", text="refund text", score=0.95),
            RetrievedChunk(id="doc_webhooks_04", text="webhook text", score=0.95),
            RetrievedChunk(id="doc_compliance_02", text="comp text", score=0.95),
            RetrievedChunk(id="doc_auth_flows_09", text="auth text", score=0.95),
            RetrievedChunk(id="doc_subscriptions_03", text="sub text", score=0.95),
            RetrievedChunk(id="doc_mandates_05", text="mandate text", score=0.95),
            RetrievedChunk(id="doc_pricing_tier_01", text="price text", score=0.95),
            RetrievedChunk(id="doc_tax_regulations_03", text="tax text", score=0.95),
        ]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        return "Accurate answer"


class TestBasicComparison:
    """Tests for basic comparator execution and delta calculations."""

    def test_compare_two_pipelines_on_dataset(self) -> None:
        """Comparator evaluates both pipelines and returns complete ComparisonReport."""
        pipe_a = FastMockPipeline()
        pipe_b = AccurateMockPipeline()
        dataset = load_dataset("examples/basic_dataset.json")

        comparator = Comparator(k=5)
        report = comparator.compare(pipe_a, pipe_b, dataset)

        assert report.dataset_name == "basic_dataset"
        assert report.pipeline_a_name == "fast_pipeline"
        assert report.pipeline_b_name == "accurate_pipeline"
        assert report.pipeline_a_report.total_queries == 5
        assert report.pipeline_b_report.total_queries == 5
        assert isinstance(report.metric_deltas, MetricDeltas)
        assert report.metric_deltas.recall_at_k >= 0.0

    def test_exact_metric_deltas_calculation(self) -> None:
        """Metric deltas are calculated strictly as Pipeline B minus Pipeline A."""
        r1_a = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"precision_at_5": 0.6, "recall_at_5": 0.5, "reciprocal_rank": 0.5},
            latency={"retrieval_ms": 100.0},
        )
        r1_b = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"precision_at_5": 0.8, "recall_at_5": 1.0, "reciprocal_rank": 1.0},
            latency={"retrieval_ms": 150.0},
        )
        rep_a = build_report([r1_a], dataset_name="ds", pipeline_name="A", k=5)
        rep_b = build_report([r1_b], dataset_name="ds", pipeline_name="B", k=5)

        comp = compare_reports(rep_a, rep_b, [r1_a], [r1_b], k=5)
        assert comp.metric_deltas.precision_at_k == pytest.approx(0.2)
        assert comp.metric_deltas.recall_at_k == pytest.approx(0.5)
        assert comp.metric_deltas.mrr == pytest.approx(0.5)
        assert comp.metric_deltas.mean_retrieval_ms == pytest.approx(50.0)

    def test_negative_latency_delta_when_b_is_faster(self) -> None:
        """When Pipeline B is faster, mean_retrieval_ms delta is negative."""
        r1_a = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"precision_at_5": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 1.0},
            latency={"retrieval_ms": 200.0},
        )
        r1_b = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"precision_at_5": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 1.0},
            latency={"retrieval_ms": 50.0},
        )
        rep_a = build_report([r1_a], dataset_name="ds", pipeline_name="A", k=5)
        rep_b = build_report([r1_b], dataset_name="ds", pipeline_name="B", k=5)

        comp = compare_reports(rep_a, rep_b, [r1_a], [r1_b], k=5)
        assert comp.metric_deltas.mean_retrieval_ms == pytest.approx(-150.0)
        assert comp.latency_winner == "B"


class TestDiagnosisDeltas:
    """Tests for diagnostic failure count deltas across all 8 categories."""

    def test_all_8_categories_present_in_diagnosis_deltas(self) -> None:
        """All 8 taxonomy categories are present in diagnosis_deltas with B - A arithmetic."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Miss",
            ),
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="B")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        assert set(comp.diagnosis_deltas.keys()) == {cat.value for cat in FailureCategory}
        assert comp.diagnosis_deltas["PASS"] == 1
        assert comp.diagnosis_deltas["WRONG_CHUNK_RETRIEVED"] == -1
        assert comp.diagnosis_deltas["WRONG_CHUNK_RANK"] == 0


class TestSemanticComparison:
    """Tests for offline mode vs judged comparison."""

    def test_no_judge_semantic_deltas_are_none(self) -> None:
        """When no judge was configured, semantic deltas are None rather than 0."""
        r_a = EvaluationResult(query_id="q1", query="q", status="completed")
        r_b = EvaluationResult(query_id="q1", query="q", status="completed")
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="B")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        assert comp.metric_deltas.answer_correctness is None
        assert comp.metric_deltas.groundedness is None

    def test_judge_enabled_semantic_deltas(self) -> None:
        """When judge evaluates queries, semantic deltas are computed accurately."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"answer_correct": False, "grounded": True, "judge_confidence": 0.9},
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"answer_correct": True, "grounded": True, "judge_confidence": 0.9},
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="B")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        assert comp.metric_deltas.answer_correctness == pytest.approx(1.0)
        assert comp.metric_deltas.groundedness == pytest.approx(0.0)

    def test_judge_failures_preserved_separately(self) -> None:
        """Judge failures are preserved in the individual reports."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            judge_error="judge failed: RateLimitError",
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"answer_correct": True, "grounded": True},
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="B")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        assert comp.pipeline_a_report.judge_failures == 1
        assert comp.pipeline_b_report.judge_failures == 0


class TestQueryOutcomesMatching:
    """Tests for matching queries by query_id and classifying transitions."""

    def test_matching_by_query_id_independent_of_order(self) -> None:
        """Results are matched strictly by query_id even if list ordering differs."""
        r1_a = EvaluationResult(
            query_id="q1",
            query="query 1",
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Miss",
            ),
        )
        r2_a = EvaluationResult(
            query_id="q2",
            query="query 2",
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
        )

        # Reversed order in Pipeline B
        r2_b = EvaluationResult(
            query_id="q2",
            query="query 2",
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
        )
        r1_b = EvaluationResult(
            query_id="q1",
            query="query 1",
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
        )

        rep_a = build_report([r1_a, r2_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r2_b, r1_b], dataset_name="ds", pipeline_name="B")

        comp = compare_reports(rep_a, rep_b, [r1_a, r2_a], [r2_b, r1_b])
        assert len(comp.query_outcomes) == 2
        outcomes_by_id = {qo.query_id: qo for qo in comp.query_outcomes}
        assert outcomes_by_id["q1"].outcome == "improved"
        assert outcomes_by_id["q2"].outcome == "unchanged"
        assert comp.queries_improved == 1
        assert comp.queries_unchanged == 1
        assert comp.queries_regressed == 0

    def test_query_regression_classification(self) -> None:
        """When Pipeline B fails a query that passed in A, it is classified as regressed."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            diagnosis=DiagnosisResult(
                category=FailureCategory.ANSWER_INCORRECT,
                severity="major",
                confidence=1.0,
                reason="Incorrect",
            ),
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="B")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        assert comp.queries_regressed == 1
        assert comp.queries_improved == 0


class TestSameSeverityOutcomeClassification:
    """Regression tests for same-severity diagnosis transitions."""

    def test_same_severity_warning_category_change_not_unchanged(self) -> None:
        """Test 1: WRONG_CHUNK_RANK -> INSUFFICIENT_CONTEXT is NOT unchanged."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.5},
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RANK,
                severity="warning",
                confidence=1.0,
                reason="Wrong rank",
            ),
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.5},
            diagnosis=DiagnosisResult(
                category=FailureCategory.INSUFFICIENT_CONTEXT,
                severity="warning",
                confidence=1.0,
                reason="Insufficient context",
            ),
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="B")
        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])

        outcome = comp.query_outcomes[0].outcome
        assert outcome != "unchanged"
        assert outcome == "regressed"

    def test_same_severity_major_category_change_not_unchanged(self) -> None:
        """Test 2: WRONG_CHUNK_RETRIEVED -> ANSWER_INCORRECT (major -> major) is NOT unchanged."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.0},
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Miss",
            ),
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.0},
            diagnosis=DiagnosisResult(
                category=FailureCategory.ANSWER_INCORRECT,
                severity="major",
                confidence=1.0,
                reason="Wrong answer",
            ),
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="B")
        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])

        outcome = comp.query_outcomes[0].outcome
        assert outcome != "unchanged"
        assert outcome == "improved"

    def test_same_category_same_metrics_is_unchanged(self) -> None:
        """Test 3: WRONG_CHUNK_RETRIEVED -> WRONG_CHUNK_RETRIEVED with same metrics is unchanged."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.0},
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Miss",
            ),
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.0},
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Miss",
            ),
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="B")
        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])

        assert comp.query_outcomes[0].outcome == "unchanged"

    def test_same_category_recall_improves(self) -> None:
        """Test 4: Same category with improved recall is classified as improved."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.0},
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Miss",
            ),
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.5},
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Miss",
            ),
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="B")
        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])

        assert comp.query_outcomes[0].outcome == "improved"

    def test_same_category_recall_decreases(self) -> None:
        """Test 5: Same category with decreased recall is classified as regressed."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.5},
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Miss",
            ),
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.0},
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Miss",
            ),
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="B")
        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])

        assert comp.query_outcomes[0].outcome == "regressed"

    def test_different_same_severity_categories_measured_recall_improvement_wins(self) -> None:
        """Test 6: Measured recall improvement overrides category priority rank."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.4},
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RANK,
                severity="warning",
                confidence=1.0,
                reason="Wrong rank",
            ),
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.8},
            diagnosis=DiagnosisResult(
                category=FailureCategory.INSUFFICIENT_CONTEXT,
                severity="warning",
                confidence=1.0,
                reason="Insufficient context",
            ),
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="B")
        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])

        assert comp.query_outcomes[0].outcome == "improved"

    def test_different_same_severity_categories_measured_recall_regression_wins(self) -> None:
        """Test 7: Measured recall regression overrides category priority rank."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.8},
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Miss",
            ),
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            metrics={"recall_at_5": 0.2},
            diagnosis=DiagnosisResult(
                category=FailureCategory.ANSWER_INCORRECT,
                severity="major",
                confidence=1.0,
                reason="Wrong answer",
            ),
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="A")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="B")
        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])

        assert comp.query_outcomes[0].outcome == "regressed"


class TestWinnerAndTradeOffDetermination:
    """Tests for winner selection rules and qualitative trade-off detection."""

    def test_clear_quality_winner(self) -> None:
        """Pipeline B is declared overall winner when Recall improves significantly."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"recall_at_5": 0.5, "precision_at_5": 0.5, "reciprocal_rank": 0.5},
            latency={"retrieval_ms": 10.0},
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"recall_at_5": 1.0, "precision_at_5": 1.0, "reciprocal_rank": 1.0},
            latency={"retrieval_ms": 10.0},
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="PipeA", k=5)
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="PipeB", k=5)

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b], k=5)
        assert comp.quality_winner == "PipeB"
        assert comp.overall_winner == "PipeB"
        assert comp.winner == "PipeB"

    def test_quality_improves_latency_degrades_trade_off(self) -> None:
        """Higher quality with higher latency triggers trade-off detection."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"recall_at_5": 0.6, "precision_at_5": 0.6, "reciprocal_rank": 0.6},
            latency={"retrieval_ms": 20.0},
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"recall_at_5": 0.9, "precision_at_5": 0.9, "reciprocal_rank": 0.9},
            latency={"retrieval_ms": 120.0},
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="PipeA", k=5)
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="PipeB", k=5)

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b], k=5)
        assert comp.quality_winner == "PipeB"
        assert comp.latency_winner == "PipeA"
        assert comp.overall_winner == "PipeB"
        assert comp.winner == "PipeB"
        assert comp.trade_off == "Higher quality <-> higher latency"
        assert "improves Recall@5" in comp.summary

    def test_tied_performance(self) -> None:
        """When quality and latency differences are within tolerance, result is TIE."""
        r_a = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"recall_at_5": 1.0, "precision_at_5": 1.0, "reciprocal_rank": 1.0},
            latency={"retrieval_ms": 10.0},
        )
        r_b = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"recall_at_5": 1.0, "precision_at_5": 1.0, "reciprocal_rank": 1.0},
            latency={"retrieval_ms": 12.0},
        )
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="PipeA", k=5)
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="PipeB", k=5)

        comp = compare_reports(
            rep_a, rep_b, [r_a], [r_b], quality_tolerance=0.02, latency_tolerance_ms=10.0, k=5
        )
        assert comp.quality_winner == "TIE"
        assert comp.latency_winner == "TIE"
        assert comp.overall_winner == "TIE"
        assert comp.winner == "TIE"


class TestJSONSerialization:
    """Tests for Pydantic serialization roundtrip."""

    def test_comparison_report_json_roundtrip(self) -> None:
        """ComparisonReport serializes to and deserializes from JSON without data loss."""
        pipe_a = FastMockPipeline()
        pipe_b = AccurateMockPipeline()
        dataset = load_dataset("examples/basic_dataset.json")

        comparator = Comparator(k=5)
        report = comparator.compare(pipe_a, pipe_b, dataset)

        json_str = report.model_dump_json(indent=2)
        deserialized = ComparisonReport.model_validate_json(json_str)

        assert deserialized.dataset_name == report.dataset_name
        assert deserialized.pipeline_a_name == report.pipeline_a_name
        assert deserialized.pipeline_b_name == report.pipeline_b_name
        assert deserialized.metric_deltas.recall_at_k == report.metric_deltas.recall_at_k
        assert deserialized.queries_improved == report.queries_improved


class TestCLICompareCommand:
    """Integration tests for the 'ragdiag compare' CLI command."""

    def test_cli_compare_success(self) -> None:
        """ragdiag compare runs successfully and outputs side-by-side terminal report."""
        result = runner.invoke(
            app,
            [
                "compare",
                "--pipeline-a",
                "examples/dense_pipeline.py",
                "--pipeline-b",
                "examples/hybrid_pipeline.py",
                "--dataset",
                "examples/basic_dataset.json",
            ],
        )
        assert result.exit_code == 0
        assert "\x1b" not in result.output
        assert "RAGDiag Comparison" in result.output
        assert "OVERALL METRICS" in result.output
        assert "FAILURE COUNTS" in result.output
        assert "QUERY TYPES" in result.output
        assert "DECISION" in result.output
        assert "QUERY OUTCOMES" in result.output

    def test_cli_compare_with_output_json(self, tmp_path) -> None:
        """ragdiag compare writes valid JSON ComparisonReport when --output is provided."""
        out_file = tmp_path / "comp.json"
        result = runner.invoke(
            app,
            [
                "compare",
                "--pipeline-a",
                "examples/dense_pipeline.py",
                "--pipeline-b",
                "examples/hybrid_pipeline.py",
                "--dataset",
                "examples/basic_dataset.json",
                "--output",
                str(out_file),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()

        raw_json = out_file.read_text(encoding="utf-8")
        parsed = ComparisonReport.model_validate_json(raw_json)
        assert parsed.pipeline_a_name == "dense_pipeline"
        assert parsed.pipeline_b_name == "hybrid_pipeline"
        assert parsed.dataset_name == "basic_dataset"

    def test_cli_compare_missing_pipeline_a(self) -> None:
        """ragdiag compare exits with code 1 when pipeline A is missing."""
        result = runner.invoke(
            app,
            [
                "compare",
                "--pipeline-a",
                "examples/nonexistent.py",
                "--pipeline-b",
                "examples/hybrid_pipeline.py",
                "--dataset",
                "examples/basic_dataset.json",
            ],
        )
        assert result.exit_code == 1
        assert "Pipeline A Load Failed" in result.output
