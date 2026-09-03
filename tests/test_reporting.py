"""Comprehensive tests for Phase 7 Diagnostic Intelligence and EvaluationReport."""

import pytest
from typer.testing import CliRunner

from ragdiag.cli.main import app
from ragdiag.diagnosis.models import DiagnosisResult, FailureCategory
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.result import EvaluationResult
from ragdiag.reporting import EvaluationReport, TopFailure, build_report
from ragdiag.reporting.insights import generate_insights

runner = CliRunner()


class TestReportCountsAndStructure:
    """Tests for basic report structure, metadata, and counts."""

    def test_report_creation_and_counts(self) -> None:
        """Report accurately computes total, completed, and failed queries."""
        r1 = EvaluationResult(
            query_id="q1",
            query="Query 1",
            status="completed",
            expected_chunk_ids=["c1"],
            retrieved_chunks=[RetrievedChunk(id="c1", text="t")],
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
            metrics={"precision_at_5": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 1.0},
            latency={"retrieval_ms": 10.0},
            query_type="factual",
        )
        r2 = EvaluationResult(
            query_id="q2",
            query="Query 2",
            status="failed",
            error="Connection error",
            diagnosis=DiagnosisResult(
                category=FailureCategory.UNKNOWN,
                severity="major",
                confidence=1.0,
                reason="Pipeline error",
            ),
            query_type="factual",
        )

        report = build_report(
            [r1, r2],
            dataset_name="test_ds",
            dataset_version="2.0",
            pipeline_name="test_pipe",
            k=5,
        )

        assert report.total_queries == 2
        assert report.completed_queries == 1
        assert report.failed_queries == 1
        assert report.dataset_name == "test_ds"
        assert report.dataset_version == "2.0"
        assert report.pipeline_name == "test_pipe"
        assert report.retrieval.k == 5
        assert report.retrieval.mean_precision_at_k == 1.0
        assert report.retrieval.mean_recall_at_k == 1.0
        assert report.retrieval.mrr == 1.0

    def test_all_8_categories_present_with_zeros(self) -> None:
        """Report retains all 8 FailureCategory values in diagnosis_counts for schema stability."""
        r1 = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
        )
        report = build_report([r1])

        expected_cats = {cat.value for cat in FailureCategory}
        assert set(report.diagnosis_counts.keys()) == expected_cats
        assert report.diagnosis_counts["PASS"] == 1
        assert report.diagnosis_counts["WRONG_CHUNK_RETRIEVED"] == 0
        assert report.diagnosis_counts["WRONG_CHUNK_RANK"] == 0
        assert report.diagnosis_counts["INSUFFICIENT_CONTEXT"] == 0
        assert report.diagnosis_counts["RETRIEVED_BUT_NOT_GROUNDED"] == 0
        assert report.diagnosis_counts["ANSWER_INCORRECT"] == 0
        assert report.diagnosis_counts["LATENCY_OUTLIER"] == 0
        assert report.diagnosis_counts["UNKNOWN"] == 0


class TestSemanticAggregationStates:
    """Tests for semantic evaluation states: no judge, judge success, and judge failure."""

    def test_no_judge_configured(self) -> None:
        """When no judge was used, semantic summary is None."""
        r = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"precision_at_5": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 1.0},
        )
        report = build_report([r])
        assert report.semantic is None
        assert report.judged_queries == 0
        assert report.judge_failures == 0

    def test_judge_configured_and_successful(self) -> None:
        """When judge evaluates queries, semantic rates are computed accurately."""
        r1 = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"answer_correct": True, "grounded": True, "judge_confidence": 0.9},
        )
        r2 = EvaluationResult(
            query_id="q2",
            query="q",
            status="completed",
            metrics={"answer_correct": False, "grounded": True, "judge_confidence": 0.8},
        )
        report = build_report([r1, r2])
        assert report.semantic is not None
        assert report.judged_queries == 2
        assert report.judge_failures == 0
        assert report.semantic.answer_correctness_rate == 0.5
        assert report.semantic.groundedness_rate == 1.0
        assert report.semantic.mean_judge_confidence == pytest.approx(0.85)

    def test_judge_failures_isolated_from_denominators(self) -> None:
        """Judge failures are counted separately and excluded from semantic denominators."""
        r1 = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            metrics={"answer_correct": True, "grounded": True, "judge_confidence": 0.95},
        )
        r2 = EvaluationResult(
            query_id="q2",
            query="q",
            status="completed",
            judge_error="judge failed: TimeoutError",
        )
        report = build_report([r1, r2])
        assert report.semantic is not None
        assert report.judged_queries == 1
        assert report.judge_failures == 1
        assert report.semantic.answer_correctness_rate == 1.0
        assert report.semantic.groundedness_rate == 1.0


class TestQueryTypeBreakdown:
    """Tests for query type metrics and diagnosis breakdowns."""

    def test_query_types_metrics_and_diagnosis(self) -> None:
        """Metrics and failure counts are partitioned across query types."""
        r_fact = EvaluationResult(
            query_id="q1",
            query="q",
            query_type="factual",
            status="completed",
            metrics={"recall_at_5": 1.0, "precision_at_5": 1.0, "reciprocal_rank": 1.0},
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
        )
        r_multi = EvaluationResult(
            query_id="q2",
            query="q",
            query_type="multi-hop",
            status="completed",
            metrics={"recall_at_5": 0.5, "precision_at_5": 0.25, "reciprocal_rank": 0.5},
            diagnosis=DiagnosisResult(
                category=FailureCategory.INSUFFICIENT_CONTEXT,
                severity="warning",
                confidence=1.0,
                reason="Partial context",
            ),
        )
        report = build_report([r_fact, r_multi])

        assert "factual" in report.metrics_by_query_type
        assert "multi-hop" in report.metrics_by_query_type

        fact_qm = report.metrics_by_query_type["factual"]
        assert fact_qm.mean_recall_at_k == 1.0
        assert fact_qm.mean_precision_at_k == 1.0
        assert fact_qm.diagnosis_counts["PASS"] == 1

        multi_qm = report.metrics_by_query_type["multi-hop"]
        assert multi_qm.mean_recall_at_k == 0.5
        assert multi_qm.mean_precision_at_k == 0.25
        assert multi_qm.diagnosis_counts["INSUFFICIENT_CONTEXT"] == 1


class TestTopFailuresRanking:
    """Tests for deterministic top-failures selection and tie-breaking."""

    def test_top_failures_deterministic_ordering(self) -> None:
        """Failures ordered by severity (major > warning), -confidence, category, query_id."""
        # Warning severity, high confidence
        r_warn = EvaluationResult(
            query_id="q_warn",
            query="q",
            diagnosis=DiagnosisResult(
                category=FailureCategory.INSUFFICIENT_CONTEXT,
                severity="warning",
                confidence=1.0,
                reason="Warning reason",
            ),
        )
        # Major severity, confidence 0.9
        r_major_1 = EvaluationResult(
            query_id="q_major_b",
            query="q",
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=0.9,
                reason="Major 1 reason",
            ),
        )
        # Major severity, confidence 0.9, tie-breaker query_id 'q_major_a' < 'q_major_b'
        r_major_2 = EvaluationResult(
            query_id="q_major_a",
            query="q",
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=0.9,
                reason="Major 2 reason",
            ),
        )
        # PASS result (must be excluded from top failures)
        r_pass = EvaluationResult(
            query_id="q_pass",
            query="q",
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
        )

        report = build_report([r_warn, r_major_1, r_major_2, r_pass], max_top_failures=5)
        assert len(report.top_failures) == 3
        # Major failures come before warning
        assert report.top_failures[0].query_id == "q_major_a"
        assert report.top_failures[1].query_id == "q_major_b"
        assert report.top_failures[2].query_id == "q_warn"
        assert all(isinstance(tf, TopFailure) for tf in report.top_failures)


class TestDeterministicInsights:
    """Tests for rule-based deterministic insight generation."""

    def test_all_passed_insight(self) -> None:
        """When all queries pass, 'All evaluated queries passed' insight is generated."""
        r = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
        )
        report = build_report([r])
        assert any("All evaluated queries passed" in ins for ins in report.overall_insights)

    def test_dominant_failure_mode_insight(self) -> None:
        """Dominant failure mode is identified and reported with count and percentage."""
        r1 = EvaluationResult(
            query_id="q1",
            query="q",
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Wrong retrieval",
            ),
        )
        r2 = EvaluationResult(
            query_id="q2",
            query="q",
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Wrong retrieval",
            ),
        )
        r3 = EvaluationResult(
            query_id="q3",
            query="q",
            diagnosis=DiagnosisResult(
                category=FailureCategory.INSUFFICIENT_CONTEXT,
                severity="warning",
                confidence=1.0,
                reason="Context",
            ),
        )
        report = build_report([r1, r2, r3])
        insights = generate_insights(report)
        assert any("Wrong-chunk retrieval is the dominant failure mode" in ins for ins in insights)

    def test_weakest_query_type_insight(self) -> None:
        """Query type with significantly lower recall is identified."""
        r_fact = EvaluationResult(
            query_id="q1",
            query="q",
            query_type="factual",
            status="completed",
            metrics={"recall_at_5": 0.95, "precision_at_5": 1.0, "reciprocal_rank": 1.0},
        )
        r_multi = EvaluationResult(
            query_id="q2",
            query="q",
            query_type="multi-hop",
            status="completed",
            metrics={"recall_at_5": 0.45, "precision_at_5": 0.5, "reciprocal_rank": 0.5},
        )
        report = build_report([r_fact, r_multi])
        insights = generate_insights(report)
        assert any(
            "multi-hop' queries have the weakest retrieval performance" in ins for ins in insights
        )

    def test_latency_concern_insight(self) -> None:
        """Latency outliers trigger latency concern insight."""
        r = EvaluationResult(
            query_id="q1",
            query="q",
            status="completed",
            diagnosis=DiagnosisResult(
                category=FailureCategory.LATENCY_OUTLIER,
                severity="warning",
                confidence=1.0,
                reason="Slow",
            ),
            latency={"retrieval_ms": 1500.0},
        )
        report = build_report([r])
        insights = generate_insights(report)
        assert any("Latency concerns detected" in ins for ins in insights)


class TestJSONSerializationAndEdgeCases:
    """Tests for Pydantic serialization roundtrip and edge case handling."""

    def test_json_serialization_roundtrip(self) -> None:
        """EvaluationReport serializes to and deserializes from JSON without data loss."""
        r = EvaluationResult(
            query_id="q1",
            query="Sample query",
            status="completed",
            expected_chunk_ids=["c1"],
            retrieved_chunks=[RetrievedChunk(id="c1", text="text")],
            metrics={"precision_at_5": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 1.0},
            latency={"retrieval_ms": 12.5},
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
            query_type="factual",
        )
        report = build_report([r], dataset_name="ds", dataset_version="1.0", pipeline_name="pipe")
        json_str = report.model_dump_json(indent=2)
        deserialized = EvaluationReport.model_validate_json(json_str)

        assert deserialized.dataset_name == report.dataset_name
        assert deserialized.total_queries == report.total_queries
        assert deserialized.retrieval.mean_precision_at_k == report.retrieval.mean_precision_at_k
        assert deserialized.diagnosis_counts["PASS"] == 1
        assert "factual" in deserialized.metrics_by_query_type

    def test_empty_results_edge_case(self) -> None:
        """Empty results list produces valid zero-initialized report without ZeroDivisionError."""
        report = build_report([])
        assert report.total_queries == 0
        assert report.completed_queries == 0
        assert report.failed_queries == 0
        assert report.retrieval.mean_precision_at_k == 0.0
        assert report.retrieval.mean_recall_at_k == 0.0
        assert report.retrieval.mrr == 0.0
        assert report.semantic is None
        assert report.top_failures == []

    def test_all_failed_queries_edge_case(self) -> None:
        """All failed queries handled safely without divide-by-zero."""
        r = EvaluationResult(
            query_id="q1",
            query="q",
            status="failed",
            error="Boom",
            diagnosis=DiagnosisResult(
                category=FailureCategory.UNKNOWN,
                severity="major",
                confidence=1.0,
                reason="Pipeline error",
            ),
        )
        report = build_report([r])
        assert report.total_queries == 1
        assert report.completed_queries == 0
        assert report.failed_queries == 1
        assert report.retrieval.mean_precision_at_k == 0.0
        assert report.diagnosis_counts["UNKNOWN"] == 1


class TestCLIReportAndOutputOption:
    """Integration test verifying CLI run displays full report and writes valid JSON output."""

    def test_cli_run_with_output_file(self, tmp_path) -> None:
        """ragdiag run writes valid JSON EvaluationReport when --output is specified."""
        output_file = tmp_path / "eval_report.json"
        pipeline_file = "examples/basic_pipeline.py"
        dataset_file = "examples/basic_dataset.json"

        result = runner.invoke(
            app,
            [
                "run",
                "--pipeline",
                pipeline_file,
                "--dataset",
                dataset_file,
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert "\x1b" not in result.output
        assert "OVERALL" in result.output
        assert "FAILURE ANALYSIS" in result.output
        assert "QUERY TYPES" in result.output
        assert "INSIGHTS" in result.output
        assert "Report written to:" in result.output

        # Verify output JSON file exists and is valid EvaluationReport
        assert output_file.exists()
        raw_json = output_file.read_text(encoding="utf-8")
        parsed_report = EvaluationReport.model_validate_json(raw_json)

        assert parsed_report.dataset_name == "basic_dataset"
        assert parsed_report.pipeline_name == "basic_pipeline"
        assert parsed_report.total_queries == 5
        assert parsed_report.completed_queries == 5
        assert parsed_report.failed_queries == 0
        assert parsed_report.retrieval.mean_recall_at_k == 1.0
        assert parsed_report.diagnosis_counts["PASS"] == 5
        assert "factual" in parsed_report.metrics_by_query_type
