"""Comprehensive tests for Feature 4: Confidence (Evaluation Evidence Reliability)."""

import io

from rich.console import Console
from typer.testing import CliRunner

from ragdiag.cli.main import app
from ragdiag.diagnosis.models import FailureCategory, get_action_for_category
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.result import EvaluationResult
from ragdiag.reporting.aggregator import build_report
from ragdiag.reporting.confidence import (
    _calculate_sample_size_score,
    _derive_confidence_level,
    compute_confidence,
)
from ragdiag.reporting.models import ConfidenceLevel, EvaluationConfidence, EvaluationReport
from ragdiag.reporting.terminal import render_terminal_report

runner = CliRunner()


def _make_eval_result(
    query_id: str,
    category: FailureCategory = FailureCategory.PASS,
    recall: float = 1.0,
    retrieval_ms: float = 20.0,
    grounded: bool | None = None,
    answer_correct: bool | None = None,
    status: str = "completed",
    error: str | None = None,
    judge_error: str | None = None,
) -> EvaluationResult:
    """Helper to construct a deterministic EvaluationResult for testing."""
    metrics: dict[str, object] = {
        "recall_at_5": recall,
        "precision_at_5": 1.0 if recall > 0 else 0.0,
        "reciprocal_rank": 1.0 if recall > 0 else 0.0,
    }
    if grounded is not None:
        metrics["grounded"] = grounded
    if answer_correct is not None:
        metrics["answer_correct"] = answer_correct

    chunks: list[RetrievedChunk] = []
    if recall > 0.0:
        chunks.append(RetrievedChunk(id="c1", text="Relevant context text.", score=0.95))

    return EvaluationResult(
        query_id=query_id,
        query=f"Query {query_id}",
        retrieved_chunks=chunks,
        expected_chunk_ids=["c1"],
        generated_answer="Generated response." if status == "completed" else "",
        expected_answer="Expected ground truth answer.",
        retrieval_duration_ms=retrieval_ms,
        generation_duration_ms=10.0,
        metrics=metrics,
        diagnosis={
            "category": category.value,
            "severity": "info" if category == FailureCategory.PASS else "major",
            "confidence": 0.95,
            "reason": f"Diagnostic reason for {category.value}",
            "action": get_action_for_category(category),
            "evidence": ["Deterministic test evidence"],
        },
        judge_result=None,
        judge_error=judge_error,
        status=status,
        error=error,
    )


class TestConfidenceScoreAndBounding:
    """Tests for confidence scoring, bounding, and determinism."""

    def test_fully_successful_large_evaluation_produces_high_confidence(self) -> None:
        """50 completed queries with judge verification yields High confidence (100.0)."""
        results = [
            _make_eval_result(f"q{i}", grounded=True, answer_correct=True) for i in range(50)
        ]
        report = build_report(results, pipeline_name="LargePipe")
        conf = report.confidence

        assert conf.score == 100.0
        assert conf.level == ConfidenceLevel.HIGH.value
        assert any("All evaluation queries completed successfully" in r for r in conf.reasons)
        assert any("Strong dataset sample size" in r for r in conf.reasons)
        assert any("Semantic judge evaluation completed successfully" in r for r in conf.reasons)

    def test_confidence_score_is_bounded_between_0_and_100(self) -> None:
        """Confidence score never exceeds 100.0 or drops below 0.0."""
        # Extreme negative: zero queries completed, crashes
        r_neg = _make_eval_result("q1", status="failed", error="Fatal error")
        report_neg = build_report([r_neg])
        assert 0.0 <= report_neg.confidence.score <= 100.0

        # Extreme positive
        results_pos = [
            _make_eval_result(f"q{i}", grounded=True, answer_correct=True) for i in range(60)
        ]
        report_pos = build_report(results_pos)
        assert 0.0 <= report_pos.confidence.score <= 100.0

    def test_identical_evaluations_produce_identical_confidence(self) -> None:
        """Re-evaluating the same report yields identical confidence scores and reasons."""
        results = [_make_eval_result(f"q{i}") for i in range(15)]
        report = build_report(results)
        conf1 = compute_confidence(report)
        conf2 = compute_confidence(report)

        assert conf1.model_dump() == conf2.model_dump()

    def test_complete_evaluation_failure_produces_very_low_confidence(self) -> None:
        """When 100% of queries crash, confidence is 0.0 (Very Low)."""
        results = [
            _make_eval_result(f"q{i}", status="failed", error="Pipeline crash") for i in range(10)
        ]
        report = build_report(results)
        conf = report.confidence

        assert conf.score == 0.0
        assert conf.level == ConfidenceLevel.VERY_LOW.value
        assert any("failed" in r.lower() for r in conf.reasons)


class TestDatasetSampleSizeCurve:
    """Tests verifying sample size saturating curve and impact on confidence."""

    def test_small_dataset_lowers_confidence(self) -> None:
        """A small dataset (3 queries) yields lower confidence than a large dataset."""
        small_results = [_make_eval_result(f"q{i}") for i in range(3)]
        large_results = [_make_eval_result(f"q{i}") for i in range(50)]

        small_report = build_report(small_results)
        large_report = build_report(large_results)

        assert small_report.confidence.score < large_report.confidence.score
        assert any("Limited dataset sample size" in r for r in small_report.confidence.reasons)

    def test_medium_dataset_produces_higher_confidence_than_small(self) -> None:
        """Medium dataset (15 queries) yields higher confidence than very small (3 queries)."""
        small_report = build_report([_make_eval_result(f"q{i}") for i in range(3)])
        med_report = build_report([_make_eval_result(f"q{i}") for i in range(15)])

        assert med_report.confidence.score > small_report.confidence.score
        assert med_report.confidence.level in (
            ConfidenceLevel.GOOD.value,
            ConfidenceLevel.HIGH.value,
        )

    def test_large_dataset_provides_strong_sample_size_evidence(self) -> None:
        """50+ queries receives the maximum 1.0 sample size score."""
        assert _calculate_sample_size_score(50) == 1.0
        assert _calculate_sample_size_score(100) == 1.0
        assert _calculate_sample_size_score(49) < 1.0
        assert _calculate_sample_size_score(49) > 0.98  # smooth, no sudden jump


class TestQueryCoverageAndFailures:
    """Tests verifying impact of query completion rates and execution crashes."""

    def test_full_query_coverage_gives_stronger_confidence_than_partial(self) -> None:
        """100% query coverage has higher confidence than partial coverage."""
        full_results = [_make_eval_result(f"q{i}") for i in range(20)]
        partial_results = [
            _make_eval_result(f"q{i}", status="completed" if i < 14 else "failed", error="Timeout")
            for i in range(20)
        ]

        full_rep = build_report(full_results)
        partial_rep = build_report(partial_results)

        assert full_rep.confidence.score > partial_rep.confidence.score

    def test_partial_query_failures_reduce_confidence_and_state_reasons(self) -> None:
        """Failed queries reduce confidence and generate specific failure reasons."""
        results = [
            _make_eval_result(f"q{i}", status="completed" if i < 8 else "failed", error="Crash")
            for i in range(10)
        ]
        report = build_report(results)
        conf = report.confidence

        assert conf.score < 80.0
        assert any("20% of evaluation queries failed" in r for r in conf.reasons)


class TestJudgeEvidenceInteractions:
    """Tests verifying interactions with LLM judge availability and failures."""

    def test_judge_available_and_successful_increases_confidence(self) -> None:
        """Successful semantic judge evaluation increases evidence confidence."""
        results_without_judge = [_make_eval_result(f"q{i}") for i in range(50)]
        results_with_judge = [
            _make_eval_result(f"q{i}", grounded=True, answer_correct=True) for i in range(50)
        ]

        rep_no_judge = build_report(results_without_judge)
        rep_judge = build_report(results_with_judge)

        assert rep_judge.confidence.score > rep_no_judge.confidence.score
        assert rep_judge.confidence.score == 100.0

    def test_judge_not_configured_does_not_cause_severe_penalty(self) -> None:
        """Retrieval-only evaluation remains legitimate with High confidence on large datasets."""
        results = [_make_eval_result(f"q{i}") for i in range(50)]
        report = build_report(results)
        conf = report.confidence

        # High confidence (e.g. 93.0) without judge
        assert conf.score >= 90.0
        assert conf.level == ConfidenceLevel.HIGH.value
        assert any(
            "Semantic judge was not configured; retrieval-only evidence." in r for r in conf.reasons
        )

    def test_judge_configured_with_failures_lowers_confidence(self) -> None:
        """When judge evaluations fail, confidence drops and reasons document failures."""
        results_clean = [
            _make_eval_result(f"q{i}", grounded=True, answer_correct=True) for i in range(20)
        ]
        results_with_judge_fails = [
            _make_eval_result(
                f"q{i}",
                grounded=True if i < 15 else None,
                answer_correct=True if i < 15 else None,
                judge_error="API Rate limit" if i >= 15 else None,
            )
            for i in range(20)
        ]

        rep_clean = build_report(results_clean)
        rep_fails = build_report(results_with_judge_fails)

        assert rep_fails.confidence.score < rep_clean.confidence.score
        assert any("5 semantic judge evaluations failed" in r for r in rep_fails.confidence.reasons)


class TestConfidenceLevelsAndInvariants:
    """Tests verifying confidence level bands, non-contradiction, and independence."""

    def test_confidence_level_boundaries_are_deterministic(self) -> None:
        """Deterministic boundary tests for High, Good, Moderate, Low, Very Low."""
        assert _derive_confidence_level(100.0) == "High"
        assert _derive_confidence_level(90.0) == "High"
        assert _derive_confidence_level(89.9) == "Good"
        assert _derive_confidence_level(75.0) == "Good"
        assert _derive_confidence_level(74.9) == "Moderate"
        assert _derive_confidence_level(60.0) == "Moderate"
        assert _derive_confidence_level(59.9) == "Low"
        assert _derive_confidence_level(40.0) == "Low"
        assert _derive_confidence_level(39.9) == "Very Low"
        assert _derive_confidence_level(0.0) == "Very Low"

    def test_reasons_are_never_contradictory(self) -> None:
        """Evaluation without judge never claims judge evaluation succeeded or failed."""
        results = [_make_eval_result(f"q{i}") for i in range(10)]
        report = build_report(results)
        conf = report.confidence

        reason_text = " ".join(conf.reasons)
        assert "completed successfully across all evaluated queries" not in reason_text
        assert "judge evaluations failed" not in reason_text
        assert "retrieval-only evidence" in reason_text

    def test_health_profile_and_confidence_remain_separate(self) -> None:
        """A poor pipeline can have High confidence, proving Health and Confidence are separate."""
        # Bad RAG pipeline: 0% recall, all WRONG_CHUNK_RETRIEVED, but 50 queries evaluated cleanly
        bad_results = [
            _make_eval_result(
                f"q{i}",
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                recall=0.0,
                grounded=False,
                answer_correct=False,
            )
            for i in range(50)
        ]
        report = build_report(bad_results)

        # Health is Critical because RAG pipeline is completely broken
        assert report.health_profile.score < 40.0
        assert report.health_profile.grade == "Critical"

        # Confidence is High because evaluation evidence is complete and reliable!
        assert report.confidence.score >= 90.0
        assert report.confidence.level == "High"


class TestReportIntegrationAndSerialization:
    """Tests for EvaluationReport integration, serialization, and CLI terminal output."""

    def test_confidence_appears_in_evaluation_report(self) -> None:
        """EvaluationReport automatically computes and includes confidence."""
        results = [_make_eval_result("q1")]
        report = build_report(results)

        assert hasattr(report, "confidence")
        assert isinstance(report.confidence, EvaluationConfidence)

    def test_json_serialization_roundtrip(self) -> None:
        """EvaluationReport with EvaluationConfidence cleanly serializes and deserializes."""
        results = [_make_eval_result(f"q{i}") for i in range(10)]
        report = build_report(results, dataset_name="ds_conf", pipeline_name="PipeConf")

        json_str = report.model_dump_json(indent=2)
        deserialized = EvaluationReport.model_validate_json(json_str)

        assert deserialized.confidence.score == report.confidence.score
        assert deserialized.confidence.level == report.confidence.level
        assert deserialized.confidence.reasons == report.confidence.reasons

    def test_legacy_evaluation_report_json_deserialization(self) -> None:
        """Legacy JSON without 'confidence' deserializes cleanly with defaults."""
        legacy_json = (
            '{"dataset_name": "ds", "dataset_version": "1.0", "total_queries": 1, '
            '"completed_queries": 1, "failed_queries": 0}'
        )
        deserialized = EvaluationReport.model_validate_json(legacy_json)
        assert deserialized.confidence is not None
        assert deserialized.confidence.score == 100.0
        assert deserialized.confidence.level == "High"

    def test_terminal_report_displays_evaluation_confidence(self) -> None:
        """Terminal rendering outputs EVALUATION CONFIDENCE section."""
        results = [_make_eval_result("q1")]
        report = build_report(results, pipeline_name="MyPipe")

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, color_system=None, width=120)
        render_terminal_report(report, console)
        output = buf.getvalue()

        assert "HEALTH PROFILE" in output
        assert "EVALUATION CONFIDENCE" in output
        assert "Score:" in output
        assert "Level:" in output
        assert "Reasons:" in output

    def test_cli_run_command_includes_evaluation_confidence(self) -> None:
        """CLI command 'ragdiag run' renders EVALUATION CONFIDENCE in stdout."""
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
        assert "EVALUATION CONFIDENCE" in result.stdout
        assert "Score:" in result.stdout
        assert "Level:" in result.stdout
