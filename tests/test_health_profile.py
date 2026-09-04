"""Comprehensive tests for Feature 3: Health Profile."""

import io

from rich.console import Console
from typer.testing import CliRunner

from ragdiag.cli.main import app
from ragdiag.diagnosis.models import FailureCategory, get_action_for_category
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.result import EvaluationResult
from ragdiag.reporting.aggregator import build_report
from ragdiag.reporting.health import _derive_grade_and_status, compute_health_profile
from ragdiag.reporting.models import EvaluationReport, HealthGrade, HealthProfile, HealthStatus
from ragdiag.reporting.terminal import render_terminal_report

runner = CliRunner()


def _make_eval_result(
    query_id: str,
    category: FailureCategory,
    recall: float = 1.0,
    precision: float = 1.0,
    mrr: float = 1.0,
    retrieval_ms: float = 20.0,
    grounded: bool | None = None,
    answer_correct: bool | None = None,
    status: str = "completed",
    error: str | None = None,
) -> EvaluationResult:
    """Helper to construct a deterministic EvaluationResult for testing."""
    metrics: dict[str, object] = {
        "recall_at_5": recall,
        "precision_at_5": precision,
        "reciprocal_rank": mrr,
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
            "severity": (
                "info"
                if category == FailureCategory.PASS
                else (
                    "warning"
                    if category
                    in (
                        FailureCategory.INSUFFICIENT_CONTEXT,
                        FailureCategory.WRONG_CHUNK_RANK,
                        FailureCategory.LATENCY_OUTLIER,
                    )
                    else "major"
                )
            ),
            "confidence": 0.95,
            "reason": f"Diagnostic reason for {category.value}",
            "action": get_action_for_category(category),
            "evidence": ["Deterministic test evidence"],
        },
        judge_result=None,
        judge_error=None,
        status=status,
        error=error,
    )


class TestHealthScoreAndBands:
    """Tests covering health scoring formula, bounding, and grade/status bands."""

    def test_perfect_evaluation_produces_excellent_score(self) -> None:
        """A flawless evaluation with 100% recall, precision, MRR and fast latency gets 100.0."""
        r = _make_eval_result(
            "q1", FailureCategory.PASS, recall=1.0, precision=1.0, mrr=1.0, retrieval_ms=15.0
        )
        report = build_report([r], pipeline_name="FastPipeline")
        hp = report.health_profile

        assert hp.score == 100.0
        assert hp.grade == HealthGrade.EXCELLENT.value
        assert hp.status == HealthStatus.HEALTHY.value
        assert any("High recall" in s for s in hp.strengths)
        assert any("Low retrieval latency" in s for s in hp.strengths)
        assert hp.weaknesses == []
        assert hp.recommendations == [get_action_for_category(FailureCategory.PASS)]

    def test_poor_evaluation_produces_critical_score(self) -> None:
        """A failing evaluation with zero retrieval, high latency, and crashes is critical."""
        r1 = _make_eval_result(
            "q1",
            FailureCategory.WRONG_CHUNK_RETRIEVED,
            recall=0.0,
            precision=0.0,
            mrr=0.0,
            retrieval_ms=2500.0,
        )
        r2 = _make_eval_result(
            "q2",
            FailureCategory.UNKNOWN,
            recall=0.0,
            precision=0.0,
            mrr=0.0,
            retrieval_ms=2500.0,
            status="failed",
            error="Connection timeout",
        )
        report = build_report([r1, r2], pipeline_name="BrokenPipeline")
        hp = report.health_profile

        assert hp.score < 40.0
        assert hp.grade == HealthGrade.CRITICAL.value
        assert hp.status == HealthStatus.CRITICAL.value
        assert len(hp.weaknesses) > 0
        assert len(hp.recommendations) > 0

    def test_score_is_always_bounded_between_0_and_100(self) -> None:
        """Health score never goes below 0.0 or above 100.0 even in extreme edge cases."""
        # Extreme negative: total failure + crashes + latency outliers
        r_neg = _make_eval_result(
            "q_bad",
            FailureCategory.UNKNOWN,
            recall=0.0,
            precision=0.0,
            mrr=0.0,
            retrieval_ms=9999.0,
            status="failed",
        )
        report_neg = build_report([r_neg])
        assert 0.0 <= report_neg.health_profile.score <= 100.0

        # Extreme positive
        r_pos = _make_eval_result(
            "q_good",
            FailureCategory.PASS,
            recall=1.0,
            precision=1.0,
            mrr=1.0,
            retrieval_ms=1.0,
            grounded=True,
            answer_correct=True,
        )
        report_pos = build_report([r_pos])
        assert 0.0 <= report_pos.health_profile.score <= 100.0

    def test_grade_boundaries_are_deterministic(self) -> None:
        """Verify boundary values for Excellent, Good, Fair, Poor, and Critical."""
        assert _derive_grade_and_status(100.0) == ("Excellent", "Healthy")
        assert _derive_grade_and_status(90.0) == ("Excellent", "Healthy")
        assert _derive_grade_and_status(89.9) == ("Good", "Healthy")
        assert _derive_grade_and_status(75.0) == ("Good", "Healthy")
        assert _derive_grade_and_status(74.9) == ("Fair", "Degraded")
        assert _derive_grade_and_status(60.0) == ("Fair", "Degraded")
        assert _derive_grade_and_status(59.9) == ("Poor", "Unhealthy")
        assert _derive_grade_and_status(40.0) == ("Poor", "Unhealthy")
        assert _derive_grade_and_status(39.9) == ("Critical", "Critical")
        assert _derive_grade_and_status(0.0) == ("Critical", "Critical")

    def test_unchanged_evaluation_produces_deterministic_output(self) -> None:
        """Running compute_health_profile twice on the same report yields identical results."""
        r = _make_eval_result("q1", FailureCategory.PASS, recall=0.8, precision=0.8, mrr=0.8)
        report = build_report([r])
        hp1 = compute_health_profile(report)
        hp2 = compute_health_profile(report)

        assert hp1.model_dump() == hp2.model_dump()


class TestSemanticAndJudgeInteractions:
    """Tests verifying behavior with and without LLM judge semantic metrics."""

    def test_missing_judge_metrics_does_not_crash(self) -> None:
        """When semantic evaluation was not performed, health profile calculates cleanly."""
        r = _make_eval_result(
            "q1", FailureCategory.PASS, recall=0.9, precision=0.8, mrr=0.9, retrieval_ms=50.0
        )
        report = build_report([r])
        assert report.semantic is None
        hp = report.health_profile

        assert hp.score > 80.0
        assert hp.grade in ("Excellent", "Good")
        # Ensure semantic strengths are NOT mentioned
        assert not any("correctness" in s.lower() for s in hp.strengths)
        assert not any("grounded" in s.lower() for s in hp.strengths)

    def test_judge_metrics_improve_score_when_high(self) -> None:
        """Semantic correctness and groundedness contribute positively to the health score."""
        r_semantic = _make_eval_result(
            "q1",
            FailureCategory.PASS,
            recall=0.7,
            precision=0.6,
            mrr=0.7,
            retrieval_ms=120.0,
            grounded=True,
            answer_correct=True,
        )
        report = build_report([r_semantic])
        hp = report.health_profile

        assert any("High answer correctness" in s for s in hp.strengths)
        assert any("Strong context groundedness" in s for s in hp.strengths)

    def test_unsupported_semantic_strengths_are_not_reported(self) -> None:
        """Without judge evaluation, semantic strengths are never asserted."""
        r = _make_eval_result("q1", FailureCategory.PASS, recall=1.0)
        report = build_report([r])
        hp = report.health_profile

        for s in hp.strengths:
            assert "answer correctness" not in s.lower()
            assert "context groundedness" not in s.lower()


class TestWeaknessesAndStrengths:
    """Tests verifying accurate extraction of weaknesses and strengths."""

    def test_low_recall_contributes_to_weakness(self) -> None:
        """Low recall (< 0.70) is flagged under weaknesses."""
        r = _make_eval_result("q1", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.4)
        report = build_report([r])
        hp = report.health_profile

        assert any("Low recall" in w for w in hp.weaknesses)

    def test_low_precision_contributes_to_weakness(self) -> None:
        """Low precision (< 0.50) is flagged under weaknesses."""
        r = _make_eval_result("q1", FailureCategory.PASS, recall=1.0, precision=0.2)
        report = build_report([r])
        hp = report.health_profile

        assert any("Poor precision" in w for w in hp.weaknesses)

    def test_poor_ranking_contributes_to_weakness(self) -> None:
        """Low MRR (< 0.60) is flagged under weaknesses."""
        r = _make_eval_result("q1", FailureCategory.WRONG_CHUNK_RANK, recall=1.0, mrr=0.25)
        report = build_report([r])
        hp = report.health_profile

        assert any("Suboptimal ranking quality" in w for w in hp.weaknesses)

    def test_high_latency_contributes_to_weakness(self) -> None:
        """Latency > 300ms is flagged under weaknesses."""
        r = _make_eval_result(
            "q1", FailureCategory.LATENCY_OUTLIER, recall=1.0, retrieval_ms=1200.0
        )
        report = build_report([r])
        hp = report.health_profile

        assert any("High retrieval latency" in w for w in hp.weaknesses)

    def test_diagnosis_failures_contribute_to_weaknesses(self) -> None:
        """Specific failure categories appear with counts in weaknesses."""
        r1 = _make_eval_result("q1", FailureCategory.WRONG_CHUNK_RETRIEVED, recall=0.0)
        r2 = _make_eval_result(
            "q2", FailureCategory.ANSWER_INCORRECT, recall=1.0, answer_correct=False
        )
        report = build_report([r1, r2])
        hp = report.health_profile

        assert any("WRONG_CHUNK_RETRIEVED" in w for w in hp.weaknesses)
        assert any("ANSWER_INCORRECT" in w for w in hp.weaknesses)

    def test_strengths_generated_from_actual_available_metrics(self) -> None:
        """Strengths accurately reflect measured high recall and low latency."""
        r = _make_eval_result("q1", FailureCategory.PASS, recall=0.95, retrieval_ms=12.0)
        report = build_report([r])
        hp = report.health_profile

        assert any("High recall" in s for s in hp.strengths)
        assert any("Low retrieval latency" in s for s in hp.strengths)


class TestRecommendations:
    """Tests verifying recommendations reuse Failure -> Action Mapping and deduplicate."""

    def test_recommendations_reuse_failure_action_mapping(self) -> None:
        """Recommendations match the deterministic get_action_for_category mapping."""
        r = _make_eval_result("q1", FailureCategory.WRONG_CHUNK_RETRIEVED, recall=0.0)
        report = build_report([r])
        hp = report.health_profile

        expected_action = get_action_for_category(FailureCategory.WRONG_CHUNK_RETRIEVED)
        assert expected_action in hp.recommendations

    def test_duplicate_recommendations_are_removed(self) -> None:
        """When multiple queries have the same failure category, the recommendation appears once."""
        r1 = _make_eval_result("q1", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.5)
        r2 = _make_eval_result("q2", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.5)
        r3 = _make_eval_result("q3", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.5)
        report = build_report([r1, r2, r3])
        hp = report.health_profile

        expected_action = get_action_for_category(FailureCategory.INSUFFICIENT_CONTEXT)
        assert hp.recommendations.count(expected_action) == 1


class TestReportIntegrationAndSerialization:
    """Tests for EvaluationReport integration, serialization, and CLI terminal output."""

    def test_health_profile_appears_in_evaluation_report(self) -> None:
        """EvaluationReport automatically includes a computed health_profile."""
        r = _make_eval_result("q1", FailureCategory.PASS)
        report = build_report([r])

        assert hasattr(report, "health_profile")
        assert isinstance(report.health_profile, HealthProfile)
        assert report.health_profile.score == 100.0

    def test_json_serialization_roundtrip(self) -> None:
        """EvaluationReport with HealthProfile cleanly serializes and deserializes."""
        r = _make_eval_result("q1", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.5)
        report = build_report([r], dataset_name="ds_test", pipeline_name="TestPipe")

        json_str = report.model_dump_json(indent=2)
        deserialized = EvaluationReport.model_validate_json(json_str)

        assert deserialized.health_profile.score == report.health_profile.score
        assert deserialized.health_profile.grade == report.health_profile.grade
        assert deserialized.health_profile.status == report.health_profile.status
        assert deserialized.health_profile.weaknesses == report.health_profile.weaknesses
        assert deserialized.health_profile.recommendations == report.health_profile.recommendations

    def test_legacy_evaluation_report_json_deserialization(self) -> None:
        """Legacy EvaluationReport JSON without health_profile deserializes with defaults."""
        legacy_json = (
            '{"dataset_name": "ds", "dataset_version": "1.0", "total_queries": 1, '
            '"completed_queries": 1, "failed_queries": 0}'
        )
        deserialized = EvaluationReport.model_validate_json(legacy_json)
        assert deserialized.health_profile is not None
        assert deserialized.health_profile.score == 100.0
        assert deserialized.health_profile.grade == "Excellent"

    def test_terminal_report_displays_health_profile(self) -> None:
        """Terminal rendering outputs HEALTH PROFILE with score, grade, and bullets."""
        r = _make_eval_result("q1", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.5)
        report = build_report([r], pipeline_name="MyPipe")

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, color_system=None, width=120)
        render_terminal_report(report, console)
        output = buf.getvalue()

        assert "HEALTH PROFILE" in output
        assert "Score:" in output
        assert "Grade:" in output
        assert "Weaknesses:" in output
        assert "Recommendations:" in output

    def test_cli_run_command_includes_health_profile(self) -> None:
        """Invoking 'ragdiag run' outputs the HEALTH PROFILE section in terminal."""
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
        assert "HEALTH PROFILE" in result.stdout
        assert "Score:" in result.stdout
        assert "Grade:" in result.stdout
