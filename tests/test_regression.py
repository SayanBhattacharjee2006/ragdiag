"""Comprehensive tests for Feature 2: Regression Analysis."""

import io

from rich.console import Console
from typer.testing import CliRunner

from ragdiag.cli.main import app
from ragdiag.comparison.comparator import compare_reports
from ragdiag.comparison.models import ComparisonReport
from ragdiag.comparison.terminal import render_comparison_terminal
from ragdiag.diagnosis.models import DiagnosisResult, FailureCategory
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.result import EvaluationResult
from ragdiag.reporting.aggregator import build_report

runner = CliRunner()


def _make_eval_result(
    query_id: str,
    category: FailureCategory,
    recall: float = 1.0,
    precision: float = 1.0,
    mrr: float = 1.0,
    retrieval_ms: float = 10.0,
    grounded: bool | None = None,
    answer_correct: bool | None = None,
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

    chunk = RetrievedChunk(id="c1", text="text")
    return EvaluationResult(
        query_id=query_id,
        query=f"Query {query_id}",
        status="completed",
        expected_chunk_ids=["c1"],
        retrieved_chunks=[chunk] if recall > 0 else [RetrievedChunk(id="c99", text="wrong")],
        metrics=metrics,
        latency={"retrieval_ms": retrieval_ms},
        diagnosis=DiagnosisResult(
            category=category,
            severity="info" if category == FailureCategory.PASS else "major",
            confidence=1.0,
            reason=f"Diagnosed as {category.value}",
        ),
    )


class TestMetricRegressions:
    """Tests verifying overall metric regression detection against tolerances."""

    def test_no_regression_when_identical(self) -> None:
        """Identical pipeline metrics produce no regressions and overall_regression=False."""
        r_a = _make_eval_result("q1", FailureCategory.PASS, recall=1.0, retrieval_ms=10.0)
        r_b = _make_eval_result("q1", FailureCategory.PASS, recall=1.0, retrieval_ms=10.0)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        ra = comp.regression_analysis

        assert not ra.overall_regression
        assert len(ra.metric_regressions) == 0
        assert ra.regressed_query_count == 0
        assert "No meaningful regressions detected" in ra.summary

    def test_recall_decrease_beyond_tolerance_detected(self) -> None:
        """Recall drop greater than quality_tolerance (0.02) is detected as a regression."""
        r_a = _make_eval_result("q1", FailureCategory.PASS, recall=0.80)
        r_b = _make_eval_result("q1", FailureCategory.WRONG_CHUNK_RANK, recall=0.69)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b], quality_tolerance=0.02)
        ra = comp.regression_analysis

        assert ra.overall_regression
        recall_reg = next((m for m in ra.metric_regressions if m.metric_name == "Recall@5"), None)
        assert recall_reg is not None
        assert recall_reg.delta == -0.11
        assert recall_reg.threshold == 0.02
        assert recall_reg.baseline_value == 0.80
        assert recall_reg.current_value == 0.69

    def test_recall_decrease_within_tolerance_not_detected(self) -> None:
        """Recall drop within quality_tolerance (0.02) is not reported as a regression."""
        r_a = _make_eval_result("q1", FailureCategory.PASS, recall=0.80)
        r_b = _make_eval_result("q1", FailureCategory.PASS, recall=0.79)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b], quality_tolerance=0.02)
        ra = comp.regression_analysis

        assert not ra.overall_regression
        assert not any(m.metric_name == "Recall@5" for m in ra.metric_regressions)

    def test_precision_and_mrr_regressions_detected(self) -> None:
        """Precision and MRR drops exceeding tolerance are captured in metric_regressions."""
        r_a = _make_eval_result("q1", FailureCategory.PASS, precision=0.80, mrr=1.0)
        r_b = _make_eval_result("q1", FailureCategory.PASS, precision=0.70, mrr=0.50)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b], quality_tolerance=0.02)
        ra = comp.regression_analysis

        m_names = [m.metric_name for m in ra.metric_regressions]
        assert "Precision@5" in m_names
        assert "MRR" in m_names

    def test_latency_increase_beyond_tolerance_detected(self) -> None:
        """Mean retrieval latency increase > 10ms is detected as performance regression."""
        r_a = _make_eval_result("q1", FailureCategory.PASS, retrieval_ms=10.0)
        r_b = _make_eval_result("q1", FailureCategory.PASS, retrieval_ms=35.0)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b], latency_tolerance_ms=10.0)
        ra = comp.regression_analysis

        assert ra.overall_regression
        lat_reg = next(
            (m for m in ra.metric_regressions if m.metric_name == "Mean Retrieval Latency"), None
        )
        assert lat_reg is not None
        assert lat_reg.delta == 25.0
        assert lat_reg.unit == "ms"
        assert "Performance regression detected" in ra.summary

    def test_latency_increase_within_tolerance_not_detected(self) -> None:
        """Mean retrieval latency increase <= 10ms is not reported as regression."""
        r_a = _make_eval_result("q1", FailureCategory.PASS, retrieval_ms=10.0)
        r_b = _make_eval_result("q1", FailureCategory.PASS, retrieval_ms=16.0)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b], latency_tolerance_ms=10.0)
        ra = comp.regression_analysis

        assert not ra.overall_regression
        assert not any(m.metric_name == "Mean Retrieval Latency" for m in ra.metric_regressions)

    def test_semantic_quality_regressions(self) -> None:
        """Answer correctness and groundedness rate drops beyond tolerance are captured."""
        r_a = _make_eval_result("q1", FailureCategory.PASS, grounded=True, answer_correct=True)
        r_b = _make_eval_result(
            "q1", FailureCategory.RETRIEVED_BUT_NOT_GROUNDED, grounded=False, answer_correct=False
        )
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b], quality_tolerance=0.02)
        ra = comp.regression_analysis

        m_names = [m.metric_name for m in ra.metric_regressions]
        assert "Answer Correctness" in m_names
        assert "Groundedness" in m_names

    def test_no_judge_missing_semantics_handled_gracefully(self) -> None:
        """When semantic evaluation is not configured, semantic metrics are cleanly omitted."""
        r_a = _make_eval_result("q1", FailureCategory.PASS)
        r_b = _make_eval_result("q1", FailureCategory.PASS)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        assert rep_a.semantic is None
        assert rep_b.semantic is None

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        ra = comp.regression_analysis
        assert not any(
            m.metric_name in ("Answer Correctness", "Groundedness") for m in ra.metric_regressions
        )


class TestQueryLevelRegressions:
    """Tests for query-level regression detection and categorization."""

    def test_pass_to_failure_detected_as_regression(self) -> None:
        """A query transitioning from PASS to INSUFFICIENT_CONTEXT is flagged as regressed."""
        r_a = _make_eval_result("q_pass", FailureCategory.PASS, recall=1.0)
        r_b = _make_eval_result("q_pass", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.5)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        ra = comp.regression_analysis

        assert ra.regressed_query_count == 1
        rq = ra.regressed_queries[0]
        assert rq.query_id == "q_pass"
        assert rq.baseline_diagnosis == "PASS"
        assert rq.current_diagnosis == "INSUFFICIENT_CONTEXT"
        assert rq.transition == "PASS -> INSUFFICIENT_CONTEXT"
        assert any(
            "q_pass: PASS -> INSUFFICIENT_CONTEXT" in imp for imp in ra.important_regressions
        )

    def test_failure_to_pass_is_improvement_not_regression(self) -> None:
        """A query transitioning from WRONG_CHUNK_RETRIEVED to PASS is not a regression."""
        r_a = _make_eval_result("q1", FailureCategory.WRONG_CHUNK_RETRIEVED, recall=0.0)
        r_b = _make_eval_result("q1", FailureCategory.PASS, recall=1.0)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        ra = comp.regression_analysis

        assert ra.regressed_query_count == 0
        assert len(ra.regressed_queries) == 0

    def test_pass_to_major_failure_ranked_highest_importance(self) -> None:
        """Transitions from PASS to Major failures are ranked at top of important_regressions."""
        r_a1 = _make_eval_result("q_major", FailureCategory.PASS, recall=1.0)
        r_b1 = _make_eval_result(
            "q_major", FailureCategory.ANSWER_INCORRECT, recall=1.0, answer_correct=False
        )

        r_a2 = _make_eval_result("q_warn", FailureCategory.PASS, recall=1.0)
        r_b2 = _make_eval_result("q_warn", FailureCategory.WRONG_CHUNK_RANK, recall=1.0)

        rep_a = build_report([r_a1, r_a2], pipeline_name="PipeA")
        rep_b = build_report([r_b1, r_b2], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a1, r_a2], [r_b1, r_b2])
        ra = comp.regression_analysis

        assert len(ra.important_regressions) == 2
        # PASS -> ANSWER_INCORRECT (major) comes before PASS -> WRONG_CHUNK_RANK (warning)
        assert "q_major: PASS -> ANSWER_INCORRECT" == ra.important_regressions[0]
        assert "q_warn: PASS -> WRONG_CHUNK_RANK" == ra.important_regressions[1]

    def test_same_category_recall_drop_regression(self) -> None:
        """When category is the same but recall drops significantly, query is marked regressed."""
        r_a = _make_eval_result("q1", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.80)
        r_b = _make_eval_result("q1", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.40)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        ra = comp.regression_analysis

        assert ra.regressed_query_count == 1
        rq = ra.regressed_queries[0]
        assert "dropped from 0.80 to 0.40" in rq.reason


class TestDiagnosisTransitionsAndCategoryIncreases:
    """Tests verifying diagnosis category transition tracking and failure count increases."""

    def test_diagnosis_transitions_grouping_and_counts(self) -> None:
        """Regressed query transitions are grouped and counted accurately."""
        r_a1 = _make_eval_result("q1", FailureCategory.PASS, recall=1.0)
        r_b1 = _make_eval_result("q1", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.5)

        r_a2 = _make_eval_result("q2", FailureCategory.PASS, recall=1.0)
        r_b2 = _make_eval_result("q2", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.5)

        r_a3 = _make_eval_result("q3", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.5)
        r_b3 = _make_eval_result("q3", FailureCategory.WRONG_CHUNK_RETRIEVED, recall=0.0)

        rep_a = build_report([r_a1, r_a2, r_a3], pipeline_name="PipeA")
        rep_b = build_report([r_b1, r_b2, r_b3], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a1, r_a2, r_a3], [r_b1, r_b2, r_b3])
        ra = comp.regression_analysis

        assert len(ra.diagnosis_regressions) == 2
        # PASS -> INSUFFICIENT_CONTEXT had 2 queries
        pass_to_insuff = next(
            (
                dt
                for dt in ra.diagnosis_regressions
                if dt.transition == "PASS -> INSUFFICIENT_CONTEXT"
            ),
            None,
        )
        assert pass_to_insuff is not None
        assert pass_to_insuff.count == 2
        assert pass_to_insuff.query_ids == ["q1", "q2"]

        # INSUFFICIENT_CONTEXT -> WRONG_CHUNK_RETRIEVED had 1 query
        insuff_to_miss = next(
            (
                dt
                for dt in ra.diagnosis_regressions
                if dt.transition == "INSUFFICIENT_CONTEXT -> WRONG_CHUNK_RETRIEVED"
            ),
            None,
        )
        assert insuff_to_miss is not None
        assert insuff_to_miss.count == 1
        assert insuff_to_miss.query_ids == ["q3"]

    def test_increased_failures_tracking(self) -> None:
        """Categories that suffered an increase in failures are recorded in increased_failures."""
        r_a1 = _make_eval_result("q1", FailureCategory.PASS)
        r_b1 = _make_eval_result("q1", FailureCategory.WRONG_CHUNK_RETRIEVED)
        rep_a = build_report([r_a1], pipeline_name="PipeA")
        rep_b = build_report([r_b1], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a1], [r_b1])
        ra = comp.regression_analysis

        assert "WRONG_CHUNK_RETRIEVED" in ra.increased_failures
        assert ra.increased_failures["WRONG_CHUNK_RETRIEVED"] == 1
        assert "PASS" not in ra.increased_failures


class TestMixedImprovementsAndTradeoffs:
    """Tests verifying behavior when some metrics improve while others regress."""

    def test_quality_improvement_with_latency_tradeoff_is_not_overall_regression(self) -> None:
        """Candidate pipeline quality improvement with latency trade-off is not regression."""
        r_a = _make_eval_result(
            "q1", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.5, retrieval_ms=10.0
        )
        r_b = _make_eval_result("q1", FailureCategory.PASS, recall=1.0, retrieval_ms=35.0)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        ra = comp.regression_analysis

        # Overall regression is False because candidate pipeline improved quality
        assert not ra.overall_regression
        # But latency regression is still tracked individually in metric_regressions
        assert any(mr.metric_name == "Mean Retrieval Latency" for mr in ra.metric_regressions)
        assert "improved overall quality despite increased retrieval latency" in ra.summary

    def test_multiple_simultaneous_regressions(self) -> None:
        """Multiple metrics dropping simultaneously are all captured in regression_analysis."""
        r_a = _make_eval_result(
            "q1", FailureCategory.PASS, recall=1.0, precision=1.0, mrr=1.0, retrieval_ms=10.0
        )
        r_b = _make_eval_result(
            "q1",
            FailureCategory.WRONG_CHUNK_RETRIEVED,
            recall=0.0,
            precision=0.0,
            mrr=0.0,
            retrieval_ms=40.0,
        )
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        ra = comp.regression_analysis

        assert ra.overall_regression
        m_names = {m.metric_name for m in ra.metric_regressions}
        assert "Recall@5" in m_names
        assert "Precision@5" in m_names
        assert "MRR" in m_names
        assert "Mean Retrieval Latency" in m_names


class TestSerializationAndTerminalOutput:
    """Tests for Pydantic serialization and terminal rendering."""

    def test_regression_analysis_json_serialization_roundtrip(self) -> None:
        """ComparisonReport with RegressionAnalysis serializes and deserializes cleanly."""
        r_a = _make_eval_result("q1", FailureCategory.PASS, recall=1.0)
        r_b = _make_eval_result("q1", FailureCategory.WRONG_CHUNK_RETRIEVED, recall=0.0)
        rep_a = build_report([r_a], dataset_name="ds", pipeline_name="PipeA")
        rep_b = build_report([r_b], dataset_name="ds", pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])
        json_str = comp.model_dump_json(indent=2)
        deserialized = ComparisonReport.model_validate_json(json_str)

        assert (
            deserialized.regression_analysis.overall_regression
            == comp.regression_analysis.overall_regression
        )
        assert (
            deserialized.regression_analysis.regressed_query_count
            == comp.regression_analysis.regressed_query_count
        )
        assert len(deserialized.regression_analysis.metric_regressions) == len(
            comp.regression_analysis.metric_regressions
        )
        assert len(deserialized.regression_analysis.important_regressions) == len(
            comp.regression_analysis.important_regressions
        )

    def test_legacy_comparison_json_deserialization(self) -> None:
        """Legacy JSON without 'regression_analysis' defaults to empty RegressionAnalysis."""
        legacy_json = (
            '{"dataset_name": "ds", "dataset_version": "1.0", "pipeline_a_name": "pa", '
            '"pipeline_b_name": "pb", "pipeline_a_report": {}, "pipeline_b_report": {}}'
        )
        deserialized = ComparisonReport.model_validate_json(legacy_json)
        assert deserialized.regression_analysis.overall_regression is False

    def test_terminal_report_displays_regression_analysis_yes(self) -> None:
        """Terminal rendering outputs REGRESSION ANALYSIS with YES and metric deltas."""
        r_a = _make_eval_result("q12", FailureCategory.PASS, recall=1.0, mrr=1.0)
        r_b = _make_eval_result("q12", FailureCategory.INSUFFICIENT_CONTEXT, recall=0.5, mrr=0.5)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        render_comparison_terminal(comp, console)
        output = buf.getvalue()

        assert "REGRESSION ANALYSIS" in output
        assert "Overall regression: YES" in output
        assert "Metric regressions:" in output
        assert "Recall@5" in output
        assert "Queries regressed: 1" in output
        assert "Important regressions:" in output
        assert "q12: PASS -> INSUFFICIENT_CONTEXT" in output

    def test_terminal_report_displays_regression_analysis_no(self) -> None:
        """Terminal rendering outputs REGRESSION ANALYSIS with NO when no regressions exist."""
        r_a = _make_eval_result("q1", FailureCategory.PASS)
        r_b = _make_eval_result("q1", FailureCategory.PASS)
        rep_a = build_report([r_a], pipeline_name="PipeA")
        rep_b = build_report([r_b], pipeline_name="PipeB")

        comp = compare_reports(rep_a, rep_b, [r_a], [r_b])

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        render_comparison_terminal(comp, console)
        output = buf.getvalue()

        assert "REGRESSION ANALYSIS" in output
        assert "Overall regression: NO" in output
        assert "No meaningful regressions detected" in output

    def test_cli_compare_command_includes_regression_analysis(self) -> None:
        """ragdiag compare CLI invocation includes REGRESSION ANALYSIS section in output."""
        result = runner.invoke(
            app,
            [
                "compare",
                "--pipeline-a",
                "examples/dense_pipeline.py",
                "--pipeline-b",
                "examples/hybrid_pipeline.py",
                "--dataset",
                "examples/demo_dataset.json",
            ],
        )
        assert result.exit_code == 0
        assert "REGRESSION ANALYSIS" in result.output
        assert "Overall regression:" in result.output
