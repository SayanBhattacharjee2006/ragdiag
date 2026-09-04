"""Comprehensive tests for Feature 1: Failure -> Action Mapping."""

import io

from rich.console import Console

from ragdiag.diagnosis.classifier import DiagnosisEngine
from ragdiag.diagnosis.models import (
    FAILURE_ACTIONS,
    DiagnosisResult,
    FailureCategory,
    get_action_for_category,
)
from ragdiag.diagnosis.rules import (
    classify_answer_failure,
    classify_context_sufficiency,
    classify_grounding_failure,
    classify_latency_outlier,
    classify_pipeline_failure,
    classify_ranking_failure,
    classify_retrieval_failure,
)
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.result import EvaluationResult
from ragdiag.reporting.aggregator import build_report
from ragdiag.reporting.models import EvaluationReport, TopFailure
from ragdiag.reporting.terminal import render_terminal_report

EXPECTED_MAPPINGS: dict[FailureCategory, str] = {
    FailureCategory.PASS: "No action required.",
    FailureCategory.WRONG_CHUNK_RETRIEVED: (
        "Review the retrieval strategy and query formulation; "
        "the pipeline retrieved irrelevant context."
    ),
    FailureCategory.WRONG_CHUNK_RANK: (
        "Improve ranking or reranking so relevant context appears earlier."
    ),
    FailureCategory.INSUFFICIENT_CONTEXT: (
        "Increase retrieval depth or improve retrieval coverage "
        "so all required context is retrieved."
    ),
    FailureCategory.RETRIEVED_BUT_NOT_GROUNDED: (
        "Improve answer grounding so the generated response "
        "stays supported by the retrieved context."
    ),
    FailureCategory.ANSWER_INCORRECT: (
        "Review the generation prompt, model behavior, and context usage for answer correctness."
    ),
    FailureCategory.LATENCY_OUTLIER: (
        "Investigate slow retrieval or generation paths and optimize the latency bottleneck."
    ),
    FailureCategory.UNKNOWN: ("Inspect the pipeline execution error and underlying integration."),
}


class TestFailureActionMapping:
    """Tests verifying the deterministic mapping for all failure categories."""

    def test_all_categories_in_failure_actions(self) -> None:
        """Every FailureCategory enum member is represented in FAILURE_ACTIONS."""
        for cat in FailureCategory:
            assert cat in FAILURE_ACTIONS
            assert isinstance(FAILURE_ACTIONS[cat], str)
            assert len(FAILURE_ACTIONS[cat]) > 0

    def test_failure_actions_count(self) -> None:
        """Exactly 8 failure categories are mapped."""
        assert len(FAILURE_ACTIONS) == 8

    def test_exact_mapping_strings(self) -> None:
        """Every failure category produces the exact expected recommendation string."""
        for cat, expected_str in EXPECTED_MAPPINGS.items():
            assert FAILURE_ACTIONS[cat] == expected_str

    def test_pass_explicit_recommendation(self) -> None:
        """PASS category has the explicit recommendation 'No action required.'."""
        assert FAILURE_ACTIONS[FailureCategory.PASS] == "No action required."
        assert FailureCategory.PASS.action == "No action required."
        assert get_action_for_category(FailureCategory.PASS) == "No action required."

    def test_category_action_property(self) -> None:
        """FailureCategory.action property returns the correct recommendation."""
        for cat in FailureCategory:
            assert cat.action == EXPECTED_MAPPINGS[cat]

    def test_get_action_for_category_with_enum(self) -> None:
        """get_action_for_category accepts FailureCategory enum instances."""
        for cat in FailureCategory:
            assert get_action_for_category(cat) == EXPECTED_MAPPINGS[cat]

    def test_get_action_for_category_with_string(self) -> None:
        """get_action_for_category accepts string values matching category names."""
        for cat in FailureCategory:
            assert get_action_for_category(cat.value) == EXPECTED_MAPPINGS[cat]

    def test_get_action_for_category_unknown_string_fallback(self) -> None:
        """get_action_for_category returns UNKNOWN action for invalid/unrecognized strings."""
        unknown_action = EXPECTED_MAPPINGS[FailureCategory.UNKNOWN]
        assert get_action_for_category("COMPLETELY_INVALID_CATEGORY") == unknown_action
        assert get_action_for_category("") == unknown_action
        assert get_action_for_category(None) == unknown_action  # type: ignore[arg-type]

    def test_mapping_is_deterministic(self) -> None:
        """Multiple repeated lookups return identical recommendations."""
        for cat in FailureCategory:
            res1 = get_action_for_category(cat)
            res2 = get_action_for_category(cat)
            assert res1 == res2
            assert res1 is FAILURE_ACTIONS[cat]


class TestDiagnosisResultActionModel:
    """Tests for DiagnosisResult action field handling, defaults, and serialization."""

    def test_diagnosis_result_default_action_for_all_categories(self) -> None:
        """DiagnosisResult automatically populates action from category when not provided."""
        for cat in FailureCategory:
            diag = DiagnosisResult(
                category=cat,
                severity="info" if cat == FailureCategory.PASS else "major",
                confidence=1.0,
                reason="Testing action default",
            )
            assert diag.action == EXPECTED_MAPPINGS[cat]

    def test_diagnosis_result_explicit_action_preserved(self) -> None:
        """Explicitly supplied action overrides the default recommendation."""
        custom = "Custom action override for specific deployment."
        diag = DiagnosisResult(
            category=FailureCategory.WRONG_CHUNK_RETRIEVED,
            severity="major",
            confidence=1.0,
            reason="Testing custom action",
            action=custom,
        )
        assert diag.action == custom

    def test_diagnosis_result_serialization_roundtrip(self) -> None:
        """DiagnosisResult serializes action to JSON and restores it on deserialization."""
        diag = DiagnosisResult(
            category=FailureCategory.INSUFFICIENT_CONTEXT,
            severity="warning",
            confidence=0.85,
            reason="Partial context retrieved",
        )
        data = diag.model_dump()
        assert "action" in data
        assert data["action"] == EXPECTED_MAPPINGS[FailureCategory.INSUFFICIENT_CONTEXT]

        json_str = diag.model_dump_json()
        assert f'"{EXPECTED_MAPPINGS[FailureCategory.INSUFFICIENT_CONTEXT]}"' in json_str

        restored = DiagnosisResult.model_validate_json(json_str)
        assert restored.action == diag.action
        assert restored.category == diag.category

    def test_diagnosis_result_legacy_json_backward_compatibility(self) -> None:
        """Deserializing legacy JSON without 'action' field automatically sets default action."""
        legacy_json = (
            '{"category": "ANSWER_INCORRECT", "severity": "major", '
            '"confidence": 0.95, "reason": "Incorrect answer", "evidence": []}'
        )
        diag = DiagnosisResult.model_validate_json(legacy_json)
        assert diag.action == EXPECTED_MAPPINGS[FailureCategory.ANSWER_INCORRECT]


class TestTopFailureActionModel:
    """Tests for TopFailure action field handling, defaults, and serialization."""

    def test_top_failure_default_action_for_all_categories(self) -> None:
        """TopFailure automatically populates action from category when not provided."""
        for cat in FailureCategory:
            tf = TopFailure(
                query_id="q1",
                query="Sample query",
                category=cat,
                severity="major",
                confidence=1.0,
                reason="Testing top failure action default",
            )
            assert tf.action == EXPECTED_MAPPINGS[cat]

    def test_top_failure_explicit_action_preserved(self) -> None:
        """Explicitly supplied action on TopFailure is preserved."""
        custom = "Custom top failure action."
        cat_enum = FailureCategory.WRONG_CHUNK_RANK
        tf = TopFailure(
            query_id="q1",
            query="Sample query",
            category=cat_enum,
            severity="warning",
            confidence=1.0,
            reason="Rank issue",
            action=custom,
        )
        assert tf.action == custom
        assert tf.category == cat_enum

    def test_top_failure_serialization_roundtrip(self) -> None:
        """TopFailure serializes and deserializes action cleanly."""
        tf = TopFailure(
            query_id="q_fail",
            query="What is 3DS?",
            category=FailureCategory.RETRIEVED_BUT_NOT_GROUNDED,
            severity="major",
            confidence=1.0,
            reason="Hallucination detected",
            evidence=["Claim not in context"],
        )
        dumped = tf.model_dump()
        assert "action" in dumped
        assert dumped["action"] == EXPECTED_MAPPINGS[FailureCategory.RETRIEVED_BUT_NOT_GROUNDED]

        restored = TopFailure.model_validate_json(tf.model_dump_json())
        assert restored.action == tf.action

    def test_top_failure_legacy_json_backward_compatibility(self) -> None:
        """Deserializing legacy TopFailure JSON without 'action' sets default action."""
        legacy_json = (
            '{"query_id": "q1", "query": "q", "category": "LATENCY_OUTLIER", '
            '"severity": "warning", "confidence": 1.0, "reason": "Slow", "evidence": []}'
        )
        tf = TopFailure.model_validate_json(legacy_json)
        assert tf.action == EXPECTED_MAPPINGS[FailureCategory.LATENCY_OUTLIER]


class TestRulesAndEngineActionIntegration:
    """Tests verifying rule functions and DiagnosisEngine return DiagnosisResults with action."""

    def test_classify_pipeline_failure_action(self) -> None:
        diag = classify_pipeline_failure("failed", "DB error")
        assert diag is not None
        assert diag.category == FailureCategory.UNKNOWN
        assert diag.action == EXPECTED_MAPPINGS[FailureCategory.UNKNOWN]

    def test_classify_retrieval_failure_action(self) -> None:
        diag = classify_retrieval_failure(["doc1"], [RetrievedChunk(id="doc2", text="t")])
        assert diag is not None
        assert diag.category == FailureCategory.WRONG_CHUNK_RETRIEVED
        assert diag.action == EXPECTED_MAPPINGS[FailureCategory.WRONG_CHUNK_RETRIEVED]

    def test_classify_ranking_failure_action(self) -> None:
        chunks = [
            RetrievedChunk(id="doc_other1", text="t"),
            RetrievedChunk(id="doc_other2", text="t"),
            RetrievedChunk(id="doc_other3", text="t"),
            RetrievedChunk(id="doc_relevant", text="t"),
        ]
        diag = classify_ranking_failure(["doc_relevant"], chunks, rank_threshold=3)
        assert diag is not None
        assert diag.category == FailureCategory.WRONG_CHUNK_RANK
        assert diag.action == EXPECTED_MAPPINGS[FailureCategory.WRONG_CHUNK_RANK]

    def test_classify_context_sufficiency_action(self) -> None:
        diag = classify_context_sufficiency(["doc1", "doc2"], [RetrievedChunk(id="doc1", text="t")])
        assert diag is not None
        assert diag.category == FailureCategory.INSUFFICIENT_CONTEXT
        assert diag.action == EXPECTED_MAPPINGS[FailureCategory.INSUFFICIENT_CONTEXT]

    def test_classify_grounding_failure_action(self) -> None:
        diag = classify_grounding_failure({"grounded": False, "answer_correct": False})
        assert diag is not None
        assert diag.category == FailureCategory.RETRIEVED_BUT_NOT_GROUNDED
        assert diag.action == EXPECTED_MAPPINGS[FailureCategory.RETRIEVED_BUT_NOT_GROUNDED]

    def test_classify_answer_failure_action(self) -> None:
        diag = classify_answer_failure({"grounded": True, "answer_correct": False})
        assert diag is not None
        assert diag.category == FailureCategory.ANSWER_INCORRECT
        assert diag.action == EXPECTED_MAPPINGS[FailureCategory.ANSWER_INCORRECT]

    def test_classify_latency_outlier_action(self) -> None:
        diag = classify_latency_outlier({"retrieval_ms": 1500.0}, latency_threshold_ms=1000.0)
        assert diag is not None
        assert diag.category == FailureCategory.LATENCY_OUTLIER
        assert diag.action == EXPECTED_MAPPINGS[FailureCategory.LATENCY_OUTLIER]

    def test_diagnosis_engine_pass_action(self) -> None:
        engine = DiagnosisEngine()
        res = EvaluationResult(
            query_id="q_pass",
            query="Pass query",
            status="completed",
            expected_chunk_ids=["doc1"],
            retrieved_chunks=[RetrievedChunk(id="doc1", text="t")],
            metrics={
                "recall_at_5": 1.0,
                "precision_at_5": 1.0,
                "answer_correct": True,
                "grounded": True,
            },
        )
        diag = engine.diagnose(res)
        assert diag.category == FailureCategory.PASS
        assert diag.action == EXPECTED_MAPPINGS[FailureCategory.PASS]

    def test_build_report_top_failures_contain_actions(self) -> None:
        """build_report populates action in every TopFailure."""
        r1 = EvaluationResult(
            query_id="q1",
            query="q1",
            status="completed",
            expected_chunk_ids=["doc1"],
            retrieved_chunks=[RetrievedChunk(id="doc2", text="t")],
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="Miss",
            ),
        )
        r2 = EvaluationResult(
            query_id="q2",
            query="q2",
            status="completed",
            expected_chunk_ids=["doc1", "doc2"],
            retrieved_chunks=[RetrievedChunk(id="doc1", text="t")],
            diagnosis=DiagnosisResult(
                category=FailureCategory.INSUFFICIENT_CONTEXT,
                severity="warning",
                confidence=1.0,
                reason="Partial",
            ),
        )

        report = build_report([r1, r2])
        assert len(report.top_failures) == 2
        for tf in report.top_failures:
            assert tf.action == EXPECTED_MAPPINGS[tf.category]

        # Serialization roundtrip of EvaluationReport with top_failures action
        json_report = report.model_dump_json()
        restored_report = EvaluationReport.model_validate_json(json_report)
        assert len(restored_report.top_failures) == 2
        for tf in restored_report.top_failures:
            assert tf.action == EXPECTED_MAPPINGS[tf.category]


class TestTerminalReportingActionOutput:
    """Tests verifying terminal report renderer prints action recommendations."""

    def test_terminal_report_displays_action_for_top_failures(self) -> None:
        """Terminal rendering outputs 'Action: <recommendation>' for each top failure."""
        r = EvaluationResult(
            query_id="q_fail_01",
            query="What is the refund turnaround?",
            status="completed",
            expected_chunk_ids=["c1"],
            retrieved_chunks=[RetrievedChunk(id="c99", text="wrong doc")],
            diagnosis=DiagnosisResult(
                category=FailureCategory.WRONG_CHUNK_RETRIEVED,
                severity="major",
                confidence=1.0,
                reason="None of the expected chunks were retrieved.",
                evidence=["Recall@5: 0.0"],
            ),
        )
        report = build_report([r], dataset_name="test_ds", pipeline_name="test_pipe")

        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=120)
        render_terminal_report(report, console)
        output = buffer.getvalue()

        assert "TOP FAILURES" in output
        assert "q_fail_01" in output
        assert "WRONG_CHUNK_RETRIEVED" in output
        assert "Action:" in output
        assert EXPECTED_MAPPINGS[FailureCategory.WRONG_CHUNK_RETRIEVED] in output

    def test_terminal_report_no_top_failures_when_all_pass(self) -> None:
        """When all queries pass, TOP FAILURES section is omitted from terminal output."""
        r = EvaluationResult(
            query_id="q_pass_01",
            query="What is 3DS?",
            status="completed",
            expected_chunk_ids=["c1"],
            retrieved_chunks=[RetrievedChunk(id="c1", text="3DS text")],
            diagnosis=DiagnosisResult(
                category=FailureCategory.PASS,
                severity="info",
                confidence=1.0,
                reason="Pass",
            ),
        )
        report = build_report([r], dataset_name="test_ds", pipeline_name="test_pipe")

        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=120)
        render_terminal_report(report, console)
        output = buffer.getvalue()

        assert "TOP FAILURES" not in output
        assert "Action:" not in output
