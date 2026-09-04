"""Tests for Feature 5: ragdiag diagnose CLI command and inspection engine."""

import io
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from ragdiag.cli.main import app
from ragdiag.diagnosis.inspector import (
    DiagnosisInspection,
    inspect_report,
    render_diagnosis_terminal,
)
from ragdiag.diagnosis.models import FailureCategory, get_action_for_category
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.result import EvaluationResult
from ragdiag.reporting.aggregator import build_report
from ragdiag.reporting.models import EvaluationReport

runner = CliRunner()


def _make_sample_report(tmp_path: Path) -> tuple[EvaluationReport, Path]:
    """Helper to construct and save a sample EvaluationReport with known failures."""
    r1 = EvaluationResult(
        query_id="q12",
        query="What is the refund turnaround time?",
        retrieved_chunks=[RetrievedChunk(id="c99", text="Irrelevant text.", score=0.1)],
        expected_chunk_ids=["c1"],
        generated_answer="Refunds take 3 days.",
        expected_answer="Refunds take 5 to 7 days.",
        retrieval_duration_ms=25.0,
        generation_duration_ms=10.0,
        metrics={"recall_at_5": 0.0, "precision_at_5": 0.0, "reciprocal_rank": 0.0},
        diagnosis={
            "category": FailureCategory.WRONG_CHUNK_RETRIEVED.value,
            "severity": "major",
            "confidence": 0.95,
            "reason": "Complete retrieval miss.",
            "action": get_action_for_category(FailureCategory.WRONG_CHUNK_RETRIEVED),
            "evidence": ["None of the expected chunks were retrieved."],
        },
        status="completed",
    )
    r2 = EvaluationResult(
        query_id="q27",
        query="Why was the card auto-debit rejected?",
        retrieved_chunks=[RetrievedChunk(id="c2", text="Mandate expired.", score=0.9)],
        expected_chunk_ids=["c2"],
        generated_answer="Unknown reason.",
        expected_answer="Mandate expired.",
        retrieval_duration_ms=30.0,
        generation_duration_ms=12.0,
        metrics={
            "recall_at_5": 1.0,
            "precision_at_5": 1.0,
            "reciprocal_rank": 1.0,
            "answer_correct": False,
            "grounded": True,
        },
        diagnosis={
            "category": FailureCategory.ANSWER_INCORRECT.value,
            "severity": "major",
            "confidence": 0.90,
            "reason": "Answer contradicts context.",
            "action": get_action_for_category(FailureCategory.ANSWER_INCORRECT),
            "evidence": ["Model hallucinated or failed to answer correctly."],
        },
        status="completed",
    )
    r3 = EvaluationResult(
        query_id="q30",
        query="What is the support contact email?",
        retrieved_chunks=[RetrievedChunk(id="c3", text="Contact support@example.com", score=0.95)],
        expected_chunk_ids=["c3"],
        generated_answer="support@example.com",
        expected_answer="support@example.com",
        retrieval_duration_ms=15.0,
        generation_duration_ms=8.0,
        metrics={"recall_at_5": 1.0, "precision_at_5": 1.0, "reciprocal_rank": 1.0},
        diagnosis={
            "category": FailureCategory.PASS.value,
            "severity": "info",
            "confidence": 1.0,
            "reason": "All checks passed.",
            "action": get_action_for_category(FailureCategory.PASS),
            "evidence": [],
        },
        status="completed",
    )

    report = build_report(
        [r1, r2, r3], dataset_name="sample_eval", dataset_version="1.0", pipeline_name="MyPipe"
    )
    report_file = tmp_path / "evaluation_report.json"
    report_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report, report_file


class TestDiagnoseCLI:
    """Tests for ragdiag diagnose CLI command."""

    def test_diagnose_valid_evaluation_report(self, tmp_path: Path, monkeypatch) -> None:
        """ragdiag diagnose loads valid report and outputs structured diagnosis."""
        monkeypatch.setenv("RAGDIAG_PERSISTENCE_DIR", str(tmp_path / ".ragdiag"))
        _, report_file = _make_sample_report(tmp_path)

        result = runner.invoke(app, ["diagnose", str(report_file)])
        assert result.exit_code == 0
        assert "RAG DIAGNOSIS" in result.stdout
        assert "3 queries evaluated" in result.stdout
        assert "1 passed" in result.stdout
        assert "2 failed" in result.stdout

    def test_diagnose_output_contains_top_failures_and_actions(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Top failure categories appear with counts and Feature 1 action recommendations."""
        monkeypatch.setenv("RAGDIAG_PERSISTENCE_DIR", str(tmp_path / ".ragdiag"))
        _, report_file = _make_sample_report(tmp_path)

        result = runner.invoke(app, ["diagnose", str(report_file)])
        assert result.exit_code == 0
        assert "TOP FAILURE MODES" in result.stdout
        assert "WRONG_CHUNK_RETRIEVED" in result.stdout
        normalized = " ".join(result.stdout.split())
        assert get_action_for_category(FailureCategory.WRONG_CHUNK_RETRIEVED) in normalized
        assert get_action_for_category(FailureCategory.ANSWER_INCORRECT) in normalized

    def test_diagnose_output_contains_important_query_details(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Important queries section shows query ID, diagnosis, reason, and action."""
        monkeypatch.setenv("RAGDIAG_PERSISTENCE_DIR", str(tmp_path / ".ragdiag"))
        _, report_file = _make_sample_report(tmp_path)

        result = runner.invoke(app, ["diagnose", str(report_file)])
        assert result.exit_code == 0
        assert "IMPORTANT QUERIES" in result.stdout
        assert "q12" in result.stdout
        assert "q27" in result.stdout

    def test_diagnose_output_contains_health_and_confidence(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Health score and Evaluation Confidence appear in diagnosis output."""
        monkeypatch.setenv("RAGDIAG_PERSISTENCE_DIR", str(tmp_path / ".ragdiag"))
        _, report_file = _make_sample_report(tmp_path)

        result = runner.invoke(app, ["diagnose", str(report_file)])
        assert result.exit_code == 0
        assert "HEALTH" in result.stdout
        assert "EVALUATION CONFIDENCE" in result.stdout
        assert "Score:" in result.stdout
        assert "Grade:" in result.stdout
        assert "Level:" in result.stdout

    def test_diagnose_missing_file_produces_error(self, tmp_path: Path) -> None:
        """Specifying a non-existent file produces a clean error panel without crash."""
        non_existent = tmp_path / "does_not_exist.json"
        result = runner.invoke(app, ["diagnose", str(non_existent)])
        assert result.exit_code == 1
        assert "File Not Found" in result.stdout

    def test_diagnose_invalid_json_produces_error(self, tmp_path: Path) -> None:
        """Specifying malformed JSON produces a clear error without Python traceback."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{invalid_json: true", encoding="utf-8")

        result = runner.invoke(app, ["diagnose", str(bad_json)])
        assert result.exit_code == 1
        assert "Invalid Evaluation Report" in result.stdout

    def test_diagnose_invalid_schema_produces_error(self, tmp_path: Path) -> None:
        """JSON that does not match EvaluationReport schema produces a clear error."""
        wrong_schema = tmp_path / "wrong.json"
        wrong_schema.write_text('{"unrelated_key": 123}', encoding="utf-8")

        result = runner.invoke(app, ["diagnose", str(wrong_schema)])
        assert result.exit_code == 1
        assert "Invalid Evaluation Report" in result.stdout

        wrong_type = tmp_path / "wrong_type.json"
        wrong_type.write_text('{"total_queries": "not_an_int"}', encoding="utf-8")
        result2 = runner.invoke(app, ["diagnose", str(wrong_type)])
        assert result2.exit_code == 1
        assert "Invalid Evaluation Report" in result2.stdout

    def test_diagnose_all_pass_evaluation(self, tmp_path: Path, monkeypatch) -> None:
        """When all queries pass, diagnosis reports clean pass without failure modes."""
        monkeypatch.setenv("RAGDIAG_PERSISTENCE_DIR", str(tmp_path / ".ragdiag"))
        r_pass = EvaluationResult(
            query_id="q1",
            query="test",
            retrieved_chunks=[RetrievedChunk(id="c1", text="text")],
            expected_chunk_ids=["c1"],
            status="completed",
            diagnosis={
                "category": FailureCategory.PASS.value,
                "severity": "info",
                "confidence": 1.0,
                "reason": "OK",
                "action": get_action_for_category(FailureCategory.PASS),
                "evidence": [],
            },
        )
        report = build_report([r_pass])
        rep_file = tmp_path / "pass_report.json"
        rep_file.write_text(report.model_dump_json(), encoding="utf-8")

        result = runner.invoke(app, ["diagnose", str(rep_file)])
        assert result.exit_code == 0
        assert "0 failed" in result.stdout
        assert "All queries passed successfully" in result.stdout


class TestInspectReportAPI:
    """Tests for Python SDK inspect_report function."""

    def test_inspect_report_returns_typed_inspection(self, tmp_path: Path) -> None:
        """inspect_report returns DiagnosisInspection model with expected attributes."""
        report, _ = _make_sample_report(tmp_path)
        diag = inspect_report(report)

        assert isinstance(diag, DiagnosisInspection)
        assert diag.total_queries == 3
        assert diag.passed_queries == 1
        assert diag.failed_queries == 2
        assert len(diag.failure_modes) == 2
        assert any(fm.category == "WRONG_CHUNK_RETRIEVED" for fm in diag.failure_modes)
        assert any(fm.category == "ANSWER_INCORRECT" for fm in diag.failure_modes)

    def test_render_diagnosis_terminal_direct(self, tmp_path: Path) -> None:
        """render_diagnosis_terminal renders cleanly to a Console buffer."""
        report, _ = _make_sample_report(tmp_path)
        buf = io.StringIO()
        c = Console(file=buf, force_terminal=False, color_system=None, width=120)
        render_diagnosis_terminal(report, c)
        output = buf.getvalue()

        assert "RAG DIAGNOSIS" in output
        assert "TOP FAILURE MODES" in output
        assert "IMPORTANT QUERIES" in output
