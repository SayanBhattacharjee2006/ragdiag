"""Tests for Feature 6: Automatic Result Persistence."""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from ragdiag.cli.main import app
from ragdiag.comparison.comparator import compare_reports
from ragdiag.comparison.models import ComparisonReport
from ragdiag.diagnosis.models import FailureCategory, get_action_for_category
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.result import EvaluationResult
from ragdiag.persistence.manager import ResultPersistence
from ragdiag.reporting.aggregator import build_report
from ragdiag.reporting.models import EvaluationReport

runner = CliRunner()


def _make_eval_result(query_id: str = "q1") -> EvaluationResult:
    return EvaluationResult(
        query_id=query_id,
        query="Test query?",
        retrieved_chunks=[RetrievedChunk(id="c1", text="Relevant context.", score=0.9)],
        expected_chunk_ids=["c1"],
        generated_answer="Answer.",
        expected_answer="Answer.",
        retrieval_duration_ms=15.0,
        generation_duration_ms=10.0,
        metrics={"recall_at_5": 1.0, "precision_at_5": 1.0, "reciprocal_rank": 1.0},
        diagnosis={
            "category": FailureCategory.PASS.value,
            "severity": "info",
            "confidence": 1.0,
            "reason": "OK",
            "action": get_action_for_category(FailureCategory.PASS),
            "evidence": [],
        },
        status="completed",
    )


class TestResultPersistenceManager:
    """Tests for ResultPersistence manager methods."""

    def test_persist_evaluation_creates_latest_and_history(self, tmp_path: Path) -> None:
        """persist_evaluation creates latest.json, latest.md, and timestamped history."""
        persistence = ResultPersistence(base_dir=tmp_path / ".ragdiag")
        report = build_report([_make_eval_result("q1")], pipeline_name="PipeA")

        res = persistence.persist_evaluation(report, run_id="run001")
        assert res.success
        assert res.json_path is not None and res.json_path.exists()
        assert res.markdown_path is not None and res.markdown_path.exists()
        assert res.history_json_path is not None and res.history_json_path.exists()
        assert res.history_markdown_path is not None and res.history_markdown_path.exists()

        # Check content
        loaded = EvaluationReport.model_validate_json(res.json_path.read_text(encoding="utf-8"))
        assert loaded.pipeline_name == "PipeA"

        md_content = res.markdown_path.read_text(encoding="utf-8")
        assert "# RAGDiag Evaluation Report" in md_content
        assert "## Health Profile" in md_content
        assert "## Evaluation Confidence" in md_content

    def test_multiple_evaluations_create_separate_history_files(self, tmp_path: Path) -> None:
        """Multiple runs update latest files while accumulating unique history entries."""
        persistence = ResultPersistence(base_dir=tmp_path / ".ragdiag")
        rep1 = build_report([_make_eval_result("q1")], pipeline_name="Pipe1")
        rep2 = build_report([_make_eval_result("q2")], pipeline_name="Pipe2")

        res1 = persistence.persist_evaluation(rep1, run_id="first")
        res2 = persistence.persist_evaluation(rep2, run_id="second")

        assert res1.success and res2.success
        assert res1.history_json_path != res2.history_json_path
        assert res1.history_json_path.exists()
        assert res2.history_json_path.exists()

        # latest should point to the latest run (Pipe2)
        latest_loaded = EvaluationReport.model_validate_json(
            res2.json_path.read_text(encoding="utf-8")
        )
        assert latest_loaded.pipeline_name == "Pipe2"

    def test_persist_comparison_creates_comparison_artifacts(self, tmp_path: Path) -> None:
        """persist_comparison writes to comparisons/ subfolder without
        colliding with evaluations.
        """
        persistence = ResultPersistence(base_dir=tmp_path / ".ragdiag")
        rep_a = build_report([_make_eval_result("q1")], pipeline_name="A")
        rep_b = build_report([_make_eval_result("q1")], pipeline_name="B")
        comp = compare_reports(rep_a, rep_b, [_make_eval_result("q1")], [_make_eval_result("q1")])

        res = persistence.persist_comparison(comp, run_id="comp01")
        assert res.success
        assert "comparisons" in str(res.json_path)
        assert res.json_path.exists()
        assert res.markdown_path.exists()

        loaded = ComparisonReport.model_validate_json(res.json_path.read_text(encoding="utf-8"))
        assert loaded.pipeline_a_name == "A"
        assert loaded.pipeline_b_name == "B"

    def test_persist_diagnosis_creates_diagnosis_artifacts(self, tmp_path: Path) -> None:
        """persist_diagnosis writes to diagnoses/ subfolder without modifying source report."""
        persistence = ResultPersistence(base_dir=tmp_path / ".ragdiag")
        report = build_report([_make_eval_result("q1")], pipeline_name="DiagPipe")

        res = persistence.persist_diagnosis(report, run_id="diag01")
        assert res.success
        assert "diagnoses" in str(res.json_path)
        assert res.json_path.exists()
        assert res.markdown_path.exists()

        md_content = res.markdown_path.read_text(encoding="utf-8")
        assert "# RAGDiag Pipeline Diagnosis Report" in md_content

    def test_persistence_failure_returns_warning_without_raising(self, tmp_path: Path) -> None:
        """When filesystem writes fail, returns PersistenceResult with warning without exception."""
        # Point to a path where parent is a file so mkdir fails
        bad_file = tmp_path / "blocker"
        bad_file.write_text("blocking file")
        persistence = ResultPersistence(base_dir=bad_file / ".ragdiag")

        report = build_report([_make_eval_result("q1")])
        res = persistence.persist_evaluation(report)

        assert not res.success
        assert res.warning is not None
        assert "Could not persist result" in res.warning


class TestCLIPersistenceIntegration:
    """Tests verifying CLI commands automatically trigger persistence."""

    def test_cli_run_automatically_persists_results(self, tmp_path: Path, monkeypatch) -> None:
        """ragdiag run automatically creates .ragdiag/evaluations/latest.json and .md."""
        persist_dir = tmp_path / ".ragdiag"
        monkeypatch.setenv("RAGDIAG_PERSISTENCE_DIR", str(persist_dir))

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
        latest_json = persist_dir / "evaluations" / "latest.json"
        latest_md = persist_dir / "evaluations" / "latest.md"

        assert latest_json.exists()
        assert latest_md.exists()

    def test_cli_compare_automatically_persists_results(self, tmp_path: Path, monkeypatch) -> None:
        """ragdiag compare automatically creates .ragdiag/comparisons/latest.json and .md."""
        persist_dir = tmp_path / ".ragdiag"
        monkeypatch.setenv("RAGDIAG_PERSISTENCE_DIR", str(persist_dir))

        result = runner.invoke(
            app,
            [
                "compare",
                "--pipeline-a",
                "examples/basic_pipeline.py",
                "--pipeline-b",
                "examples/basic_pipeline.py",
                "--dataset",
                "examples/basic_dataset.json",
            ],
        )
        assert result.exit_code == 0
        latest_json = persist_dir / "comparisons" / "latest.json"
        latest_md = persist_dir / "comparisons" / "latest.md"

        assert latest_json.exists()
        assert latest_md.exists()

    def test_cli_diagnose_automatically_persists_results(self, tmp_path: Path, monkeypatch) -> None:
        """ragdiag diagnose automatically creates .ragdiag/diagnoses/latest.json and .md."""
        persist_dir = tmp_path / ".ragdiag"
        monkeypatch.setenv("RAGDIAG_PERSISTENCE_DIR", str(persist_dir))

        report = build_report([_make_eval_result("q1")])
        report_file = tmp_path / "eval.json"
        report_file.write_text(report.model_dump_json(), encoding="utf-8")

        result = runner.invoke(app, ["diagnose", str(report_file)])
        assert result.exit_code == 0

        latest_json = persist_dir / "diagnoses" / "latest.json"
        latest_md = persist_dir / "diagnoses" / "latest.md"

        assert latest_json.exists()
        assert latest_md.exists()

    def test_persistence_failure_in_cli_does_not_fail_run(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """When persistence fails in ragdiag run, CLI succeeds with a warning."""
        persist_dir = tmp_path / ".ragdiag"
        monkeypatch.setenv("RAGDIAG_PERSISTENCE_DIR", str(persist_dir))

        from ragdiag.persistence.manager import PersistenceResult

        with patch.object(
            ResultPersistence,
            "persist_evaluation",
            return_value=PersistenceResult(
                success=False,
                warning="Simulated disk write failure",
            ),
        ):
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
            assert "Warning:" in result.stdout
            assert "Simulated disk write failure" in result.stdout
