"""Automatic result persistence manager for RAGDiag reports."""

import datetime
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ragdiag.comparison.models import ComparisonReport
from ragdiag.persistence.markdown import (
    render_comparison_markdown,
    render_diagnosis_markdown,
    render_evaluation_markdown,
)
from ragdiag.reporting.models import EvaluationReport

ReportType = Literal["evaluations", "comparisons", "diagnoses"]


@dataclass
class PersistenceResult:
    """Result of an automatic persistence operation."""

    success: bool
    json_path: Path | None = None
    markdown_path: Path | None = None
    history_json_path: Path | None = None
    history_markdown_path: Path | None = None
    warning: str | None = None


def _atomic_write_text(path: Path, content: str) -> None:
    """Safely write text to a path via temporary file replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


class ResultPersistence:
    """Manages automatic persistence of evaluation, comparison, and diagnosis results."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        if base_dir is not None:
            self.base_dir = Path(base_dir)
        else:
            env_override = os.environ.get("RAGDIAG_PERSISTENCE_DIR")
            self.base_dir = Path(env_override) if env_override else Path.cwd() / ".ragdiag"

    def _persist_payload(
        self,
        report_type: ReportType,
        json_content: str,
        markdown_content: str,
        run_id: str | None = None,
    ) -> PersistenceResult:
        """Atomically persist JSON and Markdown to latest and history destinations."""
        try:
            category_dir = self.base_dir / report_type
            history_dir = category_dir / "history"

            category_dir.mkdir(parents=True, exist_ok=True)
            history_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
            token = run_id or uuid.uuid4().hex[:6]
            file_stem = f"{timestamp}_{token}"

            latest_json = category_dir / "latest.json"
            latest_md = category_dir / "latest.md"
            hist_json = history_dir / f"{file_stem}.json"
            hist_md = history_dir / f"{file_stem}.md"

            _atomic_write_text(latest_json, json_content)
            _atomic_write_text(latest_md, markdown_content)
            _atomic_write_text(hist_json, json_content)
            _atomic_write_text(hist_md, markdown_content)

            return PersistenceResult(
                success=True,
                json_path=latest_json,
                markdown_path=latest_md,
                history_json_path=hist_json,
                history_markdown_path=hist_md,
            )
        except OSError as exc:
            return PersistenceResult(
                success=False,
                warning=f"Could not persist result to {self.base_dir}: {exc}",
            )

    def persist_evaluation(
        self, report: EvaluationReport, run_id: str | None = None
    ) -> PersistenceResult:
        """Persist EvaluationReport as latest.json, latest.md, and timestamped history."""
        json_str = report.model_dump_json(indent=2)
        md_str = render_evaluation_markdown(report)
        return self._persist_payload("evaluations", json_str, md_str, run_id=run_id)

    def persist_comparison(
        self, report: ComparisonReport, run_id: str | None = None
    ) -> PersistenceResult:
        """Persist ComparisonReport as latest.json, latest.md, and timestamped history."""
        json_str = report.model_dump_json(indent=2)
        md_str = render_comparison_markdown(report)
        return self._persist_payload("comparisons", json_str, md_str, run_id=run_id)

    def persist_diagnosis(
        self, report: EvaluationReport, run_id: str | None = None
    ) -> PersistenceResult:
        """Persist Diagnosis inspection as latest.json, latest.md, and timestamped history."""
        json_str = report.model_dump_json(indent=2)
        md_str = render_diagnosis_markdown(report)
        return self._persist_payload("diagnoses", json_str, md_str, run_id=run_id)
