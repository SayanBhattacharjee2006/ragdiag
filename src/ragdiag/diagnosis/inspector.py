"""Diagnosis inspection engine and terminal renderer for existing evaluation reports."""

from pydantic import BaseModel, Field
from rich.console import Console

from ragdiag.diagnosis.models import FailureCategory, get_action_for_category
from ragdiag.reporting.models import EvaluationReport, TopFailure

# Taxonomy severity hierarchy for deterministic ordering
SEVERITY_ORDER: dict[str, int] = {
    FailureCategory.UNKNOWN.value: 0,
    FailureCategory.WRONG_CHUNK_RETRIEVED.value: 0,
    FailureCategory.RETRIEVED_BUT_NOT_GROUNDED.value: 0,
    FailureCategory.ANSWER_INCORRECT.value: 0,
    FailureCategory.INSUFFICIENT_CONTEXT.value: 1,
    FailureCategory.WRONG_CHUNK_RANK.value: 1,
    FailureCategory.LATENCY_OUTLIER.value: 1,
    FailureCategory.PASS.value: 2,
}


class FailureModeSummary(BaseModel):
    """Aggregate summary of an observed failure category."""

    category: str
    count: int
    action: str


class DiagnosisInspection(BaseModel):
    """Structured inspection summary of an evaluated RAG pipeline's diagnostic evidence."""

    total_queries: int = 0
    passed_queries: int = 0
    failed_queries: int = 0
    failure_modes: list[FailureModeSummary] = Field(default_factory=list)
    important_queries: list[TopFailure] = Field(default_factory=list)
    health_score: float = 100.0
    health_grade: str = "Excellent"
    health_status: str = "Healthy"
    confidence_score: float = 100.0
    confidence_level: str = "High"


def inspect_report(report: EvaluationReport) -> DiagnosisInspection:
    """Extract diagnostic intelligence and failure rankings from an EvaluationReport.

    Reuses existing failure categories, top failures, health profiles, and
    action recommendations without performing new evaluation or AI inference.

    Args:
        report: Fully validated `EvaluationReport`.

    Returns:
        Structured `DiagnosisInspection` summary.
    """
    total = report.total_queries
    passed = report.diagnosis_counts.get(FailureCategory.PASS.value, 0)
    failed = total - passed

    # 1. Failure modes ordered by severity first, then count descending, then category name
    active_failures = [
        (cat, count)
        for cat, count in report.diagnosis_counts.items()
        if cat != FailureCategory.PASS.value and count > 0
    ]

    def failure_sort_key(item: tuple[str, int]) -> tuple[int, int, str]:
        cat, count = item
        sev_rank = SEVERITY_ORDER.get(cat, 99)
        return (sev_rank, -count, cat)

    sorted_failures = sorted(active_failures, key=failure_sort_key)
    failure_modes = [
        FailureModeSummary(
            category=cat,
            count=cnt,
            action=get_action_for_category(cat),
        )
        for cat, cnt in sorted_failures
    ]

    return DiagnosisInspection(
        total_queries=total,
        passed_queries=passed,
        failed_queries=failed,
        failure_modes=failure_modes,
        important_queries=report.top_failures,
        health_score=report.health_profile.score,
        health_grade=report.health_profile.grade,
        health_status=report.health_profile.status,
        confidence_score=report.confidence.score,
        confidence_level=report.confidence.level,
    )


def render_diagnosis_terminal(report: EvaluationReport, console: Console) -> None:
    """Render a structured diagnostic inspection report to a Rich Console.

    Args:
        report: EvaluationReport to inspect and display.
        console: Rich Console instance.
    """
    c = console
    diag = inspect_report(report)

    c.print("[bold cyan]RAG DIAGNOSIS[/bold cyan]")
    c.print("=" * 44)

    # Header stats
    c.print("\n[bold]Overall:[/bold]")
    c.print(f"  {diag.total_queries} queries evaluated")
    c.print(f"  [green]{diag.passed_queries} passed[/green]")
    fail_style = "red" if diag.failed_queries > 0 else "green"
    c.print(f"  [{fail_style}]{diag.failed_queries} failed[/{fail_style}]\n")

    # Top Failure Modes
    if diag.failure_modes:
        c.print("[bold]TOP FAILURE MODES[/bold]")
        c.print("-" * 44)
        for idx, fm in enumerate(diag.failure_modes, start=1):
            sev_rank = SEVERITY_ORDER.get(fm.category, 1)
            cat_color = "red" if sev_rank == 0 else "yellow"
            c.print(f"{idx}. [{cat_color}][bold]{fm.category}[/bold][/{cat_color}]")
            c.print(f"   Queries: [bold]{fm.count}[/bold]")
            c.print(f"   [cyan]Action:[/cyan] {fm.action}\n")
    else:
        c.print(
            "[bold green]All queries passed successfully! No failure modes detected.[/bold green]\n"
        )

    # Important Queries
    if diag.important_queries:
        c.print("[bold]IMPORTANT QUERIES[/bold]")
        c.print("-" * 44)
        for tf in diag.important_queries:
            cat_name = tf.category.value if hasattr(tf.category, "value") else str(tf.category)
            sev_rank = SEVERITY_ORDER.get(cat_name, 1)
            cat_color = "red" if sev_rank == 0 else "yellow"
            c.print(f"[bold]{tf.query_id}[/bold]")
            c.print(f"  Diagnosis: [{cat_color}]{cat_name}[/{cat_color}]")
            if tf.reason:
                c.print(f"  Reason:    {tf.reason}")
            if tf.evidence:
                first_ev = tf.evidence[0]
                c.print(f"  Evidence:  [dim]{first_ev}[/dim]")
            if tf.action:
                c.print(f"  [cyan]Action:[/cyan]    {tf.action}")
            c.print()

    # Health
    c.print("[bold]HEALTH[/bold]")
    c.print("-" * 44)
    h_color = (
        "green"
        if diag.health_status == "Healthy"
        else ("yellow" if diag.health_status == "Degraded" else "red")
    )
    score_str = (
        f"{diag.health_score:.0f}" if diag.health_score.is_integer() else f"{diag.health_score:.1f}"
    )
    c.print(f"Score: [{h_color}]{score_str}/100[/{h_color}]")
    c.print(f"Grade: [{h_color}]{diag.health_grade}[/{h_color}] (Status: {diag.health_status})\n")

    # Evaluation Confidence
    c.print("[bold]EVALUATION CONFIDENCE[/bold]")
    c.print("-" * 44)
    c_color = (
        "green"
        if diag.confidence_level in ("High", "Good")
        else ("yellow" if diag.confidence_level == "Moderate" else "red")
    )
    conf_str = (
        f"{diag.confidence_score:.0f}"
        if diag.confidence_score.is_integer()
        else f"{diag.confidence_score:.1f}"
    )
    c.print(f"Score: [{c_color}]{conf_str}/100[/{c_color}]")
    c.print(f"Level: [{c_color}]{diag.confidence_level}[/{c_color}]\n")
