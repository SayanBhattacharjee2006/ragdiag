"""Terminal presentation renderer for EvaluationReport using Rich."""

from rich.console import Console

from ragdiag.reporting.models import EvaluationReport

CATEGORY_SEVERITY_STYLES: dict[str, str] = {
    "PASS": "green",
    "WRONG_CHUNK_RETRIEVED": "red",
    "WRONG_CHUNK_RANK": "yellow",
    "INSUFFICIENT_CONTEXT": "yellow",
    "RETRIEVED_BUT_NOT_GROUNDED": "red",
    "ANSWER_INCORRECT": "red",
    "LATENCY_OUTLIER": "yellow",
    "UNKNOWN": "red",
}


def render_terminal_report(
    report: EvaluationReport,
    console: Console,
    total_elapsed: float | None = None,
) -> None:
    """Render a comprehensive, polished diagnostic report to a Rich Console.

    Args:
        report: Aggregated `EvaluationReport` to format and display.
        console: Configured Rich `Console` instance.
        total_elapsed: Optional total evaluation execution time in seconds.
    """
    c = console
    c.print("[bold cyan]RAGDiag[/bold cyan]")
    c.print("-" * 44)
    if report.pipeline_name:
        c.print(f"Pipeline: [bold]{report.pipeline_name}[/bold]")
    if report.dataset_name:
        ds_label = report.dataset_name
        if report.dataset_version:
            ds_label += f" (v{report.dataset_version})"
        c.print(f"Dataset:  [bold]{ds_label}[/bold]")
    c.print(f"Queries:  [bold]{report.total_queries}[/bold]\n")

    c.print("[bold green]Evaluation complete.[/bold green]\n")

    # 1. OVERALL
    c.print("[bold]OVERALL - Retrieval Metrics[/bold]")
    c.print("-" * 44)
    k = report.retrieval.k
    c.print(f"Precision@{k}:        {report.retrieval.mean_precision_at_k:.2f}")
    c.print(f"Recall@{k}:           {report.retrieval.mean_recall_at_k:.2f}")
    c.print(f"MRR:                 {report.retrieval.mrr:.2f}\n")

    if report.semantic is not None:
        c.print("[bold]Semantic Quality[/bold]")
        c.print("-" * 24)
        c_rate = (
            f"{report.semantic.answer_correctness_rate:.2f}"
            if report.semantic.answer_correctness_rate is not None
            else "N/A"
        )
        g_rate = (
            f"{report.semantic.groundedness_rate:.2f}"
            if report.semantic.groundedness_rate is not None
            else "N/A"
        )
        c.print(f"Answer Correctness:  {c_rate}")
        c.print(f"Groundedness:        {g_rate}")
        if report.semantic.mean_judge_confidence is not None:
            c.print(f"Judge Confidence:    {report.semantic.mean_judge_confidence:.2f}")
        if report.judge_failures > 0:
            c.print(f"Judge Failures:      [red]{report.judge_failures}[/red]")
        c.print()

    c.print("[bold]Retrieval Latency[/bold]")
    c.print("-" * 24)
    c.print(f"Mean:                {report.latency.mean_ms:.2f} ms")
    c.print(f"P50:                 {report.latency.p50_ms:.2f} ms")
    c.print(f"P95:                 {report.latency.p95_ms:.2f} ms")
    c.print(f"P99:                 {report.latency.p99_ms:.2f} ms\n")

    # 2. FAILURE ANALYSIS
    c.print("[bold]FAILURE ANALYSIS (Root Cause Analysis)[/bold]")
    c.print("-" * 44)
    for cat, count in report.diagnosis_counts.items():
        style = CATEGORY_SEVERITY_STYLES.get(cat, "white")
        count_display = f"[{style}]{count}[/{style}]" if count > 0 else f"[dim]{count}[/dim]"
        cat_label = f"{cat}:"
        c.print(f"{cat_label:<28} {count_display}")
    c.print()

    # 3. QUERY TYPES
    if report.metrics_by_query_type:
        c.print("[bold]QUERY TYPES[/bold]")
        c.print("-" * 44)
        for qt, qm in report.metrics_by_query_type.items():
            q_str = "query" if qm.total_queries == 1 else "queries"
            c.print(f"[bold]{qt.capitalize()}[/bold] ({qm.total_queries} {q_str})")
            line = f"  Recall@{k}: {qm.mean_recall_at_k:.2f}  MRR: {qm.mrr:.2f}"
            if qm.groundedness_rate is not None:
                line += f"  Groundedness: {qm.groundedness_rate:.2f}"
            if qm.answer_correctness_rate is not None:
                line += f"  Correctness: {qm.answer_correctness_rate:.2f}"
            c.print(line)

            # Highlight failures for this query type if any
            qt_failures = {
                cat: cnt for cat, cnt in qm.diagnosis_counts.items() if cat != "PASS" and cnt > 0
            }
            if qt_failures:
                fail_summary = ", ".join(f"{cat}: {cnt}" for cat, cnt in qt_failures.items())
                c.print(f"  [yellow]Failures:[/yellow] {fail_summary}")
            c.print()

    # 4. TOP FAILURES
    if report.top_failures:
        c.print("[bold]TOP FAILURES[/bold]")
        c.print("-" * 44)
        for tf in report.top_failures:
            style = CATEGORY_SEVERITY_STYLES.get(tf.category.value, "red")
            c.print(f"[bold]{tf.query_id}[/bold]  [{style}]{tf.category.value}[/{style}]")
            c.print(f"{tf.reason}")
            if tf.evidence:
                first_ev = tf.evidence[0]
                c.print(f"[dim]{first_ev}[/dim]")
            c.print()

    # 5. INSIGHTS
    if report.overall_insights:
        c.print("[bold]INSIGHTS[/bold]")
        c.print("-" * 44)
        for insight in report.overall_insights:
            c.print(f"* {insight}")
        c.print()

    # 6. SUMMARY FOOTER
    fail_style = "red" if report.failed_queries > 0 else "green"
    c.print(f"Completed:  [green]{report.completed_queries}[/green]")
    c.print(f"Failed:     [{fail_style}]{report.failed_queries}[/{fail_style}]")
    if report.judge_failures > 0:
        c.print(f"Judge Failures: [red]{report.judge_failures}[/red]")
    if total_elapsed is not None:
        c.print(f"Total time: {total_elapsed:.2f}s")
