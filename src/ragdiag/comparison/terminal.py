"""Side-by-side terminal presentation renderer for ComparisonReport using Rich."""

from rich.console import Console

from ragdiag.comparison.models import ComparisonReport


def _format_delta(val: float, is_percentage: bool = False, reverse_color: bool = False) -> str:
    """Format numerical delta with sign and optional color."""
    sign = "+" if val > 0 else ""
    if is_percentage:
        text = f"{sign}{val:.2f}"
    else:
        text = f"{sign}{val:.2f}"

    if abs(val) < 0.0001:
        return f"[dim]{text}[/dim]"

    if reverse_color:
        # For latency or failures, negative is good (green), positive is bad (red)
        style = "green" if val < 0 else "red"
    else:
        # For quality metrics, positive is good (green), negative is bad (red)
        style = "green" if val > 0 else "red"

    return f"[{style}]{text}[/{style}]"


def _format_count_delta(val: int, reverse_color: bool = True) -> str:
    """Format integer count delta with sign and color."""
    sign = "+" if val > 0 else ""
    text = f"{sign}{val}"
    if val == 0:
        return f"[dim]{text}[/dim]"

    if reverse_color:
        # For failures, negative is good (green), positive is bad (red)
        style = "green" if val < 0 else "red"
    else:
        style = "green" if val > 0 else "red"

    return f"[{style}]{text}[/{style}]"


def render_comparison_terminal(
    report: ComparisonReport,
    console: Console,
    total_elapsed: float | None = None,
) -> None:
    """Render a side-by-side comparative evaluation report to a Rich Console.

    Args:
        report: Aggregated `ComparisonReport`.
        console: Rich `Console` instance.
        total_elapsed: Optional total comparison wall-clock time in seconds.
    """
    c = console
    c.print("[bold cyan]RAGDiag Comparison[/bold cyan]")
    c.print("=" * 50)
    ds_label = report.dataset_name
    if report.dataset_version:
        ds_label += f" (v{report.dataset_version})"
    c.print(f"Dataset:    [bold]{ds_label}[/bold]")
    c.print(f"Pipeline A: [bold]{report.pipeline_a_name}[/bold]")
    c.print(f"Pipeline B: [bold]{report.pipeline_b_name}[/bold]\n")

    k = report.pipeline_a_report.retrieval.k

    # 1. OVERALL METRICS TABLE
    c.print("[bold]OVERALL METRICS[/bold]")
    c.print("-" * 50)
    p_a_name = report.pipeline_a_name
    p_b_name = report.pipeline_b_name
    header = f"{'Metric':<24} {p_a_name:>11} {p_b_name:>11} {'Delta (B-A)':>12}"
    c.print(f"[bold]{header}[/bold]")
    c.print("-" * 62)

    # Retrieval Metrics
    p_a = report.pipeline_a_report.retrieval.mean_precision_at_k
    p_b = report.pipeline_b_report.retrieval.mean_precision_at_k
    c.print(
        f"{'Precision@' + str(k):<24} {p_a:>11.2f} {p_b:>11.2f} "
        f"{_format_delta(report.metric_deltas.precision_at_k):>21}"
    )

    r_a = report.pipeline_a_report.retrieval.mean_recall_at_k
    r_b = report.pipeline_b_report.retrieval.mean_recall_at_k
    c.print(
        f"{'Recall@' + str(k):<24} {r_a:>11.2f} {r_b:>11.2f} "
        f"{_format_delta(report.metric_deltas.recall_at_k):>21}"
    )

    mrr_a = report.pipeline_a_report.retrieval.mrr
    mrr_b = report.pipeline_b_report.retrieval.mrr
    c.print(
        f"{'MRR':<24} {mrr_a:>11.2f} {mrr_b:>11.2f} {_format_delta(report.metric_deltas.mrr):>21}"
    )

    # Semantic Metrics (if judged)
    if report.metric_deltas.answer_correctness is not None:
        sem_a = (
            report.pipeline_a_report.semantic.answer_correctness_rate
            if report.pipeline_a_report.semantic
            else 0.0
        )
        sem_b = (
            report.pipeline_b_report.semantic.answer_correctness_rate
            if report.pipeline_b_report.semantic
            else 0.0
        )
        c_a_str = f"{sem_a:.2f}" if sem_a is not None else "N/A"
        c_b_str = f"{sem_b:.2f}" if sem_b is not None else "N/A"
        c.print(
            f"{'Answer Correctness':<24} {c_a_str:>11} {c_b_str:>11} "
            f"{_format_delta(report.metric_deltas.answer_correctness):>21}"
        )

    if report.metric_deltas.groundedness is not None:
        g_a = (
            report.pipeline_a_report.semantic.groundedness_rate
            if report.pipeline_a_report.semantic
            else 0.0
        )
        g_b = (
            report.pipeline_b_report.semantic.groundedness_rate
            if report.pipeline_b_report.semantic
            else 0.0
        )
        g_a_str = f"{g_a:.2f}" if g_a is not None else "N/A"
        g_b_str = f"{g_b:.2f}" if g_b is not None else "N/A"
        c.print(
            f"{'Groundedness':<24} {g_a_str:>11} {g_b_str:>11} "
            f"{_format_delta(report.metric_deltas.groundedness):>21}"
        )

    # Latency Metrics
    lat_mean_a = f"{report.pipeline_a_report.latency.mean_ms:.1f}ms"
    lat_mean_b = f"{report.pipeline_b_report.latency.mean_ms:.1f}ms"
    lat_delta_str = _format_delta(report.metric_deltas.mean_retrieval_ms, reverse_color=True)
    c.print(f"{'Mean Retrieval':<24} {lat_mean_a:>11} {lat_mean_b:>11} {lat_delta_str + 'ms':>23}")

    lat_p95_a = f"{report.pipeline_a_report.latency.p95_ms:.1f}ms"
    lat_p95_b = f"{report.pipeline_b_report.latency.p95_ms:.1f}ms"
    lat_p95_delta_str = _format_delta(report.metric_deltas.p95_retrieval_ms, reverse_color=True)
    c.print(
        f"{'P95 Retrieval':<24} {lat_p95_a:>11} {lat_p95_b:>11} {lat_p95_delta_str + 'ms':>23}\n"
    )

    # 2. FAILURE COUNTS TABLE
    c.print("[bold]FAILURE COUNTS[/bold]")
    c.print("-" * 50)
    header_f = (
        f"{'Category':<28} {report.pipeline_a_name:>9} {report.pipeline_b_name:>9} {'Delta':>8}"
    )
    c.print(f"[bold]{header_f}[/bold]")
    c.print("-" * 58)
    for cat, delta in report.diagnosis_deltas.items():
        cnt_a = report.pipeline_a_report.diagnosis_counts.get(cat, 0)
        cnt_b = report.pipeline_b_report.diagnosis_counts.get(cat, 0)
        # For PASS, positive is good (green). For failures, negative is good (green).
        is_rev = cat != "PASS"
        delta_str = _format_count_delta(delta, reverse_color=is_rev)
        c.print(f"{cat:<28} {cnt_a:>9} {cnt_b:>9} {delta_str:>17}")
    c.print()

    # 3. QUERY TYPES BREAKDOWN
    if report.query_type_deltas:
        c.print("[bold]QUERY TYPES[/bold]")
        c.print("-" * 50)
        for qt, qtd in report.query_type_deltas.items():
            qm_a = report.pipeline_a_report.metrics_by_query_type.get(qt)
            qm_b = report.pipeline_b_report.metrics_by_query_type.get(qt)
            r_a_val = qm_a.mean_recall_at_k if qm_a else 0.0
            r_b_val = qm_b.mean_recall_at_k if qm_b else 0.0
            mrr_a_val = qm_a.mrr if qm_a else 0.0
            mrr_b_val = qm_b.mrr if qm_b else 0.0

            r_d_str = _format_delta(qtd.recall_at_k)
            mrr_d_str = _format_delta(qtd.mrr)

            c.print(f"[bold]{qt.capitalize()}[/bold]")
            c.print(
                f"  Recall@{k}:  {r_a_val:.2f} -> {r_b_val:.2f} ({r_d_str})    "
                f"MRR: {mrr_a_val:.2f} -> {mrr_b_val:.2f} ({mrr_d_str})"
            )
            if qtd.groundedness is not None and qm_a and qm_b:
                g_a_val = qm_a.groundedness_rate or 0.0
                g_b_val = qm_b.groundedness_rate or 0.0
                g_d_str = _format_delta(qtd.groundedness)
                c.print(f"  Groundedness: {g_a_val:.2f} -> {g_b_val:.2f} ({g_d_str})")
            if qtd.total_failure_delta != 0:
                f_d_str = _format_count_delta(qtd.total_failure_delta, reverse_color=True)
                c.print(f"  Failure count delta: {f_d_str}")
            c.print()

    # 4. DECISION SECTION
    c.print("[bold]DECISION[/bold]")
    c.print("-" * 50)
    winner_color = "green" if report.overall_winner != "TIE" else "yellow"
    c.print(
        f"Overall winner: [{winner_color}][bold]{report.overall_winner}[/bold][/{winner_color}]\n"
    )
    c.print("[bold]Why:[/bold]")
    c.print(f"{report.summary}\n")
    if report.trade_off:
        c.print("[bold]Trade-off:[/bold]")
        c.print(f"[yellow]{report.trade_off}[/yellow]\n")

    # 5. QUERY OUTCOMES SUMMARY
    c.print("[bold]QUERY OUTCOMES[/bold]")
    c.print("-" * 50)
    c.print(f"Improved:  [green]{report.queries_improved}[/green]")
    c.print(f"Regressed: [red]{report.queries_regressed}[/red]")
    c.print(f"Unchanged: {report.queries_unchanged}\n")

    if total_elapsed is not None:
        c.print(f"Total comparison time: {total_elapsed:.2f}s")
