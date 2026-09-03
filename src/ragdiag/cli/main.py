"""CLI interface for RAGDiag."""

import os
import sys
import time
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

import ragdiag
from ragdiag.dataset.exceptions import DatasetLoadError, DatasetValidationError
from ragdiag.dataset.loader import load_dataset
from ragdiag.dataset.validator import validate_dataset
from ragdiag.metrics.aggregation import aggregate_metrics
from ragdiag.pipeline.exceptions import PipelineError
from ragdiag.pipeline.loader import load_pipeline
from ragdiag.runner.evaluator import Evaluator

app = typer.Typer(
    name="ragdiag",
    help="RAGDiag: RAG evaluation and root-cause diagnosis CLI.",
    no_args_is_help=True,
    add_completion=False,
)


def get_console() -> Console:
    """Return a Rich Console configured via public API based on stdout TTY status.

    When output is captured by a test runner or redirected to a non-interactive
    stream (pipe or file), terminal escape sequences are disabled for clean,
    deterministic output. When running in an interactive terminal, standard
    terminal detection and styling apply.
    """
    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return Console(force_terminal=None if is_tty else False)


console = get_console()


def version_callback(value: bool) -> None:
    """Print the version of ragdiag and exit."""
    if value:
        c = get_console()
        c.print(f"[bold cyan]ragdiag[/bold cyan] version [green]{ragdiag.__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show RAGDiag version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """RAGDiag: RAG evaluation and root-cause diagnosis CLI."""


@app.command()
def validate(
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            "-d",
            help="Path to evaluation dataset JSON file.",
        ),
    ],
) -> None:
    """Validate a golden evaluation dataset JSON file."""
    c = get_console()
    try:
        ds = load_dataset(dataset)
        summary = validate_dataset(ds)
    except (DatasetLoadError, DatasetValidationError) as err:
        c.print(
            Panel(
                f"[bold red]Validation Failed[/bold red]\n\n{err}",
                title="[bold red]Dataset Error[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from err
    except Exception as exc:
        c.print(
            Panel(
                f"[bold red]Unexpected Error[/bold red]\n\n{exc}",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from exc

    breakdown = "\n".join(
        f"  [cyan]{qt}[/cyan]: {count}" for qt, count in summary.query_type_counts.items()
    )

    summary_text = (
        f"[bold]Dataset:[/bold] {summary.name}\n"
        f"[bold]Version:[/bold] {summary.version}\n"
        f"[bold]Samples:[/bold] {summary.total_samples}\n"
        f"[bold]Query types:[/bold]\n{breakdown}\n\n"
        f"[bold green]Dataset is valid.[/bold green]"
    )

    c.print(
        Panel(
            summary_text,
            title="[bold green]Dataset Validation Summary[/bold green]",
            border_style="green",
        )
    )


@app.command()
def run(
    pipeline: Annotated[
        str,
        typer.Option(
            "--pipeline",
            "-p",
            help="Path to Python file defining a 'pipeline = MyPipeline()' adapter instance.",
        ),
    ],
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            "-d",
            help="Path to golden evaluation dataset JSON file.",
        ),
    ],
    judge: Annotated[
        str | None,
        typer.Option(
            "--judge",
            "-j",
            help="LLM judge provider (e.g. 'openai').",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model identifier for the LLM judge.",
        ),
    ] = "gpt-4o-mini",
) -> None:
    """Run RAG evaluation against a pipeline and dataset."""
    c = get_console()

    judge_instance = None
    if judge is not None:
        judge_norm = judge.strip().lower()
        if judge_norm == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                c.print(
                    Panel(
                        "[bold red]OpenAI API Key Missing[/bold red]\n\n"
                        "To evaluate with the OpenAI judge, set the [bold]OPENAI_API_KEY[/bold] "
                        "environment variable or run without '--judge'.",
                        title="[bold red]Authentication Error[/bold red]",
                        border_style="red",
                    )
                )
                raise typer.Exit(code=1)
            from ragdiag.judges.openai import OpenAIJudge

            judge_instance = OpenAIJudge(model=model)
        else:
            c.print(
                Panel(
                    f"[bold red]Unsupported Judge Provider[/bold red]\n\n"
                    f"Provider '{judge}' is not supported. Supported providers: 'openai'.",
                    title="[bold red]Configuration Error[/bold red]",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)
    try:
        loaded_pipeline = load_pipeline(pipeline)
    except PipelineError as exc:
        c.print(
            Panel(
                f"[bold red]Pipeline Load Failed[/bold red]\n\n{exc}",
                title="[bold red]Pipeline Error[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from exc

    try:
        loaded_dataset = load_dataset(dataset)
    except (DatasetLoadError, DatasetValidationError) as exc:
        c.print(
            Panel(
                f"[bold red]Dataset Load Failed[/bold red]\n\n{exc}",
                title="[bold red]Dataset Error[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from exc

    c.print("[bold cyan]RAGDiag[/bold cyan]")
    c.print("-" * 24)
    c.print(f"Pipeline: [bold]{loaded_pipeline.name}[/bold]")
    c.print(f"Dataset:  [bold]{loaded_dataset.name}[/bold]")
    c.print(f"Queries:  [bold]{len(loaded_dataset.samples)}[/bold]\n")
    c.print("[dim]Running evaluation...[/dim]\n")

    evaluator = Evaluator(k=5, judge=judge_instance)
    start_time = time.perf_counter()
    results = evaluator.evaluate(loaded_pipeline, loaded_dataset)
    total_elapsed = time.perf_counter() - start_time

    report = aggregate_metrics(results, k=5)

    c.print("[bold green]Evaluation complete.[/bold green]\n")
    c.print("[bold]Retrieval Metrics[/bold]")
    c.print("-" * 24)
    c.print(f"Precision@{report.k}:  {report.mean_precision_at_k:.2f}")
    c.print(f"Recall@{report.k}:     {report.mean_recall_at_k:.2f}")
    c.print(f"MRR:          {report.mrr:.2f}\n")

    if report.judged_queries > 0 or report.judge_failures > 0:
        judge_name = judge or "llm"
        c.print(f"[bold]Semantic Metrics (Judge: {judge_name})[/bold]")
        c.print("-" * 24)
        c_val = (
            report.answer_correctness_rate if report.answer_correctness_rate is not None else 0.0
        )
        g_val = report.groundedness_rate if report.groundedness_rate is not None else 0.0
        c.print(f"Answer correctness: {c_val:.2f}")
        c.print(f"Groundedness:       {g_val:.2f}")
        if report.mean_judge_confidence is not None:
            c.print(f"Judge confidence:   {report.mean_judge_confidence:.2f}")
        if report.judge_failures > 0:
            c.print(f"Judge failures:     [red]{report.judge_failures}[/red]")
        c.print()

    c.print("[bold]Root Cause Analysis[/bold]")
    c.print("-" * 24)
    category_labels = [
        ("PASS", "PASS:"),
        ("WRONG_CHUNK_RETRIEVED", "Wrong chunk retrieved:"),
        ("WRONG_CHUNK_RANK", "Wrong chunk rank:"),
        ("INSUFFICIENT_CONTEXT", "Insufficient context:"),
        ("RETRIEVED_BUT_NOT_GROUNDED", "Retrieved but not grounded:"),
        ("ANSWER_INCORRECT", "Answer incorrect:"),
        ("LATENCY_OUTLIER", "Latency outlier:"),
        ("UNKNOWN", "Unknown:"),
    ]
    for cat_key, label in category_labels:
        cnt = report.diagnosis_counts.get(cat_key, 0)
        c.print(f"{label:<28} {cnt}")
    c.print()

    failing_results = [
        r
        for r in results
        if (
            (hasattr(r.diagnosis, "category") and str(r.diagnosis.category) != "PASS")
            or (isinstance(r.diagnosis, dict) and r.diagnosis.get("category") != "PASS")
        )
    ]
    if failing_results:
        c.print("[bold]Top Failures[/bold]")
        c.print("-" * 24)
        for r in failing_results[:5]:
            cat_name = (
                r.diagnosis.category.value
                if hasattr(r.diagnosis.category, "value")
                else str(r.diagnosis.category)
            )
            reason = (
                r.diagnosis.reason
                if hasattr(r.diagnosis, "reason")
                else r.diagnosis.get("reason", "")
            )
            c.print(f"[bold]{r.query_id}[/bold]  [red]{cat_name}[/red]")
            c.print(f"{reason}")
            if hasattr(r.diagnosis, "evidence") and r.diagnosis.evidence:
                first_ev = r.diagnosis.evidence[0]
                c.print(f"[dim]{first_ev}[/dim]")
            c.print()

    c.print("[bold]Retrieval Latency[/bold]")
    c.print("-" * 24)
    c.print(f"Mean:   {report.retrieval_latency.mean_ms:.2f} ms")
    c.print(f"P50:    {report.retrieval_latency.p50_ms:.2f} ms")
    c.print(f"P95:    {report.retrieval_latency.p95_ms:.2f} ms")
    c.print(f"P99:    {report.retrieval_latency.p99_ms:.2f} ms\n")

    fail_style = "red" if report.failed_queries > 0 else "green"
    c.print(f"Completed:  [green]{report.completed_queries}[/green]")
    c.print(f"Failed:     [{fail_style}]{report.failed_queries}[/{fail_style}]")
    c.print(f"Total time: {total_elapsed:.2f}s")
