"""CLI interface for RAGDiag."""

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
) -> None:
    """Run RAG evaluation against a pipeline and dataset."""
    c = get_console()
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

    evaluator = Evaluator()
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
