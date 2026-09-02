"""CLI interface for RAGDiag."""

import time
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

import ragdiag
from ragdiag.dataset.exceptions import DatasetLoadError, DatasetValidationError
from ragdiag.dataset.loader import load_dataset
from ragdiag.dataset.validator import validate_dataset
from ragdiag.pipeline.exceptions import PipelineError
from ragdiag.pipeline.loader import load_pipeline
from ragdiag.runner.evaluator import Evaluator

app = typer.Typer(
    name="ragdiag",
    help="RAGDiag: RAG evaluation and root-cause diagnosis CLI.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    """Print the version of ragdiag and exit."""
    if value:
        console.print(
            f"[bold cyan]ragdiag[/bold cyan] version [green]{ragdiag.__version__}[/green]"
        )
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
    try:
        ds = load_dataset(dataset)
        summary = validate_dataset(ds)
    except (DatasetLoadError, DatasetValidationError) as err:
        console.print(
            Panel(
                f"[bold red]Validation Failed[/bold red]\n\n{err}",
                title="[bold red]Dataset Error[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from err
    except Exception as exc:
        console.print(
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

    console.print(
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
    try:
        loaded_pipeline = load_pipeline(pipeline)
    except PipelineError as exc:
        console.print(
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
        console.print(
            Panel(
                f"[bold red]Dataset Load Failed[/bold red]\n\n{exc}",
                title="[bold red]Dataset Error[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from exc

    console.print("[bold cyan]RAGDiag[/bold cyan]")
    console.print("-" * 24)
    console.print(f"Pipeline: [bold]{loaded_pipeline.name}[/bold]")
    console.print(f"Dataset:  [bold]{loaded_dataset.name}[/bold]")
    console.print(f"Queries:  [bold]{len(loaded_dataset.samples)}[/bold]\n")
    console.print("[dim]Running evaluation...[/dim]\n")

    evaluator = Evaluator()
    start_time = time.perf_counter()
    results = evaluator.evaluate(loaded_pipeline, loaded_dataset)
    total_elapsed = time.perf_counter() - start_time

    completed_count = sum(1 for r in results if r.status == "completed")
    failed_count = sum(1 for r in results if r.status == "failed")

    fail_style = "red" if failed_count > 0 else "green"

    console.print("[bold green]Evaluation complete.[/bold green]\n")
    console.print(f"Completed:  [green]{completed_count}[/green]")
    console.print(f"Failed:     [{fail_style}]{failed_count}[/{fail_style}]")
    console.print(f"Total time: {total_elapsed:.2f}s")
