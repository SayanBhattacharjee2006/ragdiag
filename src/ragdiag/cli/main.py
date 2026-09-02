"""CLI interface for RAGDiag."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

import ragdiag
from ragdiag.dataset.exceptions import DatasetLoadError, DatasetValidationError
from ragdiag.dataset.loader import load_dataset
from ragdiag.dataset.validator import validate_dataset

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
        str | None,
        typer.Option(
            "--pipeline",
            "-p",
            help="Path or module import for the Pipeline adapter.",
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Path to evaluation dataset (JSON/JSONL).",
        ),
    ] = None,
) -> None:
    """Run RAG evaluation and diagnosis against a pipeline."""
    console.print(
        Panel.fit(
            "[bold yellow]Evaluation Engine Not Implemented[/bold yellow]\n\n"
            "The evaluation and root-cause diagnosis engine is not implemented yet.\n"
            "This command will be implemented in subsequent development phases.",
            title="[bold cyan]RAGDiag[/bold cyan]",
            border_style="yellow",
        )
    )
