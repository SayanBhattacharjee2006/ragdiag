"""CLI interface for RAGDiag."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

import ragdiag

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
