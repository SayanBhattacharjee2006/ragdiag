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
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to save evaluation report as JSON.",
        ),
    ] = None,
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

    c.print("[dim]Running evaluation...[/dim]\n")

    evaluator = Evaluator(k=5, judge=judge_instance)
    start_time = time.perf_counter()
    results = evaluator.evaluate(loaded_pipeline, loaded_dataset)
    total_elapsed = time.perf_counter() - start_time

    from ragdiag.reporting import build_report, render_terminal_report

    report = build_report(
        results,
        dataset_name=loaded_dataset.name,
        dataset_version=loaded_dataset.version,
        pipeline_name=loaded_pipeline.name,
        k=5,
    )

    render_terminal_report(report, c, total_elapsed=total_elapsed)

    if output:
        try:
            json_str = report.model_dump_json(indent=2)
            with open(output, "w", encoding="utf-8") as f:
                f.write(json_str)
            c.print(f"\n[bold green]Report written to:[/bold green] {output}")
        except OSError as exc:
            c.print(f"\n[bold red]Failed to write output file '{output}':[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
