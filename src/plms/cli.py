"""Command-line interface for plms."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from plms import __version__
from plms.exceptions import InvalidRequestError, PlmsError
from plms.models import load
from plms.registry import Registry

app = typer.Typer(
    name="plms",
    help="Unified toolkit for inference across a variety of protein language models (pLMs).",
    no_args_is_help=True,
    add_completion=False,
)

models_app = typer.Typer(
    help="Inspect the models available to plms.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")

console = Console()

# Reusable argument/option definitions (module-level to keep call sites tidy).
_ModelArg = Annotated[str, typer.Argument(help="Model name or alias.")]
_FastaArg = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Input FASTA file.")
]
_OutputOpt = Annotated[Path, typer.Option("-o", "--output", help="Output directory.")]
_GpuOpt = Annotated[bool, typer.Option("--gpu/--no-gpu", help="Run the container with all GPUs.")]
_BatchOpt = Annotated[int | None, typer.Option("--batch-size", help="Override the batch size.")]


@app.callback()
def _main() -> None:
    """Unified toolkit for inference across protein language models (pLMs)."""
    # Present so Typer treats `plms` as a command group with subcommands,
    # rather than collapsing the single command into the top level.


@app.command()
def version() -> None:
    """Print the installed plms version."""
    typer.echo(__version__)


@models_app.command("list")
def models_list() -> None:
    """List the models registered with plms."""
    table = Table(title="plms models")
    table.add_column("name", style="bold")
    table.add_column("aliases")
    table.add_column("family")
    table.add_column("image")
    for entry in Registry.load().list_models():
        table.add_row(entry.name, ", ".join(entry.aliases), entry.model_family, entry.image)
    console.print(table)


@app.command()
def embed(
    model: _ModelArg,
    fasta: _FastaArg,
    output_dir: _OutputOpt,
    pooling: Annotated[
        str, typer.Option("--pooling", help="Pooling mode: mean, cls, or none.")
    ] = "mean",
    layers: Annotated[str, typer.Option("--layers", help="Comma-separated layer indices.")] = "-1",
    gpu: _GpuOpt = False,
    batch_size: _BatchOpt = None,
) -> None:
    """Compute embeddings for sequences in a FASTA file."""
    try:
        model_obj = load(model)
        result = model_obj.embed(
            fasta,
            pooling=pooling,
            layers=_parse_layers(layers),
            output_dir=output_dir,
            use_gpu=gpu,
            batch_size=batch_size,
        )
        console.print(
            f"[green]embed[/green] complete: {result.result.n_output_records} record(s) "
            f"written to [bold]{output_dir}[/bold]"
        )
        console.print(
            f"  model={model_obj.manifest.name}  "
            f"embedding_dim={model_obj.manifest.embedding_dim}  pooling={pooling}"
        )
    except PlmsError as exc:
        _fail(exc)


@app.command()
def likelihood(
    model: _ModelArg,
    fasta: _FastaArg,
    output_dir: _OutputOpt,
    gpu: _GpuOpt = False,
    batch_size: _BatchOpt = None,
) -> None:
    """Compute pseudo-log-likelihoods for sequences in a FASTA file."""
    try:
        model_obj = load(model)
        result = model_obj.likelihood(
            fasta, output_dir=output_dir, use_gpu=gpu, batch_size=batch_size
        )
        console.print(
            f"[green]likelihood[/green] complete: {result.result.n_output_records} record(s) "
            f"written to [bold]{output_dir}[/bold]"
        )
    except PlmsError as exc:
        _fail(exc)


def _parse_layers(value: str) -> list[int]:
    """Parse a comma-separated layer-index string into a list of ints."""
    try:
        return [int(token) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise InvalidRequestError(f"invalid --layers value {value!r}: {exc}") from exc


def _fail(exc: PlmsError) -> None:
    """Print a clean error message and exit with status 1."""
    console.print(f"[bold red]Error:[/bold red] {exc}")
    raise typer.Exit(1)


def main() -> None:
    """Entry point for the ``plms`` console script."""
    app()


if __name__ == "__main__":
    main()
