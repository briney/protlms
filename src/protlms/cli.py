"""Command-line interface for protlms."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from protlms import __version__
from protlms.exceptions import InvalidRequestError, ProtlmsError
from protlms.models import load
from protlms.registry import Registry
from protlms.runner import SubprocessDockerRunner, ensure_image

app = typer.Typer(
    name="protlms",
    help="Unified toolkit for inference across a variety of protein language models (pLMs).",
    no_args_is_help=True,
    add_completion=False,
)

models_app = typer.Typer(
    help="Inspect the models available to protlms.",
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
_ChunkSizeOpt = Annotated[
    int | None,
    typer.Option(
        "--chunk-size",
        help="Split the input into runs of at most N records (merged; resumable).",
    ),
]
_NoPullOpt = Annotated[
    bool,
    typer.Option("--no-pull", help="Do not pull the image if it is missing locally."),
]


@app.callback()
def _main() -> None:
    """Unified toolkit for inference across protein language models (pLMs)."""
    # Present so Typer treats `protlms` as a command group with subcommands,
    # rather than collapsing the single command into the top level.


@app.command()
def version() -> None:
    """Print the installed protlms version."""
    typer.echo(__version__)


@models_app.command("list")
def models_list() -> None:
    """List the models registered with protlms."""
    table = Table(title="protlms models")
    table.add_column("name", style="bold")
    table.add_column("aliases")
    table.add_column("family")
    table.add_column("image")
    for entry in Registry.load().list_models():
        table.add_row(entry.name, ", ".join(entry.aliases), entry.model_family, entry.image)
    console.print(table)


@app.command()
def pull(
    model: Annotated[str | None, typer.Argument(help="Model name or alias.")] = None,
    all_models: Annotated[bool, typer.Option("--all", help="Pull every registered model.")] = False,
) -> None:
    """Pull a model's container image from its registry (digest-pinned when set)."""
    registry = Registry.load()
    try:
        if all_models:
            entries = registry.list_models()
        elif model is not None:
            entries = [registry.resolve(model)]
        else:
            raise InvalidRequestError("provide a model name or --all")
        docker_runner = SubprocessDockerRunner()
        for entry in entries:
            ref = entry.pinned_ref()
            console.print(f"pulling [bold]{entry.name}[/bold] ({ref}) …")
            ensure_image(docker_runner, ref, allow_pull=True, model_name=entry.name)
            console.print(f"  [green]ok[/green] {entry.name}")
    except ProtlmsError as exc:
        _fail(exc)


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
    chunk_size: _ChunkSizeOpt = None,
    no_pull: _NoPullOpt = False,
) -> None:
    """Compute embeddings for sequences in a FASTA file."""
    try:
        model_obj = load(model, allow_pull=False if no_pull else None)
        result = model_obj.embed(
            fasta,
            pooling=pooling,
            layers=_parse_layers(layers),
            output_dir=output_dir,
            use_gpu=gpu,
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
        console.print(
            f"[green]embed[/green] complete: {result.result.n_output_records} record(s) "
            f"written to [bold]{output_dir}[/bold]"
        )
        console.print(
            f"  model={model_obj.manifest.name}  "
            f"embedding_dim={model_obj.manifest.embedding_dim}  pooling={pooling}"
        )
    except ProtlmsError as exc:
        _fail(exc)


@app.command()
def likelihood(
    model: _ModelArg,
    fasta: _FastaArg,
    output_dir: _OutputOpt,
    gpu: _GpuOpt = False,
    batch_size: _BatchOpt = None,
    chunk_size: _ChunkSizeOpt = None,
    no_pull: _NoPullOpt = False,
) -> None:
    """Compute per-sequence log-likelihoods for sequences in a FASTA file."""
    try:
        model_obj = load(model, allow_pull=False if no_pull else None)
        result = model_obj.likelihood(
            fasta, output_dir=output_dir, use_gpu=gpu, batch_size=batch_size, chunk_size=chunk_size
        )
        console.print(
            f"[green]likelihood[/green] complete: {result.result.n_output_records} record(s) "
            f"written to [bold]{output_dir}[/bold]"
        )
    except ProtlmsError as exc:
        _fail(exc)


@app.command()
def score(
    model: _ModelArg,
    variants: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Variants CSV.")
    ],
    output_dir: _OutputOpt,
    method: Annotated[
        str, typer.Option("--method", help="Scoring method: masked-marginal or wt-marginal.")
    ] = "masked-marginal",
    gpu: _GpuOpt = False,
    batch_size: _BatchOpt = None,
) -> None:
    """Score sequence variants for effect."""
    try:
        model_obj = load(model)
        result = model_obj.score(
            variants, method=method, output_dir=output_dir, use_gpu=gpu, batch_size=batch_size
        )
        console.print(
            f"[green]score[/green] complete: {result.result.n_output_records} variant(s) "
            f"written to [bold]{output_dir}[/bold] method={method}"
        )
    except ProtlmsError as exc:
        _fail(exc)


@app.command()
def generate(
    model: _ModelArg,
    prompts: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Prompts FASTA (empty sequence = unconditional).",
        ),
    ],
    output_dir: _OutputOpt,
    num_samples: Annotated[int, typer.Option("--num-samples", help="Samples per prompt.")] = 1,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature.")
    ] = 1.0,
    top_p: Annotated[float, typer.Option("--top-p", help="Nucleus sampling probability.")] = 1.0,
    max_length: Annotated[
        int | None, typer.Option("--max-length", help="Max sequence length.")
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Random seed for reproducibility.")
    ] = None,
    gpu: _GpuOpt = False,
    batch_size: _BatchOpt = None,
    chunk_size: _ChunkSizeOpt = None,
    no_pull: _NoPullOpt = False,
) -> None:
    """Generate sequences with an autoregressive model."""
    try:
        model_obj = load(model, allow_pull=False if no_pull else None)
        result = model_obj.generate(
            prompts,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            seed=seed,
            output_dir=output_dir,
            use_gpu=gpu,
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
        console.print(
            f"[green]generate[/green] complete: {result.result.n_output_records} sequence(s) "
            f"written to [bold]{output_dir}[/bold]"
        )
    except ProtlmsError as exc:
        _fail(exc)


def _parse_layers(value: str) -> list[int]:
    """Parse a comma-separated layer-index string into a list of ints."""
    try:
        return [int(token) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise InvalidRequestError(f"invalid --layers value {value!r}: {exc}") from exc


def _fail(exc: ProtlmsError) -> None:
    """Print a clean error message and exit with status 1."""
    console.print(f"[bold red]Error:[/bold red] {exc}")
    raise typer.Exit(1)


def main() -> None:
    """Entry point for the ``protlms`` console script."""
    app()


if __name__ == "__main__":
    main()
