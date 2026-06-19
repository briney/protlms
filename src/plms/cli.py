"""Command-line interface for plms."""

from __future__ import annotations

import typer

from plms import __version__

app = typer.Typer(
    name="plms",
    help="Unified toolkit for inference across a variety of protein language models (pLMs).",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _main() -> None:
    """Unified toolkit for inference across protein language models (pLMs)."""
    # Present so Typer treats `plms` as a command group with subcommands,
    # rather than collapsing the single command into the top level.


@app.command()
def version() -> None:
    """Print the installed plms version."""
    typer.echo(__version__)


def main() -> None:
    """Entry point for the ``plms`` console script."""
    app()


if __name__ == "__main__":
    main()
