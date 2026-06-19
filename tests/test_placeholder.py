"""Smoke tests for the plms package scaffold."""

from __future__ import annotations

from typer.testing import CliRunner

import plms
from plms.cli import app

runner = CliRunner()


def test_version_is_defined() -> None:
    """The package exposes a non-empty version string."""
    assert isinstance(plms.__version__, str)
    assert plms.__version__


def test_cli_version_command() -> None:
    """``plms version`` prints the package version and exits cleanly."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert plms.__version__ in result.stdout
