"""Smoke tests for the protlms package scaffold."""

from __future__ import annotations

from typer.testing import CliRunner

import protlms
from protlms.cli import app

runner = CliRunner()


def test_version_is_defined() -> None:
    """The package exposes a non-empty version string."""
    assert isinstance(protlms.__version__, str)
    assert protlms.__version__


def test_cli_version_command() -> None:
    """``protlms version`` prints the package version and exits cleanly."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert protlms.__version__ in result.stdout
