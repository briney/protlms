"""Tests for the plms command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from plms.cli import app
from plms.contract import Manifest, Result
from plms.exceptions import ModelNotFoundError
from plms.models import EmbeddingResult, LikelihoodResult, ScoreResult

runner = CliRunner()


def _manifest() -> Manifest:
    return Manifest(
        contract_version="0.1",
        name="esm2_t6_8M",
        version="1.0.0",
        description="ESM2 8M.",
        model_family="esm2",
        capabilities=["embed", "likelihood"],
        embedding_dim=320,
        max_sequence_length=1024,
        pooling_modes=["mean", "cls", "none"],
        num_layers=6,
        default_batch_size=8,
    )


def _result(capability: str, artifacts: list[dict]) -> Result:
    return Result(
        contract_version="0.1",
        capability=capability,
        model_name="esm2_t6_8M",
        n_input_records=2,
        n_output_records=2,
        artifacts=artifacts,
    )


class FakeModel:
    last_call: dict = {}

    def __init__(self) -> None:
        self.manifest = _manifest()

    def embed(self, fasta, *, pooling, layers, output_dir, use_gpu, batch_size):  # noqa: ANN001
        FakeModel.last_call = {
            "method": "embed",
            "pooling": pooling,
            "layers": list(layers),
            "use_gpu": use_gpu,
            "output_dir": output_dir,
        }
        return EmbeddingResult(
            result=_result("embed", [{"path": "embeddings.npz", "kind": "pooled_embeddings"}]),
            output_dir=Path(output_dir),
            pooling=pooling,
        )

    def likelihood(self, fasta, *, output_dir, use_gpu, batch_size):  # noqa: ANN001
        FakeModel.last_call = {"method": "likelihood", "use_gpu": use_gpu}
        return LikelihoodResult(
            result=_result("likelihood", [{"path": "likelihoods.csv", "kind": "likelihoods_csv"}]),
            output_dir=Path(output_dir),
        )

    def score(self, variants, *, method, output_dir, use_gpu, batch_size):  # noqa: ANN001
        FakeModel.last_call = {"method": "score", "scoring_method": method, "use_gpu": use_gpu}
        return ScoreResult(
            result=_result("score", [{"path": "scores.csv", "kind": "variant_scores_csv"}]),
            output_dir=Path(output_dir),
            method=method,
        )


@pytest.fixture
def fasta(tmp_path: Path) -> Path:
    path = tmp_path / "seqs.fasta"
    path.write_text(">seq1\nACDEF\n>seq2\nGHIKL\n")
    return path


@pytest.fixture
def variants_csv(tmp_path: Path) -> Path:
    path = tmp_path / "variants.csv"
    path.write_text("variant_id,wt_sequence,mutant\nv1,ACDE,A1G\n")
    return path


def test_models_list_shows_registered_models() -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "esm2-8m" in result.stdout


def test_embed_command_invokes_model(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("plms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app, ["embed", "esm2-8m", str(fasta), "-o", str(tmp_path / "out"), "--pooling", "mean"]
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["method"] == "embed"
    assert FakeModel.last_call["pooling"] == "mean"


def test_embed_command_parses_layers_and_gpu(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("plms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app,
        ["embed", "esm2-8m", str(fasta), "-o", str(tmp_path / "out"), "--layers", "-1,6", "--gpu"],
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["layers"] == [-1, 6]
    assert FakeModel.last_call["use_gpu"] is True


def test_likelihood_command_invokes_model(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("plms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(app, ["likelihood", "esm2-8m", str(fasta), "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["method"] == "likelihood"


def test_score_command_invokes_model(variants_csv: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("plms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app,
        [
            "score",
            "esm2-8m",
            str(variants_csv),
            "-o",
            str(tmp_path / "out"),
            "--method",
            "wt-marginal",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["method"] == "score"
    assert FakeModel.last_call["scoring_method"] == "wt-marginal"


def test_plms_error_reported_cleanly_with_exit_1(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    def boom(name, **kw):  # noqa: ANN001, ANN003
        raise ModelNotFoundError("unknown model 'nope'")

    monkeypatch.setattr("plms.cli.load", boom)
    result = runner.invoke(app, ["embed", "nope", str(fasta), "-o", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "unknown model" in result.stdout
    # a clean message, not a raw traceback
    assert "Traceback" not in result.stdout
