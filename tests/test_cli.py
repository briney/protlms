"""Tests for the protlms command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from protlms.cli import app
from protlms.contract import Manifest, Result
from protlms.exceptions import ModelNotFoundError
from protlms.models import (
    ContactsResult,
    EmbeddingResult,
    GenerationResult,
    LikelihoodResult,
    ScoreResult,
)
from protlms.registry import Registry

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

    def embed(self, fasta, *, pooling, layers, output_dir, use_gpu, batch_size, chunk_size):  # noqa: ANN001
        FakeModel.last_call = {
            "method": "embed",
            "pooling": pooling,
            "layers": list(layers),
            "use_gpu": use_gpu,
            "output_dir": output_dir,
            "chunk_size": chunk_size,
        }
        return EmbeddingResult(
            result=_result("embed", [{"path": "embeddings.npz", "kind": "pooled_embeddings"}]),
            output_dir=Path(output_dir),
            pooling=pooling,
        )

    def likelihood(self, fasta, *, output_dir, use_gpu, batch_size, chunk_size):  # noqa: ANN001
        FakeModel.last_call = {"method": "likelihood", "use_gpu": use_gpu, "chunk_size": chunk_size}
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

    def contacts(self, fasta, *, method, output_dir, use_gpu, batch_size):  # noqa: ANN001
        FakeModel.last_call = {"method": "contacts", "contacts_method": method, "use_gpu": use_gpu}
        return ContactsResult(
            result=_result("contacts", [{"path": "contacts/seq1.npy", "kind": "contact_map"}]),
            output_dir=Path(output_dir),
            method=method,
        )

    def generate(
        self,
        prompts,
        *,
        num_samples,
        temperature,
        top_p,
        max_length,
        seed,
        output_dir,
        use_gpu,
        batch_size,
        chunk_size,
    ):  # noqa: ANN001
        FakeModel.last_call = {
            "method": "generate",
            "num_samples": num_samples,
            "temperature": temperature,
            "top_p": top_p,
            "max_length": max_length,
            "seed": seed,
            "use_gpu": use_gpu,
            "chunk_size": chunk_size,
        }
        return GenerationResult(
            result=_result("generate", [{"path": "generated.fasta", "kind": "generated_fasta"}]),
            output_dir=Path(output_dir),
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


@pytest.fixture
def prompts(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.fasta"
    path.write_text(">p1\nACDE\n")
    return path


def test_models_list_shows_registered_models() -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "esm2-8m" in result.stdout


def test_embed_command_invokes_model(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app, ["embed", "esm2-8m", str(fasta), "-o", str(tmp_path / "out"), "--pooling", "mean"]
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["method"] == "embed"
    assert FakeModel.last_call["pooling"] == "mean"


def test_embed_command_parses_layers_and_gpu(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app,
        ["embed", "esm2-8m", str(fasta), "-o", str(tmp_path / "out"), "--layers", "-1,6", "--gpu"],
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["layers"] == [-1, 6]
    assert FakeModel.last_call["use_gpu"] is True


def test_likelihood_command_invokes_model(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(app, ["likelihood", "esm2-8m", str(fasta), "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["method"] == "likelihood"


def test_score_command_invokes_model(variants_csv: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
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


def test_protlms_error_reported_cleanly_with_exit_1(
    fasta: Path, tmp_path: Path, monkeypatch
) -> None:
    def boom(name, **kw):  # noqa: ANN001, ANN003
        raise ModelNotFoundError("unknown model 'nope'")

    monkeypatch.setattr("protlms.cli.load", boom)
    result = runner.invoke(app, ["embed", "nope", str(fasta), "-o", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "unknown model" in result.stdout
    # a clean message, not a raw traceback
    assert "Traceback" not in result.stdout


def test_score_protlms_error_reported_cleanly(
    variants_csv: Path, tmp_path: Path, monkeypatch
) -> None:
    def boom(name, **kw):  # noqa: ANN001, ANN003
        raise ModelNotFoundError("unknown model 'nope'")

    monkeypatch.setattr("protlms.cli.load", boom)
    result = runner.invoke(app, ["score", "nope", str(variants_csv), "-o", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "unknown model" in result.stdout
    assert "Traceback" not in result.stdout


def test_score_command_default_method(variants_csv: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app, ["score", "esm2-8m", str(variants_csv), "-o", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["scoring_method"] == "masked-marginal"


def test_contacts_command_invokes_model(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(app, ["contacts", "esm2-8m", str(fasta), "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["method"] == "contacts"
    assert FakeModel.last_call["contacts_method"] == "categorical-jacobian"


def test_generate_command_invokes_model(prompts: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app,
        [
            "generate",
            "progen2-small",
            str(prompts),
            "-o",
            str(tmp_path / "out"),
            "--num-samples",
            "4",
            "--seed",
            "42",
            "--temperature",
            "0.7",
            "--top-p",
            "0.95",
            "--gpu",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["method"] == "generate"
    assert FakeModel.last_call["num_samples"] == 4
    assert FakeModel.last_call["seed"] == 42
    assert FakeModel.last_call["temperature"] == 0.7
    assert FakeModel.last_call["top_p"] == 0.95
    assert FakeModel.last_call["use_gpu"] is True


def test_embed_command_forwards_chunk_size(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app,
        ["embed", "esm2-8m", str(fasta), "-o", str(tmp_path / "out"), "--chunk-size", "1000"],
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["chunk_size"] == 1000


def test_embed_command_chunk_size_defaults_none(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(app, ["embed", "esm2-8m", str(fasta), "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["chunk_size"] is None


def test_likelihood_command_forwards_chunk_size(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app,
        ["likelihood", "esm2-8m", str(fasta), "-o", str(tmp_path / "out"), "--chunk-size", "500"],
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["chunk_size"] == 500


def test_generate_command_forwards_chunk_size(prompts: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app,
        ["generate", "progen2-small", str(prompts), "-o", str(tmp_path / "o"), "--chunk-size", "8"],
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["chunk_size"] == 8


def test_pull_command_pulls_resolved_model(monkeypatch) -> None:
    calls: list[tuple[str, bool, str]] = []
    monkeypatch.setattr("protlms.cli.SubprocessDockerRunner", lambda: object())
    monkeypatch.setattr(
        "protlms.cli.ensure_image",
        lambda runner, ref, *, allow_pull, model_name: calls.append((ref, allow_pull, model_name)),
    )
    result = runner.invoke(app, ["pull", "esm2-8m"])
    assert result.exit_code == 0, result.output
    assert calls and calls[0][1] is True and calls[0][2] == "esm2-8m"


def test_pull_all_pulls_every_model(monkeypatch) -> None:
    pulled: list[str] = []
    monkeypatch.setattr("protlms.cli.SubprocessDockerRunner", lambda: object())
    monkeypatch.setattr(
        "protlms.cli.ensure_image",
        lambda runner, ref, *, allow_pull, model_name: pulled.append(model_name),
    )
    result = runner.invoke(app, ["pull", "--all"])
    assert result.exit_code == 0, result.output
    assert len(pulled) == len(Registry.load().list_models())


def test_pull_without_model_or_all_errors() -> None:
    result = runner.invoke(app, ["pull"])
    assert result.exit_code == 1


def test_embed_no_pull_threads_allow_pull_false(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_load(name, **kw):  # noqa: ANN001, ANN003
        captured.update(kw)
        return FakeModel()

    monkeypatch.setattr("protlms.cli.load", fake_load)
    result = runner.invoke(
        app, ["embed", "esm2-8m", str(fasta), "-o", str(tmp_path / "out"), "--no-pull"]
    )
    assert result.exit_code == 0, result.output
    assert captured.get("allow_pull") is False
