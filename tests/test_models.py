"""Tests for the Model integration layer, using a fake container runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plms.exceptions import (
    CapabilityNotSupportedError,
    ContainerExecutionError,
    ContractVersionError,
    InvalidRequestError,
)
from plms.io import read_fasta
from plms.models import EmbeddingResult, LikelihoodResult, Model, load
from plms.runner import RunResult, RunSpec

EMBEDDING_DIM = 320


def _manifest_json(**overrides) -> str:
    data = {
        "contract_version": "0.1",
        "name": "esm2_t6_8M",
        "version": "1.0.0",
        "description": "ESM2 8M.",
        "model_family": "esm2",
        "capabilities": ["embed", "likelihood", "score"],
        "embedding_dim": EMBEDDING_DIM,
        "max_sequence_length": 1024,
        "pooling_modes": ["mean", "cls", "none"],
        "num_layers": 6,
        "min_gpu_memory_gb": None,
        "default_batch_size": 8,
    }
    data.update(overrides)
    return json.dumps(data)


class FakeRunner:
    """A Runner that simulates a contract-compliant container in-process."""

    def __init__(self, manifest_json: str, *, behavior: str = "success") -> None:
        self.manifest_json = manifest_json
        self.behavior = behavior
        self.last_spec: RunSpec | None = None

    def manifest(self, image: str) -> str:
        return self.manifest_json

    def run(self, spec: RunSpec) -> RunResult:
        self.last_spec = spec
        argv = ["docker", "run", spec.image, *spec.command]
        if self.behavior == "error":
            err = json.dumps(
                {
                    "contract_version": "0.1",
                    "error_type": "SequenceTooLong",
                    "message": "sequence exceeds max length",
                    "details": {"id": "seq1"},
                }
            )
            return RunResult(exit_code=1, stdout="", stderr=f"loading...\n{err}", argv=argv)
        self._write_outputs(spec)
        return RunResult(exit_code=0, stdout="", stderr="", argv=argv)

    def _write_outputs(self, spec: RunSpec) -> None:
        out = spec.output_dir
        capability = spec.command[0]
        if capability == "embed":
            records = read_fasta(spec.input_dir / "seqs.fasta")
            pooling = spec.command[spec.command.index("--pooling") + 1]
            self._write_embed(out, records, pooling)
        elif capability == "likelihood":
            records = read_fasta(spec.input_dir / "seqs.fasta")
            self._write_likelihood(out, records)
        elif capability == "score":
            self._write_score(out, spec.input_dir / "variants.csv")

    def _write_embed(self, out: Path, records, pooling: str) -> None:  # noqa: ANN001
        artifacts = []
        if pooling == "none":
            pr_dir = out / "per_residue"
            pr_dir.mkdir()
            for rec in records:
                arr = np.ones((len(rec.sequence), EMBEDDING_DIM), dtype=np.float32)
                np.save(pr_dir / f"{rec.id}.npy", arr)
                artifacts.append(
                    {"path": f"per_residue/{rec.id}.npy", "kind": "per_residue_embeddings"}
                )
        else:
            vectors = {rec.id: np.ones(EMBEDDING_DIM, dtype=np.float32) for rec in records}
            np.savez(out / "embeddings.npz", **vectors)
            artifacts.append({"path": "embeddings.npz", "kind": "pooled_embeddings"})
        self._write_result(out, "embed", records, artifacts)

    def _write_likelihood(self, out: Path, records) -> None:  # noqa: ANN001
        lines = [
            "record_id,seq_len,pseudo_log_likelihood,mean_pseudo_log_likelihood,pseudo_perplexity"
        ]
        for rec in records:
            lines.append(f"{rec.id},{len(rec.sequence)},-3.5,-0.7,2.01")
        (out / "likelihoods.csv").write_text("\n".join(lines) + "\n")
        self._write_result(
            out, "likelihood", records, [{"path": "likelihoods.csv", "kind": "likelihoods_csv"}]
        )

    def _write_score(self, out: Path, variants_csv: Path) -> None:
        import csv as _csv

        with variants_csv.open(newline="") as handle:
            rows = list(_csv.DictReader(handle))
        lines = ["variant_id,mutant,n_mutations,score"]
        for r in rows:
            n = len(r["mutant"].split(":"))
            lines.append(f"{r['variant_id']},{r['mutant']},{n},-1.5")
        (out / "scores.csv").write_text("\n".join(lines) + "\n")
        (out / "result.json").write_text(
            json.dumps(
                {
                    "contract_version": "0.2",
                    "capability": "score",
                    "model_name": "esm2_t6_8M",
                    "n_input_records": len(rows),
                    "n_output_records": len(rows),
                    "artifacts": [{"path": "scores.csv", "kind": "variant_scores_csv"}],
                }
            )
        )

    def _write_result(self, out: Path, capability: str, records, artifacts) -> None:  # noqa: ANN001
        (out / "result.json").write_text(
            json.dumps(
                {
                    "contract_version": "0.1",
                    "capability": capability,
                    "model_name": "esm2_t6_8M",
                    "n_input_records": len(records),
                    "n_output_records": len(records),
                    "artifacts": artifacts,
                }
            )
        )


@pytest.fixture
def fasta(tmp_path: Path) -> Path:
    path = tmp_path / "seqs.fasta"
    path.write_text(">seq1\nACDEFGHIK\n>seq2\nLMNPQRST\n")
    return path


def _load(behavior: str = "success", **manifest_overrides) -> Model:
    runner = FakeRunner(_manifest_json(**manifest_overrides), behavior=behavior)
    return load("esm2-8m", runner=runner)


def test_load_returns_model_with_validated_manifest() -> None:
    model = _load()
    assert isinstance(model, Model)
    assert model.manifest.name == "esm2_t6_8M"
    assert model.manifest.embedding_dim == EMBEDDING_DIM


def test_load_rejects_incompatible_contract_major() -> None:
    runner = FakeRunner(_manifest_json(contract_version="1.0"))
    with pytest.raises(ContractVersionError):
        load("esm2-8m", runner=runner)


def test_embed_pooled_returns_vectors_keyed_by_id(fasta: Path, tmp_path: Path) -> None:
    model = _load()
    result = model.embed(fasta, pooling="mean", output_dir=tmp_path / "out")
    assert isinstance(result, EmbeddingResult)
    pooled = result.pooled()
    assert set(pooled) == {"seq1", "seq2"}
    assert pooled["seq1"].shape == (EMBEDDING_DIM,)


def test_embed_none_returns_per_residue(fasta: Path, tmp_path: Path) -> None:
    model = _load()
    result = model.embed(fasta, pooling="none", output_dir=tmp_path / "out")
    per_residue = result.per_residue()
    assert per_residue["seq1"].shape == (9, EMBEDDING_DIM)  # len("ACDEFGHIK") == 9


def test_embed_builds_expected_command(fasta: Path, tmp_path: Path) -> None:
    model = _load()
    model.embed(fasta, pooling="mean", layers=(-1,), output_dir=tmp_path / "out")
    cmd = model._runner.last_spec.command  # type: ignore[attr-defined]
    assert cmd[0] == "embed"
    assert cmd[cmd.index("--pooling") + 1] == "mean"
    assert cmd[cmd.index("--layers") + 1] == "-1"
    assert cmd[cmd.index("--input") + 1] == "/in/seqs.fasta"
    assert cmd[cmd.index("--output") + 1] == "/out"


def test_embed_passes_gpu_flag(fasta: Path, tmp_path: Path) -> None:
    model = _load()
    model.embed(fasta, pooling="mean", use_gpu=True, output_dir=tmp_path / "out")
    assert model._runner.last_spec.use_gpu is True  # type: ignore[attr-defined]


def test_embed_invalid_pooling_raises_before_run(fasta: Path, tmp_path: Path) -> None:
    model = _load()
    with pytest.raises(InvalidRequestError):
        model.embed(fasta, pooling="bogus", output_dir=tmp_path / "out")
    assert model._runner.last_spec is None  # type: ignore[attr-defined]


def test_embed_empty_fasta_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    model = _load()
    with pytest.raises(InvalidRequestError):
        model.embed(empty, output_dir=tmp_path / "out")


def test_likelihood_unsupported_capability_raises(fasta: Path, tmp_path: Path) -> None:
    model = _load(capabilities=["embed"])  # no likelihood
    with pytest.raises(CapabilityNotSupportedError):
        model.likelihood(fasta, output_dir=tmp_path / "out")


def test_likelihood_returns_rows(fasta: Path, tmp_path: Path) -> None:
    model = _load()
    result = model.likelihood(fasta, output_dir=tmp_path / "out")
    assert isinstance(result, LikelihoodResult)
    rows = result.rows()
    assert {r["record_id"] for r in rows} == {"seq1", "seq2"}
    assert rows[0]["pseudo_perplexity"] == pytest.approx(2.01)


def test_container_error_is_surfaced_with_structured_fields(fasta: Path, tmp_path: Path) -> None:
    model = _load(behavior="error")
    with pytest.raises(ContainerExecutionError) as excinfo:
        model.embed(fasta, pooling="mean", output_dir=tmp_path / "out")
    assert excinfo.value.error_type == "SequenceTooLong"
    assert excinfo.value.exit_code == 1


def test_embed_resolves_relative_output_dir_to_absolute(
    fasta: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    model = _load()
    result = model.embed(fasta, pooling="mean", output_dir=Path("relout"))
    assert result.output_dir.is_absolute()
    assert set(result.pooled()) == {"seq1", "seq2"}


def test_embed_without_output_dir_keeps_results_available(fasta: Path) -> None:
    model = _load()
    result = model.embed(fasta, pooling="mean")  # no output_dir
    pooled = result.pooled()  # temp dir must still be alive
    assert set(pooled) == {"seq1", "seq2"}


@pytest.fixture
def variants_csv(tmp_path: Path) -> Path:
    path = tmp_path / "variants.csv"
    path.write_text("variant_id,wt_sequence,mutant\nself,ACDEFGHIK,A1A\nsingle,ACDEFGHIK,C2A\n")
    return path


def test_score_returns_rows(variants_csv: Path, tmp_path: Path) -> None:
    from plms.models import ScoreResult

    model = _load()
    result = model.score(variants_csv, output_dir=tmp_path / "sc")
    assert isinstance(result, ScoreResult)
    rows = {r["variant_id"]: r for r in result.rows()}
    assert set(rows) == {"self", "single"}
    assert rows["single"]["n_mutations"] == 1


def test_score_builds_expected_command(variants_csv: Path, tmp_path: Path) -> None:
    model = _load()
    model.score(variants_csv, method="wt-marginal", output_dir=tmp_path / "sc")
    cmd = model._runner.last_spec.command  # type: ignore[attr-defined]
    assert cmd[0] == "score"
    assert cmd[cmd.index("--input") + 1] == "/in/variants.csv"
    assert cmd[cmd.index("--method") + 1] == "wt-marginal"


def test_score_invalid_method_raises_before_run(variants_csv: Path, tmp_path: Path) -> None:
    model = _load()
    with pytest.raises(InvalidRequestError):
        model.score(variants_csv, method="bogus", output_dir=tmp_path / "sc")
    assert model._runner.last_spec is None  # type: ignore[attr-defined]


def test_score_missing_columns_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("variant_id,mutant\nv1,A1G\n")
    model = _load()
    with pytest.raises(InvalidRequestError):
        model.score(bad, output_dir=tmp_path / "sc")


def test_score_unsupported_capability_raises(variants_csv: Path, tmp_path: Path) -> None:
    model = _load(capabilities=["embed", "likelihood"])  # no score
    with pytest.raises(CapabilityNotSupportedError):
        model.score(variants_csv, output_dir=tmp_path / "sc")
