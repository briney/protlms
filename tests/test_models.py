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
    ImageNotFoundError,
    InvalidRequestError,
)
from plms.io import read_fasta
from plms.models import EmbeddingResult, LikelihoodResult, Model, load
from plms.registry import ModelEntry, Registry
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

    def __init__(
        self, manifest_json: str, *, behavior: str = "success", present: bool = True
    ) -> None:
        self.manifest_json = manifest_json
        self.behavior = behavior
        self.present = present
        self.last_spec: RunSpec | None = None
        self.manifest_ref: str | None = None
        self.pulled: list[str] = []

    def manifest(self, image: str) -> str:
        self.manifest_ref = image
        return self.manifest_json

    def image_present(self, ref: str) -> bool:
        return self.present

    def pull(self, ref: str) -> None:
        self.pulled.append(ref)

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
        elif capability == "generate":
            records = read_fasta(spec.input_dir / "seqs.fasta")
            num_samples = int(spec.command[spec.command.index("--num-samples") + 1])
            self._write_generate(out, records, num_samples)

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
        lines = ["record_id,seq_len,log_likelihood,mean_log_likelihood,perplexity"]
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

    def _write_generate(self, out: Path, records, num_samples: int) -> None:  # noqa: ANN001
        lines = []
        out_ids = []
        for rec in records:
            for k in range(num_samples):
                rid = f"{rec.id}__sample{k}"
                out_ids.append(rid)
                lines.append(f">{rid}\nACDEFG\n")
        (out / "generated.fasta").write_text("".join(lines))
        self._write_result(
            out,
            "generate",
            records,
            [{"path": "generated.fasta", "kind": "generated_fasta", "record_ids": out_ids}],
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
    assert rows[0]["perplexity"] == pytest.approx(2.01)


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
    assert model._runner.last_spec is None  # type: ignore[attr-defined]


def test_score_unsupported_capability_raises(variants_csv: Path, tmp_path: Path) -> None:
    model = _load(capabilities=["embed", "likelihood"])  # no score
    with pytest.raises(CapabilityNotSupportedError):
        model.score(variants_csv, output_dir=tmp_path / "sc")


@pytest.fixture
def prompts(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.fasta"
    path.write_text(">p1\nACDE\n>uncond\n\n")  # second record is unconditional (empty)
    return path


def test_generate_returns_sequences(prompts: Path, tmp_path: Path) -> None:
    from plms.models import GenerationResult

    model = _load(capabilities=["embed", "likelihood", "generate"])
    result = model.generate(prompts, num_samples=2, output_dir=tmp_path / "gen")
    assert isinstance(result, GenerationResult)
    seqs = result.sequences()
    expected_ids = {"p1__sample0", "p1__sample1", "uncond__sample0", "uncond__sample1"}
    assert {r.id for r in seqs} == expected_ids


def test_generate_builds_expected_command(prompts: Path, tmp_path: Path) -> None:
    model = _load(capabilities=["generate"])
    model.generate(
        prompts,
        num_samples=3,
        temperature=0.8,
        top_p=0.9,
        seed=42,
        output_dir=tmp_path / "g",
    )
    cmd = model._runner.last_spec.command  # type: ignore[attr-defined]
    assert cmd[0] == "generate"
    assert cmd[cmd.index("--num-samples") + 1] == "3"
    assert cmd[cmd.index("--temperature") + 1] == "0.8"
    assert cmd[cmd.index("--top-p") + 1] == "0.9"
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert "--max-length" not in cmd  # omitted when None


def test_generate_unsupported_capability_raises(prompts: Path, tmp_path: Path) -> None:
    model = _load(capabilities=["embed", "likelihood"])
    with pytest.raises(CapabilityNotSupportedError):
        model.generate(prompts, output_dir=tmp_path / "g")


def test_generate_empty_prompts_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    model = _load(capabilities=["generate"])
    with pytest.raises(InvalidRequestError):
        model.generate(empty, output_dir=tmp_path / "g")


def test_embed_chunked_merges_all_records(tmp_path: Path) -> None:
    model = _load()
    fasta = tmp_path / "many.fasta"
    fasta.write_text("".join(f">s{i}\nACDEFG\n" for i in range(5)))
    result = model.embed(fasta, pooling="mean", output_dir=tmp_path / "out", chunk_size=2)
    pooled = result.pooled()
    assert set(pooled) == {f"s{i}" for i in range(5)}
    assert result.result.n_output_records == 5
    assert (tmp_path / "out" / "chunks" / "chunk_0000").is_dir()


def test_embed_chunk_size_none_keeps_single_run(fasta: Path, tmp_path: Path) -> None:
    model = _load()
    out = tmp_path / "out"
    model.embed(fasta, pooling="mean", output_dir=out, chunk_size=None)
    assert not (out / "chunks").exists()


def test_embed_single_chunk_short_circuits(fasta: Path, tmp_path: Path) -> None:
    model = _load()  # the `fasta` fixture has 2 records
    out = tmp_path / "out"
    model.embed(fasta, pooling="mean", output_dir=out, chunk_size=10)
    assert not (out / "chunks").exists()  # records <= chunk_size => single run


def test_likelihood_chunked_merges_rows(tmp_path: Path) -> None:
    model = _load()
    fasta = tmp_path / "many.fasta"
    fasta.write_text("".join(f">s{i}\nACDEFG\n" for i in range(4)))
    result = model.likelihood(fasta, output_dir=tmp_path / "out", chunk_size=2)
    rows = {r["record_id"] for r in result.rows()}
    assert rows == {f"s{i}" for i in range(4)}


def test_generate_chunked_merges_samples(tmp_path: Path) -> None:
    model = _load(capabilities=["embed", "likelihood", "generate"])
    prompts = tmp_path / "p.fasta"
    prompts.write_text("".join(f">p{i}\nAC\n" for i in range(3)))
    result = model.generate(prompts, num_samples=2, output_dir=tmp_path / "out", chunk_size=2)
    ids = {r.id for r in result.sequences()}
    assert ids == {f"p{i}__sample{k}" for i in range(3) for k in range(2)}


def _registry_with_digest() -> Registry:
    return Registry(
        [
            ModelEntry(
                name="esm2-8m",
                image="ghcr.io/briney/plms-esm2:t6_8M",
                digest="sha256:abc123",
                model_family="esm2",
            )
        ]
    )


def test_load_skips_pull_when_image_present() -> None:
    runner = FakeRunner(_manifest_json(), present=True)
    load("esm2-8m", runner=runner)
    assert runner.pulled == []


def test_load_pulls_pinned_ref_when_absent() -> None:
    runner = FakeRunner(_manifest_json(), present=False)
    load("esm2-8m", runner=runner, registry=_registry_with_digest())
    assert runner.pulled == ["ghcr.io/briney/plms-esm2@sha256:abc123"]


def test_load_runs_manifest_against_pinned_ref() -> None:
    runner = FakeRunner(_manifest_json(), present=True)
    load("esm2-8m", runner=runner, registry=_registry_with_digest())
    assert runner.manifest_ref == "ghcr.io/briney/plms-esm2@sha256:abc123"


def test_load_allow_pull_false_raises_when_absent() -> None:
    runner = FakeRunner(_manifest_json(), present=False)
    with pytest.raises(ImageNotFoundError):
        load("esm2-8m", runner=runner, allow_pull=False)


def test_load_env_no_pull_disables_pull(monkeypatch) -> None:
    monkeypatch.setenv("PLMS_NO_PULL", "1")
    runner = FakeRunner(_manifest_json(), present=False)
    with pytest.raises(ImageNotFoundError):
        load("esm2-8m", runner=runner)
