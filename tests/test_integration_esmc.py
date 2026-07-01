"""End-to-end integration test against a locally built ESM-C image.

Gated: runs only when ``PROTLMS_RUN_DOCKER_TESTS=1`` and a working Docker daemon is
available. Builds the ``esmc_300m`` image if it is not already present, then
drives the real ``protlms`` client through embed, likelihood, and score on a small
FASTA / variants CSV of real protein sequences.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import protlms

IMAGE = "ghcr.io/briney/protlms-esm-c:300m"
EMBEDDING_DIM = 960
REPO_ROOT = Path(__file__).parents[1]
TINY_FASTA = REPO_ROOT / "tests" / "data" / "tiny.fasta"
VARIANTS_CSV = REPO_ROOT / "tests" / "data" / "variants.csv"
EXPECTED_IDS = {"insulin_b", "gb1", "melittin"}


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def _gpu_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0


requires_gpu = pytest.mark.skipif(not _gpu_available(), reason="requires a GPU")


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PROTLMS_RUN_DOCKER_TESTS") != "1" or not _docker_available(),
        reason="set PROTLMS_RUN_DOCKER_TESTS=1 and ensure a Docker daemon is available",
    ),
]


@pytest.fixture(scope="session")
def esmc_image() -> str:
    """Ensure the 300M ESM-C image exists, building it if necessary."""
    present = (
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    )
    if not present:
        subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                "ESMC_CHECKPOINT=esmc_300m",
                "-t",
                IMAGE,
                str(REPO_ROOT / "containers" / "esm-c"),
            ],
            check=True,
        )
    return IMAGE


@pytest.fixture(scope="session")
def model(esmc_image: str) -> protlms.Model:
    return protlms.load("esm-c-300m", allow_pull=False)


def test_manifest_is_read_through_client(model: protlms.Model) -> None:
    assert model.manifest.name == "esmc_300m"
    assert model.manifest.embedding_dim == EMBEDDING_DIM
    assert model.manifest.num_layers == 30
    assert model.manifest.contract_version == "0.4"
    capabilities = {c.value for c in model.manifest.capabilities}
    assert {"embed", "likelihood", "score", "contacts"} <= capabilities


def test_embed_pooled_end_to_end(model: protlms.Model, tmp_path: Path) -> None:
    result = model.embed(TINY_FASTA, pooling="mean", output_dir=tmp_path / "emb")
    pooled = result.pooled()
    assert set(pooled) == EXPECTED_IDS
    for vector in pooled.values():
        assert vector.shape == (EMBEDDING_DIM,)
        assert vector.dtype == np.float32
        assert np.isfinite(vector).all()


def test_embed_per_residue_end_to_end(model: protlms.Model, tmp_path: Path) -> None:
    result = model.embed(TINY_FASTA, pooling="none", output_dir=tmp_path / "pr")
    per_residue = result.per_residue()
    assert set(per_residue) == EXPECTED_IDS
    # melittin is 26 residues long
    assert per_residue["melittin"].shape == (26, EMBEDDING_DIM)


def test_likelihood_end_to_end(model: protlms.Model, tmp_path: Path) -> None:
    result = model.likelihood(TINY_FASTA, output_dir=tmp_path / "ll")
    rows = {row["record_id"]: row for row in result.rows()}
    assert set(rows) == EXPECTED_IDS
    for row in rows.values():
        assert row["perplexity"] > 1.0
        assert math.isfinite(float(row["log_likelihood"]))
        assert row["seq_len"] > 0
    assert result.result.params["likelihood_method"] == "masked_marginal"


def test_score_masked_marginal_end_to_end(model: protlms.Model, tmp_path: Path) -> None:
    result = model.score(VARIANTS_CSV, method="masked-marginal", output_dir=tmp_path / "sc")
    rows = {r["variant_id"]: r for r in result.rows()}
    assert set(rows) == {"self", "single", "double"}
    assert rows["self"]["score"] == pytest.approx(0.0, abs=1e-5)
    assert rows["self"]["n_mutations"] == 1
    assert rows["double"]["n_mutations"] == 2
    assert math.isfinite(float(rows["single"]["score"]))


@requires_gpu
def test_contacts_end_to_end_shapes(model: protlms.Model, tmp_path: Path) -> None:
    result = model.contacts(TINY_FASTA, output_dir=tmp_path / "ct", use_gpu=True)
    maps = result.maps()
    assert set(maps) == EXPECTED_IDS
    for cmap in maps.values():
        n = cmap.shape[0]
        assert cmap.shape == (n, n)
        assert cmap.dtype == np.float32
        assert np.isfinite(cmap).all()
        assert np.allclose(cmap, cmap.T, atol=1e-4)  # symmetric


@requires_gpu
def test_evaluate_contacts_casp14_target(model: protlms.Model, tmp_path: Path) -> None:
    from protlms.eval.runner import evaluate_contacts, mean_precision

    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    src = REPO_ROOT / "tests" / "data" / "casp14" / "T1024.pdb"
    (pdb_dir / "T1024.pdb").write_bytes(src.read_bytes())
    results = evaluate_contacts(model, pdb_dir, use_gpu=True, max_length=400)
    assert len(results) == 1
    r = results[0]
    assert r.target_id == "T1024"
    assert 0.0 <= r.precision_at_l <= 1.0
    assert not math.isnan(mean_precision(results))
