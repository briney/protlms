"""End-to-end integration test against a locally built ProtBERT image.

Gated: runs only when ``PROTLMS_RUN_DOCKER_TESTS=1`` and a working Docker daemon is
available. Builds the ``prot_bert`` (UniRef100) image if it is not already present,
then drives the real ``protlms`` client through embed, likelihood, and score on a
small FASTA / variants CSV of real protein sequences.
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

IMAGE = "ghcr.io/briney/protlms-protbert:uniref100"
EMBEDDING_DIM = 1024
REPO_ROOT = Path(__file__).parents[1]
TINY_FASTA = REPO_ROOT / "tests" / "data" / "tiny.fasta"
VARIANTS_CSV = REPO_ROOT / "tests" / "data" / "variants.csv"
EXPECTED_IDS = {"insulin_b", "gb1", "melittin"}


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PROTLMS_RUN_DOCKER_TESTS") != "1" or not _docker_available(),
        reason="set PROTLMS_RUN_DOCKER_TESTS=1 and ensure a Docker daemon is available",
    ),
]


@pytest.fixture(scope="session")
def protbert_image() -> str:
    """Ensure the UniRef100 ProtBERT image exists, building it if necessary."""
    present = (
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    )
    if not present:
        subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                "PROTBERT_CHECKPOINT=prot_bert",
                "-t",
                IMAGE,
                str(REPO_ROOT / "containers" / "protbert"),
            ],
            check=True,
        )
    return IMAGE


@pytest.fixture(scope="session")
def model(protbert_image: str) -> protlms.Model:
    return protlms.load("protbert", allow_pull=False)


def test_manifest_is_read_through_client(model: protlms.Model) -> None:
    assert model.manifest.name == "prot_bert"
    assert model.manifest.embedding_dim == EMBEDDING_DIM
    capabilities = {c.value for c in model.manifest.capabilities}
    assert {"embed", "likelihood", "score"} <= capabilities


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
