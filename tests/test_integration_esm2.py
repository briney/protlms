"""End-to-end integration test against a locally built ESM2 image.

Gated: runs only when ``PLMS_RUN_DOCKER_TESTS=1`` and a working Docker daemon is
available. Builds the tiny ``esm2_t6_8M`` image if it is not already present,
then drives the real ``plms`` client through embed and likelihood on a small
FASTA of real protein sequences.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import plms

IMAGE = "plms-esm2:t6_8M"
EMBEDDING_DIM = 320
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
        os.environ.get("PLMS_RUN_DOCKER_TESTS") != "1" or not _docker_available(),
        reason="set PLMS_RUN_DOCKER_TESTS=1 and ensure a Docker daemon is available",
    ),
]


@pytest.fixture(scope="session")
def esm2_image() -> str:
    """Ensure the tiny ESM2 image exists, building it if necessary."""
    present = (
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    )
    if not present:
        subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                "ESM2_CHECKPOINT=esm2_t6_8M",
                "-t",
                IMAGE,
                str(REPO_ROOT / "containers" / "esm2"),
            ],
            check=True,
        )
    return IMAGE


@pytest.fixture(scope="session")
def model(esm2_image: str) -> plms.Model:
    return plms.load("esm2-8m")


def test_manifest_is_read_through_client(model: plms.Model) -> None:
    assert model.manifest.name == "esm2_t6_8M"
    assert model.manifest.embedding_dim == EMBEDDING_DIM
    capabilities = {c.value for c in model.manifest.capabilities}
    assert {"embed", "likelihood"} <= capabilities


def test_embed_pooled_end_to_end(model: plms.Model, tmp_path: Path) -> None:
    result = model.embed(TINY_FASTA, pooling="mean", output_dir=tmp_path / "emb")
    pooled = result.pooled()
    assert set(pooled) == EXPECTED_IDS
    for vector in pooled.values():
        assert vector.shape == (EMBEDDING_DIM,)
        assert vector.dtype == np.float32
        assert np.isfinite(vector).all()


def test_embed_per_residue_end_to_end(model: plms.Model, tmp_path: Path) -> None:
    result = model.embed(TINY_FASTA, pooling="none", output_dir=tmp_path / "pr")
    per_residue = result.per_residue()
    assert set(per_residue) == EXPECTED_IDS
    # melittin is 26 residues long
    assert per_residue["melittin"].shape == (26, EMBEDDING_DIM)


def test_likelihood_end_to_end(model: plms.Model, tmp_path: Path) -> None:
    result = model.likelihood(TINY_FASTA, output_dir=tmp_path / "ll")
    rows = {row["record_id"]: row for row in result.rows()}
    assert set(rows) == EXPECTED_IDS
    for row in rows.values():
        assert row["pseudo_perplexity"] > 1.0
        assert math.isfinite(float(row["pseudo_log_likelihood"]))
        assert row["seq_len"] > 0


def test_score_masked_marginal_end_to_end(model: plms.Model, tmp_path: Path) -> None:
    result = model.score(VARIANTS_CSV, method="masked-marginal", output_dir=tmp_path / "sc")
    rows = {r["variant_id"]: r for r in result.rows()}
    assert set(rows) == {"self", "single", "double"}
    # a self-substitution must score exactly 0
    assert rows["self"]["score"] == pytest.approx(0.0, abs=1e-5)
    assert rows["self"]["n_mutations"] == 1
    assert rows["double"]["n_mutations"] == 2
    assert math.isfinite(float(rows["single"]["score"]))


def test_score_wt_marginal_runs(model: plms.Model, tmp_path: Path) -> None:
    result = model.score(VARIANTS_CSV, method="wt-marginal", output_dir=tmp_path / "sc")
    rows = {r["variant_id"]: r for r in result.rows()}
    assert set(rows) == {"self", "single", "double"}
    assert rows["self"]["score"] == pytest.approx(0.0, abs=1e-5)
    assert math.isfinite(float(rows["single"]["score"]))
    assert math.isfinite(float(rows["double"]["score"]))


def test_manifest_now_declares_score(model: plms.Model) -> None:
    assert "score" in {c.value for c in model.manifest.capabilities}
