"""End-to-end test that chunked runs equal unchunked runs (real ESM image).

Gated: runs only when ``PROTLMS_RUN_DOCKER_TESTS=1`` and a working Docker daemon is
available. Builds the tiny ``esm2_t6_8M`` image if absent.
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

IMAGE = "ghcr.io/briney/protlms-esm:t6_8M"
REPO_ROOT = Path(__file__).parents[1]
TINY_FASTA = REPO_ROOT / "tests" / "data" / "tiny.fasta"


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
def esm_image() -> str:
    present = (
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    )
    if not present:
        subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                "ESM_HF_ID=facebook/esm2_t6_8M_UR50D",
                "--build-arg",
                "ESM_MODEL_NAME=esm2_t6_8M",
                "--build-arg",
                "ESM_MODEL_FAMILY=esm2",
                "-t",
                IMAGE,
                str(REPO_ROOT / "containers" / "esm"),
            ],
            check=True,
        )
    return IMAGE


@pytest.fixture(scope="session")
def model(esm_image: str) -> protlms.Model:
    return protlms.load("esm2-8m", allow_pull=False)


def test_embed_chunked_equals_unchunked(model: protlms.Model, tmp_path: Path) -> None:
    plain = model.embed(TINY_FASTA, pooling="mean", output_dir=tmp_path / "plain").pooled()
    chunked = model.embed(
        TINY_FASTA, pooling="mean", output_dir=tmp_path / "chunked", chunk_size=2
    ).pooled()
    assert set(chunked) == set(plain)
    for rid in plain:
        np.testing.assert_allclose(chunked[rid], plain[rid], atol=1e-5)
    assert (tmp_path / "chunked" / "chunks" / "chunk_0001").is_dir()  # 3 records / 2 => 2 chunks


def test_likelihood_chunked_equals_unchunked(model: protlms.Model, tmp_path: Path) -> None:
    plain = {
        r["record_id"]: r for r in model.likelihood(TINY_FASTA, output_dir=tmp_path / "p").rows()
    }
    chunked = {
        r["record_id"]: r
        for r in model.likelihood(TINY_FASTA, output_dir=tmp_path / "c", chunk_size=2).rows()
    }
    assert set(chunked) == set(plain)
    for rid in plain:
        assert math.isclose(
            float(chunked[rid]["log_likelihood"]), float(plain[rid]["log_likelihood"]), abs_tol=1e-4
        )
