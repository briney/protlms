"""End-to-end integration test against a locally built ProGen2 image."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import plms

IMAGE = "plms-progen2:small"
REPO_ROOT = Path(__file__).parents[1]
PROMPTS = REPO_ROOT / "tests" / "data" / "prompts.fasta"
SEQS = REPO_ROOT / "tests" / "data" / "tiny.fasta"
_AA = set("ACDEFGHIKLMNPQRSTVWY")


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
def progen2_image() -> str:
    present = (
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    )
    if not present:
        subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                "PROGEN2_CHECKPOINT=progen2-small",
                "-t",
                IMAGE,
                str(REPO_ROOT / "containers" / "progen2"),
            ],
            check=True,
        )
    return IMAGE


@pytest.fixture(scope="session")
def model(progen2_image: str) -> plms.Model:
    return plms.load("progen2-small")


def test_manifest_declares_generate_and_likelihood(model: plms.Model) -> None:
    caps = {c.value for c in model.manifest.capabilities}
    assert {"generate", "likelihood"} <= caps
    assert model.manifest.pooling_modes == []


def test_generate_is_deterministic_with_seed(model: plms.Model, tmp_path: Path) -> None:
    first = model.generate(
        PROMPTS, num_samples=2, temperature=0.8, top_p=0.9, seed=7, output_dir=tmp_path / "a"
    )
    second = model.generate(
        PROMPTS, num_samples=2, temperature=0.8, top_p=0.9, seed=7, output_dir=tmp_path / "b"
    )
    a = {r.id: r.sequence for r in first.sequences()}
    b = {r.id: r.sequence for r in second.sequences()}
    assert len(a) == 4  # 2 prompts x num_samples=2
    assert a == b  # same seed => identical output


def test_generate_produces_valid_sequences(model: plms.Model, tmp_path: Path) -> None:
    result = model.generate(
        PROMPTS, num_samples=2, max_length=64, seed=1, output_dir=tmp_path / "gen"
    )
    seqs = result.sequences()
    assert {r.id for r in seqs} == {
        "prefix1__sample0",
        "prefix1__sample1",
        "uncond__sample0",
        "uncond__sample1",
    }
    for record in seqs:
        assert record.sequence  # non-empty
        assert set(record.sequence) <= _AA  # clean amino acids only
        assert len(record.sequence) <= 64


def test_progen2_likelihood(model: plms.Model, tmp_path: Path) -> None:
    result = model.likelihood(SEQS, output_dir=tmp_path / "ll")
    rows = result.rows()
    assert len(rows) == 3
    assert result.result.params.get("likelihood_method") == "causal"
    for row in rows:
        assert math.isfinite(float(row["log_likelihood"]))
        assert row["perplexity"] > 1.0
