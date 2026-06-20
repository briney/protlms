"""Opt-in end-to-end test of GHCR pull + run.

Runs only when PLMS_RUN_GHCR_TESTS=1 and docker is available. It removes the
esm2-8m image locally, then `plms.load` with auto-pull enabled, proving the
client fetches the published image from GHCR and runs it.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

import plms
from plms.registry import Registry


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PLMS_RUN_GHCR_TESTS") != "1" or not _docker_available(),
        reason="set PLMS_RUN_GHCR_TESTS=1 with docker available to run GHCR pull tests",
    ),
]


def test_load_pulls_published_image_from_ghcr() -> None:
    ref = Registry.load().resolve("esm2-8m").pinned_ref()
    subprocess.run(["docker", "image", "rm", "-f", ref], capture_output=True)
    model = plms.load("esm2-8m", allow_pull=True)
    assert model.manifest.model_family == "esm2"
