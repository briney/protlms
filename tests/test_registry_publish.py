"""Tests for the GHCR publishing helper script."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.registry_publish import lookup_build, set_digest

SAMPLE = """\
models:
  - name: esm2-8m
    aliases: [esm2_t6_8M]
    image: ghcr.io/briney/plms-esm2:t6_8M
    model_family: esm2
    build:
      context: containers/esm2
      args: { ESM2_CHECKPOINT: esm2_t6_8M }
"""


def _yaml(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(SAMPLE)
    return path


def test_lookup_build_returns_image_context_args(tmp_path: Path) -> None:
    image, context, args = lookup_build(_yaml(tmp_path), "esm2-8m")
    assert image == "ghcr.io/briney/plms-esm2:t6_8M"
    assert context == "containers/esm2"
    assert args == {"ESM2_CHECKPOINT": "esm2_t6_8M"}


def test_lookup_build_unknown_name_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        lookup_build(_yaml(tmp_path), "nope")


def test_lookup_build_missing_build_block_raises(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("models:\n  - name: x\n    image: i\n    model_family: f\n")
    with pytest.raises(ValueError, match="no build metadata"):
        lookup_build(path, "x")


def test_set_digest_writes_and_roundtrips(tmp_path: Path) -> None:
    import yaml

    path = _yaml(tmp_path)
    set_digest(path, "esm2-8m", "sha256:abc123")
    data = yaml.safe_load(path.read_text())
    entry = next(m for m in data["models"] if m["name"] == "esm2-8m")
    assert entry["digest"] == "sha256:abc123"


def test_set_digest_rejects_bad_digest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sha256:"):
        set_digest(_yaml(tmp_path), "esm2-8m", "abc123")
