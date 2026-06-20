"""Tests for the model registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from plms.exceptions import ModelNotFoundError
from plms.registry import BuildSpec, ModelEntry, Registry


def test_default_registry_resolves_esm2_8m() -> None:
    registry = Registry.load()
    entry = registry.resolve("esm2-8m")
    assert entry.image == "ghcr.io/briney/plms-esm2:t6_8M"
    assert entry.model_family == "esm2"


def test_resolve_by_alias() -> None:
    registry = Registry.load()
    by_name = registry.resolve("esm2-8m")
    by_alias = registry.resolve("esm2_t6_8M")
    assert by_alias == by_name


def test_resolve_unknown_raises() -> None:
    registry = Registry.load()
    with pytest.raises(ModelNotFoundError):
        registry.resolve("does-not-exist")


def test_list_models_includes_phase0_models() -> None:
    names = {e.name for e in Registry.load().list_models()}
    assert {"esm2-8m", "esm2-650m"} <= names


def test_load_from_custom_path(tmp_path: Path) -> None:
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        "models:\n"
        "  - name: tiny\n"
        "    aliases: [t]\n"
        "    image: example:tiny\n"
        "    model_family: demo\n"
    )
    registry = Registry.load(yaml_path)
    assert registry.resolve("t").name == "tiny"
    assert registry.resolve("tiny").image == "example:tiny"


def test_resolve_progen2_small() -> None:
    registry = Registry.load()
    entry = registry.resolve("progen2-small")
    assert entry.image == "ghcr.io/briney/plms-progen2:small"
    assert entry.model_family == "progen2"
    assert registry.resolve("progen2_small") == entry


def test_resolve_esm_c() -> None:
    registry = Registry.load()
    e300 = registry.resolve("esm-c-300m")
    assert e300.image == "ghcr.io/briney/plms-esm-c:300m"
    assert e300.model_family == "esm-c"
    assert registry.resolve("esmc_300m") == e300
    e600 = registry.resolve("esm-c-600m")
    assert e600.image == "ghcr.io/briney/plms-esm-c:600m"
    assert e600.model_family == "esm-c"
    assert registry.resolve("esmc_600m") == e600


def _entry(**overrides: object) -> ModelEntry:
    data = dict(name="m", image="ghcr.io/briney/plms-esm2:t6_8M", model_family="esm2")
    data.update(overrides)
    return ModelEntry(**data)


def test_pinned_ref_uses_digest_and_strips_tag() -> None:
    entry = _entry(digest="sha256:abc123")
    assert entry.pinned_ref() == "ghcr.io/briney/plms-esm2@sha256:abc123"


def test_pinned_ref_without_digest_returns_image() -> None:
    assert _entry().pinned_ref() == "ghcr.io/briney/plms-esm2:t6_8M"


def test_pinned_ref_preserves_registry_host_port() -> None:
    entry = _entry(image="host:5000/repo:tag", digest="sha256:deadbeef")
    assert entry.pinned_ref() == "host:5000/repo@sha256:deadbeef"


def test_invalid_digest_rejected() -> None:
    with pytest.raises(ValueError, match="sha256:"):
        _entry(digest="abc123")


def test_build_spec_parsed_from_entry() -> None:
    entry = _entry(build={"context": "containers/esm2", "args": {"ESM2_CHECKPOINT": "esm2_t6_8M"}})
    assert isinstance(entry.build, BuildSpec)
    assert entry.build.context == "containers/esm2"
    assert entry.build.args["ESM2_CHECKPOINT"] == "esm2_t6_8M"
