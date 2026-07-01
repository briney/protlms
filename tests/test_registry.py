"""Tests for the model registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from protlms.exceptions import ModelNotFoundError
from protlms.registry import BuildSpec, ModelEntry, Registry


def test_default_registry_resolves_esm2_8m() -> None:
    registry = Registry.load()
    entry = registry.resolve("esm2-8m")
    assert entry.image == "ghcr.io/briney/protlms-esm:t6_8M"
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
    assert entry.image == "ghcr.io/briney/protlms-progen2:small"
    assert entry.model_family == "progen2"
    assert registry.resolve("progen2_small") == entry


def test_resolve_esm_c() -> None:
    registry = Registry.load()
    e300 = registry.resolve("esm-c-300m")
    assert e300.image == "ghcr.io/briney/protlms-esm-c:300m"
    assert e300.model_family == "esm-c"
    assert registry.resolve("esmc_300m") == e300
    e600 = registry.resolve("esm-c-600m")
    assert e600.image == "ghcr.io/briney/protlms-esm-c:600m"
    assert e600.model_family == "esm-c"
    assert registry.resolve("esmc_600m") == e600


def test_resolve_esm_c_6b() -> None:
    entry = Registry.load().resolve("esm-c-6b")
    assert entry.image == "ghcr.io/briney/protlms-esm-c:6b"
    assert entry.model_family == "esm-c"
    assert entry.build.context == "containers/esm-c"
    assert entry.build.args["ESMC_CHECKPOINT"] == "esmc_6b"
    assert Registry.load().resolve("esmc_6b") == entry


def test_registry_includes_all_esm_c_sizes() -> None:
    names = {e.name for e in Registry.load().list_models()}
    assert {"esm-c-300m", "esm-c-600m", "esm-c-6b"} <= names


def _entry(**overrides: object) -> ModelEntry:
    data = dict(name="m", image="ghcr.io/briney/protlms-esm2:t6_8M", model_family="esm2")
    data.update(overrides)
    return ModelEntry(**data)


def test_pinned_ref_uses_digest_and_strips_tag() -> None:
    entry = _entry(digest="sha256:abc123")
    assert entry.pinned_ref() == "ghcr.io/briney/protlms-esm2@sha256:abc123"


def test_pinned_ref_without_digest_returns_image() -> None:
    assert _entry().pinned_ref() == "ghcr.io/briney/protlms-esm2:t6_8M"


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


def test_resolve_protbert() -> None:
    registry = Registry.load()
    base = registry.resolve("protbert")
    assert base.image == "ghcr.io/briney/protlms-protbert:uniref100"
    assert base.model_family == "protbert"
    assert registry.resolve("prot_bert") == base
    bfd = registry.resolve("protbert-bfd")
    assert bfd.image == "ghcr.io/briney/protlms-protbert:bfd"
    assert bfd.model_family == "protbert"
    assert registry.resolve("prot_bert_bfd") == bfd


def test_resolve_e1() -> None:
    registry = Registry.load()
    cases = [
        ("e1-150m", "E1-150m", "150m"),
        ("e1-300m", "E1-300m", "300m"),
        ("e1-600m", "E1-600m", "600m"),
    ]
    for name, alias, tag in cases:
        entry = registry.resolve(name)
        assert entry.image == f"ghcr.io/briney/protlms-e1:{tag}"
        assert entry.model_family == "e1"
        assert registry.resolve(alias) == entry


def test_resolve_esm1b() -> None:
    registry = Registry.load()
    entry = registry.resolve("esm1b")
    assert entry.model_family == "esm1b"
    assert entry.image.startswith("ghcr.io/briney/protlms-esm:")
    assert entry.build is not None
    assert entry.build.context == "containers/esm"
    assert entry.build.args["ESM_HF_ID"] == "facebook/esm1b_t33_650M_UR50S"


def test_registry_includes_all_esm2_sizes() -> None:
    names = {e.name for e in Registry.load().list_models()}
    assert {"esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b", "esm2-15b"} <= names


def test_resolve_esm2_3b_uses_shared_context() -> None:
    entry = Registry.load().resolve("esm2-3b")
    assert entry.build.context == "containers/esm"
    assert entry.build.args["ESM_MODEL_FAMILY"] == "esm2"
