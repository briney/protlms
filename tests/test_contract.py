"""Tests for the container-contract schemas and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from plms.contract import (
    CONTRACT_VERSION,
    Capability,
    ContainerError,
    Manifest,
    OutputArtifact,
    PoolingMode,
    Result,
    check_contract_compatibility,
    parse_container_error,
    parse_contract_version,
)
from plms.exceptions import ContainerExecutionError, ContractVersionError


def _manifest_dict() -> dict:
    return {
        "contract_version": "0.1",
        "name": "esm2_t6_8M",
        "version": "1.0.0",
        "description": "ESM2 8M parameter model.",
        "model_family": "esm2",
        "capabilities": ["embed", "likelihood"],
        "embedding_dim": 320,
        "max_sequence_length": 1024,
        "pooling_modes": ["mean", "cls", "none"],
        "num_layers": 6,
        "min_gpu_memory_gb": None,
        "default_batch_size": 8,
    }


def test_contract_version_is_semantic_string() -> None:
    assert CONTRACT_VERSION == "0.3"
    assert parse_contract_version(CONTRACT_VERSION) == (0, 3)


def test_manifest_round_trip() -> None:
    manifest = Manifest.model_validate(_manifest_dict())
    assert manifest.name == "esm2_t6_8M"
    assert manifest.embedding_dim == 320
    assert Capability.EMBED in manifest.capabilities
    assert PoolingMode.MEAN in manifest.pooling_modes
    assert manifest.image_digest is None


def test_manifest_ignores_unknown_fields() -> None:
    data = _manifest_dict()
    data["some_future_field"] = {"nested": 1}
    manifest = Manifest.model_validate(data)  # must not raise
    assert manifest.name == "esm2_t6_8M"


def test_result_round_trip() -> None:
    result = Result.model_validate(
        {
            "contract_version": "0.1",
            "capability": "embed",
            "model_name": "esm2_t6_8M",
            "n_input_records": 3,
            "n_output_records": 3,
            "artifacts": [
                {
                    "path": "embeddings.npz",
                    "kind": "pooled_embeddings",
                    "shape": [3, 320],
                    "dtype": "float32",
                }
            ],
            "warnings": ["1 sequence truncated"],
            "params": {"pooling": "mean"},
        }
    )
    assert result.capability is Capability.EMBED
    assert result.artifacts[0].kind == "pooled_embeddings"
    assert result.warnings == ["1 sequence truncated"]


def test_output_artifact_optional_fields_default_to_none() -> None:
    art = OutputArtifact(path="per_residue/seq1.npy", kind="per_residue_embeddings")
    assert art.shape is None
    assert art.record_ids is None


def test_container_error_round_trip() -> None:
    err = ContainerError.model_validate(
        {"error_type": "SequenceTooLong", "message": "seq exceeds max", "details": {"id": "seq1"}}
    )
    assert err.error_type == "SequenceTooLong"
    assert err.details["id"] == "seq1"


@pytest.mark.parametrize(
    "image_version",
    ["0.1", "0.0", "0.5"],  # same major (0) => compatible
)
def test_check_compatibility_same_major_ok(image_version: str) -> None:
    check_contract_compatibility(image_version, client_version="0.1")  # must not raise


@pytest.mark.parametrize("image_version", ["1.0", "2.3"])
def test_check_compatibility_major_mismatch_raises(image_version: str) -> None:
    with pytest.raises(ContractVersionError):
        check_contract_compatibility(image_version, client_version="0.1")


def test_parse_contract_version_malformed_raises() -> None:
    with pytest.raises(ContractVersionError):
        parse_contract_version("not-a-version")


def test_parse_container_error_extracts_last_json_line() -> None:
    stderr = (
        "Loading model weights...\n"
        "Traceback noise that is not json\n"
        '{"contract_version": "0.1", "error_type": "InternalError", "message": "boom"}\n'
    )
    err = parse_container_error(stderr)
    assert err is not None
    assert err.error_type == "InternalError"
    assert err.message == "boom"


def test_parse_container_error_returns_none_when_no_json() -> None:
    assert parse_container_error("segfault\ncore dumped\n") is None


_DATA = Path(__file__).parent / "data"


def test_documented_manifest_example_validates() -> None:
    """The worked example in docs/CONTRACT.md must parse as a Manifest."""
    manifest = Manifest.model_validate_json((_DATA / "manifest.example.json").read_text())
    assert manifest.contract_version == CONTRACT_VERSION
    assert Capability.EMBED in manifest.capabilities
    assert Capability.SCORE in manifest.capabilities


def test_documented_result_example_validates() -> None:
    """The worked example in docs/CONTRACT.md must parse as a Result."""
    result = Result.model_validate_json((_DATA / "result.embed.example.json").read_text())
    assert result.capability is Capability.EMBED
    assert result.artifacts[0].kind == "pooled_embeddings"


def test_documented_score_result_example_validates() -> None:
    """The score result example in docs/CONTRACT.md must parse as a Result."""
    result = Result.model_validate_json((_DATA / "result.score.example.json").read_text())
    assert result.capability is Capability.SCORE
    assert result.artifacts[0].kind == "variant_scores_csv"


def test_documented_generate_result_example_validates() -> None:
    """The generate result example in docs/CONTRACT.md must parse as a Result."""
    result = Result.model_validate_json((_DATA / "result.generate.example.json").read_text())
    assert result.capability is Capability.GENERATE
    assert result.artifacts[0].kind == "generated_fasta"


def test_container_execution_error_carries_structured_fields() -> None:
    err = ContainerExecutionError(
        error_type="DeviceUnavailable",
        message="cuda requested but unavailable",
        details={"requested": "cuda"},
        exit_code=1,
        stderr_tail="...",
    )
    assert err.error_type == "DeviceUnavailable"
    assert err.exit_code == 1
    assert "cuda" in str(err)
