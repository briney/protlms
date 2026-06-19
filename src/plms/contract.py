"""The container contract: schemas and version handling.

This module is the executable mirror of ``docs/CONTRACT.md``. The client and
every model image agree on exactly these schemas; nothing else passes between
them. Keep this file and the contract document edited together.
"""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from plms.exceptions import ContractVersionError

#: The contract version this client speaks (``MAJOR.MINOR``).
CONTRACT_VERSION = "0.2"


class Capability(StrEnum):
    """An inference capability a model may declare and the client may request."""

    EMBED = "embed"
    LIKELIHOOD = "likelihood"
    SCORE = "score"  # implemented in contract 0.2
    GENERATE = "generate"  # reserved for a future contract minor version


class PoolingMode(StrEnum):
    """How per-residue representations are reduced to a per-sequence vector."""

    MEAN = "mean"
    CLS = "cls"
    NONE = "none"  # no pooling: return per-residue representations


class ArtifactKind(StrEnum):
    """Known values for :attr:`OutputArtifact.kind` (the field stays a free str
    so future kinds remain forward-compatible)."""

    POOLED_EMBEDDINGS = "pooled_embeddings"
    PER_RESIDUE_EMBEDDINGS = "per_residue_embeddings"
    LIKELIHOODS_CSV = "likelihoods_csv"
    VARIANT_SCORES_CSV = "variant_scores_csv"


class Manifest(BaseModel):
    """A model image's self-description, emitted by its ``manifest`` subcommand.

    Unknown fields are ignored so a newer image (minor-bumped contract) can be
    read by an older client.
    """

    model_config = ConfigDict(extra="ignore")

    contract_version: str
    name: str
    version: str
    description: str
    model_family: str
    capabilities: list[Capability]
    embedding_dim: int
    max_sequence_length: int
    pooling_modes: list[PoolingMode]
    num_layers: int
    min_gpu_memory_gb: float | None = None
    default_batch_size: int
    #: Resolved by the client at run time, not self-reported by the image.
    image_digest: str | None = None


class OutputArtifact(BaseModel):
    """One file produced by a capability run, described in ``result.json``."""

    model_config = ConfigDict(extra="ignore")

    path: str  # relative to the output mount (/out)
    kind: str  # e.g. "pooled_embeddings" | "per_residue_embeddings" | "likelihoods_csv"
    record_ids: list[str] | None = None
    shape: list[int] | None = None
    dtype: str | None = None


class Result(BaseModel):
    """The success summary a container writes to ``/out/result.json``."""

    model_config = ConfigDict(extra="ignore")

    contract_version: str
    capability: Capability
    model_name: str
    n_input_records: int
    n_output_records: int
    artifacts: list[OutputArtifact]
    warnings: list[str] = []
    params: dict[str, str] = {}


class ContainerError(BaseModel):
    """The structured error a container writes to stderr on a non-zero exit."""

    model_config = ConfigDict(extra="ignore")

    contract_version: str | None = None
    error_type: str
    message: str
    details: dict[str, str] = {}


def parse_contract_version(version: str) -> tuple[int, int]:
    """Parse a ``MAJOR.MINOR`` contract version string into a tuple of ints.

    Args:
        version: A version string such as ``"0.1"``.

    Returns:
        ``(major, minor)``.

    Raises:
        ContractVersionError: If ``version`` is not ``MAJOR.MINOR`` integers.
    """
    parts = version.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ContractVersionError(
            f"malformed contract version {version!r}; expected 'MAJOR.MINOR'"
        )
    major, minor = (int(p) for p in parts)
    return major, minor


def check_contract_compatibility(
    image_version: str, client_version: str = CONTRACT_VERSION
) -> None:
    """Verify an image's contract version is compatible with this client.

    Compatibility rule: the major versions must match. A newer image minor is
    tolerated (the client ignores unknown fields); callers may warn separately.

    Args:
        image_version: The ``contract_version`` reported by the image manifest.
        client_version: The client's contract version (defaults to
            :data:`CONTRACT_VERSION`).

    Raises:
        ContractVersionError: If the major versions differ.
    """
    image_major, _ = parse_contract_version(image_version)
    client_major, _ = parse_contract_version(client_version)
    if image_major != client_major:
        raise ContractVersionError(
            f"image contract version {image_version!r} is incompatible with client "
            f"version {client_version!r} (major version mismatch)"
        )


def parse_container_error(stderr: str) -> ContainerError | None:
    """Extract a structured :class:`ContainerError` from a container's stderr.

    The container writes its error object as a single JSON line. This scans
    stderr from the last line backwards and returns the first line that parses
    into a valid :class:`ContainerError`.

    Args:
        stderr: The full captured stderr of a container run.

    Returns:
        The parsed error, or ``None`` if no structured error line was found.
    """
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "error_type" in payload and "message" in payload:
            return ContainerError.model_validate(payload)
    return None
