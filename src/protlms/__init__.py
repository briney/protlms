"""protlms: unified toolkit for inference across a variety of protein language models (pLMs)."""

from __future__ import annotations

from protlms.contract import Capability, Manifest, PoolingMode, Result
from protlms.exceptions import (
    CapabilityNotSupportedError,
    ContainerExecutionError,
    ContractVersionError,
    FastaError,
    ImageNotFoundError,
    InvalidRequestError,
    ModelNotFoundError,
    OutputParseError,
    ProtlmsError,
    RunnerError,
)
from protlms.models import (
    EmbeddingResult,
    GenerationResult,
    LikelihoodResult,
    Model,
    ScoreResult,
    load,
)
from protlms.registry import ModelEntry, Registry

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load",
    "Model",
    "EmbeddingResult",
    "LikelihoodResult",
    "ScoreResult",
    "GenerationResult",
    "Registry",
    "ModelEntry",
    "Manifest",
    "Result",
    "Capability",
    "PoolingMode",
    "ProtlmsError",
    "ModelNotFoundError",
    "ImageNotFoundError",
    "ContractVersionError",
    "CapabilityNotSupportedError",
    "InvalidRequestError",
    "RunnerError",
    "ContainerExecutionError",
    "OutputParseError",
    "FastaError",
]
