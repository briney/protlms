"""plms: unified toolkit for inference across a variety of protein language models (pLMs)."""

from __future__ import annotations

from plms.contract import Capability, Manifest, PoolingMode, Result
from plms.exceptions import (
    CapabilityNotSupportedError,
    ContainerExecutionError,
    ContractVersionError,
    FastaError,
    ImageNotFoundError,
    InvalidRequestError,
    ModelNotFoundError,
    OutputParseError,
    PlmsError,
    RunnerError,
)
from plms.models import (
    EmbeddingResult,
    GenerationResult,
    LikelihoodResult,
    Model,
    ScoreResult,
    load,
)
from plms.registry import ModelEntry, Registry

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
    "PlmsError",
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
