"""Exception hierarchy for plms.

All errors raised by the library derive from :class:`PlmsError`, so callers can
catch the whole family with a single ``except plms.PlmsError``.
"""

from __future__ import annotations


class PlmsError(Exception):
    """Base class for every error raised by plms."""


class ModelNotFoundError(PlmsError):
    """Raised when a model name/alias cannot be resolved in the registry."""


class ImageNotFoundError(PlmsError):
    """Raised when a model's Docker image is not available locally."""


class ContractVersionError(PlmsError):
    """Raised when an image's contract version is incompatible with the client."""


class CapabilityNotSupportedError(PlmsError):
    """Raised when a model does not declare support for a requested capability."""


class InvalidRequestError(PlmsError):
    """Raised when a request fails client-side validation (e.g. bad pooling mode)."""


class RunnerError(PlmsError):
    """Raised when the container runtime could not be invoked (e.g. docker missing)."""


class OutputParseError(PlmsError):
    """Raised when a container's output directory is missing or malformed."""


class FastaError(PlmsError):
    """Raised when a FASTA input file cannot be parsed."""


class ContainerExecutionError(PlmsError):
    """Raised when a container ran but exited with a non-zero status.

    Carries the structured fields from the container's stderr error object when
    available, plus the exit code and a tail of stderr for debugging.

    Args:
        message: Human-readable error message.
        error_type: The container's ``error_type`` (e.g. ``"SequenceTooLong"``),
            or ``None`` when stderr held no structured error.
        details: Free-form details from the container's structured error.
        exit_code: The container process exit code.
        stderr_tail: The last lines of the container's stderr.
    """

    def __init__(
        self,
        message: str,
        *,
        error_type: str | None = None,
        details: dict[str, str] | None = None,
        exit_code: int | None = None,
        stderr_tail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail
