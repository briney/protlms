"""Container runtime: build a ``docker run`` invocation and execute it.

Phase 0 uses ``subprocess`` to call the local ``docker`` CLI, which keeps the
exact command transparent and loggable. Everything goes through the
:class:`Runner` protocol so a Docker-SDK implementation can be dropped in later
without touching the rest of the client.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from plms.exceptions import ImageNotFoundError, ImagePullError, RunnerError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSpec:
    """A fully-specified container run.

    Attributes:
        image: Docker image reference.
        command: The contract subcommand plus its flags (argv after the image).
        input_dir: Host directory bind-mounted read-only at ``/in``.
        output_dir: Host directory bind-mounted read-write at ``/out``.
        use_gpu: Whether to request all GPUs (``--gpus all``).
        extra_env: Environment variables to pass with ``-e``.
    """

    image: str
    command: list[str]
    input_dir: Path
    output_dir: Path
    use_gpu: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """The outcome of a container run."""

    exit_code: int
    stdout: str
    stderr: str
    argv: list[str]


class Runner(Protocol):
    """Anything that can execute a :class:`RunSpec`."""

    def run(self, spec: RunSpec) -> RunResult: ...

    def manifest(self, image: str) -> str: ...

    def image_present(self, ref: str) -> bool: ...

    def pull(self, ref: str) -> None: ...


def _current_user() -> str | None:
    """Return ``uid:gid`` for the current process, or ``None`` on non-POSIX hosts."""
    if hasattr(os, "getuid"):
        return f"{os.getuid()}:{os.getgid()}"
    return None


def build_argv(spec: RunSpec, docker_executable: str = "docker") -> list[str]:
    """Build the ``docker run`` argv for a spec.

    Args:
        spec: The run specification.
        docker_executable: The docker CLI executable (overridable for podman etc).

    Returns:
        The argument vector to pass to :func:`subprocess.run`.
    """
    # Bind-mount sources must be absolute; a relative path is interpreted by
    # docker as a named volume rather than a host directory.
    input_dir = spec.input_dir.absolute()
    output_dir = spec.output_dir.absolute()
    argv = [docker_executable, "run", "--rm"]
    user = _current_user()
    if user is not None:
        # Run as the host user so written outputs are not owned by root.
        argv += ["--user", user]
    if spec.use_gpu:
        argv += ["--gpus", "all"]
    argv += ["-v", f"{input_dir}:/in:ro", "-v", f"{output_dir}:/out:rw"]
    for key in sorted(spec.extra_env):
        argv += ["-e", f"{key}={spec.extra_env[key]}"]
    argv.append(spec.image)
    argv += spec.command
    return argv


class SubprocessDockerRunner:
    """A :class:`Runner` that shells out to the local ``docker`` CLI."""

    def __init__(self, docker_executable: str = "docker") -> None:
        self._docker = docker_executable

    def run(self, spec: RunSpec) -> RunResult:
        """Run a container and capture its result (does not raise on non-zero exit).

        Raises:
            RunnerError: If the docker executable cannot be invoked.
        """
        argv = build_argv(spec, self._docker)
        completed = self._invoke(argv)
        return RunResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            argv=argv,
        )

    def manifest(self, image: str) -> str:
        """Run an image's ``manifest`` subcommand and return its stdout.

        Raises:
            RunnerError: If the docker executable cannot be invoked.
            ImageNotFoundError: If the run exits non-zero (image absent locally
                or its manifest command failed).
        """
        argv = [self._docker, "run", "--rm", image, "manifest"]
        completed = self._invoke(argv)
        if completed.returncode != 0:
            raise ImageNotFoundError(
                f"could not read manifest for image {image!r} "
                f"(exit {completed.returncode}); is it built locally? "
                f"Build it from containers/<family>. stderr: "
                f"{completed.stderr.strip()[:500]}"
            )
        return completed.stdout

    def image_present(self, ref: str) -> bool:
        """Return True if the image is available in the local Docker store."""
        completed = self._invoke([self._docker, "image", "inspect", ref])
        return completed.returncode == 0

    def pull(self, ref: str) -> None:
        """Pull an image from its registry.

        Raises:
            RunnerError: If the docker executable cannot be invoked.
            ImagePullError: If ``docker pull`` exits non-zero.
        """
        completed = self._invoke([self._docker, "pull", ref])
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            hint = ""
            if any(t in stderr.lower() for t in ("denied", "unauthorized", "authentication")):
                hint = " (authentication failed; try `docker login ghcr.io`)"
            raise ImagePullError(
                f"failed to pull image {ref!r} (exit {completed.returncode}){hint}. "
                f"stderr: {stderr[:500]}"
            )

    def _invoke(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        logger.info("running: %s", " ".join(argv))
        try:
            return subprocess.run(argv, capture_output=True, text=True, check=False)
        except (FileNotFoundError, OSError) as exc:
            raise RunnerError(
                f"failed to invoke docker executable {self._docker!r}: {exc}"
            ) from exc
