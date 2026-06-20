"""Model registry: resolve model names/aliases to Docker image references.

The registry is backed by a human-editable YAML file. The default registry
ships inside the package (``plms/_data/models.yaml``); a custom file can be
supplied for testing or local overrides.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

from plms.exceptions import ModelNotFoundError

_DEFAULT_REGISTRY_RESOURCE = "_data/models.yaml"


class BuildSpec(BaseModel):
    """Build-only metadata for the publishing pipeline; ignored by the client."""

    context: str
    args: dict[str, str] = {}


class ModelEntry(BaseModel):
    """One registry entry mapping a model to its image."""

    name: str
    aliases: list[str] = []
    image: str
    digest: str | None = None
    model_family: str
    build: BuildSpec | None = None

    @field_validator("digest")
    @classmethod
    def _validate_digest(cls, value: str | None) -> str | None:
        """Reject digests that are not ``sha256:`` references."""
        if value is not None and not value.startswith("sha256:"):
            raise ValueError(f"digest must start with 'sha256:', got {value!r}")
        return value

    def pinned_ref(self) -> str:
        """Image reference to pull/run.

        Returns ``<repo>@<digest>`` when a digest is set (reproducible), else the
        bare ``image`` tag (e.g. a locally-built image with no published digest).
        """
        if self.digest is None:
            return self.image
        return f"{self._strip_tag(self.image)}@{self.digest}"

    @staticmethod
    def _strip_tag(image: str) -> str:
        """Drop a ``:tag`` from the final path segment of an image reference."""
        prefix, sep, last = image.rpartition("/")
        name = last.split(":", 1)[0]
        return f"{prefix}{sep}{name}" if sep else name


class Registry:
    """An in-memory model registry loaded from YAML."""

    def __init__(self, entries: list[ModelEntry]) -> None:
        self._entries = entries
        self._by_key: dict[str, ModelEntry] = {}
        for entry in entries:
            for key in (entry.name, *entry.aliases):
                self._by_key[key] = entry

    @classmethod
    def load(cls, path: Path | None = None) -> Registry:
        """Load a registry from YAML.

        Args:
            path: A YAML file to load. If ``None``, the packaged default
                registry is used.

        Returns:
            The loaded registry.
        """
        if path is None:
            text = resources.files("plms").joinpath(_DEFAULT_REGISTRY_RESOURCE).read_text()
        else:
            text = Path(path).read_text()
        data = yaml.safe_load(text) or {}
        entries = [ModelEntry.model_validate(item) for item in data.get("models", [])]
        return cls(entries)

    def resolve(self, name: str) -> ModelEntry:
        """Resolve a model name or alias to its registry entry.

        Raises:
            ModelNotFoundError: If ``name`` matches no entry.
        """
        try:
            return self._by_key[name]
        except KeyError:
            known = ", ".join(sorted(e.name for e in self._entries))
            raise ModelNotFoundError(f"unknown model {name!r}; known models: {known}") from None

    def list_models(self) -> list[ModelEntry]:
        """Return all registry entries in file order."""
        return list(self._entries)
