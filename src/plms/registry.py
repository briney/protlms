"""Model registry: resolve model names/aliases to Docker image references.

The registry is backed by a human-editable YAML file. The default registry
ships inside the package (``plms/_data/models.yaml``); a custom file can be
supplied for testing or local overrides.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel

from plms.exceptions import ModelNotFoundError

_DEFAULT_REGISTRY_RESOURCE = "_data/models.yaml"


class ModelEntry(BaseModel):
    """One registry entry mapping a model to its image."""

    name: str
    aliases: list[str] = []
    image: str
    model_family: str


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
