"""Helpers for the GHCR publishing workflow.

Reads build metadata from the packaged ``models.yaml`` and writes published image
digests back into it. Kept dependency-light (PyYAML only) so CI can run it.

Usage:
    python -m scripts.registry_publish lookup <models.yaml> <name>
    python -m scripts.registry_publish set-digest <models.yaml> <name> <sha256:...>
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def _load(models_yaml: Path) -> dict:
    """Parse a models.yaml file into a dict."""
    return yaml.safe_load(Path(models_yaml).read_text()) or {}


def _find(data: dict, name: str) -> dict:
    """Return the entry whose ``name`` matches, or raise KeyError."""
    for entry in data.get("models", []):
        if entry.get("name") == name:
            return entry
    raise KeyError(f"no model named {name!r} in registry")


def lookup_build(models_yaml: Path, name: str) -> tuple[str, str, dict[str, str]]:
    """Return ``(image, build_context, build_args)`` for a model.

    Raises:
        KeyError: If the model name is not present.
        ValueError: If the model has no ``build`` block.
    """
    entry = _find(_load(models_yaml), name)
    build = entry.get("build")
    if not build:
        raise ValueError(f"model {name!r} has no build metadata")
    return entry["image"], build["context"], dict(build.get("args", {}))


def set_digest(models_yaml: Path, name: str, digest: str) -> None:
    """Write ``digest`` onto the named entry and rewrite the file.

    Raises:
        KeyError: If the model name is not present.
        ValueError: If ``digest`` is not a ``sha256:`` reference.
    """
    if not digest.startswith("sha256:"):
        raise ValueError(f"digest must start with 'sha256:', got {digest!r}")
    data = _load(Path(models_yaml))
    _find(data, name)["digest"] = digest
    Path(models_yaml).write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def main(argv: list[str]) -> int:
    """Tiny CLI used by the publishing workflow."""
    match argv:
        case ["lookup", models_yaml, name]:
            image, context, args = lookup_build(Path(models_yaml), name)
            # Emit GITHUB_OUTPUT lines. build_args uses docker/build-push-action's
            # native newline-separated KEY=VALUE form (single line for one arg).
            build_args = "\n".join(f"{k}={v}" for k, v in args.items())
            print(f"image={image}")
            print(f"context={context}")
            print(f"build_args={build_args}")
            return 0
        case ["set-digest", models_yaml, name, digest]:
            set_digest(Path(models_yaml), name, digest)
            return 0
        case _:
            print(__doc__)
            return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
