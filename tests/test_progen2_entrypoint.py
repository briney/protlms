"""Unit tests for the ProGen2 entrypoint's torch-free helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENTRYPOINT = Path(__file__).parents[1] / "containers" / "progen2" / "entrypoint.py"


def _load():
    spec = importlib.util.spec_from_file_location("progen2_entrypoint", _ENTRYPOINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entrypoint = _load()


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("progen2-small", "hugohrban/progen2-small"),
        ("progen2-base", "hugohrban/progen2-base"),
        ("hugohrban/progen2-small", "hugohrban/progen2-small"),
    ],
)
def test_resolve_hf_id(checkpoint: str, expected: str) -> None:
    assert entrypoint.resolve_hf_id(checkpoint) == expected


def test_strip_special_tokens_keeps_amino_acids() -> None:
    # control tokens (1/2), pad, and stray markers removed; AA letters kept
    assert entrypoint.strip_special_tokens("1MAGIC2") == "MAGIC"
    assert entrypoint.strip_special_tokens("<|pad|>ACDE<|pad|>") == "ACDE"
