"""Unit tests for the ESM2 container entrypoint's pure helpers.

These exercise the dependency-light logic (no torch/transformers needed) by
loading the standalone entrypoint module by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENTRYPOINT = Path(__file__).parents[1] / "containers" / "esm2" / "entrypoint.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("esm2_entrypoint", _ENTRYPOINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entrypoint = _load_entrypoint()


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("esm2_t6_8M", "facebook/esm2_t6_8M_UR50D"),
        ("esm2_t33_650M", "facebook/esm2_t33_650M_UR50D"),
        ("esm2_t6_8M_UR50D", "facebook/esm2_t6_8M_UR50D"),
        ("facebook/esm2_t6_8M_UR50D", "facebook/esm2_t6_8M_UR50D"),
    ],
)
def test_resolve_hf_id(checkpoint: str, expected: str) -> None:
    assert entrypoint.resolve_hf_id(checkpoint) == expected


def test_sanitize_ids_replaces_unsafe_and_dedupes() -> None:
    cleaned = entrypoint.sanitize_ids(["sp|P01308|INS", "sp|P01308|INS", "ok-id"])
    assert cleaned[0] == "sp_P01308_INS"
    assert cleaned[1] == "sp_P01308_INS__1"  # collision de-duplicated
    assert cleaned[2] == "ok-id"


def test_read_fasta_matches_contract_id_rule(tmp_path: Path) -> None:
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">seq1 description here\nACDE\nFGHI\n>seq2\nKLMN\n")
    records = entrypoint.read_fasta(fasta)
    assert records == [("seq1", "ACDEFGHI"), ("seq2", "KLMN")]


def test_perplexity_from_mean() -> None:
    # mean log-likelihood of 0 => perplexity 1.0
    assert entrypoint.perplexity_from_mean(0.0) == pytest.approx(1.0)
    assert entrypoint.perplexity_from_mean(-1.0) == pytest.approx(2.718281828, rel=1e-6)


def test_parse_mutant_single_and_multi() -> None:
    assert entrypoint.parse_mutant("A24G") == [("A", 24, "G")]
    assert entrypoint.parse_mutant("A24G:T56S") == [("A", 24, "G"), ("T", 56, "S")]


def test_parse_mutant_self_substitution() -> None:
    assert entrypoint.parse_mutant("M1M") == [("M", 1, "M")]


@pytest.mark.parametrize("bad", ["", "24G", "AG", "A2", "AxG", "A-1G"])
def test_parse_mutant_malformed_raises(bad: str) -> None:
    with pytest.raises(ValueError):
        entrypoint.parse_mutant(bad)
