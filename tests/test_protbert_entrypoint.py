"""Unit tests for the ProtBERT entrypoint's torch/transformers-free helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENTRYPOINT = Path(__file__).parents[1] / "containers" / "protbert" / "entrypoint.py"


def _load():
    spec = importlib.util.spec_from_file_location("protbert_entrypoint", _ENTRYPOINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entrypoint = _load()


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("prot_bert", "Rostlab/prot_bert"),
        ("prot_bert_bfd", "Rostlab/prot_bert_bfd"),
        ("Rostlab/prot_bert", "Rostlab/prot_bert"),
    ],
)
def test_resolve_hf_id(checkpoint: str, expected: str) -> None:
    assert entrypoint.resolve_hf_id(checkpoint) == expected


def test_preprocess_spaces_residues() -> None:
    assert entrypoint.preprocess("MKTAY") == "M K T A Y"


def test_preprocess_maps_rare_residues_to_x() -> None:
    # U, Z, O, B -> X; lowercase is upper-cased; result is space-separated.
    assert entrypoint.preprocess("auzob") == "A X X X X"


def test_sanitize_ids_dedupes_collisions() -> None:
    assert entrypoint.sanitize_ids(["a/b", "a:b", "ok"]) == ["a_b", "a_b__1", "ok"]


def test_read_fasta_parses_records(tmp_path: Path) -> None:
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">one desc\nMAGIC\n>two\nACDE\nFG\n")
    assert entrypoint.read_fasta(fasta) == [("one", "MAGIC"), ("two", "ACDEFG")]


@pytest.mark.parametrize(
    ("mutant", "expected"),
    [
        ("A24G", [("A", 24, "G")]),
        ("A24G:T56S", [("A", 24, "G"), ("T", 56, "S")]),
    ],
)
def test_parse_mutant_valid(mutant: str, expected: list[tuple[str, int, str]]) -> None:
    assert entrypoint.parse_mutant(mutant) == expected


def test_parse_mutant_invalid_raises() -> None:
    with pytest.raises(ValueError):
        entrypoint.parse_mutant("not-a-mutant")


def test_perplexity_from_mean() -> None:
    assert entrypoint.perplexity_from_mean(0.0) == pytest.approx(1.0)
    assert entrypoint.perplexity_from_mean(-1.0) == pytest.approx(2.718281828, rel=1e-6)


def test_truncate_warns_and_clips() -> None:
    warnings: list[str] = []
    long_seq = "A" * (entrypoint.MAX_SEQUENCE_LENGTH + 5)
    out = entrypoint._truncate(long_seq, warnings, "big")
    assert len(out) == entrypoint.MAX_SEQUENCE_LENGTH
    assert warnings and "truncated" in warnings[0]
