"""Unit tests for the Profluent-E1 entrypoint's torch/E1-free helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENTRYPOINT = Path(__file__).parents[1] / "containers" / "e1" / "entrypoint.py"


def _load():
    spec = importlib.util.spec_from_file_location("e1_entrypoint", _ENTRYPOINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entrypoint = _load()


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("E1-150m", "Profluent-Bio/E1-150m"),
        ("E1-600m", "Profluent-Bio/E1-600m"),
        ("Profluent-Bio/E1-300m", "Profluent-Bio/E1-300m"),
    ],
)
def test_resolve_hf_id(checkpoint: str, expected: str) -> None:
    assert entrypoint.resolve_hf_id(checkpoint) == expected


def test_vocab_constants() -> None:
    # Documented E1 vocab: mask "?" is id 5; amino acids are A..Z at ids 8..33.
    assert entrypoint._MASK_ID == 5
    assert entrypoint._AA_TO_ID["A"] == 8
    assert entrypoint._AA_TO_ID["C"] == 10
    assert entrypoint._AA_TO_ID["Z"] == 33


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


def _logp_row(value_by_id: dict[int, float]) -> list[float]:
    """Build a length-34 logprob vector with the given id->value overrides."""
    vec = [0.0] * 34
    for idx, val in value_by_id.items():
        vec[idx] = val
    return vec


def test_score_variant_self_substitution_is_zero() -> None:
    wt = "ACDEF"
    logp = {1: _logp_row({entrypoint._AA_TO_ID["A"]: -3.0})}
    score, n_mut, err = entrypoint._score_variant("A1A", wt, logp)
    assert err is None
    assert n_mut == 1
    assert score == pytest.approx(0.0)


def test_score_variant_single_uses_logratio() -> None:
    wt = "ACDEF"
    logp = {1: _logp_row({entrypoint._AA_TO_ID["A"]: -2.0, entrypoint._AA_TO_ID["G"]: -0.5})}
    score, n_mut, err = entrypoint._score_variant("A1G", wt, logp)
    assert err is None
    assert score == pytest.approx(-0.5 - (-2.0))


def test_score_variant_wt_mismatch_errors() -> None:
    score, _n, err = entrypoint._score_variant("C1G", "ACDEF", {1: _logp_row({})})
    assert score is None
    assert "mismatch" in err


def test_build_manifest_150m(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entrypoint, "DEFAULT_CHECKPOINT", "E1-150m")
    m = entrypoint.build_manifest()
    assert m["contract_version"] == "0.3"
    assert m["model_family"] == "e1"
    assert m["name"] == "E1-150m"
    assert m["embedding_dim"] == 768
    assert m["num_layers"] == 20
    assert m["min_gpu_memory_gb"] is None
    assert set(m["capabilities"]) == {"embed", "likelihood", "score"}
    assert set(m["pooling_modes"]) == {"mean", "cls", "none"}


def test_build_manifest_600m(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entrypoint, "DEFAULT_CHECKPOINT", "E1-600m")
    m = entrypoint.build_manifest()
    assert m["embedding_dim"] == 1280
    assert m["num_layers"] == 30
    assert m["min_gpu_memory_gb"] == 4.0
