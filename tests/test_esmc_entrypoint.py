"""Unit tests for the ESM-C entrypoint's torch/esm-free helpers and manifest."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ENTRYPOINT = Path(__file__).parents[1] / "containers" / "esm-c" / "entrypoint.py"


def _load():
    spec = importlib.util.spec_from_file_location("esmc_entrypoint", _ENTRYPOINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entrypoint = _load()


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


@pytest.mark.parametrize(
    ("checkpoint", "dim", "layers"),
    [("esmc_300m", 960, 30), ("esmc_600m", 1152, 36), ("esmc_6b", 2560, 80)],
)
def test_build_manifest(
    monkeypatch: pytest.MonkeyPatch, checkpoint: str, dim: int, layers: int
) -> None:
    monkeypatch.setattr(entrypoint, "DEFAULT_CHECKPOINT", checkpoint)
    m = entrypoint.build_manifest()
    assert m["name"] == checkpoint
    assert m["model_family"] == "esm-c"
    assert m["contract_version"] == "0.4"
    assert m["embedding_dim"] == dim
    assert m["num_layers"] == layers
    assert m["capabilities"] == ["embed", "likelihood", "score", "contacts"]
    assert m["max_sequence_length"] == 2048


def test_jacobian_to_contacts_shape_symmetry_zero_diag() -> None:
    rng = np.random.default_rng(0)
    length = 7
    contacts = entrypoint.jacobian_to_contacts(rng.standard_normal((length, 20, length, 20)))
    assert contacts.shape == (length, length)
    assert contacts.dtype == np.float32
    assert np.allclose(contacts, contacts.T, atol=1e-5)
    assert np.allclose(np.diag(contacts), 0.0)


def test_aa_token_ids_maps_twenty_amino_acids() -> None:
    class FakeTok:
        def convert_tokens_to_ids(self, token: str) -> int:
            return ord(token)

    ids = entrypoint.aa_token_ids(FakeTok())
    assert len(ids) == 20
    assert ids[0] == ord("A")


def test_parser_has_contacts_subcommand() -> None:
    args = entrypoint.build_parser().parse_args(
        ["contacts", "--input", "/in/seqs.fasta", "--output", "/out"]
    )
    assert args.command == "contacts"
    assert args.method == "categorical-jacobian"
    assert args.func is entrypoint.cmd_contacts
