"""Unit tests for the shared ESM container entrypoint's pure helpers.

These exercise the dependency-light logic (no torch/transformers needed) by
loading the standalone entrypoint module by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENTRYPOINT = Path(__file__).parents[1] / "containers" / "esm" / "entrypoint.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("esm_entrypoint", _ENTRYPOINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entrypoint = _load_entrypoint()


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


def test_jacobian_to_contacts_shape_symmetry_zero_diag() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    length = 7
    jac = rng.standard_normal((length, 20, length, 20))
    contacts = entrypoint.jacobian_to_contacts(jac)
    assert contacts.shape == (length, length)
    assert contacts.dtype == np.float32
    assert np.allclose(contacts, contacts.T, atol=1e-5)
    assert np.allclose(np.diag(contacts), 0.0)


def test_jacobian_to_contacts_invariant_to_aa_permutation() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    length = 5
    jac = rng.standard_normal((length, 20, length, 20))
    perm = rng.permutation(20)
    permuted = jac[:, perm][:, :, :, perm]
    a = entrypoint.jacobian_to_contacts(jac)
    b = entrypoint.jacobian_to_contacts(permuted)
    assert np.allclose(a, b, atol=1e-4)


def test_aa_token_ids_maps_twenty_amino_acids() -> None:
    class FakeTok:
        def convert_tokens_to_ids(self, token: str) -> int:
            return ord(token)

    ids = entrypoint.aa_token_ids(FakeTok())
    assert len(ids) == 20
    assert ids[0] == ord("A")
    assert ids[-1] == ord("Y")


def test_write_contacts_outputs_saves_npy_and_artifacts(tmp_path: Path) -> None:
    import numpy as np

    maps = {"seqA": np.zeros((4, 4), dtype=np.float32), "seqB": np.ones((3, 3), dtype=np.float32)}
    artifacts = entrypoint.write_contacts_outputs(tmp_path, maps)
    assert (tmp_path / "contacts" / "seqA.npy").is_file()
    assert (tmp_path / "contacts" / "seqB.npy").is_file()
    kinds = {a["kind"] for a in artifacts}
    assert kinds == {"contact_map"}
    by_id = {a["record_ids"][0]: a for a in artifacts}
    assert by_id["seqA"]["shape"] == [4, 4]
    assert by_id["seqA"]["path"] == "contacts/seqA.npy"


def test_parser_has_contacts_subcommand() -> None:
    args = entrypoint.build_parser().parse_args(
        ["contacts", "--input", "/in/seqs.fasta", "--output", "/out"]
    )
    assert args.command == "contacts"
    assert args.method == "categorical-jacobian"
    assert args.func is entrypoint.cmd_contacts
