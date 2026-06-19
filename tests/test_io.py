"""Tests for FASTA parsing, input staging, and output parsing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plms.exceptions import FastaError, OutputParseError
from plms.io import (
    FastaRecord,
    load_per_residue_embeddings,
    load_pooled_embeddings,
    read_fasta,
    read_likelihoods,
    read_result,
    stage_inputs,
    write_fasta,
)

# --- FASTA reading ---------------------------------------------------------


def test_read_fasta_parses_multiple_records(tmp_path: Path) -> None:
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">seq1 first\nACDEF\n>seq2 second\nGHIKL\n")
    records = read_fasta(fasta)
    assert [r.id for r in records] == ["seq1", "seq2"]
    assert records[0].description == "seq1 first"
    assert records[0].sequence == "ACDEF"


def test_read_fasta_joins_wrapped_sequences_and_ignores_blank_lines(tmp_path: Path) -> None:
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">seq1\nACDE\nFGHI\n\n>seq2\nKLMN\n")
    records = read_fasta(fasta)
    assert records[0].sequence == "ACDEFGHI"
    assert records[1].sequence == "KLMN"


def test_read_fasta_id_is_first_whitespace_token(tmp_path: Path) -> None:
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">sp|P01308|INS_HUMAN Insulin\nFVNQHLC\n")
    (record,) = read_fasta(fasta)
    assert record.id == "sp|P01308|INS_HUMAN"
    assert record.description == "sp|P01308|INS_HUMAN Insulin"


def test_read_fasta_sequence_before_header_raises(tmp_path: Path) -> None:
    fasta = tmp_path / "bad.fasta"
    fasta.write_text("ACDEFG\n>seq1\nGHIKL\n")
    with pytest.raises(FastaError):
        read_fasta(fasta)


def test_read_fasta_empty_file_returns_empty_list(tmp_path: Path) -> None:
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    assert read_fasta(fasta) == []


def test_write_then_read_fasta_round_trip(tmp_path: Path) -> None:
    records = [FastaRecord("a", "a desc", "ACDE"), FastaRecord("b", "b", "FGHI")]
    out = tmp_path / "out.fasta"
    write_fasta(records, out)
    back = read_fasta(out)
    assert [(r.id, r.sequence) for r in back] == [("a", "ACDE"), ("b", "FGHI")]


# --- input staging ---------------------------------------------------------


def test_stage_inputs_creates_input_mount_and_normalized_fasta() -> None:
    records = [FastaRecord("seq1", "seq1 desc", "ACDEF"), FastaRecord("seq2", "seq2", "GHIKL")]
    with stage_inputs(records) as job:
        assert job.input_dir.is_dir()
        assert job.container_input_path == "/in/seqs.fasta"
        staged = job.input_dir / job.input_filename
        assert staged.is_file()
        # normalized headers carry only the id token
        reparsed = read_fasta(staged)
        assert [r.id for r in reparsed] == ["seq1", "seq2"]
        held = job.input_dir
    # cleaned up on exit
    assert not held.exists()


def test_stage_inputs_rejects_duplicate_ids() -> None:
    records = [FastaRecord("dup", "dup", "ACDE"), FastaRecord("dup", "dup", "FGHI")]
    with pytest.raises(FastaError), stage_inputs(records):
        pass


# --- output parsing --------------------------------------------------------


def _write_result(out_dir: Path, payload: dict) -> None:
    (out_dir / "result.json").write_text(json.dumps(payload))


def test_read_result_parses_result_json(tmp_path: Path) -> None:
    _write_result(
        tmp_path,
        {
            "contract_version": "0.1",
            "capability": "embed",
            "model_name": "esm2_t6_8M",
            "n_input_records": 1,
            "n_output_records": 1,
            "artifacts": [{"path": "embeddings.npz", "kind": "pooled_embeddings"}],
        },
    )
    result = read_result(tmp_path)
    assert result.model_name == "esm2_t6_8M"


def test_read_result_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OutputParseError):
        read_result(tmp_path)


def test_load_pooled_embeddings(tmp_path: Path) -> None:
    np.savez(
        tmp_path / "embeddings.npz",
        seq1=np.ones(4, dtype=np.float32),
        seq2=np.zeros(4, dtype=np.float32),
    )
    _write_result(
        tmp_path,
        {
            "contract_version": "0.1",
            "capability": "embed",
            "model_name": "m",
            "n_input_records": 2,
            "n_output_records": 2,
            "artifacts": [{"path": "embeddings.npz", "kind": "pooled_embeddings"}],
        },
    )
    result = read_result(tmp_path)
    pooled = load_pooled_embeddings(tmp_path, result)
    assert set(pooled) == {"seq1", "seq2"}
    assert pooled["seq1"].shape == (4,)


def test_load_per_residue_embeddings(tmp_path: Path) -> None:
    pr_dir = tmp_path / "per_residue"
    pr_dir.mkdir()
    np.save(pr_dir / "seq1.npy", np.ones((5, 4), dtype=np.float32))
    np.save(pr_dir / "seq2.npy", np.ones((3, 4), dtype=np.float32))
    _write_result(
        tmp_path,
        {
            "contract_version": "0.1",
            "capability": "embed",
            "model_name": "m",
            "n_input_records": 2,
            "n_output_records": 2,
            "artifacts": [
                {"path": "per_residue/seq1.npy", "kind": "per_residue_embeddings"},
                {"path": "per_residue/seq2.npy", "kind": "per_residue_embeddings"},
            ],
        },
    )
    result = read_result(tmp_path)
    per_residue = load_per_residue_embeddings(tmp_path, result)
    assert set(per_residue) == {"seq1", "seq2"}
    assert per_residue["seq1"].shape == (5, 4)


def test_read_likelihoods_coerces_numeric_columns(tmp_path: Path) -> None:
    (tmp_path / "likelihoods.csv").write_text(
        "record_id,seq_len,pseudo_log_likelihood,mean_pseudo_log_likelihood,pseudo_perplexity\n"
        "seq1,5,-3.5,-0.7,2.01\n"
    )
    _write_result(
        tmp_path,
        {
            "contract_version": "0.1",
            "capability": "likelihood",
            "model_name": "m",
            "n_input_records": 1,
            "n_output_records": 1,
            "artifacts": [{"path": "likelihoods.csv", "kind": "likelihoods_csv"}],
        },
    )
    result = read_result(tmp_path)
    rows = read_likelihoods(tmp_path, result)
    assert rows[0]["record_id"] == "seq1"
    assert rows[0]["seq_len"] == 5
    assert rows[0]["pseudo_perplexity"] == pytest.approx(2.01)


def test_stage_file_copies_input_under_dest_name(tmp_path: Path) -> None:
    from plms.io import stage_file

    src = tmp_path / "v.csv"
    src.write_text("variant_id,wt_sequence,mutant\nv1,ACDE,A1G\n")
    with stage_file(src, "variants.csv") as job:
        staged = job.input_dir / "variants.csv"
        assert staged.is_file()
        assert job.container_input_path == "/in/variants.csv"
        held = job.input_dir
    assert not held.exists()


def test_check_csv_has_columns_ok(tmp_path: Path) -> None:
    from plms.io import check_csv_has_columns

    p = tmp_path / "v.csv"
    p.write_text("variant_id,wt_sequence,mutant\nv1,ACDE,A1G\n")
    check_csv_has_columns(p, ["variant_id", "wt_sequence", "mutant"])  # must not raise


def test_check_csv_has_columns_missing_raises(tmp_path: Path) -> None:
    from plms.exceptions import InvalidRequestError
    from plms.io import check_csv_has_columns

    p = tmp_path / "v.csv"
    p.write_text("variant_id,mutant\nv1,A1G\n")
    with pytest.raises(InvalidRequestError):
        check_csv_has_columns(p, ["variant_id", "wt_sequence", "mutant"])


def test_read_variant_scores_coerces_and_handles_blanks(tmp_path: Path) -> None:
    from plms.io import read_result, read_variant_scores

    (tmp_path / "scores.csv").write_text(
        "variant_id,mutant,n_mutations,score\nself,M1M,1,0.0\nbad,Z9Q,1,\n"
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "contract_version": "0.2",
                "capability": "score",
                "model_name": "m",
                "n_input_records": 2,
                "n_output_records": 2,
                "artifacts": [{"path": "scores.csv", "kind": "variant_scores_csv"}],
            }
        )
    )
    rows = read_variant_scores(tmp_path, read_result(tmp_path))
    assert rows[0]["variant_id"] == "self"
    assert rows[0]["n_mutations"] == 1
    assert rows[0]["score"] == 0.0
    assert rows[1]["score"] is None  # blank score for an invalid row
