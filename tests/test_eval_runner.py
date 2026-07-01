"""Tests for the CASP14 contact-evaluation runner (no Docker)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from protlms.eval.runner import (
    TargetResult,
    evaluate_contacts,
    mean_precision,
    write_results_csv,
)
from protlms.io import read_fasta

_PDB_SRC = Path(__file__).parent / "data" / "casp14" / "T1024.pdb"


class _FakeContactsResult:
    def __init__(self, maps: dict[str, np.ndarray]) -> None:
        self._maps = maps

    def maps(self) -> dict[str, np.ndarray]:
        return self._maps


class FakeModel:
    """Duck-typed stand-in for protlms.Model that returns random symmetric maps."""

    def contacts(self, fasta, **_kw):  # noqa: ANN001, ANN003
        rng = np.random.default_rng(0)
        maps = {}
        for rec in read_fasta(Path(fasta)):
            n = len(rec.sequence)
            m = rng.random((n, n)).astype(np.float32)
            maps[rec.id] = (m + m.T) / 2
        return _FakeContactsResult(maps)


def test_evaluate_contacts_returns_one_result_per_target(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    (pdb_dir / "T1024.pdb").write_bytes(_PDB_SRC.read_bytes())
    results = evaluate_contacts(FakeModel(), pdb_dir)
    assert len(results) == 1
    r = results[0]
    assert r.target_id == "T1024"
    assert r.length > 50
    assert 0.0 <= r.precision_at_l <= 1.0
    assert r.n_long_range_true > 0


def test_evaluate_contacts_respects_max_length(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    (pdb_dir / "T1024.pdb").write_bytes(_PDB_SRC.read_bytes())
    results = evaluate_contacts(FakeModel(), pdb_dir, max_length=10)
    assert results == []  # target skipped (too long)


def test_write_results_csv_and_mean(tmp_path: Path) -> None:
    results = [
        TargetResult("A", 50, 10, 0.5),
        TargetResult("B", 60, 12, 0.25),
    ]
    out = tmp_path / "r.csv"
    write_results_csv(results, out)
    lines = out.read_text().splitlines()
    assert lines[0] == "target_id,length,n_long_range_true,precision_at_l"
    assert lines[1].startswith("A,50,10,0.5")
    assert math.isclose(mean_precision(results), 0.375)
