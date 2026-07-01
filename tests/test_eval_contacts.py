"""Tests for PDB parsing and the contact metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from protlms.eval.contacts import parse_pdb, true_contact_map

_PDB = Path(__file__).parent / "data" / "casp14" / "T1024.pdb"


def test_parse_pdb_extracts_sequence_and_coords() -> None:
    chain = parse_pdb(_PDB)
    n = len(chain.sequence)
    assert n > 50
    assert chain.resnums.shape == (n,)
    assert chain.cb_coords.shape == (n, 3)
    assert set(chain.sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
    assert np.all(np.diff(chain.resnums) >= 1)  # strictly increasing residue numbers


def test_true_contact_map_is_symmetric_bool_with_zero_diagonal() -> None:
    chain = parse_pdb(_PDB)
    cmap = true_contact_map(chain.cb_coords)
    n = len(chain.sequence)
    assert cmap.shape == (n, n)
    assert cmap.dtype == bool
    assert np.array_equal(cmap, cmap.T)
    assert not cmap.diagonal().any()
    assert cmap.sum() > 0  # a real fold has contacts


def test_true_contact_map_threshold_monotone() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    cmap = true_contact_map(coords, threshold=8.0)
    assert cmap[0, 1] and not cmap[0, 2]
