"""Tests for PDB parsing and the contact metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


def test_long_range_precision_at_l_known_case() -> None:
    from protlms.eval.contacts import long_range_precision_at_l

    n = 6
    resnums = np.arange(n)  # 0..5; long-range with sep=3 => |i-j| >= 3
    # eligible upper-tri pairs (sep>=3): (0,3),(0,4),(0,5),(1,4),(1,5),(2,5)
    true = np.zeros((n, n), dtype=bool)
    true[0, 3] = true[3, 0] = True  # a real long-range contact
    true[1, 5] = true[5, 1] = True  # another
    pred = np.zeros((n, n), dtype=float)
    pred[0, 3] = 0.9  # top-ranked, true
    pred[0, 4] = 0.8  # 2nd, false
    pred[1, 5] = 0.7  # 3rd, true
    pred = (pred + pred.T) / 2
    # top = 2 => picks (0,3) true and (0,4) false => precision 0.5
    assert long_range_precision_at_l(pred, true, resnums, sep=3, top=2) == 0.5
    # top defaults to N=6 but only 3 nonzero-scored + rest 0; top clipped to eligible count (6)
    # true positives among all 6 eligible = 2 => 2/6
    assert long_range_precision_at_l(pred, true, resnums, sep=3) == pytest.approx(2 / 6)


def test_long_range_precision_at_l_no_eligible_pairs_is_nan() -> None:
    from protlms.eval.contacts import long_range_precision_at_l

    n = 3
    pred = np.zeros((n, n))
    true = np.zeros((n, n), dtype=bool)
    resnums = np.arange(n)
    assert np.isnan(long_range_precision_at_l(pred, true, resnums, sep=24))


def test_long_range_precision_at_l_default_top_is_residue_count_not_pair_count() -> None:
    from protlms.eval.contacts import long_range_precision_at_l

    n = 6
    resnums = np.arange(n)  # sep=1 => all 15 upper-triangle pairs eligible
    pred = np.zeros((n, n), dtype=float)
    # top-6 pairs by score (descending): 3 true, 3 false among them
    pred[0, 1] = 0.9  # true
    pred[0, 2] = 0.8  # false
    pred[0, 3] = 0.7  # true
    pred[0, 4] = 0.6  # false
    pred[0, 5] = 0.5  # true
    pred[1, 2] = 0.4  # false
    true = np.zeros((n, n), dtype=bool)
    for a, b in [(0, 1), (0, 3), (0, 5), (1, 3), (2, 4), (3, 5)]:
        true[a, b] = True  # 6 true total; the last 3 have pred 0 => ranked below the top-6
    # default top must be L = N = 6, NOT the 15 eligible pairs.
    # precision@6 = 3 true in the top-6 / 6 = 0.5 (would be 6/15 if it scored all eligible pairs).
    assert long_range_precision_at_l(pred, true, resnums, sep=1) == 0.5


def test_long_range_precision_at_l_shape_mismatch_raises() -> None:
    from protlms.eval.contacts import long_range_precision_at_l

    with pytest.raises(ValueError):
        long_range_precision_at_l(np.zeros((4, 4)), np.zeros((4, 4), dtype=bool), np.arange(3))


@pytest.mark.parametrize("bad_top", [0, -1])
def test_long_range_precision_at_l_rejects_nonpositive_top(bad_top: int) -> None:
    from protlms.eval.contacts import long_range_precision_at_l
    from protlms.exceptions import InvalidRequestError

    n = 4
    with pytest.raises(InvalidRequestError):
        long_range_precision_at_l(
            np.zeros((n, n)), np.zeros((n, n), dtype=bool), np.arange(n), top=bad_top
        )
