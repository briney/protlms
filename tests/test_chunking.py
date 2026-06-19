"""Unit tests for client-side input chunking (no Docker)."""

from __future__ import annotations

import pytest

from plms.chunking import _check_unique_ids, _input_fingerprint, chunk_records
from plms.exceptions import FastaError, InvalidRequestError
from plms.io import FastaRecord


def _rec(i: int) -> FastaRecord:
    return FastaRecord(id=f"s{i}", description=f"s{i}", sequence="ACDE")


def test_chunk_records_even_division() -> None:
    chunks = chunk_records([_rec(i) for i in range(4)], 2)
    assert [[r.id for r in c] for c in chunks] == [["s0", "s1"], ["s2", "s3"]]


def test_chunk_records_remainder() -> None:
    chunks = chunk_records([_rec(i) for i in range(5)], 2)
    assert [len(c) for c in chunks] == [2, 2, 1]


def test_chunk_records_size_ge_len_is_single_chunk() -> None:
    chunks = chunk_records([_rec(i) for i in range(3)], 10)
    assert len(chunks) == 1 and len(chunks[0]) == 3


def test_chunk_records_size_one() -> None:
    chunks = chunk_records([_rec(i) for i in range(3)], 1)
    assert [len(c) for c in chunks] == [1, 1, 1]


def test_chunk_records_rejects_size_below_one() -> None:
    with pytest.raises(InvalidRequestError):
        chunk_records([_rec(0)], 0)


def test_check_unique_ids_raises_on_duplicate() -> None:
    with pytest.raises(FastaError):
        _check_unique_ids([_rec(0), _rec(0)])


def test_input_fingerprint_is_order_sensitive_and_stable() -> None:
    a = [_rec(0), _rec(1)]
    assert _input_fingerprint(a) == _input_fingerprint([_rec(0), _rec(1)])
    assert _input_fingerprint(a) != _input_fingerprint([_rec(1), _rec(0)])
