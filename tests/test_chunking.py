"""Unit tests for client-side input chunking (no Docker)."""

from __future__ import annotations

import json as _json
from pathlib import Path

import numpy as np
import pytest

from protlms.chunking import (
    _check_unique_ids,
    _input_fingerprint,
    chunk_records,
    merge_chunk_outputs,
)
from protlms.contract import Result
from protlms.exceptions import FastaError, InvalidRequestError
from protlms.io import FastaRecord


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


# ---------------------------------------------------------------------------
# Task 2: merge tests
# ---------------------------------------------------------------------------


def _write_chunk_result(cdir: Path, capability: str, artifacts: list[dict], n: int) -> Result:
    cdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract_version": "0.3",
        "capability": capability,
        "model_name": "esm2_t6_8M",
        "n_input_records": n,
        "n_output_records": n,
        "artifacts": artifacts,
        "warnings": [],
        "params": {"pooling": "mean"},
    }
    (cdir / "result.json").write_text(_json.dumps(payload))
    return Result.model_validate(payload)


def test_merge_pooled_embeddings(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pairs = []
    for ci, ids in enumerate([["a", "b"], ["c"]]):
        cdir = out / "chunks" / f"chunk_{ci:04d}"
        cdir.mkdir(parents=True)
        np.savez(cdir / "embeddings.npz", **{i: np.ones(320, dtype=np.float32) for i in ids})
        res = _write_chunk_result(
            cdir, "embed", [{"path": "embeddings.npz", "kind": "pooled_embeddings"}], len(ids)
        )
        pairs.append((cdir, res))
    merged = merge_chunk_outputs("embed", pairs, out)
    assert merged.n_input_records == 3 and merged.n_output_records == 3
    with np.load(out / "embeddings.npz") as npz:
        assert set(npz.files) == {"a", "b", "c"}
    art = merged.artifacts[0]
    assert art.kind == "pooled_embeddings"
    assert set(art.record_ids) == {"a", "b", "c"}
    assert art.shape == [3, 320]
    assert art.dtype == "float32"
    # result.json round-trips
    assert Result.model_validate_json((out / "result.json").read_text()).n_output_records == 3


def test_merge_per_residue(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pairs = []
    for ci, rid in enumerate(["a", "b"]):
        cdir = out / "chunks" / f"chunk_{ci:04d}"
        (cdir / "per_residue").mkdir(parents=True)
        np.save(cdir / "per_residue" / f"{rid}.npy", np.ones((4, 320), dtype=np.float32))
        res = _write_chunk_result(
            cdir,
            "embed",
            [
                {
                    "path": f"per_residue/{rid}.npy",
                    "kind": "per_residue_embeddings",
                    "record_ids": [rid],
                    "shape": [4, 320],
                    "dtype": "float32",
                }
            ],
            1,
        )
        pairs.append((cdir, res))
    merged = merge_chunk_outputs("embed", pairs, out)
    assert {a.path for a in merged.artifacts} == {"per_residue/a.npy", "per_residue/b.npy"}
    assert (out / "per_residue" / "a.npy").is_file()
    assert np.load(out / "per_residue" / "b.npy").shape == (4, 320)


def test_merge_likelihoods(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    header = "record_id,seq_len,log_likelihood,mean_log_likelihood,perplexity"
    pairs = []
    for ci, rows in enumerate([["a,4,-1.0,-0.25,1.28"], ["b,5,-2.0,-0.4,1.49"]]):
        cdir = out / "chunks" / f"chunk_{ci:04d}"
        cdir.mkdir(parents=True)
        (cdir / "likelihoods.csv").write_text("\n".join([header, *rows]) + "\n")
        res = _write_chunk_result(
            cdir, "likelihood", [{"path": "likelihoods.csv", "kind": "likelihoods_csv"}], 1
        )
        pairs.append((cdir, res))
    merged = merge_chunk_outputs("likelihood", pairs, out)
    lines = (out / "likelihoods.csv").read_text().strip().splitlines()
    assert lines[0] == header
    assert {ln.split(",")[0] for ln in lines[1:]} == {"a", "b"}
    assert set(merged.artifacts[0].record_ids) == {"a", "b"}


def test_merge_generated_fasta(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pairs = []
    for ci, body in enumerate([">p0__sample0\nACDE\n", ">p1__sample0\nGHIK\n"]):
        cdir = out / "chunks" / f"chunk_{ci:04d}"
        cdir.mkdir(parents=True)
        (cdir / "generated.fasta").write_text(body)
        res = _write_chunk_result(
            cdir, "generate", [{"path": "generated.fasta", "kind": "generated_fasta"}], 1
        )
        pairs.append((cdir, res))
    merged = merge_chunk_outputs("generate", pairs, out)
    assert set(merged.artifacts[0].record_ids) == {"p0__sample0", "p1__sample0"}
    assert (out / "generated.fasta").read_text().count(">") == 2


# ---------------------------------------------------------------------------
# Task 3: orchestration tests
# ---------------------------------------------------------------------------

from protlms.chunking import CHUNKS_DIRNAME, run_chunked  # noqa: E402


class _CountingRunChunk:
    """A run_chunk closure that writes a minimal embed output and counts calls."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, chunk, cdir):  # noqa: ANN001
        self.calls.append([r.id for r in chunk])
        cdir.mkdir(parents=True, exist_ok=True)
        np.savez(cdir / "embeddings.npz", **{r.id: np.ones(320, dtype=np.float32) for r in chunk})
        (cdir / "result.json").write_text(
            _json.dumps(
                {
                    "contract_version": "0.3",
                    "capability": "embed",
                    "model_name": "m",
                    "n_input_records": len(chunk),
                    "n_output_records": len(chunk),
                    "artifacts": [{"path": "embeddings.npz", "kind": "pooled_embeddings"}],
                    "warnings": [],
                    "params": {},
                }
            )
        )
        return Result.model_validate_json((cdir / "result.json").read_text())


def test_run_chunked_single_chunk_runs_into_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    run = _CountingRunChunk()
    recs = [_rec(i) for i in range(3)]
    merged = run_chunked(
        capability="embed", records=recs, chunk_size=10, output_dir=out, run_chunk=run
    )
    assert len(run.calls) == 1
    assert not (out / CHUNKS_DIRNAME).exists()  # no chunk layout for a single chunk
    assert merged.n_output_records == 3


def test_run_chunked_multi_chunk_merges(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    run = _CountingRunChunk()
    recs = [_rec(i) for i in range(5)]
    merged = run_chunked(
        capability="embed", records=recs, chunk_size=2, output_dir=out, run_chunk=run
    )
    assert [len(c) for c in run.calls] == [2, 2, 1]
    assert merged.n_output_records == 5
    with np.load(out / "embeddings.npz") as npz:
        assert set(npz.files) == {f"s{i}" for i in range(5)}
    assert (out / CHUNKS_DIRNAME / "chunking.json").is_file()


def test_run_chunked_skips_completed_chunks(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    recs = [_rec(i) for i in range(5)]
    # First run completes everything.
    run_chunked(
        capability="embed",
        records=recs,
        chunk_size=2,
        output_dir=out,
        run_chunk=_CountingRunChunk(),
    )
    # Second run with the same request must not re-invoke any chunk.
    run2 = _CountingRunChunk()
    merged = run_chunked(
        capability="embed", records=recs, chunk_size=2, output_dir=out, run_chunk=run2
    )
    assert run2.calls == []  # all three chunks skipped
    assert merged.n_output_records == 5


def test_run_chunked_rejects_changed_input(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    run_chunked(
        capability="embed",
        records=[_rec(i) for i in range(5)],
        chunk_size=2,
        output_dir=out,
        run_chunk=_CountingRunChunk(),
    )
    with pytest.raises(InvalidRequestError):
        run_chunked(
            capability="embed",
            records=[_rec(i) for i in range(6)],  # different fingerprint
            chunk_size=2,
            output_dir=out,
            run_chunk=_CountingRunChunk(),
        )


def test_run_chunked_rejects_duplicate_ids(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FastaError):
        run_chunked(
            capability="embed",
            records=[_rec(0), _rec(0)],
            chunk_size=1,
            output_dir=out,
            run_chunk=_CountingRunChunk(),
        )


def test_run_chunked_failed_chunk_names_index(tmp_path: Path) -> None:
    from protlms.exceptions import ContainerExecutionError

    out = tmp_path / "out"
    out.mkdir()

    def failing(chunk, cdir):  # noqa: ANN001
        raise ContainerExecutionError("boom", error_type="InternalError", exit_code=1)

    with pytest.raises(ContainerExecutionError) as excinfo:
        run_chunked(
            capability="embed",
            records=[_rec(i) for i in range(4)],
            chunk_size=2,
            output_dir=out,
            run_chunk=failing,
        )
    assert "chunk 0" in str(excinfo.value)
