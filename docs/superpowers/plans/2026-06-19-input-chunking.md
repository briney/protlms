# Input Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a client-side `chunk_size` option to `embed`/`likelihood`/`generate` that shards a large FASTA into multiple container runs, runs them sequentially (resuming completed chunks), and merges the outputs into one logical result — with no contract, container, or `score` changes.

**Architecture:** A new `src/plms/chunking.py` owns three concerns — splitting records, orchestrating per-chunk runs with resume, and merging per-capability outputs. `models.py` decides chunked-vs-single per call and delegates, passing a closure that runs one chunk into a given directory. The merged result is written to disk so the existing `EmbeddingResult`/`LikelihoodResult`/`GenerationResult` handles work unchanged.

**Tech Stack:** Python 3.11+, pydantic (`contract.Result`/`OutputArtifact`), numpy (npz merge), the existing `Runner` protocol. No new dependencies.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-06-19-input-chunking-design.md`:

- **`chunk_size: int | None = None`** added to `Model.embed`, `Model.likelihood`, `Model.generate` only. `None` = the exact current single-run behavior (byte-for-byte). `score`, `contract.py`, `runner.py`, `registry.py`, and every container are **untouched**.
- **Capabilities chunked: `embed`, `likelihood`, `generate`** (FASTA-record inputs). Not `score`.
- **Split by record count**, in file order, into consecutive chunks of ≤ `chunk_size`. `chunk_size < 1` → `InvalidRequestError`.
- **Single chunk short-circuits** to the existing single-run path (run directly into `output_dir`, no `chunks/` layout).
- **Sequential execution.** Merge-on-disk into one `output_dir`; merged `result.json` is client-synthesized and must validate against `contract.Result`.
- **Output layout (when >1 chunk):** `output_dir/chunks/chunk_NNNN/` (zero-padded to ≥4 digits) each a full container output; `output_dir/chunks/chunking.json` records `capability`, `chunk_size`, `n_records`, and `fingerprint` (sha256 of the ordered record ids).
- **Resume:** a chunk with a *valid* `result.json` is skipped. A `chunking.json` whose `fingerprint`/`chunk_size`/`capability` differs from the request raises `InvalidRequestError`. A malformed chunk `result.json` is treated as incomplete (re-run). A failed chunk raises `ContainerExecutionError` naming the chunk index.
- **Duplicate record ids across the whole input** raise `FastaError` before splitting.
- **New module `src/plms/chunking.py`** owns split/merge/orchestrate; reuses `io.py` + the `Runner`; no Docker specifics.
- **Generate seed caveat:** with a fixed seed, reproducibility holds only for a fixed `chunk_size` (documented in code/docstring).
- **Quality gates before each commit:** `ruff check src/ tests/`, `ruff format src/ tests/`, `ty check src/`, `pytest`. Commit style `<component>: <what changed>`, imperative.

---

### Task 1: chunking split + fingerprint helpers

Creates `src/plms/chunking.py` with the pure, torch/Docker-free helpers the later tasks build on.

**Files:**
- Create: `src/plms/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `plms.io.FastaRecord`, `plms.exceptions.{FastaError, InvalidRequestError}`.
- Produces: `chunk_records(records: list[FastaRecord], chunk_size: int) -> list[list[FastaRecord]]`; `_check_unique_ids(records: list[FastaRecord]) -> None`; `_input_fingerprint(records: list[FastaRecord]) -> str`. Module constants `CHUNKS_DIRNAME = "chunks"`, `CHUNKING_MANIFEST_NAME = "chunking.json"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chunking.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'plms.chunking'`.

- [ ] **Step 3: Create the module with the pure helpers**

Create `src/plms/chunking.py`:

```python
"""Client-side input chunking: split a large input into per-chunk container
runs and merge the outputs into one logical result.

This module is the only place that knows how to shard a request across multiple
container runs. It reuses :mod:`plms.io` for file I/O and drives runs through a
caller-supplied closure, so it depends on no Docker specifics. The contract,
the containers, and ``score`` are unaffected.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from plms.contract import ArtifactKind, OutputArtifact, Result
from plms.exceptions import FastaError, InvalidRequestError, OutputParseError
from plms.io import load_pooled_embeddings, read_fasta, read_result

if TYPE_CHECKING:
    from plms.io import FastaRecord

logger = logging.getLogger(__name__)

CHUNKS_DIRNAME = "chunks"
CHUNKING_MANIFEST_NAME = "chunking.json"


def chunk_records(records: list[FastaRecord], chunk_size: int) -> list[list[FastaRecord]]:
    """Split records into consecutive chunks of at most ``chunk_size`` (file order)."""
    if chunk_size < 1:
        raise InvalidRequestError(f"chunk_size must be >= 1, got {chunk_size}")
    return [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]


def _check_unique_ids(records: list[FastaRecord]) -> None:
    """Raise if any record id repeats across the whole input (before splitting)."""
    ids = [r.id for r in records]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise FastaError(f"duplicate record ids in input: {dupes}")


def _input_fingerprint(records: list[FastaRecord]) -> str:
    """A stable hash of the ordered record ids — the chunking input fingerprint."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunking.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
ruff format src/plms/chunking.py tests/test_chunking.py
ruff check src/plms/chunking.py tests/test_chunking.py
git add src/plms/chunking.py tests/test_chunking.py
git commit -m "chunking: record splitting + input fingerprint helpers"
```

---

### Task 2: per-capability output merge

Adds the merge layer that combines per-chunk outputs into one `output_dir` and synthesizes a `result.json`.

**Files:**
- Modify: `src/plms/chunking.py` (append merge functions)
- Test: `tests/test_chunking.py` (append merge tests)

**Interfaces:**
- Consumes: `chunk` helpers + imports from Task 1; `contract.{ArtifactKind, OutputArtifact, Result}`; `io.{load_pooled_embeddings, read_fasta}`.
- Produces: `merge_chunk_outputs(capability: str, pairs: list[tuple[Path, Result]], output_dir: Path) -> Result`. `pairs` is `(chunk_dir, chunk_result)` in chunk order. Writes merged artifacts + `output_dir/result.json` and returns the merged `Result`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chunking.py`:

```python
import json as _json
from pathlib import Path

import numpy as np

from plms.chunking import merge_chunk_outputs
from plms.contract import Result


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
            [{"path": f"per_residue/{rid}.npy", "kind": "per_residue_embeddings",
              "record_ids": [rid], "shape": [4, 320], "dtype": "float32"}],
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunking.py -k merge -v`
Expected: FAIL — `ImportError: cannot import name 'merge_chunk_outputs'`.

- [ ] **Step 3: Append the merge implementation**

Append to `src/plms/chunking.py`:

```python
def merge_chunk_outputs(
    capability: str,
    pairs: list[tuple[Path, Result]],
    output_dir: Path,
) -> Result:
    """Merge per-chunk outputs into ``output_dir`` and return a synthesized Result.

    Args:
        capability: ``embed``, ``likelihood``, or ``generate``.
        pairs: ``(chunk_dir, chunk_result)`` in chunk order.
        output_dir: Where merged artifacts and the merged ``result.json`` are written.
    """
    artifacts = _merge_artifacts(capability, pairs, output_dir)
    first = pairs[0][1]
    merged = Result(
        contract_version=first.contract_version,
        capability=first.capability,
        model_name=first.model_name,
        n_input_records=sum(r.n_input_records for _, r in pairs),
        n_output_records=sum(r.n_output_records for _, r in pairs),
        artifacts=artifacts,
        warnings=[w for _, r in pairs for w in r.warnings],
        params=first.params,
    )
    (output_dir / "result.json").write_text(merged.model_dump_json(indent=2))
    return merged


def _merge_artifacts(
    capability: str, pairs: list[tuple[Path, Result]], output_dir: Path
) -> list[OutputArtifact]:
    if capability == "embed":
        kinds = {a.kind for _, r in pairs for a in r.artifacts}
        if ArtifactKind.POOLED_EMBEDDINGS.value in kinds:
            return _merge_pooled(pairs, output_dir)
        return _merge_per_residue(pairs, output_dir)
    if capability == "likelihood":
        return [_merge_csv(pairs, output_dir, "likelihoods.csv", ArtifactKind.LIKELIHOODS_CSV)]
    if capability == "generate":
        return [_merge_fasta(pairs, output_dir)]
    raise InvalidRequestError(f"chunking does not support capability {capability!r}")


def _merge_pooled(pairs: list[tuple[Path, Result]], output_dir: Path) -> list[OutputArtifact]:
    merged: dict[str, np.ndarray] = {}
    for chunk_dir, result in pairs:
        merged.update(load_pooled_embeddings(chunk_dir, result))
    np.savez(output_dir / "embeddings.npz", **merged)
    dim = int(next(iter(merged.values())).shape[0])
    return [
        OutputArtifact(
            path="embeddings.npz",
            kind=ArtifactKind.POOLED_EMBEDDINGS.value,
            record_ids=list(merged),
            shape=[len(merged), dim],
            dtype="float32",
        )
    ]


def _merge_per_residue(pairs: list[tuple[Path, Result]], output_dir: Path) -> list[OutputArtifact]:
    pr_dir = output_dir / "per_residue"
    pr_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[OutputArtifact] = []
    for chunk_dir, result in pairs:
        for artifact in result.artifacts:
            if artifact.kind != ArtifactKind.PER_RESIDUE_EMBEDDINGS.value:
                continue
            name = Path(artifact.path).name
            shutil.copyfile(chunk_dir / artifact.path, pr_dir / name)
            artifacts.append(
                OutputArtifact(
                    path=f"per_residue/{name}",
                    kind=ArtifactKind.PER_RESIDUE_EMBEDDINGS.value,
                    record_ids=[Path(name).stem],
                    shape=artifact.shape,
                    dtype=artifact.dtype,
                )
            )
    return artifacts


def _merge_csv(
    pairs: list[tuple[Path, Result]], output_dir: Path, filename: str, kind: ArtifactKind
) -> OutputArtifact:
    header: str | None = None
    data_lines: list[str] = []
    for chunk_dir, result in pairs:
        artifact = next(a for a in result.artifacts if a.kind == kind.value)
        lines = (chunk_dir / artifact.path).read_text().splitlines()
        if not lines:
            continue
        if header is None:
            header = lines[0]
        data_lines.extend(lines[1:])
    (output_dir / filename).write_text("\n".join([header or "", *data_lines]) + "\n")
    ids = [row[0] for row in csv.reader(data_lines) if row]
    return OutputArtifact(path=filename, kind=kind.value, record_ids=ids)


def _merge_fasta(pairs: list[tuple[Path, Result]], output_dir: Path) -> OutputArtifact:
    parts: list[str] = []
    for chunk_dir, result in pairs:
        artifact = next(
            a for a in result.artifacts if a.kind == ArtifactKind.GENERATED_FASTA.value
        )
        text = (chunk_dir / artifact.path).read_text()
        if text and not text.endswith("\n"):
            text += "\n"
        parts.append(text)
    out_path = output_dir / "generated.fasta"
    out_path.write_text("".join(parts))
    ids = [rec.id for rec in read_fasta(out_path)]
    return OutputArtifact(
        path="generated.fasta", kind=ArtifactKind.GENERATED_FASTA.value, record_ids=ids
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunking.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
ruff format src/plms/chunking.py tests/test_chunking.py
ruff check src/plms/chunking.py tests/test_chunking.py
git add src/plms/chunking.py tests/test_chunking.py
git commit -m "chunking: per-capability output merge"
```

---

### Task 3: resumable chunk orchestration

Adds the orchestrator that splits, runs each chunk (skipping completed ones), validates the chunking manifest, and merges.

**Files:**
- Modify: `src/plms/chunking.py` (append orchestration)
- Test: `tests/test_chunking.py` (append orchestration tests)

**Interfaces:**
- Consumes: Task 1 + Task 2 functions; `io.read_result`; `exceptions.OutputParseError`.
- Produces: `run_chunked(*, capability: str, records: list[FastaRecord], chunk_size: int, output_dir: Path, run_chunk) -> Result`. `run_chunk` is a closure `(chunk_records, chunk_dir) -> Result` that runs one chunk into `chunk_dir` and returns its parsed `Result` (raising `ContainerExecutionError` on failure). Single chunk → runs directly into `output_dir`; >1 → `chunks/` layout with resume.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chunking.py`:

```python
from plms.chunking import CHUNKS_DIRNAME, run_chunked


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
    merged = run_chunked(capability="embed", records=recs, chunk_size=10, output_dir=out, run_chunk=run)
    assert len(run.calls) == 1
    assert not (out / CHUNKS_DIRNAME).exists()  # no chunk layout for a single chunk
    assert merged.n_output_records == 3


def test_run_chunked_multi_chunk_merges(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    run = _CountingRunChunk()
    recs = [_rec(i) for i in range(5)]
    merged = run_chunked(capability="embed", records=recs, chunk_size=2, output_dir=out, run_chunk=run)
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
    run_chunked(capability="embed", records=recs, chunk_size=2, output_dir=out, run_chunk=_CountingRunChunk())
    # Second run with the same request must not re-invoke any chunk.
    run2 = _CountingRunChunk()
    merged = run_chunked(capability="embed", records=recs, chunk_size=2, output_dir=out, run_chunk=run2)
    assert run2.calls == []  # all three chunks skipped
    assert merged.n_output_records == 5


def test_run_chunked_rejects_changed_input(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    run_chunked(capability="embed", records=[_rec(i) for i in range(5)], chunk_size=2, output_dir=out, run_chunk=_CountingRunChunk())
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
        run_chunked(capability="embed", records=[_rec(0), _rec(0)], chunk_size=1, output_dir=out, run_chunk=_CountingRunChunk())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunking.py -k run_chunked -v`
Expected: FAIL — `ImportError: cannot import name 'run_chunked'`.

- [ ] **Step 3: Append the orchestration implementation**

Append to `src/plms/chunking.py`:

```python
def run_chunked(
    *,
    capability: str,
    records: list[FastaRecord],
    chunk_size: int,
    output_dir: Path,
    run_chunk,  # noqa: ANN001 - closure: (list[FastaRecord], Path) -> Result
) -> Result:
    """Run a request in chunks, resuming completed chunks, and merge the outputs.

    With a single chunk this runs directly into ``output_dir`` (no ``chunks/``
    layout), matching the unchunked path. With more than one chunk each runs into
    ``output_dir/chunks/chunk_NNNN/``; a chunk whose ``result.json`` already parses
    is skipped. ``run_chunk`` runs one chunk into a directory and returns its
    parsed ``Result`` (raising on a failed run).
    """
    _check_unique_ids(records)
    chunks = chunk_records(records, chunk_size)
    if len(chunks) == 1:
        return run_chunk(chunks[0], output_dir)

    chunks_dir = output_dir / CHUNKS_DIRNAME
    chunks_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_manifest(chunks_dir, capability, chunk_size, records)

    pairs: list[tuple[Path, Result]] = []
    for index, chunk in enumerate(chunks):
        chunk_dir = chunks_dir / f"chunk_{index:04d}"
        done = _completed_result(chunk_dir)
        if done is not None:
            logger.info("chunk %d/%d already complete; skipping", index + 1, len(chunks))
            pairs.append((chunk_dir, done))
            continue
        logger.info("running chunk %d/%d (%d records)", index + 1, len(chunks), len(chunk))
        pairs.append((chunk_dir, run_chunk(chunk, chunk_dir)))
    return merge_chunk_outputs(capability, pairs, output_dir)


def _completed_result(chunk_dir: Path) -> Result | None:
    """Return a chunk's parsed Result if present and valid, else None (re-run)."""
    if not (chunk_dir / "result.json").is_file():
        return None
    try:
        return read_result(chunk_dir)
    except OutputParseError:
        return None


def _validate_or_write_manifest(
    chunks_dir: Path, capability: str, chunk_size: int, records: list[FastaRecord]
) -> None:
    path = chunks_dir / CHUNKING_MANIFEST_NAME
    current = {
        "capability": capability,
        "chunk_size": chunk_size,
        "n_records": len(records),
        "fingerprint": _input_fingerprint(records),
    }
    if path.is_file():
        prev = json.loads(path.read_text())
        if (
            prev.get("fingerprint") != current["fingerprint"]
            or prev.get("chunk_size") != chunk_size
            or prev.get("capability") != capability
        ):
            raise InvalidRequestError(
                f"chunking manifest in {chunks_dir} does not match this request "
                "(input or chunk_size changed); use a fresh output_dir"
            )
        return
    path.write_text(json.dumps(current, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunking.py -v`
Expected: PASS (all chunking unit tests).

- [ ] **Step 5: Format, lint, type-check, commit**

```bash
ruff format src/plms/chunking.py tests/test_chunking.py
ruff check src/plms/chunking.py tests/test_chunking.py
ty check src/
git add src/plms/chunking.py tests/test_chunking.py
git commit -m "chunking: resumable chunk orchestration"
```

---

### Task 4: wire `chunk_size` into the Model layer

Threads `chunk_size` through `embed`/`likelihood`/`generate`, refactoring the run path so a chunk can run into a chosen directory.

**Files:**
- Modify: `src/plms/models.py`
- Test: `tests/test_models.py` (append chunking tests)

**Interfaces:**
- Consumes: `plms.chunking.run_chunked`; existing `stage_inputs`, `read_result`, `RunSpec`, `_resolve_output_dir`, `_raise_container_error`.
- Produces: `chunk_size: int | None = None` keyword on `Model.embed`, `Model.likelihood`, `Model.generate`. New private `Model._run_into_dir(capability, staging, extra_args, out_dir, use_gpu) -> Result` and `Model._run_chunked(capability, records, extra_args, output_dir, use_gpu, chunk_size) -> tuple[Result, Path, TemporaryDirectory | None]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
def test_embed_chunked_merges_all_records(tmp_path: Path) -> None:
    model = _load()
    fasta = tmp_path / "many.fasta"
    fasta.write_text("".join(f">s{i}\nACDEFG\n" for i in range(5)))
    result = model.embed(fasta, pooling="mean", output_dir=tmp_path / "out", chunk_size=2)
    pooled = result.pooled()
    assert set(pooled) == {f"s{i}" for i in range(5)}
    assert result.result.n_output_records == 5
    assert (tmp_path / "out" / "chunks" / "chunk_0000").is_dir()


def test_embed_chunk_size_none_keeps_single_run(fasta: Path, tmp_path: Path) -> None:
    model = _load()
    out = tmp_path / "out"
    model.embed(fasta, pooling="mean", output_dir=out, chunk_size=None)
    assert not (out / "chunks").exists()


def test_embed_single_chunk_short_circuits(fasta: Path, tmp_path: Path) -> None:
    model = _load()  # the `fasta` fixture has 2 records
    out = tmp_path / "out"
    model.embed(fasta, pooling="mean", output_dir=out, chunk_size=10)
    assert not (out / "chunks").exists()  # records <= chunk_size => single run


def test_likelihood_chunked_merges_rows(tmp_path: Path) -> None:
    model = _load()
    fasta = tmp_path / "many.fasta"
    fasta.write_text("".join(f">s{i}\nACDEFG\n" for i in range(4)))
    result = model.likelihood(fasta, output_dir=tmp_path / "out", chunk_size=2)
    rows = {r["record_id"] for r in result.rows()}
    assert rows == {f"s{i}" for i in range(4)}


def test_generate_chunked_merges_samples(tmp_path: Path) -> None:
    model = _load(capabilities=["embed", "likelihood", "generate"])
    prompts = tmp_path / "p.fasta"
    prompts.write_text("".join(f">p{i}\nAC\n" for i in range(3)))
    result = model.generate(prompts, num_samples=2, output_dir=tmp_path / "out", chunk_size=2)
    ids = {r.id for r in result.sequences()}
    assert ids == {f"p{i}__sample{k}" for i in range(3) for k in range(2)}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -k chunk -v`
Expected: FAIL — `TypeError: embed() got an unexpected keyword argument 'chunk_size'`.

- [ ] **Step 3: Add the import and the run helpers**

In `src/plms/models.py`, add to the `plms.chunking` import (new line after the `plms.registry` import near the top):

```python
from plms.chunking import run_chunked
```

Replace the existing `_run` method (currently at `models.py:318-346`) with this refactor plus the two new helpers:

```python
    def _run_into_dir(
        self,
        capability: Capability,
        staging,  # contextmanager[StagedInput]
        extra_args: list[str],
        out_dir: Path,
        use_gpu: bool,
    ) -> Result:
        """Run one container job into ``out_dir`` and return its parsed Result."""
        out_dir.mkdir(parents=True, exist_ok=True)
        with staging as staged:
            command = [
                capability.value,
                "--input",
                staged.container_input_path,
                "--output",
                "/out",
                *extra_args,
            ]
            spec = RunSpec(
                image=self._entry.image,
                command=command,
                input_dir=staged.input_dir,
                output_dir=out_dir,
                use_gpu=use_gpu,
            )
            run_result = self._runner.run(spec)
        if run_result.exit_code != 0:
            self._raise_container_error(run_result)
        return read_result(out_dir)

    def _run(
        self,
        capability: Capability,
        staging,  # contextmanager[StagedInput]
        extra_args: list[str],
        output_dir: Path | None,
        use_gpu: bool,
    ) -> tuple[Result, Path, tempfile.TemporaryDirectory | None]:
        out_dir, keep = self._resolve_output_dir(output_dir)
        result = self._run_into_dir(capability, staging, extra_args, out_dir, use_gpu)
        return result, out_dir, keep

    def _run_chunked(
        self,
        capability: Capability,
        records: list[FastaRecord],
        extra_args: list[str],
        output_dir: Path | None,
        use_gpu: bool,
        chunk_size: int,
    ) -> tuple[Result, Path, tempfile.TemporaryDirectory | None]:
        out_dir, keep = self._resolve_output_dir(output_dir)

        def run_chunk(chunk: list[FastaRecord], chunk_dir: Path) -> Result:
            return self._run_into_dir(capability, stage_inputs(chunk), extra_args, chunk_dir, use_gpu)

        result = run_chunked(
            capability=capability.value,
            records=records,
            chunk_size=chunk_size,
            output_dir=out_dir,
            run_chunk=run_chunk,
        )
        return result, out_dir, keep
```

Note: `FastaRecord` is already imported under `TYPE_CHECKING` in `models.py`; the runtime annotation `list[FastaRecord]` is fine because the module has `from __future__ import annotations`.

- [ ] **Step 4: Thread `chunk_size` through the three public methods**

In `Model.embed` (`models.py:132`), add `chunk_size: int | None = None` as the last keyword parameter, and replace the run dispatch (the `result, out_dir, keep = self._run(...)` line) with:

```python
        if chunk_size is not None:
            result, out_dir, keep = self._run_chunked(
                Capability.EMBED, records, extra, output_dir, use_gpu, chunk_size
            )
        else:
            result, out_dir, keep = self._run(
                Capability.EMBED, stage_inputs(records), extra, output_dir, use_gpu
            )
```

In `Model.likelihood` (`models.py:169`), add `chunk_size: int | None = None`, and replace its run dispatch with:

```python
        if chunk_size is not None:
            result, out_dir, keep = self._run_chunked(
                Capability.LIKELIHOOD, records, extra, output_dir, use_gpu, chunk_size
            )
        else:
            result, out_dir, keep = self._run(
                Capability.LIKELIHOOD, stage_inputs(records), extra, output_dir, use_gpu
            )
```

In `Model.generate` (`models.py:234`), add `chunk_size: int | None = None`, and replace its run dispatch with:

```python
        if chunk_size is not None:
            result, out_dir, keep = self._run_chunked(
                Capability.GENERATE, records, extra, output_dir, use_gpu, chunk_size
            )
        else:
            result, out_dir, keep = self._run(
                Capability.GENERATE, stage_inputs(records), extra, output_dir, use_gpu
            )
```

Add a one-line `chunk_size:` entry to each of the three docstrings' Args sections, e.g.:
`chunk_size: If set, split the input into runs of at most this many records and merge the outputs (resumable into a persistent output_dir).`

`Model.score` is **not** modified.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: PASS — the new chunking tests and all pre-existing model tests (the `_run` refactor preserves behavior).

- [ ] **Step 6: Format, lint, type-check, commit**

```bash
ruff format src/plms/models.py tests/test_models.py
ruff check src/plms/models.py tests/test_models.py
ty check src/
git add src/plms/models.py tests/test_models.py
git commit -m "models: chunk_size on embed/likelihood/generate"
```

---

### Task 5: expose `--chunk-size` on the CLI

**Files:**
- Modify: `src/plms/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: the `chunk_size` keyword from Task 4.
- Produces: a `--chunk-size` option on the `embed`, `likelihood`, and `generate` CLI commands, forwarded as `chunk_size=`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_cli.py`, first update the `FakeModel` methods to accept `chunk_size` and record it (so the CLI can forward it). Change the three signatures and their `last_call` dicts:

```python
    def embed(self, fasta, *, pooling, layers, output_dir, use_gpu, batch_size, chunk_size):  # noqa: ANN001
        FakeModel.last_call = {
            "method": "embed",
            "pooling": pooling,
            "layers": list(layers),
            "use_gpu": use_gpu,
            "output_dir": output_dir,
            "chunk_size": chunk_size,
        }
        return EmbeddingResult(
            result=_result("embed", [{"path": "embeddings.npz", "kind": "pooled_embeddings"}]),
            output_dir=Path(output_dir),
            pooling=pooling,
        )

    def likelihood(self, fasta, *, output_dir, use_gpu, batch_size, chunk_size):  # noqa: ANN001
        FakeModel.last_call = {"method": "likelihood", "use_gpu": use_gpu, "chunk_size": chunk_size}
        return LikelihoodResult(
            result=_result("likelihood", [{"path": "likelihoods.csv", "kind": "likelihoods_csv"}]),
            output_dir=Path(output_dir),
        )
```

And add `chunk_size` to `generate`'s signature (last keyword) and to its `last_call` dict:

```python
    def generate(
        self,
        prompts,
        *,
        num_samples,
        temperature,
        top_p,
        max_length,
        seed,
        output_dir,
        use_gpu,
        batch_size,
        chunk_size,
    ):  # noqa: ANN001
        FakeModel.last_call = {
            "method": "generate",
            "num_samples": num_samples,
            "temperature": temperature,
            "top_p": top_p,
            "max_length": max_length,
            "seed": seed,
            "use_gpu": use_gpu,
            "chunk_size": chunk_size,
        }
        return GenerationResult(
            result=_result("generate", [{"path": "generated.fasta", "kind": "generated_fasta"}]),
            output_dir=Path(output_dir),
        )
```

(`score`'s `FakeModel.score` is left unchanged — no `chunk_size`.)

Then append these tests:

```python
def test_embed_command_forwards_chunk_size(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("plms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app,
        ["embed", "esm2-8m", str(fasta), "-o", str(tmp_path / "out"), "--chunk-size", "1000"],
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["chunk_size"] == 1000


def test_embed_command_chunk_size_defaults_none(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("plms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(app, ["embed", "esm2-8m", str(fasta), "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["chunk_size"] is None


def test_likelihood_command_forwards_chunk_size(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("plms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app,
        ["likelihood", "esm2-8m", str(fasta), "-o", str(tmp_path / "out"), "--chunk-size", "500"],
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["chunk_size"] == 500


def test_generate_command_forwards_chunk_size(prompts: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("plms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app,
        ["generate", "progen2-small", str(prompts), "-o", str(tmp_path / "o"), "--chunk-size", "8"],
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["chunk_size"] == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -k chunk_size -v`
Expected: FAIL — Typer reports no such option `--chunk-size` (exit code 2), so `chunk_size` is never forwarded.

- [ ] **Step 3: Add the CLI option**

In `src/plms/cli.py`, add a reusable option definition next to the others (after `_BatchOpt` at `cli.py:37`):

```python
_ChunkSizeOpt = Annotated[
    int | None,
    typer.Option(
        "--chunk-size",
        help="Split the input into runs of at most N records (merged; resumable).",
    ),
]
```

Add `chunk_size: _ChunkSizeOpt = None` as the last parameter of `embed`, `likelihood`, and `generate`, and pass `chunk_size=chunk_size` into the corresponding `model_obj.embed(...)`, `model_obj.likelihood(...)`, and `model_obj.generate(...)` calls. For example, `embed`'s call becomes:

```python
        result = model_obj.embed(
            fasta,
            pooling=pooling,
            layers=_parse_layers(layers),
            output_dir=output_dir,
            use_gpu=gpu,
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
```

(Do the analogous one-line additions for `likelihood` and `generate`. `score` is not changed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (new chunk-size tests + all pre-existing CLI tests).

- [ ] **Step 5: Format, lint, type-check, commit**

```bash
ruff format src/plms/cli.py tests/test_cli.py
ruff check src/plms/cli.py tests/test_cli.py
ty check src/
git add src/plms/cli.py tests/test_cli.py
git commit -m "cli: --chunk-size on embed/likelihood/generate"
```

---

### Task 6: end-to-end chunked == unchunked integration

Proves a chunked run produces the same result as a single run against a real image.

**Files:**
- Create: `tests/test_integration_chunking.py`
- Reuses: `tests/data/tiny.fasta` (3 records), the `plms-esm2:t6_8M` image.

**Interfaces:**
- Consumes: `plms.load("esm2-8m")`, `Model.embed`/`likelihood` with `chunk_size`.

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration_chunking.py`:

```python
"""End-to-end test that chunked runs equal unchunked runs (real ESM2 image).

Gated: runs only when ``PLMS_RUN_DOCKER_TESTS=1`` and a working Docker daemon is
available. Builds the tiny ``esm2_t6_8M`` image if absent.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import plms

IMAGE = "plms-esm2:t6_8M"
REPO_ROOT = Path(__file__).parents[1]
TINY_FASTA = REPO_ROOT / "tests" / "data" / "tiny.fasta"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PLMS_RUN_DOCKER_TESTS") != "1" or not _docker_available(),
        reason="set PLMS_RUN_DOCKER_TESTS=1 and ensure a Docker daemon is available",
    ),
]


@pytest.fixture(scope="session")
def esm2_image() -> str:
    present = (
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    )
    if not present:
        subprocess.run(
            ["docker", "build", "--build-arg", "ESM2_CHECKPOINT=esm2_t6_8M", "-t", IMAGE,
             str(REPO_ROOT / "containers" / "esm2")],
            check=True,
        )
    return IMAGE


@pytest.fixture(scope="session")
def model(esm2_image: str) -> plms.Model:
    return plms.load("esm2-8m")


def test_embed_chunked_equals_unchunked(model: plms.Model, tmp_path: Path) -> None:
    plain = model.embed(TINY_FASTA, pooling="mean", output_dir=tmp_path / "plain").pooled()
    chunked = model.embed(
        TINY_FASTA, pooling="mean", output_dir=tmp_path / "chunked", chunk_size=2
    ).pooled()
    assert set(chunked) == set(plain)
    for rid in plain:
        np.testing.assert_allclose(chunked[rid], plain[rid], atol=1e-5)
    assert (tmp_path / "chunked" / "chunks" / "chunk_0001").is_dir()  # 3 records / 2 => 2 chunks


def test_likelihood_chunked_equals_unchunked(model: plms.Model, tmp_path: Path) -> None:
    plain = {r["record_id"]: r for r in model.likelihood(TINY_FASTA, output_dir=tmp_path / "p").rows()}
    chunked = {
        r["record_id"]: r
        for r in model.likelihood(TINY_FASTA, output_dir=tmp_path / "c", chunk_size=2).rows()
    }
    assert set(chunked) == set(plain)
    for rid in plain:
        assert math.isclose(
            float(chunked[rid]["log_likelihood"]), float(plain[rid]["log_likelihood"]), abs_tol=1e-4
        )
```

- [ ] **Step 2: Run the integration test (gated)**

Run: `PLMS_RUN_DOCKER_TESTS=1 pytest tests/test_integration_chunking.py -v -m slow`
Expected: PASS. The chunked embed/likelihood results match the unchunked runs within tolerance, and the `chunks/chunk_0001` directory confirms real sharding occurred.

- [ ] **Step 3: Run the full unit suite + gates**

Run: `pytest` (gated integration skipped), then `ruff check src/ tests/`, `ruff format --check src/ tests/`, `ty check src/`.
Expected: all green/clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_chunking.py
git commit -m "test: chunked==unchunked integration (esm2)"
```

---

## Self-Review

**1. Spec coverage** — every spec section maps to a task:
- `chunk_size` on embed/likelihood/generate, `None` = current behavior → Task 4 (params + dispatch) + Task 5 (CLI).
- Capabilities embed/likelihood/generate, score untouched → Tasks 4/5 (no score changes).
- Split by record count, `<1` raises → Task 1 (`chunk_records`).
- Single-chunk short-circuit → Task 3 (`run_chunked`) + Task 4 test.
- Output layout + `chunking.json` fingerprint → Task 3 (`_validate_or_write_manifest`).
- Resume / skip valid chunks / malformed = re-run / mismatch raises → Task 3 (`_completed_result`, manifest validation) + tests.
- Duplicate ids before split → Task 1 (`_check_unique_ids`) called in Task 3.
- Merge per artifact kind + synthesized result.json validating against `Result` → Task 2.
- New `chunking.py` reusing io.py + Runner → Tasks 1-3 (module), Task 4 (closure via `_run_into_dir`).
- Failed chunk raises ContainerExecutionError → Task 4 (`_run_into_dir` calls `_raise_container_error`); propagates through `run_chunk`.
- Generate seed caveat → documented in the design; the merge is order-deterministic. (No code path needs it; noted in `embed`/`generate` docstrings via the `chunk_size` line.)
- Integration equality → Task 6.

**2. Placeholder scan** — no TBD/TODO; every code step shows complete content. Docstring `Args` additions in Task 4 Step 4 are described with the exact sentence to add.

**3. Type/name consistency** — `chunk_records`, `_check_unique_ids`, `_input_fingerprint`, `merge_chunk_outputs(capability, pairs, output_dir)`, `run_chunked(*, capability, records, chunk_size, output_dir, run_chunk)`, `CHUNKS_DIRNAME`, `CHUNKING_MANIFEST_NAME`, and `Model._run_into_dir`/`_run`/`_run_chunked` signatures are consistent across the module, the model wiring, and the tests. `pairs` is `(Path, Result)` everywhere; `run_chunk` is `(list[FastaRecord], Path) -> Result` everywhere. Capability is passed as the enum to `_run_into_dir` and as `capability.value` (str) to `run_chunked`/merge.

## Notes for the implementer

- The pre-existing `tests/test_models.py` and `tests/test_cli.py` tests must continue to pass: the `_run` refactor (Task 4) keeps identical behavior, and the CLI `FakeModel` signature changes (Task 5) add only a new keyword.
- `model_dump_json` (pydantic v2) serializes `Capability`/enum fields to their string values, so the synthesized `result.json` round-trips through `Result.model_validate_json` (exercised by the Task 2 pooled test).
- Resumability benefits require a persistent (user-provided) `output_dir`; with a temporary dir each call starts fresh. The integration test uses `tmp_path` subdirs, which persist for the test's duration.
