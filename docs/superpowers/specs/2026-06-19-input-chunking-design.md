# Design: input chunking (client-side sharding for scale)

> **Status:** Approved design. Completes Phase 1 of the plms roadmap — the
> "implement input chunking for scale" item. Purely a client-side feature; no
> contract, container, or image changes.

## Context

Today each `Model.embed`/`likelihood`/`generate`/`score` call maps to exactly one
`docker run`: stage one input file into `/in`, run, parse one `/out`. The
container loads the model once and processes the whole file (VISION's one-shot
batch model). That amortizes model-load cost across a file, but a single logical
request is still one container run — so a million-sequence FASTA is one enormous
job that can't be split, resumed after a crash, or (later) fanned out.

**Input chunking** keeps the contract unchanged and adds a client-side layer that
splits a large input into chunks, runs each as its own container job, and merges
the outputs into one logical result. VISION places this boundary in the client
and states the chunk-job seam is exactly where parallel/multi-GPU fan-out will
slot in later without an API change.

## Locked decisions

1. **Capabilities = `embed`, `likelihood`, `generate`** (the FASTA-record inputs,
   uniform per-record output merging). `score` (variants CSV, `wt_sequence`-grouped)
   is **out of scope** for v1.
2. **Chunk size = by record count.** New `chunk_size: int | None` parameter;
   `None` = current single-run behavior (fully backward-compatible). Residue-budget
   packing is a noted future enhancement.
3. **Sequential execution** in v1. The per-chunk-job boundary is where parallel
   fan-out slots in later; no API change required for that.
4. **Resumability in v1**: stable per-chunk output subdirs + completion detection,
   so re-running a job skips already-finished chunks.
5. **Merge-on-disk into one `output_dir`**: the merged result preserves the
   existing `Result` + handle contract (one dir, one `result.json`), so the
   `EmbeddingResult`/`LikelihoodResult`/`GenerationResult` handles are unchanged.
6. **No contract/container/`score` changes.**

## API & engagement

`chunk_size: int | None = None` is added to `Model.embed`, `Model.likelihood`,
and `Model.generate` (and surfaced as `--chunk-size` on the corresponding CLI
commands).

- `chunk_size=None` → the existing single-run path, byte-for-byte current
  behavior.
- `chunk_size=N` (`N ≥ 1`) → split the input FASTA records, in file order, into
  consecutive chunks of ≤ `N` records.
- If the split yields a **single chunk** (records ≤ `N`), short-circuit to the
  existing single-run path — no chunk layout for small inputs.

Chunking operates on `FastaRecord`s already parsed by `read_fasta` (so the
existing duplicate-id and empty-input checks still apply up front).

## Execution, output layout & resumability

When the split yields >1 chunk, each chunk runs into a stable subdir of the
(resolved) `output_dir`:

```
<output_dir>/
  chunks/
    chunking.json          # input fingerprint + chunk_size + capability
    chunk_0000/            # one full container output: result.json + artifacts
    chunk_0001/
    ...
  result.json              # client-synthesized merged summary
  <merged artifacts>       # embeddings.npz | likelihoods.csv | generated.fasta | per_residue/*.npy
```

- **`chunking.json`** records: the capability, `chunk_size`, the input record
  count, and a hash of the ordered record ids (the input fingerprint). It is
  written before any chunk runs.
- **Skip logic:** before running chunk *i*, if `chunk_{i:04d}/result.json` exists
  and parses as a valid `Result`, the chunk is skipped (already complete).
- **Fingerprint validation:** on a resumed run, an existing `chunking.json` whose
  fingerprint or `chunk_size` differs from the current request raises
  `InvalidRequestError` rather than silently resuming a stale job. (Chunking is
  deterministic for a given input + `chunk_size`, so chunk *i* always holds the
  same records across runs.)
- **Failure:** a non-zero chunk container exit raises `ContainerExecutionError`
  carrying the failing chunk index (and the usual structured error fields).
  Completed chunk subdirs remain on disk, so a re-run resumes.
- Resumability **benefits require a persistent (user-provided) `output_dir`**;
  with the default temporary directory chunking still works but each call starts
  fresh (the temp dir is new), so there is nothing to resume.

Chunk numbering is zero-padded to 4 digits (`chunk_0000`…); jobs exceeding 10000
chunks still sort/path correctly (Python int formatting widens the field).

## Merge semantics (per artifact kind)

After all chunks are present, the merge step reads the chunk subdirs and writes
merged artifacts + a synthesized `result.json` into `output_dir`. The merge is
idempotent and cheap relative to inference, so it always re-runs:

| Capability / artifact | Merge |
|---|---|
| pooled embeddings (`mean`/`cls`) | Union the per-chunk `embeddings.npz` dicts → one `embeddings.npz`. |
| per-residue (`none`) | Copy each chunk's `per_residue/<id>.npy` into `<output_dir>/per_residue/`. |
| likelihoods | Concatenate CSV data rows (single header) in chunk order → `likelihoods.csv`. |
| generated | Concatenate FASTA records in chunk order → `generated.fasta`. |

**Synthesized `result.json`** (mirrors `contract.Result`): `contract_version`,
`capability`, `model_name` carried from the chunks (identical); `n_input_records`
and `n_output_records` summed; `artifacts` rebuilt to point at the merged files
with unioned `record_ids` and (for pooled) shape `[total, embedding_dim]`;
`warnings` concatenated in chunk order; `params` carried from chunk 0 (the
request params are identical across chunks).

Because record ids are unique across the whole input (the duplicate-id check runs
before splitting), no key collisions occur when unioning npz dicts or per-residue
files.

**Documented caveat (generate):** with a fixed `--seed`, reproducibility holds
**for a fixed `chunk_size`**. Each chunk container reseeds and samples over its
own prompts, so changing `chunk_size` changes how prompts group across containers
and therefore the RNG trajectory. Same input + same `chunk_size` + same seed →
identical output.

## Module structure

- **New `src/plms/chunking.py`** owns the feature: `chunk_records(records,
  chunk_size) -> list[list[FastaRecord]]` (pure split); the orchestration that
  runs/skip-resumes chunks and writes `chunking.json`; and `merge_chunk_outputs`
  per capability. It reuses `io.py` (FASTA/CSV/npz read+write, `read_result`,
  `stage_inputs`) and the existing `Runner`. It depends on no Docker specifics.
- **`models.py`**: `embed`/`likelihood`/`generate` gain `chunk_size`; when set and
  the split is >1 chunk, they delegate to the chunking orchestrator (passing a
  per-chunk run closure that mirrors `_run` into a chunk subdir); otherwise they
  take the existing single-run path. The returned `*Result` handles are unchanged.
- **`cli.py`**: `--chunk-size` option added to `embed`/`likelihood`/`generate`.
- **`io.py`**: small additions if needed for merged-artifact writing (e.g. a
  helper to write a synthesized `result.json`); readers already exist.
- **Unchanged:** `contract.py`, `runner.py`, `registry.py`, every container, and
  `Model.score`.

## Testing

**Unit (FakeRunner / fixtures, no Docker):**
- `chunk_records`: even division, remainder, `chunk_size ≥ len` (single chunk),
  `chunk_size == 1`, order preserved.
- Merge correctness from fabricated chunk subdirs (small real `embeddings.npz` /
  `per_residue/*.npy` / `likelihoods.csv` / `generated.fasta` + per-chunk
  `result.json`): merged artifact has all ids, summed counts, chunk-order rows;
  synthesized `result.json` validates against `contract.Result`.
- Resumability: a pre-existing valid `chunk_0000/result.json` causes that chunk's
  run closure **not** to be invoked (assert via a counting FakeRunner); a
  `chunking.json` with a mismatched fingerprint or `chunk_size` raises
  `InvalidRequestError`.
- `chunk_size=None` and single-chunk inputs take the unchanged single-run path
  (no `chunks/` dir created).

**Integration (docker-gated, `@pytest.mark.slow`, `PLMS_RUN_DOCKER_TESTS=1`):**
- `embed` with `chunk_size=2` on `tests/data/tiny.fasta` (3 records → 2 chunks)
  against `esm2-8m`: merged `pooled()` has all 3 ids with shape `(320,)`, and the
  vectors match an unchunked `embed` run within `1e-5`.
- `likelihood` with `chunk_size=2` on the same input: merged rows cover all 3
  records and match the unchunked run.

## Verification

```bash
plms embed esm2-8m big.fasta -o out/ --chunk-size 1000
# crash/interrupt, then re-run the same command -> completed chunks are skipped
pytest                                   # unit (gated integration skipped)
PLMS_RUN_DOCKER_TESTS=1 pytest -m slow   # chunked == unchunked, end-to-end
ruff check src/ tests/ && ruff format --check src/ tests/ && ty check src/
```

## Risks

- **Merge fidelity:** the synthesized `result.json` must satisfy `contract.Result`
  and the handle readers exactly (artifact `kind`/`path`/`record_ids`/`shape`).
  Covered by the merge unit tests (validate against the pydantic model) and the
  integration equality check vs. the unchunked run.
- **Resumption safety:** skipping a chunk on a changed input is the main hazard;
  the `chunking.json` fingerprint + `chunk_size` check is the guard, tested for
  the mismatch-raises path.
- **Partial/corrupt chunk output:** a chunk dir present but with malformed
  `result.json` must be treated as not-complete (re-run), not skipped — the skip
  check requires a *valid* parse.

## Out of scope (later)

`score` chunking (CSV / `wt_sequence`-grouped); residue-budget (length-balanced)
chunk packing; parallel / multi-GPU chunk fan-out (the seam is designed for it);
streaming/iterator result handles that avoid materializing merged artifacts on
disk; cross-request result caching.
