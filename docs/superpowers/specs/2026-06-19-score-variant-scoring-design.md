# Design: `score` (variant effect scoring)

> **Status:** Approved design. Phase 1a of the plms roadmap — the third
> masked-LM capability, extending the working ESM2 image. Builds directly on the
> Phase 0 contract + client.

## Context

Phase 0 delivered the container contract plus a client that drives ESM2 for
`embed` and `likelihood`. The contract reserves `score` and `generate` but does
not implement them. This sub-project implements **`score`** — per-variant effect
scoring for masked protein language models — completing ESM2's masked-LM
capabilities and resolving the previously-open *variant input format* question
(see VISION "Open Questions"). `generate`/ProGen2, ESM-C, and input chunking are
separate later sub-projects.

The variant score for a substitution is the standard masked-LM log-odds at the
mutated position: `logP(mutant_aa | context) − logP(wildtype_aa | context)`.
Higher = more favorable under the model.

## Principles & approach

The scoring math lives **in the container**, not the client. The rejected
alternative — the client pulls raw logits and computes log-prob differences —
would force the client to understand tokenization and model internals, violating
the "lightweight client / opaque internals / no ML dependencies" principles. So
the container computes scores; the client validates input, drives the run, and
parses a scores CSV. This mirrors the existing `embed`/`likelihood` flow exactly;
no new architecture is introduced.

## Contract changes (0.1 → 0.2, minor bump)

Same major version, so a 0.1 client and a 0.2 image remain mutually readable
(unknown fields ignored); a 0.2 client warns-but-runs against a 0.1 image.

- **`Capability.SCORE`** becomes implemented; the ESM2 manifest adds `"score"`
  to `capabilities`.
- **New subcommand:**
  ```
  score --input /in/variants.csv --output /out
        [--method masked-marginal|wt-marginal] [--batch-size N] [--device cpu|cuda]
  ```
- **Input `variants.csv`** (columns, header required):
  | column | meaning |
  |---|---|
  | `variant_id` | unique id for the row's variant |
  | `wt_sequence` | full wild-type sequence (repeated across rows that share a WT) |
  | `mutant` | mutation string: `{WT}{pos}{MUT}`, **1-indexed**, multi-mutants colon-separated (`A24G:T56S`) |

  A self-substitution (`A24A`) is valid and must score exactly `0.0`.
- **Output `scores.csv`** (columns): `variant_id, mutant, n_mutations, score`.
  New artifact kind `variant_scores_csv` (`ArtifactKind.VARIANT_SCORES_CSV`).
- `result.json` unchanged in shape (`capability="score"`, one
  `variant_scores_csv` artifact, `warnings`, `params` echoes `method`).
- Bump `CONTRACT_VERSION` to `"0.2"`. Update `docs/CONTRACT.md` (score subcommand,
  input/output schemas, version-compatibility table) and add a worked-example
  `scores.csv`/`result.json` validated by the contract drift-guard test.

## Scoring semantics (ESM2 container)

Group input rows by identical `wt_sequence` to amortize forward passes.

- **masked-marginal** (default): for each *unique mutated position* appearing in
  any variant of a WT, mask that position in the WT, run a forward pass, take
  `log_softmax` over the vocabulary. Per-substitution score =
  `logP(mut_aa) − logP(wt_aa)` at that position. A variant's score is the **sum**
  over its substitutions (standard additive multi-mutant approximation). Cost ≈
  number of unique mutated positions per WT (batched by `--batch-size`).
- **wt-marginal** (`--method wt-marginal`): a single *unmasked* forward pass per
  WT; read logits at the mutated positions and apply the same log-odds
  difference. Cost = 1 forward pass per WT.
- **Per-row validation:** the stated WT residue must equal `wt_sequence[pos-1]`
  and `1 ≤ pos ≤ len(wt_sequence)`. Invalid rows get a blank `score` and an entry
  in `result.warnings`; the batch does **not** fail. Sequences longer than
  `max_sequence_length` are truncated (with a warning), as for the other
  capabilities; variants referencing truncated-away positions become invalid rows.
- `model.eval()`, `torch.no_grad()`, `torch.amp` autocast on CUDA — consistent
  with the existing entrypoint.

## Client changes (mirror embed/likelihood; stays ML-free)

- **`contract.py`** — implement `Capability.SCORE` use, add
  `ArtifactKind.VARIANT_SCORES_CSV`, bump `CONTRACT_VERSION = "0.2"`.
- **`io.py`** — stage a CSV input into `/in` under a fixed name (generalize the
  current FASTA-only staging to also stage a named CSV; validate the required
  header columns → `InvalidRequestError`), and `read_variant_scores(out_dir,
  result)` → `list[dict]` with numeric `score`/`n_mutations`.
- **`models.py`** — `Model.score(variants_csv, *, method="masked-marginal",
  output_dir=None, use_gpu=False, batch_size=None) -> ScoreResult`. Validates the
  capability is supported, `method` is one of the two allowed values
  (`InvalidRequestError` otherwise), and the CSV has the required header. Returns
  a `ScoreResult` dataclass with `.rows()` (lazy CSV parse). Reuses the shared
  `_run_capability` helper; the only capability-specific bits are the staged
  input filename (`variants.csv`) and the extra args (`--method`).
- **`cli.py`** — `plms score MODEL variants.csv -o OUT [--method masked-marginal|wt-marginal] [--gpu/--no-gpu] [--batch-size N]`.
- **`__init__.py`** — export `ScoreResult`.

`_run_capability` currently assumes a FASTA input staged as `seqs.fasta`. It will
be generalized so the staged input filename and the pre-run record/column
validation are capability-specific, while staging, mounting, running, error
handling, and `result.json` parsing stay shared.

## ESM2 container changes

Add `cmd_score` to `containers/esm2/entrypoint.py`: a pure `parse_mutant(s) ->
list[(wt, pos, mut)]` helper, WT-residue validation, WT grouping, and the
masked-marginal / wt-marginal computations (reusing the existing batched-forward
machinery). Add `"score"` to the manifest `capabilities` and bump the manifest
`contract_version` to `"0.2"`. The image is rebuilt; the registry entry is
unchanged.

## Testing

**Unit (no Docker, no torch):**
- `test_contract.py` — `CONTRACT_VERSION == "0.2"`; `variant_scores_csv` kind;
  worked-example `scores.csv`/result validate.
- `test_io.py` — CSV staging + required-column validation (missing column →
  `InvalidRequestError`); `read_variant_scores` numeric coercion.
- `test_models.py` (FakeRunner) — `score` happy path returns `ScoreResult`;
  `CapabilityNotSupportedError` when manifest lacks `score`; `InvalidRequestError`
  for a bad `method` and for a CSV missing columns; correct command construction
  (`score --input /in/variants.csv ... --method ...`).
- `test_cli.py` — `plms score` invokes `Model.score` and renders a summary;
  `PlmsError` → clean exit 1.
- `test_esm2_entrypoint.py` — `parse_mutant("A24G:T56S")` →
  `[("A",24,"G"),("T",56,"S")]`; malformed strings raise; WT-residue mismatch
  detected; self-substitution parses.

**Integration (docker-gated, `@pytest.mark.slow`, `PLMS_RUN_DOCKER_TESTS=1`):**
- Rebuild `plms-esm2:t6_8M` (now contract 0.2).
- `tests/data/variants.csv`: one WT (a short real sequence, e.g. GB1) with a few
  mutants **including a self-substitution** and at least one multi-mutant.
- Assert: all scores finite; the **self-substitution scores ≈ 0.0** (deterministic
  sanity check); `n_mutations` correct; both `masked-marginal` and `wt-marginal`
  run and produce one row per input variant.

## Verification

```bash
docker build --build-arg ESM2_CHECKPOINT=esm2_t6_8M -t plms-esm2:t6_8M containers/esm2
docker run --rm plms-esm2:t6_8M manifest        # capabilities now include "score"
plms score esm2-8m tests/data/variants.csv -o out/
plms score esm2-8m tests/data/variants.csv -o out/ --method wt-marginal --gpu
```
Then: `pytest` (unit) green; `PLMS_RUN_DOCKER_TESTS=1 pytest -m slow` green;
`ruff check`, `ruff format --check`, `ty check src/` clean.

## Out of scope (later sub-projects)

`generate` + ProGen2; ESM-C container; input chunking for large scans;
alternative scoring methods beyond masked/wt-marginal; non-substitution variants
(indels). Multi-mutant scoring uses the additive masked-marginal approximation
only.
