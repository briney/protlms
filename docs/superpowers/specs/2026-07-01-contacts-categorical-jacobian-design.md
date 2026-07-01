# Design: `contacts` (categorical-Jacobian contact prediction) + CASP14 evaluation

> **Status:** Approved design. Adds a new masked-LM capability (`contacts`) to the
> container contract, generalizes the ESM2 image into a shared ESM masked-LM image
> (covering ESM-1b + all ESM-2 sizes), migrates the ESM-C family off the ESM SDK
> onto HuggingFace `transformers`, and adds an in-package evaluation harness that
> scores emergent structural knowledge as **long-range precision@L** on CASP14.

## Context & goal

We want to measure the emergent structural knowledge of several pLMs — ESM-1b,
ESM-2 (all sizes), and ESM-C (all sizes) to start — using **long-range
precision@L** computed from the **categorical Jacobian** (Zhang, Wayment-Steele,
… Ovchinnikov, *PNAS* 2024, ["Protein language models learn evolutionary
statistics of interacting sequence motifs"](https://www.pnas.org/doi/10.1073/pnas.2406285121)).
The evaluation dataset is **CASP14**, whose PDB files live at
`~/projects/esm-c/data/casp14/` (34 `.pdb` targets).

The categorical Jacobian is model-specific to compute (tokenizer, logit access,
mutation mechanics) but produces a tiny, model-agnostic artifact: an `(L, L)`
contact-score matrix. That boundary maps cleanly onto the existing contract
philosophy — the **container** computes the analysis (as `score` bakes in
masked-marginal), the **client** stays ML-free, and a new model "just works" once
it publishes a contract-compliant image and a registry entry.

## Reference method (pinned to the authors' code)

Ported verbatim from
[`zzhangzzhang/pLMs-interpretability`](https://github.com/zzhangzzhang/pLMs-interpretability)
and [Ovchinnikov's ColabBio](https://github.com/sokrypton/ColabBio/tree/main/categorical_jacobian).

**Categorical Jacobian** (`get_categorical_jacobian`): feed the **unmasked**
wild-type sequence. For each position `n`, tile the tokenized sequence 20×, set
position `n` to each of the 20 standard amino acids, forward-pass, and read logits
at *every* position over the 20 AA tokens → `(20, L, 20)`. Stack over positions
and subtract the WT baseline → tensor `J` of shape `(L, 20, L, 20)`. Cost ≈ `L`
forward passes of batch-20 (≈ `20·L` sequences of length `L`).

**Post-processing** (their exact order, ported as a shared helper):
1. Center over all four axes: `for i in range(4): J -= J.mean(i, keepdims=True)`.
2. Symmetrize the 4-D tensor: `J = (J + J.transpose(2,3,0,1)) / 2`.
3. Frobenius norm over the two AA axes: `S = sqrt((J**2).sum((1,3)))` → `(L, L)`.
4. Zero the diagonal.
5. **APC** (average product correction): `S -= (S.sum(0)*S.sum(1)) / S.sum()`, re-zero diagonal.
6. Symmetrize the `(L, L)`: `S = (S + S.T) / 2`.

**Generalization vs. the reference:** the reference hardcodes ESM's `4:24` token
block and `arange(4,24)` mutation ids. Our container maps the 20 AA letters →
token ids **via the tokenizer** and identifies residue positions via the
tokenizer's special-token mask, so the identical code serves ESM-1b, ESM-2, and
ESM-C.

**Metric** (`top_L`, from the same repo):
- **True contact:** Cβ–Cβ distance `< 8.0 Å` (Cα for glycine).
- **Long-range:** sequence separation `|i − j| ≥ 24`.
- **precision@L:** rank eligible pairs by predicted score (descending), take the
  top `L` where `L = len(sequence)`, precision = fraction that are true contacts.
  `sep` and the top-`k` fraction (L, L/2, L/5) are configurable; defaults match
  the paper (`sep=24`, top-`L`).
- **Resolved-residues only:** CASP PDBs have gaps; scoring is restricted to pairs
  of residues that are resolved in the structure.

## Scope & non-goals

**In scope:** contract `contacts` capability (0.4 bump); shared ESM image; ESM-C
family migration to `transformers`; `contacts` on both images; client
`Model.contacts` + IO parsing; `protlms.eval` (PDB→contacts, precision@L, CASP14
runner); `protlms contacts` and `protlms eval contacts` CLI; registry entries for
ESM-1b, all ESM-2 sizes, all ESM-C sizes; tests.

**Non-goals:** attention-map contact heads (`predict_contacts`); other structural
metrics (TM-score, distograms); alternative Jacobian variants (masked, autograd);
non-substitution analysis; MSA/coevolution baselines; a GPU cluster batch runner.

## Contract changes (0.3 → 0.4, minor bump)

Same major version → a 0.3 client and a 0.4 image stay mutually readable (unknown
fields ignored); a 0.4 client may warn-but-run against a 0.3 image. Edit
`contract.py` and `docs/CONTRACT.md` together (the drift-guard test enforces it).

- **`Capability.CONTACTS = "contacts"`**; images that support it add `"contacts"`
  to their manifest `capabilities`.
- **New subcommand:**
  ```
  contacts --input /in/seqs.fasta --output /out
           [--method categorical-jacobian] [--batch-size N] [--device cpu|cuda]
  ```
  `--method` is a named enum (only `categorical-jacobian` for now), mirroring how
  `score` names its method for forward-compatibility. **Multi-record FASTA** is
  supported (looped internally) — so a full CASP14 sweep is **one container run
  per model** (weights load once).
- **Output:** one `(L, L)` float32 array per record at `contacts/<id>.npy`, plus
  `result.json`. New artifact kind `contact_map` (`ArtifactKind.CONTACT_MAP`),
  one artifact per record (mirrors per-residue embeddings), each carrying
  `record_ids=[id]`, `shape=[L, L]`, `dtype="float32"`.
- `result.json`: `capability="contacts"`, `params` echoes `{method, device}`;
  `warnings` used for truncation.
- Bump `CONTRACT_VERSION = "0.4"`; update the version-compatibility table and add
  a worked-example `result.contacts.example.json` under `tests/data/`.

## Container changes

### Generalize `containers/esm2/` → `containers/esm/`

A shared `EsmForMaskedLM` image serving **ESM-1b + all ESM-2 sizes** (same
architecture, same `transformers` load path). Build args:
- `HF_ID` — full HuggingFace id (e.g. `facebook/esm2_t33_650M_UR50D`,
  `facebook/esm1b_t33_650M_UR50S`).
- `MODEL_NAME` — clean manifest `name` (e.g. `esm2_t33_650M`, `esm1b_t33_650M`).
- `MODEL_FAMILY` — manifest `model_family` (`esm2` or `esm1b`).

This rename touches: the Dockerfile, `entrypoint.py` (env vars `ESM2_CHECKPOINT`
→ the new build args; drop the esm2-only `resolve_hf_id` suffix logic in favor of
a passed-through `HF_ID`), `tests/test_esm2_entrypoint.py` →
`tests/test_esm_entrypoint.py`, the esm2 integration test, the GHCR image name
(`protlms-esm2` → `protlms-esm`), and the registry entries/`build.context`. The
publish workflow's build matrix updates accordingly.

### `contacts` implementation (shared ESM entrypoint)

Add `cmd_contacts` plus pure, unit-testable helpers:
- `aa_token_ids(tokenizer) -> list[int]` — the 20 AA letters → token ids.
- `categorical_jacobian(model, tokenizer, seq, batch_size, device) -> (L,20,L,20)`
  — the mutate-and-read-logits loop, tokenizer-driven, `--batch-size` controls
  mutated-sequences-per-forward-pass for memory.
- `jacobian_to_contacts(J) -> (L,L)` — the exact 6-step post-processing above
  (pure numpy; the primary unit-test target).

`model.eval()`, `torch.no_grad()`, `torch.amp` autocast on CUDA, consistent with
the existing entrypoint. Sequences over `max_sequence_length` are truncated with a
warning. Manifest gains `"contacts"`; `contract_version` → `"0.4"`.

### `containers/esm-c/`: migrate ESM SDK → `transformers`, add `contacts`

Replace the `esm==3.2.3` SDK path with `AutoTokenizer` + `AutoModelForMaskedLM`
(`trust_remote_code=True`) for **all** ESM-C checkpoints (300M, 600M, 6B), so the
family is unified on `transformers` and `contacts`/`likelihood`/`score`/`embed`
all read logits/hidden states the same way. `embed` reads `output.hidden_states`;
`likelihood`/`score`/`contacts` read `output.logits`. The Jacobian helpers are
shared logic (same math; different tokenizer/model), factored so both entrypoints
call the same numpy post-processing.

- **Verify at build time:** exact HF repo ids and the `AutoModelForMaskedLM`
  logits path per checkpoint (candidates: `biohub/esmc-300m-2024-12`,
  `biohub/ESMC-600M`, `biohub/ESMC-6B`; fallback: the Synthyra `ESM++` ports,
  which expose `AutoModelForMaskedLM` + `logits` with `trust_remote_code=True`).
  Pin the resolved ids in the Dockerfile/registry build args.
- Base image / Python: keep whatever the chosen HF path requires; drop the
  SDK-specific Python-3.12 pin if no longer needed. Bake weights via `_prefetch`,
  `HF_HUB_OFFLINE=1` at runtime (unchanged pattern).
- Existing ESM-C unit/integration tests update to the transformers path; the
  `esmc_300m`/`esmc_600m` numeric outputs may shift slightly (SDK vs. HF weights
  are the same, but tokenization/precision paths differ) — integration assertions
  stay tolerance-based / sanity-based, not exact-value.

## Client changes (`protlms`, stays ML-free; numpy already core)

- **`contract.py`** — `Capability.CONTACTS`, `ArtifactKind.CONTACT_MAP`,
  `CONTRACT_VERSION = "0.4"`.
- **`io.py`** — `load_contact_maps(out_dir, result) -> dict[str, np.ndarray]`
  (id → `(L, L)`), reading the `contact_map` artifacts (mirrors
  `load_per_residue_embeddings`).
- **`models.py`** — `Model.contacts(fasta, *, method="categorical-jacobian",
  output_dir=None, use_gpu=False, batch_size=None, chunk_size=None) ->
  ContactsResult`. Validates the capability + method; reuses the shared
  `_run` / `_run_chunked` machinery (staged input `seqs.fasta`, extra args
  `--method [--batch-size]`). `ContactsResult` dataclass with `.maps()` (lazy).
- **`__init__.py`** — export `ContactsResult`.

## Evaluation harness (`protlms.eval`, bundles biopython)

BioPython is added to the **core** dependencies (user decision — no separate
`[eval]` extra). It is not an ML dependency; the "no ML deps" principle is
preserved.

- **`protlms/eval/contacts.py`** (pure, numpy + biopython):
  - `parse_pdb(pdb: Path) -> PdbChain` → `sequence: str`, `resnums: np.ndarray`,
    `cb_coords: (N,3) float` (Cβ, Cα for Gly), for resolved residues in order.
  - `true_contact_map(cb_coords, *, threshold=8.0) -> (N,N) bool`.
  - `long_range_precision_at_l(pred, true, resnums, *, sep=24, top=None) -> float`
    — restrict to `|resnum_i − resnum_j| ≥ sep` (residue numbers respect gaps),
    rank by `pred`, take top `top` (default `N`), return true-positive fraction.
    Gap policy: model input = resolved-residue sequence; separation uses PDB
    residue numbers; only resolved pairs scored — matching the reference's
    "restrict to PDB residues".
- **`protlms/eval/runner.py`**:
  `evaluate_contacts(model_name, pdb_dir, *, sep, top, use_gpu, batch_size) ->
  list[TargetResult]` — parse every PDB, build **one** FASTA of all target
  sequences, call `Model.contacts(...)` once, then score each target and return
  per-target `precision_at_l` + the mean. Writes a results CSV (numpy + `csv`
  only; **no pandas**).

## CLI changes (`cli.py`)

- **`protlms contacts MODEL FASTA -o OUT [--method categorical-jacobian] [--gpu/--no-gpu] [--batch-size N]`**
  — predict `(L, L)` maps only.
- **`protlms eval contacts MODEL... --pdb-dir DIR [--out results.csv] [--sep 24] [--top L] [--gpu/--no-gpu] [--batch-size N]`**
  — the CASP14 benchmark, `rich` progress bar, prints a per-model summary
  (per-target P@L + mean).

## Registry & models (`_data/models.yaml`)

- **ESM (shared image, `build.context: containers/esm`):** `esm1b`
  (`facebook/esm1b_t33_650M_UR50S`); ESM-2 `esm2_t6_8M`, `esm2_t12_35M`,
  `esm2_t30_150M`, `esm2_t33_650M`, `esm2_t36_3B`, `esm2_t48_15B`. GHCR image
  `ghcr.io/briney/protlms-esm:<tag>`.
- **ESM-C (transformers):** `esmc_300m`, `esmc_600m`, `esmc_6b`.
- All ESM/ESM-C manifests declare `contacts`.
- The **giants** (esm2-3B/15B, esm-c-6B) are registered now, but the benchmark
  defaults to the ≤650M-class; large models are opt-in (see caveats).

## Data flow (CASP14 benchmark, end to end)

`protlms eval contacts esm2-650m --pdb-dir ~/projects/esm-c/data/casp14/` →
`eval.runner` parses each PDB (`sequence`, `resnums`, `cb_coords`) → writes one
FASTA of all targets → `protlms.load("esm2-650m").contacts(fasta)` → one container
run computes the categorical Jacobian + post-processing per target, writing
`contacts/<id>.npy` → client loads the `(L,L)` maps → `eval.contacts` builds true
contact maps and computes long-range precision@L per target → CSV + mean.

## Testing

**Unit (no Docker, no torch):**
- `test_contract.py` — `CONTRACT_VERSION == "0.4"`; `contact_map` kind; worked
  `result.contacts.example.json` validates.
- `test_io.py` — `load_contact_maps` returns id→`(L,L)` arrays.
- `test_models.py` (FakeRunner) — `contacts` happy path → `ContactsResult`;
  `CapabilityNotSupportedError` when unsupported; correct argv
  (`contacts --input /in/seqs.fasta ... --method ...`).
- `test_cli.py` — `protlms contacts` and `protlms eval contacts` invoke the model
  and render summaries; `ProtlmsError` → clean exit 1.
- `test_esm_entrypoint.py` — `jacobian_to_contacts` on a small synthetic
  `(L,20,L,20)` tensor: correct `(L,L)` shape, zero diagonal, symmetry, and a
  hand-checkable APC result; `aa_token_ids` returns 20 ids.
- `test_eval_contacts.py` — `parse_pdb` + `true_contact_map` on one real CASP14
  PDB (resolved-residue count, symmetric boolean map, `≥` some contacts);
  `long_range_precision_at_l` on a hand-constructed `pred`/`true`/`resnums` case
  (known answer, incl. the `sep` filter and gap-aware separation).

**Integration (docker-gated, `@pytest.mark.slow`, `PROTLMS_RUN_DOCKER_TESTS=1`):**
- Build `protlms-esm:t6_8M` (contract 0.4); `manifest` includes `contacts`.
- `contacts` on one short CASP14 target → `(L,L)` map; end-to-end
  `evaluate_contacts` on that target yields a finite P@L in `[0,1]` above a small
  random-baseline threshold (structural signal sanity check).
- ESM-C transformers image: `embed`/`likelihood`/`score`/`contacts` smoke test on
  a short sequence (shapes + finite values).

## Verification

```bash
docker build --build-arg HF_ID=facebook/esm2_t6_8M_UR50D \
  --build-arg MODEL_NAME=esm2_t6_8M --build-arg MODEL_FAMILY=esm2 \
  -t protlms-esm:t6_8M containers/esm
docker run --rm protlms-esm:t6_8M manifest        # capabilities include "contacts"
protlms contacts esm2-8m tests/data/short.fasta -o out/
protlms eval contacts esm2-8m --pdb-dir ~/projects/esm-c/data/casp14/ --out patl.csv
```
Then: `pytest` (unit) green; `PROTLMS_RUN_DOCKER_TESTS=1 pytest -m slow` green;
`ruff check src/ tests/`, `ruff format --check src/ tests/`, `ty check src/` clean.

## Risks & caveats

- **Compute cost ≈ O(20·L³).** Cheap on 8M–650M; **impractical on esm2-3B/15B and
  esm-c-6B** for long targets. Registered but opt-in; benchmark defaults to
  ≤650M-class.
- **Very long CASP14 targets** (e.g. T1044 ≈ 2180 aa) exceed `max_sequence_length`
  and are Jacobian-infeasible; they are truncated-with-warning and effectively
  out of the practical sweep. The runner should skip/flag targets beyond a
  configurable length cap rather than attempt them.
- **ESM-C HF logit path** is the load-bearing unknown for the family migration:
  confirm repo ids + `AutoModelForMaskedLM` logits before wiring all three
  checkpoints; Synthyra `ESM++` is the fallback. Migrating 300M/600M off the SDK
  means re-validating their existing capabilities against the new path.
- **ESM `token_dropout`:** HF `EsmForMaskedLM` applies token-dropout scaling; we
  replicate the reference's (unmasked, default-settings) behavior and note it as a
  knob if contact quality regresses.

## Resolved decisions

- Contract bump to **0.4** (approved).
- Giants (esm2-3B/15B, esm-c-6B) in the registry **now** (approved).
- **All** ESM-C checkpoints migrate to `transformers` — the whole family unified
  (approved).
- Eval lives **in-package** (`protlms.eval` + `protlms eval contacts`), biopython
  bundled into the **core** install, no separate extra (approved).
- ESM-1b served by a **generalized shared `esm` container** (approved).
- Defaults chosen (flag to revisit): multi-record FASTA for `contacts`; `--method`
  as a named enum; resolved-sequence + residue-number separation for gaps; numpy +
  csv (no pandas) in the eval harness.
