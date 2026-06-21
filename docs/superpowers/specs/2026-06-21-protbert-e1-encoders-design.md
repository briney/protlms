# Design: ProtBERT + Profluent-E1 containers (two masked-LM encoders)

> **Status:** Approved design. Expands the model stable with two more bidirectional
> masked-LM encoders, following the established encoder pattern (ESM2 / ESM-C).
> Phase 3 (benchmarking) is deliberately paused; this is additive model
> integration only — **zero client changes**.

## Context

The toolkit already proves that masked-LM encoders hide cleanly behind the
container contract: ESM2 (HuggingFace `AutoModelForMaskedLM`) and ESM-C (native
`esm` SDK) both expose **`embed` + `likelihood` + `score`** with no client code.
This sub-project adds two more encoders in the same mold:

- **ProtBERT** (ProtTrans, Rostlab) — a vanilla `BertForMaskedLM`. The cheapest
  possible addition: it is structurally identical to the ESM2 path, differing
  only in tokenization preprocessing. Follows the **esm2** template.
- **Profluent-E1** (Profluent Bio, 2025) — a newer encoder explicitly pitched as a
  "drop-in replacement for ESM," but with a custom architecture and its own
  package (not loadable via `transformers.AutoModel`). It can run single-sequence
  or retrieval-augmented; **only single-sequence mode is in scope.** Follows the
  **esm-c** template (native SDK, load-free manifest table, flash-attn omitted).

Both are MLM encoders, so neither needs `generate`. Both reuse the contract's
existing `embed`/`likelihood`/`score` semantics at `contract_version "0.3"`.

## Locked decisions

1. **Capabilities = `embed`, `likelihood`, `score`** for both. No `generate`
   (neither is autoregressive).
2. **Scoring = the toolkit's shared masked-marginal / wt-marginal logic on raw
   model logits** for both models — the *same* code path as ESM2/ESM-C. For E1
   this means **not** using the model's bundled `E1Scorer`. Rationale: identical
   `score`/`likelihood` semantics across every model in the stable, so that
   future Phase-3 benchmarking differences reflect the model, not the scoring
   code. (Trade-off: slightly less faithful to E1's published fitness pipeline.)
3. **ProtBERT checkpoints:** ship **both** `Rostlab/prot_bert` (UniRef100) and
   `Rostlab/prot_bert_bfd` (BFD). Same architecture; they isolate training-data
   effects. The routine integration test runs against `prot_bert` only.
4. **E1 checkpoints:** ship **all three** — `E1-150m`, `E1-300m`, `E1-600m`. The
   routine integration test runs against `E1-150m` (smallest; doubles as the
   CI/demo image — there is no sub-100M variant).
5. **E1 mode = single-sequence only.** `context_seqs` is never passed; no
   retrieval/homolog API surface is exposed by the entrypoint.
6. **flash-attn is NOT installed in the E1 image** (matches the esm-c precedent);
   E1's `flex_attention` fallback keeps the image CPU-runnable.

## Contract changes

**None.** `embed`/`likelihood`/`score` and the `likelihoods.csv` schema already
exist as of `0.3`. Both models declare `contract_version "0.3"`. `docs/CONTRACT.md`
and `src/protlms/contract.py` are untouched.

## Client changes

**No production-code changes.** `contract.py`, `models.py`, `io.py`, `cli.py` are
already generic over these three capabilities (all exercised by ESM2 today). The
only client-side touches:

- **`src/protlms/_data/models.yaml`** — five new entries:
  - `protbert` (alias `prot_bert`) → `ghcr.io/briney/protlms-protbert:uniref100`,
    family `protbert`, build arg `PROTBERT_CHECKPOINT=prot_bert`
  - `protbert-bfd` (alias `prot_bert_bfd`) → `…protlms-protbert:bfd`, family
    `protbert`, build arg `PROTBERT_CHECKPOINT=prot_bert_bfd`
  - `e1-150m` (alias `E1-150m`) → `…protlms-e1:150m`, family `e1`, build arg
    `E1_CHECKPOINT=E1-150m`
  - `e1-300m` (alias `E1-300m`) → `…protlms-e1:300m`, family `e1`
  - `e1-600m` (alias `E1-600m`) → `…protlms-e1:600m`, family `e1`
- **`tests/test_registry.py`** — assert each new name/alias resolves to its image
  and family.
- **`.github/workflows/publish-image.yaml`** — add the five new
  `(context, build-arg, image:tag)` rows to the publish matrix.

No `__init__.py` export changes (no new result type; `EmbeddingResult` /
`LikelihoodResult` / variant scoring are already exported).

## ProtBERT container (`containers/protbert/`)

`Dockerfile`, `entrypoint.py`, `README.md`. The entrypoint is the **esm2
entrypoint structure** verbatim (argparse subcommands
`manifest`/`embed`/`likelihood`/`score`/`_prefetch`; heavy imports inside
functions so the pure helpers stay torch-free and unit-testable; per-container
duplication of `sanitize_ids` / `read_fasta` / `parse_mutant` /
`perplexity_from_mean` / `_score_variant` / `_truncate`, as with every container).

ProtBERT-specific pieces:

- **`resolve_hf_id`:** map `prot_bert` → `Rostlab/prot_bert`, `prot_bert_bfd` →
  `Rostlab/prot_bert_bfd`; a value containing `/` is treated as a full HF id.
- **Preprocessing shim (the one real delta):** a `preprocess(seq) -> str` helper
  applied wherever a raw sequence is tokenized:
  ```python
  def preprocess(seq: str) -> str:
      return " ".join(re.sub(r"[UZOB]", "X", seq.upper()))
  ```
  i.e. rare residues `U/Z/O/B → X`, then **space-separate** every residue. The
  tokenizer is loaded with `do_lower_case=False`. Because each residue becomes
  exactly one token between `[CLS]` and `[SEP]`, the existing
  `hidden[1 : 1+len(seq)]` residue slicing, the masked-marginal PLL loop, and
  `convert_tokens_to_ids(aa)` scoring all carry over unchanged. Sequence *length*
  for truncation/PLL is measured on the raw (un-spaced) sequence.
- **Load:** `AutoTokenizer.from_pretrained(hf_id, do_lower_case=False)` +
  `AutoModelForMaskedLM.from_pretrained(hf_id, torch_dtype=torch.float32)`; mask
  token is `[MASK]` (`tokenizer.mask_token_id`). `pick_device` validates explicit
  `cuda` exactly like ESM2.
- **`embed`:** arbitrary `--layers` supported via `output_hidden_states=True`
  (like ESM2). `mean` = mean over residues; `cls` = the `[CLS]` (index 0) vector;
  `none` = per-residue `(L, 1024)` array.
- **`likelihood` / `score`:** identical to ESM2 — masked-marginal PLL
  (`params.likelihood_method = "masked_marginal"`), plus `masked-marginal`
  (default) and `wt-marginal` scoring with the shared `_score_variant` logic.
- **Manifest (from `AutoConfig`, no drift):** `name = PROTBERT_CHECKPOINT`,
  `model_family = "protbert"`, `capabilities = ["embed","likelihood","score"]`,
  `embedding_dim = config.hidden_size` (1024), `num_layers =
  config.num_hidden_layers` (30), `pooling_modes = ["mean","cls","none"]`,
  `max_sequence_length = 1024` (conservative; ProtBERT was trained at 512/2048,
  `max_position_embeddings` is 40000), `min_gpu_memory_gb = null` (runs on CPU),
  `default_batch_size = 8`.
- **Dockerfile:** same `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` base as
  ESM2; `pip install` the **pinned** `transformers` (the esm2 pin, e.g.
  `4.46.3`, validated against vanilla BERT); `ARG PROTBERT_CHECKPOINT=prot_bert`
  + matching `ENV`; bake weights via `_prefetch`; `HF_HUB_OFFLINE=1` at runtime;
  `ENTRYPOINT` = the contract CLI. Weights are ungated (AFL-3.0); no HF token.
- **README.md:** build commands + the `PROTBERT_CHECKPOINT` build arg, a
  checkpoint table (`prot_bert` → UniRef100, `prot_bert_bfd` → BFD; both
  1024/30/~420M), the tokenization note (UZOB→X + spacing), baked-weights/offline
  note, and standalone `manifest` debugging — mirroring `containers/esm2/README.md`.

## Profluent-E1 container (`containers/e1/`)

`Dockerfile`, `entrypoint.py`, `README.md`. The entrypoint mirrors the **esm-c**
shape: native package SDK, load-free manifest table, final-layer-only embeddings.
Same torch-free pure helpers as the other containers.

E1-specific pieces:

- **Load:** `from E1.modeling import E1ForMaskedLM`;
  `model = E1ForMaskedLM.from_pretrained(resolve_hf_id(E1_CHECKPOINT)).eval().to(device)`
  where `resolve_hf_id` maps `E1-150m` → `Profluent-Bio/E1-150m`, etc. Inputs are
  built with `E1BatchPreparer().get_batch_kwargs([seq], device=device)`, which
  yields `input_ids`, `within_seq_position_ids`, `global_position_ids`,
  `sequence_ids`. The forward returns `outputs.logits` `(B, L, 34)` and
  `outputs.embeddings` `(B, L, D)`. Mask token is `?` (id 5).
- **Single-sequence only:** the entrypoint passes exactly one sequence per record
  and never supplies homolog context, so `sequence_ids` is trivial and the whole
  token span (minus boundary tokens) is the query. Boundary tokens (BOS/EOS) are
  identified via the batch preparer's boundary-token mask
  (`get_boundary_token_mask`) rather than hardcoded indices.
- **`embed` (final layer only, like esm-c):** non-`-1` `--layers` → an
  `InvalidInput` error. `mean` = mean over residue (non-boundary) positions;
  `cls` = the BOS-position vector; `none` = per-residue `(L, D)` array.
- **`likelihood`:** masked-marginal PLL — for each residue position, set its
  `input_ids` entry to the `?` mask id, batch by `--batch-size`, `log_softmax`
  the logits, sum the true-residue log-prob. `params.likelihood_method =
  "masked_marginal"`. Emits the neutral `likelihoods.csv`.
- **`score`:** `masked-marginal` (default) + `wt-marginal`, shared
  `_score_variant` logic. AA→token id via E1's tokenizer.
- **Manifest (load-free `_MODEL_INFO` table, like esm-c):**
  `model_family = "e1"`, `capabilities = ["embed","likelihood","score"]`,
  `pooling_modes = ["mean","cls","none"]`, `max_sequence_length = 2048` (model
  supports 8192 within-seq, but masked-marginal is O(L) forward passes), per
  checkpoint:

  | checkpoint | embedding_dim | num_layers | min_gpu_memory_gb |
  |---|---|---|---|
  | `E1-150m` | 768  | 20 | null |
  | `E1-300m` | 1024 | 20 | 2.0 (hint) |
  | `E1-600m` | 1280 | 30 | 4.0 (hint) |

  `default_batch_size = 8`. (GPU-recommended but CPU-capable via `flex_attention`.)
- **Dockerfile:** a **newer base** than ESM2 — a `pytorch/pytorch:2.7.x-cuda12.x`
  image (E1 requires torch ≥2.7,<2.9 and Python ≥3.12). Install the custom `E1`
  package **pinned to a specific commit/tag**:
  `pip install "E1 @ git+https://github.com/Profluent-AI/E1.git@<pin>"` (pulls its
  `transformers <4.57`, `einops`, `tokenizers`, `kernels`, etc.). **No flash-attn.**
  `ARG E1_CHECKPOINT=E1-150m` + `ENV`; bake weights via `_prefetch`
  (`E1ForMaskedLM.from_pretrained(...)` populating the HF cache); `HF_HUB_OFFLINE=1`
  at runtime; `ENTRYPOINT` = the contract CLI. Repos are ungated; no HF token.
- **README.md:** build commands + `E1_CHECKPOINT` build arg, the size table above,
  a single-sequence-only note, the baked-weights/offline note, the masked-marginal
  O(L) likelihood note, and the **E1 weights attribution requirement** — reproduce
  the upstream `NOTICE`/`ATTRIBUTION` text (weights are free for research +
  commercial use *with attribution*; code is Apache-2.0).

## Testing

**Unit (no docker/torch):**
- `tests/test_protbert_entrypoint.py` — pure-helper tests, plus explicit
  **`preprocess` cases**: `UZOB→X` substitution, space-joining, length measured on
  the raw sequence; `resolve_hf_id` mapping; `sanitize_ids`, `read_fasta`,
  `parse_mutant`, `perplexity_from_mean`, `_truncate`.
- `tests/test_e1_entrypoint.py` — pure-helper tests + `resolve_hf_id`
  (`E1-150m` → `Profluent-Bio/E1-150m`) + the final-layer-only guard.
- `tests/test_registry.py` — the five new names/aliases resolve to their images
  and families (`protbert`, `e1`).

**Integration (docker-gated, `@pytest.mark.slow`, `PROTLMS_RUN_DOCKER_TESTS=1`):**
- `tests/test_integration_protbert.py`, mirroring `test_integration_esm2.py`,
  against the built `protbert:uniref100` image: `manifest` shows family
  `protbert` / `embedding_dim 1024` / `num_layers 30`; `embed` mean → `(1024,)`
  per record and `none` → `(L, 1024)`; `likelihood` → finite values,
  `params.likelihood_method == "masked_marginal"`; `score` → self-sub scores 0,
  single-mutant finite.
- `tests/test_integration_e1.py`, same shape, against the built `e1:150m` image
  (`embedding_dim 768`, `num_layers 20`, family `e1`).

## Verification

```bash
# ProtBERT
docker build --build-arg PROTBERT_CHECKPOINT=prot_bert -t protlms-protbert:uniref100 containers/protbert
docker run --rm protlms-protbert:uniref100 manifest
protlms embed protbert seqs.fasta -o out/ --pooling mean
protlms score protbert variants.csv -o out/

# E1
docker build --build-arg E1_CHECKPOINT=E1-150m -t protlms-e1:150m containers/e1
docker run --rm protlms-e1:150m manifest
protlms likelihood e1-150m seqs.fasta -o out/
```
Then `pytest` (unit) green; `PROTLMS_RUN_DOCKER_TESTS=1 pytest -m slow` green for
ProtBERT + E1; `ruff check src/ tests/`, `ruff format --check src/ tests/`,
`ty check src/` clean.

## Risks

- **E1 is the main implementation unknown.** Three things must be confirmed
  against the pinned commit during the build: (1) the exact `E1ForMaskedLM`
  forward signature and that `outputs.logits` / `outputs.embeddings` are returned
  without extra flags; (2) the `E1BatchPreparer` boundary-token mask API used to
  strip BOS/EOS; (3) that the **CPU path works without flash-attn** via
  `flex_attention` (and is fast enough for a tiny-FASTA integration test). The E1
  embed-shape and likelihood integration tests are where these are proven.
- **Base-image / dependency reconciliation for E1:** E1 needs torch ≥2.7 and
  Python ≥3.12, so it gets its own newer base image distinct from the
  esm2/protbert 2.5.1 base. Pin a concrete `pytorch` tag whose torch satisfies
  E1's range; pin the E1 git commit so `transformers <4.57` resolves
  reproducibly. This is isolated to the E1 image, so it cannot affect the others.
- **ProtBERT tokenization correctness:** the space-join must yield one token per
  residue; covered directly by the `preprocess` unit tests and the embed-shape
  integration assertion. Note: `UZOB→X` rewrites those rare residues, so a
  `wt_sequence` containing them will mismatch on `score` (the WT-residue check) —
  acceptable and rare; documented in the README.
- **Image size:** there is no tiny ProtBERT/E1 demo model (smallest are ~420M /
  150M), so integration images are larger than esm2's 8M. Mitigated by the tests
  being `slow`-gated and opt-in; routine tests build only the smallest of each.

## Out of scope (later / not this sub-project)

- E1 **retrieval-augmented mode** (homolog context) — explicitly deferred.
- E1's bundled `E1Scorer` fitness pipeline (we use shared scoring; see decision 2).
- Other ProtTrans models (ProtT5, ProtAlbert, ProtElectra, ProtXLNet) and the
  community `DistilProtBert`.
- Any new capability or contract change; input chunking; Phase-3 benchmarking.
