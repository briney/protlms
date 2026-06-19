# Design: ESM-C container (masked LM, native `esm` SDK)

> **Status:** Approved design. Completes Phase 1 of the plms roadmap — adds the
> third first-wave model (ESM-C), a modern general masked LM. Builds on the
> Phase 0 contract + client and the Phase 1 `score`/`generate` work.

## Context

ESM-C (EvolutionaryScale, "ESM Cambrian") is a bidirectional masked protein
language model — the same architectural family as ESM2, just newer and stronger.
It therefore exercises exactly the capabilities the contract already supports and
ESM2 already proves: **`embed`, `likelihood`, `score`** (no `generate`).

The point of adding it is to validate that a *second masked-LM family, with a
completely different SDK and tokenizer*, hides cleanly behind the contract with
**zero client changes** — the work is entirely a new `containers/esm-c/` plus a
registry entry and tests. VISION.md lists ESM2 + ESM-C + ProGen2 as the first
wave and already sketches `containers/esm-c/` in the repo layout.

## Locked decisions

1. **Backend = native EvolutionaryScale `esm` SDK** (`ESMC.from_pretrained`), not
   the community `transformers` ESM++ port. Rationale: original weights, the
   maintained path, and a single forward that yields both logits and embeddings.
   Cost: the entrypoint writes its own load/logits/embedding extraction rather
   than reusing ESM2's `AutoModelForMaskedLM` code.
2. **Capabilities = `embed`, `likelihood`, `score`.** No `generate` (ESM-C is not
   autoregressive).
3. **Checkpoints:** register **both** `esm-c-300m` and `esm-c-600m`; default build
   arg `ESMC_CHECKPOINT=esmc_300m`; the routine end-to-end build/integration test
   runs against **300M only** (600M is wired and buildable but not in the routine
   test path). Mirrors ESM2's 8M/650M setup.
4. **Scope = local open weights only.** The 6B model is Forge-API-only and breaks
   the offline-weights-baked-into-the-image model; it is out of scope.

## Contract changes

**None.** `embed`/`likelihood`/`score` and the neutralized `likelihoods.csv`
schema already exist as of `0.3`. ESM-C declares `contract_version "0.3"` and is
a purely additive model integration. `docs/CONTRACT.md` and
`src/plms/contract.py` are untouched.

## Client changes

**No production-code changes.** `contract.py`, `models.py`, `io.py`, `cli.py`
stay as-is — they are already generic over these three capabilities (all
exercised today by ESM2). The only client-side touches:

- **`src/plms/_data/models.yaml`** — two entries:
  - `esm-c-300m` (alias `esmc_300m`) → `plms-esm-c:300m`, family `esm-c`
  - `esm-c-600m` (alias `esmc_600m`) → `plms-esm-c:600m`, family `esm-c`
- **`tests/test_registry.py`** — assert both names/aliases resolve to their
  images and family.

No `__init__.py` export changes (no new result type; `EmbeddingResult` /
`LikelihoodResult` / variant scoring are already exported).

## ESM-C container (`containers/esm-c/`)

`Dockerfile`, `entrypoint.py`, `README.md`. Versioned with the client, excluded
from the wheel. The entrypoint mirrors the ESM2 entrypoint's structure
(argparse subcommands `manifest`/`embed`/`likelihood`/`score`/`_prefetch`; heavy
imports inside functions so the pure helpers — `sanitize_ids`, `read_fasta`,
`parse_mutant`, `perplexity_from_mean`, `_score_variant`, `_truncate` — stay
torch-free and unit-testable). Each container is standalone, so these helpers are
duplicated from ESM2 by design (same pattern as `containers/progen2/`).

ESM-C-specific pieces:

- **Load:** `from esm.models.esmc import ESMC`;
  `model = ESMC.from_pretrained(ESMC_CHECKPOINT).eval().to(device)` with
  `ESMC_CHECKPOINT ∈ {esmc_300m, esmc_600m}`. The tokenizer comes off the model
  (`model.tokenizer`, an `EsmSequenceTokenizer`) and exposes BOS/EOS and a mask
  token id. `pick_device` validates an explicit `cuda` request exactly like ESM2.
- **Forward (shared path):** a single call returning per-position vocab logits
  `(1, T, V)` **and** hidden embeddings `(1, T, D)` — via the SDK's
  `model.logits(encoded, LogitsConfig(sequence=True, return_embeddings=True))`.
  `embed` reads the embeddings; `likelihood`/`score` read the logits. (Exact SDK
  call surface is pinned/confirmed against the installed `esm` version in the
  plan; a lower-level `model(input_ids)` forward is the documented fallback.)
- **`embed`:** strip BOS/EOS → residues at `1 : 1+len(seq)`. `mean` = mean over
  residues; `cls` = the BOS vector; `none` = the per-residue `(L, D)` array. Same
  pooling scheme and `embeddings.npz` / `per_residue/<id>.npy` outputs as ESM2.
- **`likelihood`:** masked-marginal pseudo-log-likelihood — O(L) masked forwards
  per sequence (batched by `--batch-size`), summing the log-softmax log-prob of
  each true residue under masking. Emits the neutral `likelihoods.csv`
  (`record_id, seq_len, log_likelihood, mean_log_likelihood, perplexity`);
  `result.json` `params.likelihood_method = "masked_marginal"`.
- **`score`:** `masked-marginal` (default) + `wt-marginal`, identical
  `_score_variant` logic to ESM2 (1-indexed `{WT}{pos}{MUT}`, colon-separated
  multi-mutants, self-substitution scores 0, invalid → blank score + warning).
  AA→token id via the ESM-C tokenizer.

### Manifest

`contract_version "0.3"`, `name = ESMC_CHECKPOINT`, `model_family = "esm-c"`,
`capabilities = ["embed","likelihood","score"]`, `pooling_modes =
["mean","cls","none"]`. `embedding_dim` and `num_layers` are **derived from the
loaded model** (not hardcoded) to avoid drift — expected 960/30 for 300M,
1152/36 for 600M. `max_sequence_length` is a documented constant (truncation +
warning beyond it, as in ESM2). `min_gpu_memory_gb`: `null` for 300M (runs on
CPU); a small non-null hint for 600M. `default_batch_size` set sensibly.

### Dockerfile & README

- **Dockerfile:** same `pytorch/pytorch:*-cuda*-runtime` base as ESM2;
  `pip install` a **pinned** `esm` version; `ARG ESMC_CHECKPOINT=esmc_300m` +
  matching `ENV`; bake weights at build via the hidden `_prefetch`
  (`ESMC.from_pretrained(ESMC_CHECKPOINT)` populating the HF cache layer);
  `HF_HUB_OFFLINE=1` at runtime; `ENTRYPOINT` = the contract CLI. **No HF token
  needed** — the 300M/600M repos are open (Cambrian Open License), publicly
  downloadable.
- **README.md:** build commands + the `ESMC_CHECKPOINT` build arg, a checkpoint
  table (300M → 960/30, 600M → 1152/36), baked-weights/offline note, the
  masked-marginal O(L) likelihood note, and standalone `manifest` debugging —
  mirroring `containers/esm2/README.md`.

## Testing

**Unit (no docker/torch):**
- `tests/test_esmc_entrypoint.py` — pure-helper tests: `sanitize_ids`
  (collision de-dup), `read_fasta`, `parse_mutant` (single/multi/invalid),
  `perplexity_from_mean`, `_truncate` warning behavior.
- `tests/test_registry.py` — both `esm-c-300m`/`esm-c-600m` names + aliases
  resolve to their images and family `esm-c`.

**Integration (docker-gated, `@pytest.mark.slow`, `PLMS_RUN_DOCKER_TESTS=1`):**
`tests/test_integration_esmc.py`, mirroring `tests/test_integration_esm2.py`,
against the built **300M** image:
- `manifest` shows `model_family "esm-c"`, `capabilities
  ["embed","likelihood","score"]`, `contract_version "0.3"`, `embedding_dim 960`,
  `num_layers 30`.
- `embed --pooling mean` on a tiny FASTA → `embeddings.npz` with one
  `(960,)` float32 vector per record; `--pooling none` → per-residue `(L, 960)`.
- `likelihood` → finite `log_likelihood`, `perplexity > 1`, one row per record,
  and `result.json` `params.likelihood_method == "masked_marginal"`.
- `score` on a small `variants.csv` → a self-substitution scores 0 and a
  single-mutant gets a finite score.

## Verification

```bash
docker build --build-arg ESMC_CHECKPOINT=esmc_300m -t plms-esm-c:300m containers/esm-c
docker run --rm plms-esm-c:300m manifest
plms embed      esm-c-300m seqs.fasta     -o out/ --pooling mean
plms likelihood esm-c-300m seqs.fasta     -o out/
plms score      esm-c-300m variants.csv   -o out/
```
Then `pytest` (unit) green; `PLMS_RUN_DOCKER_TESTS=1 pytest -m slow` green for
ESM-C; `ruff check src/ tests/`, `ruff format --check src/ tests/`,
`ty check src/` clean.

## Risks

- **`esm` SDK API surface + version pin** is the main implementation unknown: the
  exact load call, the logits/embeddings accessor (`model.logits(...,
  LogitsConfig(...))` vs. a lower-level forward), and the tokenizer's
  special-token layout (BOS/EOS indices, `mask_token_id`) must be confirmed
  against the pinned `esm` version. The plan pins a specific version; the
  container build + the embed-shape and likelihood integration tests are where
  these are proven.
- **torch reconciliation:** the `esm` package may pin a torch version that
  differs from the pytorch base image. The plan reconciles this (compatible base
  image tag, or install strategy) so the build is consistent and reproducible.
- **Weights are open** (no HF token). Low-likelihood contingency: if HF later
  gates the repo behind a click-through, pass `HF_TOKEN` as a build secret;
  document in the README.

## Out of scope (later / not this sub-project)

ESM-C 6B and the Forge API path (no local weights); a `transformers`/ESM++
backend; any new capability or contract change; input chunking; routine
build/test of the 600M image (wired + buildable, but not in the test path).
