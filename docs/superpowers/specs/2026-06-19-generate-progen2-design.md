# Design: `generate` + ProGen2 (autoregressive generation)

> **Status:** Approved design. Phase 1b of the plms roadmap — the fourth
> capability (`generate`) plus a second model family (ProGen2, autoregressive),
> proving the contract spans a fundamentally different architecture. Builds on
> the Phase 0 contract + client and the Phase 1a `score` work.

## Context

The contract so far covers `embed`, `likelihood`, and `score` — all exercised by
ESM2, a masked LM. `generate` is reserved but unimplemented, and every model to
date is a masked LM. This sub-project implements **`generate`** and adds
**ProGen2**, a decoder-only autoregressive protein LM, which:

- exercises the only remaining capability (`generate`), and
- proves the contract generalizes across architectures (masked vs. autoregressive).

ProGen2 also supports a true (left-to-right, causal) sequence log-likelihood, so
it declares **`generate` + `likelihood`**. This forces a small, beneficial
contract refinement: the shared `likelihoods.csv` schema becomes
**method-neutral** so masked-marginal (ESM2) and causal (ProGen2) likelihoods are
directly comparable.

`embed` for ProGen2, ESM-C, and input chunking remain out of scope (later
sub-projects).

## Locked decisions

1. **ProGen2 scope = `generate` + `likelihood`** (not `embed`).
2. **Prompt model:** input is a FASTA of prefixes; each record's sequence is a
   prefix to continue; an **empty sequence = unconditional** sampling.
3. **Sampling knobs (v1):** `num_samples`, `temperature`, `top_p`, `max_length`,
   `seed`.
4. **Model source:** a HuggingFace community port loaded via
   `AutoModelForCausalLM(..., trust_remote_code=True)`; demo/CI checkpoint
   `progen2-small` (151M).
5. **Likelihood schema:** neutralize column names now (cross-cutting; touches
   ESM2).

## Contract changes (0.2 → 0.3, minor bump)

Same major, so a 0.2 client/image and a 0.3 client/image remain mutually
readable (unknown fields ignored).

- **`Capability.GENERATE`** becomes implemented; ProGen2's manifest declares
  `capabilities: ["generate", "likelihood"]`.
- **New artifact kind** `ArtifactKind.GENERATED_FASTA = "generated_fasta"`.
- **`generate` subcommand:**
  ```
  generate --input /in/prompts.fasta --output /out
           [--num-samples N] [--temperature T] [--top-p P] [--max-length L]
           [--seed S] [--batch-size N] [--device cpu|cuda]
  ```
- **Neutralized likelihood schema** (was `pseudo_log_likelihood,
  mean_pseudo_log_likelihood, pseudo_perplexity`):
  `record_id, seq_len, log_likelihood, mean_log_likelihood, perplexity`.
  The masked-vs-causal distinction moves to `result.json`
  `params.likelihood_method` ∈ {`masked_marginal`, `causal`}.
- **Manifest:** no schema change. A generate-only-ish model reports its real
  `embedding_dim`/`num_layers` and `pooling_modes: []` (it does not support
  `embed`, so the client never requests pooling). `max_sequence_length` doubles
  as the generation length ceiling.
- Bump `CONTRACT_VERSION = "0.3"`. Update `docs/CONTRACT.md` (generate
  subcommand, generated-fasta output, neutralized likelihood schema + method
  param, version table) and add a worked-example `result.json` for `generate`
  validated by the drift-guard test.

## `generate` semantics (ProGen2 container)

- Read `prompts.fasta`. Each record → a prompt; **empty sequence = unconditional**
  (sample from the model's start token). At least one record required.
- For each prompt, sample `num_samples` continuations via HF
  `model.generate(do_sample=True, temperature=T, top_p=P, max_length=L,
  num_return_sequences=num_samples)`. `--max-length` defaults to
  `max_sequence_length` when omitted.
- `--seed` → `torch.manual_seed(seed)` before generation; the seed (and all
  sampling params) are echoed into `result.json` params for reproducibility.
- **Output `generated.fasta`:** clean amino-acid sequences with ProGen2's
  control/special tokens stripped; headers `{prompt_id}__sample{k}` for
  `k = 0..num_samples-1`. One `generated_fasta` artifact.
- `model.eval()`, `torch.no_grad()`, `torch.amp` autocast on CUDA — consistent
  with the ESM2 entrypoint.

## `likelihood` semantics (ProGen2, causal)

- True left-to-right log-likelihood: one forward pass over each sequence; sum the
  log-softmax log-probabilities of the actual next tokens across positions.
  `mean_log_likelihood = log_likelihood / seq_len`;
  `perplexity = exp(-mean_log_likelihood)`.
- Emits the neutralized `likelihoods.csv`; `result.json`
  `params.likelihood_method = "causal"`.

## ESM2 cross-cutting change

Rename ESM2's likelihood CSV columns to the neutral names and set
`params.likelihood_method = "masked_marginal"`. Rebuild the ESM2 image. Update
the ESM2 likelihood integration-test assertions (`perplexity` not
`pseudo_perplexity`). ESM2's manifest also bumps to `contract_version "0.3"`.
This is justified: the likelihood schema is shared infrastructure that ProGen2
now also uses; neutralizing it while there is a single existing consumer is the
cheap moment.

## Client changes (mirror existing capabilities; stays ML-free)

- **`contract.py`** — implement `GENERATE`, add `GENERATED_FASTA`, bump
  `CONTRACT_VERSION = "0.3"`.
- **`io.py`** — `read_generated(out_dir, result) -> list[FastaRecord]` (locates
  the `generated_fasta` artifact and parses it with the existing `read_fasta`);
  update `_LIKELIHOOD_COLUMN_TYPES` keys to `log_likelihood`,
  `mean_log_likelihood`, `perplexity`.
- **`models.py`** — `GenerationResult` dataclass with
  `sequences() -> list[FastaRecord]` (lazy); `Model.generate(prompts_fasta, *,
  num_samples=1, temperature=1.0, top_p=1.0, max_length=None, seed=None,
  output_dir=None, use_gpu=False, batch_size=None) -> GenerationResult`.
  Validates `Capability.GENERATE`; reads records (≥1 required; empty sequences
  allowed) via the existing `_read_records`; stages via `stage_inputs`; builds
  the `generate` command (only emitting flags the caller set). Reuses `_run`.
- **`cli.py`** — `plms generate MODEL prompts.fasta -o OUT [--num-samples N]
  [--temperature T] [--top-p P] [--max-length L] [--seed S] [--gpu/--no-gpu]
  [--batch-size N]`.
- **`registry.py`** — add `progen2-small` (alias `progen2_small`) →
  `plms-progen2:small`, family `progen2`.
- **`__init__.py`** — export `GenerationResult`.

`_read_records` currently rejects empty input and warns on overlong sequences;
it already permits records with empty sequences, so unconditional prompts pass
unchanged.

## ProGen2 container (`containers/progen2/`)

`Dockerfile`, `entrypoint.py`, `README.md`. Excluded from the wheel.

- **Dockerfile:** same pytorch CUDA base as ESM2; `pip install transformers`;
  `ARG PROGEN2_CHECKPOINT=progen2-small` mapped to the HF port id; bake weights
  at build via a hidden `_prefetch` (`from_pretrained(..., trust_remote_code=True)`);
  `HF_HUB_OFFLINE=1` at runtime; entrypoint = the contract CLI.
- **`entrypoint.py`** (argparse, heavy imports inside functions so pure helpers
  stay torch-free): `manifest` (capabilities `["generate","likelihood"]`,
  `embedding_dim`/`num_layers` from config, `pooling_modes: []`,
  `max_sequence_length`, `contract_version "0.3"`); `generate` (sampling +
  control-token stripping + `generated.fasta`); `likelihood` (causal LL +
  neutralized CSV + `likelihood_method="causal"`); `_prefetch`. Structured
  `ContainerError` on stderr + non-zero exit on failure, matching the contract.
- **`README.md`:** build commands, the `PROGEN2_CHECKPOINT` build arg, baked
  weights, `trust_remote_code` note, standalone `manifest` debugging.

## Testing

**Unit (no docker/torch):**
- `test_contract.py` — `CONTRACT_VERSION == "0.3"`; `generated_fasta` kind;
  worked-example generate `result.json` validates.
- `test_io.py` — `read_generated` parses a `generated.fasta` keyed/ordered as
  written; `read_likelihoods` uses the neutral column keys.
- `test_models.py` (FakeRunner) — `generate` returns a `GenerationResult` whose
  `sequences()` reflects the staged prompts × `num_samples`; command-building
  emits only the flags set; `CapabilityNotSupportedError` when a model lacks
  `generate`; `InvalidRequestError` on an empty prompts file.
- `test_cli.py` — `plms generate` invokes `Model.generate` with the right kwargs;
  `PlmsError` → clean exit 1.
- ProGen2 entrypoint pure-helper tests (e.g. a control-token-stripping / output
  header helper).
- Update the existing ESM2 likelihood unit/fixture expectations to the neutral
  column names.

**Integration (docker-gated, `@pytest.mark.slow`, `PLMS_RUN_DOCKER_TESTS=1`):**
- Build `plms-progen2:small`; `docker run ... manifest` shows
  `capabilities ["generate","likelihood"]`, `contract_version "0.3"`.
- **Determinism anchor:** `generate` with a fixed `--seed` run twice produces
  identical `generated.fasta` (the deterministic correctness check for this
  sub-project).
- A `prompts.fasta` with one prefix record and one empty (unconditional) record,
  `num_samples=2` → 4 output records, each a valid amino-acid string no longer
  than `max_length`, headers `{prompt_id}__sample{0,1}`.
- ProGen2 `likelihood` on a tiny FASTA → finite `log_likelihood`, `perplexity > 1`,
  one row per record.
- Rebuild the ESM2 image; its likelihood integration test now asserts the neutral
  `perplexity` column and `params.likelihood_method == "masked_marginal"`.

## Verification

```bash
docker build --build-arg PROGEN2_CHECKPOINT=progen2-small -t plms-progen2:small containers/progen2
docker run --rm plms-progen2:small manifest
plms generate progen2-small prompts.fasta -o out/ --num-samples 4 --temperature 0.8 --top-p 0.9 --seed 42
plms likelihood progen2-small seqs.fasta -o out/
```
Then `pytest` (unit) green; `PLMS_RUN_DOCKER_TESTS=1 pytest -m slow` green (both
ProGen2 and the rebuilt ESM2); `ruff check`, `ruff format --check`,
`ty check src/` clean.

## Risks

- **ProGen2 HF port compatibility** is the main implementation unknown: the exact
  community port (e.g. a `progen2-small` repo) must load via
  `AutoModelForCausalLM(..., trust_remote_code=True)`, expose a working
  `.generate()`, and ship a tokenizer whose decode yields clean amino-acid
  strings (with control/special tokens identifiable for stripping). The plan
  pins a specific port; the container build + the generate integration test are
  where this is proven. If the chosen port misbehaves, the fallback is vendoring
  the Salesforce ProGen2 modeling/tokenizer files into `containers/progen2/`
  (a container-internal change that does not affect the client or contract).
- **Sampling determinism** under a fixed seed holds for a given device/dtype; the
  determinism integration test runs both generations on the same device.

## Out of scope (later sub-projects)

`embed` for ProGen2; scoring/ranking generated samples inside `generate` (run
`likelihood` on the output FASTA instead); directional (N→C / C→N) generation
controls; beam search; ESM-C; input chunking. Sampling is single-prompt,
single-pass; no batching across prompts beyond what `model.generate` provides.
