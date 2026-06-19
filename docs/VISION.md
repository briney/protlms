# plms — Vision

> **Status:** Draft / source of truth for the high-level vision. Decisions captured
> here reflect deliberate choices; sections marked _(proposed)_ are sensible
> defaults still open for revision.

## Mission

`plms` is a lightweight, pip-installable Python package and CLI that provides a
**single, unified interface for inference across many protein language models
(pLMs)** — so embeddings, likelihoods, variant-effect scores, and generated
sequences can be obtained the same way regardless of the underlying model.

## The Problem

Protein language models are proliferating (ESM2, ESM-C, ProtT5, ProGen2, …),
and each ships its own:

- API surface, tokenizer, and special-token conventions,
- weights and download mechanics, and
- **mutually incompatible dependency stack** — e.g. one model pins `torch<=2.10`
  while another requires `torch>=2.11`, making it impossible to install both in
  one environment.

The consequences: using more than one model in a single workflow is painful,
swapping one model for another means rewriting glue code, and benchmarking
across models is an exercise in dependency-hell archaeology and irreproducible
one-off scripts.

## What `plms` Is

- A **lightweight client / orchestrator**. It presents one consistent API and
  CLI and hides every model-specific detail behind a standardized boundary.
- The package itself carries **no ML dependencies** (no `torch`, no
  `transformers`). It depends only on orchestration tooling.
- Each model is shipped as a **fully standalone Docker image** — its own
  dependency stack *and its weights* baked in. The image is the unit of
  versioning and reproducibility.
- `plms` talks to these images through a **standardized container contract**
  (see below). Adding a model means publishing a contract-compliant image, not
  changing the client.

## What `plms` Is Not (Non-Goals)

- **Not** a training or fine-tuning framework — inference only.
- **Not** a place where ML dependencies or model weights live inside the Python
  package. Those belong in containers.
- **Not** a long-running inference server. v1 is one-shot batch execution
  (see [Execution Model](#execution-model-one-shot-batch)).
- **Not** (yet) a benchmark suite. The API is designed to make benchmarking
  trivial, but the harness itself is a later phase.
- **Not** specialized for antibody or other domain-specific models. Antibody
  language models carry their own complexities (paired vs. unpaired chains,
  numbering schemes, region-aware features) that would broaden scope. `plms`
  deliberately targets **general** protein language models and aims to do that
  one job very well.

## Design Principles

1. **Lightweight client, heavy containers.** All weight and dependency burden
   lives behind the container boundary; the installed package stays tiny.
2. **Contract over special-casing.** The client never knows about a specific
   model. It speaks one contract; models comply with it.
3. **Stateless, reproducible jobs.** Every job is a fresh container run with
   pinned image digests and recorded configuration.
4. **Opaque internals.** Tokenization, pooling math, and model quirks are the
   container's responsibility. The client deals in sequences and standard
   options.
5. **Scale-ready, not over-engineered.** The design must not preclude
   large-scale inputs, but v1 keeps the machinery simple.

## Architecture Overview

```
  user code / CLI
        │
        ▼
  ┌───────────────┐     reads manifest, builds `docker run`,
  │  plms client  │ ──▶ mounts I/O, parses outputs, surfaces errors
  └───────────────┘
        │ Docker SDK (local daemon)
        ▼
  ┌──────────────────────────────────────────┐
  │  model image  (weights + deps baked in)   │
  │  ├─ manifest  (declares capabilities)     │
  │  └─ internal CLI: embed | likelihood |    │
  │                   score | generate        │
  │  reads /in (ro)  →  writes /out (rw)       │
  └──────────────────────────────────────────┘
```

### The Client (`plms` package)

- **Dependencies (intended):** Docker SDK, `pydantic`, `typer`, `rich`,
  `pyyaml`, and `numpy` (only to load result arrays). Explicitly **no** deep-
  learning libraries.
- **Responsibilities:** model registry & name resolution, request validation
  against the model's manifest, Docker lifecycle management, mount and input
  chunking, output parsing into Python objects, and clear error surfacing.

### Execution Model (One-Shot Batch)

- A **job is a single `docker run`**. The container loads the model once,
  processes an entire input file (e.g. a multi-record FASTA), writes results to
  a mounted output directory, and exits.
- The unit of work is therefore a **file of many sequences**, never a single
  sequence — this amortizes the multi-GB model-load cost across the batch.
- Jobs are stateless and reproducible; nothing persists between runs except the
  outputs the user asked for.

### Orchestration

`plms` drives the **local Docker daemon** via the Docker SDK:

1. Resolve a model name to an image reference.
2. Ensure the image is present (v1: built locally from in-tree Dockerfiles).
3. Run it with the right `--gpus`, read-only input mount, writable output
   mount, and the contract subcommand + arguments.
4. Check exit code and capture logs.
5. Parse outputs into Python objects and clean up.

Remote registries / pre-existing remote endpoints are explicitly a later phase
(see [Roadmap](#roadmap)).

## The Unified Interface (Capabilities)

Four operations make up the v1 API. Each maps to one contract subcommand. Not
every model supports all four — the manifest declares support and the client
validates requests before launching a container.

| Capability | What it returns |
|---|---|
| **Embeddings** | Per-residue and pooled (mean / cls / …) representations. |
| **Likelihoods** | Sequence-level pseudo-perplexity / pseudo-log-likelihood. |
| **Variant scoring** | Masked-marginal / mutant-vs-wildtype effect scores. |
| **Generation** | Conditional/unconditional sampled sequences (generative pLMs). |

**Illustrative** Python and CLI surface _(provisional)_:

```python
import plms

model = plms.load("esm2-650m")                       # resolves & checks the image
emb   = model.embed("seqs.fasta", pooling="mean")    # -> array handle
ll    = model.likelihood("seqs.fasta")               # -> per-sequence scores
vfx   = model.score("variants.csv")                  # -> variant-effect table
gen   = plms.load("progen2").generate("prompts.fasta", num_samples=10)
```

```bash
plms models list
plms embed     esm2-650m seqs.fasta     -o out/ --pooling mean
plms likelihood esm2-650m seqs.fasta    -o out/
plms score     esm2-650m variants.csv   -o out/
plms generate  progen2   prompts.fasta  -o out/ --num-samples 10
```

## The Container Contract

The contract is the linchpin of the whole project: it is the *only* thing the
client and the images agree on.

### Manifest

Each image emits a JSON manifest via a `manifest` subcommand (so it can be
introspected with a one-shot run, no file extraction needed). Fields _(proposed)_:

- `name`, `version`, `description`, `model_family`
- `capabilities`: subset of `["embed", "likelihood", "score", "generate"]`
- `embedding_dim`, `max_sequence_length`, supported `pooling` modes
- hardware hints: `min_gpu_memory_gb`, default batch size
- `image_digest`, `contract_version`

### Internal CLI

A conventional entrypoint exposing standardized subcommands:

```
<entrypoint> manifest                                  # prints JSON to stdout
<entrypoint> embed      --input /in/... --output /out/ [--pooling ...] [--layers ...]
<entrypoint> likelihood --input /in/... --output /out/
<entrypoint> score      --input /in/... --output /out/
<entrypoint> generate   --input /in/... --output /out/ [--num-samples N] [--temperature T]
```

### I/O Convention

- Inputs arrive in a **read-only** mounted directory; outputs go to a
  **writable** mounted directory.
- A `result.json` summary describes what was produced (files, shapes, counts).
- Exit code `0` = success; non-zero = failure with a structured error on stderr.

### Versioning

`contract_version` lets the client check image/client compatibility and refuse
to run mismatched pairs with a clear message.

## Data Formats _(proposed)_

- **Inputs:** FASTA for sequences; CSV/TSV with a defined schema for variants.
- **Outputs:** pooled embeddings → `.npy` / `.npz`; ragged per-residue
  embeddings → HDF5 (or one file per record); likelihoods & variant scores →
  CSV / Parquet; generated sequences → FASTA.

## Scale & Performance

The design must **not preclude large-scale inputs** (millions of sequences),
even though v1 targets moderate sizes:

- The client **shards large inputs** into chunks and runs jobs over them,
  streaming/concatenating outputs so a single logical request can span many
  container runs.
- Chunk sizes are chosen to amortize the per-job model-load cost.
- **Resumability** (skip already-completed shards) is desirable.
- Parallel / multi-GPU fan-out is a later optimization, but the chunking
  boundary is where it will slot in without an API change.

## Repository Layout

```
plms/
├── pyproject.toml
├── docs/
│   └── VISION.md
├── src/plms/              # lightweight client (no ML deps)
│   ├── cli.py             # Typer entry point (`plms`)
│   ├── registry.py        # model name → image resolution
│   ├── contract.py        # manifest schema + I/O conventions
│   ├── runner.py          # Docker lifecycle, mounts, chunking
│   └── io.py              # input/output format handling
├── containers/            # build contexts — EXCLUDED from the wheel
│   ├── esm2/
│   ├── esm-c/
│   └── progen2/           # each: Dockerfile + entrypoint + contract impl
├── tests/
├── scripts/
└── configs/
```

`containers/` is versioned alongside the client so the contract evolves in
lockstep, but it is excluded from the installed package to keep
`pip install plms` lightweight.

## First Models

The first wave is **ESM2, ESM-C, and ProGen2**. Together they:

- cover a general masked-LM workhorse (ESM2), a modern general model (ESM-C),
  and an autoregressive generative model (ProGen2), and
- exercise **all four capabilities** end-to-end, proving the contract
  generalizes across architectures.

Additional general-purpose models (e.g. ProtT5) are natural fast follows; each
new model further validates that model-specific tokenization and preprocessing
hide cleanly behind the contract.

## Roadmap

- **Phase 0 — Contract + skeleton.** Lock the container contract spec, build the
  `plms` client skeleton, and get **one model (ESM2)** working end-to-end with a
  locally built image.
- **Phase 1 — Generalize.** Add ESM-C and ProGen2; finalize the contract across
  all four capabilities; implement input chunking for scale.
- **Phase 2 — Distribution.** Image hosting (e.g. GHCR) with auto-pull and
  digest pinning; optional connection to pre-existing remote endpoints.
- **Phase 3 — Benchmarking.** A harness for standard tasks, datasets, and
  metrics with cross-model comparison (likely a companion concern).
- **Later.** Additional general pLMs; parallel/multi-GPU fan-out; further
  capabilities (attention/contacts, structure) as demand warrants.

## Open Questions / Deferred Decisions

- Exact on-disk output formats (HDF5 vs `.npz` for ragged per-residue arrays).
- Image hosting/registry choice and how to handle multi-GB images.
- Parallel / multi-GPU fan-out strategy for large-scale jobs.
- Whether/how to cache results across identical requests.
- Variant input specification (mutation strings vs. explicit full sequences).
