# The protlms Container Contract

> **Contract version:** `0.4`
>
> This document is the agreement between the `protlms` client and every model
> image. It is the *only* thing the two sides share. Its schemas are mirrored
> exactly by the Pydantic models in [`src/protlms/contract.py`](../src/protlms/contract.py);
> **edit the two together.** The worked examples here are checked in under
> [`tests/data/`](../tests/data/) and validated by `tests/test_contract.py`, so a
> drift between this document and the code fails a test.

A model image is a black box that exposes a small command-line interface. The
client never imports model code, never knows a model's tokenizer, and never
sees its weights. It resolves a model name to an image, runs the image with a
read-only input mount and a writable output mount, and parses the files the
image leaves behind.

---

## 1. Versioning & compatibility

`contract_version` is a `MAJOR.MINOR` string. The client refuses to run against
an image whose **major** differs from its own.

| Image vs. client | Behavior |
|---|---|
| Same major, image minor ≤ client minor | Run normally. |
| Same major, image minor > client minor | Run; the client may warn. Unknown manifest/result fields are ignored. |
| Different major | Refuse with `ContractVersionError`. |

Forward compatibility relies on the client ignoring unknown fields
(`extra="ignore"` on every model). Adding a new capability or field is a
**minor** bump; changing or removing existing semantics is a **major** bump.

---

## 2. Internal CLI

The image's `ENTRYPOINT` is the contract CLI. It MUST expose these subcommands:

```
<entry> manifest
        # Print the manifest JSON (section 3) to stdout and exit 0.
        # Requires no mounts — introspectable with a one-shot `docker run`.

<entry> embed      --input /in/seqs.fasta --output /out
                   [--pooling mean|cls|none] [--layers -1] [--batch-size N] [--device cpu|cuda]

<entry> likelihood --input /in/seqs.fasta --output /out
                   [--batch-size N] [--device cpu|cuda]

<entry> score      --input /in/variants.csv --output /out
                   [--method masked-marginal|wt-marginal] [--batch-size N] [--device cpu|cuda]

<entry> generate   --input /in/prompts.fasta --output /out
                   [--num-samples N] [--temperature T] [--top-p P] [--max-length L]
                   [--seed S] [--batch-size N] [--device cpu|cuda]

<entry> contacts   --input /in/seqs.fasta --output /out
                   [--method categorical-jacobian] [--batch-size N] [--device cpu|cuda]
```

**Flag conventions**

- `--input` — a path inside the container, always under the read-only `/in`.
- `--output` — a directory inside the container, always the writable `/out`.
- `--pooling` — `mean`/`cls` produce one pooled vector per record; `none`
  produces per-residue arrays. The manifest declares which modes are supported.
- `--layers` — comma-separated transformer layer indices; negative indexing
  allowed (`-1` = last). At `0.1` a single index is the supported path.
- `--device` — `cuda` is only valid when the container was launched with
  `--gpus`. If `cuda` is requested but unavailable, the image emits a
  `DeviceUnavailable` error (section 6).

The `protlms` client always launches with mounts `-v <host_in>:/in:ro` and
`-v <host_out>:/out:rw`, and passes `--gpus all` when a GPU is requested.

---

## 3. Manifest schema

Emitted as JSON on stdout by the `manifest` subcommand. Mirrors
`protlms.contract.Manifest`.

| Field | Type | Notes |
|---|---|---|
| `contract_version` | string | `MAJOR.MINOR`. |
| `name` | string | Canonical model name (e.g. `esm2_t6_8M`). |
| `version` | string | Model/image semantic version. |
| `description` | string | Human-readable summary. |
| `model_family` | string | e.g. `esm2`. |
| `capabilities` | string[] | Subset of `embed`, `likelihood`, `score`, `generate`, `contacts`. |
| `embedding_dim` | int | Representation width. |
| `max_sequence_length` | int | Longer inputs are truncated (recorded in `warnings`). |
| `pooling_modes` | string[] | Subset of `mean`, `cls`, `none`. Empty if the model does not support `embed`. |
| `num_layers` | int | Transformer layer count (bounds `--layers`). |
| `min_gpu_memory_gb` | float \| null | `null` ⇒ runs comfortably on CPU. |
| `default_batch_size` | int | Used when `--batch-size` is omitted. |
| `image_digest` | string \| null | Optional. **Resolved by the client at run time** (`docker image inspect`); images need not self-report it. |

**Worked example** ([`tests/data/manifest.example.json`](../tests/data/manifest.example.json)):

```json
{
  "contract_version": "0.4",
  "name": "esm2_t6_8M",
  "version": "1.0.0",
  "description": "ESM2 8M-parameter masked protein language model.",
  "model_family": "esm2",
  "capabilities": ["embed", "likelihood", "score", "contacts"],
  "embedding_dim": 320,
  "max_sequence_length": 1024,
  "pooling_modes": ["mean", "cls", "none"],
  "num_layers": 6,
  "min_gpu_memory_gb": null,
  "default_batch_size": 8
}
```

---

## 4. Input & output conventions

### Inputs

- `embed` / `likelihood`: a FASTA file at `/in/seqs.fasta`. The client
  normalizes headers so each record header is just its id token.
- `generate`: a FASTA file at `/in/prompts.fasta` where each record's sequence
  is treated as a prefix. An empty sequence triggers unconditional sampling.
- `score`: a CSV at `/in/variants.csv` with the schema described below.
- `contacts`: a FASTA file at `/in/seqs.fasta` (same as `embed`/`likelihood`).

### Outputs

Every successful run writes `/out/result.json` (section 5) plus the artifact
files it describes. The client reads `result.json` to discover outputs rather
than globbing the directory.

| Capability | Files written under `/out` |
|---|---|
| `embed`, pooled (`mean`/`cls`) | `embeddings.npz` — one `(embedding_dim,)` float32 array per record, keyed by record id. |
| `embed`, per-residue (`none`) | `per_residue/<id>.npy` — one `(L, embedding_dim)` float32 array per record. |
| `likelihood` | `likelihoods.csv` (schema below). |
| `score` | `scores.csv` (schema below). |
| `generate` | `generated.fasta` — clean amino-acid sequences (control/special tokens stripped). Headers are `{prompt_id}__sample{k}` for k=0..num_samples-1. |
| `contacts` | `contacts/<id>.npy` — one `(L, L)` float32 contact-score matrix per record. |

**Record ids** are sanitized for use as filenames / npz keys: characters
outside `[A-Za-z0-9._-]` become `_`, and collisions are de-duplicated with a
`__N` suffix. With clean ids (the common case) the sanitized id equals the
original.

**`likelihoods.csv` columns:**

| Column | Meaning |
|---|---|
| `record_id` | The (sanitized) record id. |
| `seq_len` | Number of scored residues. |
| `log_likelihood` | Σ over positions of log P(true residue). |
| `mean_log_likelihood` | `log_likelihood / seq_len` (length-normalized). |
| `perplexity` | `exp(-mean_log_likelihood)`. |

The scoring method is recorded in `result.json` under `params.likelihood_method`,
whose value is `"masked_marginal"` (masked LMs such as ESM2) or `"causal"`
(autoregressive LMs such as ProGen2).

**`variants.csv` input columns (for `score`):**

| Column | Meaning |
|---|---|
| `variant_id` | A unique identifier for the variant. |
| `wt_sequence` | The wild-type reference sequence (full length). |
| `mutant` | A mutation descriptor; 1-indexed notation `{WT}{pos}{MUT}` or colon-separated multi-mutants (`A24G:T56S`). A self-substitution (e.g. `A24A`) scores exactly 0. |

**`scores.csv` output columns (for `score`):**

| Column | Meaning |
|---|---|
| `variant_id` | From the input; allows cross-referencing. |
| `mutant` | The mutation descriptor from the input. |
| `n_mutations` | Number of individual mutations in the descriptor (1 for `A24G`, 2 for `A24G:T56S`). |
| `score` | The score value; blank if the variant is invalid (WT-residue mismatch, out-of-range position, or malformed descriptor). See `result.warnings`. |

---

## 5. `result.json` schema

The success summary written to `/out/result.json`. Mirrors
`protlms.contract.Result` + `OutputArtifact`.

| Field | Type | Notes |
|---|---|---|
| `contract_version` | string | |
| `capability` | string | The capability that produced this output. |
| `model_name` | string | |
| `n_input_records` | int | |
| `n_output_records` | int | May differ from input if records were skipped. |
| `artifacts` | OutputArtifact[] | See below. |
| `warnings` | string[] | e.g. truncation notices. Defaults to `[]`. |
| `params` | object(string→string) | Echo of the request params for reproducibility. |

**OutputArtifact**

| Field | Type | Notes |
|---|---|---|
| `path` | string | Relative to `/out`. |
| `kind` | string | `pooled_embeddings`, `per_residue_embeddings`, `likelihoods_csv`, `variant_scores_csv`, `generated_fasta`, or `contact_map` (free string for forward compatibility). |
| `record_ids` | string[] \| null | Records contained in this artifact. |
| `shape` | int[] \| null | Logical array shape, if applicable. |
| `dtype` | string \| null | e.g. `float32`. |

**Worked example (embed)** ([`tests/data/result.embed.example.json`](../tests/data/result.embed.example.json)):

```json
{
  "contract_version": "0.3",
  "capability": "embed",
  "model_name": "esm2_t6_8M",
  "n_input_records": 3,
  "n_output_records": 3,
  "artifacts": [
    {
      "path": "embeddings.npz",
      "kind": "pooled_embeddings",
      "record_ids": ["insulin_b", "gb1", "melittin"],
      "shape": [3, 320],
      "dtype": "float32"
    }
  ],
  "warnings": [],
  "params": {"pooling": "mean", "layers": "-1"}
}
```

**Worked example (score)** ([`tests/data/result.score.example.json`](../tests/data/result.score.example.json)):

```json
{
  "contract_version": "0.3",
  "capability": "score",
  "model_name": "esm2_t6_8M",
  "n_input_records": 3,
  "n_output_records": 3,
  "artifacts": [
    {
      "path": "scores.csv",
      "kind": "variant_scores_csv",
      "record_ids": ["self", "single", "double"]
    }
  ],
  "warnings": [],
  "params": {"method": "masked-marginal"}
}
```

**Worked example (generate)** ([`tests/data/result.generate.example.json`](../tests/data/result.generate.example.json)):

```json
{
  "contract_version": "0.3",
  "capability": "generate",
  "model_name": "progen2-small",
  "n_input_records": 2,
  "n_output_records": 4,
  "artifacts": [
    {"path": "generated.fasta", "kind": "generated_fasta",
     "record_ids": ["prompt1__sample0", "prompt1__sample1", "uncond__sample0", "uncond__sample1"]}
  ],
  "warnings": [],
  "params": {"num_samples": "2", "temperature": "0.8", "top_p": "0.9", "seed": "42"}
}
```

**Worked example (contacts)** ([`tests/data/result.contacts.example.json`](../tests/data/result.contacts.example.json)):

```json
{
  "contract_version": "0.4",
  "capability": "contacts",
  "model_name": "esm2_t6_8M",
  "n_input_records": 2,
  "n_output_records": 2,
  "artifacts": [
    {"path": "contacts/gb1.npy", "kind": "contact_map", "record_ids": ["gb1"], "shape": [56, 56], "dtype": "float32"},
    {"path": "contacts/insulin_b.npy", "kind": "contact_map", "record_ids": ["insulin_b"], "shape": [30, 30], "dtype": "float32"}
  ],
  "warnings": [],
  "params": {"method": "categorical-jacobian", "device": "auto"}
}
```

---

## 6. Error contract

- **Success:** exit code `0` **and** `/out/result.json` present.
- **Failure:** non-zero exit code. The image writes a single JSON object as the
  last line of **stderr**, matching `protlms.contract.ContainerError`:

```json
{"contract_version": "0.3", "error_type": "SequenceTooLong",
 "message": "sequence 'seq1' exceeds max_sequence_length", "details": {"id": "seq1"}}
```

| Field | Type | Notes |
|---|---|---|
| `contract_version` | string \| null | |
| `error_type` | string | e.g. `InvalidInput`, `UnsupportedCapability`, `SequenceTooLong`, `DeviceUnavailable`, `InternalError`. |
| `message` | string | Human-readable. |
| `details` | object(string→string) | Free-form context. Defaults to `{}`. |

The client scans stderr from the last line backwards for the first valid error
object and raises `ContainerExecutionError` carrying its fields, the exit code,
and a tail of stderr. If no structured error is found (e.g. a segfault), it
raises with the raw stderr tail so the failure is still legible.

---

## 7. Adding a model

1. Create `containers/<family>/` with a `Dockerfile` and an entrypoint that
   implements the subcommands above for the capabilities you support.
2. Bake weights into the image at build time (reproducible, offline runtime).
3. Emit a manifest whose `contract_version` matches a major version the client
   supports.
4. Register the image in [`src/protlms/_data/models.yaml`](../src/protlms/_data/models.yaml).

The client does not change. See [`containers/esm/`](../containers/esm/) for a
reference implementation.
