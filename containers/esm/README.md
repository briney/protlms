# ESM container

A contract-compliant Docker image wrapping the [ESM](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
family of masked protein language models. It serves both ESM-1b and all ESM-2
sizes (same `EsmForMaskedLM` architecture) and implements the protlms
container contract (see [`../../docs/CONTRACT.md`](../../docs/CONTRACT.md))
using HuggingFace `transformers`, exposing the `manifest`, `embed`, and
`likelihood` subcommands.

The checkpoint, name, and family are selected at build time via the
`ESM_HF_ID`, `ESM_MODEL_NAME`, and `ESM_MODEL_FAMILY` build args, and the
weights are baked into the image, so runtime requires no network access.

## Building

```bash
# tiny demo / CI model (fast to build, ~8M params)
docker build --build-arg ESM_HF_ID=facebook/esm2_t6_8M_UR50D \
  --build-arg ESM_MODEL_NAME=esm2_t6_8M --build-arg ESM_MODEL_FAMILY=esm2 \
  -t protlms-esm:t6_8M containers/esm

# standard workhorse (~650M params)
docker build --build-arg ESM_HF_ID=facebook/esm2_t33_650M_UR50D \
  --build-arg ESM_MODEL_NAME=esm2_t33_650M --build-arg ESM_MODEL_FAMILY=esm2 \
  -t protlms-esm:t33_650M containers/esm

# ESM-1b
docker build --build-arg ESM_HF_ID=facebook/esm1b_t33_650M_UR50S \
  --build-arg ESM_MODEL_NAME=esm1b_t33_650M --build-arg ESM_MODEL_FAMILY=esm1b \
  -t protlms-esm:esm1b_650M containers/esm
```

`ESM_HF_ID` accepts any HuggingFace model id compatible with
`AutoModelForMaskedLM`; `ESM_MODEL_NAME` and `ESM_MODEL_FAMILY` populate the
manifest's `name` and `model_family` fields.

## Running directly (debugging)

```bash
# introspect the manifest (no mounts needed)
docker run --rm protlms-esm:t6_8M manifest

# embed sequences (CPU)
docker run --rm -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-esm:t6_8M embed --input /in/seqs.fasta --output /out --pooling mean

# on GPU
docker run --rm --gpus all -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-esm:t6_8M likelihood --input /in/seqs.fasta --output /out
```

Normally you do not run these by hand — the `protlms` client builds these commands
for you (`protlms embed esm2-8m seqs.fasta -o out/`).

## Models

| Checkpoint | Family | Params | embedding_dim | layers |
|---|---|---|---|---|
| `esm2_t6_8M` | esm2 | 8M | 320 | 6 |
| `esm2_t12_35M` | esm2 | 35M | 480 | 12 |
| `esm2_t30_150M` | esm2 | 150M | 640 | 30 |
| `esm2_t33_650M` | esm2 | 650M | 1280 | 33 |
| `esm1b_t33_650M` | esm1b | 650M | 1280 | 33 |

`esm2_t6_8M` is the demo/CI default; `esm2_t33_650M` is the standard workhorse.

## Notes

- `likelihood` uses masked-marginal pseudo-log-likelihood, which costs O(L)
  forward passes per sequence. This is fine for short sequences; a single-pass
  approximation is a possible future fast-path.
- The image runs on CPU when launched without `--gpus`, and uses CUDA with
  mixed precision when launched with `--gpus all`.
