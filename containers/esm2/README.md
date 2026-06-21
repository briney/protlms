# ESM2 container

A contract-compliant Docker image wrapping the [ESM2](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
masked protein language model. It implements the protlms container contract
(see [`../../docs/CONTRACT.md`](../../docs/CONTRACT.md)) using HuggingFace
`transformers`, and exposes the `manifest`, `embed`, and `likelihood`
subcommands.

The checkpoint is selected at build time via the `ESM2_CHECKPOINT` build arg and
its weights are baked into the image, so runtime requires no network access.

## Building

```bash
# tiny demo / CI model (fast to build, ~8M params)
docker build --build-arg ESM2_CHECKPOINT=esm2_t6_8M -t protlms-esm2:t6_8M containers/esm2

# standard workhorse (~650M params)
docker build --build-arg ESM2_CHECKPOINT=esm2_t33_650M -t protlms-esm2:t33_650M containers/esm2
```

`ESM2_CHECKPOINT` accepts a short ESM2 name (`esm2_t6_8M`, `esm2_t33_650M`, …),
which is resolved to `facebook/<name>_UR50D`, or a full HuggingFace id.

## Running directly (debugging)

```bash
# introspect the manifest (no mounts needed)
docker run --rm protlms-esm2:t6_8M manifest

# embed sequences (CPU)
docker run --rm -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-esm2:t6_8M embed --input /in/seqs.fasta --output /out --pooling mean

# on GPU
docker run --rm --gpus all -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-esm2:t6_8M likelihood --input /in/seqs.fasta --output /out
```

Normally you do not run these by hand — the `protlms` client builds these commands
for you (`protlms embed esm2-8m seqs.fasta -o out/`).

## Models

| Checkpoint | Params | embedding_dim | layers |
|---|---|---|---|
| `esm2_t6_8M` | 8M | 320 | 6 |
| `esm2_t12_35M` | 35M | 480 | 12 |
| `esm2_t30_150M` | 150M | 640 | 30 |
| `esm2_t33_650M` | 650M | 1280 | 33 |

`esm2_t6_8M` is the demo/CI default; `esm2_t33_650M` is the standard workhorse.

## Notes

- `likelihood` uses masked-marginal pseudo-log-likelihood, which costs O(L)
  forward passes per sequence. This is fine for short sequences; a single-pass
  approximation is a possible future fast-path.
- The image runs on CPU when launched without `--gpus`, and uses CUDA with
  mixed precision when launched with `--gpus all`.
