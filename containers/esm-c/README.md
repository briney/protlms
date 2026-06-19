# ESM-C container

A contract-compliant Docker image wrapping the
[ESM-C](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12) masked
protein language model (EvolutionaryScale). It implements the plms container
contract (see [`../../docs/CONTRACT.md`](../../docs/CONTRACT.md)) using the native
`esm` SDK, and exposes the `manifest`, `embed`, `likelihood`, and `score`
subcommands.

The checkpoint is selected at build time via the `ESMC_CHECKPOINT` build arg and
its weights are baked into the image, so runtime requires no network access.

## Building

```bash
# 300M (demo / CI default)
docker build --build-arg ESMC_CHECKPOINT=esmc_300m -t plms-esm-c:300m containers/esm-c

# 600M
docker build --build-arg ESMC_CHECKPOINT=esmc_600m -t plms-esm-c:600m containers/esm-c
```

`ESMC_CHECKPOINT` accepts `esmc_300m` or `esmc_600m`. The 300M/600M weights are
open (Cambrian Open License) and download without authentication. The 6B model is
EvolutionaryScale Forge API-only and is not supported by this image.

## Running directly (debugging)

```bash
docker run --rm plms-esm-c:300m manifest

docker run --rm -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  plms-esm-c:300m embed --input /in/seqs.fasta --output /out --pooling mean

docker run --rm --gpus all -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  plms-esm-c:300m likelihood --input /in/seqs.fasta --output /out
```

Normally you do not run these by hand — the `plms` client builds these commands
for you (`plms embed esm-c-300m seqs.fasta -o out/`).

## Models

| Checkpoint | Params | embedding_dim | layers |
|---|---|---|---|
| `esmc_300m` | 300M | 960 | 30 |
| `esmc_600m` | 600M | 1152 | 36 |

## Notes

- Uses the native `esm` SDK (`ESMC.from_pretrained`), which requires Python 3.12;
  this base image therefore differs from the ESM2 image.
- `likelihood` uses masked-marginal pseudo-log-likelihood (O(L) forward passes per
  sequence) and records `params.likelihood_method = "masked_marginal"`.
- `embed` returns the **final-layer** representation; `--layers` must be `-1`
  (the client default). Other layer indices return an `InvalidInput` error.
- flash-attn is intentionally not installed, so the SDK uses standard attention; the image runs on CPU and uses the GPU when launched with --gpus all.
