# ESM-C container

A contract-compliant Docker image wrapping the ESM-C masked protein language
model, served via HuggingFace `transformers` using the MIT-licensed weights
published by [biohub](https://github.com/Biohub/esm). The biohub `esm` package
registers the `ESMCForMaskedLM` architecture with transformers' auto classes
(native `transformers` has no built-in `esmc` support) and is installed
directly from its git repository, pinned to a fixed commit (no PyPI release
yet). It implements the protlms container contract (see
[`../../docs/CONTRACT.md`](../../docs/CONTRACT.md)) and exposes the
`manifest`, `embed`, `likelihood`, `score`, and `contacts` subcommands.

The checkpoint is selected at build time via the `ESMC_CHECKPOINT` build arg and
its weights are baked into the image, so runtime requires no network access.

## Building

```bash
# 300M (demo / CI default)
docker build --build-arg ESMC_CHECKPOINT=esmc_300m -t protlms-esm-c:300m containers/esm-c

# 600M / 6B
docker build --build-arg ESMC_CHECKPOINT=esmc_600m -t protlms-esm-c:600m containers/esm-c
docker build --build-arg ESMC_CHECKPOINT=esmc_6b   -t protlms-esm-c:6b   containers/esm-c
```

`ESMC_CHECKPOINT` accepts `esmc_300m`, `esmc_600m`, or `esmc_6b`. All three
checkpoints are MIT-licensed biohub weights and download without
authentication.

## Running directly (debugging)

```bash
docker run --rm protlms-esm-c:300m manifest

docker run --rm -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-esm-c:300m embed --input /in/seqs.fasta --output /out --pooling mean

docker run --rm --gpus all -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-esm-c:300m likelihood --input /in/seqs.fasta --output /out

docker run --rm --gpus all -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-esm-c:300m contacts --input /in/seqs.fasta --output /out
```

Normally you do not run these by hand — the `protlms` client builds these commands
for you (`protlms embed esm-c-300m seqs.fasta -o out/`).

## Models

| Checkpoint | Params | embedding_dim | layers |
|---|---|---|---|
| `esmc_300m` | 300M | 960 | 30 |
| `esmc_600m` | 600M | 1152 | 36 |
| `esmc_6b` | 6B | 2560 | 80 |

## Notes

- Uses HuggingFace `transformers` (`AutoModelForMaskedLM`/`AutoTokenizer`), with
  the biohub `esm` package installed solely to register the `ESMCForMaskedLM`
  architecture; this base image requires Python 3.12.
- `likelihood` uses masked-marginal pseudo-log-likelihood (O(L) forward passes per
  sequence) and records `params.likelihood_method = "masked_marginal"`.
- `contacts` predicts an (L, L) contact map per sequence via the categorical
  Jacobian (Zhang/Ovchinnikov pipeline: symmetrize, Frobenius norm over the
  amino-acid axes, average product correction).
- `embed` returns the **final-layer** representation; `--layers` must be `-1`
  (the client default). Other layer indices return an `InvalidInput` error.
- flash-attn is intentionally not installed, so the model uses standard attention.
  The image runs on CPU and uses the GPU when launched with `--gpus all`.
