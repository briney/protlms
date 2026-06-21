# Profluent-E1 container (single-sequence mode)

A contract-compliant Docker image wrapping the
[Profluent-E1](https://huggingface.co/Profluent-Bio/E1-150m) masked protein
language model. It implements the protlms container contract (see
[`../../docs/CONTRACT.md`](../../docs/CONTRACT.md)) using the custom `E1` package,
and exposes the `manifest`, `embed`, `likelihood`, and `score` subcommands.

**Single-sequence mode only.** E1 can also run retrieval-augmented (with homolog
context); this image deliberately does not expose that — every sequence is scored
on its own.

The checkpoint is selected at build time via the `E1_CHECKPOINT` build arg and its
weights are baked into the image, so runtime requires no network access.

## Building

```bash
# 150m (demo / CI default)
docker build --build-arg E1_CHECKPOINT=E1-150m -t protlms-e1:150m containers/e1

# 300m / 600m
docker build --build-arg E1_CHECKPOINT=E1-300m -t protlms-e1:300m containers/e1
docker build --build-arg E1_CHECKPOINT=E1-600m -t protlms-e1:600m containers/e1
```

`E1_CHECKPOINT` accepts `E1-150m`, `E1-300m`, or `E1-600m`, resolved to
`Profluent-Bio/<name>`. The `E1` package is pinned to a specific git commit (see
the `E1_REF` build arg in the Dockerfile) for reproducibility.

## Running directly (debugging)

```bash
docker run --rm protlms-e1:150m manifest

docker run --rm -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-e1:150m embed --input /in/seqs.fasta --output /out --pooling mean

docker run --rm --gpus all -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-e1:150m likelihood --input /in/seqs.fasta --output /out
```

Normally you do not run these by hand — the `protlms` client builds these commands
for you (`protlms embed e1-150m seqs.fasta -o out/`).

## Models

| Checkpoint | Params | embedding_dim | layers |
|---|---|---|---|
| `E1-150m` | ~154M | 768 | 20 |
| `E1-300m` | ~274M | 1024 | 20 |
| `E1-600m` | ~641M | 1280 | 30 |

## Notes

- Uses the custom `E1` package (`E1ForMaskedLM`, `E1BatchPreparer`), which requires
  Python 3.12; this base image therefore differs from the ESM2/ProtBERT images.
- `likelihood` uses masked-marginal pseudo-log-likelihood (O(L) forward passes per
  sequence) and records `params.likelihood_method = "masked_marginal"`. The mask
  token is `?`.
- `embed` returns the **final-layer** representation; `--layers` must be `-1` (the
  client default). Other layer indices return an `InvalidInput` error. `cls`
  pooling uses the `<bos>` vector; `mean` averages over residue positions.
- `max_sequence_length = 2048` (the model supports longer within-sequence context,
  but masked-marginal scoring is O(L) forward passes).
- flash-attn is not installed; the model runs on CPU via the flex_attention
  fallback and uses the GPU when launched with `--gpus all` (recommended for the
  larger checkpoints).

## License & attribution

The `E1` **code** is Apache-2.0. The **weights** are free for research and
commercial use **with attribution** under Profluent's model license. If you use
this image or its outputs, follow the upstream attribution requirements in the
[E1 repository](https://github.com/Profluent-AI/E1) `NOTICE`/`ATTRIBUTION` files.
The paper text is separately licensed CC-BY-NC-ND.
