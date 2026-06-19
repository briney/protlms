# ProGen2 container

A contract-compliant Docker image wrapping the [ProGen2](https://huggingface.co/hugohrban/progen2-small)
autoregressive protein language model. It implements the plms container contract
(see [`../../docs/CONTRACT.md`](../../docs/CONTRACT.md)) and exposes the
`manifest`, `generate`, `likelihood`, and hidden `_prefetch` subcommands.

The checkpoint is selected at build time via the `PROGEN2_CHECKPOINT` build arg
and its weights are baked into the image, so runtime requires no network access.

## Building

```bash
# small model (~300M params, 605 MB weights)
docker build --build-arg PROGEN2_CHECKPOINT=progen2-small -t plms-progen2:small containers/progen2

# base model (~760M params)
docker build --build-arg PROGEN2_CHECKPOINT=progen2-base -t plms-progen2:base containers/progen2
```

`PROGEN2_CHECKPOINT` accepts a short name (`progen2-small`, `progen2-base`, …)
which is resolved to `hugohrban/<name>`, or a full HuggingFace id.

## Running directly (debugging)

```bash
# introspect the manifest (no mounts needed)
docker run --rm plms-progen2:small manifest

# generate sequences from prompts (CPU)
mkdir -p in out
printf '>p1\nMAGIC\n>uncond\n\n' > in/prompts.fasta
docker run --rm -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  plms-progen2:small generate \
    --input /in/prompts.fasta --output /out \
    --num-samples 2 --temperature 1.0 --seed 42

# compute causal log-likelihoods (CPU)
docker run --rm -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  plms-progen2:small likelihood --input /in/seqs.fasta --output /out

# GPU (requires --gpus flag at runtime)
docker run --rm --gpus all -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  plms-progen2:small generate --input /in/prompts.fasta --output /out --device cuda
```

Normally you do not run these by hand — the `plms` client builds these commands
for you.

## Models

| Checkpoint | Params | embedding_dim | layers |
|---|---|---|---|
| `progen2-small` | ~300M | 1024 | 12 |
| `progen2-base` | ~760M | 1536 | 18 |

## Port-specific notes (`hugohrban/progen2-*`)

- **Tokenizer**: uses `tokenizers.Tokenizer.from_pretrained()` from the HF
  `tokenizers` library, **not** `AutoTokenizer`. This requires the `tokenizers`
  package (installed separately from `transformers`).
- **Generation**: `model.generate()` is not available; the entrypoint uses a
  manual autoregressive sampling loop with `past_key_values` caching.
- **Special tokens**: PAD=0 (`<|pad|>`), BOS=1 (`<|bos|>`), EOS=2 (`<|eos|>`).
  The "1" and "2" numeric characters in sequences are vocabulary tokens used as
  organism-context markers in the original ProGen2 training format; they are
  stripped from generated output alongside pad/BOS/EOS.
- **`trust_remote_code=True`** is required for `AutoModelForCausalLM` and
  `AutoConfig` because the port ships a custom `modeling_progen.py`.
- **Unconditional generation**: an empty FASTA sequence (header only) uses a
  BOS token as the sole prompt, triggering unconditional sampling.

## Notes

- `likelihood` uses true causal left-to-right log-likelihood in a single forward
  pass per sequence (O(1) forward passes, vs. O(L) for masked-marginal ESM2).
- The image runs on CPU when launched without `--gpus`, and uses CUDA when
  launched with `--gpus all`.
