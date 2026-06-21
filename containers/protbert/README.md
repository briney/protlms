# ProtBERT container

A contract-compliant Docker image wrapping the
[ProtBERT](https://huggingface.co/Rostlab/prot_bert) masked protein language model
(ProtTrans / Rostlab). It implements the protlms container contract (see
[`../../docs/CONTRACT.md`](../../docs/CONTRACT.md)) using HuggingFace
`transformers`, and exposes the `manifest`, `embed`, `likelihood`, and `score`
subcommands.

The checkpoint is selected at build time via the `PROTBERT_CHECKPOINT` build arg
and its weights are baked into the image, so runtime requires no network access.

## Building

```bash
# UniRef100 (demo / CI default)
docker build --build-arg PROTBERT_CHECKPOINT=prot_bert -t protlms-protbert:uniref100 containers/protbert

# BFD
docker build --build-arg PROTBERT_CHECKPOINT=prot_bert_bfd -t protlms-protbert:bfd containers/protbert
```

`PROTBERT_CHECKPOINT` accepts `prot_bert` (UniRef100) or `prot_bert_bfd` (BFD),
resolved to `Rostlab/<name>`, or a full HuggingFace id. Both checkpoints are
released under AFL-3.0 and download without authentication.

## Running directly (debugging)

```bash
docker run --rm protlms-protbert:uniref100 manifest

docker run --rm -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-protbert:uniref100 embed --input /in/seqs.fasta --output /out --pooling mean

docker run --rm --gpus all -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-protbert:uniref100 likelihood --input /in/seqs.fasta --output /out
```

Normally you do not run these by hand — the `protlms` client builds these commands
for you (`protlms embed protbert seqs.fasta -o out/`).

## Models

| Checkpoint | Training data | Params | embedding_dim | layers |
|---|---|---|---|---|
| `prot_bert` | UniRef100 | ~420M | 1024 | 30 |
| `prot_bert_bfd` | BFD | ~420M | 1024 | 30 |

## Notes

- **Tokenization:** ProtBERT expects whitespace-separated residues and was trained
  with the rare residues U, Z, O, B mapped to X. The entrypoint applies both
  transformations automatically (`preprocess`), so the client just passes plain
  FASTA. One nuance for `score`: a wild-type residue that is U/Z/O/B still matches
  the raw input (the WT-residue check passes), but because the model context is
  X-substituted, scores at or near such rare residues are approximate. This is
  uncommon — standard sequences are unaffected.
- `likelihood` uses masked-marginal pseudo-log-likelihood (O(L) forward passes per
  sequence) and records `params.likelihood_method = "masked_marginal"`.
- `embed` supports arbitrary `--layers` via hidden states; `cls` pooling uses the
  `[CLS]` token, `mean` averages over residue positions.
- `max_sequence_length = 1024` (ProtBERT was trained at 512/2048; longer inputs are
  truncated with a warning).
- The image runs on CPU when launched without `--gpus`, and uses CUDA with mixed
  precision when launched with `--gpus all`.
