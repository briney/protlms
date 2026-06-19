# plms

Unified toolkit for inference across a variety of protein language models (pLMs).

`plms` is a lightweight client/CLI that gives you **one** interface for embeddings
and likelihoods regardless of the underlying model. The client itself carries no
ML dependencies — each model ships as a standalone Docker image, and the client
talks to images through a standardized [container contract](docs/CONTRACT.md).
See [docs/VISION.md](docs/VISION.md) for the full design.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

Build a model image (the tiny ESM2 demo model), then run inference through the
client:

```bash
# build the demo image (weights baked in)
docker build --build-arg ESM2_CHECKPOINT=esm2_t6_8M -t plms-esm2:t6_8M containers/esm2

plms models list                                          # available models
plms embed      esm2-8m seqs.fasta -o out/ --pooling mean # pooled embeddings (.npz)
plms embed      esm2-8m seqs.fasta -o out/ --pooling none # per-residue embeddings (.npy)
plms likelihood esm2-8m seqs.fasta -o out/                # pseudo-log-likelihoods (.csv)
plms embed      esm2-8m seqs.fasta -o out/ --gpu          # run on GPU
```

```python
import plms

model = plms.load("esm2-8m")
emb = model.embed("seqs.fasta", pooling="mean")
print(emb.pooled())               # {record_id: (embedding_dim,) array}

ll = model.likelihood("seqs.fasta")
print(ll.rows())                  # per-sequence likelihood / perplexity
```

## Project layout

| Path | What it is |
|---|---|
| `src/plms/contract.py` | Contract schemas (manifest, result, errors). |
| `src/plms/registry.py` | Model name → image resolution (`_data/models.yaml`). |
| `src/plms/runner.py` | Docker invocation behind a swappable `Runner` interface. |
| `src/plms/io.py` | FASTA parsing, input staging, output parsing. |
| `src/plms/models.py` | `plms.load()` and the unified `Model` interface. |
| `containers/esm2/` | Reference contract-compliant model image. |
| `docs/CONTRACT.md` | The container contract specification. |

## Development

```bash
pytest                       # run the test suite
ruff check src/ tests/       # lint
ruff format src/ tests/      # format
ty check src/                # type check
```

All tooling is configured in `pyproject.toml` — there are no separate config files.

## License

MIT — see [LICENSE](LICENSE).
