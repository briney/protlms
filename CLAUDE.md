# protlms

Unified toolkit for inference across a variety of protein language models (pLMs).

## Quick Reference

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
ty check src/

# CLI
protlms --help
```

## Project Structure

```
src/protlms/           # Main package code
  cli.py            # Typer-based command-line interface (entry point: `protlms`)
tests/              # Test suite (mirrors src structure)
```

## Code Conventions

- Python 3.11+ — use modern syntax (type unions with `|`, `match` statements, etc.)
- All public functions and classes need docstrings (Google style)
- Type hints on all function signatures
- Tests go in `tests/` mirroring the src structure: `src/foo/bar.py` → `tests/test_bar.py`
- Ruff handles formatting and linting — don't override its defaults beyond pyproject.toml config

## Before Committing

1. `ruff check --fix src/ tests/` — auto-fix lint issues
2. `ruff format src/ tests/` — format code
3. `pytest` — all tests pass
4. Write a meaningful commit message: `<component>: <what changed and why>`

## Architecture

<!-- Update this section as the project develops. Describe the main components,
     how data flows, and any non-obvious design decisions. -->

The goal is a single, consistent inference interface that wraps a variety of
protein language models so that embeddings, likelihoods, and downstream
predictions can be obtained through one API regardless of the underlying model.
The client carries **no ML dependencies**: each model ships as a standalone
Docker image (weights + deps baked in), and the client talks to images through a
standardized **container contract** ([`docs/CONTRACT.md`](docs/CONTRACT.md)).
Adding a model means publishing a contract-compliant image and a registry entry,
not changing the client. See [`docs/VISION.md`](docs/VISION.md) for the full
rationale.

### Module layout

- `contract.py` — Pydantic schemas the client and images agree on: `Manifest`,
  `Result`/`OutputArtifact`, `ContainerError`, plus `CONTRACT_VERSION` and the
  `Capability`/`PoolingMode` enums. Mirrors `docs/CONTRACT.md` exactly.
- `registry.py` — `Registry`/`ModelEntry`: resolve a model name/alias to a Docker
  image, backed by the packaged `_data/models.yaml`.
- `runner.py` — the `Runner` protocol + `SubprocessDockerRunner`. `build_argv`
  constructs the exact `docker run` command (mounts, `--gpus`); the protocol seam
  lets a Docker-SDK runner drop in later without touching the client.
- `io.py` — FASTA parsing (`read_fasta`), input staging (`stage_inputs`), and
  output parsing (`read_result`, `load_*_embeddings`, `read_likelihoods`).
- `models.py` — the integration layer: `protlms.load(name)` → `Model` with
  `embed`/`likelihood`, returning `EmbeddingResult`/`LikelihoodResult`. Validates
  requests against the manifest, stages inputs, drives the runner, parses outputs.
- `exceptions.py` — the `ProtlmsError` hierarchy.
- `cli.py` — thin Typer wrapper (`protlms models list|embed|likelihood`).

### Data flow

`protlms.load(name)` resolves the image (`registry`), reads its manifest
(`runner.manifest`), and checks contract compatibility. `model.embed(...)`
validates the request against the manifest, stages the FASTA into a temp `/in`
(`io`), builds a `docker run` command (`runner`), executes it, then parses
`/out` (`result.json` + arrays/CSV) back into Python objects.

### Containers

`containers/<family>/` holds each model's build context (Dockerfile + entrypoint
implementing the contract). It is versioned with the client but excluded from the
wheel. `containers/esm/` is the reference implementation — a shared ESM
masked-LM image serving both ESM-1b and ESM-2 (HuggingFace `transformers`;
checkpoint, name, and family chosen by the `ESM_HF_ID`, `ESM_MODEL_NAME`, and
`ESM_MODEL_FAMILY` build args). It now also supports the `contacts` capability.
