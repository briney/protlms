# plms

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
plms --help
```

## Project Structure

```
src/plms/           # Main package code
  cli.py            # Typer-based command-line interface (entry point: `plms`)
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
protein language models (e.g. ESM, ProtT5, AntiBERTy, and others) so that
embeddings, likelihoods, and downstream predictions can be obtained through one
API regardless of the underlying model. As models are added, document each
backend adapter and the shared interface they implement here.

TODO: Fill in module layout (model adapters, shared interface, tokenization,
batching) as the project takes shape.
