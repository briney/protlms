# Agent Instructions

General instructions for AI coding agents working on this project.
Not specific to any single agent or tool.

## Project Overview

`plms` is a unified toolkit for running inference across a variety of protein
language models (pLMs). It aims to provide one consistent API for obtaining
embeddings, sequence likelihoods, and downstream predictions regardless of which
underlying model (ESM, ProtT5, antibody-specific models, etc.) is used.

## Architecture

<!-- Describe the main modules/components, their responsibilities, and how they interact.
     Include data flow if relevant. -->

TODO: Fill in as the project develops. Expected shape: a shared inference
interface, per-model backend adapters, tokenization/batching utilities, and the
`plms` CLI (`src/plms/cli.py`) as a thin wrapper over the library.

## Development Workflow

- Install: `pip install -e ".[dev]"`
- Test: `pytest`
- Lint: `ruff check src/ tests/`
- Format: `ruff format src/ tests/`
- Type check: `mypy src/`

Always run tests and lint before considering work complete.

## Conventions

- **Formatting**: Ruff handles this. Do not manually adjust formatting.
- **Imports**: Sorted by ruff (isort rules). Do not manually reorder.
- **Types**: All function signatures must have type annotations. Use modern syntax (`X | None` not `Optional[X]`).
- **Docstrings**: Google style. Required on all public functions, classes, and modules.
- **Tests**: Every new function or class should have corresponding tests. Use `pytest` fixtures, not `setUp`/`tearDown`.
- **Error handling**: Prefer specific exceptions over bare `except`. Define custom exceptions in an `exceptions.py` module if the project needs them.
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE for constants.

## What Not to Do

- Do not add dependencies without explicit approval. Prefer the standard library when reasonable.
- Do not create separate config files (`.flake8`, `pytest.ini`, `mypy.ini`, etc.). All tool config lives in `pyproject.toml`.
- Do not use `setup.py`, `setup.cfg`, or `requirements.txt`.
- Do not modify CI configuration without explicit approval.
- Do not add `# type: ignore` without a comment explaining why.
- Do not write tests that depend on external services or network access without mocking.

## Sensitive Areas

<!-- List files, modules, or patterns that need extra care — e.g., security-critical code,
     performance-sensitive paths, or tricky algorithms. Agents should be extra cautious
     modifying these and should flag changes for human review. -->

TODO: Fill in as the project develops. Model weight downloads and any code that
hits external model registries will warrant extra care.
