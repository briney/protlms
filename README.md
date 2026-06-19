# plms

Unified toolkit for inference across a variety of protein language models (pLMs).

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
plms --help        # show available commands
plms version       # print the installed version
```

```python
import plms

print(plms.__version__)
```

## Development

```bash
pytest                       # run the test suite
ruff check src/ tests/       # lint
ruff format src/ tests/      # format
mypy src/                    # type check
```

All tooling is configured in `pyproject.toml` — there are no separate config files.

## License

MIT — see [LICENSE](LICENSE).
