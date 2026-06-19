# `generate` + ProGen2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `generate` capability and a ProGen2 (autoregressive) container, and neutralize the shared likelihood schema so masked-marginal (ESM2) and causal (ProGen2) likelihoods are comparable.

**Architecture:** Mirrors the existing contract pattern — generation + likelihood math live in the ProGen2 container; the client validates input, drives `docker run`, and parses outputs. ProGen2 loads from a HuggingFace community port via `AutoModelForCausalLM(..., trust_remote_code=True)`. Contract bumps 0.2 → 0.3 (minor; same major stays mutually readable).

**Tech Stack:** Python 3.11+, Pydantic v2, Typer, NumPy (client); HuggingFace `transformers` + PyTorch (containers). pytest, ruff, ty.

## Global Constraints

- Python 3.11+; `from __future__ import annotations` in every source module **except `cli.py`**. Client carries **no ML deps** (no torch/transformers/pandas); stdlib `csv`, NumPy only for arrays.
- Google docstrings on public functions/classes; full type hints; functions < ~50 lines; ruff line length 100; `ruff check`/`ruff format`/`ty check src/` stay clean.
- `CONTRACT_VERSION = "0.3"`. `src/plms/contract.py` mirrors `docs/CONTRACT.md`.
- New artifact kind: `ArtifactKind.GENERATED_FASTA = "generated_fasta"`.
- **Neutralized likelihood CSV columns (exact):** `record_id,seq_len,log_likelihood,mean_log_likelihood,perplexity`. The method is recorded in `result.json` `params.likelihood_method` ∈ {`masked_marginal` (ESM2), `causal` (ProGen2)}.
- `generate` subcommand flags: `--num-samples N --temperature T --top-p P --max-length L --seed S --batch-size N --device cpu|cuda`. Input `prompts.fasta`; **empty sequence = unconditional**. Output `generated.fasta`, headers `{prompt_id}__sample{k}` (k=0..num_samples-1), clean amino-acid sequences (control/special tokens stripped).
- ProGen2 manifest: `capabilities: ["generate", "likelihood"]`, `pooling_modes: []`, real `embedding_dim`/`num_layers`, `contract_version "0.3"`. Demo/CI checkpoint `progen2-small` → HF port `hugohrban/progen2-small`, loaded with `trust_remote_code=True`.
- ESM2 image is rebuilt and bumped to contract `0.3`; its likelihood output uses the neutral columns + `likelihood_method="masked_marginal"`.

---

### Task 1: Contract — 0.3, generated_fasta kind, fixtures

**Files:**
- Modify: `src/plms/contract.py`, `tests/data/manifest.example.json`, `tests/data/result.embed.example.json`, `tests/data/result.score.example.json`, `tests/test_contract.py`
- Create: `tests/data/result.generate.example.json`

**Interfaces:**
- Produces: `CONTRACT_VERSION == "0.3"`; `ArtifactKind.GENERATED_FASTA == "generated_fasta"`.

- [ ] **Step 1: Update fixtures + tests (failing)**

Set `"contract_version": "0.3"` in `manifest.example.json`, `result.embed.example.json`, and `result.score.example.json`. Create `tests/data/result.generate.example.json`:

```json
{
  "contract_version": "0.3",
  "capability": "generate",
  "model_name": "progen2-small",
  "n_input_records": 2,
  "n_output_records": 4,
  "artifacts": [
    {"path": "generated.fasta", "kind": "generated_fasta",
     "record_ids": ["prompt1__sample0", "prompt1__sample1", "uncond__sample0", "uncond__sample1"]}
  ],
  "warnings": [],
  "params": {"num_samples": "2", "temperature": "0.8", "top_p": "0.9", "seed": "42"}
}
```

In `tests/test_contract.py`: change `test_contract_version_is_semantic_string` to assert `CONTRACT_VERSION == "0.3"` and `parse_contract_version(CONTRACT_VERSION) == (0, 3)`. Add:

```python
def test_documented_generate_result_example_validates() -> None:
    """The generate result example in docs/CONTRACT.md must parse as a Result."""
    result = Result.model_validate_json((_DATA / "result.generate.example.json").read_text())
    assert result.capability is Capability.GENERATE
    assert result.artifacts[0].kind == "generated_fasta"
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_contract.py -q`
Expected: FAIL — `CONTRACT_VERSION` is `"0.2"`; `GENERATED_FASTA` not defined.

- [ ] **Step 3: Implement**

In `src/plms/contract.py` set `CONTRACT_VERSION = "0.3"` and add to `ArtifactKind`:

```python
    GENERATED_FASTA = "generated_fasta"
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_contract.py -q` → PASS. Then `python -m pytest -q` — existing tests still pass (0.2 manifests remain compatible: same major).

- [ ] **Step 5: Commit**

```bash
git add src/plms/contract.py tests/test_contract.py tests/data/
git commit -m "contract: bump to 0.3 and add generated_fasta artifact kind"
```

---

### Task 2: io — neutralize likelihood columns + read_generated

**Files:**
- Modify: `src/plms/io.py`, `tests/test_io.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `_artifacts`, `ArtifactKind`, `Result`, `read_fasta`, `FastaRecord` (existing in io).
- Produces: `read_generated(out_dir: Path, result: Result) -> list[FastaRecord]`; `read_likelihoods` now returns rows keyed `record_id, seq_len, log_likelihood, mean_log_likelihood, perplexity`.

- [ ] **Step 1: Write/adjust failing tests**

In `tests/test_io.py`, update `test_read_likelihoods_coerces_numeric_columns` to the neutral columns and add a generated-reader test:

```python
def test_read_likelihoods_coerces_numeric_columns(tmp_path: Path) -> None:
    (tmp_path / "likelihoods.csv").write_text(
        "record_id,seq_len,log_likelihood,mean_log_likelihood,perplexity\nseq1,5,-3.5,-0.7,2.01\n"
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "contract_version": "0.3",
                "capability": "likelihood",
                "model_name": "m",
                "n_input_records": 1,
                "n_output_records": 1,
                "artifacts": [{"path": "likelihoods.csv", "kind": "likelihoods_csv"}],
            }
        )
    )
    result = read_result(tmp_path)
    rows = read_likelihoods(tmp_path, result)
    assert rows[0]["seq_len"] == 5
    assert rows[0]["perplexity"] == pytest.approx(2.01)


def test_read_generated(tmp_path: Path) -> None:
    from plms.io import read_generated

    (tmp_path / "generated.fasta").write_text(">p__sample0\nACDE\n>p__sample1\nFGHI\n")
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "contract_version": "0.3",
                "capability": "generate",
                "model_name": "m",
                "n_input_records": 1,
                "n_output_records": 2,
                "artifacts": [{"path": "generated.fasta", "kind": "generated_fasta"}],
            }
        )
    )
    result = read_result(tmp_path)
    records = read_generated(tmp_path, result)
    assert [r.id for r in records] == ["p__sample0", "p__sample1"]
    assert records[0].sequence == "ACDE"
```

In `tests/test_models.py`, update `FakeRunner._write_likelihood` to emit the neutral header/row and update the likelihood assertion:

```python
    def _write_likelihood(self, out: Path, records) -> None:  # noqa: ANN001
        lines = ["record_id,seq_len,log_likelihood,mean_log_likelihood,perplexity"]
        for rec in records:
            lines.append(f"{rec.id},{len(rec.sequence)},-3.5,-0.7,2.01")
        (out / "likelihoods.csv").write_text("\n".join(lines) + "\n")
        self._write_result(
            out, "likelihood", records, [{"path": "likelihoods.csv", "kind": "likelihoods_csv"}]
        )
```
and in `test_likelihood_returns_rows` change `rows[0]["pseudo_perplexity"]` to `rows[0]["perplexity"]`.

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_io.py tests/test_models.py -q`
Expected: FAIL — `read_generated` missing; `read_likelihoods` still maps `pseudo_*` keys.

- [ ] **Step 3: Implement in `src/plms/io.py`**

Change `_LIKELIHOOD_COLUMN_TYPES` to the neutral keys:

```python
_LIKELIHOOD_COLUMN_TYPES: dict[str, type] = {
    "seq_len": int,
    "log_likelihood": float,
    "mean_log_likelihood": float,
    "perplexity": float,
}
```

Add `read_generated` near the other readers:

```python
def read_generated(out_dir: Path, result: Result) -> list[FastaRecord]:
    """Load generated sequences from the ``generated_fasta`` artifact.

    Raises:
        OutputParseError: If no generated_fasta artifact is present.
    """
    artifacts = _artifacts(result, ArtifactKind.GENERATED_FASTA)
    if not artifacts:
        raise OutputParseError("result declares no generated_fasta artifact")
    return read_fasta(out_dir / artifacts[0].path)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_io.py tests/test_models.py tests/test_contract.py -q` → PASS. Then `python -m pytest -q` green.

- [ ] **Step 5: Commit**

```bash
git add src/plms/io.py tests/test_io.py tests/test_models.py
git commit -m "io: neutralize likelihood columns and add read_generated"
```

---

### Task 3: models — GenerationResult + Model.generate

**Files:**
- Modify: `src/plms/models.py`, `src/plms/__init__.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `read_generated`, `stage_inputs`, `read_fasta`/`_read_records`, `Capability.GENERATE`, `RunSpec`.
- Produces: `GenerationResult.sequences() -> list[FastaRecord]`; `Model.generate(prompts_fasta, *, num_samples=1, temperature=1.0, top_p=1.0, max_length=None, seed=None, output_dir=None, use_gpu=False, batch_size=None) -> GenerationResult`.

- [ ] **Step 1: Write failing tests**

In `tests/test_models.py`, extend `FakeRunner` to simulate `generate` (add a `generate` branch to `_write_outputs` and a writer that reads the staged prompts and writes `num_samples` records per prompt):

```python
        elif capability == "generate":
            records = read_fasta(spec.input_dir / "seqs.fasta")
            num_samples = int(spec.command[spec.command.index("--num-samples") + 1])
            self._write_generate(out, records, num_samples)
```
```python
    def _write_generate(self, out: Path, records, num_samples: int) -> None:  # noqa: ANN001
        lines = []
        out_ids = []
        for rec in records:
            for k in range(num_samples):
                rid = f"{rec.id}__sample{k}"
                out_ids.append(rid)
                lines.append(f">{rid}\nACDEFG\n")
        (out / "generated.fasta").write_text("".join(lines))
        self._write_result(
            out,
            "generate",
            records,
            [{"path": "generated.fasta", "kind": "generated_fasta", "record_ids": out_ids}],
        )
```
Note: `FakeRunner._write_result` sets `n_output_records=len(records)`; for the generate tests assert on `sequences()` length rather than `n_output_records`. Add a prompts fixture and tests:

```python
@pytest.fixture
def prompts(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.fasta"
    path.write_text(">p1\nACDE\n>uncond\n\n")  # second record is unconditional (empty)
    return path


def test_generate_returns_sequences(prompts: Path, tmp_path: Path) -> None:
    from plms.models import GenerationResult

    model = _load(capabilities=["embed", "likelihood", "generate"])
    result = model.generate(prompts, num_samples=2, output_dir=tmp_path / "gen")
    assert isinstance(result, GenerationResult)
    seqs = result.sequences()
    assert {r.id for r in seqs} == {"p1__sample0", "p1__sample1", "uncond__sample0", "uncond__sample1"}


def test_generate_builds_expected_command(prompts: Path, tmp_path: Path) -> None:
    model = _load(capabilities=["generate"])
    model.generate(prompts, num_samples=3, temperature=0.8, top_p=0.9, seed=42, output_dir=tmp_path / "g")
    cmd = model._runner.last_spec.command  # type: ignore[attr-defined]
    assert cmd[0] == "generate"
    assert cmd[cmd.index("--num-samples") + 1] == "3"
    assert cmd[cmd.index("--temperature") + 1] == "0.8"
    assert cmd[cmd.index("--top-p") + 1] == "0.9"
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert "--max-length" not in cmd  # omitted when None


def test_generate_unsupported_capability_raises(prompts: Path, tmp_path: Path) -> None:
    model = _load(capabilities=["embed", "likelihood"])
    with pytest.raises(CapabilityNotSupportedError):
        model.generate(prompts, output_dir=tmp_path / "g")


def test_generate_empty_prompts_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    model = _load(capabilities=["generate"])
    with pytest.raises(InvalidRequestError):
        model.generate(empty, output_dir=tmp_path / "g")
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_models.py -q`
Expected: FAIL — `Model.generate`/`GenerationResult` missing.

- [ ] **Step 3: Implement in `src/plms/models.py`**

Add `read_generated` to the `plms.io` import block. Add the dataclass after `ScoreResult`:

```python
@dataclass
class GenerationResult:
    """Handle to the outputs of a ``generate`` run (FASTA parsed lazily)."""

    result: Result
    output_dir: Path
    _keepalive: tempfile.TemporaryDirectory | None = field(default=None, repr=False)

    def sequences(self) -> list[FastaRecord]:
        """Return the generated sequences (headers ``{prompt_id}__sample{k}``)."""
        return read_generated(self.output_dir, self.result)
```

Add the method after `score`:

```python
    def generate(
        self,
        prompts_fasta: str | Path,
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_length: int | None = None,
        seed: int | None = None,
        output_dir: Path | None = None,
        use_gpu: bool = False,
        batch_size: int | None = None,
    ) -> GenerationResult:
        """Sample sequences from an autoregressive model.

        Args:
            prompts_fasta: FASTA of prompt prefixes; an empty sequence means
                unconditional sampling. At least one record is required.
            num_samples: Samples to draw per prompt.
            temperature: Sampling temperature.
            top_p: Nucleus-sampling probability mass.
            max_length: Maximum sequence length (model default if ``None``).
            seed: Random seed for reproducible sampling.
            output_dir: Where to write outputs; a temporary directory if ``None``.
            use_gpu: Request all GPUs for the container run.
            batch_size: Override the model's default batch size.

        Raises:
            CapabilityNotSupportedError: If the model does not support generation.
            InvalidRequestError: If the prompts file contains no records.
            ContainerExecutionError: If the container run fails.
        """
        self._require_capability(Capability.GENERATE)
        records = self._read_records(prompts_fasta)
        extra = [
            "--num-samples", str(num_samples),
            "--temperature", str(temperature),
            "--top-p", str(top_p),
        ]
        if max_length is not None:
            extra += ["--max-length", str(max_length)]
        if seed is not None:
            extra += ["--seed", str(seed)]
        if batch_size is not None:
            extra += ["--batch-size", str(batch_size)]
        result, out_dir, keep = self._run(
            Capability.GENERATE, stage_inputs(records), extra, output_dir, use_gpu
        )
        return GenerationResult(result=result, output_dir=out_dir, _keepalive=keep)
```

`FastaRecord` is already imported under `TYPE_CHECKING`; if not, add `from plms.io import FastaRecord` to that block. In `src/plms/__init__.py`, add `GenerationResult` to the `plms.models` import and `__all__`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_models.py -q` → PASS; then `python -m pytest -q` green; `ty check src/` clean.

- [ ] **Step 5: Commit**

```bash
git add src/plms/models.py src/plms/__init__.py tests/test_models.py
git commit -m "models: add Model.generate and GenerationResult"
```

---

### Task 4: cli `generate` + registry progen2-small

**Files:**
- Modify: `src/plms/cli.py`, `src/plms/_data/models.yaml`, `tests/test_cli.py`, `tests/test_registry.py`

**Interfaces:**
- Produces: `plms generate ...` command; registry resolves `progen2-small`/`progen2_small` → `plms-progen2:small`.

- [ ] **Step 1: Write failing tests**

In `tests/test_registry.py` add:

```python
def test_resolve_progen2_small() -> None:
    registry = Registry.load()
    entry = registry.resolve("progen2-small")
    assert entry.image == "plms-progen2:small"
    assert entry.model_family == "progen2"
    assert registry.resolve("progen2_small") == entry
```

In `tests/test_cli.py` add a `generate` method to `FakeModel`, a `prompts` fixture, and a test:

```python
    def generate(self, prompts, *, num_samples, temperature, top_p, max_length, seed,
                 output_dir, use_gpu, batch_size):  # noqa: ANN001
        FakeModel.last_call = {"method": "generate", "num_samples": num_samples, "seed": seed}
        from plms.models import GenerationResult

        return GenerationResult(
            result=_result("generate", [{"path": "generated.fasta", "kind": "generated_fasta"}]),
            output_dir=Path(output_dir),
        )


@pytest.fixture
def prompts(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.fasta"
    path.write_text(">p1\nACDE\n")
    return path


def test_generate_command_invokes_model(prompts: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("plms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app, ["generate", "progen2-small", str(prompts), "-o", str(tmp_path / "out"),
              "--num-samples", "4", "--seed", "42"]
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["method"] == "generate"
    assert FakeModel.last_call["num_samples"] == 4
    assert FakeModel.last_call["seed"] == 42
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_cli.py tests/test_registry.py -q`
Expected: FAIL — no `generate` command; `progen2-small` not in registry.

- [ ] **Step 3: Implement**

Append to `src/plms/_data/models.yaml`:

```yaml
  - name: progen2-small
    aliases: [progen2_small]
    image: plms-progen2:small
    model_family: progen2
```

Add to `src/plms/cli.py` after the `score` command:

```python
@app.command()
def generate(
    model: _ModelArg,
    prompts: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True,
            help="Prompts FASTA (empty sequence = unconditional).",
        ),
    ],
    output_dir: _OutputOpt,
    num_samples: Annotated[int, typer.Option("--num-samples", help="Samples per prompt.")] = 1,
    temperature: Annotated[float, typer.Option("--temperature", help="Sampling temperature.")] = 1.0,
    top_p: Annotated[float, typer.Option("--top-p", help="Nucleus sampling probability.")] = 1.0,
    max_length: Annotated[int | None, typer.Option("--max-length", help="Max sequence length.")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed for reproducibility.")] = None,
    gpu: _GpuOpt = False,
    batch_size: _BatchOpt = None,
) -> None:
    """Generate sequences with an autoregressive model."""
    try:
        model_obj = load(model)
        result = model_obj.generate(
            prompts,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            seed=seed,
            output_dir=output_dir,
            use_gpu=gpu,
            batch_size=batch_size,
        )
        console.print(
            f"[green]generate[/green] complete: {result.result.n_output_records} sequence(s) "
            f"written to [bold]{output_dir}[/bold]"
        )
    except PlmsError as exc:
        _fail(exc)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_cli.py tests/test_registry.py -q` → PASS; `plms --help` lists `generate`; `python -m pytest -q` green.

- [ ] **Step 5: Commit**

```bash
git add src/plms/cli.py src/plms/_data/models.yaml tests/test_cli.py tests/test_registry.py
git commit -m "cli: add plms generate command and register progen2-small"
```

---

### Task 5: docs — CONTRACT.md for generate + neutral likelihood

**Files:**
- Modify: `docs/CONTRACT.md`

- [ ] **Step 1: Update the document**

Read `docs/CONTRACT.md` fully first, then:
- Version table / header: contract version `0.3`. `generate` is now implemented; with `score` already done, the only remaining reserved item is none (all four capabilities implemented) — update the reserved note accordingly.
- Section 2 (Internal CLI): promote `generate` to implemented with flags `--num-samples N --temperature T --top-p P --max-length L --seed S --batch-size N --device cpu|cuda`.
- Section 4 (I/O): add `generate` input (`prompts.fasta`; empty sequence = unconditional) and output (`generated.fasta`, headers `{prompt_id}__sample{k}`, clean amino-acid sequences; artifact kind `generated_fasta`). **Update the likelihood output schema** to the neutral columns `record_id,seq_len,log_likelihood,mean_log_likelihood,perplexity`, and document `result.json` `params.likelihood_method` ∈ {`masked_marginal`, `causal`}.
- Section 3 manifest example: keep at contract_version `0.3`; note that a model may declare `pooling_modes: []` when it does not support `embed`.
- Section 5: add `generated_fasta` to the OutputArtifact `kind` enumeration; add a generate worked example referencing `tests/data/result.generate.example.json`.

- [ ] **Step 2: Verify drift-guard passes**

Run: `python -m pytest tests/test_contract.py -q` → PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/CONTRACT.md
git commit -m "docs: document generate and the neutralized likelihood schema (v0.3)"
```

---

### Task 6: ESM2 — neutralize likelihood columns + contract 0.3

**Files:**
- Modify: `containers/esm2/entrypoint.py`

**Interfaces:**
- Produces: ESM2 `likelihood` emits `record_id,seq_len,log_likelihood,mean_log_likelihood,perplexity` and `result.json` `params.likelihood_method = "masked_marginal"`; manifest `contract_version "0.3"`.

- [ ] **Step 1: Update the entrypoint**

In `containers/esm2/entrypoint.py`:
1. Set the module constant `CONTRACT_VERSION = "0.3"`.
2. In `cmd_likelihood`, change the CSV header row from
   `"record_id,seq_len,pseudo_log_likelihood,mean_pseudo_log_likelihood,pseudo_perplexity"`
   to `"record_id,seq_len,log_likelihood,mean_log_likelihood,perplexity"` (the computed values are unchanged — only the column names).
3. In `_write_capability_result`, add the likelihood method to params for the likelihood capability:

```python
    params = {"device": args.device or "auto"}
    if capability == "embed":
        params |= {"pooling": args.pooling, "layers": args.layers}
    elif capability == "likelihood":
        params |= {"likelihood_method": "masked_marginal"}
```

- [ ] **Step 2: Verify (no torch needed for these checks)**

Run: `python -m pytest tests/test_esm2_entrypoint.py -q` → PASS (pure helpers unaffected). `ruff check containers/esm2/entrypoint.py` and `ruff format --check containers/esm2/entrypoint.py` → clean. (The renamed-column behavior is verified end-to-end when the image is rebuilt in Task 8.)

- [ ] **Step 3: Commit**

```bash
git add containers/esm2/entrypoint.py
git commit -m "esm2: neutralize likelihood columns; manifest contract 0.3"
```

---

### Task 7: ProGen2 container (build + smoke)

**Files:**
- Create: `containers/progen2/Dockerfile`, `containers/progen2/entrypoint.py`, `containers/progen2/README.md`, `tests/test_progen2_entrypoint.py`

**Interfaces:**
- Produces a contract-compliant image `plms-progen2:small` exposing `manifest`/`generate`/`likelihood`/`_prefetch`. Pure helper `resolve_hf_id(checkpoint) -> str` and `strip_special_tokens(text) -> str` are unit-testable without torch.

This task uses Docker. You will iterate against the real HF port until the build + smoke runs succeed — the ProGen2 tokenizer/`generate` API is the one model-specific unknown. Read the `hugohrban/progen2-small` model card/config to confirm the load + generate calls; adapt the concrete code below if the port's API differs, keeping the contract behavior identical.

- [ ] **Step 1: Pure-helper tests**

`tests/test_progen2_entrypoint.py` (loads the standalone module by path, like `tests/test_esm2_entrypoint.py`):

```python
"""Unit tests for the ProGen2 entrypoint's torch-free helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENTRYPOINT = Path(__file__).parents[1] / "containers" / "progen2" / "entrypoint.py"


def _load():
    spec = importlib.util.spec_from_file_location("progen2_entrypoint", _ENTRYPOINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entrypoint = _load()


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("progen2-small", "hugohrban/progen2-small"),
        ("progen2-base", "hugohrban/progen2-base"),
        ("hugohrban/progen2-small", "hugohrban/progen2-small"),
    ],
)
def test_resolve_hf_id(checkpoint: str, expected: str) -> None:
    assert entrypoint.resolve_hf_id(checkpoint) == expected


def test_strip_special_tokens_keeps_amino_acids() -> None:
    # control tokens (1/2), pad, and stray markers removed; AA letters kept
    assert entrypoint.strip_special_tokens("1MAGIC2") == "MAGIC"
    assert entrypoint.strip_special_tokens("<|pad|>ACDE<|pad|>") == "ACDE"
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_progen2_entrypoint.py -q`
Expected: FAIL — module file does not exist yet.

- [ ] **Step 3: Write `containers/progen2/entrypoint.py`**

```python
#!/usr/bin/env python
"""Contract-compliant entrypoint for the ProGen2 model image.

Implements the plms container contract (docs/CONTRACT.md) for the ProGen2
autoregressive protein language model via a HuggingFace community port loaded
with trust_remote_code. Exposes manifest / generate / likelihood / _prefetch.

torch and transformers are imported inside functions so the pure helpers stay
importable without the ML stack.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

CONTRACT_VERSION = "0.3"
MAX_SEQUENCE_LENGTH = 1024
DEFAULT_BATCH_SIZE = 8
DEFAULT_CHECKPOINT = os.environ.get("PROGEN2_CHECKPOINT", "progen2-small")

# Non-amino-acid characters to drop from decoded output (control tokens etc.).
_NON_AA = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def resolve_hf_id(checkpoint: str) -> str:
    """Resolve a short ProGen2 name to a HuggingFace port id."""
    if "/" in checkpoint:
        return checkpoint
    return f"hugohrban/{checkpoint}"


def strip_special_tokens(text: str) -> str:
    """Reduce a decoded ProGen2 string to canonical amino-acid letters."""
    return _NON_AA.sub("", text.upper())


def write_result(output_dir: Path, payload: dict) -> None:
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2))


def emit_error_and_exit(error_type: str, message: str, **details: str) -> None:
    error = {
        "contract_version": CONTRACT_VERSION,
        "error_type": error_type,
        "message": message,
        "details": {k: str(v) for k, v in details.items()},
    }
    print(json.dumps(error), file=sys.stderr)
    raise SystemExit(1)


def pick_device(requested: str | None) -> str:
    import torch

    available = "cuda" if torch.cuda.is_available() else "cpu"
    if requested in (None, "auto"):
        return available
    if requested == "cuda" and available != "cuda":
        emit_error_and_exit("DeviceUnavailable", "cuda requested but no GPU is available")
    return requested


def load_model(device: str):  # noqa: ANN201
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = resolve_hf_id(DEFAULT_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval().to(device)
    return tokenizer, model


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks).upper()))
            header = line[1:].split(maxsplit=1)[0] if line[1:].split() else line[1:]
            chunks = []
        elif line:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks).upper()))
    return records


def build_manifest() -> dict:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(resolve_hf_id(DEFAULT_CHECKPOINT), trust_remote_code=True)
    n_layer = getattr(config, "n_layer", None) or getattr(config, "num_hidden_layers", 0)
    n_embd = getattr(config, "n_embd", None) or getattr(config, "hidden_size", 0)
    return {
        "contract_version": CONTRACT_VERSION,
        "name": DEFAULT_CHECKPOINT,
        "version": "1.0.0",
        "description": f"ProGen2 autoregressive protein language model ({DEFAULT_CHECKPOINT}).",
        "model_family": "progen2",
        "capabilities": ["generate", "likelihood"],
        "embedding_dim": int(n_embd),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "pooling_modes": [],
        "num_layers": int(n_layer),
        "min_gpu_memory_gb": None,
        "default_batch_size": DEFAULT_BATCH_SIZE,
    }


def cmd_manifest(_args: argparse.Namespace) -> None:
    print(json.dumps(build_manifest()))


def cmd_generate(args: argparse.Namespace) -> None:
    import torch

    device = pick_device(args.device)
    tokenizer, model = load_model(device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    max_length = args.max_length or MAX_SEQUENCE_LENGTH
    records = read_fasta(Path(args.input))

    out_lines: list[str] = []
    out_ids: list[str] = []
    for rid, prefix in records:
        input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_length=max_length,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        for k in range(args.num_samples):
            seq = strip_special_tokens(tokenizer.decode(generated[k]))
            sample_id = f"{rid}__sample{k}"
            out_ids.append(sample_id)
            out_lines.append(f">{sample_id}\n{seq}\n")

    output_dir = Path(args.output)
    (output_dir / "generated.fasta").write_text("".join(out_lines))
    write_result(
        output_dir,
        {
            "contract_version": CONTRACT_VERSION,
            "capability": "generate",
            "model_name": DEFAULT_CHECKPOINT,
            "n_input_records": len(records),
            "n_output_records": len(out_ids),
            "artifacts": [
                {"path": "generated.fasta", "kind": "generated_fasta", "record_ids": out_ids}
            ],
            "warnings": [],
            "params": {
                "num_samples": str(args.num_samples),
                "temperature": str(args.temperature),
                "top_p": str(args.top_p),
                "max_length": str(max_length),
                "seed": str(args.seed),
            },
        },
    )


def cmd_likelihood(args: argparse.Namespace) -> None:
    import torch

    device = pick_device(args.device)
    tokenizer, model = load_model(device)
    records = read_fasta(Path(args.input))
    rows = ["record_id,seq_len,log_likelihood,mean_log_likelihood,perplexity"]
    warnings: list[str] = []
    for rid, seq in records:
        if len(seq) > MAX_SEQUENCE_LENGTH:
            warnings.append(f"sequence {rid!r} truncated to {MAX_SEQUENCE_LENGTH}")
            seq = seq[:MAX_SEQUENCE_LENGTH]
        ll = _causal_log_likelihood(tokenizer, model, seq, device)
        mean = ll / max(len(seq), 1)
        rows.append(f"{rid},{len(seq)},{ll:.6f},{mean:.6f},{__import__('math').exp(-mean):.6f}")
    output_dir = Path(args.output)
    (output_dir / "likelihoods.csv").write_text("\n".join(rows) + "\n")
    write_result(
        output_dir,
        {
            "contract_version": CONTRACT_VERSION,
            "capability": "likelihood",
            "model_name": DEFAULT_CHECKPOINT,
            "n_input_records": len(records),
            "n_output_records": len(records),
            "artifacts": [{"path": "likelihoods.csv", "kind": "likelihoods_csv"}],
            "warnings": warnings,
            "params": {"likelihood_method": "causal", "device": args.device or "auto"},
        },
    )


def _causal_log_likelihood(tokenizer, model, seq: str, device: str) -> float:  # noqa: ANN001
    import torch

    input_ids = tokenizer(seq, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    # predict token t+1 from position t
    targets = input_ids[0, 1:]
    token_lp = log_probs[0, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(token_lp.sum())


def cmd_prefetch(_args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = resolve_hf_id(DEFAULT_CHECKPOINT)
    AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(hf_id, trust_remote_code=True)
    print(f"prefetched {hf_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="progen2", description="ProGen2 plms contract entrypoint.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("manifest").set_defaults(func=cmd_manifest)

    gen = sub.add_parser("generate")
    gen.add_argument("--input", required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--num-samples", type=int, default=1, dest="num_samples")
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    gen.add_argument("--max-length", type=int, default=None, dest="max_length")
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    gen.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    gen.set_defaults(func=cmd_generate)

    lik = sub.add_parser("likelihood")
    lik.add_argument("--input", required=True)
    lik.add_argument("--output", required=True)
    lik.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    lik.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    lik.set_defaults(func=cmd_likelihood)

    sub.add_parser("_prefetch").set_defaults(func=cmd_prefetch)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - top-level: report as structured error
        emit_error_and_exit("InternalError", str(exc), exception=type(exc).__name__)


if __name__ == "__main__":
    main()
```

(If reading the port's card shows the tokenizer needs an explicit BOS for unconditional prompts, or a different decode call, adapt `cmd_generate`/`load_model` accordingly — keep `strip_special_tokens` as the output cleaner and the contract I/O identical. Replace the inline `__import__('math')` with a top-level `import math` if you prefer; either is fine for ruff.)

- [ ] **Step 4: Write the Dockerfile + README**

`containers/progen2/Dockerfile` (mirror `containers/esm2/Dockerfile`):

```dockerfile
# ProGen2 model image for the plms container contract.
#   docker build --build-arg PROGEN2_CHECKPOINT=progen2-small -t plms-progen2:small containers/progen2
# Weights are baked in at build time; runs CPU by default, GPU with --gpus.
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG PROGEN2_CHECKPOINT=progen2-small
ENV PROGEN2_CHECKPOINT=${PROGEN2_CHECKPOINT} \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "transformers==4.46.3"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

RUN python /app/entrypoint.py _prefetch

ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["python", "/app/entrypoint.py"]
```

`containers/progen2/README.md`: document the build commands, the `PROGEN2_CHECKPOINT` build arg, baked weights, the `trust_remote_code` requirement, and standalone `manifest` debugging (model the structure on `containers/esm2/README.md`).

- [ ] **Step 5: Run pure-helper tests + build + smoke**

```bash
python -m pytest tests/test_progen2_entrypoint.py -q          # pure helpers pass
ruff check containers/progen2/entrypoint.py tests/test_progen2_entrypoint.py
ruff format --check containers/progen2/entrypoint.py tests/test_progen2_entrypoint.py
docker build --build-arg PROGEN2_CHECKPOINT=progen2-small -t plms-progen2:small containers/progen2
docker run --rm plms-progen2:small manifest        # capabilities ["generate","likelihood"], contract 0.3
# smoke generate (write a tiny prompts dir first):
mkdir -p /tmp/pg/in /tmp/pg/out && printf '>p1\nMAGIC\n>uncond\n\n' > /tmp/pg/in/prompts.fasta
docker run --rm -v /tmp/pg/in:/in:ro -v /tmp/pg/out:/out:rw plms-progen2:small \
  generate --input /in/prompts.fasta --output /out --num-samples 2 --seed 1
cat /tmp/pg/out/generated.fasta    # 4 records, valid AA strings
```
Iterate the entrypoint until the build, `manifest`, and smoke `generate` all succeed and produce clean amino-acid sequences. If the HF port cannot be made to load/generate after reasonable effort, STOP and report BLOCKED with the error — the fallback is vendoring Salesforce ProGen2 code (a container-internal change).

- [ ] **Step 6: Commit**

```bash
git add containers/progen2/ tests/test_progen2_entrypoint.py
git commit -m "progen2: contract-compliant container (manifest/generate/likelihood)"
```

---

### Task 8: Integration — generate end-to-end + ESM2 likelihood update + full verification

**Files:**
- Create: `tests/data/prompts.fasta`, `tests/test_integration_progen2.py`
- Modify: `tests/test_integration_esm2.py`

**Interfaces:**
- Consumes: `plms.load("progen2-small")` and `plms.load("esm2-8m")` (both contract 0.3).

- [ ] **Step 1: Create prompts data**

`tests/data/prompts.fasta` (one short prefix + one unconditional empty record):

```
>prefix1
MAGIC
>uncond

```

- [ ] **Step 2: Write the ProGen2 integration test**

`tests/test_integration_progen2.py` (model the gating/fixtures on `tests/test_integration_esm2.py`):

```python
"""End-to-end integration test against a locally built ProGen2 image."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import plms

IMAGE = "plms-progen2:small"
REPO_ROOT = Path(__file__).parents[1]
PROMPTS = REPO_ROOT / "tests" / "data" / "prompts.fasta"
SEQS = REPO_ROOT / "tests" / "data" / "tiny.fasta"
_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PLMS_RUN_DOCKER_TESTS") != "1" or not _docker_available(),
        reason="set PLMS_RUN_DOCKER_TESTS=1 and ensure a Docker daemon is available",
    ),
]


@pytest.fixture(scope="session")
def progen2_image() -> str:
    present = (
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    )
    if not present:
        subprocess.run(
            ["docker", "build", "--build-arg", "PROGEN2_CHECKPOINT=progen2-small",
             "-t", IMAGE, str(REPO_ROOT / "containers" / "progen2")],
            check=True,
        )
    return IMAGE


@pytest.fixture(scope="session")
def model(progen2_image: str) -> plms.Model:
    return plms.load("progen2-small")


def test_manifest_declares_generate_and_likelihood(model: plms.Model) -> None:
    caps = {c.value for c in model.manifest.capabilities}
    assert {"generate", "likelihood"} <= caps
    assert model.manifest.pooling_modes == []


def test_generate_is_deterministic_with_seed(model: plms.Model, tmp_path: Path) -> None:
    first = model.generate(PROMPTS, num_samples=2, temperature=0.8, top_p=0.9, seed=7,
                           output_dir=tmp_path / "a")
    second = model.generate(PROMPTS, num_samples=2, temperature=0.8, top_p=0.9, seed=7,
                            output_dir=tmp_path / "b")
    a = {r.id: r.sequence for r in first.sequences()}
    b = {r.id: r.sequence for r in second.sequences()}
    assert a == b  # same seed => identical output


def test_generate_produces_valid_sequences(model: plms.Model, tmp_path: Path) -> None:
    result = model.generate(PROMPTS, num_samples=2, max_length=64, seed=1,
                            output_dir=tmp_path / "gen")
    seqs = result.sequences()
    assert {r.id for r in seqs} == {
        "prefix1__sample0", "prefix1__sample1", "uncond__sample0", "uncond__sample1"
    }
    for record in seqs:
        assert record.sequence  # non-empty
        assert set(record.sequence) <= _AA  # clean amino acids only
        assert len(record.sequence) <= 64


def test_progen2_likelihood(model: plms.Model, tmp_path: Path) -> None:
    rows = model.likelihood(SEQS, output_dir=tmp_path / "ll").rows()
    assert len(rows) == 3
    for row in rows:
        assert math.isfinite(float(row["log_likelihood"]))
        assert row["perplexity"] > 1.0
```

- [ ] **Step 3: Update the ESM2 integration test for neutral columns**

In `tests/test_integration_esm2.py`, the `test_likelihood_end_to_end` currently asserts on `pseudo_perplexity`. Change its assertions to the neutral column and check the method param:

```python
def test_likelihood_end_to_end(model: plms.Model, tmp_path: Path) -> None:
    result = model.likelihood(TINY_FASTA, output_dir=tmp_path / "ll")
    rows = {row["record_id"]: row for row in result.rows()}
    assert set(rows) == EXPECTED_IDS
    for row in rows.values():
        assert row["perplexity"] > 1.0
        assert math.isfinite(float(row["log_likelihood"]))
        assert row["seq_len"] > 0
    assert result.result.params["likelihood_method"] == "masked_marginal"
```

- [ ] **Step 4: Rebuild both images and run integration**

```bash
docker build --build-arg ESM2_CHECKPOINT=esm2_t6_8M -t plms-esm2:t6_8M containers/esm2
docker build --build-arg PROGEN2_CHECKPOINT=progen2-small -t plms-progen2:small containers/progen2
PLMS_RUN_DOCKER_TESTS=1 python -m pytest tests/test_integration_esm2.py tests/test_integration_progen2.py -v
```
Expected: all pass — ESM2 likelihood now reports neutral columns + `masked_marginal`; ProGen2 generation is deterministic under a fixed seed and yields clean AA sequences.

- [ ] **Step 5: Full verification gate**

```bash
ruff check src/ tests/ containers/
ruff format --check src/ tests/ containers/
ty check src/
python -m pytest -q                                   # unit green, integration skipped
PLMS_RUN_DOCKER_TESTS=1 python -m pytest -q -m slow    # all integration green
```

- [ ] **Step 6: Commit**

```bash
git add tests/data/prompts.fasta tests/test_integration_progen2.py tests/test_integration_esm2.py
git commit -m "test: end-to-end generate integration (ProGen2) + ESM2 neutral-likelihood update"
```

---

## Self-Review

**Spec coverage:**
- Contract 0.3 + generated_fasta → Task 1. ✓
- Neutralized likelihood columns + `likelihood_method` → client (Task 2), ESM2 (Task 6), docs (Task 5), integration (Task 8). ✓
- `generate` subcommand + flags → Task 7 (parser) + documented Task 5. ✓
- `read_generated`, `Model.generate`, `GenerationResult`, CLI, registry → Tasks 2, 3, 4. ✓
- ProGen2 container (HF port, trust_remote_code, baked weights, generate+likelihood) → Task 7. ✓
- Empty-prompt = unconditional; ≥1 record → Task 3 (`_read_records`) + Task 7 (`cmd_generate`) + Task 8 data. ✓
- Determinism (fixed seed) → Task 8. ✓
- ProGen2 manifest pooling_modes [] → Task 7 + asserted Task 8. ✓

**Placeholder scan:** Container generate/tokenizer code is concrete with an explicit "adapt to the real port API" instruction scoped to Task 7 (which has docker to verify) — not a deferred placeholder. No TBD/TODO elsewhere.

**Type consistency:** `Model.generate(...) -> GenerationResult`; `GenerationResult.sequences() -> list[FastaRecord]`; `read_generated(out_dir, result) -> list[FastaRecord]`; neutral likelihood keys identical across io reader, ESM2 writer, ProGen2 writer, and tests; `resolve_hf_id`/`strip_special_tokens` signatures match their tests.

## Risks
- **ProGen2 HF port** is the live unknown; Task 7 is structured as build-and-iterate against the real port with docker, with a documented vendoring fallback. Keep the contract I/O fixed regardless of port specifics.
- **Generation cost on CPU:** keep integration `max_length` small (≤64) so CPU runs stay fast.
- **Cross-task likelihood rename:** Task 2 updates io + its test consumers together (suite stays green); Task 6 changes the ESM2 writer; Task 8 rebuilds ESM2 and updates its integration assertion. Between Task 2 and Task 8, the ESM2 *image* still emits old column names but is not rebuilt/tested until Task 8, so no test observes the mismatch.
