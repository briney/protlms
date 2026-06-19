# `score` (variant effect scoring) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `score` capability that computes per-variant effect scores for ESM2, completing the masked-LM capability set.

**Architecture:** The scoring math runs inside the ESM2 container (client stays ML-free). The client validates the variants CSV, drives a `docker run`, and parses a scores CSV — exactly mirroring the existing `embed`/`likelihood` flow. Contract bumps 0.1 → 0.2 (minor; same major stays mutually readable).

**Tech Stack:** Python 3.11+, Pydantic v2, Typer, NumPy (client); HuggingFace `transformers` + PyTorch (container). Tooling: pytest, ruff, ty.

## Global Constraints

- Python 3.11+; modern syntax (`X | None`, `match`, `StrEnum`). `from __future__ import annotations` in every source module **except `cli.py`** (Typer reads runtime annotations there).
- Client carries **no ML dependencies** (no torch/transformers/pandas/biopython). NumPy only to load arrays; stdlib `csv` for CSV.
- Google-style docstrings on public functions/classes; full type hints; functions < ~50 lines.
- `ruff check`, `ruff format`, `ty check src/` must stay clean. Line length 100. Tests mirror `src/` layout under `tests/`.
- Contract schemas in `src/plms/contract.py` mirror `docs/CONTRACT.md` exactly; edit together. `CONTRACT_VERSION` is `MAJOR.MINOR`.
- Variant `mutant` notation: `{WT}{pos}{MUT}`, **1-indexed**, multi-mutants colon-separated (`A24G:T56S`). Self-substitution (`A24A`) scores exactly `0.0`. Multi-mutant score = additive sum of per-substitution masked/wt-marginal log-odds.
- Variants CSV columns: `variant_id, wt_sequence, mutant`. Scores CSV columns: `variant_id, mutant, n_mutations, score`. New artifact kind `variant_scores_csv`.
- Invalid variant rows (WT-residue mismatch / out-of-range position / malformed mutant) get a blank score and a `result.warnings` entry; the batch does NOT fail.

---

### Task 1: Contract — version bump + scores artifact kind + example fixtures

**Files:**
- Modify: `src/plms/contract.py`
- Modify: `tests/data/manifest.example.json`, `tests/data/result.embed.example.json`
- Create: `tests/data/result.score.example.json`
- Modify: `tests/test_contract.py`

**Interfaces:**
- Produces: `CONTRACT_VERSION == "0.2"`; `ArtifactKind.VARIANT_SCORES_CSV == "variant_scores_csv"`.

- [ ] **Step 1: Update the contract version + example fixtures, then write/adjust failing tests**

In `tests/data/manifest.example.json` set `"contract_version": "0.2"` and `"capabilities": ["embed", "likelihood", "score"]`.
In `tests/data/result.embed.example.json` set `"contract_version": "0.2"`.
Create `tests/data/result.score.example.json`:

```json
{
  "contract_version": "0.2",
  "capability": "score",
  "model_name": "esm2_t6_8M",
  "n_input_records": 3,
  "n_output_records": 3,
  "artifacts": [
    {"path": "scores.csv", "kind": "variant_scores_csv", "record_ids": ["self", "single", "double"]}
  ],
  "warnings": [],
  "params": {"method": "masked-marginal"}
}
```

In `tests/test_contract.py`: change `test_contract_version_is_semantic_string` to assert `CONTRACT_VERSION == "0.2"` and `parse_contract_version(CONTRACT_VERSION) == (0, 2)`. Update `test_documented_manifest_example_validates` to also assert `Capability.SCORE in manifest.capabilities`. Add:

```python
def test_documented_score_result_example_validates() -> None:
    result = Result.model_validate_json((_DATA / "result.score.example.json").read_text())
    assert result.capability is Capability.SCORE
    assert result.artifacts[0].kind == "variant_scores_csv"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_contract.py -q`
Expected: FAIL — `CONTRACT_VERSION` is still `"0.1"`; `variant_scores_csv` not yet a known kind (the score example still validates since `kind` is a free string, but the version + manifest-capability assertions fail).

- [ ] **Step 3: Implement the contract changes**

In `src/plms/contract.py`, set the version constant:

```python
CONTRACT_VERSION = "0.2"
```

Add to the `ArtifactKind` StrEnum:

```python
    VARIANT_SCORES_CSV = "variant_scores_csv"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_contract.py -q`
Expected: PASS. Then `python -m pytest -q` — all still green (existing model tests use a 0.1 manifest, which a 0.2 client still accepts: same major).

- [ ] **Step 5: Commit**

```bash
git add src/plms/contract.py tests/test_contract.py tests/data/
git commit -m "contract: bump to 0.2 and add variant_scores_csv artifact kind"
```

---

### Task 2: io — CSV staging, column validation, scores reader

**Files:**
- Modify: `src/plms/io.py`
- Modify: `tests/test_io.py`

**Interfaces:**
- Consumes: `StagedInput`, `_artifacts`, `Result`, `ArtifactKind` (existing in `io`/`contract`).
- Produces:
  - `stage_file(src: Path, dest_name: str) -> contextmanager[StagedInput]`
  - `check_csv_has_columns(path: Path, required: Iterable[str]) -> None` (raises `InvalidRequestError`)
  - `read_variant_scores(out_dir: Path, result: Result) -> list[dict[str, str | int | float | None]]`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_io.py` (import `stage_file`, `check_csv_has_columns`, `read_variant_scores` from `plms.io`, and `InvalidRequestError` from `plms.exceptions`):

```python
def test_stage_file_copies_input_under_dest_name() -> None:
    import tempfile
    from plms.io import stage_file

    src = Path(tempfile.mkdtemp()) / "v.csv"
    src.write_text("variant_id,wt_sequence,mutant\nv1,ACDE,A1G\n")
    with stage_file(src, "variants.csv") as job:
        staged = job.input_dir / "variants.csv"
        assert staged.is_file()
        assert job.container_input_path == "/in/variants.csv"
        held = job.input_dir
    assert not held.exists()


def test_check_csv_has_columns_ok(tmp_path: Path) -> None:
    from plms.io import check_csv_has_columns

    p = tmp_path / "v.csv"
    p.write_text("variant_id,wt_sequence,mutant\nv1,ACDE,A1G\n")
    check_csv_has_columns(p, ["variant_id", "wt_sequence", "mutant"])  # must not raise


def test_check_csv_has_columns_missing_raises(tmp_path: Path) -> None:
    from plms.exceptions import InvalidRequestError
    from plms.io import check_csv_has_columns

    p = tmp_path / "v.csv"
    p.write_text("variant_id,mutant\nv1,A1G\n")
    with pytest.raises(InvalidRequestError):
        check_csv_has_columns(p, ["variant_id", "wt_sequence", "mutant"])


def test_read_variant_scores_coerces_and_handles_blanks(tmp_path: Path) -> None:
    import json

    from plms.io import read_result, read_variant_scores

    (tmp_path / "scores.csv").write_text(
        "variant_id,mutant,n_mutations,score\nself,M1M,1,0.0\nbad,Z9Q,1,\n"
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "contract_version": "0.2",
                "capability": "score",
                "model_name": "m",
                "n_input_records": 2,
                "n_output_records": 2,
                "artifacts": [{"path": "scores.csv", "kind": "variant_scores_csv"}],
            }
        )
    )
    rows = read_variant_scores(tmp_path, read_result(tmp_path))
    assert rows[0]["variant_id"] == "self"
    assert rows[0]["n_mutations"] == 1
    assert rows[0]["score"] == 0.0
    assert rows[1]["score"] is None  # blank score for an invalid row
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_io.py -q`
Expected: FAIL — `ImportError` for `stage_file` / `check_csv_has_columns` / `read_variant_scores`.

- [ ] **Step 3: Implement in `src/plms/io.py`**

Add `import shutil` to the stdlib imports and `InvalidRequestError` to the exceptions import:

```python
from plms.exceptions import FastaError, InvalidRequestError, OutputParseError
```

Add `stage_file` after `stage_inputs`:

```python
@contextmanager
def stage_file(src: Path, dest_name: str) -> Iterator[StagedInput]:
    """Stage an arbitrary input file into a temporary directory bound at /in.

    Args:
        src: The host file to stage.
        dest_name: The filename it should have inside the input directory.

    Yields:
        A :class:`StagedInput` pointing at the host input directory.
    """
    with tempfile.TemporaryDirectory(prefix="plms-in-") as tmp:
        input_dir = Path(tmp)
        shutil.copyfile(src, input_dir / dest_name)
        yield StagedInput(input_dir=input_dir, input_filename=dest_name)


def check_csv_has_columns(path: Path, required: Iterable[str]) -> None:
    """Validate that a CSV file's header contains all required columns.

    Raises:
        InvalidRequestError: If any required column is absent.
    """
    with Path(path).open(newline="") as handle:
        header = next(csv.reader(handle), [])
    missing = [column for column in required if column not in header]
    if missing:
        raise InvalidRequestError(f"variants CSV {path} is missing column(s): {missing}")
```

Replace the body of `read_likelihoods` with a shared helper and add `read_variant_scores`. Add the score column types constant near `_LIKELIHOOD_COLUMN_TYPES`:

```python
_SCORE_COLUMN_TYPES: dict[str, type] = {"n_mutations": int, "score": float}


def _read_csv_artifact(
    out_dir: Path,
    result: Result,
    kind: ArtifactKind,
    column_types: dict[str, type],
) -> list[dict[str, str | int | float | None]]:
    """Read a single CSV artifact, coercing numeric columns (blanks -> None)."""
    artifacts = _artifacts(result, kind)
    if not artifacts:
        raise OutputParseError(f"result declares no {kind.value} artifact")
    rows: list[dict[str, str | int | float | None]] = []
    with (out_dir / artifacts[0].path).open(newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row: dict[str, str | int | float | None] = {}
            for key, value in raw_row.items():
                caster = column_types.get(key, str)
                if caster is not str and value == "":
                    row[key] = None
                else:
                    row[key] = caster(value)
            rows.append(row)
    return rows


def read_likelihoods(
    out_dir: Path, result: Result
) -> list[dict[str, str | int | float | None]]:
    """Read the likelihoods CSV, coercing known numeric columns."""
    return _read_csv_artifact(out_dir, result, ArtifactKind.LIKELIHOODS_CSV, _LIKELIHOOD_COLUMN_TYPES)


def read_variant_scores(
    out_dir: Path, result: Result
) -> list[dict[str, str | int | float | None]]:
    """Read the variant scores CSV (blank score => None for invalid rows)."""
    return _read_csv_artifact(out_dir, result, ArtifactKind.VARIANT_SCORES_CSV, _SCORE_COLUMN_TYPES)
```

(Delete the old `read_likelihoods` body that inlined the CSV loop.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_io.py -q`
Expected: PASS (including the existing `test_read_likelihoods_coerces_numeric_columns`, whose behavior is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/plms/io.py tests/test_io.py
git commit -m "io: add CSV staging, column validation, and variant-scores reader"
```

---

### Task 3: models — generalize the run helper, add Model.score + ScoreResult

**Files:**
- Modify: `src/plms/models.py`, `src/plms/__init__.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `stage_file`, `check_csv_has_columns`, `read_variant_scores`, `stage_inputs`, `read_fasta` (from `io`); `Capability.SCORE`, `RunSpec` (existing).
- Produces:
  - `ScoreResult` dataclass with `.rows() -> list[dict[str, str | int | float | None]]`.
  - `Model.score(variants_csv: str | Path, *, method: str = "masked-marginal", output_dir: Path | None = None, use_gpu: bool = False, batch_size: int | None = None) -> ScoreResult`.
  - Internal: `_run(self, capability, staging, extra_args, output_dir, use_gpu)` now takes a staging **context manager** instead of a FASTA path; `_read_records(self, fasta)` helper.

- [ ] **Step 1: Write the failing tests**

In `tests/test_models.py`, extend the `FakeRunner` to also simulate `score` (it already reads the staged input dir). Add a `_write_score` branch and the score result writer:

```python
    def _write_outputs(self, spec: RunSpec) -> None:
        out = spec.output_dir
        capability = spec.command[0]
        if capability == "embed":
            records = read_fasta(spec.input_dir / "seqs.fasta")
            pooling = spec.command[spec.command.index("--pooling") + 1]
            self._write_embed(out, records, pooling)
        elif capability == "likelihood":
            records = read_fasta(spec.input_dir / "seqs.fasta")
            self._write_likelihood(out, records)
        elif capability == "score":
            self._write_score(out, spec.input_dir / "variants.csv")

    def _write_score(self, out: Path, variants_csv: Path) -> None:
        import csv as _csv

        with variants_csv.open(newline="") as handle:
            rows = list(_csv.DictReader(handle))
        lines = ["variant_id,mutant,n_mutations,score"]
        for r in rows:
            n = len(r["mutant"].split(":"))
            lines.append(f"{r['variant_id']},{r['mutant']},{n},-1.5")
        (out / "scores.csv").write_text("\n".join(lines) + "\n")
        (out / "result.json").write_text(
            json.dumps(
                {
                    "contract_version": "0.2",
                    "capability": "score",
                    "model_name": "esm2_t6_8M",
                    "n_input_records": len(rows),
                    "n_output_records": len(rows),
                    "artifacts": [{"path": "scores.csv", "kind": "variant_scores_csv"}],
                }
            )
        )
```

Add a variants fixture and tests:

```python
@pytest.fixture
def variants_csv(tmp_path: Path) -> Path:
    path = tmp_path / "variants.csv"
    path.write_text(
        "variant_id,wt_sequence,mutant\nself,ACDEFGHIK,A1A\nsingle,ACDEFGHIK,C2A\n"
    )
    return path


def test_score_returns_rows(variants_csv: Path, tmp_path: Path) -> None:
    from plms.models import ScoreResult

    model = _load()
    result = model.score(variants_csv, output_dir=tmp_path / "sc")
    assert isinstance(result, ScoreResult)
    rows = {r["variant_id"]: r for r in result.rows()}
    assert set(rows) == {"self", "single"}
    assert rows["single"]["n_mutations"] == 1


def test_score_builds_expected_command(variants_csv: Path, tmp_path: Path) -> None:
    model = _load()
    model.score(variants_csv, method="wt-marginal", output_dir=tmp_path / "sc")
    cmd = model._runner.last_spec.command  # type: ignore[attr-defined]
    assert cmd[0] == "score"
    assert cmd[cmd.index("--input") + 1] == "/in/variants.csv"
    assert cmd[cmd.index("--method") + 1] == "wt-marginal"


def test_score_invalid_method_raises_before_run(variants_csv: Path, tmp_path: Path) -> None:
    model = _load()
    with pytest.raises(InvalidRequestError):
        model.score(variants_csv, method="bogus", output_dir=tmp_path / "sc")
    assert model._runner.last_spec is None  # type: ignore[attr-defined]


def test_score_missing_columns_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("variant_id,mutant\nv1,A1G\n")
    model = _load()
    with pytest.raises(InvalidRequestError):
        model.score(bad, output_dir=tmp_path / "sc")


def test_score_unsupported_capability_raises(variants_csv: Path, tmp_path: Path) -> None:
    model = _load(capabilities=["embed", "likelihood"])  # no score
    with pytest.raises(CapabilityNotSupportedError):
        model.score(variants_csv, output_dir=tmp_path / "sc")
```

Note: the default `_manifest_json()` in this file lists `["embed", "likelihood"]`. Update its default `capabilities` to `["embed", "likelihood", "score"]` so `_load()` supports score (the `test_likelihood_unsupported_capability_raises` and new `test_score_unsupported_capability_raises` tests pass an explicit `capabilities=` override).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_models.py -q`
Expected: FAIL — `Model` has no `score`; `ScoreResult` not importable.

- [ ] **Step 3: Implement in `src/plms/models.py`**

Update imports from `plms.io`:

```python
from plms.io import (
    check_csv_has_columns,
    load_per_residue_embeddings,
    load_pooled_embeddings,
    read_fasta,
    read_likelihoods,
    read_result,
    read_variant_scores,
    stage_file,
    stage_inputs,
)
```

Add the `ScoreResult` dataclass after `LikelihoodResult`:

```python
@dataclass
class ScoreResult:
    """Handle to the outputs of a ``score`` run (CSV parsed lazily)."""

    result: Result
    output_dir: Path
    method: str
    _keepalive: tempfile.TemporaryDirectory | None = field(default=None, repr=False)

    def rows(self) -> list[dict[str, str | int | float | None]]:
        """Return one row per variant: variant_id, mutant, n_mutations, score."""
        return read_variant_scores(self.output_dir, self.result)
```

Add a `_read_records` helper and refactor `_run` to take a staging context manager. Replace the current `_run` method and the input-reading portion of `embed`/`likelihood`:

```python
    def _read_records(self, fasta: str | Path) -> list:
        records = read_fasta(Path(fasta))
        if not records:
            raise InvalidRequestError(f"input FASTA {fasta} contains no records")
        too_long = [
            r.id for r in records if len(r.sequence) > self._manifest.max_sequence_length
        ]
        if too_long:
            logger.warning(
                "%d sequence(s) exceed max_sequence_length=%d and will be truncated by the "
                "container: %s",
                len(too_long),
                self._manifest.max_sequence_length,
                too_long[:5],
            )
        return records

    def _run(
        self,
        capability: Capability,
        staging,  # contextmanager[StagedInput]
        extra_args: list[str],
        output_dir: Path | None,
        use_gpu: bool,
    ) -> tuple[Result, Path, tempfile.TemporaryDirectory | None]:
        out_dir, keep = self._resolve_output_dir(output_dir)
        with staging as staged:
            command = [
                capability.value,
                "--input",
                staged.container_input_path,
                "--output",
                "/out",
                *extra_args,
            ]
            spec = RunSpec(
                image=self._entry.image,
                command=command,
                input_dir=staged.input_dir,
                output_dir=out_dir,
                use_gpu=use_gpu,
            )
            run_result = self._runner.run(spec)
        if run_result.exit_code != 0:
            self._raise_container_error(run_result)
        return read_result(out_dir), out_dir, keep
```

Update `embed` to use the helpers (replace its body's run call):

```python
        records = self._read_records(fasta)
        extra = ["--pooling", pooling, "--layers", ",".join(str(x) for x in layers)]
        if batch_size is not None:
            extra += ["--batch-size", str(batch_size)]
        result, out_dir, keep = self._run(
            Capability.EMBED, stage_inputs(records), extra, output_dir, use_gpu
        )
        return EmbeddingResult(result=result, output_dir=out_dir, pooling=pooling, _keepalive=keep)
```

Update `likelihood` similarly:

```python
        records = self._read_records(fasta)
        extra = ["--batch-size", str(batch_size)] if batch_size is not None else []
        result, out_dir, keep = self._run(
            Capability.LIKELIHOOD, stage_inputs(records), extra, output_dir, use_gpu
        )
        return LikelihoodResult(result=result, output_dir=out_dir, _keepalive=keep)
```

Add the `score` method after `likelihood`:

```python
    def score(
        self,
        variants_csv: str | Path,
        *,
        method: str = "masked-marginal",
        output_dir: Path | None = None,
        use_gpu: bool = False,
        batch_size: int | None = None,
    ) -> ScoreResult:
        """Score sequence variants for effect.

        Args:
            variants_csv: CSV with columns ``variant_id, wt_sequence, mutant``.
            method: ``"masked-marginal"`` (default) or ``"wt-marginal"``.
            output_dir: Where to write outputs; a temporary directory if ``None``.
            use_gpu: Request all GPUs for the container run.
            batch_size: Override the model's default batch size.

        Raises:
            CapabilityNotSupportedError: If the model does not support scoring.
            InvalidRequestError: If ``method`` is invalid or the CSV lacks columns.
            ContainerExecutionError: If the container run fails.
        """
        self._require_capability(Capability.SCORE)
        if method not in ("masked-marginal", "wt-marginal"):
            raise InvalidRequestError(
                f"unsupported scoring method {method!r}; "
                "choose 'masked-marginal' or 'wt-marginal'"
            )
        path = Path(variants_csv)
        check_csv_has_columns(path, ("variant_id", "wt_sequence", "mutant"))
        extra = ["--method", method]
        if batch_size is not None:
            extra += ["--batch-size", str(batch_size)]
        result, out_dir, keep = self._run(
            Capability.SCORE, stage_file(path, "variants.csv"), extra, output_dir, use_gpu
        )
        return ScoreResult(result=result, output_dir=out_dir, method=method, _keepalive=keep)
```

In `src/plms/__init__.py`, add `ScoreResult` to the import from `plms.models` and to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_models.py tests/test_io.py tests/test_contract.py -q`
Expected: PASS (the existing embed/likelihood tests still pass — behavior is unchanged by the `_run` refactor).

- [ ] **Step 5: Commit**

```bash
git add src/plms/models.py src/plms/__init__.py tests/test_models.py
git commit -m "models: add Model.score and ScoreResult; generalize run staging"
```

---

### Task 4: cli — `plms score`

**Files:**
- Modify: `src/plms/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load`, `ScoreResult` (via `model.score`).
- Produces: `plms score MODEL VARIANTS_CSV -o OUT [--method ...] [--gpu/--no-gpu] [--batch-size N]`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_cli.py`, add a `score` method to `FakeModel` and tests:

```python
    def score(self, variants, *, method, output_dir, use_gpu, batch_size):  # noqa: ANN001
        FakeModel.last_call = {"method": "score", "scoring_method": method, "use_gpu": use_gpu}
        from plms.models import ScoreResult

        return ScoreResult(
            result=_result("score", [{"path": "scores.csv", "kind": "variant_scores_csv"}]),
            output_dir=Path(output_dir),
            method=method,
        )


@pytest.fixture
def variants_csv(tmp_path: Path) -> Path:
    path = tmp_path / "variants.csv"
    path.write_text("variant_id,wt_sequence,mutant\nv1,ACDE,A1G\n")
    return path


def test_score_command_invokes_model(variants_csv: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("plms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app, ["score", "esm2-8m", str(variants_csv), "-o", str(tmp_path / "out"),
              "--method", "wt-marginal"]
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["method"] == "score"
    assert FakeModel.last_call["scoring_method"] == "wt-marginal"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -q`
Expected: FAIL — no `score` command registered (Typer exits non-zero / "No such command").

- [ ] **Step 3: Implement in `src/plms/cli.py`**

Add a `score` command after `likelihood`:

```python
@app.command()
def score(
    model: _ModelArg,
    variants: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Variants CSV.")
    ],
    output_dir: _OutputOpt,
    method: Annotated[
        str, typer.Option("--method", help="Scoring method: masked-marginal or wt-marginal.")
    ] = "masked-marginal",
    gpu: _GpuOpt = False,
    batch_size: _BatchOpt = None,
) -> None:
    """Score sequence variants for effect."""
    try:
        model_obj = load(model)
        result = model_obj.score(
            variants, method=method, output_dir=output_dir, use_gpu=gpu, batch_size=batch_size
        )
        console.print(
            f"[green]score[/green] complete: {result.result.n_output_records} variant(s) "
            f"written to [bold]{output_dir}[/bold]  method={method}"
        )
    except PlmsError as exc:
        _fail(exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -q && plms --help`
Expected: PASS; `plms --help` lists a `score` command.

- [ ] **Step 5: Commit**

```bash
git add src/plms/cli.py tests/test_cli.py
git commit -m "cli: add plms score command"
```

---

### Task 5: docs — update CONTRACT.md for `score`

**Files:**
- Modify: `docs/CONTRACT.md`

**Interfaces:** none (documentation; mirrors Task 1 schema + Task 6 behavior).

- [ ] **Step 1: Update the document**

- In the version table / section 1, change the contract version to `0.2` and note that `score` is now implemented (move it out of "reserved").
- In section 2 (Internal CLI), promote `score` from reserved to implemented with its flags:
  `score --input /in/variants.csv --output /out [--method masked-marginal|wt-marginal] [--batch-size N] [--device cpu|cuda]`.
- In section 4 (I/O), add the `variants.csv` input schema (`variant_id, wt_sequence, mutant`; 1-indexed `{WT}{pos}{MUT}`, colon-separated multi-mutants; self-substitution scores 0) and the `scores.csv` output schema (`variant_id, mutant, n_mutations, score`; artifact kind `variant_scores_csv`; invalid rows → blank score + `result.warnings`).
- Add a worked example referencing `tests/data/result.score.example.json`.
- Update the manifest worked-example JSON in section 3 to `"contract_version": "0.2"` and `capabilities` including `"score"` (matching `tests/data/manifest.example.json`).

- [ ] **Step 2: Verify the drift-guard still passes**

Run: `python -m pytest tests/test_contract.py -q`
Expected: PASS (the example fixtures from Task 1 validate; the doc now matches them).

- [ ] **Step 3: Commit**

```bash
git add docs/CONTRACT.md
git commit -m "docs: document the score capability in CONTRACT.md (v0.2)"
```

---

### Task 6: ESM2 container — implement `cmd_score`

**Files:**
- Modify: `containers/esm2/entrypoint.py`
- Modify: `tests/test_esm2_entrypoint.py`

**Interfaces:**
- Consumes: existing `load_model`, `pick_device`, `_truncate`, `write_result`, `build_parser`.
- Produces (pure, unit-testable): `parse_mutant(mutant: str) -> list[tuple[str, int, str]]`.
- Produces (container-only): `cmd_score(args)`; manifest now lists `"score"` and `contract_version == "0.2"`.

- [ ] **Step 1: Write the failing pure-helper tests**

Add to `tests/test_esm2_entrypoint.py`:

```python
def test_parse_mutant_single_and_multi() -> None:
    assert entrypoint.parse_mutant("A24G") == [("A", 24, "G")]
    assert entrypoint.parse_mutant("A24G:T56S") == [("A", 24, "G"), ("T", 56, "S")]


def test_parse_mutant_self_substitution() -> None:
    assert entrypoint.parse_mutant("M1M") == [("M", 1, "M")]


@pytest.mark.parametrize("bad", ["", "24G", "AG", "A2", "AxG", "A-1G"])
def test_parse_mutant_malformed_raises(bad: str) -> None:
    with pytest.raises(ValueError):
        entrypoint.parse_mutant(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_esm2_entrypoint.py -q`
Expected: FAIL — `entrypoint` has no `parse_mutant`.

- [ ] **Step 3: Implement in `containers/esm2/entrypoint.py`**

Add `import re` (top) and the constant + parser near the other pure helpers:

```python
_MUTANT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def parse_mutant(mutant: str) -> list[tuple[str, int, str]]:
    """Parse a mutation string like ``A24G`` or ``A24G:T56S`` (1-indexed)."""
    subs: list[tuple[str, int, str]] = []
    for token in mutant.split(":"):
        match = _MUTANT_RE.match(token.strip())
        if not match:
            raise ValueError(f"invalid mutation token {token!r}")
        subs.append((match.group(1).upper(), int(match.group(2)), match.group(3).upper()))
    return subs
```

Add the scoring helpers and `cmd_score` (torch imported inside, consistent with the rest of the file):

```python
def _masked_position_logprobs(tokenizer, model, seq, positions, batch_size, device):  # noqa: ANN001
    """Map each 1-indexed position to its masked log-softmax vector over the vocab."""
    import torch

    input_ids = tokenizer(seq, return_tensors="pt")["input_ids"][0]
    ordered = sorted(positions)
    out: dict[int, "object"] = {}
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        batch = input_ids.repeat(len(chunk), 1)
        for row, pos in enumerate(chunk):
            batch[row, pos] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(input_ids=batch.to(device)).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        for row, pos in enumerate(chunk):
            out[pos] = log_probs[row, pos].cpu().numpy()
    return out


def _wt_position_logprobs(tokenizer, model, seq, positions, device):  # noqa: ANN001
    """Map each 1-indexed position to its unmasked (WT-context) log-softmax vector."""
    import torch

    enc = tokenizer(seq, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)[0].cpu().numpy()
    return {pos: log_probs[pos] for pos in positions}


def _score_variant(mutant, wt_seq, logp_by_pos, tokenizer):  # noqa: ANN001
    """Return (score, n_mutations, error_message_or_None) for one variant."""
    try:
        subs = parse_mutant(mutant)
    except ValueError as exc:
        return None, 0, str(exc)
    total = 0.0
    for wt_aa, pos, mut_aa in subs:
        if not 1 <= pos <= len(wt_seq):
            return None, len(subs), f"position {pos} out of range for {mutant}"
        if wt_seq[pos - 1] != wt_aa:
            return None, len(subs), f"WT residue mismatch at {pos} in {mutant}"
        vec = logp_by_pos[pos]
        wt_id = tokenizer.convert_tokens_to_ids(wt_aa)
        mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
        total += float(vec[mut_id] - vec[wt_id])
    return total, len(subs), None


def cmd_score(args: argparse.Namespace) -> None:
    import csv as csv_module

    device = pick_device(args.device)
    tokenizer, model = load_model(device)
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE
    with Path(args.input).open(newline="") as handle:
        rows = list(csv_module.DictReader(handle))
    warnings: list[str] = []
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["wt_sequence"], []).append(row)

    out_rows = ["variant_id,mutant,n_mutations,score"]
    for wt_seq, group in groups.items():
        seq = _truncate(wt_seq, warnings, "<wt>")
        positions = _valid_positions(group, seq)
        if args.method == "wt-marginal":
            logp = _wt_position_logprobs(tokenizer, model, seq, positions, device)
        else:
            logp = _masked_position_logprobs(tokenizer, model, seq, positions, batch_size, device)
        for row in group:
            score, n_mut, err = _score_variant(row["mutant"], seq, logp, tokenizer)
            if err is not None:
                warnings.append(f"{row['variant_id']}: {err}")
            score_str = "" if score is None else f"{score:.6f}"
            out_rows.append(f"{row['variant_id']},{row['mutant']},{n_mut},{score_str}")

    output_dir = Path(args.output)
    (output_dir / "scores.csv").write_text("\n".join(out_rows) + "\n")
    artifacts = [
        {"path": "scores.csv", "kind": "variant_scores_csv", "record_ids": [r["variant_id"] for r in rows]}
    ]
    write_result(
        output_dir,
        {
            "contract_version": CONTRACT_VERSION,
            "capability": "score",
            "model_name": DEFAULT_CHECKPOINT,
            "n_input_records": len(rows),
            "n_output_records": len(rows),
            "artifacts": artifacts,
            "warnings": warnings,
            "params": {"method": args.method, "device": args.device or "auto"},
        },
    )


def _valid_positions(group: list[dict], wt_seq: str) -> set[int]:
    """Collect in-range mutated positions across a WT group (for logit computation)."""
    positions: set[int] = set()
    for row in group:
        try:
            for _wt, pos, _mut in parse_mutant(row["mutant"]):
                if 1 <= pos <= len(wt_seq):
                    positions.add(pos)
        except ValueError:
            continue
    return positions
```

Update `build_manifest` to add `"score"` to the `capabilities` list:

```python
        "capabilities": ["embed", "likelihood", "score"],
```

Register the `score` subcommand in `build_parser` (alongside `likelihood`):

```python
    score = sub.add_parser("score")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--method", default="masked-marginal", choices=["masked-marginal", "wt-marginal"])
    score.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    score.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    score.set_defaults(func=cmd_score)
```

(`CONTRACT_VERSION` in this file is already `"0.2"`-aligned via the constant at the top — confirm it reads `CONTRACT_VERSION = "0.2"`; bump it if it still says `"0.1"`.)

- [ ] **Step 4: Run pure-helper tests + lint the container**

Run: `python -m pytest tests/test_esm2_entrypoint.py -q && ruff check containers/esm2/entrypoint.py`
Expected: PASS; ruff clean. (Container scoring correctness is verified in Task 7.)

- [ ] **Step 5: Commit**

```bash
git add containers/esm2/entrypoint.py tests/test_esm2_entrypoint.py
git commit -m "esm2: implement score (masked-marginal + wt-marginal) at contract 0.2"
```

---

### Task 7: Integration — rebuild image, end-to-end score test, full verification

**Files:**
- Create: `tests/data/variants.csv`
- Modify: `tests/test_integration_esm2.py`

**Interfaces:**
- Consumes: `plms.load("esm2-8m")` (now contract 0.2 with `score`).

- [ ] **Step 1: Create the integration data**

`tests/data/variants.csv` (GB1 WT — the same 56-aa sequence used in `tiny.fasta`; includes a self-substitution and a multi-mutant):

```
variant_id,wt_sequence,mutant
self,MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE,M1M
single,MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE,T2A
double,MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE,M1L:T2A
```

- [ ] **Step 2: Write the integration test**

Add to `tests/test_integration_esm2.py`:

```python
VARIANTS_CSV = REPO_ROOT / "tests" / "data" / "variants.csv"


def test_score_masked_marginal_end_to_end(model: plms.Model, tmp_path: Path) -> None:
    result = model.score(VARIANTS_CSV, method="masked-marginal", output_dir=tmp_path / "sc")
    rows = {r["variant_id"]: r for r in result.rows()}
    assert set(rows) == {"self", "single", "double"}
    # a self-substitution must score exactly 0
    assert rows["self"]["score"] == pytest.approx(0.0, abs=1e-5)
    assert rows["self"]["n_mutations"] == 1
    assert rows["double"]["n_mutations"] == 2
    assert math.isfinite(float(rows["single"]["score"]))


def test_score_wt_marginal_runs(model: plms.Model, tmp_path: Path) -> None:
    result = model.score(VARIANTS_CSV, method="wt-marginal", output_dir=tmp_path / "sc")
    rows = {r["variant_id"]: r for r in result.rows()}
    assert rows["self"]["score"] == pytest.approx(0.0, abs=1e-5)
    assert math.isfinite(float(rows["double"]["score"]))


def test_manifest_now_declares_score(model: plms.Model) -> None:
    assert "score" in {c.value for c in model.manifest.capabilities}
```

- [ ] **Step 3: Rebuild the image and run the integration test**

Run:
```bash
docker build --build-arg ESM2_CHECKPOINT=esm2_t6_8M -t plms-esm2:t6_8M containers/esm2
docker run --rm plms-esm2:t6_8M manifest   # capabilities include "score", contract_version 0.2
PLMS_RUN_DOCKER_TESTS=1 python -m pytest tests/test_integration_esm2.py -v
```
Expected: image builds; manifest shows `score`; all integration tests PASS (self-substitution scores ≈ 0).

- [ ] **Step 4: Full verification gate**

Run:
```bash
ruff check src/ tests/ containers/
ruff format --check src/ tests/ containers/
ty check src/
python -m pytest -q                                   # unit suite green, integration skipped
PLMS_RUN_DOCKER_TESTS=1 python -m pytest -q -m slow    # integration green
```
Expected: all clean/green. Manually: `plms score esm2-8m tests/data/variants.csv -o out/` prints a summary and writes `out/scores.csv` owned by the host user.

- [ ] **Step 5: Commit**

```bash
git add tests/data/variants.csv tests/test_integration_esm2.py
git commit -m "test: end-to-end score integration on ESM2 (self-mutation scores 0)"
```

---

## Self-Review

**Spec coverage:**
- Contract 0.1→0.2 + `variant_scores_csv` → Task 1. ✓
- `score` subcommand + flags → Task 6 (`build_parser`) + documented Task 5. ✓
- Input `variants.csv` schema + validation → Task 2 (`check_csv_has_columns`) + Task 3 (`Model.score`). ✓
- Output `scores.csv` schema + reader + blank-score handling → Task 2 (`read_variant_scores`) + Task 6 (writer). ✓
- masked-marginal (default) + wt-marginal, WT grouping, per-row validation, additive multi-mutants → Task 6. ✓
- Client `Model.score`/`ScoreResult`/CLI/export → Tasks 3, 4. ✓
- CONTRACT.md + worked example → Tasks 1, 5. ✓
- Unit tests (contract/io/models/cli/entrypoint) → Tasks 1–4, 6. Integration (self-mutation ≈ 0, both methods) → Task 7. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `Model.score(...) -> ScoreResult`; `ScoreResult.rows() -> list[dict[str, str | int | float | None]]`; `read_variant_scores`/`_read_csv_artifact` share that return type; `parse_mutant -> list[tuple[str, int, str]]`; `_run(..., staging, ...)` signature consistently used by embed/likelihood/score. ✓

## Notes on risk
- `_run` refactor changes a private signature used by `embed`/`likelihood`; their existing tests guard against regression (Task 3 Step 4).
- The `tokenizer.convert_tokens_to_ids` lookup assumes single-character residue tokens (true for ESM2); validated end-to-end by the self-substitution = 0 check (Task 7).
- masked-marginal cost is one masked forward per unique mutated position per WT — fine for the tiny test; large DMS scans should prefer `--method wt-marginal` (1 forward/WT) until input chunking lands.
