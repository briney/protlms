# Phase 2 — GHCR Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `plms.load(name)` fetch model images from GHCR on demand, run them pinned to an immutable digest, and ship a manual CI workflow that publishes images and writes the digest back to the registry.

**Architecture:** All new runtime behavior lives behind the existing `Runner` seam: two new primitives (`image_present`, `pull`) plus a module-level `ensure_image` orchestrator that `load()` calls before reading the manifest. The registry gains a `digest` field (runs use `<repo>@<digest>`) and a producer-only `build` block. A `workflow_dispatch` GitHub Action builds/pushes one image and opens a PR pinning its digest.

**Tech Stack:** Python 3.11+, Pydantic v2, Typer, pytest, Docker CLI (subprocess), GitHub Actions, GHCR.

## Global Constraints

- Python 3.11+; modern syntax (`X | Y` unions, `match`, etc.). One line each below, verbatim from the spec:
- The client carries **no ML dependencies** — everything new is client-side orchestration (registry/runner/CLI) plus CI.
- Container images are **`linux/amd64` only** (GPU/CUDA); no multi-arch.
- GHCR packages are **public**; the client never manages `docker login`. Visibility is a documented one-time manual toggle, not set by the workflow.
- Runs are **digest-pinned** when a digest is present; the human-readable tag stays in `models.yaml` for docs.
- Contract schema is **unchanged** by this work.
- Before every commit: `ruff check src/ tests/`, `ruff format src/ tests/`, `ty check src/`, `pytest` must pass.
- Commit message format: `<component>: <what changed and why>`.

---

### Task 1: Registry — `digest` + `build` fields and `pinned_ref()`

**Files:**
- Modify: `src/plms/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: nothing (foundation task).
- Produces:
  - `BuildSpec(BaseModel)` with `context: str`, `args: dict[str, str] = {}`.
  - `ModelEntry` gains `digest: str | None = None` and `build: BuildSpec | None = None`.
  - `ModelEntry.pinned_ref() -> str` — returns `"<repo>@<digest>"` when `digest` is set (with the `:tag` stripped from the final path segment), else `image` verbatim.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_registry.py`:

```python
from plms.registry import BuildSpec, ModelEntry


def _entry(**overrides) -> ModelEntry:
    data = dict(name="m", image="ghcr.io/briney/plms-esm2:t6_8M", model_family="esm2")
    data.update(overrides)
    return ModelEntry(**data)


def test_pinned_ref_uses_digest_and_strips_tag() -> None:
    entry = _entry(digest="sha256:abc123")
    assert entry.pinned_ref() == "ghcr.io/briney/plms-esm2@sha256:abc123"


def test_pinned_ref_without_digest_returns_image() -> None:
    assert _entry().pinned_ref() == "ghcr.io/briney/plms-esm2:t6_8M"


def test_pinned_ref_preserves_registry_host_port() -> None:
    entry = _entry(image="host:5000/repo:tag", digest="sha256:deadbeef")
    assert entry.pinned_ref() == "host:5000/repo@sha256:deadbeef"


def test_invalid_digest_rejected() -> None:
    with pytest.raises(ValueError, match="sha256:"):
        _entry(digest="abc123")


def test_build_spec_parsed_from_entry() -> None:
    entry = _entry(build={"context": "containers/esm2", "args": {"ESM2_CHECKPOINT": "esm2_t6_8M"}})
    assert isinstance(entry.build, BuildSpec)
    assert entry.build.context == "containers/esm2"
    assert entry.build.args["ESM2_CHECKPOINT"] == "esm2_t6_8M"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_registry.py -k "pinned_ref or invalid_digest or build_spec" -v`
Expected: FAIL — `ImportError: cannot import name 'BuildSpec'` / `ModelEntry` has no `pinned_ref`.

- [ ] **Step 3: Implement the schema changes**

In `src/plms/registry.py`, change the import line `from pydantic import BaseModel` to:

```python
from pydantic import BaseModel, field_validator
```

Replace the `ModelEntry` class (currently lines 21-27) with:

```python
class BuildSpec(BaseModel):
    """Build-only metadata for the publishing pipeline; ignored by the client."""

    context: str
    args: dict[str, str] = {}


class ModelEntry(BaseModel):
    """One registry entry mapping a model to its image."""

    name: str
    aliases: list[str] = []
    image: str
    digest: str | None = None
    model_family: str
    build: BuildSpec | None = None

    @field_validator("digest")
    @classmethod
    def _validate_digest(cls, value: str | None) -> str | None:
        """Reject digests that are not ``sha256:`` references."""
        if value is not None and not value.startswith("sha256:"):
            raise ValueError(f"digest must start with 'sha256:', got {value!r}")
        return value

    def pinned_ref(self) -> str:
        """Image reference to pull/run.

        Returns ``<repo>@<digest>`` when a digest is set (reproducible), else the
        bare ``image`` tag (e.g. a locally-built image with no published digest).
        """
        if self.digest is None:
            return self.image
        return f"{self._strip_tag(self.image)}@{self.digest}"

    @staticmethod
    def _strip_tag(image: str) -> str:
        """Drop a ``:tag`` from the final path segment of an image reference."""
        prefix, sep, last = image.rpartition("/")
        name = last.split(":", 1)[0]
        return f"{prefix}{sep}{name}" if sep else name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (all, including pre-existing tests).

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff check src/plms/registry.py tests/test_registry.py && ruff format src/plms/registry.py tests/test_registry.py
ty check src/
git add src/plms/registry.py tests/test_registry.py
git commit -m "registry: add digest + build fields and pinned_ref()"
```

---

### Task 2: Runner — `ImagePullError`, `image_present`, `pull`

**Files:**
- Modify: `src/plms/exceptions.py`
- Modify: `src/plms/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ImagePullError(RunnerError)` in `plms.exceptions`.
  - `Runner` protocol methods `image_present(self, ref: str) -> bool` and `pull(self, ref: str) -> None`.
  - `SubprocessDockerRunner.image_present` (`docker image inspect`) and `.pull` (`docker pull`, raising `ImagePullError` on non-zero).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runner.py` (note: `ImagePullError` import added to the existing exceptions import line):

```python
from plms.exceptions import ImagePullError  # add alongside existing imports


def test_image_present_true_on_zero_exit(monkeypatch) -> None:
    def fake_run(argv, capture_output, text, check):  # noqa: ANN001
        assert argv == ["docker", "image", "inspect", "img@sha256:abc"]
        return subprocess.CompletedProcess(argv, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert SubprocessDockerRunner().image_present("img@sha256:abc") is True


def test_image_present_false_on_nonzero_exit(monkeypatch) -> None:
    def fake_run(argv, capture_output, text, check):  # noqa: ANN001
        return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="No such image")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert SubprocessDockerRunner().image_present("img@sha256:abc") is False


def test_pull_success_invokes_docker_pull(monkeypatch) -> None:
    captured = {}

    def fake_run(argv, capture_output, text, check):  # noqa: ANN001
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, returncode=0, stdout="pulled", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    SubprocessDockerRunner().pull("ghcr.io/briney/plms-esm2@sha256:abc")
    assert captured["argv"] == ["docker", "pull", "ghcr.io/briney/plms-esm2@sha256:abc"]


def test_pull_nonzero_raises_image_pull_error(monkeypatch) -> None:
    def fake_run(argv, capture_output, text, check):  # noqa: ANN001
        return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="network unreachable")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ImagePullError, match="network unreachable"):
        SubprocessDockerRunner().pull("img@sha256:abc")


def test_pull_auth_error_adds_login_hint(monkeypatch) -> None:
    def fake_run(argv, capture_output, text, check):  # noqa: ANN001
        return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="denied: access forbidden")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ImagePullError, match="docker login ghcr.io"):
        SubprocessDockerRunner().pull("img@sha256:abc")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runner.py -k "image_present or pull" -v`
Expected: FAIL — `ImportError: cannot import name 'ImagePullError'` / `SubprocessDockerRunner` has no `image_present`.

- [ ] **Step 3: Add the exception**

In `src/plms/exceptions.py`, after the `RunnerError` class (currently lines 34-35), add:

```python
class ImagePullError(RunnerError):
    """Raised when ``docker pull`` fails (network, auth, or unknown digest)."""
```

- [ ] **Step 4: Add the runner primitives**

In `src/plms/runner.py`, change the exceptions import (currently `from plms.exceptions import ImageNotFoundError, RunnerError`) to:

```python
from plms.exceptions import ImageNotFoundError, ImagePullError, RunnerError
```

Add two methods to the `Runner` protocol (after `def manifest(self, image: str) -> str: ...`):

```python
    def image_present(self, ref: str) -> bool: ...

    def pull(self, ref: str) -> None: ...
```

In `SubprocessDockerRunner`, add these methods (after `manifest`, before `_invoke`):

```python
    def image_present(self, ref: str) -> bool:
        """Return True if the image is available in the local Docker store."""
        completed = self._invoke([self._docker, "image", "inspect", ref])
        return completed.returncode == 0

    def pull(self, ref: str) -> None:
        """Pull an image from its registry.

        Raises:
            RunnerError: If the docker executable cannot be invoked.
            ImagePullError: If ``docker pull`` exits non-zero.
        """
        completed = self._invoke([self._docker, "pull", ref])
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            hint = ""
            if any(t in stderr.lower() for t in ("denied", "unauthorized", "authentication")):
                hint = " (authentication failed; try `docker login ghcr.io`)"
            raise ImagePullError(
                f"failed to pull image {ref!r} (exit {completed.returncode}){hint}. "
                f"stderr: {stderr[:500]}"
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_runner.py -v`
Expected: PASS.

- [ ] **Step 6: Lint, type-check, commit**

```bash
ruff check src/plms/runner.py src/plms/exceptions.py tests/test_runner.py && ruff format src/plms/runner.py src/plms/exceptions.py tests/test_runner.py
ty check src/
git add src/plms/runner.py src/plms/exceptions.py tests/test_runner.py
git commit -m "runner: add image_present/pull primitives and ImagePullError"
```

---

### Task 3: Runner — `ensure_image` orchestrator

**Files:**
- Modify: `src/plms/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `Runner.image_present`, `Runner.pull` (Task 2); `ImageNotFoundError`, `ImagePullError` (Tasks 2 / existing).
- Produces: module-level `ensure_image(runner: Runner, ref: str, *, allow_pull: bool, model_name: str) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runner.py`:

```python
from plms.exceptions import ImageNotFoundError  # add alongside existing imports
from plms.runner import ensure_image  # add alongside existing imports


class _RecordingRunner:
    """A minimal Runner stand-in that records pulls."""

    def __init__(self, *, present: bool, pull_error: Exception | None = None) -> None:
        self._present = present
        self._pull_error = pull_error
        self.pulled: list[str] = []

    def image_present(self, ref: str) -> bool:
        return self._present

    def pull(self, ref: str) -> None:
        self.pulled.append(ref)
        if self._pull_error is not None:
            raise self._pull_error


def test_ensure_image_noop_when_present() -> None:
    runner = _RecordingRunner(present=True)
    ensure_image(runner, "img@sha256:abc", allow_pull=True, model_name="m")
    assert runner.pulled == []


def test_ensure_image_pulls_when_absent_and_allowed() -> None:
    runner = _RecordingRunner(present=False)
    ensure_image(runner, "img@sha256:abc", allow_pull=True, model_name="m")
    assert runner.pulled == ["img@sha256:abc"]


def test_ensure_image_raises_when_absent_and_pull_disabled() -> None:
    runner = _RecordingRunner(present=False)
    with pytest.raises(ImageNotFoundError, match="plms pull m"):
        ensure_image(runner, "img@sha256:abc", allow_pull=False, model_name="m")


def test_ensure_image_propagates_pull_error() -> None:
    runner = _RecordingRunner(present=False, pull_error=ImagePullError("boom"))
    with pytest.raises(ImagePullError, match="boom"):
        ensure_image(runner, "img@sha256:abc", allow_pull=True, model_name="m")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runner.py -k ensure_image -v`
Expected: FAIL — `ImportError: cannot import name 'ensure_image'`.

- [ ] **Step 3: Implement `ensure_image`**

In `src/plms/runner.py`, add this module-level function after the `Runner` protocol class (before `_current_user`):

```python
def ensure_image(runner: Runner, ref: str, *, allow_pull: bool, model_name: str) -> None:
    """Ensure an image is present locally, pulling it when permitted.

    Args:
        runner: The container runner.
        ref: The image reference to ensure (typically a digest-pinned ref).
        allow_pull: Whether to pull when the image is absent.
        model_name: The model name, used only for error messages.

    Raises:
        ImageNotFoundError: If the image is absent and ``allow_pull`` is False.
        ImagePullError: If a pull is attempted and fails.
    """
    if runner.image_present(ref):
        return
    if not allow_pull:
        raise ImageNotFoundError(
            f"image {ref!r} for model {model_name!r} is not available locally and "
            f"pulling is disabled. Run `plms pull {model_name}` or unset PLMS_NO_PULL."
        )
    logger.info("pulling image %s for model %s", ref, model_name)
    runner.pull(ref)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff check src/plms/runner.py tests/test_runner.py && ruff format src/plms/runner.py tests/test_runner.py
ty check src/
git add src/plms/runner.py tests/test_runner.py
git commit -m "runner: add ensure_image orchestrator with pull escape hatch"
```

---

### Task 4: `load()` auto-pull + pinned-ref runs

**Files:**
- Modify: `src/plms/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `entry.pinned_ref()` (Task 1); `ensure_image` (Task 3).
- Produces:
  - `load(name, *, runner=None, registry=None, allow_pull: bool | None = None) -> Model`.
  - `Model.__init__(self, entry, runner, manifest, image_ref: str)` storing `self._image_ref`; runs use `self._image_ref`.
  - Env resolution: `allow_pull=None` → `False` iff `PLMS_NO_PULL` is truthy (`1`/`true`/`yes`), else `True`.

- [ ] **Step 1: Update the test `FakeRunner` and write failing tests**

In `tests/test_models.py`, add `present`/recording support to `FakeRunner.__init__` and two methods. Change the `__init__` signature/body (currently starts at line 46) to:

```python
    def __init__(self, manifest_json: str, *, behavior: str = "success", present: bool = True) -> None:
        self.manifest_json = manifest_json
        self.behavior = behavior
        self.present = present
        self.last_spec: RunSpec | None = None
        self.manifest_ref: str | None = None
        self.pulled: list[str] = []
```

Change `manifest` to record the ref:

```python
    def manifest(self, image: str) -> str:
        self.manifest_ref = image
        return self.manifest_json
```

Add after `manifest`:

```python
    def image_present(self, ref: str) -> bool:
        return self.present

    def pull(self, ref: str) -> None:
        self.pulled.append(ref)
```

Then append these tests to `tests/test_models.py`:

```python
from plms.exceptions import ImageNotFoundError  # add alongside existing imports
from plms.registry import ModelEntry, Registry  # add alongside existing imports


def _registry_with_digest() -> Registry:
    return Registry(
        [
            ModelEntry(
                name="esm2-8m",
                image="ghcr.io/briney/plms-esm2:t6_8M",
                digest="sha256:abc123",
                model_family="esm2",
            )
        ]
    )


def test_load_skips_pull_when_image_present() -> None:
    runner = FakeRunner(_manifest_json(), present=True)
    load("esm2-8m", runner=runner)
    assert runner.pulled == []


def test_load_pulls_pinned_ref_when_absent() -> None:
    runner = FakeRunner(_manifest_json(), present=False)
    load("esm2-8m", runner=runner, registry=_registry_with_digest())
    assert runner.pulled == ["ghcr.io/briney/plms-esm2@sha256:abc123"]


def test_load_runs_manifest_against_pinned_ref() -> None:
    runner = FakeRunner(_manifest_json(), present=True)
    load("esm2-8m", runner=runner, registry=_registry_with_digest())
    assert runner.manifest_ref == "ghcr.io/briney/plms-esm2@sha256:abc123"


def test_load_allow_pull_false_raises_when_absent() -> None:
    runner = FakeRunner(_manifest_json(), present=False)
    with pytest.raises(ImageNotFoundError):
        load("esm2-8m", runner=runner, allow_pull=False)


def test_load_env_no_pull_disables_pull(monkeypatch) -> None:
    monkeypatch.setenv("PLMS_NO_PULL", "1")
    runner = FakeRunner(_manifest_json(), present=False)
    with pytest.raises(ImageNotFoundError):
        load("esm2-8m", runner=runner)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -k "pull or pinned or no_pull" -v`
Expected: FAIL — `load()` has no `allow_pull` kwarg / `Model.__init__` got unexpected `image_ref`.

- [ ] **Step 3: Implement the `load`/`Model` changes**

In `src/plms/models.py`:

Add `import os` to the stdlib imports (after `import logging`).

Change the runner import (currently `from plms.runner import Runner, RunSpec, SubprocessDockerRunner`) to:

```python
from plms.runner import Runner, RunSpec, SubprocessDockerRunner, ensure_image
```

Change `Model.__init__` (lines 118-121) to:

```python
    def __init__(
        self, entry: ModelEntry, runner: Runner, manifest: Manifest, image_ref: str
    ) -> None:
        self._entry = entry
        self._runner = runner
        self._manifest = manifest
        self._image_ref = image_ref
```

Change the `RunSpec` image argument (line 369) from `image=self._entry.image,` to:

```python
                image=self._image_ref,
```

Add this helper just above `def load(` (line 447):

```python
def _resolve_allow_pull(allow_pull: bool | None) -> bool:
    """Resolve pull policy: explicit arg wins, else consult ``PLMS_NO_PULL``."""
    if allow_pull is not None:
        return allow_pull
    no_pull = os.environ.get("PLMS_NO_PULL", "").strip().lower() in {"1", "true", "yes"}
    return not no_pull
```

Replace the `load` function body (lines 447-473) with:

```python
def load(
    name: str,
    *,
    runner: Runner | None = None,
    registry: Registry | None = None,
    allow_pull: bool | None = None,
) -> Model:
    """Resolve a model name and return a ready-to-use :class:`Model`.

    Resolves the name against the registry, ensures the pinned image is present
    locally (pulling it when permitted), reads the image's manifest, checks
    contract compatibility, and constructs the model.

    Args:
        name: A model name or alias known to the registry.
        runner: The container runner (defaults to a local docker subprocess runner).
        registry: The model registry (defaults to the packaged registry).
        allow_pull: Whether to pull a missing image. ``None`` (default) consults
            the ``PLMS_NO_PULL`` environment variable.

    Raises:
        ModelNotFoundError: If the name is unknown.
        ImageNotFoundError: If the image is absent and pulling is disabled.
        ImagePullError: If the image must be pulled and the pull fails.
        ContractVersionError: If the image's contract major version mismatches.
    """
    runner = runner or SubprocessDockerRunner()
    registry = registry or Registry.load()
    entry = registry.resolve(name)
    ref = entry.pinned_ref()
    ensure_image(runner, ref, allow_pull=_resolve_allow_pull(allow_pull), model_name=name)
    manifest = Manifest.model_validate_json(runner.manifest(ref))
    check_contract_compatibility(manifest.contract_version)
    return Model(entry, runner, manifest, image_ref=ref)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: PASS (all, including pre-existing).

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff check src/plms/models.py tests/test_models.py && ruff format src/plms/models.py tests/test_models.py
ty check src/
git add src/plms/models.py tests/test_models.py
git commit -m "models: auto-pull pinned image on load with PLMS_NO_PULL escape hatch"
```

---

### Task 5: CLI — `plms pull` and `--no-pull`

**Files:**
- Modify: `src/plms/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ensure_image` (Task 3), `SubprocessDockerRunner` (existing), `Registry` (existing), `load(allow_pull=...)` (Task 4).
- Produces:
  - `plms pull <model>` and `plms pull --all`.
  - `--no-pull` option on `embed`, `likelihood`, `score`, `generate`, threading `allow_pull=False` into `load()` (the four commands that call `load`; spec named embed/likelihood/generate — score included for consistency).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
from plms.registry import Registry  # add alongside existing imports


def test_pull_command_pulls_resolved_model(monkeypatch) -> None:
    calls: list[tuple[str, bool, str]] = []
    monkeypatch.setattr("plms.cli.SubprocessDockerRunner", lambda: object())
    monkeypatch.setattr(
        "plms.cli.ensure_image",
        lambda runner, ref, *, allow_pull, model_name: calls.append((ref, allow_pull, model_name)),
    )
    result = runner.invoke(app, ["pull", "esm2-8m"])
    assert result.exit_code == 0, result.output
    assert calls and calls[0][1] is True and calls[0][2] == "esm2-8m"


def test_pull_all_pulls_every_model(monkeypatch) -> None:
    pulled: list[str] = []
    monkeypatch.setattr("plms.cli.SubprocessDockerRunner", lambda: object())
    monkeypatch.setattr(
        "plms.cli.ensure_image",
        lambda runner, ref, *, allow_pull, model_name: pulled.append(model_name),
    )
    result = runner.invoke(app, ["pull", "--all"])
    assert result.exit_code == 0, result.output
    assert len(pulled) == len(Registry.load().list_models())


def test_pull_without_model_or_all_errors() -> None:
    result = runner.invoke(app, ["pull"])
    assert result.exit_code == 1


def test_embed_no_pull_threads_allow_pull_false(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_load(name, **kw):  # noqa: ANN001, ANN003
        captured.update(kw)
        return FakeModel()

    monkeypatch.setattr("plms.cli.load", fake_load)
    result = runner.invoke(
        app, ["embed", "esm2-8m", str(fasta), "-o", str(tmp_path / "out"), "--no-pull"]
    )
    assert result.exit_code == 0, result.output
    assert captured.get("allow_pull") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -k "pull or no_pull" -v`
Expected: FAIL — no `pull` command / `embed` rejects `--no-pull`.

- [ ] **Step 3: Implement the CLI changes**

In `src/plms/cli.py`:

Change the imports. Replace `from plms.exceptions import InvalidRequestError, PlmsError` with:

```python
from plms.exceptions import InvalidRequestError, PlmsError
from plms.runner import SubprocessDockerRunner, ensure_image
```

Add a reusable option after `_ChunkSizeOpt` (line 44):

```python
_NoPullOpt = Annotated[
    bool,
    typer.Option("--no-pull", help="Do not pull the image if it is missing locally."),
]
```

Add the `pull` command after `models_list` (after line 70):

```python
@app.command()
def pull(
    model: Annotated[str | None, typer.Argument(help="Model name or alias.")] = None,
    all_models: Annotated[bool, typer.Option("--all", help="Pull every registered model.")] = False,
) -> None:
    """Pull a model's container image from its registry (digest-pinned when set)."""
    registry = Registry.load()
    try:
        if all_models:
            entries = registry.list_models()
        elif model is not None:
            entries = [registry.resolve(model)]
        else:
            raise InvalidRequestError("provide a model name or --all")
        docker_runner = SubprocessDockerRunner()
        for entry in entries:
            ref = entry.pinned_ref()
            console.print(f"pulling [bold]{entry.name}[/bold] ({ref}) …")
            ensure_image(docker_runner, ref, allow_pull=True, model_name=entry.name)
            console.print(f"  [green]ok[/green] {entry.name}")
    except PlmsError as exc:
        _fail(exc)
```

Add `no_pull: _NoPullOpt = False,` to the signatures of `embed`, `likelihood`, `score`, and `generate` (as the last parameter), and change each `model_obj = load(model)` call to:

```python
        model_obj = load(model, allow_pull=False if no_pull else None)
```

(There are four such call sites — one per command.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff check src/plms/cli.py tests/test_cli.py && ruff format src/plms/cli.py tests/test_cli.py
ty check src/
git add src/plms/cli.py tests/test_cli.py
git commit -m "cli: add plms pull command and --no-pull flag"
```

---

### Task 6: Migrate `models.yaml` to GHCR refs + `build` blocks

**Files:**
- Modify: `src/plms/_data/models.yaml`
- Modify: `tests/test_registry.py`
- Modify: `tests/test_integration_esm2.py`, `tests/test_integration_esmc.py`, `tests/test_integration_progen2.py`

**Interfaces:**
- Consumes: `BuildSpec`/`digest` schema (Task 1); `load(allow_pull=...)` (Task 4).
- Produces: registry entries pointing at `ghcr.io/briney/…` images with `build` blocks (digests added later by CI).

- [ ] **Step 1: Update the failing registry assertion**

In `tests/test_registry.py`, change `test_default_registry_resolves_esm2_8m`'s image assertion to:

```python
    assert entry.image == "ghcr.io/briney/plms-esm2:t6_8M"
```

Run: `pytest tests/test_registry.py::test_default_registry_resolves_esm2_8m -v`
Expected: FAIL (still `plms-esm2:t6_8M`).

- [ ] **Step 2: Rewrite `models.yaml`**

Replace the `models:` block in `src/plms/_data/models.yaml` with (digests are intentionally absent until CI publishes; `build` is producer-only metadata ignored by the client):

```yaml
models:
  - name: esm2-8m
    aliases: [esm2_t6_8M]
    image: ghcr.io/briney/plms-esm2:t6_8M
    model_family: esm2
    build:
      context: containers/esm2
      args: { ESM2_CHECKPOINT: esm2_t6_8M }
  - name: esm2-650m
    aliases: [esm2_t33_650M]
    image: ghcr.io/briney/plms-esm2:t33_650M
    model_family: esm2
    build:
      context: containers/esm2
      args: { ESM2_CHECKPOINT: esm2_t33_650M }
  - name: progen2-small
    aliases: [progen2_small]
    image: ghcr.io/briney/plms-progen2:small
    model_family: progen2
    build:
      context: containers/progen2
      args: { PROGEN2_CHECKPOINT: progen2-small }
  - name: esm-c-300m
    aliases: [esmc_300m]
    image: ghcr.io/briney/plms-esm-c:300m
    model_family: esm-c
    build:
      context: containers/esm-c
      args: { ESMC_CHECKPOINT: esmc_300m }
  - name: esm-c-600m
    aliases: [esmc_600m]
    image: ghcr.io/briney/plms-esm-c:600m
    model_family: esm-c
    build:
      context: containers/esm-c
      args: { ESMC_CHECKPOINT: esmc_600m }
```

Also update the header comment's "Phase 0 images are built locally" note to reflect GHCR publishing (one line): `# Images are published to GHCR; digests are pinned by .github/workflows/publish-image.yaml.`

- [ ] **Step 3: Keep integration tests on the local build (no GHCR pull)**

These tests build the image locally; they must pin to the GHCR tag and pass `allow_pull=False` so `load` uses the local build rather than pulling. Apply per file:

`tests/test_integration_esm2.py`:
```python
IMAGE = "ghcr.io/briney/plms-esm2:t6_8M"
```
```python
    return plms.load("esm2-8m", allow_pull=False)
```

`tests/test_integration_esmc.py`:
```python
IMAGE = "ghcr.io/briney/plms-esm-c:300m"
```
```python
    return plms.load("esm-c-300m", allow_pull=False)
```

`tests/test_integration_progen2.py`:
```python
IMAGE = "ghcr.io/briney/plms-progen2:small"
```
```python
    return plms.load("progen2-small", allow_pull=False)
```

- [ ] **Step 4: Run the fast suite to verify it passes**

Run: `pytest -m "not slow" -v`
Expected: PASS. (The integration tests are `slow`+docker-gated and won't run here; the registry test now passes.)

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff check tests/ && ruff format tests/
ty check src/
git add src/plms/_data/models.yaml tests/test_registry.py tests/test_integration_esm2.py tests/test_integration_esmc.py tests/test_integration_progen2.py
git commit -m "registry: point models.yaml at GHCR images with build metadata"
```

---

### Task 7: Publishing helper script (`scripts/registry_publish.py`)

**Files:**
- Create: `scripts/registry_publish.py`
- Create: `tests/test_registry_publish.py`

**Interfaces:**
- Consumes: `models.yaml` structure (Task 6).
- Produces (pure, importable functions; CLI via `python -m`):
  - `lookup_build(models_yaml: Path, name: str) -> tuple[str, str, dict[str, str]]` → `(image, context, args)`.
  - `set_digest(models_yaml: Path, name: str, digest: str) -> None` (rewrites the file, preserving the `name`→`digest` mapping).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_registry_publish.py`:

```python
"""Tests for the GHCR publishing helper script."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.registry_publish import lookup_build, set_digest

SAMPLE = """\
models:
  - name: esm2-8m
    aliases: [esm2_t6_8M]
    image: ghcr.io/briney/plms-esm2:t6_8M
    model_family: esm2
    build:
      context: containers/esm2
      args: { ESM2_CHECKPOINT: esm2_t6_8M }
"""


def _yaml(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(SAMPLE)
    return path


def test_lookup_build_returns_image_context_args(tmp_path: Path) -> None:
    image, context, args = lookup_build(_yaml(tmp_path), "esm2-8m")
    assert image == "ghcr.io/briney/plms-esm2:t6_8M"
    assert context == "containers/esm2"
    assert args == {"ESM2_CHECKPOINT": "esm2_t6_8M"}


def test_lookup_build_unknown_name_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        lookup_build(_yaml(tmp_path), "nope")


def test_lookup_build_missing_build_block_raises(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("models:\n  - name: x\n    image: i\n    model_family: f\n")
    with pytest.raises(ValueError, match="no build metadata"):
        lookup_build(path, "x")


def test_set_digest_writes_and_roundtrips(tmp_path: Path) -> None:
    import yaml

    path = _yaml(tmp_path)
    set_digest(path, "esm2-8m", "sha256:abc123")
    data = yaml.safe_load(path.read_text())
    entry = next(m for m in data["models"] if m["name"] == "esm2-8m")
    assert entry["digest"] == "sha256:abc123"


def test_set_digest_rejects_bad_digest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sha256:"):
        set_digest(_yaml(tmp_path), "esm2-8m", "abc123")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_registry_publish.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.registry_publish'`.

- [ ] **Step 3: Implement the script**

Create `scripts/__init__.py` (empty file, so `scripts` is importable in tests):

```python
```

Create `scripts/registry_publish.py`:

```python
"""Helpers for the GHCR publishing workflow.

Reads build metadata from the packaged ``models.yaml`` and writes published image
digests back into it. Kept dependency-light (PyYAML only) so CI can run it.

Usage:
    python -m scripts.registry_publish lookup <models.yaml> <name>
    python -m scripts.registry_publish set-digest <models.yaml> <name> <sha256:...>
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def _load(models_yaml: Path) -> dict:
    """Parse a models.yaml file into a dict."""
    return yaml.safe_load(Path(models_yaml).read_text()) or {}


def _find(data: dict, name: str) -> dict:
    """Return the entry whose ``name`` matches, or raise KeyError."""
    for entry in data.get("models", []):
        if entry.get("name") == name:
            return entry
    raise KeyError(f"no model named {name!r} in registry")


def lookup_build(models_yaml: Path, name: str) -> tuple[str, str, dict[str, str]]:
    """Return ``(image, build_context, build_args)`` for a model.

    Raises:
        KeyError: If the model name is not present.
        ValueError: If the model has no ``build`` block.
    """
    entry = _find(_load(models_yaml), name)
    build = entry.get("build")
    if not build:
        raise ValueError(f"model {name!r} has no build metadata")
    return entry["image"], build["context"], dict(build.get("args", {}))


def set_digest(models_yaml: Path, name: str, digest: str) -> None:
    """Write ``digest`` onto the named entry and rewrite the file.

    Raises:
        KeyError: If the model name is not present.
        ValueError: If ``digest`` is not a ``sha256:`` reference.
    """
    if not digest.startswith("sha256:"):
        raise ValueError(f"digest must start with 'sha256:', got {digest!r}")
    data = _load(Path(models_yaml))
    _find(data, name)["digest"] = digest
    Path(models_yaml).write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def main(argv: list[str]) -> int:
    """Tiny CLI used by the publishing workflow."""
    match argv:
        case ["lookup", models_yaml, name]:
            image, context, args = lookup_build(Path(models_yaml), name)
            # Emit GITHUB_OUTPUT lines. build_args uses docker/build-push-action's
            # native newline-separated KEY=VALUE form (single line for one arg).
            build_args = "\n".join(f"{k}={v}" for k, v in args.items())
            print(f"image={image}")
            print(f"context={context}")
            print(f"build_args={build_args}")
            return 0
        case ["set-digest", models_yaml, name, digest]:
            set_digest(Path(models_yaml), name, digest)
            return 0
        case _:
            print(__doc__)
            return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_registry_publish.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff check scripts/ tests/test_registry_publish.py && ruff format scripts/ tests/test_registry_publish.py
ty check scripts/registry_publish.py
git add scripts/__init__.py scripts/registry_publish.py tests/test_registry_publish.py
git commit -m "scripts: add registry_publish lookup/set-digest helpers"
```

---

### Task 8: GHCR publishing workflow

**Files:**
- Create: `.github/workflows/publish-image.yaml`

**Interfaces:**
- Consumes: `scripts/registry_publish.py` (Task 7); `src/plms/_data/models.yaml` (Task 6).
- Produces: a `workflow_dispatch` action that builds+pushes one model image to GHCR and opens a PR pinning its digest.

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/publish-image.yaml`:

```yaml
name: Publish model image

on:
  workflow_dispatch:
    inputs:
      model:
        description: Registry model name to build and publish (e.g. esm2-8m)
        required: true
        type: string

permissions:
  contents: write
  packages: write
  pull-requests: write

env:
  MODELS_YAML: src/plms/_data/models.yaml

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Free disk space
        uses: jlumbroso/free-disk-space@main
        with:
          tool-cache: true
          large-packages: false

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install PyYAML
        run: pip install pyyaml

      - name: Resolve build metadata
        id: meta
        run: python -m scripts.registry_publish lookup "$MODELS_YAML" "${{ inputs.model }}" >> "$GITHUB_OUTPUT"

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        id: build
        uses: docker/build-push-action@v6
        with:
          context: ${{ steps.meta.outputs.context }}
          file: ${{ steps.meta.outputs.context }}/Dockerfile
          platforms: linux/amd64
          push: true
          tags: ${{ steps.meta.outputs.image }}
          build-args: ${{ steps.meta.outputs.build_args }}

      - name: Pin digest in registry
        run: |
          DIGEST="${{ steps.build.outputs.digest }}"
          python -m scripts.registry_publish set-digest "$MODELS_YAML" "${{ inputs.model }}" "$DIGEST"

      - name: Open digest-pin PR
        uses: peter-evans/create-pull-request@v6
        with:
          branch: pin/${{ inputs.model }}
          title: "registry: pin ${{ inputs.model }} to published digest"
          commit-message: "registry: pin ${{ inputs.model }} to ${{ steps.build.outputs.digest }}"
          body: |
            Automated digest pin for `${{ inputs.model }}`.
            Image: `${{ steps.meta.outputs.image }}`
            Digest: `${{ steps.build.outputs.digest }}`

            > First-time packages default to private. Set the GHCR package to
            > public once in the repo's package settings so unauthenticated
            > `docker pull` works.
          add-paths: ${{ env.MODELS_YAML }}
```

> Note: `build_args` is emitted as the action's native newline-separated `KEY=VALUE` form (Task 7). All current models have exactly one build arg, so the value is a single line and round-trips cleanly through `$GITHUB_OUTPUT`. If a future model needs multiple build args, switch the `lookup` output to a `$GITHUB_OUTPUT` heredoc (multiline-output syntax) so the embedded newline survives.

- [ ] **Step 2: Validate the workflow YAML parses**

Run: `python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/publish-image.yaml').read_text()); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/publish-image.yaml
git commit -m "ci: add manual GHCR publish workflow with digest-pin PR"
```

---

### Task 9: Opt-in GHCR pull integration test

**Files:**
- Create: `tests/test_integration_ghcr.py`

**Interfaces:**
- Consumes: `load(allow_pull=...)` (Task 4); the published `esm2-8m` image (only when present).
- Produces: a `slow`, opt-in test that exercises a real pull-then-run, skipped unless explicitly enabled and docker is available.

- [ ] **Step 1: Write the test**

Create `tests/test_integration_ghcr.py`:

```python
"""Opt-in end-to-end test of GHCR pull + run.

Runs only when PLMS_RUN_GHCR_TESTS=1 and docker is available. It removes the
esm2-8m image locally, then `plms.load` with auto-pull enabled, proving the
client fetches the published image from GHCR and runs it.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

import plms
from plms.registry import Registry


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PLMS_RUN_GHCR_TESTS") != "1" or not _docker_available(),
        reason="set PLMS_RUN_GHCR_TESTS=1 with docker available to run GHCR pull tests",
    ),
]


def test_load_pulls_published_image_from_ghcr() -> None:
    ref = Registry.load().resolve("esm2-8m").pinned_ref()
    subprocess.run(["docker", "image", "rm", "-f", ref], capture_output=True)
    model = plms.load("esm2-8m", allow_pull=True)
    assert model.manifest.model_family == "esm2"
```

- [ ] **Step 2: Verify it is collected but skipped by default**

Run: `pytest tests/test_integration_ghcr.py -v`
Expected: 1 skipped (reason: GHCR tests not enabled).

- [ ] **Step 3: Lint, format, commit**

```bash
ruff check tests/test_integration_ghcr.py && ruff format tests/test_integration_ghcr.py
git add tests/test_integration_ghcr.py
git commit -m "test: opt-in GHCR pull+run integration test"
```

---

## Final verification

- [ ] Run the full fast suite: `pytest -m "not slow"` → all pass.
- [ ] `ruff check src/ tests/ scripts/` and `ruff format --check src/ tests/ scripts/` → clean.
- [ ] `ty check src/` → clean.
- [ ] Confirm `plms pull --help`, `plms embed --help` show the new options.
