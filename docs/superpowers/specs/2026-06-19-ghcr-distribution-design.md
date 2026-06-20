# Phase 2 — GHCR Distribution (publish → pull → pin)

**Date:** 2026-06-19
**Status:** Approved (design)
**Roadmap:** VISION.md "Phase 2 — Distribution"

## Summary

Make `plms` model images distributable: publish each model's container image to
GitHub Container Registry (GHCR), have the client **auto-pull** the right image on
demand, and **pin every run to an image digest** for reproducibility. This closes
the publish → pull → pin loop so a user can `plms.load("esm2-8m")` on a fresh host
with no manual `docker build`.

Out of scope (deferred to a later phase): connecting to pre-existing **remote
inference endpoints** (a different architecture — live HTTP server vs. running a
container locally).

## Goals

- Registry entries reference GHCR images and carry an immutable digest.
- `load()` ensures the pinned image is present locally, pulling it when missing.
- A documented, controllable publishing pipeline pushes images to GHCR and pins
  the registry to the resulting digest via a reviewable PR.
- The client keeps **no ML dependencies**; everything new is client-side
  orchestration (registry/runner/CLI) plus CI.

## Non-Goals

- Remote inference endpoints.
- Private packages / auth management in the client (see Assumptions).
- Multi-arch images (GPU/CUDA → `linux/amd64` only).

## Decisions (from brainstorming)

| Topic | Decision |
| --- | --- |
| Scope | Full publish → pull → pin loop; remote endpoints deferred. |
| Pinning representation | `image` (human tag) **and** `digest` (`sha256:…`) fields; runs use `<repo>@<digest>`. CI writes the digest back. |
| Pull UX | Auto-pull on `load()` when missing; `PLMS_NO_PULL` env / `--no-pull` flag escape hatch; explicit `plms pull` command. |
| Visibility / auth | **All images public.** Client does not manage login; it surfaces a clear auth-hint on pull failure. |
| Publishing trigger | `workflow_dispatch` (manual), builds+pushes one model, opens a PR writing the digest back to `models.yaml`. |
| Pull-logic location | **Approach A:** primitives (`image_present`, `pull`) on the `Runner` protocol; orchestration in a module-level `ensure_image` helper. |

## Architecture

All new runtime behavior sits behind the existing `Runner` seam, so a future
Docker-SDK runner inherits pulling for free and tests can inject a fake runner.

### 1. Registry schema (`registry.py`)

`ModelEntry` gains two optional fields and one method:

```yaml
- name: esm2-8m
  aliases: [esm2_t6_8M]
  image: ghcr.io/briney/plms-esm2:t6_8M   # human-readable tag
  digest: sha256:abc123…                  # immutable; written back by CI
  model_family: esm2
  build:                                  # build-only metadata; client ignores it
    context: containers/esm2
    args: { ESM2_CHECKPOINT: facebook/esm2_t6_8M_UR50D }
```

- `digest: str | None = None` — bare `sha256:…` value. A Pydantic validator
  rejects values that do not start with `sha256:`.
- `build: BuildSpec | None = None` — `BuildSpec` is `{ context: str, args: dict[str, str] }`.
  Consumer-facing client code never reads it; only the publishing pipeline does.
  Keeping it here preserves "add a model = one registry entry."
- Method:

  ```python
  def pinned_ref(self) -> str:
      """Image ref to run/pull: '<repo>@<digest>' if digest set, else the tag.

      Strips ':tag' from the final path segment of `image` and appends
      '@<digest>'. With no digest (locally-built dev images), returns `image`
      unchanged.
      """
  ```

  Tag-stripping operates on the final `/`-segment only, so a registry host with a
  port (e.g. `host:5000/repo:tag`) is handled correctly. GHCR has no port.

### 2. Runner (`runner.py`)

Two primitives added to the `Runner` protocol:

```python
def image_present(self, ref: str) -> bool: ...   # `docker image inspect <ref>` exit 0
def pull(self, ref: str) -> None: ...            # `docker pull <ref>`; raises ImagePullError on non-zero
```

Orchestration is a module-level function (shared across runners; keeps the
protocol to two primitives):

```python
def ensure_image(runner: Runner, ref: str, *, allow_pull: bool, model_name: str) -> None:
    if runner.image_present(ref):
        return
    if not allow_pull:
        raise ImageNotFoundError(...)  # hint: `plms pull <model_name>` / unset PLMS_NO_PULL
    runner.pull(ref)
```

`manifest()` and `run()` keep their signatures; they simply receive the pinned ref.

### 3. Load flow (`models.py`)

`load()` gains `allow_pull: bool | None = None`. Resolution order:
explicit arg → `False` if `PLMS_NO_PULL` is truthy → default `True`.

```python
runner = runner or SubprocessDockerRunner()
registry = registry or Registry.load()
entry = registry.resolve(name)
ref = entry.pinned_ref()
ensure_image(runner, ref, allow_pull=_resolve_allow_pull(allow_pull), model_name=name)
manifest = Manifest.model_validate_json(runner.manifest(ref))
check_contract_compatibility(manifest.contract_version)
return Model(entry, runner, manifest, image_ref=ref)
```

`Model` stores `image_ref` and uses it for `RunSpec(image=self._image_ref, …)`
(today it passes `entry.image` at `models.py:369`), so embed/likelihood/generate
also execute the pinned digest.

### 4. CLI (`cli.py`)

- `plms pull <model>` — resolve entry, `ensure_image(allow_pull=True)` (pull if missing).
- `plms pull --all` — iterate every registry entry.
- `--no-pull` flag added to `embed` / `likelihood` / `generate`, threading
  `allow_pull=False` into `load()`.

## Publishing pipeline

### Build metadata

Lives in the optional `build:` block on each `ModelEntry` (above): single source
of truth, ignored by the client at runtime. (Alternative considered: a separate
`containers/build-matrix.yaml` — rejected for duplicating name→image and drifting.)

### Workflow (`.github/workflows/publish-image.yaml`)

- **Trigger:** `workflow_dispatch` with input `model` (a registry `name`).
- **Permissions:** `packages: write`, `contents: write`, `pull-requests: write`.
- **Steps:**
  1. Checkout; free disk space (CUDA+weights images are multi-GB; ubuntu-latest
     has ~14 GB free).
  2. Read `models.yaml`, resolve the chosen `name` → `image` (tag),
     `build.context`, `build.args` (via `scripts/registry_publish.py`).
  3. `docker login ghcr.io` with `GITHUB_TOKEN`.
  4. Build (`linux/amd64`) and push to the `image` tag.
  5. Capture the pushed digest (`docker buildx imagetools inspect` / `RepoDigests`).
  6. Write `sha256:…` into the matching entry's `digest:` in `models.yaml`.
  7. Open a PR (`peter-evans/create-pull-request`), e.g.
     `registry: pin esm2-8m to <short-digest>`.

A human triggers the build, reviews the digest bump in the PR, and **merging is
what pins it** — no surprise mutations to the committed registry.

### Visibility

Packages are public (project decision). A newly created GHCR package defaults to
private, and the default `GITHUB_TOKEN` typically lacks the scope to change
package visibility. So visibility is **not** flipped by the workflow: after a
package's first publish, an owner sets it to public once in the repo's package
settings (a documented one-time manual step). Subsequent publishes to the same
package inherit that visibility.

## Error handling (`exceptions.py`)

One new exception, `ImagePullError(RunnerError)`, raised by
`SubprocessDockerRunner.pull()` on a non-zero `docker pull`. Message carries the
ref, exit code, and stderr tail; if stderr looks auth-related
(`denied`/`unauthorized`), it appends a "try `docker login ghcr.io`" hint.

Failure modes stay distinct:

- `ImageNotFoundError` — image absent **and** we won't fetch (`allow_pull=False`);
  hint points to `plms pull <model>` / unsetting `PLMS_NO_PULL`. `manifest()`
  keeps raising this as a backstop, but the normal path is `ensure_image`.
- `ImagePullError` — fetch attempted and failed (network / auth / unknown digest).
- `RunnerError` (docker executable missing) — unchanged.

## Testing

- **Unit — `pinned_ref()`:** digest present → `<repo>@sha256:…` with `:tag`
  stripped from the final segment; digest `None` → `image` verbatim; host-with-port
  edge; invalid digest (no `sha256:` prefix) → validation error.
- **Unit — `ensure_image` with a `FakeRunner`** that records calls: present →
  no pull; absent + allow → pulls; absent + deny → `ImageNotFoundError`; pull
  raises → `ImagePullError` propagates.
- **Unit — `load()` allow_pull resolution:** explicit arg wins; `PLMS_NO_PULL`
  honored (monkeypatched env); default pulls. FakeRunner asserts `manifest()` /
  `run()` receive the **pinned ref**, not the tag.
- **Unit — CLI:** `plms pull <model>` and `--all` (CliRunner + injected fake
  runner/registry, per existing `test_cli.py` pattern); `--no-pull` threads
  `allow_pull=False`.
- **Unit — publishing scripts:** `scripts/registry_publish.py` pure functions
  (`lookup_build(name)`, `set_digest(models_yaml, name, digest)`) tested against a
  sample `models.yaml` — round-trips the digest and preserves comments/formatting.
  The Actions YAML itself is not unit-tested; its Python is.
- **Integration (slow, docker-gated):** existing ESM2 end-to-end test stays. A real
  GHCR-pull test is opt-in — `@pytest.mark.slow`, skipped unless the image is
  published — so normal CI never pulls multi-GB images. The `ensure_image`
  no-op-when-present path is covered against a locally-built image by tag.

## Assumptions & caveats (recorded, non-blocking)

- **All-public visibility assumes redistribution is permitted for every baked-in
  weight set.** ESM2 (MIT) and ProGen2 are open; ESM-C (EvolutionaryScale) has
  license terms that may restrict weight redistribution. Accepted as a project
  decision; revisit per-model if licensing requires it (the schema already allows
  a future per-entry visibility flag).
- GitHub-hosted runners may be tight on disk/time for the largest images;
  self-hosted runners are the fallback.

## Files touched

- `src/plms/registry.py` — `digest`, `build` fields; `BuildSpec`; `pinned_ref()`.
- `src/plms/runner.py` — `image_present`, `pull` on the protocol + impl;
  `ensure_image` helper.
- `src/plms/models.py` — `load(allow_pull=…)`, env resolution, `Model.image_ref`.
- `src/plms/exceptions.py` — `ImagePullError`.
- `src/plms/cli.py` — `plms pull` command; `--no-pull` on embed/likelihood/generate.
- `src/plms/_data/models.yaml` — GHCR image refs, `build:` blocks (digests via CI).
- `scripts/registry_publish.py` — build lookup + digest write-back (testable).
- `.github/workflows/publish-image.yaml` — manual build/push + digest-PR workflow.
- `tests/` — `test_registry.py`, `test_runner.py`, `test_models.py`, `test_cli.py`,
  `test_registry_publish.py`, plus a slow GHCR-pull integration test.
