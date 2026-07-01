# ESM-C → transformers migration + `contacts` — Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the ESM-C container off the EvolutionaryScale `esm` SDK onto HuggingFace `transformers` (uniform `.logits`/`hidden_states` interface) using the MIT-licensed **biohub** ESM-C weights, add the `contacts` capability, and register all three sizes (300M/600M/6B).

**Architecture:** ESM-C has no native `transformers` support (configs carry `model_type: esmc`, `auto_map: null`); the **biohub `esm` package** (`pip install esm@git+https://github.com/Biohub/esm.git`) registers `ESMCForMaskedLM` into transformers' auto-classes on import. The rewritten entrypoint then mirrors the shipped `containers/esm/entrypoint.py` exactly — `AutoModelForMaskedLM` + `AutoTokenizer`, embeddings from `output_hidden_states`, likelihood/score/contacts from `output.logits` — with an ESM-C-specific load path and a load-free manifest.

**Tech Stack:** Python 3.12, PyTorch, HuggingFace `transformers` (~4.57.x, pulled by the `esm` package), the biohub `esm` package, numpy, biopython (client eval, already shipped), Docker, pytest.

**Prerequisite:** Plan 1 is merged into `main` (contract 0.4 with `Capability.CONTACTS`/`ArtifactKind.CONTACT_MAP`, `Model.contacts`, `protlms.eval`, `protlms contacts` / `protlms eval contacts` CLI, and the shared `containers/esm/` image with the categorical-Jacobian helpers). This plan reuses all of that unchanged; it only adds/rewrites the ESM-C container and registry entries.

## Global Constraints

- **Client is untouched.** This plan changes only `containers/esm-c/`, `src/protlms/_data/models.yaml`, and the ESM-C tests. No changes to `contract.py`, `models.py`, `io.py`, `cli.py`, or `protlms.eval` (contract already at 0.4 with `contacts`).
- **Container `CONTRACT_VERSION = "0.4"`**; manifest `capabilities` = `["embed", "likelihood", "score", "contacts"]`; `model_family = "esm-c"`.
- **Weights (MIT, biohub):** `biohub/ESMC-300M`, `biohub/ESMC-600M`, `biohub/ESMC-6B`. Load-free manifest architecture table (verified from each repo's `config.json`):
  | checkpoint | hf_id | embedding_dim (`d_model`) | num_layers (`n_layers`) |
  |---|---|---|---|
  | `esmc_300m` | `biohub/ESMC-300M` | 960 | 30 |
  | `esmc_600m` | `biohub/ESMC-600M` | 1152 | 36 |
  | `esmc_6b` | `biohub/ESMC-6B` | 2560 | 80 |
  All three: `vocab_size=64`, `mask_token_id=32`, `pad_token_id=1`.
- **Load recipe (verified from the biohub card + EvolutionaryScale/Biohub repo README):** `import esm` (registers `ESMCForMaskedLM`) then `AutoModelForMaskedLM.from_pretrained(hf_id)` / `AutoTokenizer.from_pretrained(hf_id)`; forward → `output.logits` `(B, L, 64)`; `output_hidden_states=True` → `output.hidden_states`. No `trust_remote_code` needed for the model once `esm` is imported.
- **`MAX_SEQUENCE_LENGTH = 2048`** (unchanged from the current ESM-C image).
- **Standalone-container pattern:** the ESM-C entrypoint is self-contained — the categorical-Jacobian helpers are **duplicated** from `containers/esm/entrypoint.py` (matches the esm2/esm-c/progen2/protbert/e1 convention; do not import across containers).
- Python 3.11+, Google-style docstrings, type hints, ruff (line length 100). Container entrypoints are not under `ty check src/`, but `ruff check containers/esm-c/entrypoint.py` must pass. Heavy imports (`torch`, `transformers`, `esm`) go **inside** functions so the pure helpers unit-test without them.
- **Tests:** unit tests in `tests/test_esmc_entrypoint.py` (torch/esm-free, load the module by path); Docker integration in `tests/test_integration_esmc.py` gated by `@pytest.mark.slow` + `PROTLMS_RUN_DOCKER_TESTS=1`. **Run tests with `python -m pytest`** (bare `pytest` is a different interpreter without protlms). `python`/`pip` on PATH are the correct conda env.
- Commit style: `<component>: <what changed and why>`.

---

### Task 1: Rewrite the ESM-C entrypoint (transformers path) + `contacts`

**Files:**
- Modify: `containers/esm-c/entrypoint.py` (full rewrite)
- Test: `tests/test_esmc_entrypoint.py`

**Interfaces:**
- Produces (module-level, torch/esm-free): `sanitize_ids`, `read_fasta`, `perplexity_from_mean`, `parse_mutant`, `_truncate`, `aa_token_ids(tokenizer) -> list[int]`, `jacobian_to_contacts(jac) -> np.ndarray`, `write_contacts_outputs(output_dir, id_to_map) -> list[dict]`, `build_manifest() -> dict`, `build_parser()`.
- Produces (model-dependent, verified in Task 3): `load_model(device) -> (tokenizer, model)`, `_embed_one`, `_pseudo_log_likelihood`, `_masked_position_logprobs`, `_wt_position_logprobs`, `categorical_jacobian`, `cmd_*`.

**Approach:** The shipped `containers/esm/entrypoint.py` is already a transformers masked-LM contract entrypoint with the full contacts implementation. Start from it and apply the ESM-C-specific changes below. **Read `containers/esm/entrypoint.py` first** — you are copying its structure verbatim except where listed.

- [ ] **Step 1: Write the failing unit tests**

Replace the manifest tests in `tests/test_esmc_entrypoint.py` and add jacobian/aa/parser tests. Keep the existing `_load()` module-by-path loader and the pure-helper tests (`test_sanitize_ids_dedupes_collisions`, `test_read_fasta_parses_records`, `test_parse_mutant_*`, `test_perplexity_from_mean`, `test_truncate_warns_and_clips`). Replace `test_build_manifest_300m`/`_600m` with:

```python
import numpy as np


@pytest.mark.parametrize(
    ("checkpoint", "dim", "layers"),
    [("esmc_300m", 960, 30), ("esmc_600m", 1152, 36), ("esmc_6b", 2560, 80)],
)
def test_build_manifest(monkeypatch: pytest.MonkeyPatch, checkpoint: str, dim: int, layers: int) -> None:
    monkeypatch.setattr(entrypoint, "DEFAULT_CHECKPOINT", checkpoint)
    m = entrypoint.build_manifest()
    assert m["name"] == checkpoint
    assert m["model_family"] == "esm-c"
    assert m["contract_version"] == "0.4"
    assert m["embedding_dim"] == dim
    assert m["num_layers"] == layers
    assert m["capabilities"] == ["embed", "likelihood", "score", "contacts"]
    assert m["max_sequence_length"] == 2048


def test_jacobian_to_contacts_shape_symmetry_zero_diag() -> None:
    rng = np.random.default_rng(0)
    length = 7
    contacts = entrypoint.jacobian_to_contacts(rng.standard_normal((length, 20, length, 20)))
    assert contacts.shape == (length, length)
    assert contacts.dtype == np.float32
    assert np.allclose(contacts, contacts.T, atol=1e-5)
    assert np.allclose(np.diag(contacts), 0.0)


def test_aa_token_ids_maps_twenty_amino_acids() -> None:
    class FakeTok:
        def convert_tokens_to_ids(self, token: str) -> int:
            return ord(token)

    ids = entrypoint.aa_token_ids(FakeTok())
    assert len(ids) == 20
    assert ids[0] == ord("A")


def test_parser_has_contacts_subcommand() -> None:
    args = entrypoint.build_parser().parse_args(
        ["contacts", "--input", "/in/seqs.fasta", "--output", "/out"]
    )
    assert args.command == "contacts"
    assert args.method == "categorical-jacobian"
    assert args.func is entrypoint.cmd_contacts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_esmc_entrypoint.py -v`
Expected: FAIL (`jacobian_to_contacts`/`aa_token_ids`/`cmd_contacts` missing; manifest has contract 0.3 and no contacts; `esmc_6b` absent).

- [ ] **Step 3: Rewrite `containers/esm-c/entrypoint.py`**

Copy `containers/esm/entrypoint.py` to `containers/esm-c/entrypoint.py`, then apply exactly these changes (everything else — the pure helpers, `_embed_one`, `_pseudo_log_likelihood`, `_masked_position_logprobs`, `_wt_position_logprobs`, `_score_variant`, `aa_token_ids`, `jacobian_to_contacts`, `write_contacts_outputs`, `categorical_jacobian`, `cmd_embed`, `cmd_likelihood`, `cmd_score`, `cmd_contacts`, `_write_capability_result`, `build_parser`, `main` — is copied unchanged):

1. **Module docstring:** replace with a description of the ESM-C image via `transformers` + biohub weights, noting: "Heavy imports (`torch`, `transformers`, `esm`) happen inside functions ... Importing the biohub `esm` package registers the `ESMCForMaskedLM` architecture with transformers' auto classes."

2. **Constants block** — replace the esm globals (`HF_ID`/`MODEL_NAME`/`MODEL_FAMILY`) with:

```python
CONTRACT_VERSION = "0.4"
MAX_SEQUENCE_LENGTH = 2048
DEFAULT_BATCH_SIZE = 8
DEFAULT_CHECKPOINT = os.environ.get("ESMC_CHECKPOINT", "esmc_300m")
MODEL_FAMILY = "esm-c"

# checkpoint name -> architecture facts (keeps `manifest` load-free; verified from
# each biohub repo's config.json: d_model / n_layers).
_MODEL_INFO: dict[str, dict[str, object]] = {
    "esmc_300m": {"hf_id": "biohub/ESMC-300M", "embedding_dim": 960, "num_layers": 30, "min_gpu_memory_gb": None},
    "esmc_600m": {"hf_id": "biohub/ESMC-600M", "embedding_dim": 1152, "num_layers": 36, "min_gpu_memory_gb": 4.0},
    "esmc_6b": {"hf_id": "biohub/ESMC-6B", "embedding_dim": 2560, "num_layers": 80, "min_gpu_memory_gb": 24.0},
}
```

3. **`load_model`** — replace the esm version with (note `import esm` to register the arch, and returning `(tokenizer, model)` as the contacts helpers expect):

```python
def load_model(device: str):  # noqa: ANN201 - returns (tokenizer, model)
    """Load the tokenizer and masked-LM model for the configured biohub checkpoint."""
    import esm  # noqa: F401 - registers ESMCForMaskedLM with transformers auto classes
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    hf_id = _MODEL_INFO[DEFAULT_CHECKPOINT]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForMaskedLM.from_pretrained(hf_id, torch_dtype=torch.float32)
    model.eval().to(device)
    return tokenizer, model
```

4. **`build_manifest`** — replace the esm (AutoConfig-based) version with a load-free version driven by `_MODEL_INFO`:

```python
def build_manifest() -> dict:
    """Build the manifest from the checkpoint-keyed architecture table (load-free)."""
    info = _MODEL_INFO[DEFAULT_CHECKPOINT]
    return {
        "contract_version": CONTRACT_VERSION,
        "name": DEFAULT_CHECKPOINT,
        "version": "1.0.0",
        "description": f"ESM-C masked protein language model ({DEFAULT_CHECKPOINT}).",
        "model_family": MODEL_FAMILY,
        "capabilities": ["embed", "likelihood", "score", "contacts"],
        "embedding_dim": info["embedding_dim"],
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "pooling_modes": ["mean", "cls", "none"],
        "num_layers": info["num_layers"],
        "min_gpu_memory_gb": info["min_gpu_memory_gb"],
        "default_batch_size": DEFAULT_BATCH_SIZE,
    }
```

5. **`cmd_prefetch`** — replace with the transformers/esm prefetch:

```python
def cmd_prefetch(_args: argparse.Namespace) -> None:
    """Bake weights into the image at build time (populate the HF cache)."""
    import esm  # noqa: F401 - registers ESMCForMaskedLM
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    hf_id = _MODEL_INFO[DEFAULT_CHECKPOINT]["hf_id"]
    AutoTokenizer.from_pretrained(hf_id)
    AutoModelForMaskedLM.from_pretrained(hf_id)
    print(f"prefetched {hf_id}")
```

6. **`build_parser`** — change `prog="esm"` to `prog="esm-c"` (keep the `contacts` subparser and all others unchanged from the esm copy).

7. **Result `model_name`** — the esm copy uses `MODEL_NAME` in `cmd_score` and `_write_capability_result` and `cmd_contacts`. Replace every `MODEL_NAME` with `DEFAULT_CHECKPOINT` (there is no `MODEL_NAME` global in ESM-C).

> Note: `cmd_embed` copied from esm supports arbitrary `--layers` via `output_hidden_states` — this is fine (ESM-C exposes `hidden_states`); the old ESM-C image's `--layers -1`-only restriction is intentionally dropped.

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `python -m pytest tests/test_esmc_entrypoint.py -v`
Expected: PASS (module imports without torch/transformers/esm; all pure/manifest/jacobian/parser tests green).

- [ ] **Step 5: Lint + commit**

```bash
ruff check containers/esm-c/entrypoint.py tests/ && ruff format containers/esm-c/entrypoint.py tests/
git add containers/esm-c/entrypoint.py tests/test_esmc_entrypoint.py
git commit -m "esm-c: rewrite entrypoint on transformers + biohub weights + contacts"
```

---

### Task 2: ESM-C Dockerfile (biohub `esm` git) + registry + README

**Files:**
- Modify: `containers/esm-c/Dockerfile`
- Modify: `containers/esm-c/README.md`
- Modify: `src/protlms/_data/models.yaml`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: the rewritten entrypoint's `_prefetch` (Task 1).
- Produces: registry entries `esm-c-300m`, `esm-c-600m` (updated), `esm-c-6b` (new), all `build.context: containers/esm-c`, image `ghcr.io/briney/protlms-esm-c:<tag>`, `build.args: { ESMC_CHECKPOINT: esmc_<size> }`.

- [ ] **Step 1: Write the failing registry tests**

Add to `tests/test_registry.py`:

```python
def test_resolve_esm_c_6b() -> None:
    entry = Registry.load().resolve("esm-c-6b")
    assert entry.image == "ghcr.io/briney/protlms-esm-c:6b"
    assert entry.model_family == "esm-c"
    assert entry.build.context == "containers/esm-c"
    assert entry.build.args["ESMC_CHECKPOINT"] == "esmc_6b"
    assert Registry.load().resolve("esmc_6b") == entry


def test_registry_includes_all_esm_c_sizes() -> None:
    names = {e.name for e in Registry.load().list_models()}
    assert {"esm-c-300m", "esm-c-600m", "esm-c-6b"} <= names
```

(The existing `test_resolve_esm_c` for 300m/600m stays and must still pass — do not change those entries' `name`/`aliases`/`image`/`context`/`args` keys, which are unchanged by this plan.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_registry.py -k esm_c -v`
Expected: FAIL (`esm-c-6b` not present).

- [ ] **Step 3: Add the `esm-c-6b` registry entry**

In `src/protlms/_data/models.yaml`, after the `esm-c-600m` entry, add:

```yaml
  - name: esm-c-6b
    aliases: [esmc_6b]
    image: ghcr.io/briney/protlms-esm-c:6b
    model_family: esm-c
    build:
      context: containers/esm-c
      args: { ESMC_CHECKPOINT: esmc_6b }
```

(The `esm-c-300m`/`esm-c-600m` entries already use `context: containers/esm-c` + `ESMC_CHECKPOINT` — leave them as-is; the Dockerfile change below makes them build the new transformers image.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_registry.py -k esm_c -v`
Expected: PASS.

- [ ] **Step 5: Resolve and pin the biohub `esm` package commit**

The biohub `esm` package has no PyPI release yet, so pin a commit for reproducibility (mirrors how the E1 image pins `E1_REF`).

Run: `git ls-remote https://github.com/Biohub/esm.git main`
Record the 40-char SHA it prints (call it `<ESM_SHA>`). If the org resolves differently (case/redirect), use the resolved `https://github.com/<org>/esm.git` URL.

- [ ] **Step 6: Rewrite `containers/esm-c/Dockerfile`**

```dockerfile
# ESM-C model image for the protlms container contract.
#
# ESM-C via HuggingFace transformers, using the MIT-licensed biohub ESM-C weights.
# The biohub `esm` package registers the ESMCForMaskedLM architecture with
# transformers' auto classes (native transformers has no `esmc` support).
#
# Build (300M, default / CI):
#   docker build --build-arg ESMC_CHECKPOINT=esmc_300m -t protlms-esm-c:300m containers/esm-c
# Build (600M / 6B):
#   docker build --build-arg ESMC_CHECKPOINT=esmc_600m -t protlms-esm-c:600m containers/esm-c
#   docker build --build-arg ESMC_CHECKPOINT=esmc_6b   -t protlms-esm-c:6b   containers/esm-c
#
# Weights are baked in at build time, so runtime needs no network access.
# The image runs on CPU by default and uses the GPU when launched with --gpus.

ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

# Pinned biohub esm package commit (no PyPI release yet). Resolve via:
#   git ls-remote https://github.com/Biohub/esm.git main
ARG ESM_REF=<ESM_SHA>
ARG ESMC_CHECKPOINT=esmc_300m
ENV ESMC_CHECKPOINT=${ESMC_CHECKPOINT} \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

# git + build tools for the git-based esm install; esm pulls torch + transformers (~4.57.x).
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "esm @ git+https://github.com/Biohub/esm.git@${ESM_REF}"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

# Bake the checkpoint's weights into the image (populates the HF cache layer).
RUN python /app/entrypoint.py _prefetch

# Enforce offline weights at runtime for reproducibility.
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/app/entrypoint.py"]
```

Replace `<ESM_SHA>` with the SHA from Step 5.

- [ ] **Step 7: Update `containers/esm-c/README.md`**

Rewrite to describe: ESM-C via transformers + biohub MIT weights (300M/600M/6B), the `esm` package as the enabler, capabilities now including `contacts`, and the build commands from the Dockerfile header. Remove references to the `esm==3.2.3` SDK and the "300M/600M only" framing.

- [ ] **Step 8: Lint + commit**

```bash
ruff check src/ tests/ && ruff format src/ tests/
git add containers/esm-c/Dockerfile containers/esm-c/README.md src/protlms/_data/models.yaml tests/test_registry.py
git commit -m "esm-c: transformers/biohub Dockerfile (pinned esm) + esm-c-6b registry entry"
```

---

### Task 3: Docker integration — build 300M, verify load + full contract end-to-end

**Files:**
- Modify: `tests/test_integration_esmc.py`

**Interfaces:**
- Consumes: the rewritten entrypoint (Task 1) + Dockerfile/registry (Task 2); `protlms.load`, `Model.embed/likelihood/score/contacts`, `protlms.eval.runner.evaluate_contacts`; `tests/data/tiny.fasta`, `tests/data/variants.csv`, `tests/data/casp14/T1024.pdb` (all shipped).

**Gating:** `@pytest.mark.slow` + `PROTLMS_RUN_DOCKER_TESTS=1` (existing `pytestmark`). This task is the **probe**: it is the first real load of the biohub weights via transformers and the first execution of ESM-C's `categorical_jacobian`.

**Likely adjustment point (read before running):** `categorical_jacobian` (copied from esm) finds residue positions via `tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)` (flag==0 → residue), and `_embed_one`/`_pseudo_log_likelihood` (copied from esm) assume one BOS at index 0 and one EOS at the end. If the integration reveals ESM-C's tokenizer differs (e.g., `get_special_tokens_mask` unsupported, or a different special-token layout), fix within this task: replace the residue-position derivation with one based on the probe-confirmed layout (e.g. strip the known BOS/EOS indices), keeping the `(L,20,L,20)` construction identical. Confirm the actual layout first with a one-off probe (Step 1).

- [ ] **Step 1: Build the 300M image and probe the load (record the facts)**

```bash
docker build --build-arg ESMC_CHECKPOINT=esmc_300m -t ghcr.io/briney/protlms-esm-c:300m containers/esm-c
docker run --rm ghcr.io/briney/protlms-esm-c:300m manifest
docker run --rm ghcr.io/briney/protlms-esm-c:300m python -c "
import esm  # register
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('biohub/ESMC-300M')
model = AutoModelForMaskedLM.from_pretrained('biohub/ESMC-300M', torch_dtype=torch.float32).eval()
ids = tok('MPRTEINSEQ', return_tensors='pt')['input_ids'][0]
print('ids', ids.tolist())
out = model(input_ids=ids.unsqueeze(0), output_hidden_states=True)
print('logits', tuple(out.logits.shape), 'hidden', len(out.hidden_states))
print('special_mask', tok.get_special_tokens_mask(ids.tolist(), already_has_special_tokens=True))
print('mask_id', tok.mask_token_id)
print('aa_ids', [tok.convert_tokens_to_ids(a) for a in 'ACDEFGHIKLMNPQRSTVWY'], 'unk', tok.unk_token_id)
"
```
Expected: manifest shows `esm-c`, `embedding_dim 960`, `num_layers 30`, `contract_version 0.4`, `contacts` in capabilities. Logits shape `(1, L, 64)`; `hidden` = 31; special mask marks BOS/EOS; no AA maps to `<unk>`. Record the special-token layout in the task report; if it diverges from the esm assumptions, apply the fix described above to `containers/esm-c/entrypoint.py` and rebuild before Step 2.

- [ ] **Step 2: Update the integration tests**

Edit `tests/test_integration_esmc.py`. Keep the module gating and the `IMAGE`/`EMBEDDING_DIM`/`REPO_ROOT`/`TINY_FASTA`/`VARIANTS_CSV`/`EXPECTED_IDS` constants. Update the build fixture's `--build-arg` to remain `ESMC_CHECKPOINT=esmc_300m` (unchanged) and keep `IMAGE = "ghcr.io/briney/protlms-esm-c:300m"`. Update the manifest test and add contacts coverage:

```python
def test_manifest_is_read_through_client(model: protlms.Model) -> None:
    assert model.manifest.name == "esmc_300m"
    assert model.manifest.embedding_dim == EMBEDDING_DIM  # 960
    assert model.manifest.num_layers == 30
    assert model.manifest.contract_version == "0.4"
    capabilities = {c.value for c in model.manifest.capabilities}
    assert {"embed", "likelihood", "score", "contacts"} <= capabilities


def test_contacts_end_to_end_shapes(model: protlms.Model, tmp_path: Path) -> None:
    result = model.contacts(TINY_FASTA, output_dir=tmp_path / "ct")
    maps = result.maps()
    assert set(maps) == EXPECTED_IDS
    for cmap in maps.values():
        n = cmap.shape[0]
        assert cmap.shape == (n, n)
        assert cmap.dtype == np.float32
        assert np.isfinite(cmap).all()
        assert np.allclose(cmap, cmap.T, atol=1e-4)


def test_evaluate_contacts_casp14_target(model: protlms.Model, tmp_path: Path) -> None:
    from protlms.eval.runner import evaluate_contacts, mean_precision

    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    src = REPO_ROOT / "tests" / "data" / "casp14" / "T1024.pdb"
    (pdb_dir / "T1024.pdb").write_bytes(src.read_bytes())
    results = evaluate_contacts(model, pdb_dir, max_length=400)
    assert len(results) == 1
    assert results[0].target_id == "T1024"
    assert 0.0 <= results[0].precision_at_l <= 1.0
    assert not math.isnan(mean_precision(results))
```

Keep the existing `test_embed_*`, `test_likelihood_end_to_end`, and `test_score_*` tests — they must still pass via the new transformers path (they assert shapes / finiteness / self-substitution ≈ 0, which are path-independent; leave their assertions as-is, they are tolerance/sanity based).

- [ ] **Step 3: Run the full ESM-C integration suite**

Run:
```bash
PROTLMS_RUN_DOCKER_TESTS=1 python -m pytest tests/test_integration_esmc.py -m slow -v
```
Expected: image builds (or reused); manifest declares `contacts` + contract 0.4; embed/likelihood/score pass on the transformers path; `contacts` produces `(n,n)` float32 symmetric finite maps; CASP14 `evaluate_contacts` yields a finite T1024 precision@L in `[0,1]`. Record the observed T1024 precision@L (ESM-C 300M should beat a random baseline substantially — a real number, not a pass/fail).

- [ ] **Step 4: Commit**

```bash
ruff check tests/ && ruff format tests/
git add tests/test_integration_esmc.py containers/esm-c/entrypoint.py
git commit -m "test: esm-c transformers + contacts end-to-end integration"
```
(Include `containers/esm-c/entrypoint.py` only if Step 1 required a residue-position fix.)

---

### Final verification

- [ ] **Fast suite:** `python -m pytest -m "not slow" -p no:cacheprovider` → all green (registry + esm-c entrypoint unit tests included).
- [ ] **Lint:** `ruff check src/ tests/ containers/esm-c/entrypoint.py`, `ruff format --check src/ tests/` → clean.
- [ ] **CLI smoke (no Docker):** `protlms models list` shows `esm-c-300m`/`esm-c-600m`/`esm-c-6b` on `protlms-esm-c`.
- [ ] **(Docker) real benchmark, optional:** `PROTLMS_RUN_DOCKER_TESTS=1 python -m pytest tests/test_integration_esmc.py -m slow -q`, then `protlms eval contacts esm-c-300m --pdb-dir ~/projects/esm-c/data/casp14/ --out patl_esmc300m.csv --max-length 400`.

## Risks & caveats

- **The `esm` git dependency has no PyPI release.** Pinning a commit SHA (Task 2) makes builds reproducible; if the repo/URL casing differs, the implementer resolves it via `git ls-remote` in Task 2.
- **Tokenizer/residue-position layout is the one real unknown** — concentrated in Task 3's probe, with a specified fix path. Everything upstream is a verbatim port of the proven esm entrypoint.
- **6B is enormous:** baking `biohub/ESMC-6B` weights makes a very large image, and the categorical Jacobian (~O(20·L³)) is impractical on 6B for long targets — registered but opt-in; the benchmark defaults to 300M/600M. Only the 300M image is integration-tested.
- **transformers version:** the biohub configs were authored with `transformers 4.57.6`; the `esm` package pulls a compatible version. Do not pin a transformers 5.x in the image (no native `esmc`, and the `esm` package targets 4.57.x).

## Self-review notes

- **Spec coverage:** all-ESM-C-to-transformers migration (Task 1 entrypoint + Task 2 Dockerfile); `contacts` added (Task 1, ported helpers); register all three sizes incl. 6B (Task 2); manifest 0.4 + contacts (Task 1); integration incl. contacts + CASP14 (Task 3). Client/contract/eval unchanged (already shipped in Plan 1).
- **Placeholder scan:** the only intentional fill-in is `<ESM_SHA>` (resolved in Task 2 Step 5 via `git ls-remote` and pinned) — this is a runtime-resolved value, not a vague requirement.
- **Type/name consistency:** `DEFAULT_CHECKPOINT`, `_MODEL_INFO` (with `hf_id`), `MODEL_FAMILY`, `load_model → (tokenizer, model)`, `aa_token_ids`/`jacobian_to_contacts`/`write_contacts_outputs`/`categorical_jacobian`/`cmd_contacts`, and the `ESMC_CHECKPOINT` build arg are used consistently across tasks and match the shipped esm entrypoint's contacts interface.
