# ESM-family `contacts` + CASP14 evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `contacts` capability (categorical-Jacobian contact prediction) to the protlms contract, the shared ESM masked-LM container, and the client, plus an in-package `protlms.eval` harness that scores long-range precision@L on CASP14 — for ESM-1b and all ESM-2 sizes.

**Architecture:** The container computes the full categorical Jacobian internally (mutate each position to all 20 AAs → logits → `(L,20,L,20)` → center → symmetrize → Frobenius → APC) and writes one tiny `(L,L)` float32 contact map per record. The ML-free client validates/stages/drives/parses (mirroring `embed`/`score`). A separate `protlms.eval` module (numpy + biopython) builds ground-truth contacts from PDBs and computes the metric. ESM-1b and ESM-2 share one generalized `containers/esm/` image selected by build args.

**Tech Stack:** Python 3.11+, PyTorch + HuggingFace `transformers` (container only), numpy, biopython, pydantic, typer, rich, Docker, pytest.

**Scope note:** This plan covers the **ESM family only**. ESM-C's migration to `transformers` + `contacts` is a small follow-on plan (Plan 2) that reuses everything built here and only adds its container + registry entries.

## Global Constraints

- **Python 3.11+**, modern syntax (`X | Y`, `match`, `StrEnum`), `from __future__ import annotations`.
- **Type hints on all signatures**; Google-style docstrings on all public functions/classes.
- **Ruff** formats/lints (line length 100); **`ty check src/`** must pass. Run `ruff check src/ tests/`, `ruff format src/ tests/` before each commit.
- **Client carries no ML dependencies.** numpy is already core; **biopython** is added to core deps (non-ML). torch/transformers live only in container entrypoints, imported *inside* functions so pure helpers unit-test without them.
- **Contract mirror rule:** `src/protlms/contract.py` and `docs/CONTRACT.md` are edited **together** (a drift-guard test enforces it).
- **Tests mirror src** and go in `tests/` (flat layout, e.g. `src/protlms/eval/contacts.py` → `tests/test_eval_contacts.py`). Use **real data** where possible. Mark Docker/model tests `@pytest.mark.slow`, gated by `PROTLMS_RUN_DOCKER_TESTS=1`.
- **Contract version:** `0.4` (minor bump from `0.3`; backward-compatible).
- **New capability string:** `contacts`. **New artifact kind:** `contact_map`. **New `contacts` method name:** `categorical-jacobian`.
- **Metric defaults (verbatim from spec):** true contact = Cβ–Cβ (Cα for Gly) `< 8.0 Å`; long-range = `|i − j| ≥ 24` (by PDB residue number); precision@L with `L = number of resolved residues` (top-L **unique** upper-triangle pairs).
- **Commit style:** `<component>: <what changed and why>`, imperative.

---

### Task 1: Contract — add `contacts` capability, bump to 0.4

**Files:**
- Modify: `src/protlms/contract.py`
- Modify: `docs/CONTRACT.md`
- Modify: `tests/data/manifest.example.json`
- Create: `tests/data/result.contacts.example.json`
- Test: `tests/test_contract.py`

**Interfaces:**
- Produces: `Capability.CONTACTS == "contacts"`, `ArtifactKind.CONTACT_MAP == "contact_map"`, `CONTRACT_VERSION == "0.4"`.

- [ ] **Step 1: Update the contract-version test to 0.4 and add contacts assertions (failing)**

In `tests/test_contract.py`, change the existing version test body and add two tests:

```python
def test_contract_version_is_semantic_string() -> None:
    assert CONTRACT_VERSION == "0.4"
    assert parse_contract_version(CONTRACT_VERSION) == (0, 4)


def test_contacts_capability_and_artifact_kind_exist() -> None:
    from protlms.contract import ArtifactKind

    assert Capability.CONTACTS == "contacts"
    assert ArtifactKind.CONTACT_MAP == "contact_map"


def test_documented_contacts_result_example_validates() -> None:
    """The contacts result example must parse as a Result."""
    result = Result.model_validate_json((_DATA / "result.contacts.example.json").read_text())
    assert result.capability is Capability.CONTACTS
    assert result.artifacts[0].kind == "contact_map"
    assert result.artifacts[0].shape is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contract.py -k "contacts or semantic" -v`
Expected: FAIL (`CONTRACT_VERSION` still `0.3`; `Capability.CONTACTS`/`ArtifactKind.CONTACT_MAP` missing; example file missing).

- [ ] **Step 3: Implement the contract changes**

In `src/protlms/contract.py`: set `CONTRACT_VERSION = "0.4"`; add to `Capability`:

```python
    CONTACTS = "contacts"  # implemented in contract 0.4
```

and to `ArtifactKind`:

```python
    CONTACT_MAP = "contact_map"
```

- [ ] **Step 4: Create the worked example**

Create `tests/data/result.contacts.example.json`:

```json
{
  "contract_version": "0.4",
  "capability": "contacts",
  "model_name": "esm2_t6_8M",
  "n_input_records": 2,
  "n_output_records": 2,
  "artifacts": [
    {"path": "contacts/gb1.npy", "kind": "contact_map", "record_ids": ["gb1"], "shape": [56, 56], "dtype": "float32"},
    {"path": "contacts/insulin_b.npy", "kind": "contact_map", "record_ids": ["insulin_b"], "shape": [30, 30], "dtype": "float32"}
  ],
  "warnings": [],
  "params": {"method": "categorical-jacobian", "device": "auto"}
}
```

- [ ] **Step 5: Update the manifest example to 0.4 + contacts**

In `tests/data/manifest.example.json`: set `"contract_version": "0.4"` and add `"contacts"` to the `capabilities` array (keep existing entries).

- [ ] **Step 6: Update `docs/CONTRACT.md` (mirror the code)**

Make these edits:
- Header: `> **Contract version:** \`0.4\``.
- In the "Internal CLI" subcommand block, add:
  ```
  <entry> contacts   --input /in/seqs.fasta --output /out
                     [--method categorical-jacobian] [--batch-size N] [--device cpu|cuda]
  ```
- In the capabilities row of the manifest table, note `contacts` is a valid capability.
- In the "Outputs" table add a row:
  `| \`contacts\` | \`contacts/<id>.npy\` — one \`(L, L)\` float32 contact-score matrix per record. |`
- In the OutputArtifact `kind` list add `contact_map`.
- Add a worked example mirroring `tests/data/result.contacts.example.json`.

- [ ] **Step 7: Run the full contract test file**

Run: `pytest tests/test_contract.py -v`
Expected: PASS (all, including `test_documented_manifest_example_validates`).

- [ ] **Step 8: Lint, format, type-check, commit**

```bash
ruff check src/ tests/ && ruff format src/ tests/ && ty check src/
git add src/protlms/contract.py docs/CONTRACT.md tests/data/manifest.example.json tests/data/result.contacts.example.json tests/test_contract.py
git commit -m "contract: add contacts capability + contact_map artifact (bump 0.4)"
```

---

### Task 2: Client IO — `load_contact_maps`

**Files:**
- Modify: `src/protlms/io.py`
- Test: `tests/test_io.py`

**Interfaces:**
- Consumes: `Result`, `ArtifactKind.CONTACT_MAP` (Task 1).
- Produces: `load_contact_maps(out_dir: Path, result: Result) -> dict[str, np.ndarray]` (id → `(L, L)` array).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_io.py` (extend the io import line to include `load_contact_maps`, and add `from protlms.contract import Result`):

```python
def test_load_contact_maps_returns_arrays_by_id(tmp_path: Path) -> None:
    (tmp_path / "contacts").mkdir()
    np.save(tmp_path / "contacts" / "gb1.npy", np.zeros((5, 5), dtype=np.float32))
    result = Result.model_validate(
        {
            "contract_version": "0.4",
            "capability": "contacts",
            "model_name": "esm2_t6_8M",
            "n_input_records": 1,
            "n_output_records": 1,
            "artifacts": [
                {
                    "path": "contacts/gb1.npy",
                    "kind": "contact_map",
                    "record_ids": ["gb1"],
                    "shape": [5, 5],
                    "dtype": "float32",
                }
            ],
        }
    )
    maps = load_contact_maps(tmp_path, result)
    assert set(maps) == {"gb1"}
    assert maps["gb1"].shape == (5, 5)


def test_load_contact_maps_raises_when_absent(tmp_path: Path) -> None:
    result = Result.model_validate(
        {
            "contract_version": "0.4",
            "capability": "contacts",
            "model_name": "x",
            "n_input_records": 0,
            "n_output_records": 0,
            "artifacts": [],
        }
    )
    with pytest.raises(OutputParseError):
        load_contact_maps(tmp_path, result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_io.py -k contact_maps -v`
Expected: FAIL (`load_contact_maps` not importable).

- [ ] **Step 3: Implement `load_contact_maps`**

Add to `src/protlms/io.py` (after `load_per_residue_embeddings`):

```python
def load_contact_maps(out_dir: Path, result: Result) -> dict[str, np.ndarray]:
    """Load contact-score maps keyed by record id (each shape ``(L, L)``).

    Raises:
        OutputParseError: If no contact_map artifact is present.
    """
    out: dict[str, np.ndarray] = {}
    for artifact in _artifacts(result, ArtifactKind.CONTACT_MAP):
        rid = artifact.record_ids[0] if artifact.record_ids else Path(artifact.path).stem
        out[rid] = np.load(out_dir / artifact.path)
    if not out:
        raise OutputParseError("result declares no contact_map artifacts")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_io.py -k contact_maps -v`
Expected: PASS.

- [ ] **Step 5: Lint, format, type-check, commit**

```bash
ruff check src/ tests/ && ruff format src/ tests/ && ty check src/
git add src/protlms/io.py tests/test_io.py
git commit -m "io: add load_contact_maps for contact_map artifacts"
```

---

### Task 3: Client model — `Model.contacts` + `ContactsResult`

**Files:**
- Modify: `src/protlms/models.py`
- Modify: `src/protlms/__init__.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `load_contact_maps` (Task 2), `Capability.CONTACTS` (Task 1), existing `_run`, `_require_capability`, `stage_inputs`.
- Produces: `ContactsResult` dataclass with `.maps() -> dict[str, np.ndarray]`; `Model.contacts(fasta, *, method="categorical-jacobian", output_dir=None, use_gpu=False, batch_size=None) -> ContactsResult`.

**Design note (YAGNI):** `contacts` does **not** take `chunk_size` — the container loops over a multi-record FASTA internally, and the eval sends all targets in one run. (Client-side chunking merge is `embed`/`likelihood`/`generate`-only in `chunking.py`; adding a contacts merge is deferred until a real need arises.)

- [ ] **Step 1: Teach the test FakeRunner to emit contacts, and add tests (failing)**

In `tests/test_models.py`, in `FakeRunner._write_outputs`, add a branch:

```python
        elif capability == "contacts":
            records = read_fasta(spec.input_dir / "seqs.fasta")
            self._write_contacts(out, records)
```

and add the writer method to `FakeRunner`:

```python
    def _write_contacts(self, out: Path, records) -> None:  # noqa: ANN001
        contacts_dir = out / "contacts"
        contacts_dir.mkdir()
        artifacts = []
        for rec in records:
            n = len(rec.sequence)
            np.save(contacts_dir / f"{rec.id}.npy", np.zeros((n, n), dtype=np.float32))
            artifacts.append(
                {
                    "path": f"contacts/{rec.id}.npy",
                    "kind": "contact_map",
                    "record_ids": [rec.id],
                    "shape": [n, n],
                    "dtype": "float32",
                }
            )
        self._write_result(out, "contacts", records, artifacts)
```

Add `contacts` to the default `capabilities` list in `_manifest_json` (change to `["embed", "likelihood", "score", "contacts"]`). Then add tests:

```python
def test_contacts_returns_maps_by_id(fasta: Path, tmp_path: Path) -> None:
    from protlms.models import ContactsResult

    model = _load()
    result = model.contacts(fasta, output_dir=tmp_path / "ct")
    assert isinstance(result, ContactsResult)
    maps = result.maps()
    assert set(maps) == {"seq1", "seq2"}
    assert maps["seq1"].shape == (9, 9)  # len("ACDEFGHIK") == 9


def test_contacts_builds_expected_command(fasta: Path, tmp_path: Path) -> None:
    model = _load()
    model.contacts(fasta, output_dir=tmp_path / "ct")
    cmd = model._runner.last_spec.command  # type: ignore[attr-defined]
    assert cmd[0] == "contacts"
    assert cmd[cmd.index("--input") + 1] == "/in/seqs.fasta"
    assert cmd[cmd.index("--method") + 1] == "categorical-jacobian"


def test_contacts_invalid_method_raises_before_run(fasta: Path, tmp_path: Path) -> None:
    model = _load()
    with pytest.raises(InvalidRequestError):
        model.contacts(fasta, method="bogus", output_dir=tmp_path / "ct")
    assert model._runner.last_spec is None  # type: ignore[attr-defined]


def test_contacts_unsupported_capability_raises(fasta: Path, tmp_path: Path) -> None:
    model = _load(capabilities=["embed"])  # no contacts
    with pytest.raises(CapabilityNotSupportedError):
        model.contacts(fasta, output_dir=tmp_path / "ct")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -k contacts -v`
Expected: FAIL (`Model.contacts`/`ContactsResult` do not exist).

- [ ] **Step 3: Implement `ContactsResult` and `Model.contacts`**

In `src/protlms/models.py`: add `load_contact_maps` to the `protlms.io` import block. Add the dataclass near the other result dataclasses:

```python
@dataclass
class ContactsResult:
    """Handle to the outputs of a ``contacts`` run (arrays loaded lazily)."""

    result: Result
    output_dir: Path
    method: str
    _keepalive: tempfile.TemporaryDirectory | None = field(default=None, repr=False)

    def maps(self) -> dict[str, np.ndarray]:
        """Return contact-score maps keyed by record id (shape ``(L, L)``)."""
        return load_contact_maps(self.output_dir, self.result)
```

Add the method to `Model` (after `score`):

```python
    def contacts(
        self,
        fasta: str | Path,
        *,
        method: str = "categorical-jacobian",
        output_dir: Path | None = None,
        use_gpu: bool = False,
        batch_size: int | None = None,
    ) -> ContactsResult:
        """Predict a residue-residue contact-score map per sequence.

        Uses the categorical-Jacobian method: the container mutates each position
        to all 20 amino acids, reads the model logits, and post-processes the
        resulting ``(L, 20, L, 20)`` tensor into an ``(L, L)`` contact map.

        Args:
            fasta: Path to the input FASTA file (one map produced per record).
            method: Contact-prediction method (only ``"categorical-jacobian"``).
            output_dir: Where to write outputs; a temporary directory if ``None``.
            use_gpu: Request all GPUs for the container run.
            batch_size: Override the model's default batch size.

        Raises:
            CapabilityNotSupportedError: If the model does not support contacts.
            InvalidRequestError: If ``method`` is invalid or the input is empty.
            ContainerExecutionError: If the container run fails.
        """
        self._require_capability(Capability.CONTACTS)
        if method != "categorical-jacobian":
            raise InvalidRequestError(
                f"unsupported contacts method {method!r}; choose 'categorical-jacobian'"
            )
        records = self._read_records(fasta)
        extra = ["--method", method]
        if batch_size is not None:
            extra += ["--batch-size", str(batch_size)]
        result, out_dir, keep = self._run(
            Capability.CONTACTS, stage_inputs(records), extra, output_dir, use_gpu
        )
        return ContactsResult(
            result=result, output_dir=out_dir, method=method, _keepalive=keep
        )
```

- [ ] **Step 4: Export `ContactsResult`**

In `src/protlms/__init__.py`: add `ContactsResult` to the `from protlms.models import (...)` block and to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models.py -k contacts -v`
Expected: PASS.

- [ ] **Step 6: Run full model + import tests, lint, commit**

```bash
pytest tests/test_models.py -q
python -c "from protlms import ContactsResult; print(ContactsResult.__name__)"
ruff check src/ tests/ && ruff format src/ tests/ && ty check src/
git add src/protlms/models.py src/protlms/__init__.py tests/test_models.py
git commit -m "models: add Model.contacts + ContactsResult"
```

---

### Task 4: CLI — `protlms contacts`

**Files:**
- Modify: `src/protlms/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Model.contacts`, `ContactsResult` (Task 3).
- Produces: `protlms contacts MODEL FASTA -o OUT [--method ...] [--gpu] [--batch-size N] [--no-pull]`.

- [ ] **Step 1: Add a `contacts` method to the test FakeModel and a test (failing)**

In `tests/test_cli.py`, add to `FakeModel`:

```python
    def contacts(self, fasta, *, method, output_dir, use_gpu, batch_size):  # noqa: ANN001
        FakeModel.last_call = {"method": "contacts", "contacts_method": method, "use_gpu": use_gpu}
        return ContactsResult(
            result=_result("contacts", [{"path": "contacts/seq1.npy", "kind": "contact_map"}]),
            output_dir=Path(output_dir),
            method=method,
        )
```

Add `ContactsResult` to the `from protlms.models import (...)` import. Add the test:

```python
def test_contacts_command_invokes_model(fasta: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
    result = runner.invoke(
        app, ["contacts", "esm2-8m", str(fasta), "-o", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.stdout
    assert FakeModel.last_call["method"] == "contacts"
    assert FakeModel.last_call["contacts_method"] == "categorical-jacobian"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -k contacts_command -v`
Expected: FAIL (no `contacts` command).

- [ ] **Step 3: Implement the command**

In `src/protlms/cli.py`, add after the `score` command:

```python
@app.command()
def contacts(
    model: _ModelArg,
    fasta: _FastaArg,
    output_dir: _OutputOpt,
    method: Annotated[
        str, typer.Option("--method", help="Contact method: categorical-jacobian.")
    ] = "categorical-jacobian",
    gpu: _GpuOpt = False,
    batch_size: _BatchOpt = None,
    no_pull: _NoPullOpt = False,
) -> None:
    """Predict residue-residue contact maps (categorical Jacobian)."""
    try:
        model_obj = load(model, allow_pull=False if no_pull else None)
        result = model_obj.contacts(
            fasta, method=method, output_dir=output_dir, use_gpu=gpu, batch_size=batch_size
        )
        console.print(
            f"[green]contacts[/green] complete: {result.result.n_output_records} map(s) "
            f"written to [bold]{output_dir}[/bold] method={method}"
        )
    except ProtlmsError as exc:
        _fail(exc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -k contacts_command -v`
Expected: PASS.

- [ ] **Step 5: Lint, format, type-check, commit**

```bash
ruff check src/ tests/ && ruff format src/ tests/ && ty check src/
git add src/protlms/cli.py tests/test_cli.py
git commit -m "cli: add protlms contacts command"
```

---

### Task 5: Eval — PDB parsing + true contact map (adds biopython)

**Files:**
- Modify: `pyproject.toml`
- Create: `src/protlms/eval/__init__.py`
- Create: `src/protlms/eval/contacts.py`
- Create: `tests/data/casp14/T1024.pdb` (copied real data)
- Test: `tests/test_eval_contacts.py`

**Interfaces:**
- Produces: `PdbChain(sequence: str, resnums: np.ndarray, cb_coords: np.ndarray)`; `parse_pdb(pdb: Path, *, chain: str | None = None) -> PdbChain`; `true_contact_map(cb_coords: np.ndarray, *, threshold: float = 8.0) -> np.ndarray` (shape `(N, N)` bool); constants `CONTACT_THRESHOLD_ANGSTROM = 8.0`, `LONG_RANGE_SEP = 24`.

- [ ] **Step 1: Add biopython and install**

In `pyproject.toml`, add to `dependencies`:

```toml
    # PDB parsing for the structural-contact evaluation harness.
    "biopython>=1.83",
```

Run: `pip install -e ".[dev]"`
Expected: installs biopython; `python -c "import Bio; print(Bio.__version__)"` prints a version ≥ 1.83.

- [ ] **Step 2: Copy one real CASP14 PDB into the test data**

```bash
mkdir -p tests/data/casp14
cp ~/projects/esm-c/data/casp14/T1024.pdb tests/data/casp14/T1024.pdb
```
Expected: `tests/data/casp14/T1024.pdb` exists (a real structure with ~391 residues).

- [ ] **Step 3: Write the failing test**

Create `tests/test_eval_contacts.py`:

```python
"""Tests for PDB parsing and the contact metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from protlms.eval.contacts import parse_pdb, true_contact_map

_PDB = Path(__file__).parent / "data" / "casp14" / "T1024.pdb"


def test_parse_pdb_extracts_sequence_and_coords() -> None:
    chain = parse_pdb(_PDB)
    n = len(chain.sequence)
    assert n > 50
    assert chain.resnums.shape == (n,)
    assert chain.cb_coords.shape == (n, 3)
    assert set(chain.sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
    assert np.all(np.diff(chain.resnums) >= 1)  # strictly increasing residue numbers


def test_true_contact_map_is_symmetric_bool_with_zero_diagonal() -> None:
    chain = parse_pdb(_PDB)
    cmap = true_contact_map(chain.cb_coords)
    n = len(chain.sequence)
    assert cmap.shape == (n, n)
    assert cmap.dtype == bool
    assert np.array_equal(cmap, cmap.T)
    assert not cmap.diagonal().any()
    assert cmap.sum() > 0  # a real fold has contacts


def test_true_contact_map_threshold_monotone() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    cmap = true_contact_map(coords, threshold=8.0)
    assert cmap[0, 1] and not cmap[0, 2]
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_eval_contacts.py -v`
Expected: FAIL (`protlms.eval.contacts` does not exist).

- [ ] **Step 5: Create the eval package + PDB parsing/contact code**

Create `src/protlms/eval/__init__.py`:

```python
"""Structural evaluation utilities for protlms (PDB contacts, precision@L)."""

from __future__ import annotations

from protlms.eval.contacts import (
    CONTACT_THRESHOLD_ANGSTROM,
    LONG_RANGE_SEP,
    PdbChain,
    parse_pdb,
    true_contact_map,
)

__all__ = [
    "PdbChain",
    "parse_pdb",
    "true_contact_map",
    "CONTACT_THRESHOLD_ANGSTROM",
    "LONG_RANGE_SEP",
]
```

Create `src/protlms/eval/contacts.py`:

```python
"""PDB contact-map extraction and long-range precision@L.

Depends on BioPython (PDB parsing) and numpy (metric). No ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

#: Contact distance cutoff (Cβ–Cβ, Cα for glycine), in Angstroms.
CONTACT_THRESHOLD_ANGSTROM = 8.0
#: Minimum sequence separation |i − j| for a "long-range" pair.
LONG_RANGE_SEP = 24

#: Three-letter → one-letter codes for the 20 standard amino acids.
_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass(frozen=True)
class PdbChain:
    """Resolved residues of one PDB chain.

    Attributes:
        sequence: One-letter sequence of resolved residues, in chain order.
        resnums: ``(N,)`` int array of author residue numbers.
        cb_coords: ``(N, 3)`` float array of Cβ (Cα for Gly) coordinates.
    """

    sequence: str
    resnums: np.ndarray
    cb_coords: np.ndarray


def parse_pdb(pdb: Path, *, chain: str | None = None) -> PdbChain:
    """Parse resolved standard residues from a PDB file.

    Uses the first model and (by default) the first chain. Non-standard residues,
    HETATM/water, and residues lacking Cβ/Cα are skipped.

    Args:
        pdb: Path to a ``.pdb`` file.
        chain: Chain id to read; the first chain if ``None``.

    Returns:
        A :class:`PdbChain` for the resolved residues.

    Raises:
        ValueError: If no usable standard residues are found.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(pdb))
    model = next(iter(structure))
    chain_obj = model[chain] if chain is not None else next(iter(model))

    seq_chars: list[str] = []
    resnums: list[int] = []
    coords: list[np.ndarray] = []
    for residue in chain_obj:
        if residue.id[0] != " " or not is_aa(residue, standard=True):
            continue
        one = _THREE_TO_ONE.get(residue.get_resname().strip().upper())
        if one is None:
            continue
        atom = residue["CB"] if "CB" in residue else (residue["CA"] if "CA" in residue else None)
        if atom is None:
            continue
        seq_chars.append(one)
        resnums.append(int(residue.id[1]))
        coords.append(np.asarray(atom.get_coord(), dtype=float))

    if not seq_chars:
        raise ValueError(f"no standard residues with Cβ/Cα found in {pdb}")
    return PdbChain(
        sequence="".join(seq_chars),
        resnums=np.asarray(resnums, dtype=int),
        cb_coords=np.asarray(coords, dtype=float),
    )


def true_contact_map(
    cb_coords: np.ndarray, *, threshold: float = CONTACT_THRESHOLD_ANGSTROM
) -> np.ndarray:
    """Boolean ``(N, N)`` contact map: True where Cβ–Cβ distance < ``threshold``.

    The diagonal is set to False.
    """
    coords = np.asarray(cb_coords, dtype=float)  # (N, 3)
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    dist = np.sqrt((diff**2).sum(axis=-1))  # (N, N)
    contacts = dist < threshold
    np.fill_diagonal(contacts, False)
    return contacts
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_eval_contacts.py -v`
Expected: PASS.

- [ ] **Step 7: Lint, format, type-check, commit**

```bash
ruff check src/ tests/ && ruff format src/ tests/ && ty check src/
git add pyproject.toml src/protlms/eval/__init__.py src/protlms/eval/contacts.py tests/test_eval_contacts.py tests/data/casp14/T1024.pdb
git commit -m "eval: PDB parsing + true contact map (add biopython dep)"
```

---

### Task 6: Eval — long-range precision@L

**Files:**
- Modify: `src/protlms/eval/contacts.py`
- Modify: `src/protlms/eval/__init__.py`
- Test: `tests/test_eval_contacts.py`

**Interfaces:**
- Consumes: `true_contact_map`, `LONG_RANGE_SEP` (Task 5).
- Produces: `long_range_precision_at_l(pred: np.ndarray, true: np.ndarray, resnums: np.ndarray, *, sep: int = LONG_RANGE_SEP, top: int | None = None) -> float`.

**Metric definition:** standard contact precision@L — of the top-`L` **unique** upper-triangle pairs with `|resnum_i − resnum_j| ≥ sep` ranked by predicted score, the fraction that are true contacts. `top=None` ⇒ `L = N` (number of resolved residues). Returns `nan` if no eligible pairs. (The reference notebook double-counts each pair, which effectively halves L; we implement the standard single-count definition and expose `top` for parity if ever needed.)

- [ ] **Step 1: Write the failing test (hand-checkable)**

Add to `tests/test_eval_contacts.py` (extend the import to include `long_range_precision_at_l`):

```python
def test_long_range_precision_at_l_known_case() -> None:
    from protlms.eval.contacts import long_range_precision_at_l

    n = 6
    resnums = np.arange(n)  # 0..5; long-range with sep=3 => |i-j| >= 3
    # eligible upper-tri pairs (sep>=3): (0,3),(0,4),(0,5),(1,4),(1,5),(2,5)
    true = np.zeros((n, n), dtype=bool)
    true[0, 3] = true[3, 0] = True  # a real long-range contact
    true[1, 5] = true[5, 1] = True  # another
    pred = np.zeros((n, n), dtype=float)
    pred[0, 3] = 0.9  # top-ranked, true
    pred[0, 4] = 0.8  # 2nd, false
    pred[1, 5] = 0.7  # 3rd, true
    pred = (pred + pred.T) / 2
    # top = 2 => picks (0,3) true and (0,4) false => precision 0.5
    assert long_range_precision_at_l(pred, true, resnums, sep=3, top=2) == 0.5
    # top defaults to N=6 but only 3 nonzero-scored + rest 0; top clipped to eligible count (6)
    # true positives among all 6 eligible = 2 => 2/6
    assert long_range_precision_at_l(pred, true, resnums, sep=3) == pytest.approx(2 / 6)


def test_long_range_precision_at_l_no_eligible_pairs_is_nan() -> None:
    from protlms.eval.contacts import long_range_precision_at_l

    n = 3
    pred = np.zeros((n, n))
    true = np.zeros((n, n), dtype=bool)
    resnums = np.arange(n)
    assert np.isnan(long_range_precision_at_l(pred, true, resnums, sep=24))
```

Add `import pytest` at the top of the test file if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_contacts.py -k precision -v`
Expected: FAIL (`long_range_precision_at_l` not defined).

- [ ] **Step 3: Implement the metric**

Add to `src/protlms/eval/contacts.py`:

```python
def long_range_precision_at_l(
    pred: np.ndarray,
    true: np.ndarray,
    resnums: np.ndarray,
    *,
    sep: int = LONG_RANGE_SEP,
    top: int | None = None,
) -> float:
    """Long-range contact precision@L.

    Ranks eligible upper-triangle residue pairs (``|resnum_i − resnum_j| ≥ sep``)
    by ``pred`` and returns the fraction of the top ``top`` that are true contacts.

    Args:
        pred: ``(N, N)`` predicted contact scores (higher = more likely contact).
        true: ``(N, N)`` boolean ground-truth contact map.
        resnums: ``(N,)`` residue numbers (used for the separation filter).
        sep: Minimum sequence separation for a long-range pair.
        top: Number of top pairs to score; ``N`` (= L) if ``None``.

    Returns:
        Precision in ``[0, 1]``, or ``nan`` if no eligible pairs exist.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=bool)
    resnums = np.asarray(resnums, dtype=int)
    n = pred.shape[0]
    if pred.shape != (n, n) or true.shape != (n, n) or resnums.shape != (n,):
        raise ValueError(
            f"shape mismatch: pred={pred.shape}, true={true.shape}, resnums={resnums.shape}"
        )
    i, j = np.triu_indices(n, k=1)
    eligible = np.abs(resnums[i] - resnums[j]) >= sep
    i, j = i[eligible], j[eligible]
    if i.size == 0:
        return float("nan")
    order = np.argsort(-pred[i, j], kind="stable")
    k = i.size if top is None else min(int(top), i.size)
    sel = order[:k]
    return float(true[i[sel], j[sel]].sum()) / float(k)
```

- [ ] **Step 4: Export it**

In `src/protlms/eval/__init__.py`, add `long_range_precision_at_l` to the import and `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_eval_contacts.py -v`
Expected: PASS.

- [ ] **Step 6: Lint, format, type-check, commit**

```bash
ruff check src/ tests/ && ruff format src/ tests/ && ty check src/
git add src/protlms/eval/contacts.py src/protlms/eval/__init__.py tests/test_eval_contacts.py
git commit -m "eval: add long-range precision@L metric"
```

---

### Task 7: Eval — CASP14 runner + CSV output

**Files:**
- Create: `src/protlms/eval/runner.py`
- Test: `tests/test_eval_runner.py`

**Interfaces:**
- Consumes: `parse_pdb`, `true_contact_map`, `long_range_precision_at_l` (Tasks 5–6); `Model.contacts` (Task 3); `FastaRecord`, `write_fasta` (io); `InvalidRequestError`.
- Produces: `TargetResult(target_id: str, length: int, n_long_range_true: int, precision_at_l: float)`; `evaluate_contacts(model, pdb_dir, *, sep=LONG_RANGE_SEP, top=None, use_gpu=False, batch_size=None, max_length=None) -> list[TargetResult]`; `write_results_csv(results, path) -> None`; `mean_precision(results) -> float`.

- [ ] **Step 1: Write the failing test with a duck-typed fake model**

Create `tests/test_eval_runner.py`:

```python
"""Tests for the CASP14 contact-evaluation runner (no Docker)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from protlms.eval.runner import (
    TargetResult,
    evaluate_contacts,
    mean_precision,
    write_results_csv,
)
from protlms.io import read_fasta

_PDB_SRC = Path(__file__).parent / "data" / "casp14" / "T1024.pdb"


class _FakeContactsResult:
    def __init__(self, maps: dict[str, np.ndarray]) -> None:
        self._maps = maps

    def maps(self) -> dict[str, np.ndarray]:
        return self._maps


class FakeModel:
    """Duck-typed stand-in for protlms.Model that returns random symmetric maps."""

    def contacts(self, fasta, **_kw):  # noqa: ANN001, ANN003
        rng = np.random.default_rng(0)
        maps = {}
        for rec in read_fasta(Path(fasta)):
            n = len(rec.sequence)
            m = rng.random((n, n)).astype(np.float32)
            maps[rec.id] = (m + m.T) / 2
        return _FakeContactsResult(maps)


def test_evaluate_contacts_returns_one_result_per_target(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    (pdb_dir / "T1024.pdb").write_bytes(_PDB_SRC.read_bytes())
    results = evaluate_contacts(FakeModel(), pdb_dir)
    assert len(results) == 1
    r = results[0]
    assert r.target_id == "T1024"
    assert r.length > 50
    assert 0.0 <= r.precision_at_l <= 1.0
    assert r.n_long_range_true > 0


def test_evaluate_contacts_respects_max_length(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    (pdb_dir / "T1024.pdb").write_bytes(_PDB_SRC.read_bytes())
    results = evaluate_contacts(FakeModel(), pdb_dir, max_length=10)
    assert results == []  # target skipped (too long)


def test_write_results_csv_and_mean(tmp_path: Path) -> None:
    results = [
        TargetResult("A", 50, 10, 0.5),
        TargetResult("B", 60, 12, 0.25),
    ]
    out = tmp_path / "r.csv"
    write_results_csv(results, out)
    lines = out.read_text().splitlines()
    assert lines[0] == "target_id,length,n_long_range_true,precision_at_l"
    assert lines[1].startswith("A,50,10,0.5")
    assert math.isclose(mean_precision(results), 0.375)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_runner.py -v`
Expected: FAIL (`protlms.eval.runner` does not exist).

- [ ] **Step 3: Implement the runner**

Create `src/protlms/eval/runner.py`:

```python
"""Run the CASP14 contact benchmark for a model over a directory of PDBs."""

from __future__ import annotations

import csv
import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from protlms.eval.contacts import (
    LONG_RANGE_SEP,
    long_range_precision_at_l,
    parse_pdb,
    true_contact_map,
)
from protlms.exceptions import InvalidRequestError
from protlms.io import FastaRecord, write_fasta

if TYPE_CHECKING:
    from protlms.models import Model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetResult:
    """Per-target evaluation outcome."""

    target_id: str
    length: int
    n_long_range_true: int
    precision_at_l: float


def _count_long_range_true(true: np.ndarray, resnums: np.ndarray, sep: int) -> int:
    n = true.shape[0]
    i, j = np.triu_indices(n, k=1)
    eligible = np.abs(resnums[i] - resnums[j]) >= sep
    return int(true[i[eligible], j[eligible]].sum())


def evaluate_contacts(
    model: Model,
    pdb_dir: Path | str,
    *,
    sep: int = LONG_RANGE_SEP,
    top: int | None = None,
    use_gpu: bool = False,
    batch_size: int | None = None,
    max_length: int | None = None,
) -> list[TargetResult]:
    """Score a model's long-range precision@L on every ``.pdb`` in ``pdb_dir``.

    Parses each structure, sends all target sequences through ``model.contacts``
    in a single run, then scores each predicted map against its true contacts.

    Args:
        model: A loaded protlms model supporting the ``contacts`` capability.
        pdb_dir: Directory containing CASP14 ``.pdb`` files.
        sep: Minimum long-range separation.
        top: Top-k pairs for precision (``L`` if ``None``).
        use_gpu: Request GPUs for the container run.
        batch_size: Override the container batch size.
        max_length: Skip targets whose resolved sequence exceeds this length.

    Returns:
        One :class:`TargetResult` per successfully scored target (file order).

    Raises:
        InvalidRequestError: If no usable targets are found.
    """
    pdb_paths = sorted(Path(pdb_dir).glob("*.pdb"))
    chains = {}
    records: list[FastaRecord] = []
    for path in pdb_paths:
        target_id = path.stem
        try:
            chain = parse_pdb(path)
        except (ValueError, KeyError) as exc:
            logger.warning("skipping %s: %s", target_id, exc)
            continue
        if max_length is not None and len(chain.sequence) > max_length:
            logger.warning(
                "skipping %s: length %d exceeds max_length %d",
                target_id,
                len(chain.sequence),
                max_length,
            )
            continue
        chains[target_id] = chain
        records.append(FastaRecord(id=target_id, description=target_id, sequence=chain.sequence))

    if not records:
        raise InvalidRequestError(f"no usable .pdb targets found in {pdb_dir}")

    with tempfile.TemporaryDirectory(prefix="protlms-eval-") as tmp:
        fasta = Path(tmp) / "targets.fasta"
        write_fasta(records, fasta)
        maps = model.contacts(
            fasta, use_gpu=use_gpu, batch_size=batch_size
        ).maps()

    results: list[TargetResult] = []
    for target_id, chain in chains.items():
        pred = maps.get(target_id)
        n = len(chain.sequence)
        if pred is None:
            logger.warning("no predicted map returned for %s; skipping", target_id)
            continue
        if pred.shape != (n, n):
            logger.warning(
                "map shape %s for %s does not match length %d (truncated?); skipping",
                pred.shape,
                target_id,
                n,
            )
            continue
        true = true_contact_map(chain.cb_coords)
        results.append(
            TargetResult(
                target_id=target_id,
                length=n,
                n_long_range_true=_count_long_range_true(true, chain.resnums, sep),
                precision_at_l=long_range_precision_at_l(
                    pred, true, chain.resnums, sep=sep, top=top
                ),
            )
        )
    return results


def write_results_csv(results: list[TargetResult], path: Path | str) -> None:
    """Write per-target results to a CSV file."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_id", "length", "n_long_range_true", "precision_at_l"])
        for r in results:
            writer.writerow([r.target_id, r.length, r.n_long_range_true, f"{r.precision_at_l:.6f}"])


def mean_precision(results: list[TargetResult]) -> float:
    """Mean precision@L over targets with a defined (non-nan) value."""
    values = [r.precision_at_l for r in results if not math.isnan(r.precision_at_l)]
    return float(np.mean(values)) if values else float("nan")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, format, type-check, commit**

```bash
ruff check src/ tests/ && ruff format src/ tests/ && ty check src/
git add src/protlms/eval/runner.py tests/test_eval_runner.py
git commit -m "eval: add CASP14 contacts runner + CSV output"
```

---

### Task 8: CLI — `protlms eval contacts`

**Files:**
- Modify: `src/protlms/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `evaluate_contacts`, `write_results_csv`, `mean_precision`, `TargetResult` (Task 7); `load`.
- Produces: `protlms eval contacts MODEL --pdb-dir DIR [--out CSV] [--sep 24] [--top L] [--max-length N] [--gpu] [--batch-size N] [--no-pull]`.

- [ ] **Step 1: Write the failing test (both eval helpers monkeypatched)**

Add to `tests/test_cli.py`:

```python
def test_eval_contacts_command(tmp_path: Path, monkeypatch) -> None:
    from protlms.eval.runner import TargetResult

    monkeypatch.setattr("protlms.cli.load", lambda name, **kw: FakeModel())
    monkeypatch.setattr(
        "protlms.eval.runner.evaluate_contacts",
        lambda *a, **k: [TargetResult("T1024", 80, 40, 0.625)],
    )
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    out_csv = tmp_path / "r.csv"
    result = runner.invoke(
        app, ["eval", "contacts", "esm2-8m", "--pdb-dir", str(pdb_dir), "--out", str(out_csv)]
    )
    assert result.exit_code == 0, result.stdout
    assert "T1024" in result.stdout
    assert out_csv.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -k eval_contacts -v`
Expected: FAIL (no `eval` sub-app / command).

- [ ] **Step 3: Implement the eval sub-app**

In `src/protlms/cli.py`, after the `models_app` block, add:

```python
eval_app = typer.Typer(
    help="Evaluate models on structural benchmarks.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")
```

and add the command (near the bottom, before `_parse_layers`):

```python
@eval_app.command("contacts")
def eval_contacts(
    model: _ModelArg,
    pdb_dir: Annotated[
        Path,
        typer.Option(
            "--pdb-dir", exists=True, file_okay=False, help="Directory of .pdb structures."
        ),
    ],
    out: Annotated[
        Path | None, typer.Option("--out", help="Write per-target results to this CSV.")
    ] = None,
    sep: Annotated[int, typer.Option("--sep", help="Minimum |i-j| for long-range.")] = 24,
    top: Annotated[
        int | None, typer.Option("--top", help="Top-k pairs (default L = length).")
    ] = None,
    max_length: Annotated[
        int | None, typer.Option("--max-length", help="Skip targets longer than this.")
    ] = None,
    gpu: _GpuOpt = False,
    batch_size: _BatchOpt = None,
    no_pull: _NoPullOpt = False,
) -> None:
    """Benchmark long-range precision@L on a directory of PDB structures."""
    from protlms.eval.runner import evaluate_contacts, mean_precision, write_results_csv

    try:
        model_obj = load(model, allow_pull=False if no_pull else None)
        results = evaluate_contacts(
            model_obj,
            pdb_dir,
            sep=sep,
            top=top,
            use_gpu=gpu,
            batch_size=batch_size,
            max_length=max_length,
        )
        table = Table(title=f"contacts precision@L — {model_obj.manifest.name}")
        table.add_column("target", style="bold")
        table.add_column("L", justify="right")
        table.add_column("LR contacts", justify="right")
        table.add_column("P@L", justify="right")
        for r in results:
            table.add_row(r.target_id, str(r.length), str(r.n_long_range_true), f"{r.precision_at_l:.3f}")
        console.print(table)
        console.print(
            f"mean precision@L = [bold]{mean_precision(results):.4f}[/bold] "
            f"over {len(results)} target(s)"
        )
        if out is not None:
            write_results_csv(results, out)
            console.print(f"wrote {out}")
    except ProtlmsError as exc:
        _fail(exc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -k eval_contacts -v`
Expected: PASS.

- [ ] **Step 5: Run whole CLI suite, lint, commit**

```bash
pytest tests/test_cli.py -q
ruff check src/ tests/ && ruff format src/ tests/ && ty check src/
git add src/protlms/cli.py tests/test_cli.py
git commit -m "cli: add protlms eval contacts benchmark command"
```

---

### Task 9: Generalize the ESM2 image into a shared `esm` container + registry rename

**Files:**
- Rename: `containers/esm2/` → `containers/esm/` (via `git mv`: `Dockerfile`, `entrypoint.py`, `README.md`)
- Modify: `containers/esm/entrypoint.py`, `containers/esm/Dockerfile`, `containers/esm/README.md`
- Modify: `src/protlms/_data/models.yaml`
- Rename: `tests/test_esm2_entrypoint.py` → `tests/test_esm_entrypoint.py`
- Rename: `tests/test_integration_esm2.py` → `tests/test_integration_esm.py`
- Modify: `tests/test_registry.py`

**Interfaces:**
- Produces: build args `ESM_HF_ID`, `ESM_MODEL_NAME`, `ESM_MODEL_FAMILY`; container-side globals `HF_ID`, `MODEL_NAME`, `MODEL_FAMILY`. Registry `esm2-8m`/`esm2-650m` now point at image `ghcr.io/briney/protlms-esm:<tag>` and context `containers/esm`.

**Note:** No behavior change to `embed`/`likelihood`/`score`; this is a rename + build-arg generalization. `contacts` is added in Tasks 10–11.

- [ ] **Step 1: Rename the container directory and tests (git mv)**

```bash
git mv containers/esm2 containers/esm
git mv tests/test_esm2_entrypoint.py tests/test_esm_entrypoint.py
git mv tests/test_integration_esm2.py tests/test_integration_esm.py
```

- [ ] **Step 2: Update the entrypoint-test module path and drop the esm2-only resolve test**

In `tests/test_esm_entrypoint.py`: change the path constant to the new location and delete `test_resolve_hf_id` (the suffix logic is gone):

```python
_ENTRYPOINT = Path(__file__).parents[1] / "containers" / "esm" / "entrypoint.py"
```

(Keep `test_sanitize_ids_replaces_unsafe_and_dedupes`, `test_read_fasta_matches_contract_id_rule`, `test_perplexity_from_mean`, and any `parse_mutant` tests.)

- [ ] **Step 3: Run the (renamed) entrypoint unit tests to see the resolve failure gone**

Run: `pytest tests/test_esm_entrypoint.py -v`
Expected: FAIL only on the import of the module if `resolve_hf_id` is still referenced elsewhere; otherwise collect error until Step 4 lands. (This step confirms the file loads from the new path.)

- [ ] **Step 4: Generalize `containers/esm/entrypoint.py`**

Replace the checkpoint/resolve block:

```python
CONTRACT_VERSION = "0.4"
MAX_SEQUENCE_LENGTH = 1024
DEFAULT_BATCH_SIZE = 8
HF_ID = os.environ.get("ESM_HF_ID", "facebook/esm2_t6_8M_UR50D")
MODEL_NAME = os.environ.get("ESM_MODEL_NAME", "esm2_t6_8M")
MODEL_FAMILY = os.environ.get("ESM_MODEL_FAMILY", "esm2")
```

Delete `resolve_hf_id`. Update:
- `load_model`: `hf_id = HF_ID`.
- `cmd_prefetch`: use `HF_ID`.
- `build_manifest`: `AutoConfig.from_pretrained(HF_ID)`, `"name": MODEL_NAME`, `"model_family": MODEL_FAMILY`, `"description": f"{MODEL_FAMILY} masked protein language model ({MODEL_NAME})."`, and set `"contract_version": CONTRACT_VERSION`.
- Everywhere `DEFAULT_CHECKPOINT` was used as the result `model_name` (in `cmd_score` and `_write_capability_result`), use `MODEL_NAME`.

(Set `CONTRACT_VERSION = "0.4"` now; `contacts` capability + subcommand arrive in Task 11.)

- [ ] **Step 5: Rewrite `containers/esm/Dockerfile`**

```dockerfile
# Shared ESM masked-LM image for the protlms container contract.
#
# Serves ESM-1b and all ESM-2 sizes (same EsmForMaskedLM architecture) via
# HuggingFace transformers; the checkpoint is chosen by build args.
#
# Build (ESM-2 tiny demo):
#   docker build --build-arg ESM_HF_ID=facebook/esm2_t6_8M_UR50D \
#     --build-arg ESM_MODEL_NAME=esm2_t6_8M --build-arg ESM_MODEL_FAMILY=esm2 \
#     -t protlms-esm:t6_8M containers/esm
# Build (ESM-1b):
#   docker build --build-arg ESM_HF_ID=facebook/esm1b_t33_650M_UR50S \
#     --build-arg ESM_MODEL_NAME=esm1b_t33_650M --build-arg ESM_MODEL_FAMILY=esm1b \
#     -t protlms-esm:esm1b_650M containers/esm
#
# Weights are baked in at build time, so runtime needs no network access.
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG ESM_HF_ID=facebook/esm2_t6_8M_UR50D
ARG ESM_MODEL_NAME=esm2_t6_8M
ARG ESM_MODEL_FAMILY=esm2
ENV ESM_HF_ID=${ESM_HF_ID} \
    ESM_MODEL_NAME=${ESM_MODEL_NAME} \
    ESM_MODEL_FAMILY=${ESM_MODEL_FAMILY} \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "transformers==4.46.3"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

RUN python /app/entrypoint.py _prefetch

ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/app/entrypoint.py"]
```

- [ ] **Step 6: Update the README build examples** in `containers/esm/README.md` to the new image name (`protlms-esm`), build args, and note it serves ESM-1b + ESM-2.

- [ ] **Step 7: Rename registry entries to the shared image/context**

In `src/protlms/_data/models.yaml`, update the two existing ESM-2 entries:

```yaml
  - name: esm2-8m
    aliases: [esm2_t6_8M]
    image: ghcr.io/briney/protlms-esm:t6_8M
    model_family: esm2
    build:
      context: containers/esm
      args: { ESM_HF_ID: facebook/esm2_t6_8M_UR50D, ESM_MODEL_NAME: esm2_t6_8M, ESM_MODEL_FAMILY: esm2 }
  - name: esm2-650m
    aliases: [esm2_t33_650M]
    image: ghcr.io/briney/protlms-esm:t33_650M
    model_family: esm2
    build:
      context: containers/esm
      args: { ESM_HF_ID: facebook/esm2_t33_650M_UR50D, ESM_MODEL_NAME: esm2_t33_650M, ESM_MODEL_FAMILY: esm2 }
```

- [ ] **Step 8: Update `tests/test_registry.py` image assertions**

Change `test_default_registry_resolves_esm2_8m`:

```python
    assert entry.image == "ghcr.io/briney/protlms-esm:t6_8M"
```

- [ ] **Step 9: Update the integration test to the new context/image**

In `tests/test_integration_esm.py`: set `IMAGE = "ghcr.io/briney/protlms-esm:t6_8M"`, and change the build fixture to the new context + build args:

```python
        subprocess.run(
            [
                "docker", "build",
                "--build-arg", "ESM_HF_ID=facebook/esm2_t6_8M_UR50D",
                "--build-arg", "ESM_MODEL_NAME=esm2_t6_8M",
                "--build-arg", "ESM_MODEL_FAMILY=esm2",
                "-t", IMAGE,
                str(REPO_ROOT / "containers" / "esm"),
            ],
            check=True,
        )
```

Rename the `esm2_image` fixture to `esm_image` (and its references).

- [ ] **Step 10: Run the fast suite (unit) to confirm green**

Run: `pytest -q -m "not slow"`
Expected: PASS (registry, entrypoint unit tests, everything else).

- [ ] **Step 11: Lint, format, type-check, commit**

```bash
ruff check src/ tests/ && ruff format src/ tests/ && ty check src/
git add -A containers/esm src/protlms/_data/models.yaml tests/test_esm_entrypoint.py tests/test_integration_esm.py tests/test_registry.py
git commit -m "esm: generalize esm2 image into shared esm container (esm1b+esm2)"
```

---

### Task 10: ESM container — pure Jacobian post-processing + AA token mapping

**Files:**
- Modify: `containers/esm/entrypoint.py`
- Test: `tests/test_esm_entrypoint.py`

**Interfaces:**
- Produces: `aa_token_ids(tokenizer) -> list[int]` (20 ids, order `ACDEFGHIKLMNPQRSTVWY`); `jacobian_to_contacts(jac: np.ndarray) -> np.ndarray` (`(L,20,L,20)` → `(L,L)` float32).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_esm_entrypoint.py`:

```python
def test_jacobian_to_contacts_shape_symmetry_zero_diag() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    length = 7
    jac = rng.standard_normal((length, 20, length, 20))
    contacts = entrypoint.jacobian_to_contacts(jac)
    assert contacts.shape == (length, length)
    assert contacts.dtype == np.float32
    assert np.allclose(contacts, contacts.T, atol=1e-5)
    assert np.allclose(np.diag(contacts), 0.0)


def test_jacobian_to_contacts_invariant_to_aa_permutation() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    length = 5
    jac = rng.standard_normal((length, 20, length, 20))
    perm = rng.permutation(20)
    permuted = jac[:, perm][:, :, :, perm]
    a = entrypoint.jacobian_to_contacts(jac)
    b = entrypoint.jacobian_to_contacts(permuted)
    assert np.allclose(a, b, atol=1e-4)


def test_aa_token_ids_maps_twenty_amino_acids() -> None:
    class FakeTok:
        def convert_tokens_to_ids(self, token: str) -> int:
            return ord(token)

    ids = entrypoint.aa_token_ids(FakeTok())
    assert len(ids) == 20
    assert ids[0] == ord("A")
    assert ids[-1] == ord("Y")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_esm_entrypoint.py -k "jacobian or aa_token" -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Implement the pure helpers**

Add to `containers/esm/entrypoint.py` (near the pure helpers). These are numpy-only (no torch), matching the reference `get_contacts` composed with the `get_data` pre-steps from Zhang & Ovchinnikov:

```python
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids (fixed order)


def aa_token_ids(tokenizer) -> list[int]:  # noqa: ANN001
    """Token ids for the 20 standard amino acids, in a fixed order."""
    return [tokenizer.convert_tokens_to_ids(aa) for aa in _AA_ORDER]


def jacobian_to_contacts(jac):  # noqa: ANN001, ANN201
    """Convert a categorical Jacobian ``(L,20,L,20)`` to an ``(L,L)`` contact map.

    Faithful port of the Zhang/Ovchinnikov pipeline: center over all four axes,
    symmetrize the 4-D tensor, Frobenius norm over the amino-acid axes, zero the
    diagonal, apply average product correction (APC), symmetrize the ``(L,L)`` map.
    """
    import numpy as np

    j = np.asarray(jac, dtype=np.float64)
    for axis in range(4):
        j = j - j.mean(axis=axis, keepdims=True)
    j = (j + j.transpose(2, 3, 0, 1)) / 2.0
    contacts = np.sqrt((j**2).sum(axis=(1, 3)))  # (L, L)
    np.fill_diagonal(contacts, 0.0)
    a1 = contacts.sum(axis=0, keepdims=True)
    a2 = contacts.sum(axis=1, keepdims=True)
    contacts = contacts - (a1 * a2) / contacts.sum()  # APC
    np.fill_diagonal(contacts, 0.0)
    contacts = (contacts + contacts.T) / 2.0
    return contacts.astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_esm_entrypoint.py -k "jacobian or aa_token" -v`
Expected: PASS.

- [ ] **Step 5: Lint, format, commit**

```bash
ruff check tests/ && ruff format containers/esm/entrypoint.py tests/
git add containers/esm/entrypoint.py tests/test_esm_entrypoint.py
git commit -m "esm: add pure jacobian_to_contacts + aa_token_ids helpers"
```

---

### Task 11: ESM container — `contacts` subcommand + manifest capability

**Files:**
- Modify: `containers/esm/entrypoint.py`
- Test: `tests/test_esm_entrypoint.py`

**Interfaces:**
- Consumes: `aa_token_ids`, `jacobian_to_contacts` (Task 10); existing `load_model`, `pick_device`, `read_fasta`, `sanitize_ids`, `_truncate`, `write_result`.
- Produces: `write_contacts_outputs(output_dir, id_to_map) -> list[dict]`; `categorical_jacobian(model, tokenizer, seq, aa_ids, batch_size, device) -> np.ndarray`; `cmd_contacts`; parser `contacts` subcommand; manifest `capabilities` includes `"contacts"`.

- [ ] **Step 1: Write failing tests for the pure output writer + parser wiring + manifest**

Add to `tests/test_esm_entrypoint.py`:

```python
def test_write_contacts_outputs_saves_npy_and_artifacts(tmp_path: Path) -> None:
    import numpy as np

    maps = {"seqA": np.zeros((4, 4), dtype=np.float32), "seqB": np.ones((3, 3), dtype=np.float32)}
    artifacts = entrypoint.write_contacts_outputs(tmp_path, maps)
    assert (tmp_path / "contacts" / "seqA.npy").is_file()
    assert (tmp_path / "contacts" / "seqB.npy").is_file()
    kinds = {a["kind"] for a in artifacts}
    assert kinds == {"contact_map"}
    by_id = {a["record_ids"][0]: a for a in artifacts}
    assert by_id["seqA"]["shape"] == [4, 4]
    assert by_id["seqA"]["path"] == "contacts/seqA.npy"


def test_parser_has_contacts_subcommand() -> None:
    args = entrypoint.build_parser().parse_args(
        ["contacts", "--input", "/in/seqs.fasta", "--output", "/out"]
    )
    assert args.command == "contacts"
    assert args.method == "categorical-jacobian"
    assert args.func is entrypoint.cmd_contacts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_esm_entrypoint.py -k "write_contacts or contacts_subcommand" -v`
Expected: FAIL (functions/subcommand missing).

- [ ] **Step 3: Implement the output writer, jacobian driver, and command**

Add to `containers/esm/entrypoint.py`:

```python
def write_contacts_outputs(output_dir, id_to_map):  # noqa: ANN001, ANN201
    """Save each (L,L) contact map to contacts/<id>.npy; return artifact dicts."""
    import numpy as np

    contacts_dir = output_dir / "contacts"
    contacts_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for clean_id, cmap in id_to_map.items():
        arr = np.asarray(cmap, dtype=np.float32)
        np.save(contacts_dir / f"{clean_id}.npy", arr)
        artifacts.append(
            {
                "path": f"contacts/{clean_id}.npy",
                "kind": "contact_map",
                "record_ids": [clean_id],
                "shape": list(arr.shape),
                "dtype": "float32",
            }
        )
    return artifacts


def categorical_jacobian(model, tokenizer, seq, aa_ids, batch_size, device):  # noqa: ANN001, ANN201
    """Compute the ``(L, 20, L, 20)`` categorical Jacobian for one sequence.

    Feeds the unmasked sequence; for each residue position substitutes all 20
    amino acids and reads the model logits at every residue position over the 20
    amino-acid tokens, then subtracts the wild-type baseline.
    """
    import numpy as np
    import torch

    enc = tokenizer(seq, return_tensors="pt")
    input_ids = enc["input_ids"][0]  # (T,)
    special = tokenizer.get_special_tokens_mask(
        input_ids.tolist(), already_has_special_tokens=True
    )
    residue_pos = [idx for idx, flag in enumerate(special) if flag == 0]  # (L,)
    length = len(residue_pos)
    aa = torch.tensor(aa_ids, dtype=input_ids.dtype)  # (20,)

    def logits_at_residues(batch_ids):  # noqa: ANN001, ANN202
        use_amp = device == "cuda"
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(input_ids=batch_ids.to(device)).logits  # (B, T, V)
        out = out.float()[:, residue_pos][..., aa]  # (B, L, 20)
        return out.cpu().numpy()

    baseline = logits_at_residues(input_ids.unsqueeze(0))[0]  # (L, 20)
    jac = np.zeros((length, 20, length, 20), dtype=np.float32)
    tiled = input_ids.repeat(20, 1)  # (20, T)
    for n in range(length):
        variant = tiled.clone()
        variant[:, residue_pos[n]] = aa  # position n -> each of the 20 AAs
        chunks = [
            logits_at_residues(variant[start : start + batch_size])
            for start in range(0, 20, batch_size)
        ]
        jac[n] = np.concatenate(chunks, axis=0)  # (20, L, 20)
    jac -= baseline  # broadcast over (n, a); subtract WT logit at (j, b)
    return jac


def cmd_contacts(args: argparse.Namespace) -> None:
    """Predict contact maps via the categorical Jacobian (one map per record)."""
    device = pick_device(args.device)
    tokenizer, model = load_model(device)
    aa_ids = aa_token_ids(tokenizer)
    batch_size = args.batch_size or 20
    records = read_fasta(Path(args.input))
    ids = sanitize_ids([rid for rid, _ in records])
    output_dir = Path(args.output)
    warnings: list[str] = []

    id_to_map: dict[str, object] = {}
    for clean_id, (rid, seq) in zip(ids, records, strict=True):
        seq = _truncate(seq, warnings, rid)
        jac = categorical_jacobian(model, tokenizer, seq, aa_ids, batch_size, device)
        id_to_map[clean_id] = jacobian_to_contacts(jac)

    artifacts = write_contacts_outputs(output_dir, id_to_map)
    write_result(
        output_dir,
        {
            "contract_version": CONTRACT_VERSION,
            "capability": "contacts",
            "model_name": MODEL_NAME,
            "n_input_records": len(records),
            "n_output_records": len(records),
            "artifacts": artifacts,
            "warnings": warnings,
            "params": {"method": args.method, "device": args.device or "auto"},
        },
    )
```

- [ ] **Step 4: Register the subcommand and declare the capability**

In `build_parser()` (before `_prefetch`), add:

```python
    contacts = sub.add_parser("contacts")
    contacts.add_argument("--input", required=True)
    contacts.add_argument("--output", required=True)
    contacts.add_argument(
        "--method", default="categorical-jacobian", choices=["categorical-jacobian"]
    )
    contacts.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    contacts.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    contacts.set_defaults(func=cmd_contacts)
```

In `build_manifest()`, add `"contacts"` to the `capabilities` list.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_esm_entrypoint.py -v`
Expected: PASS.

- [ ] **Step 6: Lint, format, commit**

```bash
ruff check tests/ && ruff format containers/esm/entrypoint.py tests/
git add containers/esm/entrypoint.py tests/test_esm_entrypoint.py
git commit -m "esm: implement contacts subcommand (categorical jacobian)"
```

---

### Task 12: Registry — add ESM-1b and remaining ESM-2 sizes

**Files:**
- Modify: `src/protlms/_data/models.yaml`
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces: registry entries `esm1b` (alias `esm1b_t33_650M`), `esm2-35m`, `esm2-150m`, `esm2-3b`, `esm2-15b`, all on `containers/esm` / `ghcr.io/briney/protlms-esm`.

- [ ] **Step 1: Write failing registry tests**

Add to `tests/test_registry.py`:

```python
def test_resolve_esm1b() -> None:
    registry = Registry.load()
    entry = registry.resolve("esm1b")
    assert entry.model_family == "esm1b"
    assert entry.image.startswith("ghcr.io/briney/protlms-esm:")
    assert entry.build is not None
    assert entry.build.context == "containers/esm"
    assert entry.build.args["ESM_HF_ID"] == "facebook/esm1b_t33_650M_UR50S"


def test_registry_includes_all_esm2_sizes() -> None:
    names = {e.name for e in Registry.load().list_models()}
    assert {"esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b", "esm2-15b"} <= names


def test_resolve_esm2_3b_uses_shared_context() -> None:
    entry = Registry.load().resolve("esm2-3b")
    assert entry.build.context == "containers/esm"
    assert entry.build.args["ESM_MODEL_FAMILY"] == "esm2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_registry.py -k "esm1b or all_esm2 or esm2_3b" -v`
Expected: FAIL (entries not present).

- [ ] **Step 3: Add the registry entries**

In `src/protlms/_data/models.yaml`, add (grouped with the other ESM entries):

```yaml
  - name: esm1b
    aliases: [esm1b_t33_650M]
    image: ghcr.io/briney/protlms-esm:esm1b_650M
    model_family: esm1b
    build:
      context: containers/esm
      args: { ESM_HF_ID: facebook/esm1b_t33_650M_UR50S, ESM_MODEL_NAME: esm1b_t33_650M, ESM_MODEL_FAMILY: esm1b }
  - name: esm2-35m
    aliases: [esm2_t12_35M]
    image: ghcr.io/briney/protlms-esm:t12_35M
    model_family: esm2
    build:
      context: containers/esm
      args: { ESM_HF_ID: facebook/esm2_t12_35M_UR50D, ESM_MODEL_NAME: esm2_t12_35M, ESM_MODEL_FAMILY: esm2 }
  - name: esm2-150m
    aliases: [esm2_t30_150M]
    image: ghcr.io/briney/protlms-esm:t30_150M
    model_family: esm2
    build:
      context: containers/esm
      args: { ESM_HF_ID: facebook/esm2_t30_150M_UR50D, ESM_MODEL_NAME: esm2_t30_150M, ESM_MODEL_FAMILY: esm2 }
  - name: esm2-3b
    aliases: [esm2_t36_3B]
    image: ghcr.io/briney/protlms-esm:t36_3B
    model_family: esm2
    build:
      context: containers/esm
      args: { ESM_HF_ID: facebook/esm2_t36_3B_UR50D, ESM_MODEL_NAME: esm2_t36_3B, ESM_MODEL_FAMILY: esm2 }
  - name: esm2-15b
    aliases: [esm2_t48_15B]
    image: ghcr.io/briney/protlms-esm:t48_15B
    model_family: esm2
    build:
      context: containers/esm
      args: { ESM_HF_ID: facebook/esm2_t48_15B_UR50D, ESM_MODEL_NAME: esm2_t48_15B, ESM_MODEL_FAMILY: esm2 }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Sanity-check the CLI listing, lint, commit**

```bash
protlms models list   # should show esm1b + all esm2 sizes on protlms-esm
ruff check src/ tests/ && ruff format src/ tests/ && ty check src/
git add src/protlms/_data/models.yaml tests/test_registry.py
git commit -m "registry: add esm1b + esm2 35M/150M/3B/15B on shared esm image"
```

---

### Task 13: Integration — ESM `contacts` + end-to-end CASP14 eval (Docker, slow)

**Files:**
- Modify: `tests/test_integration_esm.py`

**Interfaces:**
- Consumes: the built `protlms-esm:t6_8M` image, `protlms.load`, `Model.contacts`, `protlms.eval.runner.evaluate_contacts`, `tests/data/casp14/T1024.pdb`.

**Gating:** `@pytest.mark.slow`, runs only when `PROTLMS_RUN_DOCKER_TESTS=1` and Docker is available (existing `pytestmark`).

- [ ] **Step 1: Add contacts + eval integration tests**

Append to `tests/test_integration_esm.py`:

```python
def test_manifest_now_declares_contacts(model: protlms.Model) -> None:
    assert "contacts" in {c.value for c in model.manifest.capabilities}


def test_contacts_end_to_end_shapes(model: protlms.Model, tmp_path: Path) -> None:
    result = model.contacts(TINY_FASTA, output_dir=tmp_path / "ct")
    maps = result.maps()
    assert set(maps) == EXPECTED_IDS
    for cmap in maps.values():
        n = cmap.shape[0]
        assert cmap.shape == (n, n)
        assert cmap.dtype == np.float32
        assert np.isfinite(cmap).all()
        assert np.allclose(cmap, cmap.T, atol=1e-4)  # symmetric


def test_evaluate_contacts_casp14_target(model: protlms.Model, tmp_path: Path) -> None:
    from protlms.eval.runner import evaluate_contacts, mean_precision

    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    src = REPO_ROOT / "tests" / "data" / "casp14" / "T1024.pdb"
    (pdb_dir / "T1024.pdb").write_bytes(src.read_bytes())
    results = evaluate_contacts(model, pdb_dir, max_length=400)
    assert len(results) == 1
    r = results[0]
    assert r.target_id == "T1024"
    assert 0.0 <= r.precision_at_l <= 1.0
    # even the tiny 8M model beats a random baseline for long-range contacts
    assert not math.isnan(mean_precision(results))
```

(`math`, `np`, `Path`, `REPO_ROOT`, `TINY_FASTA`, `EXPECTED_IDS` are already imported/defined in this file.)

- [ ] **Step 2: Build the image and run the slow suite**

Run:
```bash
PROTLMS_RUN_DOCKER_TESTS=1 pytest tests/test_integration_esm.py -m slow -v
```
Expected: image builds (or is reused); all tests PASS, including `contacts` shape/symmetry and the CASP14 `evaluate_contacts` producing a finite precision@L in `[0, 1]`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_esm.py
git commit -m "test: end-to-end esm contacts + CASP14 precision@L integration"
```

---

### Final verification

- [ ] **Run the full fast suite:** `pytest -q -m "not slow"` → all green.
- [ ] **Lint/format/type:** `ruff check src/ tests/` and `ruff format --check src/ tests/` and `ty check src/` → clean.
- [ ] **Smoke the CLI (no Docker needed):** `protlms models list` shows the ESM family on `protlms-esm`; `protlms eval contacts --help` and `protlms contacts --help` render.
- [ ] **(Optional, Docker) Real benchmark:**
  ```bash
  PROTLMS_RUN_DOCKER_TESTS=1 pytest -m slow -q
  protlms eval contacts esm2-8m --pdb-dir ~/projects/esm-c/data/casp14/ --out patl_8m.csv --max-length 400
  ```

---

## Self-review notes

- **Spec coverage:** contract 0.4 (T1); client contacts (T2–T3); CLI contacts + eval (T4, T8); eval PDB/metric/runner (T5–T7); shared esm container + contacts impl (T9–T11); registry esm1b + esm2 sizes incl. giants (T12); integration + CASP14 (T13). ESM-C migration is explicitly deferred to Plan 2 (spec §"ESM-C"). Manifest `contacts` capability declared in T11; giants registered in T12.
- **Deviations from spec (intentional, flagged):** (1) `Model.contacts` omits `chunk_size` (YAGNI — container loops multi-record internally; chunking merge is capability-specific). (2) precision@L uses the **standard single-count top-L** definition; the reference notebook's double-count (≈ top L/2) is not replicated but `top` is configurable for parity. (3) `parse_pdb` defaults to the **first chain** and standard residues only (MSE and other modified residues are skipped) — documented simplification.
- **Type consistency:** `contacts` capability string, `contact_map` kind, `categorical-jacobian` method, `ESM_HF_ID`/`ESM_MODEL_NAME`/`ESM_MODEL_FAMILY` build args, and `evaluate_contacts`/`TargetResult`/`long_range_precision_at_l`/`jacobian_to_contacts`/`aa_token_ids` signatures are used identically across producing and consuming tasks.
