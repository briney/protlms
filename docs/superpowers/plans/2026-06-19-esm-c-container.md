# ESM-C Container Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a contract-compliant `containers/esm-c/` image (plus two registry entries and tests) wrapping the ESM-C masked protein language model for `embed`, `likelihood`, and `score` — with zero client production-code changes.

**Architecture:** ESM-C is a bidirectional masked LM (same capability surface as ESM2). It ships as a standalone Docker image whose entrypoint implements the protlms container contract using EvolutionaryScale's native `esm` SDK (`ESMC.from_pretrained`). The client never changes — it resolves a new registry name to the image and speaks the existing contract.

**Tech Stack:** Python 3.12, the `esm` package (`esm==3.2.3`, EvolutionaryScale), PyTorch (pulled transitively), Docker. The client side touches only `models.yaml` (YAML) and a pytest test.

## Global Constraints

These apply to **every** task. Exact values copied from the design spec (`docs/superpowers/specs/2026-06-19-esm-c-container-design.md`):

- **Contract version:** `"0.3"`. No contract changes — `docs/CONTRACT.md` and `src/protlms/contract.py` are NOT edited.
- **No client production-code changes.** Only `src/protlms/_data/models.yaml` (registry data) and tests are touched on the client side. `contract.py`, `models.py`, `io.py`, `cli.py`, `runner.py`, `registry.py` are untouched.
- **Capabilities:** `embed`, `likelihood`, `score`. No `generate`.
- **Backend:** native `esm` SDK only (`from esm.models.esmc import ESMC`). Not the transformers/ESM++ port.
- **Checkpoints:** `esmc_300m` (default + tested) and `esmc_600m` (wired + buildable, not in the routine test path). Checkpoint strings are exactly `"esmc_300m"` / `"esmc_600m"`.
- **`use_flash_attn=False`** always — keeps the image CPU-buildable and CPU-runnable with no flash-attn dependency.
- **Base image:** `python:3.12-slim-bookworm`. The `esm` package requires Python `>=3.12,<3.13`, so ESM2's `pytorch/pytorch:2.5.1` base (Python 3.11) cannot be reused.
- **`max_sequence_length = 2048`** (documented ceiling; longer inputs truncated with a warning, as in ESM2). **`default_batch_size = 8`**.
- **Manifest dims come from a checkpoint-keyed table** (`_MODEL_INFO`), keeping `manifest` model-load-free so `protlms.load()` stays fast. (Refinement of the spec's "derive from model": the checkpoint name pins the architecture, so the table is equally drift-proof and far cheaper.) 300M → `embedding_dim 960`, `num_layers 30`; 600M → `1152`, `36`.
- **Embed layer support:** final layer only (`--layers -1`, the client default). Any other layer index → a structured `InvalidInput` error. Documented in the README.
- **Per-residue / pooling layout:** ESM-C tokenization is BOS at index 0, residue `i` (1-indexed) at token index `i`, EOS last — identical to ESM2. Strip BOS/EOS for per-residue; `cls` pooling = the BOS vector.
- **Standalone container code:** `containers/esm-c/entrypoint.py` duplicates pure helpers from ESM2 by design (each container is self-contained; same pattern as `containers/progen2/`). Heavy imports (`torch`, `esm`) live **inside functions** so pure helpers unit-test without the ML stack.
- **Quality gates (run before each commit that touches `src/`/`tests/`):** `ruff check src/ tests/`, `ruff format src/ tests/`, `ty check src/`. The container entrypoint under `containers/` follows the same style but is not part of the package; still run `ruff format` on it.
- **Commit messages:** `<component>: <what changed and why>`, imperative.

---

### Task 1: Registry entries for ESM-C (client side, no Docker)

Adds the two ESM-C registry entries and a test proving they resolve. This is the only client-side change and is independently shippable.

**Files:**
- Modify: `src/protlms/_data/models.yaml` (append two entries)
- Test: `tests/test_registry.py` (append one test)

**Interfaces:**
- Consumes: `protlms.registry.Registry.load()` / `.resolve(name)` → `ModelEntry(name, aliases, image, model_family)` (existing).
- Produces: resolvable names `esm-c-300m` (alias `esmc_300m`) → image `protlms-esm-c:300m`, and `esm-c-600m` (alias `esmc_600m`) → image `protlms-esm-c:600m`, both `model_family="esm-c"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_resolve_esm_c() -> None:
    registry = Registry.load()
    e300 = registry.resolve("esm-c-300m")
    assert e300.image == "protlms-esm-c:300m"
    assert e300.model_family == "esm-c"
    assert registry.resolve("esmc_300m") == e300
    e600 = registry.resolve("esm-c-600m")
    assert e600.image == "protlms-esm-c:600m"
    assert e600.model_family == "esm-c"
    assert registry.resolve("esmc_600m") == e600
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py::test_resolve_esm_c -v`
Expected: FAIL — `ModelNotFoundError: unknown model 'esm-c-300m'`.

- [ ] **Step 3: Add the registry entries**

Append to `src/protlms/_data/models.yaml` (after the `progen2-small` entry):

```yaml
  - name: esm-c-300m
    aliases: [esmc_300m]
    image: protlms-esm-c:300m
    model_family: esm-c
  - name: esm-c-600m
    aliases: [esmc_600m]
    image: protlms-esm-c:600m
    model_family: esm-c
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (all registry tests, including the new one).

- [ ] **Step 5: Commit**

```bash
git add src/protlms/_data/models.yaml tests/test_registry.py
git commit -m "registry: add esm-c-300m/600m entries"
```

---

### Task 2: ESM-C entrypoint (contract CLI) + pure-helper unit tests

Creates the full standalone entrypoint. The pure helpers and the manifest are proven now by unit tests (no torch/esm needed); the model-backed subcommands use lazy imports and are proven later by the Docker integration test (Task 4).

**Files:**
- Create: `containers/esm-c/entrypoint.py`
- Test: `tests/test_esmc_entrypoint.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Tasks 3–4): a CLI module exposing subcommands `manifest`, `embed`, `likelihood`, `score`, `_prefetch`; pure helpers `sanitize_ids(list[str])->list[str]`, `read_fasta(Path)->list[tuple[str,str]]`, `parse_mutant(str)->list[tuple[str,int,str]]`, `perplexity_from_mean(float)->float`, `_truncate(str,list[str],str)->str`; and `build_manifest()->dict`. Env var `ESMC_CHECKPOINT` selects the checkpoint (default `"esmc_300m"`).

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_esmc_entrypoint.py`:

```python
"""Unit tests for the ESM-C entrypoint's torch/esm-free helpers and manifest."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENTRYPOINT = Path(__file__).parents[1] / "containers" / "esm-c" / "entrypoint.py"


def _load():
    spec = importlib.util.spec_from_file_location("esmc_entrypoint", _ENTRYPOINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entrypoint = _load()


def test_sanitize_ids_dedupes_collisions() -> None:
    assert entrypoint.sanitize_ids(["a/b", "a:b", "ok"]) == ["a_b", "a_b__1", "ok"]


def test_read_fasta_parses_records(tmp_path: Path) -> None:
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">one desc\nMAGIC\n>two\nACDE\nFG\n")
    assert entrypoint.read_fasta(fasta) == [("one", "MAGIC"), ("two", "ACDEFG")]


@pytest.mark.parametrize(
    ("mutant", "expected"),
    [
        ("A24G", [("A", 24, "G")]),
        ("A24G:T56S", [("A", 24, "G"), ("T", 56, "S")]),
    ],
)
def test_parse_mutant_valid(mutant: str, expected: list[tuple[str, int, str]]) -> None:
    assert entrypoint.parse_mutant(mutant) == expected


def test_parse_mutant_invalid_raises() -> None:
    with pytest.raises(ValueError):
        entrypoint.parse_mutant("not-a-mutant")


def test_perplexity_from_mean() -> None:
    assert entrypoint.perplexity_from_mean(0.0) == pytest.approx(1.0)
    assert entrypoint.perplexity_from_mean(-1.0) == pytest.approx(2.718281828, rel=1e-6)


def test_truncate_warns_and_clips() -> None:
    warnings: list[str] = []
    long_seq = "A" * (entrypoint.MAX_SEQUENCE_LENGTH + 5)
    out = entrypoint._truncate(long_seq, warnings, "big")
    assert len(out) == entrypoint.MAX_SEQUENCE_LENGTH
    assert warnings and "truncated" in warnings[0]


def test_build_manifest_300m(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entrypoint, "DEFAULT_CHECKPOINT", "esmc_300m")
    m = entrypoint.build_manifest()
    assert m["contract_version"] == "0.3"
    assert m["model_family"] == "esm-c"
    assert m["name"] == "esmc_300m"
    assert m["embedding_dim"] == 960
    assert m["num_layers"] == 30
    assert set(m["capabilities"]) == {"embed", "likelihood", "score"}
    assert set(m["pooling_modes"]) == {"mean", "cls", "none"}


def test_build_manifest_600m(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entrypoint, "DEFAULT_CHECKPOINT", "esmc_600m")
    m = entrypoint.build_manifest()
    assert m["embedding_dim"] == 1152
    assert m["num_layers"] == 36
    assert m["min_gpu_memory_gb"] == 4.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_esmc_entrypoint.py -v`
Expected: FAIL at collection — `FileNotFoundError`/load error because `containers/esm-c/entrypoint.py` does not exist yet.

- [ ] **Step 3: Create the entrypoint**

Create `containers/esm-c/entrypoint.py` with this exact content:

```python
#!/usr/bin/env python
"""Contract-compliant entrypoint for the ESM-C model image.

Implements the protlms container contract (see docs/CONTRACT.md) for the ESM-C
masked protein language model via EvolutionaryScale's native ``esm`` SDK. Exposes
the ``manifest``, ``embed``, ``likelihood``, and ``score`` subcommands plus a
hidden ``_prefetch`` used at build time to bake weights into the image.

Heavy imports (``torch``, ``esm``) happen inside the functions that need them, so
the pure helpers can be unit-tested without the ML stack installed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

CONTRACT_VERSION = "0.3"
MAX_SEQUENCE_LENGTH = 2048
DEFAULT_BATCH_SIZE = 8
DEFAULT_CHECKPOINT = os.environ.get("ESMC_CHECKPOINT", "esmc_300m")

# Architecture facts keyed by checkpoint name. Keeping these here lets `manifest`
# stay model-load-free (the checkpoint name pins the architecture, so there is no
# drift risk): embedding_dim is the model width, num_layers the transformer depth.
_MODEL_INFO: dict[str, dict[str, object]] = {
    "esmc_300m": {"embedding_dim": 960, "num_layers": 30, "min_gpu_memory_gb": None},
    "esmc_600m": {"embedding_dim": 1152, "num_layers": 36, "min_gpu_memory_gb": 4.0},
}

_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_MUTANT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


# --- pure helpers (unit-testable without torch/esm) ------------------------


def sanitize_ids(ids: list[str]) -> list[str]:
    """Sanitize record ids for filenames/keys, de-duplicating collisions."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in ids:
        clean = _ID_SAFE.sub("_", raw) or "seq"
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}__{seen[clean]}"
        else:
            seen[clean] = 0
        out.append(clean)
    return out


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into ``(id, sequence)`` tuples."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks).upper()))
            header = line[1:].split(maxsplit=1)[0] if line[1:].split() else line[1:]
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks).upper()))
    return records


def perplexity_from_mean(mean_log_likelihood: float) -> float:
    """Pseudo-perplexity from a mean pseudo-log-likelihood."""
    import math

    return math.exp(-mean_log_likelihood)


def parse_mutant(mutant: str) -> list[tuple[str, int, str]]:
    """Parse a mutation string like ``A24G`` or ``A24G:T56S`` (1-indexed)."""
    subs: list[tuple[str, int, str]] = []
    for token in mutant.split(":"):
        match = _MUTANT_RE.match(token.strip())
        if not match:
            raise ValueError(f"invalid mutation token {token!r}")
        subs.append((match.group(1).upper(), int(match.group(2)), match.group(3).upper()))
    return subs


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


def _truncate(seq: str, warnings: list[str], record_id: str) -> str:
    if len(seq) > MAX_SEQUENCE_LENGTH:
        warnings.append(f"sequence {record_id!r} truncated to {MAX_SEQUENCE_LENGTH} residues")
        return seq[:MAX_SEQUENCE_LENGTH]
    return seq


# --- contract I/O ----------------------------------------------------------


def write_result(output_dir: Path, payload: dict) -> None:
    """Write the ``result.json`` success summary."""
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2))


def emit_error_and_exit(error_type: str, message: str, **details: str) -> None:
    """Write a structured ContainerError to stderr and exit non-zero."""
    error = {
        "contract_version": CONTRACT_VERSION,
        "error_type": error_type,
        "message": message,
        "details": {k: str(v) for k, v in details.items()},
    }
    print(json.dumps(error), file=sys.stderr)
    raise SystemExit(1)


def build_manifest() -> dict:
    """Build the manifest dict from the checkpoint-keyed architecture table."""
    info = _MODEL_INFO[DEFAULT_CHECKPOINT]
    return {
        "contract_version": CONTRACT_VERSION,
        "name": DEFAULT_CHECKPOINT,
        "version": "1.0.0",
        "description": f"ESM-C masked protein language model ({DEFAULT_CHECKPOINT}).",
        "model_family": "esm-c",
        "capabilities": ["embed", "likelihood", "score"],
        "embedding_dim": info["embedding_dim"],
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "pooling_modes": ["mean", "cls", "none"],
        "num_layers": info["num_layers"],
        "min_gpu_memory_gb": info["min_gpu_memory_gb"],
        "default_batch_size": DEFAULT_BATCH_SIZE,
    }


# --- model helpers ---------------------------------------------------------


def pick_device(requested: str | None) -> str:
    """Choose the torch device, validating an explicit ``cuda`` request."""
    import torch

    available = "cuda" if torch.cuda.is_available() else "cpu"
    if requested in (None, "auto"):
        return available
    if requested == "cuda" and available != "cuda":
        emit_error_and_exit("DeviceUnavailable", "cuda requested but no GPU is available")
    return requested


def load_model(device: str):  # noqa: ANN201 - returns an ESMC module
    """Load the ESM-C model for the configured checkpoint (flash-attn disabled)."""
    import torch
    from esm.models.esmc import ESMC

    model = ESMC.from_pretrained(
        DEFAULT_CHECKPOINT,
        device=torch.device(device),
        use_flash_attn=False,
    )
    model.eval()
    return model


def _encode_ids(model, seq: str):  # noqa: ANN001, ANN202
    """Tokenize a sequence to a 1-D token-id tensor (incl BOS/EOS) on CPU."""
    from esm.sdk.api import ESMProtein

    tensor = model.encode(ESMProtein(sequence=seq))
    return tensor.sequence.cpu()  # (T,) long


def _embed_one(model, seq: str, device: str):  # noqa: ANN001, ANN202
    """Return (per-residue (L, D), cls (D,)) float32 arrays for one sequence."""
    import numpy as np
    import torch

    ids = _encode_ids(model, seq).unsqueeze(0).to(device)  # (1, T)
    with torch.no_grad():
        out = model(sequence_tokens=ids)
    emb = out.embeddings[0].float().cpu().numpy()  # (T, D)
    residue = emb[1 : 1 + len(seq)].astype(np.float32)  # strip BOS/EOS
    cls_vec = emb[0].astype(np.float32)
    return residue, cls_vec


def _pseudo_log_likelihood(model, seq: str, batch_size: int, device: str) -> float:  # noqa: ANN001
    """Masked-marginal pseudo-log-likelihood: O(L) masked forward passes."""
    import torch

    ids = _encode_ids(model, seq)  # (T,)
    mask_id = model.tokenizer.mask_token_id
    positions = list(range(1, len(ids) - 1))  # residues only
    total = 0.0
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        batch = ids.repeat(len(chunk), 1)  # (b, T)
        for row, pos in enumerate(chunk):
            batch[row, pos] = mask_id
        with torch.no_grad():
            out = model(sequence_tokens=batch.to(device))
        log_probs = torch.log_softmax(out.sequence_logits.float(), dim=-1)
        for row, pos in enumerate(chunk):
            total += float(log_probs[row, pos, ids[pos]])
    return total


def _masked_position_logprobs(model, seq, positions, batch_size, device):  # noqa: ANN001
    """Map each 1-indexed position to its masked log-softmax vector over the vocab."""
    import torch

    ids = _encode_ids(model, seq)
    mask_id = model.tokenizer.mask_token_id
    ordered = sorted(positions)
    out: dict[int, object] = {}
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        batch = ids.repeat(len(chunk), 1)
        for row, pos in enumerate(chunk):
            batch[row, pos] = mask_id
        with torch.no_grad():
            res = model(sequence_tokens=batch.to(device))
        log_probs = torch.log_softmax(res.sequence_logits.float(), dim=-1)
        for row, pos in enumerate(chunk):
            out[pos] = log_probs[row, pos].cpu().numpy()
    return out


def _wt_position_logprobs(model, seq, positions, device):  # noqa: ANN001
    """Map each 1-indexed position to its unmasked (WT-context) log-softmax vector."""
    import torch

    ids = _encode_ids(model, seq).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(sequence_tokens=ids)
    log_probs = torch.log_softmax(out.sequence_logits.float(), dim=-1)[0].cpu().numpy()
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


# --- subcommands -----------------------------------------------------------


def cmd_manifest(_args: argparse.Namespace) -> None:
    print(json.dumps(build_manifest()))


def cmd_embed(args: argparse.Namespace) -> None:
    import numpy as np

    layer = int(args.layers.split(",")[0])
    if layer != -1:
        emit_error_and_exit(
            "InvalidInput",
            "esm-c image supports only the final layer (--layers -1)",
            layers=args.layers,
        )
    device = pick_device(args.device)
    model = load_model(device)
    records = read_fasta(Path(args.input))
    ids = sanitize_ids([rid for rid, _ in records])
    output_dir = Path(args.output)
    warnings: list[str] = []

    pooled: dict[str, object] = {}
    artifacts: list[dict] = []
    per_residue_dir = output_dir / "per_residue"
    if args.pooling == "none":
        per_residue_dir.mkdir(parents=True, exist_ok=True)

    dim = _MODEL_INFO[DEFAULT_CHECKPOINT]["embedding_dim"]
    for clean_id, (rid, seq) in zip(ids, records, strict=True):
        seq = _truncate(seq, warnings, rid)
        residue, cls_vec = _embed_one(model, seq, device)
        if args.pooling == "mean":
            pooled[clean_id] = residue.mean(axis=0)
        elif args.pooling == "cls":
            pooled[clean_id] = cls_vec
        else:  # none
            _save_npy(per_residue_dir / f"{clean_id}.npy", residue)
            artifacts.append(
                {
                    "path": f"per_residue/{clean_id}.npy",
                    "kind": "per_residue_embeddings",
                    "record_ids": [clean_id],
                    "shape": list(residue.shape),
                    "dtype": "float32",
                }
            )

    if args.pooling in ("mean", "cls"):
        np.savez(output_dir / "embeddings.npz", **pooled)
        artifacts.append(
            {
                "path": "embeddings.npz",
                "kind": "pooled_embeddings",
                "record_ids": ids,
                "shape": [len(ids), dim],
                "dtype": "float32",
            }
        )
    _write_capability_result(output_dir, "embed", records, artifacts, warnings, args)


def cmd_likelihood(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    model = load_model(device)
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE
    records = read_fasta(Path(args.input))
    ids = sanitize_ids([rid for rid, _ in records])
    output_dir = Path(args.output)
    warnings: list[str] = []

    rows = ["record_id,seq_len,log_likelihood,mean_log_likelihood,perplexity"]
    for clean_id, (rid, seq) in zip(ids, records, strict=True):
        seq = _truncate(seq, warnings, rid)
        pll = _pseudo_log_likelihood(model, seq, batch_size, device)
        mean = pll / max(len(seq), 1)
        rows.append(f"{clean_id},{len(seq)},{pll:.6f},{mean:.6f},{perplexity_from_mean(mean):.6f}")
    (output_dir / "likelihoods.csv").write_text("\n".join(rows) + "\n")

    artifacts = [{"path": "likelihoods.csv", "kind": "likelihoods_csv", "record_ids": ids}]
    _write_capability_result(output_dir, "likelihood", records, artifacts, warnings, args)


def cmd_score(args: argparse.Namespace) -> None:
    """Score variants (masked-marginal or wt-marginal) from a CSV input."""
    import csv as csv_module

    device = pick_device(args.device)
    model = load_model(device)
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE
    with Path(args.input).open(newline="") as handle:
        rows = list(csv_module.DictReader(handle))
    warnings: list[str] = []
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["wt_sequence"], []).append(row)

    out_rows: list[list[object]] = []
    for wt_seq, group in groups.items():
        seq = _truncate(wt_seq, warnings, "<wt>")
        positions = _valid_positions(group, seq)
        if args.method == "wt-marginal":
            logp = _wt_position_logprobs(model, seq, positions, device)
        else:
            logp = _masked_position_logprobs(model, seq, positions, batch_size, device)
        for row in group:
            score, n_mut, err = _score_variant(row["mutant"], seq, logp, model.tokenizer)
            if err is not None:
                warnings.append(f"{row['variant_id']}: {err}")
            score_str = "" if score is None else f"{score:.6f}"
            out_rows.append([row["variant_id"], row["mutant"], n_mut, score_str])

    output_dir = Path(args.output)
    with (output_dir / "scores.csv").open("w", newline="") as handle:
        writer = csv_module.writer(handle)
        writer.writerow(["variant_id", "mutant", "n_mutations", "score"])
        writer.writerows(out_rows)
    artifacts = [
        {
            "path": "scores.csv",
            "kind": "variant_scores_csv",
            "record_ids": [r["variant_id"] for r in rows],
        }
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


def cmd_prefetch(_args: argparse.Namespace) -> None:
    """Bake weights into the image at build time (populate the HF cache)."""
    import torch
    from esm.models.esmc import ESMC

    ESMC.from_pretrained(
        DEFAULT_CHECKPOINT,
        device=torch.device("cpu"),
        use_flash_attn=False,
    )
    print(f"prefetched {DEFAULT_CHECKPOINT}")


# --- shared result writer + arg parsing ------------------------------------


def _save_npy(path: Path, array) -> None:  # noqa: ANN001
    import numpy as np

    np.save(path, array.astype(np.float32))


def _write_capability_result(
    output_dir: Path,
    capability: str,
    records: list,
    artifacts: list[dict],
    warnings: list[str],
    args: argparse.Namespace,
) -> None:
    params = {"device": args.device or "auto"}
    if capability == "embed":
        params |= {"pooling": args.pooling, "layers": args.layers}
    elif capability == "likelihood":
        params |= {"likelihood_method": "masked_marginal"}
    write_result(
        output_dir,
        {
            "contract_version": CONTRACT_VERSION,
            "capability": capability,
            "model_name": DEFAULT_CHECKPOINT,
            "n_input_records": len(records),
            "n_output_records": len(records),
            "artifacts": artifacts,
            "warnings": warnings,
            "params": params,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="esm-c", description="ESM-C protlms contract entrypoint.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("manifest").set_defaults(func=cmd_manifest)

    embed = sub.add_parser("embed")
    embed.add_argument("--input", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--pooling", default="mean", choices=["mean", "cls", "none"])
    embed.add_argument("--layers", default="-1")
    embed.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    embed.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    embed.set_defaults(func=cmd_embed)

    likelihood = sub.add_parser("likelihood")
    likelihood.add_argument("--input", required=True)
    likelihood.add_argument("--output", required=True)
    likelihood.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    likelihood.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    likelihood.set_defaults(func=cmd_likelihood)

    score = sub.add_parser("score")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument(
        "--method", default="masked-marginal", choices=["masked-marginal", "wt-marginal"]
    )
    score.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    score.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    score.set_defaults(func=cmd_score)

    sub.add_parser("_prefetch").set_defaults(func=cmd_prefetch)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - top-level: report as a structured error
        emit_error_and_exit("InternalError", str(exc), exception=type(exc).__name__)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `pytest tests/test_esmc_entrypoint.py -v`
Expected: PASS (all helper + manifest tests). No torch/esm import occurs.

- [ ] **Step 5: Format and commit**

```bash
ruff format containers/esm-c/entrypoint.py tests/test_esmc_entrypoint.py
ruff check tests/test_esmc_entrypoint.py
git add containers/esm-c/entrypoint.py tests/test_esmc_entrypoint.py
git commit -m "esm-c: contract entrypoint (manifest/embed/likelihood/score) + helper tests"
```

---

### Task 3: ESM-C Dockerfile + README

Makes the image buildable: Python-3.12 base, `esm` installed, weights baked in, offline at runtime.

**Files:**
- Create: `containers/esm-c/Dockerfile`
- Create: `containers/esm-c/README.md`

**Interfaces:**
- Consumes: `containers/esm-c/entrypoint.py` (Task 2), env var `ESMC_CHECKPOINT`.
- Produces: a buildable image tagged `protlms-esm-c:300m` (and `:600m`) whose `ENTRYPOINT` is the contract CLI. Used by Task 4.

- [ ] **Step 1: Create the Dockerfile**

Create `containers/esm-c/Dockerfile`:

```dockerfile
# ESM-C model image for the protlms container contract.
#
# Build (300M, default / CI):
#   docker build --build-arg ESMC_CHECKPOINT=esmc_300m -t protlms-esm-c:300m containers/esm-c
# Build (600M):
#   docker build --build-arg ESMC_CHECKPOINT=esmc_600m -t protlms-esm-c:600m containers/esm-c
#
# Weights are baked in at build time, so runtime needs no network access.
# The image runs on CPU by default and uses the GPU when launched with --gpus.
#
# The esm package requires Python 3.12 (>=3.12,<3.13), so this base differs from
# the ESM2 image's pytorch base. flash-attn is intentionally not installed; the
# entrypoint always loads with use_flash_attn=False.

ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

ARG ESMC_CHECKPOINT=esmc_300m
ENV ESMC_CHECKPOINT=${ESMC_CHECKPOINT} \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "esm==3.2.3"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

# Bake the checkpoint's weights into the image (populates the HF cache layer).
RUN python /app/entrypoint.py _prefetch

# Enforce offline weights at runtime for reproducibility.
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/app/entrypoint.py"]
```

> **Build-failure fallback (only if `pip install esm` fails on a missing compiler):** add, before the `pip install` line:
> `RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*`
> Most `esm` dependencies ship cp312 wheels, so this is usually unnecessary.

- [ ] **Step 2: Build the image to verify it succeeds**

Run: `docker build --build-arg ESMC_CHECKPOINT=esmc_300m -t protlms-esm-c:300m containers/esm-c`
Expected: build completes; the final `_prefetch` layer prints `prefetched esmc_300m`. (First build is slow: it pulls torch + esm and downloads the 300M weights.)

- [ ] **Step 3: Smoke-test the manifest**

Run: `docker run --rm protlms-esm-c:300m manifest`
Expected: one line of JSON with `"model_family": "esm-c"`, `"embedding_dim": 960`, `"num_layers": 30`, `"capabilities": ["embed","likelihood","score"]`, `"contract_version": "0.3"`.

- [ ] **Step 4: Create the README**

Create `containers/esm-c/README.md`:

```markdown
# ESM-C container

A contract-compliant Docker image wrapping the
[ESM-C](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12) masked
protein language model (EvolutionaryScale). It implements the protlms container
contract (see [`../../docs/CONTRACT.md`](../../docs/CONTRACT.md)) using the native
`esm` SDK, and exposes the `manifest`, `embed`, `likelihood`, and `score`
subcommands.

The checkpoint is selected at build time via the `ESMC_CHECKPOINT` build arg and
its weights are baked into the image, so runtime requires no network access.

## Building

```bash
# 300M (demo / CI default)
docker build --build-arg ESMC_CHECKPOINT=esmc_300m -t protlms-esm-c:300m containers/esm-c

# 600M
docker build --build-arg ESMC_CHECKPOINT=esmc_600m -t protlms-esm-c:600m containers/esm-c
```

`ESMC_CHECKPOINT` accepts `esmc_300m` or `esmc_600m`. The 300M/600M weights are
open (Cambrian Open License) and download without authentication. The 6B model is
EvolutionaryScale Forge API-only and is not supported by this image.

## Running directly (debugging)

```bash
docker run --rm protlms-esm-c:300m manifest

docker run --rm -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-esm-c:300m embed --input /in/seqs.fasta --output /out --pooling mean

docker run --rm --gpus all -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-esm-c:300m likelihood --input /in/seqs.fasta --output /out
```

Normally you do not run these by hand — the `protlms` client builds these commands
for you (`protlms embed esm-c-300m seqs.fasta -o out/`).

## Models

| Checkpoint | Params | embedding_dim | layers |
|---|---|---|---|
| `esmc_300m` | 300M | 960 | 30 |
| `esmc_600m` | 600M | 1152 | 36 |

## Notes

- Uses the native `esm` SDK (`ESMC.from_pretrained`), which requires Python 3.12;
  this base image therefore differs from the ESM2 image.
- `likelihood` uses masked-marginal pseudo-log-likelihood (O(L) forward passes per
  sequence) and records `params.likelihood_method = "masked_marginal"`.
- `embed` returns the **final-layer** representation; `--layers` must be `-1`
  (the client default). Other layer indices return an `InvalidInput` error.
- `use_flash_attn=False` always — the image needs no flash-attn dependency and
  runs on CPU; with `--gpus all` it uses the GPU via standard attention.
```

- [ ] **Step 5: Commit**

```bash
git add containers/esm-c/Dockerfile containers/esm-c/README.md
git commit -m "esm-c: Dockerfile (python:3.12 base, esm SDK, baked weights) + README"
```

---

### Task 4: End-to-end Docker integration test

Proves the model-backed subcommands work through the real `protlms` client against the built 300M image. Gated like the ESM2/ProGen2 integration tests.

**Files:**
- Create: `tests/test_integration_esmc.py`
- Reuses: `tests/data/tiny.fasta` (`insulin_b`, `gb1`, `melittin`), `tests/data/variants.csv` (`self`, `single`, `double`).

**Interfaces:**
- Consumes: the registry entry `esm-c-300m` (Task 1), the image `protlms-esm-c:300m` (Task 3), `protlms.load`/`Model.embed`/`.likelihood`/`.score` (existing client API).
- Produces: nothing downstream (final task).

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration_esmc.py`:

```python
"""End-to-end integration test against a locally built ESM-C image.

Gated: runs only when ``PROTLMS_RUN_DOCKER_TESTS=1`` and a working Docker daemon is
available. Builds the ``esmc_300m`` image if it is not already present, then
drives the real ``protlms`` client through embed, likelihood, and score on a small
FASTA / variants CSV of real protein sequences.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import protlms

IMAGE = "protlms-esm-c:300m"
EMBEDDING_DIM = 960
REPO_ROOT = Path(__file__).parents[1]
TINY_FASTA = REPO_ROOT / "tests" / "data" / "tiny.fasta"
VARIANTS_CSV = REPO_ROOT / "tests" / "data" / "variants.csv"
EXPECTED_IDS = {"insulin_b", "gb1", "melittin"}


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PROTLMS_RUN_DOCKER_TESTS") != "1" or not _docker_available(),
        reason="set PROTLMS_RUN_DOCKER_TESTS=1 and ensure a Docker daemon is available",
    ),
]


@pytest.fixture(scope="session")
def esmc_image() -> str:
    """Ensure the 300M ESM-C image exists, building it if necessary."""
    present = (
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    )
    if not present:
        subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                "ESMC_CHECKPOINT=esmc_300m",
                "-t",
                IMAGE,
                str(REPO_ROOT / "containers" / "esm-c"),
            ],
            check=True,
        )
    return IMAGE


@pytest.fixture(scope="session")
def model(esmc_image: str) -> protlms.Model:
    return protlms.load("esm-c-300m")


def test_manifest_is_read_through_client(model: protlms.Model) -> None:
    assert model.manifest.name == "esmc_300m"
    assert model.manifest.embedding_dim == EMBEDDING_DIM
    capabilities = {c.value for c in model.manifest.capabilities}
    assert {"embed", "likelihood", "score"} <= capabilities


def test_embed_pooled_end_to_end(model: protlms.Model, tmp_path: Path) -> None:
    result = model.embed(TINY_FASTA, pooling="mean", output_dir=tmp_path / "emb")
    pooled = result.pooled()
    assert set(pooled) == EXPECTED_IDS
    for vector in pooled.values():
        assert vector.shape == (EMBEDDING_DIM,)
        assert vector.dtype == np.float32
        assert np.isfinite(vector).all()


def test_embed_per_residue_end_to_end(model: protlms.Model, tmp_path: Path) -> None:
    result = model.embed(TINY_FASTA, pooling="none", output_dir=tmp_path / "pr")
    per_residue = result.per_residue()
    assert set(per_residue) == EXPECTED_IDS
    # melittin is 26 residues long
    assert per_residue["melittin"].shape == (26, EMBEDDING_DIM)


def test_likelihood_end_to_end(model: protlms.Model, tmp_path: Path) -> None:
    result = model.likelihood(TINY_FASTA, output_dir=tmp_path / "ll")
    rows = {row["record_id"]: row for row in result.rows()}
    assert set(rows) == EXPECTED_IDS
    for row in rows.values():
        assert row["perplexity"] > 1.0
        assert math.isfinite(float(row["log_likelihood"]))
        assert row["seq_len"] > 0
    assert result.result.params["likelihood_method"] == "masked_marginal"


def test_score_masked_marginal_end_to_end(model: protlms.Model, tmp_path: Path) -> None:
    result = model.score(VARIANTS_CSV, method="masked-marginal", output_dir=tmp_path / "sc")
    rows = {r["variant_id"]: r for r in result.rows()}
    assert set(rows) == {"self", "single", "double"}
    assert rows["self"]["score"] == pytest.approx(0.0, abs=1e-5)
    assert rows["self"]["n_mutations"] == 1
    assert rows["double"]["n_mutations"] == 2
    assert math.isfinite(float(rows["single"]["score"]))
```

- [ ] **Step 2: Run the integration test (gated)**

Run: `PROTLMS_RUN_DOCKER_TESTS=1 pytest tests/test_integration_esmc.py -v -m slow`
Expected: PASS. The session fixture builds `protlms-esm-c:300m` on first run (slow), then all five tests pass. This is where the SDK forward path (`out.embeddings`, `out.sequence_logits`), the BOS/EOS slicing, the mask-token PLL, and tokenizer AA→id mapping are proven correct.

> **If `embeddings`/`sequence_logits` shapes or attribute names differ from this plan** (the one residual SDK unknown): confirm against the installed `esm==3.2.3` with a one-off `docker run --rm protlms-esm-c:300m python -c "..."`, and adjust `_embed_one` / `_pseudo_log_likelihood` accordingly. The documented fallback is the high-level `model.logits(model.encode(ESMProtein(sequence=seq)), LogitsConfig(sequence=True, return_embeddings=True))` API returning `out.logits.sequence` and `out.embeddings`.

- [ ] **Step 3: Run the full unit suite to confirm no regressions**

Run: `pytest`
Expected: PASS (gated integration tests skipped without the env var). Then `ruff check src/ tests/`, `ruff format --check src/ tests/`, `ty check src/` clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_esmc.py
git commit -m "test: end-to-end ESM-C integration (embed/likelihood/score)"
```

---

## Self-Review

**1. Spec coverage** — every spec section maps to a task:
- Backend = native `esm` SDK → Task 2 (`load_model`, `_encode_ids`, forward path) + Task 3 (`pip install esm`).
- Capabilities embed/likelihood/score, no generate → Task 2 (`build_parser` has no `generate`).
- Two checkpoints, default+test 300M → Task 1 (both registry entries), Task 3 (build arg, both build commands), Task 4 (tests 300M).
- No contract / client-logic changes → Global Constraints + only `models.yaml`/tests touched.
- Manifest fields (family, dims, pooling, contract_version) → Task 2 `build_manifest` + unit tests; Task 3 smoke-test.
- Pooling/BOS-EOS scheme → Task 2 `_embed_one`.
- Masked-marginal likelihood + `likelihood_method` → Task 2 `_pseudo_log_likelihood` + `_write_capability_result`; Task 4 assertion.
- Score masked/wt-marginal → Task 2 `cmd_score` + helpers; Task 4 assertion.
- Dockerfile (Python 3.12 base, baked weights, offline), README → Task 3.
- Tests (unit pure-helper, registry, gated integration) → Tasks 1, 2, 4.
- Open weights / no HF token → Task 3 README.

**2. Placeholder scan** — no TBD/TODO; every code step has complete content. The two "fallback" call-outs (Dockerfile compiler, SDK shape mismatch) are explicit, bounded contingencies with concrete remedies, not deferred work — the primary path is fully specified.

**3. Type/name consistency** — checkpoint strings `esmc_300m`/`esmc_600m`, image tags `protlms-esm-c:300m`/`:600m`, registry names `esm-c-300m`/`esm-c-600m`, `EMBEDDING_DIM=960`, `model_family="esm-c"`, and helper names (`sanitize_ids`, `read_fasta`, `parse_mutant`, `perplexity_from_mean`, `_truncate`, `build_manifest`, `_embed_one`, `_pseudo_log_likelihood`, `_masked_position_logprobs`, `_wt_position_logprobs`, `_score_variant`) are consistent across the entrypoint, unit tests, Dockerfile, README, and integration test.

## Deviations from the spec (flagged)

1. **Manifest dims via a checkpoint-keyed table**, not derived from the loaded model — keeps `manifest` (and thus `protlms.load`) model-load-free; equally drift-proof since the checkpoint name pins the architecture.
2. **`max_sequence_length = 2048`** chosen for the "documented constant" the spec left open (ESM2 uses 1024; ESM-C handles longer context).
3. **Embed restricted to the final layer (`--layers -1`)** — avoids shipping unverified hidden-state-indexing code; documented in the README. Arbitrary intermediate layers are a possible future enhancement.

**As-built correction (not in the embedded code above):** the released `esm==3.2.3` has no `use_flash_attn` parameter on `ESMC.from_pretrained` (it exists only on unreleased GitHub `main`), so the shipped entrypoint omits that kwarg from both `from_pretrained` calls — flash-attn is auto-disabled because the package is not installed. The code blocks above still show `use_flash_attn=False`; the shipped `containers/esm-c/entrypoint.py` is authoritative.
```
