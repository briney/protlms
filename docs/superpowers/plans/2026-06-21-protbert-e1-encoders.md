# ProtBERT + Profluent-E1 Encoder Containers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two contract-compliant model images — `containers/protbert/` (two checkpoints) and `containers/e1/` (three sizes, single-sequence mode) — plus registry entries and tests, for `embed`/`likelihood`/`score`, with zero client production-code changes.

**Architecture:** Both models are bidirectional masked-LM encoders, so each ships as a standalone Docker image whose entrypoint implements the existing `0.3` container contract for `embed`/`likelihood`/`score`. ProtBERT follows the **esm2** template (HuggingFace `AutoModelForMaskedLM`, config-derived manifest) plus a tokenization preprocessing shim. E1 follows the **esm-c** template (custom-package SDK, load-free `_MODEL_INFO` manifest table, final-layer-only embeddings). The client never changes — it resolves new registry names to images and speaks the existing contract.

**Tech Stack:** Python 3.11 (ProtBERT image) / Python 3.12 (E1 image), HuggingFace `transformers` (ProtBERT), the custom `E1` package from GitHub (E1), PyTorch, Docker. The client side touches only `src/protlms/_data/models.yaml` (YAML) and pytest tests.

## Global Constraints

These apply to **every** task. Exact values copied from the design spec (`docs/superpowers/specs/2026-06-21-protbert-e1-encoders-design.md`):

- **Contract version:** `"0.3"`. No contract changes — `docs/CONTRACT.md` and `src/protlms/contract.py` are NOT edited.
- **No client production-code changes.** Only `src/protlms/_data/models.yaml` (registry data) and tests are touched on the client side. `contract.py`, `models.py`, `io.py`, `cli.py`, `runner.py`, `registry.py` are untouched.
- **No publish-workflow changes.** `.github/workflows/publish-image.yaml` is registry-driven (it resolves `context`/`image`/`build_args` from `models.yaml` via `scripts.registry_publish lookup`), so a new registry entry with a `build:` block is publishable as-is. Do NOT edit the workflow.
- **Capabilities:** `embed`, `likelihood`, `score`. No `generate` (neither model is autoregressive).
- **Scoring = the toolkit's shared masked-marginal / wt-marginal logic on raw model logits** for both models — the same code path as ESM2/ESM-C. For E1 this means NOT using the bundled `E1Scorer`.
- **E1 single-sequence mode only.** Never pass homolog/`context_seqs`. No retrieval API surface.
- **Standalone container code:** each `entrypoint.py` duplicates the pure helpers (`sanitize_ids`, `read_fasta`, `parse_mutant`, `perplexity_from_mean`, `_valid_positions`, `_truncate`, `write_result`, `emit_error_and_exit`) by design — each container is self-contained (same pattern as `containers/esm2/`, `containers/esm-c/`). Heavy imports (`torch`, `transformers`, `E1`) live **inside functions** so the pure helpers unit-test without the ML stack.
- **Image references** (match the GHCR convention already in `models.yaml`):
  - `ghcr.io/briney/protlms-protbert:uniref100` (build arg `PROTBERT_CHECKPOINT=prot_bert`)
  - `ghcr.io/briney/protlms-protbert:bfd` (build arg `PROTBERT_CHECKPOINT=prot_bert_bfd`)
  - `ghcr.io/briney/protlms-e1:150m` / `:300m` / `:600m` (build arg `E1_CHECKPOINT=E1-150m` / `E1-300m` / `E1-600m`)
- **ProtBERT:** `model_family = "protbert"`; checkpoint strings exactly `"prot_bert"` / `"prot_bert_bfd"` → resolved to `Rostlab/<name>`; `max_sequence_length = 1024`; `default_batch_size = 8`; `min_gpu_memory_gb = null`; dims from `AutoConfig` (1024 / 30 layers). Mask token `[MASK]`.
- **ProtBERT preprocessing (mandatory, the one real delta from esm2):** `preprocess(seq) = " ".join(re.sub(r"[UZOB]", "X", seq.upper()))` — rare residues `U/Z/O/B → X`, then space-separate every residue; tokenizer loaded with `do_lower_case=False`. Sequence length for truncation/PLL is measured on the **raw** (un-spaced) sequence.
- **E1:** `model_family = "e1"`; checkpoint strings exactly `"E1-150m"` / `"E1-300m"` / `"E1-600m"` → resolved to `Profluent-Bio/<name>`; `max_sequence_length = 2048`; `default_batch_size = 8`. Manifest dims from a load-free `_MODEL_INFO` table: 150m → 768/20/`null`, 300m → 1024/20/`2.0`, 600m → 1280/30/`4.0`. Mask token `?` (id 5); amino-acid token ids are letters `A..Z` at ids `8..33`.
- **E1 base image = `python:3.12-slim-bookworm`** (NOT a `pytorch/pytorch` image): E1 requires Python `>=3.12,<3.14`, and the official PyTorch images ship Python 3.11. torch (`>=2.7,<2.9`) is pulled transitively by `pip install E1`. **flash-attn is NOT installed**; E1's `flex_attention` fallback is used. (Mirrors how esm-c uses `python:3.12-slim-bookworm`.)
- **E1 embed = final layer only** (`--layers -1`, the client default). Any other layer index → a structured `InvalidInput` error.
- **Quality gates (run before each commit that touches `src/`/`tests/`):** `ruff check src/ tests/`, `ruff format src/ tests/`, `ty check src/`. The container entrypoints under `containers/` follow the same style but are not part of the package; still run `ruff format` on them.
- **Commit messages:** `<component>: <what changed and why>`, imperative.

---

### Task 1: Registry entries for ProtBERT (client side, no Docker)

Adds the two ProtBERT registry entries and a test proving they resolve. Independently shippable.

**Files:**
- Modify: `src/protlms/_data/models.yaml` (append two entries)
- Test: `tests/test_registry.py` (append one test)

**Interfaces:**
- Consumes: `protlms.registry.Registry.load()` / `.resolve(name)` → `ModelEntry(name, aliases, image, model_family, build)` (existing).
- Produces: resolvable names `protbert` (alias `prot_bert`) → image `ghcr.io/briney/protlms-protbert:uniref100`, and `protbert-bfd` (alias `prot_bert_bfd`) → image `ghcr.io/briney/protlms-protbert:bfd`, both `model_family="protbert"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_resolve_protbert() -> None:
    registry = Registry.load()
    base = registry.resolve("protbert")
    assert base.image == "ghcr.io/briney/protlms-protbert:uniref100"
    assert base.model_family == "protbert"
    assert registry.resolve("prot_bert") == base
    bfd = registry.resolve("protbert-bfd")
    assert bfd.image == "ghcr.io/briney/protlms-protbert:bfd"
    assert bfd.model_family == "protbert"
    assert registry.resolve("prot_bert_bfd") == bfd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py::test_resolve_protbert -v`
Expected: FAIL — `ModelNotFoundError: unknown model 'protbert'`.

- [ ] **Step 3: Add the registry entries**

Append to `src/protlms/_data/models.yaml` (after the last existing entry):

```yaml
  - name: protbert
    aliases: [prot_bert]
    image: ghcr.io/briney/protlms-protbert:uniref100
    model_family: protbert
    build:
      context: containers/protbert
      args: { PROTBERT_CHECKPOINT: prot_bert }
  - name: protbert-bfd
    aliases: [prot_bert_bfd]
    image: ghcr.io/briney/protlms-protbert:bfd
    model_family: protbert
    build:
      context: containers/protbert
      args: { PROTBERT_CHECKPOINT: prot_bert_bfd }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (all registry tests, including the new one).

- [ ] **Step 5: Commit**

```bash
git add src/protlms/_data/models.yaml tests/test_registry.py
git commit -m "registry: add protbert + protbert-bfd entries"
```

---

### Task 2: ProtBERT entrypoint (contract CLI) + pure-helper unit tests

Creates the full standalone entrypoint. The pure helpers (including the `preprocess` shim and `resolve_hf_id`) are proven now by unit tests (no torch/transformers needed); the model-backed subcommands use lazy imports and are proven later by the Docker integration test (Task 4).

**Files:**
- Create: `containers/protbert/entrypoint.py`
- Test: `tests/test_protbert_entrypoint.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Tasks 3–4): a CLI module exposing subcommands `manifest`, `embed`, `likelihood`, `score`, `_prefetch`; pure helpers `resolve_hf_id(str)->str`, `preprocess(str)->str`, `sanitize_ids(list[str])->list[str]`, `read_fasta(Path)->list[tuple[str,str]]`, `parse_mutant(str)->list[tuple[str,int,str]]`, `perplexity_from_mean(float)->float`, `_truncate(str,list[str],str)->str`; and `build_manifest()->dict`. Env var `PROTBERT_CHECKPOINT` selects the checkpoint (default `"prot_bert"`).

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_protbert_entrypoint.py`:

```python
"""Unit tests for the ProtBERT entrypoint's torch/transformers-free helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENTRYPOINT = Path(__file__).parents[1] / "containers" / "protbert" / "entrypoint.py"


def _load():
    spec = importlib.util.spec_from_file_location("protbert_entrypoint", _ENTRYPOINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entrypoint = _load()


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("prot_bert", "Rostlab/prot_bert"),
        ("prot_bert_bfd", "Rostlab/prot_bert_bfd"),
        ("Rostlab/prot_bert", "Rostlab/prot_bert"),
    ],
)
def test_resolve_hf_id(checkpoint: str, expected: str) -> None:
    assert entrypoint.resolve_hf_id(checkpoint) == expected


def test_preprocess_spaces_residues() -> None:
    assert entrypoint.preprocess("MKTAY") == "M K T A Y"


def test_preprocess_maps_rare_residues_to_x() -> None:
    # U, Z, O, B -> X; lowercase is upper-cased; result is space-separated.
    assert entrypoint.preprocess("auzob") == "A X X X X"


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_protbert_entrypoint.py -v`
Expected: FAIL at collection — load error because `containers/protbert/entrypoint.py` does not exist yet.

- [ ] **Step 3: Create the entrypoint**

Create `containers/protbert/entrypoint.py` with this exact content:

```python
#!/usr/bin/env python
"""Contract-compliant entrypoint for the ProtBERT model image.

Implements the protlms container contract (see docs/CONTRACT.md) for the ProtBERT
masked protein language model (ProtTrans / Rostlab) via HuggingFace
``transformers``. Exposes the ``manifest``, ``embed``, ``likelihood``, and
``score`` subcommands plus a hidden ``_prefetch`` used at build time to bake
weights into the image.

This file is intentionally dependency-light at import time: ``torch`` and
``transformers`` are imported inside the functions that need them, so the pure
helpers can be unit-tested without the ML stack installed.
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
DEFAULT_CHECKPOINT = os.environ.get("PROTBERT_CHECKPOINT", "prot_bert")

_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_MUTANT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")
_RARE_AA = re.compile(r"[UZOB]")


# --- pure helpers (unit-testable without torch) ----------------------------


def resolve_hf_id(checkpoint: str) -> str:
    """Resolve a short ProtBERT checkpoint name to a HuggingFace model id.

    ``prot_bert`` -> ``Rostlab/prot_bert``. A value already containing ``/`` is
    treated as a full HuggingFace id and returned unchanged.
    """
    if "/" in checkpoint:
        return checkpoint
    return f"Rostlab/{checkpoint}"


def preprocess(seq: str) -> str:
    """Format a raw sequence for ProtBERT's tokenizer.

    ProtBERT expects whitespace-separated residues and was trained with the rare
    residues U, Z, O, B mapped to X. Each residue becomes exactly one token, so
    token index ``i+1`` (after ``[CLS]``) corresponds to residue ``i``.
    """
    return " ".join(_RARE_AA.sub("X", seq.upper()))


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


def load_model(device: str):  # noqa: ANN201 - returns (tokenizer, model)
    """Load the tokenizer and masked-LM model for the configured checkpoint."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    hf_id = resolve_hf_id(DEFAULT_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(hf_id, torch_dtype=torch.float32)
    model.eval().to(device)
    return tokenizer, model


def build_manifest() -> dict:
    """Build the manifest dict from the checkpoint's config."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(resolve_hf_id(DEFAULT_CHECKPOINT))
    return {
        "contract_version": CONTRACT_VERSION,
        "name": DEFAULT_CHECKPOINT,
        "version": "1.0.0",
        "description": f"ProtBERT masked protein language model ({DEFAULT_CHECKPOINT}).",
        "model_family": "protbert",
        "capabilities": ["embed", "likelihood", "score"],
        "embedding_dim": int(config.hidden_size),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "pooling_modes": ["mean", "cls", "none"],
        "num_layers": int(config.num_hidden_layers),
        "min_gpu_memory_gb": None,
        "default_batch_size": DEFAULT_BATCH_SIZE,
    }


# --- subcommands -----------------------------------------------------------


def cmd_manifest(_args: argparse.Namespace) -> None:
    print(json.dumps(build_manifest()))


def cmd_embed(args: argparse.Namespace) -> None:
    import torch

    device = pick_device(args.device)
    tokenizer, model = load_model(device)
    layer = int(args.layers.split(",")[0])
    records = read_fasta(Path(args.input))
    ids = sanitize_ids([rid for rid, _ in records])
    output_dir = Path(args.output)
    warnings: list[str] = []

    pooled: dict[str, object] = {}
    artifacts: list[dict] = []
    per_residue_dir = output_dir / "per_residue"
    if args.pooling == "none":
        per_residue_dir.mkdir(parents=True, exist_ok=True)

    for clean_id, (rid, seq) in zip(ids, records, strict=True):
        seq = _truncate(seq, warnings, rid)
        residue, cls_vec = _embed_one(tokenizer, model, seq, layer, device)
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
        import numpy as np

        np.savez(output_dir / "embeddings.npz", **pooled)
        artifacts.append(
            {
                "path": "embeddings.npz",
                "kind": "pooled_embeddings",
                "record_ids": ids,
                "shape": [len(ids), model.config.hidden_size],
                "dtype": "float32",
            }
        )
    _write_capability_result(output_dir, "embed", records, artifacts, warnings, args)
    del torch  # silence unused-import linters; torch is used transitively above


def _embed_one(tokenizer, model, seq: str, layer: int, device: str):  # noqa: ANN001, ANN202
    import numpy as np
    import torch

    enc = tokenizer(preprocess(seq), return_tensors="pt").to(device)
    use_amp = device == "cuda"
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
        out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[layer][0].float().cpu().numpy()  # (T, D)
    residue = hidden[1 : 1 + len(seq)].astype(np.float32)  # strip [CLS]/[SEP]
    cls_vec = hidden[0].astype(np.float32)
    return residue, cls_vec


def cmd_likelihood(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    tokenizer, model = load_model(device)
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE
    records = read_fasta(Path(args.input))
    ids = sanitize_ids([rid for rid, _ in records])
    output_dir = Path(args.output)
    warnings: list[str] = []

    rows = ["record_id,seq_len,log_likelihood,mean_log_likelihood,perplexity"]
    for clean_id, (rid, seq) in zip(ids, records, strict=True):
        seq = _truncate(seq, warnings, rid)
        pll = _pseudo_log_likelihood(tokenizer, model, seq, batch_size, device)
        mean = pll / max(len(seq), 1)
        rows.append(f"{clean_id},{len(seq)},{pll:.6f},{mean:.6f},{perplexity_from_mean(mean):.6f}")
    (output_dir / "likelihoods.csv").write_text("\n".join(rows) + "\n")

    artifacts = [{"path": "likelihoods.csv", "kind": "likelihoods_csv", "record_ids": ids}]
    _write_capability_result(output_dir, "likelihood", records, artifacts, warnings, args)


def _pseudo_log_likelihood(tokenizer, model, seq: str, batch_size: int, device: str) -> float:  # noqa: ANN001
    """Masked-marginal pseudo-log-likelihood: O(L) masked forward passes."""
    import torch

    enc = tokenizer(preprocess(seq), return_tensors="pt")
    input_ids = enc["input_ids"][0]  # (L+2,)
    positions = list(range(1, len(input_ids) - 1))  # residues only
    total = 0.0
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        batch = input_ids.repeat(len(chunk), 1)  # (b, L+2)
        for row, pos in enumerate(chunk):
            batch[row, pos] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(input_ids=batch.to(device)).logits  # (b, L+2, V)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        for row, pos in enumerate(chunk):
            total += float(log_probs[row, pos, input_ids[pos]])
    return total


def cmd_prefetch(_args: argparse.Namespace) -> None:
    """Bake weights into the image at build time (populate the HF cache)."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    hf_id = resolve_hf_id(DEFAULT_CHECKPOINT)
    AutoTokenizer.from_pretrained(hf_id, do_lower_case=False)
    AutoModelForMaskedLM.from_pretrained(hf_id)
    print(f"prefetched {hf_id}")


def _masked_position_logprobs(tokenizer, model, seq, positions, batch_size, device):  # noqa: ANN001
    """Map each 1-indexed position to its masked log-softmax vector over the vocab."""
    import torch

    input_ids = tokenizer(preprocess(seq), return_tensors="pt")["input_ids"][0]
    ordered = sorted(positions)
    out: dict[int, object] = {}
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

    enc = tokenizer(preprocess(seq), return_tensors="pt").to(device)
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
    """Score variants (masked-marginal or wt-marginal) from a CSV input."""
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

    out_rows: list[list[object]] = []
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
    parser = argparse.ArgumentParser(
        prog="protbert", description="ProtBERT protlms contract entrypoint."
    )
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

Run: `pytest tests/test_protbert_entrypoint.py -v`
Expected: PASS (all helper tests, including `preprocess` and `resolve_hf_id`). No torch/transformers import occurs.

- [ ] **Step 5: Format and commit**

```bash
ruff format containers/protbert/entrypoint.py tests/test_protbert_entrypoint.py
ruff check tests/test_protbert_entrypoint.py
git add containers/protbert/entrypoint.py tests/test_protbert_entrypoint.py
git commit -m "protbert: contract entrypoint (manifest/embed/likelihood/score) + helper tests"
```

---

### Task 3: ProtBERT Dockerfile + README

Makes the image buildable: ESM2's pytorch base, `transformers` pinned, weights baked in, offline at runtime.

**Files:**
- Create: `containers/protbert/Dockerfile`
- Create: `containers/protbert/README.md`

**Interfaces:**
- Consumes: `containers/protbert/entrypoint.py` (Task 2), env var `PROTBERT_CHECKPOINT`.
- Produces: a buildable image tagged `ghcr.io/briney/protlms-protbert:uniref100` (and `:bfd`) whose `ENTRYPOINT` is the contract CLI. Used by Task 4.

- [ ] **Step 1: Create the Dockerfile**

Create `containers/protbert/Dockerfile`:

```dockerfile
# ProtBERT model image for the protlms container contract.
#
# Build (UniRef100 / CI default):
#   docker build --build-arg PROTBERT_CHECKPOINT=prot_bert -t protlms-protbert:uniref100 containers/protbert
# Build (BFD):
#   docker build --build-arg PROTBERT_CHECKPOINT=prot_bert_bfd -t protlms-protbert:bfd containers/protbert
#
# Weights are baked in at build time, so runtime needs no network access.
# The image runs on CPU by default and uses the GPU when launched with --gpus.

ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG PROTBERT_CHECKPOINT=prot_bert
ENV PROTBERT_CHECKPOINT=${PROTBERT_CHECKPOINT} \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "transformers==4.46.3"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

# Bake the checkpoint's weights into the image (populates the HF cache layer).
RUN python /app/entrypoint.py _prefetch

# Enforce offline weights at runtime for reproducibility.
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/app/entrypoint.py"]
```

- [ ] **Step 2: Build the image to verify it succeeds**

Run: `docker build --build-arg PROTBERT_CHECKPOINT=prot_bert -t ghcr.io/briney/protlms-protbert:uniref100 containers/protbert`
Expected: build completes; the final `_prefetch` layer prints `prefetched Rostlab/prot_bert`. (First build downloads the ~1.7 GB weights.)

- [ ] **Step 3: Smoke-test the manifest**

Run: `docker run --rm ghcr.io/briney/protlms-protbert:uniref100 manifest`
Expected: one line of JSON with `"model_family": "protbert"`, `"name": "prot_bert"`, `"embedding_dim": 1024`, `"num_layers": 30`, `"capabilities": ["embed","likelihood","score"]`, `"contract_version": "0.3"`.

- [ ] **Step 4: Create the README**

Create `containers/protbert/README.md`:

```markdown
# ProtBERT container

A contract-compliant Docker image wrapping the
[ProtBERT](https://huggingface.co/Rostlab/prot_bert) masked protein language model
(ProtTrans / Rostlab). It implements the protlms container contract (see
[`../../docs/CONTRACT.md`](../../docs/CONTRACT.md)) using HuggingFace
`transformers`, and exposes the `manifest`, `embed`, `likelihood`, and `score`
subcommands.

The checkpoint is selected at build time via the `PROTBERT_CHECKPOINT` build arg
and its weights are baked into the image, so runtime requires no network access.

## Building

```bash
# UniRef100 (demo / CI default)
docker build --build-arg PROTBERT_CHECKPOINT=prot_bert -t protlms-protbert:uniref100 containers/protbert

# BFD
docker build --build-arg PROTBERT_CHECKPOINT=prot_bert_bfd -t protlms-protbert:bfd containers/protbert
```

`PROTBERT_CHECKPOINT` accepts `prot_bert` (UniRef100) or `prot_bert_bfd` (BFD),
resolved to `Rostlab/<name>`, or a full HuggingFace id. Both checkpoints are
released under AFL-3.0 and download without authentication.

## Running directly (debugging)

```bash
docker run --rm protlms-protbert:uniref100 manifest

docker run --rm -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-protbert:uniref100 embed --input /in/seqs.fasta --output /out --pooling mean

docker run --rm --gpus all -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-protbert:uniref100 likelihood --input /in/seqs.fasta --output /out
```

Normally you do not run these by hand — the `protlms` client builds these commands
for you (`protlms embed protbert seqs.fasta -o out/`).

## Models

| Checkpoint | Training data | Params | embedding_dim | layers |
|---|---|---|---|---|
| `prot_bert` | UniRef100 | ~420M | 1024 | 30 |
| `prot_bert_bfd` | BFD | ~420M | 1024 | 30 |

## Notes

- **Tokenization:** ProtBERT expects whitespace-separated residues and was trained
  with the rare residues U, Z, O, B mapped to X. The entrypoint applies both
  transformations automatically (`preprocess`), so the client just passes plain
  FASTA. A side effect: a `wt_sequence` that literally contains U/Z/O/B will fail
  the `score` WT-residue check (these residues are rewritten to X). This is rare.
- `likelihood` uses masked-marginal pseudo-log-likelihood (O(L) forward passes per
  sequence) and records `params.likelihood_method = "masked_marginal"`.
- `embed` supports arbitrary `--layers` via hidden states; `cls` pooling uses the
  `[CLS]` token, `mean` averages over residue positions.
- `max_sequence_length = 1024` (ProtBERT was trained at 512/2048; longer inputs are
  truncated with a warning).
- The image runs on CPU when launched without `--gpus`, and uses CUDA with mixed
  precision when launched with `--gpus all`.
```

- [ ] **Step 5: Commit**

```bash
git add containers/protbert/Dockerfile containers/protbert/README.md
git commit -m "protbert: Dockerfile (pytorch base, transformers, baked weights) + README"
```

---

### Task 4: ProtBERT end-to-end Docker integration test

Proves the model-backed subcommands work through the real `protlms` client against the built UniRef100 image. Gated like the ESM2/ESM-C integration tests.

**Files:**
- Create: `tests/test_integration_protbert.py`
- Reuses: `tests/data/tiny.fasta` (`insulin_b`, `gb1`, `melittin`), `tests/data/variants.csv` (`self`, `single`, `double`).

**Interfaces:**
- Consumes: the registry entry `protbert` (Task 1), the image `ghcr.io/briney/protlms-protbert:uniref100` (Task 3), `protlms.load`/`Model.embed`/`.likelihood`/`.score` (existing client API).
- Produces: nothing downstream.

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration_protbert.py`:

```python
"""End-to-end integration test against a locally built ProtBERT image.

Gated: runs only when ``PROTLMS_RUN_DOCKER_TESTS=1`` and a working Docker daemon is
available. Builds the ``prot_bert`` (UniRef100) image if it is not already present,
then drives the real ``protlms`` client through embed, likelihood, and score on a
small FASTA / variants CSV of real protein sequences.
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

IMAGE = "ghcr.io/briney/protlms-protbert:uniref100"
EMBEDDING_DIM = 1024
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
def protbert_image() -> str:
    """Ensure the UniRef100 ProtBERT image exists, building it if necessary."""
    present = (
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    )
    if not present:
        subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                "PROTBERT_CHECKPOINT=prot_bert",
                "-t",
                IMAGE,
                str(REPO_ROOT / "containers" / "protbert"),
            ],
            check=True,
        )
    return IMAGE


@pytest.fixture(scope="session")
def model(protbert_image: str) -> protlms.Model:
    return protlms.load("protbert")


def test_manifest_is_read_through_client(model: protlms.Model) -> None:
    assert model.manifest.name == "prot_bert"
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

> **Note on `variants.csv`:** if any of its `wt_sequence` values happen to contain U/Z/O/B, the `score` WT-residue check will reject those rows (ProtBERT rewrites them to X). `tests/data/variants.csv` uses standard residues, so this does not apply — but if a future edit introduces them, expect a warning + blank score for those rows.

- [ ] **Step 2: Run the integration test (gated)**

Run: `PROTLMS_RUN_DOCKER_TESTS=1 pytest tests/test_integration_protbert.py -v -m slow`
Expected: PASS. The session fixture builds `ghcr.io/briney/protlms-protbert:uniref100` on first run (slow), then all five tests pass. This is where the `preprocess` space-join (one token per residue), `[CLS]`/`[SEP]` slicing, masked-marginal PLL, and `convert_tokens_to_ids` scoring are proven correct end-to-end.

- [ ] **Step 3: Run the full unit suite to confirm no regressions**

Run: `pytest`
Expected: PASS (gated integration tests skipped without the env var). Then `ruff check src/ tests/`, `ruff format --check src/ tests/`, `ty check src/` clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_protbert.py
git commit -m "test: end-to-end ProtBERT integration (embed/likelihood/score)"
```

---

### Task 5: Registry entries for E1 (client side, no Docker)

Adds the three E1 registry entries and a test proving they resolve. Independently shippable.

**Files:**
- Modify: `src/protlms/_data/models.yaml` (append three entries)
- Test: `tests/test_registry.py` (append one test)

**Interfaces:**
- Consumes: `protlms.registry.Registry.load()` / `.resolve(name)` (existing).
- Produces: resolvable names `e1-150m`/`e1-300m`/`e1-600m` (aliases `E1-150m`/`E1-300m`/`E1-600m`) → images `ghcr.io/briney/protlms-e1:{150m,300m,600m}`, all `model_family="e1"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_resolve_e1() -> None:
    registry = Registry.load()
    cases = [
        ("e1-150m", "E1-150m", "150m"),
        ("e1-300m", "E1-300m", "300m"),
        ("e1-600m", "E1-600m", "600m"),
    ]
    for name, alias, tag in cases:
        entry = registry.resolve(name)
        assert entry.image == f"ghcr.io/briney/protlms-e1:{tag}"
        assert entry.model_family == "e1"
        assert registry.resolve(alias) == entry
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py::test_resolve_e1 -v`
Expected: FAIL — `ModelNotFoundError: unknown model 'e1-150m'`.

- [ ] **Step 3: Add the registry entries**

Append to `src/protlms/_data/models.yaml` (after the ProtBERT entries):

```yaml
  - name: e1-150m
    aliases: [E1-150m]
    image: ghcr.io/briney/protlms-e1:150m
    model_family: e1
    build:
      context: containers/e1
      args: { E1_CHECKPOINT: E1-150m }
  - name: e1-300m
    aliases: [E1-300m]
    image: ghcr.io/briney/protlms-e1:300m
    model_family: e1
    build:
      context: containers/e1
      args: { E1_CHECKPOINT: E1-300m }
  - name: e1-600m
    aliases: [E1-600m]
    image: ghcr.io/briney/protlms-e1:600m
    model_family: e1
    build:
      context: containers/e1
      args: { E1_CHECKPOINT: E1-600m }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (all registry tests, including the new one).

- [ ] **Step 5: Commit**

```bash
git add src/protlms/_data/models.yaml tests/test_registry.py
git commit -m "registry: add e1-150m/300m/600m entries"
```

---

### Task 6: E1 entrypoint (contract CLI) + pure-helper/manifest/score unit tests

Creates the full standalone entrypoint. The pure helpers, the load-free manifest, AND the pure `_score_variant` (which uses the constant `_AA_TO_ID` map, not a live tokenizer) are proven now by unit tests. The model-backed subcommands use lazy `E1` imports and are proven later by the Docker integration test (Task 8).

**Files:**
- Create: `containers/e1/entrypoint.py`
- Test: `tests/test_e1_entrypoint.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Tasks 7–8): a CLI module exposing subcommands `manifest`, `embed`, `likelihood`, `score`, `_prefetch`; pure helpers `resolve_hf_id`, `sanitize_ids`, `read_fasta`, `parse_mutant`, `perplexity_from_mean`, `_truncate`, `_score_variant(mutant, wt_seq, logp_by_pos)->tuple`; constants `_MASK_ID=5`, `_AA_TO_ID` (`A=8 .. Z=33`); and `build_manifest()->dict`. Env var `E1_CHECKPOINT` selects the checkpoint (default `"E1-150m"`).

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_e1_entrypoint.py`:

```python
"""Unit tests for the Profluent-E1 entrypoint's torch/E1-free helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENTRYPOINT = Path(__file__).parents[1] / "containers" / "e1" / "entrypoint.py"


def _load():
    spec = importlib.util.spec_from_file_location("e1_entrypoint", _ENTRYPOINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entrypoint = _load()


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("E1-150m", "Profluent-Bio/E1-150m"),
        ("E1-600m", "Profluent-Bio/E1-600m"),
        ("Profluent-Bio/E1-300m", "Profluent-Bio/E1-300m"),
    ],
)
def test_resolve_hf_id(checkpoint: str, expected: str) -> None:
    assert entrypoint.resolve_hf_id(checkpoint) == expected


def test_vocab_constants() -> None:
    # Documented E1 vocab: mask "?" is id 5; amino acids are A..Z at ids 8..33.
    assert entrypoint._MASK_ID == 5
    assert entrypoint._AA_TO_ID["A"] == 8
    assert entrypoint._AA_TO_ID["C"] == 10
    assert entrypoint._AA_TO_ID["Z"] == 33


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


def _logp_row(value_by_id: dict[int, float]) -> list[float]:
    """Build a length-34 logprob vector with the given id->value overrides."""
    vec = [0.0] * 34
    for idx, val in value_by_id.items():
        vec[idx] = val
    return vec


def test_score_variant_self_substitution_is_zero() -> None:
    wt = "ACDEF"
    logp = {1: _logp_row({entrypoint._AA_TO_ID["A"]: -3.0})}
    score, n_mut, err = entrypoint._score_variant("A1A", wt, logp)
    assert err is None
    assert n_mut == 1
    assert score == pytest.approx(0.0)


def test_score_variant_single_uses_logratio() -> None:
    wt = "ACDEF"
    logp = {1: _logp_row({entrypoint._AA_TO_ID["A"]: -2.0, entrypoint._AA_TO_ID["G"]: -0.5})}
    score, n_mut, err = entrypoint._score_variant("A1G", wt, logp)
    assert err is None
    assert score == pytest.approx(-0.5 - (-2.0))


def test_score_variant_wt_mismatch_errors() -> None:
    score, _n, err = entrypoint._score_variant("C1G", "ACDEF", {1: _logp_row({})})
    assert score is None
    assert "mismatch" in err


def test_build_manifest_150m(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entrypoint, "DEFAULT_CHECKPOINT", "E1-150m")
    m = entrypoint.build_manifest()
    assert m["contract_version"] == "0.3"
    assert m["model_family"] == "e1"
    assert m["name"] == "E1-150m"
    assert m["embedding_dim"] == 768
    assert m["num_layers"] == 20
    assert m["min_gpu_memory_gb"] is None
    assert set(m["capabilities"]) == {"embed", "likelihood", "score"}
    assert set(m["pooling_modes"]) == {"mean", "cls", "none"}


def test_build_manifest_600m(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entrypoint, "DEFAULT_CHECKPOINT", "E1-600m")
    m = entrypoint.build_manifest()
    assert m["embedding_dim"] == 1280
    assert m["num_layers"] == 30
    assert m["min_gpu_memory_gb"] == 4.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_e1_entrypoint.py -v`
Expected: FAIL at collection — load error because `containers/e1/entrypoint.py` does not exist yet.

- [ ] **Step 3: Create the entrypoint**

Create `containers/e1/entrypoint.py` with this exact content:

```python
#!/usr/bin/env python
"""Contract-compliant entrypoint for the Profluent-E1 model image.

Implements the protlms container contract (see docs/CONTRACT.md) for the
Profluent-E1 masked protein language model in SINGLE-SEQUENCE mode via the custom
``E1`` package. Exposes the ``manifest``, ``embed``, ``likelihood``, and ``score``
subcommands plus a hidden ``_prefetch`` used at build time to bake weights into
the image. Retrieval-augmented (homolog-context) mode is intentionally not exposed.

Heavy imports (``torch``, ``E1``) happen inside the functions that need them, so
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
DEFAULT_CHECKPOINT = os.environ.get("E1_CHECKPOINT", "E1-150m")

# Architecture facts keyed by checkpoint name. Keeping these here lets `manifest`
# stay model-load-free (the checkpoint name pins the architecture): embedding_dim
# is the model width, num_layers the transformer depth.
_MODEL_INFO: dict[str, dict[str, object]] = {
    "E1-150m": {"embedding_dim": 768, "num_layers": 20, "min_gpu_memory_gb": None},
    "E1-300m": {"embedding_dim": 1024, "num_layers": 20, "min_gpu_memory_gb": 2.0},
    "E1-600m": {"embedding_dim": 1280, "num_layers": 30, "min_gpu_memory_gb": 4.0},
}

# E1 tokenizer vocab (from the model's tokenizer.json): the mask token is "?"
# (id 5); amino-acid tokens are the 26 uppercase letters A..Z at ids 8..33.
# Residue positions in a tokenized sequence are exactly those whose id is an AA id;
# boundary tokens (<bos>=1, <eos>=2, etc.) fall outside that range. (Verified in
# the integration test; see README.)
_MASK_ID = 5
_AA_TO_ID = {aa: 8 + i for i, aa in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
_AA_IDS = frozenset(_AA_TO_ID.values())

_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_MUTANT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


# --- pure helpers (unit-testable without torch/E1) -------------------------


def resolve_hf_id(checkpoint: str) -> str:
    """Resolve a short E1 checkpoint name to a HuggingFace model id.

    ``E1-150m`` -> ``Profluent-Bio/E1-150m``. A value already containing ``/`` is
    treated as a full HuggingFace id and returned unchanged.
    """
    if "/" in checkpoint:
        return checkpoint
    return f"Profluent-Bio/{checkpoint}"


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


def _score_variant(mutant, wt_seq, logp_by_pos):  # noqa: ANN001
    """Return (score, n_mutations, error_message_or_None) for one variant.

    Uses the constant ``_AA_TO_ID`` map (no live tokenizer), so this is pure and
    unit-testable. ``logp_by_pos`` maps a 1-indexed residue position to a vocab
    log-prob vector (indexable by amino-acid token id).
    """
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
        total += float(vec[_AA_TO_ID[mut_aa]] - vec[_AA_TO_ID[wt_aa]])
    return total, len(subs), None


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
        "description": (
            f"Profluent-E1 masked protein language model ({DEFAULT_CHECKPOINT}), "
            "single-sequence mode."
        ),
        "model_family": "e1",
        "capabilities": ["embed", "likelihood", "score"],
        "embedding_dim": info["embedding_dim"],
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "pooling_modes": ["mean", "cls", "none"],
        "num_layers": info["num_layers"],
        "min_gpu_memory_gb": info["min_gpu_memory_gb"],
        "default_batch_size": DEFAULT_BATCH_SIZE,
    }


# --- model helpers (E1 package; proven by the Docker integration test) -----


def pick_device(requested: str | None) -> str:
    """Choose the torch device, validating an explicit ``cuda`` request."""
    import torch

    available = "cuda" if torch.cuda.is_available() else "cpu"
    if requested in (None, "auto"):
        return available
    if requested == "cuda" and available != "cuda":
        emit_error_and_exit("DeviceUnavailable", "cuda requested but no GPU is available")
    return requested


def load_model(device: str):  # noqa: ANN201 - returns an E1ForMaskedLM module
    """Load the E1 model for the configured checkpoint."""
    from E1.modeling import E1ForMaskedLM

    model = E1ForMaskedLM.from_pretrained(resolve_hf_id(DEFAULT_CHECKPOINT))
    model.eval().to(device)
    return model


_PREPARER = None


def _preparer():  # noqa: ANN202
    """Return a cached E1BatchPreparer (single-sequence batches)."""
    global _PREPARER
    if _PREPARER is None:
        from E1.batch_preparer import E1BatchPreparer

        _PREPARER = E1BatchPreparer()
    return _PREPARER


def _prepare(seq: str, device: str):  # noqa: ANN202
    """Tokenize one sequence (single-sequence mode) into E1 forward kwargs.

    Returns ``(batch, residue_positions)`` where ``residue_positions`` are the
    token indices holding amino acids (boundary tokens excluded).
    """
    batch = _preparer().get_batch_kwargs([seq], device=device)
    ids = batch["input_ids"][0].tolist()
    residue_positions = [i for i, t in enumerate(ids) if t in _AA_IDS]
    return batch, residue_positions


def _forward(model, batch):  # noqa: ANN001, ANN202
    """Single forward pass returning (logits (B,T,V), embeddings (B,T,D))."""
    import torch

    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            within_seq_position_ids=batch["within_seq_position_ids"],
            global_position_ids=batch["global_position_ids"],
            sequence_ids=batch["sequence_ids"],
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
    return out.logits, out.embeddings


def _masked_logits(model, batch, token_indices):  # noqa: ANN001, ANN202
    """Forward ``len(token_indices)`` masked copies; return logits (b, T, V)."""
    import torch

    b = len(token_indices)
    input_ids = batch["input_ids"].repeat(b, 1).clone()
    for row, pos in enumerate(token_indices):
        input_ids[row, pos] = _MASK_ID
    masked = {
        "input_ids": input_ids,
        "within_seq_position_ids": batch["within_seq_position_ids"].repeat(b, 1),
        "global_position_ids": batch["global_position_ids"].repeat(b, 1),
        "sequence_ids": batch["sequence_ids"].repeat(b, 1),
        "past_key_values": None,
        "use_cache": False,
        "output_attentions": False,
        "output_hidden_states": False,
    }
    with torch.no_grad():
        out = model(**masked)
    return out.logits


def _embed_one(model, seq: str, device: str):  # noqa: ANN001, ANN202
    """Return (per-residue (L, D), cls (D,)) float32 arrays for one sequence."""
    import numpy as np

    batch, residue_positions = _prepare(seq, device)
    _logits, embeddings = _forward(model, batch)
    emb = embeddings[0].float().cpu().numpy()  # (T, D)
    residue = emb[residue_positions].astype(np.float32)  # (L, D)
    cls_vec = emb[0].astype(np.float32)  # <bos> vector
    return residue, cls_vec


def _pseudo_log_likelihood(model, seq: str, batch_size: int, device: str) -> float:  # noqa: ANN001
    """Masked-marginal pseudo-log-likelihood: O(L) masked forward passes."""
    import torch

    batch, residue_positions = _prepare(seq, device)
    true_ids = batch["input_ids"][0]
    total = 0.0
    for start in range(0, len(residue_positions), batch_size):
        chunk = residue_positions[start : start + batch_size]
        logits = _masked_logits(model, batch, chunk)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        for row, pos in enumerate(chunk):
            total += float(log_probs[row, pos, true_ids[pos]])
    return total


def _masked_position_logprobs(model, seq, positions, batch_size, device):  # noqa: ANN001
    """Map each 1-indexed residue position to its masked log-softmax vector."""
    import torch

    batch, residue_positions = _prepare(seq, device)
    ordered = sorted(positions)
    out: dict[int, object] = {}
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        tok_idx = [residue_positions[p - 1] for p in chunk]
        logits = _masked_logits(model, batch, tok_idx)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        for row, p in enumerate(chunk):
            out[p] = log_probs[row, tok_idx[row]].cpu().numpy()
    return out


def _wt_position_logprobs(model, seq, positions, device):  # noqa: ANN001
    """Map each 1-indexed residue position to its unmasked (WT-context) vector."""
    import torch

    batch, residue_positions = _prepare(seq, device)
    logits, _emb = _forward(model, batch)
    log_probs = torch.log_softmax(logits.float(), dim=-1)[0].cpu().numpy()
    return {p: log_probs[residue_positions[p - 1]] for p in positions}


# --- subcommands -----------------------------------------------------------


def cmd_manifest(_args: argparse.Namespace) -> None:
    print(json.dumps(build_manifest()))


def cmd_embed(args: argparse.Namespace) -> None:
    import numpy as np

    layer = int(args.layers.split(",")[0])
    if layer != -1:
        emit_error_and_exit(
            "InvalidInput",
            "e1 image supports only the final layer (--layers -1)",
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
            score, n_mut, err = _score_variant(row["mutant"], seq, logp)
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
    from E1.modeling import E1ForMaskedLM

    hf_id = resolve_hf_id(DEFAULT_CHECKPOINT)
    E1ForMaskedLM.from_pretrained(hf_id)
    print(f"prefetched {hf_id}")


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
    parser = argparse.ArgumentParser(prog="e1", description="Profluent-E1 protlms contract entrypoint.")
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

Run: `pytest tests/test_e1_entrypoint.py -v`
Expected: PASS (helpers, vocab constants, pure `_score_variant`, and all three manifest cases). No torch/E1 import occurs.

- [ ] **Step 5: Format and commit**

```bash
ruff format containers/e1/entrypoint.py tests/test_e1_entrypoint.py
ruff check tests/test_e1_entrypoint.py
git add containers/e1/entrypoint.py tests/test_e1_entrypoint.py
git commit -m "e1: contract entrypoint (manifest/embed/likelihood/score, single-seq) + helper tests"
```

---

### Task 7: E1 Dockerfile + README (pinned `E1` package, Python 3.12 base)

Makes the image buildable: Python-3.12 base, the `E1` package installed from a pinned git commit, weights baked in, offline at runtime.

**Files:**
- Create: `containers/e1/Dockerfile`
- Create: `containers/e1/README.md`

**Interfaces:**
- Consumes: `containers/e1/entrypoint.py` (Task 6), env var `E1_CHECKPOINT`.
- Produces: a buildable image tagged `ghcr.io/briney/protlms-e1:150m` (and `:300m`/`:600m`) whose `ENTRYPOINT` is the contract CLI. Used by Task 8.

- [ ] **Step 1: Resolve and record the `E1` git commit to pin**

Run: `git ls-remote https://github.com/Profluent-AI/E1.git HEAD`
Expected: prints a 40-char SHA. Copy it; use it as `<E1_SHA>` in the Dockerfile below (replace the literal). This pins the build for reproducibility (there is no PyPI release of `E1`).

- [ ] **Step 2: Create the Dockerfile**

Create `containers/e1/Dockerfile` (replace `<E1_SHA>` with the SHA from Step 1):

```dockerfile
# Profluent-E1 model image for the protlms container contract (single-sequence).
#
# Build (150m, default / CI):
#   docker build --build-arg E1_CHECKPOINT=E1-150m -t protlms-e1:150m containers/e1
# Build (300m / 600m):
#   docker build --build-arg E1_CHECKPOINT=E1-300m -t protlms-e1:300m containers/e1
#   docker build --build-arg E1_CHECKPOINT=E1-600m -t protlms-e1:600m containers/e1
#
# Weights are baked in at build time, so runtime needs no network access.
# The image runs on CPU by default and uses the GPU when launched with --gpus.
#
# The E1 package requires Python >=3.12 (the official pytorch images ship 3.11),
# so this base is python:3.12-slim and torch is pulled transitively by `pip
# install E1`. flash-attn is intentionally not installed; E1 falls back to
# flex_attention.

ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

ARG E1_CHECKPOINT=E1-150m
ARG E1_REF=<E1_SHA>
ENV E1_CHECKPOINT=${E1_CHECKPOINT} \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

# git is needed for the VCS install; build-essential covers any sdist deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "E1 @ git+https://github.com/Profluent-AI/E1.git@${E1_REF}"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

# Bake the checkpoint's weights into the image (populates the HF cache layer).
RUN python /app/entrypoint.py _prefetch

# Enforce offline weights at runtime for reproducibility.
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/app/entrypoint.py"]
```

- [ ] **Step 3: Build the image to verify it succeeds**

Run: `docker build --build-arg E1_CHECKPOINT=E1-150m -t ghcr.io/briney/protlms-e1:150m containers/e1`
Expected: build completes; the final `_prefetch` layer prints `prefetched Profluent-Bio/E1-150m`. (First build pulls torch + the E1 package and downloads the 150m weights.)

> **Build-failure fallback (only if `pip install E1` fails):** the most likely cause is a dependency needing a CUDA toolchain. The `git build-essential` apt line above usually suffices. If a specific dep (e.g. `flash-attn`) is being pulled as a hard requirement, confirm it is an *optional* extra in E1's `pyproject.toml` and install the base package only (no extras) — flash-attn is not required because the entrypoint relies on the flex_attention fallback.

- [ ] **Step 4: Smoke-test the manifest**

Run: `docker run --rm ghcr.io/briney/protlms-e1:150m manifest`
Expected: one line of JSON with `"model_family": "e1"`, `"name": "E1-150m"`, `"embedding_dim": 768`, `"num_layers": 20`, `"capabilities": ["embed","likelihood","score"]`, `"contract_version": "0.3"`.

- [ ] **Step 5: Verify the E1 API assumptions inside the image**

Run this probe to confirm the three runtime assumptions the entrypoint's model helpers depend on (so any mismatch surfaces here, not as a confusing integration failure):

```bash
docker run --rm ghcr.io/briney/protlms-e1:150m python -c "
from E1.batch_preparer import E1BatchPreparer
b = E1BatchPreparer().get_batch_kwargs(['ACDEFGHIKLMNPQRSTVWY'], device='cpu')
ids = b['input_ids'][0].tolist()
print('batch keys:', sorted(b))
print('input ids:', ids)
aa = [t for t in ids if 8 <= t <= 33]
# Expect 20 AA tokens for the 20 standard residues, ids 8..33, with A==8.
assert len(aa) == 20, aa
assert b['input_ids'][0].tolist().count(5) == 0  # no mask in a plain sequence
print('OK: residue tokens detected via AA id range 8..33')
"
```
Expected: `batch keys: ['global_position_ids', 'input_ids', 'sequence_ids', 'within_seq_position_ids']` and `OK: residue tokens detected ...`.

> **If the probe fails** (different batch keys, AA ids not in 8..33, or a different attribute layout): the documented vocab/forward contract differs from this plan. Confirm the real layout with a one-off `docker run ... python -c "..."` that prints the tokenizer vocab and a forward output's attributes, then adjust `_AA_TO_ID`/`_MASK_ID`/`_prepare`/`_forward` in `containers/e1/entrypoint.py` accordingly and re-run Task 6's unit tests + this probe. The pure-helper and manifest tests are unaffected.

- [ ] **Step 6: Create the README**

Create `containers/e1/README.md`:

```markdown
# Profluent-E1 container (single-sequence mode)

A contract-compliant Docker image wrapping the
[Profluent-E1](https://huggingface.co/Profluent-Bio/E1-150m) masked protein
language model. It implements the protlms container contract (see
[`../../docs/CONTRACT.md`](../../docs/CONTRACT.md)) using the custom `E1` package,
and exposes the `manifest`, `embed`, `likelihood`, and `score` subcommands.

**Single-sequence mode only.** E1 can also run retrieval-augmented (with homolog
context); this image deliberately does not expose that — every sequence is scored
on its own.

The checkpoint is selected at build time via the `E1_CHECKPOINT` build arg and its
weights are baked into the image, so runtime requires no network access.

## Building

```bash
# 150m (demo / CI default)
docker build --build-arg E1_CHECKPOINT=E1-150m -t protlms-e1:150m containers/e1

# 300m / 600m
docker build --build-arg E1_CHECKPOINT=E1-300m -t protlms-e1:300m containers/e1
docker build --build-arg E1_CHECKPOINT=E1-600m -t protlms-e1:600m containers/e1
```

`E1_CHECKPOINT` accepts `E1-150m`, `E1-300m`, or `E1-600m`, resolved to
`Profluent-Bio/<name>`. The `E1` package is pinned to a specific git commit (see
the `E1_REF` build arg in the Dockerfile) for reproducibility.

## Running directly (debugging)

```bash
docker run --rm protlms-e1:150m manifest

docker run --rm -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-e1:150m embed --input /in/seqs.fasta --output /out --pooling mean

docker run --rm --gpus all -v "$PWD/in:/in:ro" -v "$PWD/out:/out:rw" \
  protlms-e1:150m likelihood --input /in/seqs.fasta --output /out
```

Normally you do not run these by hand — the `protlms` client builds these commands
for you (`protlms embed e1-150m seqs.fasta -o out/`).

## Models

| Checkpoint | Params | embedding_dim | layers |
|---|---|---|---|
| `E1-150m` | ~154M | 768 | 20 |
| `E1-300m` | ~274M | 1024 | 20 |
| `E1-600m` | ~641M | 1280 | 30 |

## Notes

- Uses the custom `E1` package (`E1ForMaskedLM`, `E1BatchPreparer`), which requires
  Python 3.12; this base image therefore differs from the ESM2/ProtBERT images.
- `likelihood` uses masked-marginal pseudo-log-likelihood (O(L) forward passes per
  sequence) and records `params.likelihood_method = "masked_marginal"`. The mask
  token is `?`.
- `embed` returns the **final-layer** representation; `--layers` must be `-1` (the
  client default). Other layer indices return an `InvalidInput` error. `cls`
  pooling uses the `<bos>` vector; `mean` averages over residue positions.
- `max_sequence_length = 2048` (the model supports longer within-sequence context,
  but masked-marginal scoring is O(L) forward passes).
- flash-attn is not installed; the model runs on CPU via the flex_attention
  fallback and uses the GPU when launched with `--gpus all` (recommended for the
  larger checkpoints).

## License & attribution

The `E1` **code** is Apache-2.0. The **weights** are free for research and
commercial use **with attribution** under Profluent's model license. If you use
this image or its outputs, follow the upstream attribution requirements in the
[E1 repository](https://github.com/Profluent-AI/E1) `NOTICE`/`ATTRIBUTION` files.
The paper text is separately licensed CC-BY-NC-ND.
```

- [ ] **Step 7: Commit**

```bash
git add containers/e1/Dockerfile containers/e1/README.md
git commit -m "e1: Dockerfile (python:3.12 base, pinned E1 package, baked weights) + README"
```

---

### Task 8: E1 end-to-end Docker integration test

Proves the model-backed subcommands work through the real `protlms` client against the built 150m image. Gated like the other integration tests.

**Files:**
- Create: `tests/test_integration_e1.py`
- Reuses: `tests/data/tiny.fasta`, `tests/data/variants.csv`.

**Interfaces:**
- Consumes: the registry entry `e1-150m` (Task 5), the image `ghcr.io/briney/protlms-e1:150m` (Task 7), `protlms.load`/`Model.embed`/`.likelihood`/`.score` (existing client API).
- Produces: nothing downstream (final task).

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration_e1.py`:

```python
"""End-to-end integration test against a locally built Profluent-E1 image.

Gated: runs only when ``PROTLMS_RUN_DOCKER_TESTS=1`` and a working Docker daemon is
available. Builds the ``E1-150m`` image if it is not already present, then drives
the real ``protlms`` client through embed, likelihood, and score on a small FASTA /
variants CSV of real protein sequences.

Note: E1 has no flash-attn in the image and uses the flex_attention fallback. If
the host is CPU-only and flex_attention cannot run on CPU for the installed torch,
run this test on a GPU host (the client auto-selects CUDA when available).
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

IMAGE = "ghcr.io/briney/protlms-e1:150m"
EMBEDDING_DIM = 768
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
def e1_image() -> str:
    """Ensure the 150m E1 image exists, building it if necessary."""
    present = (
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    )
    if not present:
        subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                "E1_CHECKPOINT=E1-150m",
                "-t",
                IMAGE,
                str(REPO_ROOT / "containers" / "e1"),
            ],
            check=True,
        )
    return IMAGE


@pytest.fixture(scope="session")
def model(e1_image: str) -> protlms.Model:
    return protlms.load("e1-150m")


def test_manifest_is_read_through_client(model: protlms.Model) -> None:
    assert model.manifest.name == "E1-150m"
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

Run: `PROTLMS_RUN_DOCKER_TESTS=1 pytest tests/test_integration_e1.py -v -m slow`
Expected: PASS. The session fixture builds `ghcr.io/briney/protlms-e1:150m` on first run (slow), then all five tests pass. This is where the E1 forward path (`out.logits`, `out.embeddings`), the residue-position detection, the masked-marginal PLL, and the `_AA_TO_ID` scoring map are proven correct end-to-end. The `self`-substitution scoring exactly 0 is invariant to the AA id mapping; the embed-shape `(26, 768)` assertion proves residue detection.

> **If embed shapes / forward attributes differ:** re-run the Task 7 Step 5 probe and adjust `_forward` / `_embed_one` (e.g. attribute names) accordingly, then re-run. If the test fails only on a CPU-only host with a flex_attention error, run it on a GPU host (this is the flagged top risk) — the image itself is still correct.

- [ ] **Step 3: Run the full unit suite to confirm no regressions**

Run: `pytest`
Expected: PASS (gated integration tests skipped without the env var). Then `ruff check src/ tests/`, `ruff format --check src/ tests/`, `ty check src/` clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_e1.py
git commit -m "test: end-to-end Profluent-E1 integration (embed/likelihood/score)"
```

---

## Self-Review

**1. Spec coverage** — every spec section maps to a task:
- Capabilities embed/likelihood/score, no generate → Tasks 2 & 6 (`build_parser` has no `generate`).
- Shared scoring (no `E1Scorer`) → Task 6 (`_score_variant` uses `_AA_TO_ID`, `cmd_score` uses the shared masked/wt-marginal helpers).
- ProtBERT both checkpoints, test prot_bert → Task 1 (both registry entries), Task 3 (build arg, both build commands), Task 4 (tests UniRef100).
- ProtBERT esm2-template + preprocessing shim → Task 2 (`preprocess`, applied in `_embed_one`/`_pseudo_log_likelihood`/`_masked_position_logprobs`/`_wt_position_logprobs`; `do_lower_case=False`).
- E1 all three sizes, test 150m, single-sequence only → Task 5 (three entries), Task 6 (`_prepare` passes one seq, no context), Task 7 (build args), Task 8 (tests 150m).
- E1 esm-c template, load-free `_MODEL_INFO`, final-layer-only → Task 6 (`build_manifest`, `cmd_embed` layer guard).
- E1 base image python:3.12, no flash-attn, pinned package → Task 7 (Dockerfile, `E1_REF`).
- No contract / client-logic / publish-workflow changes → Global Constraints + only `models.yaml`/tests touched.
- Manifest fields → Task 2 (ProtBERT, via AutoConfig; smoke-tested in Task 3) + Task 6 (E1 unit-tested) + Task 7 smoke-test.
- Masked-marginal likelihood + `likelihood_method` → Tasks 2 & 6 `_pseudo_log_likelihood` + `_write_capability_result`; Tasks 4 & 8 assertions.
- Score masked/wt-marginal → Tasks 2 & 6 `cmd_score` + helpers; Tasks 4 & 8 assertions.
- Dockerfiles + READMEs (incl. E1 attribution, ProtBERT tokenization note) → Tasks 3 & 7.
- Tests (unit pure-helper, registry, gated integration) → Tasks 1, 2, 4, 5, 6, 8.

**2. Placeholder scan** — no TBD/TODO; every code step has complete content. The single intentional fill-in is `<E1_SHA>` in Task 7, resolved by the explicit `git ls-remote` step that precedes it — an actionable instruction, not deferred work. The two "fallback" call-outs (E1 build failure, E1 API mismatch) are explicit, bounded contingencies with concrete remedies, not vague error handling; the primary path is fully specified.

**3. Type/name consistency** — checkpoint strings (`prot_bert`/`prot_bert_bfd`, `E1-150m`/`E1-300m`/`E1-600m`), image refs (`ghcr.io/briney/protlms-protbert:uniref100`/`:bfd`, `ghcr.io/briney/protlms-e1:{150m,300m,600m}`), registry names (`protbert`/`protbert-bfd`, `e1-150m`/`e1-300m`/`e1-600m`), `model_family` (`protbert`/`e1`), `EMBEDDING_DIM` (1024 ProtBERT / 768 E1-150m), and helper names are consistent across each entrypoint, its unit tests, Dockerfile, README, and integration test. Note the deliberate signature difference: ProtBERT's `_score_variant(mutant, wt_seq, logp_by_pos, tokenizer)` takes a tokenizer (uses `convert_tokens_to_ids`), while E1's `_score_variant(mutant, wt_seq, logp_by_pos)` omits it (uses the constant `_AA_TO_ID`) — each is internally consistent with its own `cmd_score` call site.

## Deviations from the spec (flagged)

1. **No publish-workflow edits.** The spec listed "publish-workflow matrix rows," but `.github/workflows/publish-image.yaml` is registry-driven (resolves build metadata from `models.yaml` via `scripts.registry_publish lookup`). Adding registry entries with a `build:` block is sufficient; the workflow is untouched.
2. **E1 base image = `python:3.12-slim-bookworm`**, not the spec's `pytorch/pytorch:2.7.x` — the official PyTorch images ship Python 3.11, but E1 requires Python ≥3.12. torch (≥2.7) is pulled transitively by `pip install E1`. This mirrors the esm-c container's resolution of the same constraint.
3. **E1 scoring/vocab via documented constants** (`_MASK_ID=5`, `_AA_TO_ID` for `A..Z` at `8..33`) rather than a live E1 tokenizer object — this keeps `_score_variant` pure and unit-testable and decouples scoring from the (less-documented) E1 tokenizer API. The constants are verified against the running model by the Task 7 Step 5 probe and the Task 8 integration test, with a concrete fallback if they differ.
