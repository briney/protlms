#!/usr/bin/env python
"""Contract-compliant entrypoint for the ESM2 model image.

Implements the protlms container contract (see docs/CONTRACT.md) for the ESM2
masked protein language model via HuggingFace ``transformers``. Exposes the
``manifest``, ``embed``, ``likelihood``, and ``score`` subcommands plus a hidden
``_prefetch`` used at build time to bake weights into the image.

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
DEFAULT_CHECKPOINT = os.environ.get("ESM2_CHECKPOINT", "esm2_t6_8M")

_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_MUTANT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


# --- pure helpers (unit-testable without torch) ----------------------------


def resolve_hf_id(checkpoint: str) -> str:
    """Resolve a short ESM2 checkpoint name to a HuggingFace model id.

    ``esm2_t6_8M`` -> ``facebook/esm2_t6_8M_UR50D``. A value already containing
    ``/`` is treated as a full HuggingFace id and returned unchanged.
    """
    if "/" in checkpoint:
        return checkpoint
    suffix = "" if checkpoint.endswith("_UR50D") else "_UR50D"
    return f"facebook/{checkpoint}{suffix}"


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
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
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
        "description": f"ESM2 masked protein language model ({DEFAULT_CHECKPOINT}).",
        "model_family": "esm2",
        "capabilities": ["embed", "likelihood", "score"],
        "embedding_dim": int(config.hidden_size),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "pooling_modes": ["mean", "cls", "none"],
        "num_layers": int(config.num_hidden_layers),
        "min_gpu_memory_gb": None,
        "default_batch_size": DEFAULT_BATCH_SIZE,
    }


def _truncate(seq: str, warnings: list[str], record_id: str) -> str:
    if len(seq) > MAX_SEQUENCE_LENGTH:
        warnings.append(f"sequence {record_id!r} truncated to {MAX_SEQUENCE_LENGTH} residues")
        return seq[:MAX_SEQUENCE_LENGTH]
    return seq


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

    enc = tokenizer(seq, return_tensors="pt").to(device)
    use_amp = device == "cuda"
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
        out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[layer][0].float().cpu().numpy()  # (T, D)
    residue = hidden[1 : 1 + len(seq)].astype(np.float32)  # strip <cls>/<eos>
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

    enc = tokenizer(seq, return_tensors="pt")
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
    AutoTokenizer.from_pretrained(hf_id)
    AutoModelForMaskedLM.from_pretrained(hf_id)
    print(f"prefetched {hf_id}")


def _masked_position_logprobs(tokenizer, model, seq, positions, batch_size, device):  # noqa: ANN001
    """Map each 1-indexed position to its masked log-softmax vector over the vocab."""
    import torch

    input_ids = tokenizer(seq, return_tensors="pt")["input_ids"][0]
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
    parser = argparse.ArgumentParser(prog="esm2", description="ESM2 protlms contract entrypoint.")
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
