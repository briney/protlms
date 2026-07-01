#!/usr/bin/env python
"""Contract-compliant entrypoint for the ESM-C model image.

Implements the protlms container contract (see docs/CONTRACT.md) for the ESM-C
masked protein language model via HuggingFace ``transformers`` and the biohub
``esm`` package's weights. Exposes the ``manifest``, ``embed``, ``likelihood``,
``score``, and ``contacts`` subcommands plus a hidden ``_prefetch`` used at
build time to bake weights into the image.

Heavy imports (``torch``, ``transformers``, ``esm``) happen inside functions
that need them, so the pure helpers can be unit-tested without the ML stack
installed. Importing the biohub ``esm`` package registers the
``ESMCForMaskedLM`` architecture with transformers' auto classes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

CONTRACT_VERSION = "0.4"
MAX_SEQUENCE_LENGTH = 2048
DEFAULT_BATCH_SIZE = 8
DEFAULT_CHECKPOINT = os.environ.get("ESMC_CHECKPOINT", "esmc_300m")
MODEL_FAMILY = "esm-c"

# checkpoint name -> architecture facts (keeps `manifest` load-free; verified from
# each biohub repo's config.json: d_model / n_layers).
_MODEL_INFO: dict[str, dict[str, object]] = {
    "esmc_300m": {
        "hf_id": "biohub/ESMC-300M",
        "embedding_dim": 960,
        "num_layers": 30,
        "min_gpu_memory_gb": None,
    },
    "esmc_600m": {
        "hf_id": "biohub/ESMC-600M",
        "embedding_dim": 1152,
        "num_layers": 36,
        "min_gpu_memory_gb": 4.0,
    },
    "esmc_6b": {
        "hf_id": "biohub/ESMC-6B",
        "embedding_dim": 2560,
        "num_layers": 80,
        "min_gpu_memory_gb": 24.0,
    },
}

_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_MUTANT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


# --- pure helpers (unit-testable without torch) ----------------------------


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
    """Load the tokenizer and masked-LM model for the configured biohub checkpoint."""
    import esm  # noqa: F401 - registers ESMCForMaskedLM with transformers auto classes
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    hf_id = _MODEL_INFO[DEFAULT_CHECKPOINT]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForMaskedLM.from_pretrained(hf_id, torch_dtype=torch.float32)
    model.eval().to(device)
    return tokenizer, model


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
                # ESMCConfig has no `hidden_size` attribute (it uses `d_model`); use the
                # load-free architecture table instead of reading the model config.
                "shape": [len(ids), _MODEL_INFO[DEFAULT_CHECKPOINT]["embedding_dim"]],
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
    import esm  # noqa: F401 - registers ESMCForMaskedLM
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    hf_id = _MODEL_INFO[DEFAULT_CHECKPOINT]["hf_id"]
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
    special = tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
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
            "model_name": DEFAULT_CHECKPOINT,
            "n_input_records": len(records),
            "n_output_records": len(records),
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

    contacts = sub.add_parser("contacts")
    contacts.add_argument("--input", required=True)
    contacts.add_argument("--output", required=True)
    contacts.add_argument(
        "--method", default="categorical-jacobian", choices=["categorical-jacobian"]
    )
    contacts.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    contacts.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    contacts.set_defaults(func=cmd_contacts)

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
