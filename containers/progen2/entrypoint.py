#!/usr/bin/env python
"""Contract-compliant entrypoint for the ProGen2 model image.

Implements the protlms container contract (docs/CONTRACT.md) for the ProGen2
autoregressive protein language model via a HuggingFace community port loaded
with trust_remote_code. Exposes manifest / generate / likelihood / _prefetch.

Port-specific notes (hugohrban/progen2-small):
- Tokenizer is ``tokenizers.Tokenizer`` (HF tokenizers library), not AutoTokenizer.
  Load via ``Tokenizer.from_pretrained(hf_id)``.
- Special tokens: PAD=0 ``<|pad|>``, BOS=1 ``<|bos|>``, EOS=2 ``<|eos|>``.
- Numeric tokens "1" and "2" in sequences are amino-acid context markers
  (e.g. "1SEQUENCE2" = organism-conditioned), NOT control tokens.
- model.generate() is not available; generation uses a manual sampling loop
  with forward passes.

torch and transformers are imported inside functions so the pure helpers stay
importable without the ML stack.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

CONTRACT_VERSION = "0.3"
MAX_SEQUENCE_LENGTH = 1024
DEFAULT_BATCH_SIZE = 8
DEFAULT_CHECKPOINT = os.environ.get("PROGEN2_CHECKPOINT", "progen2-small")

# Non-amino-acid characters to drop from decoded output.
# Step 1: strip angle-bracket tokens like <|pad|>, <|bos|>, <|eos|>.
_ANGLE_TOKEN = re.compile(r"<\|[^|>]*\|>")
# Step 2: drop anything not a canonical amino-acid letter.
_NON_AA = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


# --- pure helpers (unit-testable without torch) ----------------------------


def resolve_hf_id(checkpoint: str) -> str:
    """Resolve a short ProGen2 name to a HuggingFace port id.

    Args:
        checkpoint: Short name like ``progen2-small`` or a full HuggingFace id.

    Returns:
        Full HuggingFace model id, e.g. ``hugohrban/progen2-small``.
    """
    if "/" in checkpoint:
        return checkpoint
    return f"hugohrban/{checkpoint}"


def strip_special_tokens(text: str) -> str:
    """Reduce a decoded ProGen2 string to canonical amino-acid letters.

    Removes numeric context markers ("1", "2"), pad tokens (``<|pad|>``),
    BOS/EOS markers, and any other non-amino-acid characters.

    Args:
        text: Raw decoded string from the ProGen2 tokenizer.

    Returns:
        Uppercase string containing only standard amino-acid letters.
    """
    text = _ANGLE_TOKEN.sub("", text)
    return _NON_AA.sub("", text.upper())


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


def pick_device(requested: str | None) -> str:
    """Choose the torch device, validating an explicit ``cuda`` request."""
    import torch

    available = "cuda" if torch.cuda.is_available() else "cpu"
    if requested in (None, "auto"):
        return available
    if requested == "cuda" and available != "cuda":
        emit_error_and_exit("DeviceUnavailable", "cuda requested but no GPU is available")
    return requested


def load_tokenizer(hf_id: str):  # noqa: ANN201
    """Load the ProGen2 tokenizer from the HF port.

    Uses ``tokenizers.Tokenizer`` (not AutoTokenizer) as required by the port.
    Padding is disabled for generation.
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(hf_id)
    tok.no_padding()
    return tok


def load_model(hf_id: str, device: str):  # noqa: ANN201
    """Load the ProGen2 causal LM model.

    Args:
        hf_id: Full HuggingFace model id.
        device: Torch device string (``cpu`` or ``cuda``).

    Returns:
        Loaded and eval-mode model on the requested device.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval().to(device)
    return model


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into ``(id, sequence)`` tuples.

    An empty sequence (header only, or header followed by blank lines) is
    returned as an empty string — used for unconditional generation prompts.
    """
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
    """Build the manifest dict from the checkpoint's config.

    Returns:
        Manifest dict conforming to contract version 0.3.
    """
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


# --- generation helpers ----------------------------------------------------


def _sample_sequence(
    model,
    tokenizer,
    prompt_ids: list[int],
    max_length: int,
    temperature: float,
    top_p: float,
    device: str,
    eos_id: int,
) -> list[int]:
    """Sample a single sequence using a manual autoregressive loop.

    Args:
        model: Loaded ProGen2 causal LM.
        tokenizer: ProGen2 tokenizers.Tokenizer instance.
        prompt_ids: Token ids for the conditioning prefix (may be empty for
            unconditional sampling).
        max_length: Maximum total sequence length including prompt.
        temperature: Sampling temperature.
        top_p: Nucleus sampling cumulative probability threshold.
        device: Torch device.
        eos_id: Token id for end-of-sequence; generation stops on emission.

    Returns:
        List of generated token ids (prompt + new tokens, without EOS).
    """
    import torch
    import torch.nn.functional as F

    generated = list(prompt_ids)
    input_tensor = torch.tensor([generated], dtype=torch.long, device=device)

    with torch.no_grad():
        past_key_values = None
        for _ in range(max_length - len(generated)):
            outputs = model(input_tensor, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]  # (1, vocab)
            past_key_values = outputs.past_key_values

            if temperature > 0:
                logits = logits / temperature

            probs = F.softmax(logits.float(), dim=-1)

            # Nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                # Remove tokens once cumulative prob exceeds top_p (keep at least one)
                remove_mask = cumprobs - sorted_probs > top_p
                sorted_probs[remove_mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_idx.gather(-1, next_token)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            tok_id = int(next_token.item())
            if tok_id == eos_id:
                break
            generated.append(tok_id)
            input_tensor = next_token  # feed only the last token when using cache

    return generated


def _encode_prompt(tokenizer, prefix: str) -> list[int]:
    """Encode a sequence prefix for use as a generation prompt.

    For an empty prefix (unconditional), returns just the BOS token (id=1).
    Otherwise encodes the prefix directly (the tokenizer does not add BOS
    automatically with the hugohrban port).

    Args:
        tokenizer: ProGen2 tokenizers.Tokenizer instance.
        prefix: Raw amino-acid prefix string (may be empty).

    Returns:
        List of token ids.
    """
    BOS_ID = 1
    if not prefix:
        return [BOS_ID]
    return tokenizer.encode(prefix).ids


def _decode_ids(tokenizer, ids: list[int]) -> str:
    """Decode token ids to a string, then strip to canonical amino acids.

    Args:
        tokenizer: ProGen2 tokenizers.Tokenizer instance.
        ids: Token id list (may include special tokens).

    Returns:
        Clean uppercase amino-acid string.
    """
    text = tokenizer.decode(ids)
    return strip_special_tokens(text)


# --- subcommands -----------------------------------------------------------


def cmd_manifest(_args: argparse.Namespace) -> None:
    """Print manifest JSON to stdout."""
    print(json.dumps(build_manifest()))


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate protein sequences from prompts.

    Each FASTA record's sequence is used as a prefix; an empty sequence
    triggers unconditional sampling. Outputs ``generated.fasta`` and
    ``result.json`` to the output directory.
    """
    import torch

    device = pick_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    hf_id = resolve_hf_id(DEFAULT_CHECKPOINT)
    tokenizer = load_tokenizer(hf_id)
    model = load_model(hf_id, device)

    max_length = args.max_length or MAX_SEQUENCE_LENGTH
    records = read_fasta(Path(args.input))

    EOS_ID = 2  # <|eos|> token id in hugohrban/progen2-* port

    out_lines: list[str] = []
    out_ids: list[str] = []
    warnings: list[str] = []
    for rid, prefix in records:
        prompt_ids = _encode_prompt(tokenizer, prefix)
        if len(prompt_ids) >= max_length:
            warnings.append(f"prompt {rid!r} length >= max_length; no new tokens generated")
        for k in range(args.num_samples):
            generated_ids = _sample_sequence(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                max_length=max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                eos_id=EOS_ID,
            )
            seq = _decode_ids(tokenizer, generated_ids)
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
            "warnings": warnings,
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
    """Compute causal left-to-right log-likelihoods for sequences.

    Outputs ``likelihoods.csv`` and ``result.json`` to the output directory.
    """
    device = pick_device(args.device)
    hf_id = resolve_hf_id(DEFAULT_CHECKPOINT)
    tokenizer = load_tokenizer(hf_id)
    model = load_model(hf_id, device)

    records = read_fasta(Path(args.input))
    rows = ["record_id,seq_len,log_likelihood,mean_log_likelihood,perplexity"]
    warnings: list[str] = []
    n_scored = 0
    for rid, seq in records:
        if not seq:
            warnings.append(f"skipping empty sequence {rid!r}")
            continue
        if len(seq) > MAX_SEQUENCE_LENGTH:
            warnings.append(f"sequence {rid!r} truncated to {MAX_SEQUENCE_LENGTH}")
            seq = seq[:MAX_SEQUENCE_LENGTH]
        ll = _causal_log_likelihood(tokenizer, model, seq, device)
        # Assumes one token per residue (true for ProGen2 amino-acid vocabulary).
        mean = ll / max(len(seq), 1)
        rows.append(f"{rid},{len(seq)},{ll:.6f},{mean:.6f},{math.exp(-mean):.6f}")
        n_scored += 1
    output_dir = Path(args.output)
    (output_dir / "likelihoods.csv").write_text("\n".join(rows) + "\n")
    write_result(
        output_dir,
        {
            "contract_version": CONTRACT_VERSION,
            "capability": "likelihood",
            "model_name": DEFAULT_CHECKPOINT,
            "n_input_records": len(records),
            "n_output_records": n_scored,
            "artifacts": [{"path": "likelihoods.csv", "kind": "likelihoods_csv"}],
            "warnings": warnings,
            "params": {"likelihood_method": "causal", "device": args.device or "auto"},
        },
    )


def _causal_log_likelihood(tokenizer, model, seq: str, device: str) -> float:  # noqa: ANN001
    """Compute true causal log-likelihood for a sequence.

    Sums log P(token_t | token_{0..t-1}) over all positions using a single
    forward pass.

    Convention note: the score is the left-to-right log-likelihood over the
    residue-token sequence as encoded by the tokenizer, WITHOUT a prepended BOS
    control token. It is internally consistent and comparable across sequences
    scored by the same model, but is NOT directly comparable to ESM2's
    masked-marginal pseudo-log-likelihood, which uses a different scoring
    objective (masked language modelling).

    Args:
        tokenizer: ProGen2 tokenizers.Tokenizer instance.
        model: Loaded ProGen2 causal LM.
        seq: Amino-acid sequence string.
        device: Torch device string.

    Returns:
        Total log-likelihood (sum over positions).
    """
    import torch

    ids = tokenizer.encode(seq).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids).logits  # (1, L, V)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    # Causal: position t predicts token t+1
    targets = input_ids[0, 1:]  # (L-1,)
    token_lp = log_probs[0, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(token_lp.sum())


def cmd_prefetch(_args: argparse.Namespace) -> None:
    """Bake weights into the image at build time (populate the HF cache)."""
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    hf_id = resolve_hf_id(DEFAULT_CHECKPOINT)
    Tokenizer.from_pretrained(hf_id)
    AutoModelForCausalLM.from_pretrained(hf_id, trust_remote_code=True)
    print(f"prefetched {hf_id}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser with all contract subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="progen2", description="ProGen2 protlms contract entrypoint."
    )
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
    """Entry point for the ProGen2 container CLI.

    Args:
        argv: Optional argument list; defaults to sys.argv.
    """
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - top-level: report as structured error
        emit_error_and_exit("InternalError", str(exc), exception=type(exc).__name__)


if __name__ == "__main__":
    main()
