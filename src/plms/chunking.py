"""Client-side input chunking: split a large input into per-chunk container
runs and merge the outputs into one logical result.

This module is the only place that knows how to shard a request across multiple
container runs. It reuses :mod:`plms.io` for file I/O and drives runs through a
caller-supplied closure, so it depends on no Docker specifics. The contract,
the containers, and ``score`` are unaffected.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from plms.contract import ArtifactKind, OutputArtifact, Result
from plms.exceptions import FastaError, InvalidRequestError
from plms.io import load_pooled_embeddings, read_fasta

if TYPE_CHECKING:
    from plms.io import FastaRecord

logger = logging.getLogger(__name__)

CHUNKS_DIRNAME = "chunks"
CHUNKING_MANIFEST_NAME = "chunking.json"


def chunk_records(records: list[FastaRecord], chunk_size: int) -> list[list[FastaRecord]]:
    """Split records into consecutive chunks of at most ``chunk_size`` (file order)."""
    if chunk_size < 1:
        raise InvalidRequestError(f"chunk_size must be >= 1, got {chunk_size}")
    return [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]


def _check_unique_ids(records: list[FastaRecord]) -> None:
    """Raise if any record id repeats across the whole input (before splitting)."""
    ids = [r.id for r in records]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise FastaError(f"duplicate record ids in input: {dupes}")


def _input_fingerprint(records: list[FastaRecord]) -> str:
    """A stable hash of the ordered record ids — the chunking input fingerprint."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def merge_chunk_outputs(
    capability: str,
    pairs: list[tuple[Path, Result]],
    output_dir: Path,
) -> Result:
    """Merge per-chunk outputs into ``output_dir`` and return a synthesized Result.

    Args:
        capability: ``embed``, ``likelihood``, or ``generate``.
        pairs: ``(chunk_dir, chunk_result)`` in chunk order.
        output_dir: Where merged artifacts and the merged ``result.json`` are written.
    """
    artifacts = _merge_artifacts(capability, pairs, output_dir)
    first = pairs[0][1]
    merged = Result(
        contract_version=first.contract_version,
        capability=first.capability,
        model_name=first.model_name,
        n_input_records=sum(r.n_input_records for _, r in pairs),
        n_output_records=sum(r.n_output_records for _, r in pairs),
        artifacts=artifacts,
        warnings=[w for _, r in pairs for w in r.warnings],
        params=first.params,
    )
    (output_dir / "result.json").write_text(merged.model_dump_json(indent=2))
    return merged


def _merge_artifacts(
    capability: str, pairs: list[tuple[Path, Result]], output_dir: Path
) -> list[OutputArtifact]:
    if capability == "embed":
        kinds = {a.kind for _, r in pairs for a in r.artifacts}
        if ArtifactKind.POOLED_EMBEDDINGS.value in kinds:
            return _merge_pooled(pairs, output_dir)
        return _merge_per_residue(pairs, output_dir)
    if capability == "likelihood":
        return [_merge_csv(pairs, output_dir, "likelihoods.csv", ArtifactKind.LIKELIHOODS_CSV)]
    if capability == "generate":
        return [_merge_fasta(pairs, output_dir)]
    raise InvalidRequestError(f"chunking does not support capability {capability!r}")


def _merge_pooled(pairs: list[tuple[Path, Result]], output_dir: Path) -> list[OutputArtifact]:
    merged: dict[str, np.ndarray] = {}
    for chunk_dir, result in pairs:
        merged.update(load_pooled_embeddings(chunk_dir, result))
    np.savez(output_dir / "embeddings.npz", **merged)
    sample = next(iter(merged.values()))
    return [
        OutputArtifact(
            path="embeddings.npz",
            kind=ArtifactKind.POOLED_EMBEDDINGS.value,
            record_ids=list(merged),
            shape=[len(merged), int(sample.shape[0])],
            dtype=str(sample.dtype),
        )
    ]


def _merge_per_residue(pairs: list[tuple[Path, Result]], output_dir: Path) -> list[OutputArtifact]:
    pr_dir = output_dir / "per_residue"
    pr_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[OutputArtifact] = []
    for chunk_dir, result in pairs:
        for artifact in result.artifacts:
            if artifact.kind != ArtifactKind.PER_RESIDUE_EMBEDDINGS.value:
                continue
            name = Path(artifact.path).name
            shutil.copyfile(chunk_dir / artifact.path, pr_dir / name)
            artifacts.append(
                OutputArtifact(
                    path=f"per_residue/{name}",
                    kind=ArtifactKind.PER_RESIDUE_EMBEDDINGS.value,
                    record_ids=[Path(name).stem],
                    shape=artifact.shape,
                    dtype=artifact.dtype,
                )
            )
    return artifacts


def _merge_csv(
    pairs: list[tuple[Path, Result]], output_dir: Path, filename: str, kind: ArtifactKind
) -> OutputArtifact:
    header: str | None = None
    data_lines: list[str] = []
    for chunk_dir, result in pairs:
        artifact = next(a for a in result.artifacts if a.kind == kind.value)
        lines = (chunk_dir / artifact.path).read_text().splitlines()
        if not lines:
            continue
        if header is None:
            header = lines[0]
        data_lines.extend(lines[1:])
    (output_dir / filename).write_text("\n".join([header or "", *data_lines]) + "\n")
    ids = [row[0] for row in csv.reader(data_lines) if row]
    return OutputArtifact(path=filename, kind=kind.value, record_ids=ids)


def _merge_fasta(pairs: list[tuple[Path, Result]], output_dir: Path) -> OutputArtifact:
    parts: list[str] = []
    for chunk_dir, result in pairs:
        artifact = next(a for a in result.artifacts if a.kind == ArtifactKind.GENERATED_FASTA.value)
        text = (chunk_dir / artifact.path).read_text()
        if text and not text.endswith("\n"):
            text += "\n"
        parts.append(text)
    out_path = output_dir / "generated.fasta"
    out_path.write_text("".join(parts))
    ids = [rec.id for rec in read_fasta(out_path)]
    return OutputArtifact(
        path="generated.fasta", kind=ArtifactKind.GENERATED_FASTA.value, record_ids=ids
    )
