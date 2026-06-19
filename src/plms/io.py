"""Input/output handling: FASTA parsing, input staging, and output parsing.

This module touches the filesystem but knows nothing about Docker. ``numpy`` is
imported only to load result arrays produced by containers.
"""

from __future__ import annotations

import csv
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from plms.contract import ArtifactKind, Result
from plms.exceptions import FastaError, OutputParseError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

#: The fixed filename a staged FASTA is normalized to inside the input mount.
STAGED_FASTA_NAME = "seqs.fasta"

#: Numeric columns in the likelihoods CSV and the type to coerce them to.
_LIKELIHOOD_COLUMN_TYPES: dict[str, type] = {
    "seq_len": int,
    "pseudo_log_likelihood": float,
    "mean_pseudo_log_likelihood": float,
    "pseudo_perplexity": float,
}


@dataclass(frozen=True)
class FastaRecord:
    """A single FASTA record.

    Attributes:
        id: The first whitespace-delimited token of the header.
        description: The full header line (without the leading ``>``).
        sequence: The (newline-joined, uppercased) residue sequence.
    """

    id: str
    description: str
    sequence: str


def read_fasta(path: Path) -> list[FastaRecord]:
    """Parse a FASTA file into records.

    Args:
        path: Path to a FASTA file.

    Returns:
        The records in file order (empty list for an empty file).

    Raises:
        FastaError: If sequence data appears before any header line.
    """
    records: list[FastaRecord] = []
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if header is not None:
            record_id = header.split(maxsplit=1)[0] if header.split() else header
            records.append(FastaRecord(record_id, header, "".join(chunks).upper()))

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            chunks = []
        else:
            if header is None:
                raise FastaError(f"sequence data before any '>' header in {path}")
            chunks.append(line)
    flush()
    return records


def write_fasta(records: Iterable[FastaRecord], path: Path) -> None:
    """Write records to ``path`` as FASTA, using only the record id as header."""
    lines = [f">{record.id}\n{record.sequence}\n" for record in records]
    path.write_text("".join(lines))


@dataclass(frozen=True)
class StagedInput:
    """The host input mount for one staged container run."""

    input_dir: Path
    input_filename: str = STAGED_FASTA_NAME

    @property
    def container_input_path(self) -> str:
        """The input path as seen inside the container (under ``/in``)."""
        return f"/in/{self.input_filename}"


@contextmanager
def stage_inputs(records: list[FastaRecord]) -> Iterator[StagedInput]:
    """Stage records into a temporary, read-only-bound input directory.

    Writes a normalized FASTA (id-only headers) into a temporary directory that
    is removed on context exit. The output directory is managed separately by
    the caller, since outputs must outlive the run.

    Args:
        records: The records to stage.

    Yields:
        A :class:`StagedInput` pointing at the host input directory.

    Raises:
        FastaError: If two records share the same id.
    """
    ids = [r.id for r in records]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise FastaError(f"duplicate record ids in input: {sorted(duplicates)}")

    with tempfile.TemporaryDirectory(prefix="plms-in-") as tmp:
        input_dir = Path(tmp)
        write_fasta(records, input_dir / STAGED_FASTA_NAME)
        yield StagedInput(input_dir=input_dir)


def read_result(out_dir: Path) -> Result:
    """Load and parse ``result.json`` from a container's output directory.

    Args:
        out_dir: The host output directory the container wrote to.

    Returns:
        The parsed :class:`~plms.contract.Result`.

    Raises:
        OutputParseError: If ``result.json`` is missing or malformed.
    """
    result_path = out_dir / "result.json"
    if not result_path.is_file():
        raise OutputParseError(f"no result.json found in {out_dir}")
    try:
        return Result.model_validate_json(result_path.read_text())
    except ValueError as exc:  # pydantic ValidationError is a ValueError
        raise OutputParseError(f"malformed result.json in {out_dir}: {exc}") from exc


def _artifacts(result: Result, kind: ArtifactKind) -> list:
    return [a for a in result.artifacts if a.kind == kind]


def load_pooled_embeddings(out_dir: Path, result: Result) -> dict[str, np.ndarray]:
    """Load pooled embeddings (one vector per record) keyed by record id.

    Raises:
        OutputParseError: If no pooled-embeddings artifact is present.
    """
    artifacts = _artifacts(result, ArtifactKind.POOLED_EMBEDDINGS)
    if not artifacts:
        raise OutputParseError("result declares no pooled_embeddings artifact")
    with np.load(out_dir / artifacts[0].path) as npz:
        return {key: npz[key] for key in npz.files}


def load_per_residue_embeddings(out_dir: Path, result: Result) -> dict[str, np.ndarray]:
    """Load per-residue embeddings keyed by record id (the artifact filename stem)."""
    out: dict[str, np.ndarray] = {}
    for artifact in _artifacts(result, ArtifactKind.PER_RESIDUE_EMBEDDINGS):
        out[Path(artifact.path).stem] = np.load(out_dir / artifact.path)
    if not out:
        raise OutputParseError("result declares no per_residue_embeddings artifacts")
    return out


def read_likelihoods(out_dir: Path, result: Result) -> list[dict[str, str | int | float]]:
    """Read the likelihoods CSV, coercing known numeric columns.

    Raises:
        OutputParseError: If no likelihoods artifact is present.
    """
    artifacts = _artifacts(result, ArtifactKind.LIKELIHOODS_CSV)
    if not artifacts:
        raise OutputParseError("result declares no likelihoods_csv artifact")
    rows: list[dict[str, str | int | float]] = []
    with (out_dir / artifacts[0].path).open(newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row: dict[str, str | int | float] = {}
            for key, value in raw_row.items():
                caster = _LIKELIHOOD_COLUMN_TYPES.get(key, str)
                row[key] = caster(value)
            rows.append(row)
    return rows
