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
        Empty if every parsed target is excluded by ``max_length`` or has no
        usable predicted map.

    Raises:
        InvalidRequestError: If no ``.pdb`` file in ``pdb_dir`` could be parsed.
    """
    pdb_paths = sorted(Path(pdb_dir).glob("*.pdb"))
    chains = {}
    records: list[FastaRecord] = []
    parsed_any = False
    for path in pdb_paths:
        target_id = path.stem
        try:
            chain = parse_pdb(path)
        except (ValueError, KeyError) as exc:
            logger.warning("skipping %s: %s", target_id, exc)
            continue
        parsed_any = True
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

    if not parsed_any:
        raise InvalidRequestError(f"no usable .pdb targets found in {pdb_dir}")
    if not records:
        return []

    with tempfile.TemporaryDirectory(prefix="protlms-eval-") as tmp:
        fasta = Path(tmp) / "targets.fasta"
        write_fasta(records, fasta)
        maps = model.contacts(fasta, use_gpu=use_gpu, batch_size=batch_size).maps()

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
