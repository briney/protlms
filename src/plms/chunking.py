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
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from plms.contract import ArtifactKind, OutputArtifact, Result
from plms.exceptions import FastaError, InvalidRequestError, OutputParseError
from plms.io import load_pooled_embeddings, read_fasta, read_result

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
