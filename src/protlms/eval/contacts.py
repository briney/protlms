"""PDB contact-map extraction and long-range precision@L.

Depends on BioPython (PDB parsing) and numpy (metric). No ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

if TYPE_CHECKING:
    from pathlib import Path

#: Contact distance cutoff (Cβ–Cβ, Cα for glycine), in Angstroms.
CONTACT_THRESHOLD_ANGSTROM = 8.0
#: Minimum sequence separation |i − j| for a "long-range" pair.
LONG_RANGE_SEP = 24

#: Three-letter → one-letter codes for the 20 standard amino acids.
_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
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
        # noqa justification: Bio.PDB.Residue has no `.get` method (unlike a dict),
        # so ruff's SIM401 "use .get()" rewrite would raise AttributeError.
        if "CB" in residue:
            atom = residue["CB"]
        elif "CA" in residue:  # noqa: SIM401
            atom = residue["CA"]
        else:
            atom = None
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
