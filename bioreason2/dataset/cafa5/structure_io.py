# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Structure-file parsing, kept free of training-only dependencies.

`collate.py` imports the trainer's processor and protein modules, which pull in
`trl`. eval.py runs in the vLLM environment where it is absent by design, so
importing the coordinate parsers from `collate` made eval die with
ModuleNotFoundError the moment structure conditioning was switched on. These
functions need only numpy, torch, Biopython and esm, so they live here and
`collate` re-exports them.
"""

import numpy as np
import torch
from Bio.PDB import MMCIFParser, is_aa
from esm.sdk.api import ESMProtein


def _coords_from_cif(cif_path, chain_id="A", atom_order=["N", "CA", "C"]):
    """Load protein structure coordinates from a CIF file.

    Args:
        cif_path (str): Path to the CIF file.
        chain_id (str): Chain ID to extract. Defaults to "A".
        atom_order (list): List of atom names to extract in order. Defaults to ["N", "CA", "C"].

    Returns:
        numpy.ndarray: Coordinate array with shape [L, len(atom_order), 3] where L is the number of residues.
    """
    # Parse with BioPython
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)
    model = structure[0]

    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} not found in {cif_path}")

    chain = model[chain_id]
    coords = []

    # Collect residue info
    for residue in chain:
        if not is_aa(residue, standard=True):
            continue

        atom_coords = []
        for atom_name in atom_order:
            try:
                coord = residue[atom_name].get_coord()
            except KeyError:
                coord = np.full(3, np.nan)
            atom_coords.append(coord)
        coords.append(atom_coords)

    coords = np.array(coords)  # shape: [L, len(atom_order), 3]

    return coords


def _coords_from_pdb(pdb_path, chain_id="A"):
    """Load protein structure coordinates from a PDB file.

    Args:
        pdb_path (str): Path to the PDB file.
        chain_id (str): Chain ID to extract. Defaults to "A".

    Returns:
        numpy.ndarray: Coordinate array with shape [L, 3, 3] (N, CA, C).
    """
    coords = ESMProtein.from_pdb(pdb_path, chain_id=chain_id).coordinates
    # `from_pdb` returns ESM3's atom37 representation as a torch.Tensor of shape
    # [L, 37, 3], while `_coords_from_cif` returns an ndarray of shape [L, 3, 3].
    # The callers assume the CIF form -- the padding buffer is [B, L, 3, 3] and
    # uses torch.from_numpy -- so a .pdb would raise TypeError and, if it did
    # not, would not fit the buffer. Normalise here: atom37 indices 0/1/2 are
    # N/CA/C (verified against N-CA 1.465 A and CA-C 1.532 A bond lengths).
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    coords = np.asarray(coords)
    if coords.ndim == 3 and coords.shape[1] > 3:
        coords = coords[:, :3, :]
    return np.ascontiguousarray(coords)


# --- sequence-verified loading --------------------------------------------
# The collator's only check was `coords.shape[0] >= len(sequence)`, then it took
# coords[:len(sequence)]. That never verifies the structure belongs to the
# protein, so a mismatched file produced confident, silently wrong embeddings.
#
# Measured over the 8,630-protein test set (6,295 with a structure):
#   exact                       5,994
#   prefix / near-identical        85   usable
#   OFFSET_by_N                    66   right protein, WRONG WINDOW under the old code
#   internal fragment              38
#   different sequence            112   e.g. AlphaFold models the full precursor
#                                       while the dataset carries the mature peptide
# i.e. ~3.4% of structured proteins were misaligned or mismatched.

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def _coords_and_seq(path, chain_id="A"):
    """Backbone [L,3,3] plus the one-letter sequence the file itself reports."""
    if path.endswith(".pdb"):
        from esm.utils.structure.protein_chain import ProteinChain
        chain = ProteinChain.from_pdb(path, chain_id=chain_id)
        coords = np.asarray(chain.atom37_positions)[:, :3, :]
        return np.ascontiguousarray(coords), chain.sequence
    parser = MMCIFParser(QUIET=True)
    model = parser.get_structure("protein", path)[0]
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} not found in {path}")
    coords, seq = [], []
    for residue in model[chain_id]:
        if not is_aa(residue, standard=True):
            continue
        atoms = []
        for name in ("N", "CA", "C"):
            try:
                atoms.append(residue[name].get_coord())
            except KeyError:
                atoms.append(np.full(3, np.nan))
        coords.append(atoms)
        seq.append(_THREE_TO_ONE.get(residue.get_resname().upper(), "X"))
    return np.asarray(coords), "".join(seq)


def backbone_for_sequence(path, sequence, chain_id="A"):
    """Backbone coordinates ALIGNED to `sequence`, or None if they do not match.

    Returning None means "no usable structure", which every caller already
    handles as sequence-only -- strictly better than silently embedding the
    wrong residues.
    """
    coords, struct_seq = _coords_and_seq(path, chain_id)
    if not struct_seq or coords.shape[0] == 0:
        return None
    n = min(len(struct_seq), len(sequence))
    if n and struct_seq[:n] == sequence[:n]:
        return coords[: len(sequence)] if len(struct_seq) >= len(sequence) else None
    offset = struct_seq.find(sequence)
    if offset >= 0:
        return coords[offset : offset + len(sequence)]
    # A few residues differing (non-standard codes rendered X) is fine at offset 0.
    if n and sum(a == b for a, b in zip(struct_seq[:n], sequence[:n])) / n >= _MIN_IDENTITY \
       and len(struct_seq) >= len(sequence):
        return coords[: len(sequence)]
    # Last resort: the sequence may sit at an offset AND differ by a few residues
    # (an isoform of the same gene product), which the exact substring search
    # above misses. Slide it and keep the best offset if it clears the identity
    # bar. Anything below that is a genuinely different form and is dropped --
    # misaligned coordinates are worse than none.
    best = _best_offset(struct_seq, sequence)
    if best is not None:
        return coords[best : best + len(sequence)]
    return None


_MIN_IDENTITY = 0.95


def _best_offset(struct_seq: str, sequence: str):
    """Offset maximising identity, or None if the best is below _MIN_IDENTITY."""
    m, k = len(struct_seq), len(sequence)
    if k == 0 or m < k:
        return None
    best_off, best_id = None, 0.0
    for off in range(m - k + 1):
        window = struct_seq[off : off + k]
        ident = sum(a == b for a, b in zip(window, sequence)) / k
        if ident > best_id:
            best_id, best_off = ident, off
        if best_id == 1.0:
            break
    return best_off if best_id >= _MIN_IDENTITY else None


def backbone_resolving_alternates(path, sequence, chain_id="A"):
    """`backbone_for_sequence`, but fall back to sibling files for this protein.

    assets/structures carries several AlphaFold releases (v4 .cif, v6 .pdb) and
    the release AlphaFold modelled can differ between them: 15 proteins have a
    sibling that matches the dataset sequence while the recorded structure_path
    does not (10 test ones want the v4 .cif, 5 train ones want the v6 .pdb).
    Trying siblings recovers those without any download.

    The recorded path is always tried first, so behaviour is unchanged for the
    ~99% that already match and no extra filesystem work happens for them.
    """
    import glob
    import os as _os

    coords = backbone_for_sequence(path, sequence, chain_id)
    if coords is not None:
        return coords
    stem = _os.path.basename(path)
    # AF-<ACCESSION>-F1-model_vN.<ext> -> glob on the accession
    parts = stem.split("-")
    accession = parts[1] if len(parts) > 2 and parts[0] == "AF" else None
    if not accession:
        return None
    for cand in sorted(glob.glob(_os.path.join(_os.path.dirname(path), f"*{accession}*"))):
        if _os.path.basename(cand) == stem:
            continue
        try:
            alt = backbone_for_sequence(cand, sequence, chain_id)
        except Exception:
            continue
        if alt is not None:
            return alt
    return None
