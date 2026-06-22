#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torsion-preference helpers for the ChemEM simulation tab.

This module holds pure (ChimeraX/OpenMM-free) geometry + RDKit helpers used by
``SimulationJob`` to:

  * enumerate the rotatable torsions of a ligand (RDKit bond perception),
  * rigidly rotate a torsion's "moved" atoms about its bond axis,
  * measure a dihedral angle, and
  * detect the minima of a scanned energy-vs-angle profile.

The rotatable-bond enumeration is lifted from ``ion_fixer.py``
(``_find_allowed_ligand_torsions`` / ``_bfs_side_atoms`` /
``_choose_torsion_terminal_neighbor``) so it can be reused without an
``IonFixer`` instance.
"""

from collections import deque
import numpy as np
from rdkit import Chem


# ---------------------------------------------------------------------------
# Rotatable-torsion enumeration (lifted from ion_fixer.py)
# ---------------------------------------------------------------------------
def bfs_side_atoms(mol, start_idx, blocked_idx):
    """Connected component reached from ``start_idx`` with the edge to
    ``blocked_idx`` removed (i.e. the atoms that move when the start--blocked
    bond is rotated)."""
    out = set()
    q = deque([start_idx])
    seen = {blocked_idx}
    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        out.add(u)
        atom = mol.GetAtomWithIdx(int(u))
        for nbr in atom.GetNeighbors():
            v = nbr.GetIdx()
            if v not in seen:
                q.append(v)
    return out


def choose_terminal_neighbor(mol, center_idx, exclude_idx, prefer_atoms=None):
    """Pick a neighbour of ``center_idx`` (other than ``exclude_idx``) to act as
    a terminal atom of the dihedral, preferring heavy atoms."""
    nbrs = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(int(center_idx)).GetNeighbors()
        if n.GetIdx() != int(exclude_idx)
    ]
    if not nbrs:
        return None

    if prefer_atoms is not None:
        heavy_pref = [i for i in nbrs if i in prefer_atoms and mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
        if heavy_pref:
            return heavy_pref[0]
        any_pref = [i for i in nbrs if i in prefer_atoms]
        if any_pref:
            return any_pref[0]

    heavy = [i for i in nbrs if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    if heavy:
        return heavy[0]
    return nbrs[0]


def find_rotatable_torsions(mol):
    """Identify rotatable single, acyclic torsions of an RDKit mol.

    Returns a list of dicts: ``{"atoms": (i, a, b, l), "moved_atoms": [...],
    "fixed_atoms": [...], "bond": (a, b)}`` with all indices in RDKit atom-index
    space. ``moved_atoms`` is the side that rotates about the a--b bond (chosen as
    the smaller side); ``fixed_atoms`` is the complementary side. The caller can let
    the user swap which side is treated as moving.
    """
    torsions = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue
        if bond.IsInRing():
            continue

        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())

        if mol.GetAtomWithIdx(a).GetDegree() < 2 or mol.GetAtomWithIdx(b).GetDegree() < 2:
            continue

        # Each end must have a heavy neighbour besides the other end, otherwise
        # this is a trivial terminal rotor (e.g. a methyl or -OH) with no
        # meaningful torsion profile. This matches the standard rotatable-bond
        # definition (so butane reports its single central bond, not 3).
        a_heavy = [n for n in mol.GetAtomWithIdx(a).GetNeighbors()
                   if n.GetIdx() != b and n.GetAtomicNum() > 1]
        b_heavy = [n for n in mol.GetAtomWithIdx(b).GetNeighbors()
                   if n.GetIdx() != a and n.GetAtomicNum() > 1]
        if not a_heavy or not b_heavy:
            continue

        side_b = bfs_side_atoms(mol, b, a)
        side_a = bfs_side_atoms(mol, a, b)

        # rotate the smaller side; orient = (axis_a, axis_b, moved_side)
        if len(side_a) <= len(side_b):
            orient = (b, a, side_a)
        else:
            orient = (a, b, side_b)

        atom_a, atom_b, moved_side = orient
        # The complementary (larger) side; the user may choose to rotate this one
        # instead, in which case the moved_side is held fixed.
        fixed_side = side_b if moved_side is side_a else side_a

        i = choose_terminal_neighbor(mol, atom_a, atom_b, prefer_atoms=None)
        l = choose_terminal_neighbor(mol, atom_b, atom_a, prefer_atoms=moved_side)
        if i is None or l is None:
            continue

        torsions.append(
            {
                "atoms": (int(i), int(atom_a), int(atom_b), int(l)),
                "moved_atoms": sorted(int(x) for x in moved_side),
                "fixed_atoms": sorted(int(x) for x in fixed_side),
                "bond": (int(atom_a), int(atom_b)),
            }
        )
    return torsions


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def rotate_points(xyz, axis_a_idx, axis_b_idx, moved_idxs, delta_deg):
    """Rigidly rotate ``moved_idxs`` rows of ``xyz`` (N, 3) about the directed
    axis ``axis_a_idx`` -> ``axis_b_idx`` by ``delta_deg`` degrees.

    Returns a NEW array; ``xyz`` is not modified. Units of ``xyz`` are
    arbitrary but must be consistent (we pass angstrom). Rotating the moved
    side by Δ about the bond axis changes the i-a-b-l dihedral by Δ.
    """
    out = np.array(xyz, dtype=float, copy=True)
    if not len(moved_idxs):
        return out

    pa = out[axis_a_idx]
    pb = out[axis_b_idx]
    axis = pb - pa
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return out
    axis = axis / norm

    # Negate so a +delta rotation of the moved atoms INCREASES the i-a-b-l
    # dihedral by delta (the IUPAC dihedral sign convention is opposite to a
    # right-handed Rodrigues rotation about the a->b axis).
    theta = np.radians(-float(delta_deg))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    moved = np.asarray(list(moved_idxs), dtype=int)
    v = out[moved] - pa                      # (M, 3) relative to axis origin
    # Rodrigues' rotation formula
    dot = v @ axis                           # (M,)
    cross = np.cross(np.broadcast_to(axis, v.shape), v)
    rotated = (v * cos_t
               + cross * sin_t
               + np.outer(dot * (1.0 - cos_t), axis))
    out[moved] = rotated + pa
    return out


def dihedral_deg(p_i, p_a, p_b, p_l):
    """Signed dihedral angle (degrees, -180..180) of points i-a-b-l."""
    p_i = np.asarray(p_i, dtype=float)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    p_l = np.asarray(p_l, dtype=float)

    b1 = p_a - p_i
    b2 = p_b - p_a
    b3 = p_l - p_b

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))

    x = n1 @ n2
    y = m1 @ n2
    return float(np.degrees(np.arctan2(y, x)))


# ---------------------------------------------------------------------------
# Minima detection on a wrapped (periodic) energy profile
# ---------------------------------------------------------------------------
def detect_minima(angles, energies):
    """Return the local minima of a periodic energy-vs-angle profile.

    ``angles`` are evenly spaced over [0, 360). A sample is a local minimum if
    it is <= both circular neighbours (strictly < at least one side to avoid
    flat plateaus emitting many minima). The lowest is flagged ``is_global``.

    Returns a list of ``{"angle": float, "energy": float, "is_global": bool}``
    sorted by angle.
    """
    n = len(energies)
    if n == 0:
        return []
    e = list(energies)
    minima = []
    for k in range(n):
        prev_e = e[(k - 1) % n]
        next_e = e[(k + 1) % n]
        if e[k] <= prev_e and e[k] <= next_e and (e[k] < prev_e or e[k] < next_e):
            minima.append({"angle": float(angles[k]), "energy": float(e[k]), "is_global": False})

    if not minima:
        # degenerate (perfectly flat / monotone-circular) -> single global min
        gk = int(np.argmin(e))
        minima = [{"angle": float(angles[gk]), "energy": float(e[gk]), "is_global": True}]
        return minima

    gmin = min(minima, key=lambda m: m["energy"])
    gmin["is_global"] = True
    minima.sort(key=lambda m: m["angle"])
    return minima
