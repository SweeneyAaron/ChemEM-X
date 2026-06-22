#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In-place, pose-preserving isomer editing for tracked ligands in the Ligand tab.

Two stereochemical edits, both applied directly to the ChimeraX model
coordinates (no model swap, no running simulation):

  * cis/trans (E/Z) double bonds - flipped by rigidly rotating the smaller
    substituent side 180 deg about the C=C axis (reuses ``rotate_points``).
  * R/S chiral centres - inverted by swapping the 3D positions of the two
    smallest substituent branches (a transposition flips parity). Each branch
    is rigidly rotated about the centre onto the other's bond direction so
    internal geometry and bond lengths are preserved.

After either edit the stereochemistry is re-perceived from the new 3D geometry
(``AssignStereochemistryFrom3D``) and the matcher's rdmol conformer is kept in
sync with the model via ``torsion_minima.write_moved``.
"""

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from chimerax.ChemEM.simulate import torsion as _torsion
from chimerax.ChemEM.ligand import torsion_minima as _tmin


_STEREO_LABEL = {
    "STEREOE": "E",
    "STEREOZ": "Z",
    "STEREOTRANS": "trans",
    "STEREOCIS": "cis",
}


def _live_mol(record):
    """Copy of the matcher rdmol whose conformer holds the ligand's current
    model-local coordinates, with stereochemistry perceived from that geometry."""
    matcher = record.get("atom_matcher")
    rdmol = getattr(matcher, "rdmol", None)
    if rdmol is None or rdmol.GetNumConformers() == 0:
        return None
    mol = Chem.Mol(rdmol)
    conf = mol.GetConformer()
    xyz = _tmin.local_coords(record)
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(float(xyz[i][0]), float(xyz[i][1]), float(xyz[i][2])))
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


# ---------------------------------------------------------------------------
# Listing the stereo features
# ---------------------------------------------------------------------------
def list_isomers(record):
    """Return ``{"double_bonds": [...], "chiral_centers": [...]}`` for the UI."""
    mol = _live_mol(record)
    if mol is None:
        return {"double_bonds": [], "chiral_centers": []}
    matcher = record.get("atom_matcher")

    double_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.rdchem.BondType.DOUBLE:
            continue
        if bond.IsInRing():
            continue
        stereo = bond.GetStereo()
        name = str(stereo).rsplit(".", 1)[-1]  # e.g. 'STEREOE'
        if name not in _STEREO_LABEL:
            continue
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        double_bonds.append({
            "id": "db|{0}|{1}".format(a, b),
            "bond": [a, b],
            "stereo": _STEREO_LABEL[name],
            "label": "{0}={1}".format(_tmin._atom_label(matcher, a), _tmin._atom_label(matcher, b)),
        })

    chiral_centers = []
    try:
        centers = Chem.FindMolChiralCenters(mol, useLegacyImplementation=False, includeUnassigned=False)
    except TypeError:
        # Older RDKit lacks the keyword; fall back to the positional form.
        centers = Chem.FindMolChiralCenters(mol, False)
    for idx, rs in centers:
        chiral_centers.append({
            "id": "rs|{0}".format(int(idx)),
            "center": int(idx),
            "rs": str(rs),
            "label": "{0} ({1})".format(_tmin._atom_label(matcher, int(idx)), rs),
        })

    return {"double_bonds": double_bonds, "chiral_centers": chiral_centers}


# ---------------------------------------------------------------------------
# cis/trans (E/Z) flip
# ---------------------------------------------------------------------------
def toggle_double_bond(record, bond):
    """Flip a non-ring stereo double bond E<->Z by rotating the smaller
    substituent side 180 deg about the C=C axis. Returns ``(ok, message)``."""
    matcher = record.get("atom_matcher")
    rdmol = getattr(matcher, "rdmol", None)
    if rdmol is None:
        return False, "Ligand has no structure."
    a, b = int(bond[0]), int(bond[1])
    if rdmol.GetBondBetweenAtoms(a, b) is None:
        return False, "Double bond not found."

    xyz = _tmin.local_coords(record)
    side_a = _torsion.bfs_side_atoms(rdmol, a, b)
    side_b = _torsion.bfs_side_atoms(rdmol, b, a)
    # Rotating either side 180 deg about the bond axis flips E<->Z; move the
    # smaller side for minimal disturbance. The bond atom on the rotated side
    # lies on the axis and is unaffected.
    if len(side_a) <= len(side_b):
        moved, axis = side_a, (b, a)
    else:
        moved, axis = side_b, (a, b)

    rot = _torsion.rotate_points(xyz, axis[0], axis[1], moved, 180.0)
    _tmin.write_moved(record, rot, moved)
    Chem.AssignStereochemistryFrom3D(rdmol)
    return True, None


# ---------------------------------------------------------------------------
# R/S inversion
# ---------------------------------------------------------------------------
def rotation_between_vectors(u, v):
    """Rotation matrix sending unit vector ``u`` onto unit vector ``v``."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return np.eye(3)
    u = u / nu
    v = v / nv
    w = np.cross(u, v)
    c = float(np.dot(u, v))
    s = float(np.linalg.norm(w))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # antiparallel: 180 deg about any axis perpendicular to u
        perp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, perp)
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0.0, -w[2], w[1]],
                  [w[2], 0.0, -w[0]],
                  [-w[1], w[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def invert_chiral_center(record, center_idx):
    """Invert a stereocentre R<->S by swapping the spatial positions of its two
    smallest, disjoint substituent branches. Returns ``(ok, message)``.

    Ring stereocentres (whose branches loop back through the ring and overlap)
    cannot be inverted by an in-place transposition and are reported as a skip."""
    matcher = record.get("atom_matcher")
    rdmol = getattr(matcher, "rdmol", None)
    if rdmol is None:
        return False, "Ligand has no structure."
    c = int(center_idx)
    atom = rdmol.GetAtomWithIdx(c)
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
    if len(neighbors) < 3:
        return False, "Not a tetrahedral stereocentre."

    branches = {n: _torsion.bfs_side_atoms(rdmol, n, c) for n in neighbors}
    # Order neighbours by branch size; choose the two smallest that are disjoint
    # and do not loop back to the centre (i.e. not a ring stereocentre).
    order = sorted(neighbors, key=lambda n: (len(branches[n]), n))
    pick = None
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            n1, n2 = order[i], order[j]
            b1, b2 = branches[n1], branches[n2]
            if c in b1 or c in b2:
                continue
            if b1 & b2:
                continue
            pick = (n1, n2, b1, b2)
            break
        if pick is not None:
            break
    if pick is None:
        return False, "Ring stereocentre - in-place inversion is not supported."

    n1, n2, b1, b2 = pick
    xyz = _tmin.local_coords(record)
    c_pos = xyz[c]
    u1 = xyz[n1] - c_pos
    u2 = xyz[n2] - c_pos
    R1 = rotation_between_vectors(u1, u2)  # branch 1 -> branch 2's direction
    R2 = rotation_between_vectors(u2, u1)  # branch 2 -> branch 1's direction

    new_xyz = xyz.copy()
    for idx in b1:
        new_xyz[idx] = c_pos + R1 @ (xyz[idx] - c_pos)
    for idx in b2:
        new_xyz[idx] = c_pos + R2 @ (xyz[idx] - c_pos)

    _tmin.write_moved(record, new_xyz, list(b1) + list(b2))
    Chem.AssignStereochemistryFrom3D(rdmol)
    return True, None
