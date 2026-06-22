#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone (no-simulation) torsion-minima flicking for tracked ligands in the
Ligand tab.

The ChemEM backend's export step writes clean intrinsic torsion-preference
profiles for every ligand to ``<exported_dir>/torsion_profiles.json`` (see
``ChemEM/protocols/export_simulation/export_simulation.py``). Each entry carries
the four dihedral ``atom_names``, the ligand-local RDKit ``local_indices`` of
those atoms, and a normalised energy ``profile`` (angle in -180..180, energy in
0..1 relative).

This module reads those profiles, matches each one to a tracked ligand's
rotatable bonds, detects the energy minima, and rigidly rotates the ligand's
ChimeraX model *in place* so the user can step through the minima - all without
a running OpenMM simulation. It reuses the pure-geometry helpers in
``chimerax.ChemEM.simulate.torsion``.
"""

import os
import json
import numpy as np
from rdkit.Geometry import Point3D
from chimerax.ChemEM.simulate import torsion as _torsion


# Delimiter for the string command payloads (ligand_id <SEP> id). Sending dicts
# would route through Parameters.get_from_query, so everything is a flat string.
SEP = "\x1f"


# ---------------------------------------------------------------------------
# Locating + loading the exported profiles
# ---------------------------------------------------------------------------
def resolve_exported_dir(chemem, ligand_id=None):
    """Directory that holds ``torsion_profiles.json`` for ``ligand_id``, or None.

    Preference order:
      1. A per-ligand directory written by the Ligand-tab "Export torsions" action
         (``chemem._ligand_torsion_export_dirs[ligand_id]``).
      2. The most recent full simulation export
         (``chemem.current_simulation.simulation.exported_dir``)."""
    per_ligand = getattr(chemem, "_ligand_torsion_export_dirs", None) or {}
    if ligand_id is not None:
        d = per_ligand.get(ligand_id)
        if d and os.path.isdir(str(d)):
            return str(d)

    cs = getattr(chemem, "current_simulation", None)
    sim = getattr(cs, "simulation", None)
    exported_dir = getattr(sim, "exported_dir", None)
    if exported_dir and os.path.isdir(str(exported_dir)):
        return str(exported_dir)
    return None


def load_torsion_profiles(exported_dir):
    """Parse ``<exported_dir>/torsion_profiles.json`` into a doc dict, or None."""
    if not exported_dir:
        return None
    path = os.path.join(exported_dir, "torsion_profiles.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Geometry helpers (model-local coordinate space)
# ---------------------------------------------------------------------------
def local_coords(record):
    """(N, 3) array of the ligand's atoms in model-LOCAL coordinates, indexed by
    RDKit atom index. Mapped atoms are read from the live ChimeraX model
    (``atom.coord``); any unmapped atom falls back to the rdmol conformer so the
    array is dense. Working in local space keeps rotation axes self-consistent
    regardless of any whole-model move the user has made."""
    matcher = record.get("atom_matcher")
    rdmol = getattr(matcher, "rdmol", None)
    if rdmol is None:
        return None
    n = rdmol.GetNumAtoms()
    conf = rdmol.GetConformer()
    rd_to_cx = getattr(matcher, "rd_to_cx", {}) or {}
    xyz = np.zeros((n, 3), dtype=float)
    for rd_idx in range(n):
        cx = rd_to_cx.get(rd_idx)
        if cx is not None:
            c = cx.coord
            xyz[rd_idx] = (float(c[0]), float(c[1]), float(c[2]))
        else:
            p = conf.GetAtomPosition(rd_idx)
            xyz[rd_idx] = (p.x, p.y, p.z)
    return xyz


def write_moved(record, new_xyz, moved_idxs):
    """Write ``new_xyz`` for ``moved_idxs`` back to BOTH the ChimeraX model
    (mapped atoms) and the matcher's rdmol conformer (so stereo perception and
    subsequent edits read consistent geometry)."""
    matcher = record.get("atom_matcher")
    rdmol = getattr(matcher, "rdmol", None)
    conf = rdmol.GetConformer()
    rd_to_cx = getattr(matcher, "rd_to_cx", {}) or {}
    for rd_idx in moved_idxs:
        rd_idx = int(rd_idx)
        x, y, z = (float(new_xyz[rd_idx][0]), float(new_xyz[rd_idx][1]), float(new_xyz[rd_idx][2]))
        cx = rd_to_cx.get(rd_idx)
        if cx is not None:
            cx.coord = np.array([x, y, z], dtype=float)
        conf.SetAtomPosition(rd_idx, Point3D(x, y, z))


def dihedral(xyz, i, a, b, l):
    return _torsion.dihedral_deg(xyz[i], xyz[a], xyz[b], xyz[l])


def _ang_diff(x, y):
    return (x - y + 180.0) % 360.0 - 180.0


def next_minimum_angle(minima_angles, cur, direction):
    """Circular nearest minimum in the step direction (port of
    ``SimulationJob._next_minimum``)."""
    if not minima_angles:
        return cur
    if direction > 0:
        cand = sorted(((ma - cur) % 360.0, ma) for ma in minima_angles)
    else:
        cand = sorted(((cur - ma) % 360.0, ma) for ma in minima_angles)
    for d, ma in cand:
        if d > 1e-6:
            return ma
    return cand[0][1]


def nearest_minimum_index(minima, angle):
    if not minima:
        return None
    diffs = [abs(_ang_diff(angle, m["angle"])) for m in minima]
    return int(np.argmin(diffs))


# ---------------------------------------------------------------------------
# Matching exported profiles to a tracked ligand's rotatable bonds
# ---------------------------------------------------------------------------
def _atom_label(matcher, rd_idx):
    cx = (getattr(matcher, "rd_to_cx", {}) or {}).get(int(rd_idx))
    nm = getattr(cx, "name", None) if cx is not None else None
    if nm:
        return str(nm)
    rdmol = getattr(matcher, "rdmol", None)
    if rdmol is not None:
        try:
            return "{0}{1}".format(rdmol.GetAtomWithIdx(int(rd_idx)).GetSymbol(), rd_idx)
        except Exception:
            pass
    return str(rd_idx)


def _name_to_rd(matcher):
    """Map upper-cased ChimeraX atom name -> rd index (for the fallback join)."""
    out = {}
    for rd_idx, cx in (getattr(matcher, "rd_to_cx", {}) or {}).items():
        nm = getattr(cx, "name", None)
        if nm:
            out.setdefault(str(nm).upper(), int(rd_idx))
    return out


def _valid_chain(rdmol, quad):
    """True if ``quad`` is four distinct, in-range atoms forming a bonded chain
    a-b-c-d in ``rdmol``. This validates that an entry's indices/names address
    THIS ligand in THIS index space before we trust them."""
    n = rdmol.GetNumAtoms()
    if len(quad) != 4 or len(set(quad)) != 4:
        return False
    if any(x < 0 or x >= n for x in quad):
        return False
    for x, y in ((quad[0], quad[1]), (quad[1], quad[2]), (quad[2], quad[3])):
        if rdmol.GetBondBetweenAtoms(int(x), int(y)) is None:
            return False
    return True


def _entry_rd_quad(entry, rdmol, name_to_rd):
    """Resolve a profile entry's dihedral to four rd indices in ``rdmol``, or None.

    Primary: ``local_indices`` interpreted directly as rd indices (the tracked
    rdmol is read from the same SDF the backend parametrised, so indices align).
    Fallback: ``atom_names`` mapped through the ChimeraX atom-name map. Both are
    accepted only if they form a bonded 4-chain in ``rdmol``."""
    li = entry.get("local_indices")
    if li and len(li) == 4:
        try:
            cand = [int(x) for x in li]
            if _valid_chain(rdmol, cand):
                return cand
        except (TypeError, ValueError):
            pass

    names = entry.get("atom_names")
    if names and len(names) == 4:
        cand = []
        for nm in names:
            rid = name_to_rd.get(str(nm).upper())
            if rid is None:
                cand = None
                break
            cand.append(rid)
        if cand and _valid_chain(rdmol, cand):
            return cand
    return None


def _angle_offset(xyz, quad, our):
    """Constant degrees offset between the backend dihedral (``quad``) and our
    dihedral (``our``) at the current geometry, such that
    ``our_angle = backend_angle - offset``. Mirrors
    ``SimulationJob._imported_angle_offset``."""
    try:
        bk = dihedral(xyz, quad[0], quad[1], quad[2], quad[3])
        ours = dihedral(xyz, our[0], our[1], our[2], our[3])
        if not (np.isfinite(bk) and np.isfinite(ours)):
            return 0.0
        return float(bk - ours)
    except Exception:
        return 0.0


def build_ligand_torsion_rows(record, doc, ligand_id):
    """One row per rotatable bond of the tracked ligand that has a matching
    exported profile. Returns ``[]`` when nothing matches.

    Each row::

        {key, ligand_id, label, rd_atoms=[i,a,b,l], moved_atoms,
         angles, energies, minima, current_angle, current_idx}
    """
    matcher = record.get("atom_matcher")
    rdmol = getattr(matcher, "rdmol", None)
    if rdmol is None:
        return []
    entries = (doc or {}).get("torsions", []) or []
    if not entries:
        return []

    name_to_rd = _name_to_rd(matcher)
    by_central = {}  # frozenset({rd_a, rd_b}) -> (entry, rd quad)
    for entry in entries:
        quad = _entry_rd_quad(entry, rdmol, name_to_rd)
        if quad is None:
            continue
        by_central.setdefault(frozenset((quad[1], quad[2])), (entry, quad))
    if not by_central:
        return []

    xyz = local_coords(record)
    rows = []
    for t in _torsion.find_rotatable_torsions(rdmol):
        i, a, b, l = t["atoms"]
        hit = by_central.get(frozenset((a, b)))
        if hit is None:
            continue
        entry, quad = hit
        offset = _angle_offset(xyz, quad, (i, a, b, l))

        pts = []
        ok = True
        for ang, e in entry.get("profile", []):
            ev = float(e)
            if not np.isfinite(ev):
                ok = False
                break
            pts.append(((float(ang) - offset) % 360.0, ev))
        if not ok or not pts:
            continue
        pts.sort(key=lambda p: p[0])
        angles = [p[0] for p in pts]
        energies = [p[1] for p in pts]
        minima = _torsion.detect_minima(angles, energies)
        cur = dihedral(xyz, i, a, b, l) % 360.0

        rows.append({
            "key": "{0}|{1}|{2}".format(ligand_id, a, b),
            "ligand_id": ligand_id,
            "label": "{0}–{1}".format(_atom_label(matcher, a), _atom_label(matcher, b)),
            "rd_atoms": [int(i), int(a), int(b), int(l)],
            "moved_atoms": [int(x) for x in t["moved_atoms"]],
            "angles": angles,
            "energies": energies,
            "minima": minima,
            "current_angle": cur,
            "current_idx": nearest_minimum_index(minima, cur),
        })
    return rows


def apply_torsion_target(record, row, target_angle_deg):
    """Rigidly rotate the row's moved side so the i-a-b-l dihedral reaches
    ``target_angle_deg``, then write the result to the model + conformer."""
    xyz = local_coords(record)
    i, a, b, l = row["rd_atoms"]
    moved = row["moved_atoms"]
    cur = dihedral(xyz, i, a, b, l)
    delta = float(target_angle_deg) - float(cur)
    rot = _torsion.rotate_points(xyz, a, b, moved, delta)
    write_moved(record, rot, moved)
    return True


# ---------------------------------------------------------------------------
# UI payloads
# ---------------------------------------------------------------------------
def slim_row(row):
    """Compact row for the list view (drops the bulky angle/energy arrays)."""
    minima = row.get("minima") or []
    return {
        "key": row["key"],
        "label": row["label"],
        "minima": minima,
        "n_minima": len(minima),
        "current_angle": row.get("current_angle"),
        "current_idx": row.get("current_idx"),
    }


def profile_payload(row):
    """Full SVG-profile payload (matches the Simulate-tab renderTorsionProfile shape)."""
    return {
        "key": row["key"],
        "label": row["label"],
        "points": [[float(a), float(e)] for a, e in zip(row["angles"], row["energies"])],
        "minima": row.get("minima") or [],
        "current_idx": row.get("current_idx"),
        "current_angle": row.get("current_angle"),
    }
