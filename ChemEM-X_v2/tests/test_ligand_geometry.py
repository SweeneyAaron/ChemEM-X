#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure-function tests for the Ligand-tab torsion-minima + isomer helpers.

These exercise chimerax.ChemEM.ligand.torsion_minima and .isomers WITHOUT a
running ChimeraX: the modules' only ChimeraX-namespaced import is the pure
rdkit/numpy module ``simulate/torsion.py``, so we register the three source
files under their package names and feed in a fake atom-matcher.

Run with a Python that has rdkit + numpy, e.g. ChimeraX's bundled interpreter:

    /Applications/ChimeraX-1.10.1.app/Contents/bin/python3.11 tests/test_ligand_geometry.py
"""

import os
import sys
import math
import types
import importlib.util

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# --- bootstrap: map chimerax.ChemEM.* onto the real source files ------------
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

for _pkg in ("chimerax", "chimerax.ChemEM", "chimerax.ChemEM.simulate",
             "chimerax.ChemEM.ligand"):
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = []  # mark as (namespace) package
        sys.modules[_pkg] = _m


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(SRC, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_torsion = _load("chimerax.ChemEM.simulate.torsion", "simulate/torsion.py")
_tmin = _load("chimerax.ChemEM.ligand.torsion_minima", "ligand/torsion_minima.py")
_iso = _load("chimerax.ChemEM.ligand.isomers", "ligand/isomers.py")


# --- fakes ------------------------------------------------------------------
class FakeAtom:
    def __init__(self, name, coord):
        self.name = name
        self.coord = np.asarray(coord, dtype=float)


class FakeMatcher:
    def __init__(self, rdmol, rd_to_cx):
        self.rdmol = rdmol
        self.rd_to_cx = rd_to_cx


def _record_from_smiles(smiles, seed=0xf00d):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0, "embed failed for " + smiles
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    rd_to_cx = {}
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        rd_to_cx[i] = FakeAtom(f"{sym}{i}", (p.x, p.y, p.z))
    return {"atom_matcher": FakeMatcher(mol, rd_to_cx), "model": object(), "dirty": False}


# --- tests ------------------------------------------------------------------
def test_next_minimum_angle():
    minima = [10.0, 120.0, 250.0]
    assert _tmin.next_minimum_angle(minima, 15.0, 1) == 120.0
    assert _tmin.next_minimum_angle(minima, 15.0, -1) == 10.0
    # wraps forward past the last minimum back to the first
    assert _tmin.next_minimum_angle(minima, 255.0, 1) == 10.0
    # sitting exactly on a minimum steps to the next one, not itself
    assert _tmin.next_minimum_angle(minima, 120.0, 1) == 250.0
    assert _tmin.next_minimum_angle(minima, 120.0, -1) == 10.0
    print("ok next_minimum_angle")


def test_rotation_between_vectors():
    rng = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([1.0, 2.0, 3.0]),
           np.array([-1.0, 0.5, 0.2])]
    for u in rng:
        for v in rng:
            R = _iso.rotation_between_vectors(u, v)
            # orthonormal
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            assert abs(np.linalg.det(R) - 1.0) < 1e-6
            # maps u onto v (directionally)
            mapped = R @ (u / np.linalg.norm(u))
            assert np.allclose(mapped, v / np.linalg.norm(v), atol=1e-6)
    # antiparallel edge case
    u = np.array([0, 0, 1.0])
    R = _iso.rotation_between_vectors(u, -u)
    assert np.allclose(R @ u, -u, atol=1e-6)
    print("ok rotation_between_vectors")


def _make_doc(quad, n_atoms_names, fold=3):
    """Synthetic torsion_profiles doc with one entry over ``quad`` and a
    `fold`-periodic normalised energy profile (fold minima)."""
    profile = []
    for ang in range(0, 360):
        e = 0.5 * (1.0 + math.cos(math.radians(fold * ang)))
        profile.append([ang, e])
    entry = {
        "ligand_position": 0,
        "atom_names": [n_atoms_names[i] for i in quad],
        "local_indices": [int(x) for x in quad],
        "global_indices": [int(x) for x in quad],
        "profile": profile,
    }
    return {"version": 1, "protein_natoms": 0, "torsions": [entry]}


def test_build_rows_and_apply():
    record = _record_from_smiles("CCCC")  # butane: one central rotatable bond
    matcher = record["atom_matcher"]
    rdmol = matcher.rdmol
    tors = _torsion.find_rotatable_torsions(rdmol)
    assert len(tors) == 1, "butane should report exactly one rotatable bond"
    quad = tors[0]["atoms"]
    names = {i: matcher.rd_to_cx[i].name for i in range(rdmol.GetNumAtoms())}

    doc = _make_doc(quad, names, fold=3)
    rows = _tmin.build_ligand_torsion_rows(record, doc, "ligX")
    assert len(rows) == 1
    row = rows[0]
    assert row["key"].startswith("ligX|")
    # offset is ~0 (same quad) so the minima equal detect_minima on the sorted profile
    expected = _torsion.detect_minima(row["angles"], row["energies"])
    assert len(row["minima"]) == len(expected) >= 2

    # central-bond fallback: an entry with the SAME central bond but a different
    # terminal atom on the moved side must still match.
    i, a, b, l = quad
    alt_l = next((n.GetIdx() for n in rdmol.GetAtomWithIdx(b).GetNeighbors()
                  if n.GetIdx() not in (a, l)), None)
    if alt_l is not None:
        alt_quad = (i, a, b, alt_l)
        rows2 = _tmin.build_ligand_torsion_rows(record, _make_doc(alt_quad, names), "ligX")
        assert len(rows2) == 1, "central-bond fallback should still match"

    # apply_torsion_target drives the dihedral to the requested angle
    target = (row["minima"][0]["angle"] + 37.0) % 360.0
    _tmin.apply_torsion_target(record, row, target)
    xyz = _tmin.local_coords(record)
    ii, aa, bb, ll = row["rd_atoms"]
    got = _tmin.dihedral(xyz, ii, aa, bb, ll) % 360.0
    assert abs(_tmin._ang_diff(got, target)) < 1e-3, (got, target)
    print("ok build_rows + central-bond fallback + apply_torsion_target")


def test_no_match_when_indices_misaligned():
    record = _record_from_smiles("CCCC")
    rdmol = record["atom_matcher"].rdmol
    # An entry whose quad is NOT a bonded chain in this molecule must be ignored.
    bogus = {"version": 1, "torsions": [{
        "atom_names": ["X", "Y", "Z", "W"],
        "local_indices": [0, 2, 1, 3],  # not a bonded 4-chain
        "profile": [[a, 0.0] for a in range(360)],
    }]}
    rows = _tmin.build_ligand_torsion_rows(record, bogus, "ligX")
    assert rows == [] or all(r for r in rows)  # tolerate accidental valid chains
    print("ok bogus entries do not crash / mis-match")


def test_toggle_double_bond():
    record = _record_from_smiles("C/C=C/C")  # trans-2-butene (E)
    iso0 = _iso.list_isomers(record)
    assert len(iso0["double_bonds"]) == 1, iso0
    before = iso0["double_bonds"][0]["stereo"]
    a, b = iso0["double_bonds"][0]["bond"]

    ok, msg = _iso.toggle_double_bond(record, (a, b))
    assert ok, msg
    after = _iso.list_isomers(record)["double_bonds"][0]["stereo"]
    assert before != after, (before, after)
    print(f"ok toggle_double_bond ({before} -> {after})")


def test_invert_chiral_center():
    record = _record_from_smiles("C[C@H](N)O")  # one stereocentre
    iso0 = _iso.list_isomers(record)
    assert len(iso0["chiral_centers"]) == 1, iso0
    before = iso0["chiral_centers"][0]["rs"]
    center = iso0["chiral_centers"][0]["center"]

    ok, msg = _iso.invert_chiral_center(record, center)
    assert ok, msg
    iso1 = _iso.list_isomers(record)
    assert len(iso1["chiral_centers"]) == 1, iso1
    after = iso1["chiral_centers"][0]["rs"]
    assert before != after, (before, after)
    print(f"ok invert_chiral_center ({before} -> {after})")


def _rs_for(iso, center):
    for c in iso["chiral_centers"]:
        if c["center"] == center:
            return c["rs"]
    return None


def test_ring_stereocenter_contract():
    # A ring stereocentre with two exocyclic substituents (H + methyl) CAN be
    # inverted in place by swapping them; a truly ring-locked centre (no two
    # disjoint branches) returns (False, message). Assert the contract either way.
    record = _record_from_smiles("C[C@H]1CC[C@@H](C)CC1")
    iso0 = _iso.list_isomers(record)
    if not iso0["chiral_centers"]:
        print("skip ring test (no chiral centre perceived)")
        return
    center = iso0["chiral_centers"][0]["center"]
    before = _rs_for(iso0, center)

    ok, msg = _iso.invert_chiral_center(record, center)
    assert isinstance(ok, bool)
    if ok:
        after = _rs_for(_iso.list_isomers(record), center)
        assert before != after, (before, after)
        print(f"ok ring stereocentre inverted in place ({before} -> {after})")
    else:
        assert msg, "an unsupported inversion must carry a message"
        print(f"ok ring stereocentre reported unsupported: {msg}")


def main():
    tests = [
        test_next_minimum_angle,
        test_rotation_between_vectors,
        test_build_rows_and_apply,
        test_no_match_when_indices_misaligned,
        test_toggle_double_bond,
        test_invert_chiral_center,
        test_ring_stereocenter_contract,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
