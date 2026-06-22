#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI glue for the Score tab.

Turns any pipeline ligand (tracked, docked, refined, or live in a simulation)
into the inputs the ``scoring`` engine needs, builds a ``MapLike`` from the
current ChimeraX density map, holds the comparison table, and exports it.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np

from chimerax.ChemEM.core.tools import mol_from_sdf
from chimerax.ChemEM.core.tracked_ligands import (
    rdmol_with_live_coords,
    tracked_ligand_display_name,
)
from chimerax.ChemEM.score import scoring


# ---------------------------------------------------------------------------
# Map adapter
# ---------------------------------------------------------------------------

def maplike_from_volume(volume):
    """Build a ``scoring.MapLike`` from a ChimeraX ``Volume`` (full region).

    Uses ``data_origin_and_step()`` (origin/apix in xyz Å) and the full data
    matrix (z, y, x) - the same idiom the simulation MapBias force uses - so the
    map sits in the same coordinate frame as the atoms' scene coordinates.
    """
    if volume is None:
        return None
    origin, apix = volume.data_origin_and_step()
    try:
        density = volume.full_matrix()
    except Exception:
        density = volume.matrix()
    # resolution is passed separately to the metric drivers (from the Setup-tab
    # 'resolution' parameter); MapLike.resolution is unused by the scoring math.
    return scoring.MapLike(origin, apix, np.asarray(density, dtype=np.float64), 0.0)


# ---------------------------------------------------------------------------
# Heavy-atom extraction helpers
# ---------------------------------------------------------------------------

def _heavy_from_rdmol(mol):
    """(coords_A, masses) for heavy atoms of an RDKit mol with a conformer."""
    if mol is None or mol.GetNumConformers() == 0:
        return None, None
    conf = mol.GetConformer()
    coords, masses = [], []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([float(p.x), float(p.y), float(p.z)])
        masses.append(float(atom.GetMass()))
    if not coords:
        return None, None
    return np.asarray(coords, dtype=float), np.asarray(masses, dtype=float)


def _heavy_from_cx_atoms(atoms):
    """(coords_A, masses) for heavy ChimeraX atoms, using scene coordinates so
    GUI moves of the model are reflected."""
    coords, masses = [], []
    for a in atoms:
        elem = a.element
        if getattr(elem, "number", 0) <= 1:
            continue
        c = a.scene_coord
        coords.append([float(c[0]), float(c[1]), float(c[2])])
        masses.append(float(getattr(elem, "mass", 12.0)))
    if not coords:
        return None, None
    return np.asarray(coords, dtype=float), np.asarray(masses, dtype=float)


# ---------------------------------------------------------------------------
# Score targets
# ---------------------------------------------------------------------------

@dataclass
class ScoreTarget:
    key: str                 # stable unique id, e.g. "tracked:lig1"
    source: str              # tracked | docked | refined | simulation | selection
    display_name: str
    coords_A: np.ndarray = None
    masses: np.ndarray = None
    rdmol: object = None
    can_mmgbsa: bool = False
    note: str = ""

    @property
    def n_atoms(self):
        return 0 if self.coords_A is None else int(self.coords_A.shape[0])

    @property
    def valid(self):
        return self.coords_A is not None and self.coords_A.shape[0] > 0


def _tracked_targets(chemem):
    out = []
    tracked = getattr(chemem, "_tracked_ligands", {}) or {}
    for ligand_id, record in tracked.items():
        mol = rdmol_with_live_coords(record)
        coords, masses = _heavy_from_rdmol(mol)
        name = tracked_ligand_display_name(record, fallback_name=str(ligand_id))
        out.append(ScoreTarget(
            key=f"tracked:{ligand_id}", source="tracked", display_name=name,
            coords_A=coords, masses=masses, rdmol=mol))
    return out


def _loaded_solution_targets(chemem, results, source):
    """Targets from a dock/refine ``LoadedSolutions`` object (SDF poses)."""
    out = []
    if results is None:
        return out
    directories = getattr(results, "directories", []) or []
    all_results = getattr(results, "results", []) or []
    for directory, solutions in zip(directories, all_results):
        for sol in solutions:
            full_path = sol.full_path
            mol = None
            try:
                mol = mol_from_sdf(full_path, remove_hs=False)
            except Exception:
                mol = None
            coords, masses = _heavy_from_rdmol(mol)
            score_str = ""
            if getattr(sol, "score", None) is not None:
                score_str = f"  [{sol.score}]"
            name = f"{directory} / {sol.display_name}{score_str}"
            out.append(ScoreTarget(
                key=f"{source}:{full_path}", source=source, display_name=name,
                coords_A=coords, masses=masses, rdmol=mol))
    return out


def _simulation_targets(chemem):
    out = []
    sim = getattr(chemem, "current_simulation", None)
    if sim is None:
        return out
    model = getattr(sim, "simulation_model", None)
    if model is None:
        return out
    can = _simulation_complex(chemem) is not None
    for residue in model.residues:
        if not str(residue.name).startswith("LIG"):
            continue
        coords, masses = _heavy_from_cx_atoms(residue.atoms)
        out.append(ScoreTarget(
            key=f"simulation:{residue.name}", source="simulation",
            display_name=f"Simulation {residue.name}",
            coords_A=coords, masses=masses, can_mmgbsa=can))
    return out


def gather_score_targets(chemem):
    """Resolve every scoreable pipeline ligand with fresh (live) coordinates."""
    targets = []
    targets.extend(_tracked_targets(chemem))
    targets.extend(_loaded_solution_targets(chemem, getattr(chemem, "dock_results", None), "docked"))
    targets.extend(_loaded_solution_targets(chemem, getattr(chemem, "refine_results", None), "refined"))
    targets.extend(_simulation_targets(chemem))
    return targets


def selection_target(chemem):
    """A ScoreTarget built from the current ChimeraX atom selection."""
    try:
        from chimerax.atomic import selected_atoms
        atoms = list(selected_atoms(chemem.session))
    except Exception:
        atoms = []
        for group in chemem.session.selection.items("atoms"):
            atoms.extend(list(group))
    if not atoms:
        return None
    coords, masses = _heavy_from_cx_atoms(atoms)
    return ScoreTarget(key="selection", source="selection",
                       display_name=f"Selection ({len(atoms)} atoms)",
                       coords_A=coords, masses=masses)


def targets_payload(targets):
    """JSON-able list for the pick-list UI."""
    return [{
        "key": t.key,
        "source": t.source,
        "display_name": t.display_name,
        "n_atoms": t.n_atoms,
        "valid": t.valid,
        "can_mmgbsa": bool(t.can_mmgbsa),
    } for t in targets]


# ---------------------------------------------------------------------------
# Scoring a target
# ---------------------------------------------------------------------------

def score_target(target, emmap, resolution, which):
    """Compute the requested map scores for one target. Returns a dict of
    column->value (only the requested/available columns)."""
    out = {}
    if not target.valid:
        out["error"] = "no atoms"
        return out

    map_metrics = [w for w in which if w in scoring.DENSITY_METRICS]
    if map_metrics:
        out.update(scoring.score_map_metrics(
            target.coords_A, target.masses, emmap, resolution, which=map_metrics))

    if "qscore" in which:
        q = scoring.score_qscore(target.coords_A, emmap)
        out["qscore_mean"] = q["qscore_mean"]
        out["qscore_min"] = q["qscore_min"]
    return out


# ---------------------------------------------------------------------------
# MMGBSA glue (uses the active simulation's parametrised complex)
# ---------------------------------------------------------------------------

def _simulation_complex(chemem):
    """The active simulation's parametrised ParmEd complex_structure, or None."""
    sim = getattr(chemem, "current_simulation", None)
    if sim is None:
        return None
    inner = getattr(sim, "simulation", None)
    return getattr(inner, "complex_structure", None)


def build_mmgbsa_complex(chemem):
    """Deep-copy the active simulation complex and refresh its coordinates from
    the live simulation positions. Returns a ParmEd structure or None."""
    import copy

    sim = getattr(chemem, "current_simulation", None)
    struct = _simulation_complex(chemem)
    if sim is None or struct is None:
        return None

    struct = copy.deepcopy(struct)
    inner = getattr(sim, "simulation", None)
    pos = None
    for getter in ("get_positions_as_numpy", "get_positions"):
        fn = getattr(inner, getter, None)
        if callable(fn):
            try:
                pos = np.asarray(fn(), dtype=float)
                break
            except Exception:
                pos = None
    if pos is not None and pos.shape[0] == len(struct.atoms):
        struct.coordinates = pos
    return struct


def compute_mmgbsa_for_simulation(chemem):
    """Run MMGBSA on the active simulation complex. Returns (components, dG)."""
    struct = build_mmgbsa_complex(chemem)
    if struct is None:
        raise RuntimeError(
            "MMGBSA needs an active simulation containing the ligand. "
            "Build a simulation on the Build tab first.")
    platform = getattr(chemem, "platform", None)
    return scoring.compute_mmgbsa(struct, platform_name=platform)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

class ScoreTable:
    """One row per scored ligand; columns filled in as scores are computed."""

    MAP_COLUMNS = ["ccc", "mi", "sci", "qscore_mean", "qscore_min"]
    MMGBSA_COLUMNS = ["mmgbsa_dG", "EEL", "VDW", "EGB", "ECAV"]
    COLUMNS = MAP_COLUMNS + MMGBSA_COLUMNS

    def __init__(self):
        self.rows = {}   # key -> {"key","name","source", <columns...>}
        self._order = []

    def _row(self, key, name, source):
        if key not in self.rows:
            self.rows[key] = {"key": key, "name": name, "source": source}
            self._order.append(key)
        else:
            # keep latest display name/source
            self.rows[key]["name"] = name
            self.rows[key]["source"] = source
        return self.rows[key]

    def update_scores(self, key, name, source, scores):
        row = self._row(key, name, source)
        for col in self.COLUMNS:
            if col in scores and scores[col] is not None:
                row[col] = round(float(scores[col]), 4)
        if scores.get("error"):
            row["error"] = scores["error"]
        return row

    def update_mmgbsa(self, key, name, source, components, deltaG):
        row = self._row(key, name, source)
        row["mmgbsa_dG"] = round(float(deltaG), 4)
        for k in ("EEL", "VDW", "EGB", "ECAV"):
            if k in components:
                row[k] = round(float(components[k]), 4)
        return row

    def set_pending(self, key, name, source, column):
        row = self._row(key, name, source)
        row[column] = "..."
        return row

    def clear(self):
        self.rows = {}
        self._order = []

    def payload(self):
        return {
            "columns": self.COLUMNS,
            "rows": [self.rows[k] for k in self._order if k in self.rows],
        }

    def export(self, output_dir, basename="chemem_scores"):
        """Write CSV + JSON of the table. Returns (csv_path, json_path)."""
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{basename}.csv")
        json_path = os.path.join(output_dir, f"{basename}.json")

        header = ["name", "source"] + self.COLUMNS
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for key in self._order:
                row = self.rows.get(key, {})
                writer.writerow([row.get("name", ""), row.get("source", "")]
                                + [row.get(c, "") for c in self.COLUMNS])

        with open(json_path, "w") as f:
            json.dump(self.payload(), f, indent=2)

        return csv_path, json_path
