#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for the Refine tab.

The Refine tab drives two backend protocols:
  * smart_ligand_refine2  (CLI --smart-ligand-refine2 / -slr2) -> <output>/smart_refine/
  * refine                (CLI --refine)                        -> <output>/refine/

Both refine an *existing* near-fit ligand pose, so the refine tab feeds the
backend explicit ligand SDFs chosen by the user from two sources:
  * docked solutions   (chemem.dock_results) -> SDF already on disk
  * tracked ligands    (chemem._tracked_ligands) -> live coords written to inputs/

@author: aaron.sweeney
"""

import os
import shutil
import uuid
from datetime import datetime
from rdkit import Chem
from chimerax.ChemEM.dock.tools import DockSolution, LoadedSolutions, ChemEMSetUp, mkdir
from chimerax.ChemEM.core.tracked_ligands import safe_file_stem

# smart_ligand_refine2 embeds per-pose scores as SDF properties; the MD `refine`
# protocol writes plain coordinates with no score, so this prop is simply absent.
SR2_SCORE_PROP = "sr2_best_raw_score"


def make_refine_run_dir(base_output, label):
    """Create and return a fresh per-run output directory.

    Each refine run is isolated under ``<base_output>/refine_runs/<label>_<ts>_<uid>``
    so successive runs (and Dock) never overwrite each other's conf / results.
    """
    if not base_output:
        return None
    run_id = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(base_output, "refine_runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def resolve_refine_ligand_paths(chemem, selected, run_dir):
    """Resolve the user's selected poses to fresh input-copy SDF paths.

    ``selected`` is a list of dicts ``{"source": "docked"|"tracked", "key": <str>}``
    where ``key`` is the docked-solution full_path or the tracked-ligand id.

    A *new copy* of every input is saved into ``<run_dir>/inputs/`` before
    refinement so originals are never modified:
      * tracked -> live ChimeraX coordinates written via the dock/ion-fixer writer
        (captures any GUI edits);
      * docked  -> the on-disk SDF copied in verbatim.
    """
    if not run_dir:
        return []

    inputs_dir = os.path.join(run_dir, "inputs")
    mkdir(inputs_dir)

    paths = []
    for index, item in enumerate(selected or []):
        source = item.get("source")
        key = item.get("key")
        if not key:
            continue

        if source == "docked":
            if not os.path.exists(key):
                continue
            stem = safe_file_stem(f"{index}_{os.path.splitext(os.path.basename(key))[0]}")
            dest = os.path.join(inputs_dir, f"{stem}.sdf")
            try:
                shutil.copy(key, dest)
            except OSError:
                continue
            paths.append(dest)
            continue

        # tracked ligand: look up the record (normalised id lookup) and write live coords
        record = chemem.get_tracked_ligand_record(key)
        if record is None:
            continue
        ligand_path = ChemEMSetUp._write_tracked_ligand_to_inputs(
            record, inputs_dir, f"{index}_{key}"
        )
        if ligand_path is not None:
            paths.append(ligand_path)

    return paths


def get_refine_results(results_dir, score_prop=SR2_SCORE_PROP):
    """Parse ``Ligand_*.sdf`` files in ``results_dir`` into a LoadedSolutions.

    Mirrors the dock results payload (a single pseudo-directory whose name is the
    results-dir basename) so the existing render/View/Hide/Add machinery is reused.
    Reads ``score_prop`` from each SDF when present (smart_refine output); leaves
    the score ``None`` otherwise (MD refine output has no embedded scores).
    """
    all_solutions = LoadedSolutions()
    if not results_dir or not os.path.exists(results_dir):
        return all_solutions

    sdf_files = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith("Ligand_") and f.lower().endswith(".sdf")
    )

    solutions = []
    for fname in sdf_files:
        full_path = os.path.join(results_dir, fname)
        score = None
        try:
            suppl = Chem.SDMolSupplier(full_path, removeHs=False, sanitize=False)
            mol = suppl[0] if len(suppl) else None
            if mol is not None and mol.HasProp(score_prop):
                score = round(float(mol.GetProp(score_prop)), 3)
        except Exception:
            score = None
        display_name = os.path.splitext(fname)[0]
        solutions.append(DockSolution(full_path, display_name, score))

    if solutions:
        all_solutions.directories.append(os.path.basename(results_dir.rstrip(os.sep)))
        all_solutions.results.append(solutions)

    return all_solutions
