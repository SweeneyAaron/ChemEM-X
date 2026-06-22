#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for tracked ligand model metadata and SDF writing.
"""

import os
from rdkit import Chem
from rdkit.Geometry import Point3D


def normalise_model_id(model_id):
    if model_id is None:
        return None

    if isinstance(model_id, (tuple, list)):
        if len(model_id) == 0:
            return None
        return ".".join(str(i) for i in model_id)

    model_id_text = str(model_id).strip()
    if not model_id_text:
        return None
    return model_id_text.lstrip("#")


def safe_file_stem(value):
    text = str(value)
    safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in text)
    safe = safe.strip("._")
    return safe if safe else "ligand"


def tracked_ligand_display_name(record, fallback_name="ligand"):
    source = record.get("source_sdf_path")
    if source:
        source_base = os.path.basename(str(source).strip())
        if source_base:
            return source_base

    model = record.get("model")
    model_name = getattr(model, "name", None)
    if model_name:
        return str(model_name)

    return fallback_name


def default_save_file_name(record, ligand_id=None, suffix="_moved.sdf"):
    source = record.get("source_sdf_path")
    if source:
        stem = os.path.splitext(os.path.basename(source))[0]
    else:
        model = record.get("model")
        model_name = getattr(model, "name", None) or "ligand"
        model_id = None
        if model is not None and getattr(model, "id", None) is not None:
            model_id = tuple(model.id)
        else:
            model_id = record.get("model_id")
        model_id_text = normalise_model_id(model_id) or "unknown_model"
        stem = f"{model_name}_{model_id_text}"

    if ligand_id is not None:
        stem = f"{stem}_{ligand_id}"

    return f"{safe_file_stem(stem)}{suffix}"


def rdmol_with_live_coords(record):
    """Return a copy of the tracked rdmol with conformer positions refreshed
    from the live ChimeraX atom coordinates (via the atom matcher)."""
    matcher = record.get("atom_matcher")
    if matcher is None:
        return None

    rd_mol = getattr(matcher, "rdmol", None)
    if rd_mol is None or rd_mol.GetNumConformers() == 0:
        return None

    rd_to_cx = getattr(matcher, "rd_to_cx", {})
    if not rd_to_cx:
        return None

    mol = Chem.Mol(rd_mol)
    conf = mol.GetConformer()

    for rd_idx, cx_atom in rd_to_cx.items():
        # Use scene_coord (= scene_position * local coord), NOT coord, so the
        # written pose reflects BOTH coordset edits AND whole-model moves the user
        # makes in the GUI (moving the model only changes its position transform,
        # leaving local coord untouched). This also matches the protein PDB, which
        # ChimeraX `save` writes in scene coordinates when no relModel is given.
        coord = getattr(cx_atom, "scene_coord", None)
        if coord is None:
            coord = getattr(cx_atom, "coord", None)
        if coord is None:
            continue
        x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
        conf.SetAtomPosition(int(rd_idx), Point3D(x, y, z))

    return mol


def write_tracked_ligand_sdf(record, output_path):
    mol = rdmol_with_live_coords(record)
    if mol is None:
        return None

    writer = Chem.SDWriter(output_path)
    try:
        writer.write(mol)
    finally:
        writer.close()

    return output_path


def build_protonated_mol_preserving_pose(original_rdmol, protonated_smiles):
    """Build a new RDKit mol from a protonated SMILES while preserving the
    heavy-atom 3D pose of ``original_rdmol``.

    Protonation only changes hydrogen count and formal charges, so the heavy-atom
    skeleton is shared. Heavy-atom coordinates are copied from the original mol;
    hydrogens are then placed from the existing geometry via ``AddHs(addCoords=True)``.
    Formal charges come for free from parsing the protonated SMILES.
    """
    if original_rdmol is None or not protonated_smiles:
        return None

    target = Chem.MolFromSmiles(protonated_smiles)
    if target is None:
        return None

    # Heavy-atom skeleton of the original (keeps its 3D conformer).
    orig_heavy = Chem.RemoveHs(Chem.Mol(original_rdmol))
    if orig_heavy.GetNumConformers() == 0:
        return None
    orig_conf = orig_heavy.GetConformer()

    # Map original heavy-atom index -> target heavy-atom index.
    pairs = []
    match = target.GetSubstructMatch(orig_heavy)
    if match and len(match) == orig_heavy.GetNumAtoms():
        # match[o_i] is the target index corresponding to orig_heavy atom o_i.
        pairs = [(o_i, match[o_i]) for o_i in range(orig_heavy.GetNumAtoms())]
    else:
        # Fallback to MCS when the direct match fails (aromaticity/kekulization).
        from rdkit.Chem import rdFMCS
        mcs = rdFMCS.FindMCS(
            [target, orig_heavy],
            matchValences=False,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=10,
        )
        if not mcs or not mcs.smartsString:
            return None
        patt = Chem.MolFromSmarts(mcs.smartsString)
        if patt is None:
            return None
        t_match = target.GetSubstructMatch(patt)
        o_match = orig_heavy.GetSubstructMatch(patt)
        if not t_match or not o_match or len(t_match) != len(o_match):
            return None
        pairs = list(zip(o_match, t_match))

    if not pairs:
        return None

    # Seed a conformer on the target with the original heavy-atom coordinates.
    conf = Chem.Conformer(target.GetNumAtoms())
    for o_i, t_i in pairs:
        pos = orig_conf.GetAtomPosition(int(o_i))
        conf.SetAtomPosition(int(t_i), Point3D(pos.x, pos.y, pos.z))
    target.RemoveAllConformers()
    target.AddConformer(conf, assignId=True)

    Chem.SanitizeMol(target)
    target_h = Chem.AddHs(target, addCoords=True)
    return target_h
