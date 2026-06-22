#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:43:40 2025

@author: aaron.sweeney
"""

import os
import json
import tempfile
from rdkit import Chem
from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.ligand.tools import Protonate
from chimerax.ChemEM.core.tracked_ligands import (
    default_save_file_name,
    write_tracked_ligand_sdf,
    rdmol_with_live_coords,
    build_protonated_mol_preserving_pose,
    safe_file_stem,
)
from chimerax.ChemEM.core.tools import (
    _build_atom_matcher_from_model_and_sdf,
    launch_chemem_job,
    EXPORT_LIGAND_TORSIONS,
)
from chimerax.ChemEM.core.parameters import PathParameter, Parameters
from chimerax.ChemEM.dock.tools import ChemEMSetUp
from chimerax.ChemEM.simulate.tools import build_model_without_ligands
from chimerax.ChemEM.ligand import torsion_minima as _tmin
from chimerax.ChemEM.ligand import isomers as _iso
from chimerax import open_command
from chimerax.core.commands import run as cx_run


def _js_call(chemem, fn, *args):
    """Call a JS function (guarded by a typeof check) with JSON-encoded args."""
    arg_str = ", ".join(json.dumps(a) for a in args)
    chemem.run_js_code(f'if (typeof {fn} === "function") {fn}({arg_str});')

class ProtonateSmiles(Command):
    @classmethod 
    def js_code(cls, smiles):
        js_code = f'addSmilesToProtonationList("{smiles}");'
        return js_code 
    
    @classmethod
    def run(cls,chemem, query):
        #need to delete the list and images at this stage TODO!
        print('---------HERERERERER')
        chemem.run_js_code("removeAllProtonatedSmiles();")
        p  = Protonate.from_query(query)
        p.protonate()
        
        for state in p.protonation_states:
            js_code = cls.js_code(state)
            chemem.run_js_code(js_code)
        
        chemem.current_protonation_states = p
        
class CleanProtonationImage(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_protonation_states is not None:
            if chemem.current_protonation_states.current_image_file is not None:
                chemem.current_protonation_states.remove_temporary_file()

class ShowLigandSmilesImage(Command):
    
    @classmethod 
    def js_code(cls, image_path):
        js_code = f'displayChemicalStructure("{image_path}");'
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        
        if chemem.current_protonation_states is not None:
            if chemem.current_protonation_states.current_image_file is not None:
                chemem.current_protonation_states.remove_temporary_file()
            try:
                
                image_idx = chemem.current_protonation_states.protonation_states.index(query)
                chemem.current_protonation_states.save_image_temporarily(image_idx)
                file_path = chemem.current_protonation_states.current_image_file.name
                if file_path is not None:
                    
                    js_code = cls.js_code(file_path)
                    chemem.run_js_code(js_code)
                    
                    #chemem.current_protonation_states.remove_temporary_file()
            except ValueError:
                js_code = f'alert("Smiles image not found {query}");'
                chemem.run_js_code(js_code)
                

class GetTrackedLigands(Command):
    @classmethod
    def run(cls, chemem, query):
        if hasattr(chemem, "push_tracked_ligands_to_ui"):
            chemem.push_tracked_ligands_to_ui()


class ViewTrackedLigand(Command):
    @staticmethod
    def render_2d_structure(chemem, record):
        """Render a tracked ligand's 2D depiction into the #chemical-structure viewer."""
        matcher = record.get("atom_matcher")
        rdmol = getattr(matcher, "rdmol", None)
        if rdmol is None:
            return

        # Clean up any previous temporary image (mirrors ShowLigandSmilesImage).
        if chemem.current_protonation_states is not None:
            if chemem.current_protonation_states.current_image_file is not None:
                chemem.current_protonation_states.remove_temporary_file()

        p = Protonate("")
        img = p.draw_molecule_from_mol(rdmol)
        if img is None:
            return

        p.images = [img]
        p.save_image_temporarily(0)
        # Stash so the temp file is cleaned on the next render cycle.
        chemem.current_protonation_states = p
        chemem.run_js_code(ShowLigandSmilesImage.js_code(p.current_image_file.name))

    @classmethod
    def run(cls, chemem, query):
        record = chemem.get_tracked_ligand_record(query)
        if record is None:
            chemem.run_js_code('alert("Tracked ligand not found.");')
            return

        model = record.get("model")
        model_id = getattr(model, "id", None)
        if model_id is None:
            chemem.run_js_code('alert("Tracked ligand model is no longer available.");')
            return

        model_spec = f'#{ ".".join(str(i) for i in model_id) }'
        cx_run(chemem.session, f"show {model_spec}")
        cx_run(chemem.session, f"view {model_spec}")

        cls.render_2d_structure(chemem, record)


class ProtonateTrackedLigand(Command):
    @classmethod
    def run(cls, chemem, query):
        # query is a ProtonationParameter; query.smiles carries the tracked id.
        record = chemem.get_tracked_ligand_record(query.smiles)
        if record is None:
            chemem.run_js_code('alert("Tracked ligand not found.");')
            return

        matcher = record.get("atom_matcher")
        rdmol = getattr(matcher, "rdmol", None)
        if rdmol is None:
            chemem.run_js_code('alert("Tracked ligand has no structure to protonate.");')
            return

        try:
            heavy = Chem.RemoveHs(Chem.Mol(rdmol))
            smiles = Chem.MolToSmiles(heavy)
        except Exception as e:
            chemem.run_js_code(
                f"alert({json.dumps(f'Unable to derive SMILES from tracked ligand: {e}')});"
            )
            return

        chemem.run_js_code("removeAllProtonatedSmiles();")
        p = Protonate(smiles, query.min_pH, query.max_pH, query.pka_prec)
        p.protonate()

        tracked_id = chemem._normalise_ligand_id(query.smiles)
        for state in p.protonation_states:
            chemem.run_js_code(
                f"addSmilesToProtonationList({json.dumps(state)}, {json.dumps(tracked_id)});"
            )

        chemem.current_protonation_states = p


class ApplyProtonationToTrackedLigand(Command):
    @staticmethod
    def _sync_ligand_parameter(chemem, tracked_id, new_path):
        params = chemem.parameters.parameters.get("Ligands", [])
        for p in params:
            if chemem._normalise_ligand_id(p.name) == tracked_id:
                p.value = new_path
                return
        chemem.parameters.add_list_parameter("Ligands", PathParameter(tracked_id, new_path))

    @classmethod
    def run(cls, chemem, query):
        # query is a SmilesParameter: name == tracked id, value == chosen SMILES.
        tracked_id = chemem._normalise_ligand_id(query.name)
        record = chemem.get_tracked_ligand_record(tracked_id)
        if record is None:
            chemem.run_js_code('alert("Tracked ligand not found.");')
            return

        original = rdmol_with_live_coords(record)
        if original is None:
            chemem.run_js_code('alert("Tracked ligand has no structure to protonate.");')
            return

        try:
            new_mol = build_protonated_mol_preserving_pose(original, query.value)
        except Exception as e:
            chemem.run_js_code(
                f"alert({json.dumps(f'Could not build protonated ligand: {e}')});"
            )
            return

        if new_mol is None:
            chemem.run_js_code('alert("Could not transfer pose to protonated ligand.");')
            return

        tmp_dir = tempfile.mkdtemp(prefix="chemem_proton_")
        new_sdf = os.path.join(tmp_dir, "protonated.sdf")
        try:
            writer = Chem.SDWriter(new_sdf)
            try:
                writer.write(new_mol)
            finally:
                writer.close()
        except Exception as e:
            chemem.run_js_code(
                f"alert({json.dumps(f'Could not write protonated SDF: {e}')});"
            )
            return

        old_model = record.get("model")

        # Open the new model BEFORE removing the old one. The 'add models' trigger
        # fires while the old (current) model is still valid, and we point the
        # tracked record at the new model before removing the old, so the prune on
        # the 'remove models' trigger does not drop our record.
        opened = cx_run(chemem.session, f'open "{new_sdf}"')
        new_model = next((m for m in opened if hasattr(m, "atoms")), None)
        if new_model is None:
            chemem.run_js_code('alert("Failed to open protonated ligand.");')
            return

        matcher = _build_atom_matcher_from_model_and_sdf(new_model, new_sdf)
        record.update({
            "model": new_model,
            "model_id": tuple(new_model.id),
            "atom_matcher": matcher,
            "source_sdf_path": new_sdf,
            "dirty": True,
            "last_saved_path": None,
        })

        # Now remove the old model; the record already references the new one.
        if (
            old_model is not None
            and old_model is not new_model
            and getattr(old_model, "id", None) is not None
            and chemem.session.models.have_id(tuple(old_model.id))
        ):
            chemem.session.models.remove([old_model])

        cls._sync_ligand_parameter(chemem, tracked_id, new_sdf)
        chemem.run_js_code(
            f"if (typeof updateLigandListEntry === 'function') "
            f"updateLigandListEntry({json.dumps(str(tracked_id))}, {json.dumps(new_sdf)});"
        )

        if hasattr(chemem, "push_tracked_ligands_to_ui"):
            chemem.push_tracked_ligands_to_ui()

        # Repaint 2D + 3D for the replaced model.
        ViewTrackedLigand.run(chemem, tracked_id)


class SaveTrackedLigandSdf(Command):
    @classmethod
    def run(cls, chemem, query):
        record = chemem.get_tracked_ligand_record(query)
        if record is None:
            chemem.run_js_code('alert("Tracked ligand not found.");')
            return

        output_path = record.get("last_saved_path") or record.get("source_sdf_path")
        if not output_path:
            chemem.run_js_code(
                'alert("No previous save path found. Use Save As for this ligand first.");'
            )
            return

        output_dir = os.path.dirname(output_path) or "."
        if not os.path.isdir(output_dir):
            chemem.run_js_code(
                f"alert({json.dumps(f'Output directory does not exist: {output_dir}')});"
            )
            return

        try:
            saved_path = write_tracked_ligand_sdf(record, output_path)
        except Exception as e:
            chemem.run_js_code(
                f"alert({json.dumps(f'Unable to save tracked ligand SDF: {e}')} );"
            )
            return

        if saved_path is None:
            chemem.run_js_code('alert("Unable to save tracked ligand SDF.");')
            return

        record["last_saved_path"] = saved_path
        record["dirty"] = False
        if hasattr(chemem, "push_tracked_ligands_to_ui"):
            chemem.push_tracked_ligands_to_ui()
        message_json = json.dumps(f"Saved ligand to: {saved_path}")
        chemem.run_js_code(
            f'if (typeof showTrackedLigandStatus === "function") showTrackedLigandStatus({message_json}, "success");'
        )


class SaveTrackedLigandSdfAs(Command):
    @staticmethod
    def _next_available_path(path):
        if not os.path.exists(path):
            return path

        base, ext = os.path.splitext(path)
        idx = 1
        candidate = f"{base}_{idx}{ext}"
        while os.path.exists(candidate):
            idx += 1
            candidate = f"{base}_{idx}{ext}"
        return candidate

    @classmethod
    def run(cls, chemem, query):
        record = chemem.get_tracked_ligand_record(query)
        if record is None:
            chemem.run_js_code('alert("Tracked ligand not found.");')
            return

        folder_dialog = open_command.dialog.OpenFolderDialog(
            chemem.session.ui.main_window, chemem.session
        )
        output_dir = folder_dialog.get_path()
        if output_dir is None:
            return

        if not os.path.isdir(output_dir):
            chemem.run_js_code(f"alert({json.dumps(f'Invalid output directory: {output_dir}')});")
            return

        default_name = default_save_file_name(record, ligand_id=query)
        output_path = os.path.join(output_dir, default_name)
        output_path = cls._next_available_path(output_path)

        try:
            saved_path = write_tracked_ligand_sdf(record, output_path)
        except Exception as e:
            chemem.run_js_code(
                f"alert({json.dumps(f'Unable to save tracked ligand SDF: {e}')} );"
            )
            return

        if saved_path is None:
            chemem.run_js_code('alert("Unable to save tracked ligand SDF.");')
            return

        record["last_saved_path"] = saved_path
        record["dirty"] = False
        if hasattr(chemem, "push_tracked_ligands_to_ui"):
            chemem.push_tracked_ligands_to_ui()
        message_json = json.dumps(f"Saved ligand to: {saved_path}")
        chemem.run_js_code(
            f'if (typeof showTrackedLigandStatus === "function") showTrackedLigandStatus({message_json}, "success");'
        )


# ===========================================================================
# Ligand-tab torsion-minima flicking (standalone; reads exported profiles).
# ===========================================================================
class LigandTorsionRows(Command):
    """Build the rotatable-bond rows (with minima) for a tracked ligand from the
    exported torsion_profiles.json and push them to the Ligand tab."""

    @classmethod
    def run(cls, chemem, query):
        ligand_id = chemem._normalise_ligand_id(query)
        record = chemem.get_tracked_ligand_record(query)
        if record is None:
            chemem._ligand_torsion_rows = {}
            _js_call(chemem, "renderLigandTorsionRows", [])
            _js_call(chemem, "showLigandTorsionStatus", "Tracked ligand not found.")
            return

        exported = _tmin.resolve_exported_dir(chemem, ligand_id)
        doc = _tmin.load_torsion_profiles(exported) if exported else None
        if doc is None:
            chemem._ligand_torsion_rows = {}
            _js_call(chemem, "renderLigandTorsionRows", [])
            _js_call(chemem, "showLigandTorsionStatus",
                     "No exported torsion profiles for this ligand. Use \"Export torsions\".")
            return

        try:
            rows = _tmin.build_ligand_torsion_rows(record, doc, ligand_id)
        except Exception as e:
            _js_call(chemem, "alert", f"Could not build torsion rows: {e}")
            return

        chemem._ligand_torsion_rows = {r["key"]: r for r in rows}
        _js_call(chemem, "renderLigandTorsionRows", [_tmin.slim_row(r) for r in rows])
        if rows:
            _js_call(chemem, "showLigandTorsionStatus", "")
        else:
            _js_call(chemem, "showLigandTorsionStatus",
                     "No exported torsion profiles match this ligand's rotatable bonds.")


def _step_ligand_torsion(chemem, key, direction):
    rows = getattr(chemem, "_ligand_torsion_rows", {}) or {}
    row = rows.get(key)
    if row is None:
        _js_call(chemem, "alert", "Torsion not found; re-select the ligand.")
        return

    record = chemem.get_tracked_ligand_record(row["ligand_id"])
    if record is None or record.get("model") is None:
        _js_call(chemem, "alert", "Tracked ligand model is no longer available.")
        return

    minima = row.get("minima") or []
    if not minima:
        _js_call(chemem, "alert", "No minima for this bond.")
        return

    try:
        xyz = _tmin.local_coords(record)
        i, a, b, l = row["rd_atoms"]
        cur = _tmin.dihedral(xyz, i, a, b, l)
        target = _tmin.next_minimum_angle([m["angle"] for m in minima], cur, direction)
        _tmin.apply_torsion_target(record, row, target)
    except Exception as e:
        _js_call(chemem, "alert", f"Could not step torsion: {e}")
        return

    record["dirty"] = True
    if hasattr(chemem, "push_tracked_ligands_to_ui"):
        chemem.push_tracked_ligands_to_ui()

    row["current_angle"] = float(target) % 360.0
    row["current_idx"] = _tmin.nearest_minimum_index(minima, row["current_angle"])
    _js_call(chemem, "renderLigandTorsionProfile", _tmin.profile_payload(row))


class StepLigandTorsionForward(Command):
    @classmethod
    def run(cls, chemem, query):
        _step_ligand_torsion(chemem, str(query), 1)


class StepLigandTorsionBackward(Command):
    @classmethod
    def run(cls, chemem, query):
        _step_ligand_torsion(chemem, str(query), -1)


class ShowLigandTorsionProfile(Command):
    @classmethod
    def run(cls, chemem, query):
        key = str(query)
        rows = getattr(chemem, "_ligand_torsion_rows", {}) or {}
        row = rows.get(key)
        if row is None:
            return
        record = chemem.get_tracked_ligand_record(row["ligand_id"])
        if record is not None and record.get("model") is not None:
            try:
                xyz = _tmin.local_coords(record)
                i, a, b, l = row["rd_atoms"]
                row["current_angle"] = _tmin.dihedral(xyz, i, a, b, l) % 360.0
                row["current_idx"] = _tmin.nearest_minimum_index(row.get("minima") or [],
                                                                 row["current_angle"])
            except Exception:
                pass
        _js_call(chemem, "renderLigandTorsionProfile", _tmin.profile_payload(row))


class ExportLigandTorsions(Command):
    """Generate the genuine intrinsic torsion profiles for a tracked ligand by
    running the ChemEM backend export (protein + this ligand) into a per-ligand
    subdirectory of the output folder. On completion the Ligand-tab flicker is
    refreshed (see CHEMEM.remove_task). query is the ligand_id string."""

    @classmethod
    def run(cls, chemem, query):
        ligand_id = chemem._normalise_ligand_id(query)
        record = chemem.get_tracked_ligand_record(query)
        if record is None or record.get("model") is None:
            _js_call(chemem, "alert", "Tracked ligand not available.")
            return

        backend = chemem.parameters.get_value("chememBackendPath")
        if not backend:
            _js_call(chemem, "alert", "Set the ChemEM backend in the Setup tab first.")
            return

        output_param = chemem.parameters.get_parameter("output")
        if output_param is None or not output_param.value:
            _js_call(chemem, "alert", "Set the output directory in the Setup tab first.")
            return

        protein = chemem.parameters.get_parameter("current_model")
        if protein is None:
            _js_call(chemem, "alert",
                     "Load a structure (current model) before exporting torsions.")
            return

        # Capture the ligand's current pose into a fresh SDF for the backend.
        tmp_dir = tempfile.mkdtemp(prefix="chemem_lig_torsion_")
        lig_sdf = os.path.join(tmp_dir, f"{safe_file_stem(ligand_id)}.sdf")
        try:
            if write_tracked_ligand_sdf(record, lig_sdf) is None:
                _js_call(chemem, "alert", "Could not write the ligand SDF for export.")
                return
        except Exception as e:
            _js_call(chemem, "alert", f"Could not write the ligand SDF: {e}")
            return

        # Isolate this run in its own subdirectory so it neither clobbers nor is
        # clobbered by a full simulation export. os.makedirs because ChemEMSetUp's
        # mkdir() does not create parents.
        run_dir = os.path.join(output_param.value, "ligand_torsions", safe_file_stem(ligand_id))
        try:
            os.makedirs(run_dir, exist_ok=True)
        except Exception as e:
            _js_call(chemem, "alert", f"Could not create export directory: {e}")
            return

        # A ligand-stripped protein copy is required boilerplate for the export
        # protocol; the torsion profiles themselves are per-ligand intrinsic and
        # do not depend on it.
        model_no_ligands = build_model_without_ligands(protein)
        chemem.session.models.add([model_no_ligands])

        build_parameters = Parameters()
        build_parameters.add(output_param)
        build_parameters.parameters["current_model"] = model_no_ligands

        try:
            chemem_setup = ChemEMSetUp.from_parameters_object(
                chemem.session,
                build_parameters,
                backend,
                ["export"],
                ligand_paths=[lig_sdf],
                output_override=run_dir,
            )
            command = chemem_setup.run_command
        except Exception as e:
            chemem.session.models.remove([model_no_ligands])
            _js_call(chemem, "alert", f"Could not prepare the torsion export: {e}")
            return

        # The completion handler reads this to refresh the right ligand's flicker.
        chemem._pending_ligand_torsion_export = {"ligand_id": ligand_id, "dir": run_dir}

        launch_chemem_job(chemem, command, EXPORT_LIGAND_TORSIONS, "Export Ligand Torsions")
        chemem.session.models.remove([model_no_ligands])

        _js_call(chemem, "showLigandTorsionStatus",
                 "Exporting torsions via the ChemEM backend… the profile will load when the job finishes.")


# ===========================================================================
# Ligand-tab in-place isomer editing (E/Z flip + R/S inversion).
# ===========================================================================
class ShowLigandIsomers(Command):
    @classmethod
    def run(cls, chemem, query):
        ligand_id = chemem._normalise_ligand_id(query)
        record = chemem.get_tracked_ligand_record(query)
        if record is None:
            _js_call(chemem, "renderLigandIsomers",
                     {"ligand_id": ligand_id, "double_bonds": [], "chiral_centers": []})
            return
        try:
            iso = _iso.list_isomers(record)
        except Exception as e:
            _js_call(chemem, "alert", f"Could not analyse stereochemistry: {e}")
            return
        payload = {"ligand_id": ligand_id}
        payload.update(iso)
        _js_call(chemem, "renderLigandIsomers", payload)


class ToggleLigandDoubleBond(Command):
    @classmethod
    def run(cls, chemem, query):
        parts = str(query).split(_tmin.SEP)
        if len(parts) != 2:
            return
        ligand_id, db_id = parts
        record = chemem.get_tracked_ligand_record(ligand_id)
        if record is None or record.get("model") is None:
            _js_call(chemem, "alert", "Tracked ligand not available.")
            return
        try:
            _, sa, sb = db_id.split("|")
            ok, msg = _iso.toggle_double_bond(record, (int(sa), int(sb)))
        except Exception as e:
            _js_call(chemem, "alert", f"Could not flip double bond: {e}")
            return
        if not ok:
            _js_call(chemem, "alert", msg or "Could not flip this double bond.")
            return
        record["dirty"] = True
        if hasattr(chemem, "push_tracked_ligands_to_ui"):
            chemem.push_tracked_ligands_to_ui()
        ShowLigandIsomers.run(chemem, ligand_id)
        ViewTrackedLigand.render_2d_structure(chemem, record)


class InvertLigandChiralCenter(Command):
    @classmethod
    def run(cls, chemem, query):
        parts = str(query).split(_tmin.SEP)
        if len(parts) != 2:
            return
        ligand_id, rs_id = parts
        record = chemem.get_tracked_ligand_record(ligand_id)
        if record is None or record.get("model") is None:
            _js_call(chemem, "alert", "Tracked ligand not available.")
            return
        try:
            _, sidx = rs_id.split("|")
            ok, msg = _iso.invert_chiral_center(record, int(sidx))
        except Exception as e:
            _js_call(chemem, "alert", f"Could not invert chiral centre: {e}")
            return
        if not ok:
            _js_call(chemem, "alert", msg or "Could not invert this chiral centre.")
            return
        record["dirty"] = True
        if hasattr(chemem, "push_tracked_ligands_to_ui"):
            chemem.push_tracked_ligands_to_ui()
        ShowLigandIsomers.run(chemem, ligand_id)
        ViewTrackedLigand.render_2d_structure(chemem, record)

