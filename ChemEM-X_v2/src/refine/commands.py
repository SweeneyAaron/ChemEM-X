#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commands for the Refine tab.

Mirrors the Dock tab: the JS pushes the chosen protocol + options via
AddRefineParameter (building chemem.refine_parameters), the selected poses via
SetRefinePoses, then fires RunRefine. RunRefine reuses ChemEMSetUp to write the
conf + build the `chemem <conf> --<protocol> --<opt> <val>` command and launches
it as a job. LoadRefineJob parses the protocol-specific output directory.

@author: aaron.sweeney
"""

import os
import json
from chimerax.core.commands import run
from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.dock.tools import ChemEMSetUp
from chimerax.ChemEM.refine.tools import (
    resolve_refine_ligand_paths,
    get_refine_results,
    make_refine_run_dir,
    SR2_SCORE_PROP,
)
from chimerax.ChemEM.core.tools import (
    CHEMEM_JOB,
    get_output_from_conf,
    launch_chemem_job,
    register_tracked_ligand,
)


class ClearRefineParameters(Command):
    @classmethod
    def run(cls, chemem, query):
        # runRefine() pushes the full UI state every run; clear first so stale
        # protocol/option tokens from a previous run (e.g. a different method, or
        # an unchecked toggle) cannot leak into this command.
        chemem.refine_parameters._clear()


class AddRefineParameter(Command):
    @classmethod
    def run(cls, chemem, query):
        chemem.refine_parameters.add(query)


class SetRefinePoses(Command):
    @classmethod
    def run(cls, chemem, query):
        # query is a StringParameter whose value is the JSON-encoded selection
        # list: [{"source": "docked"|"tracked", "key": <full_path|ligand_id>}, ...]
        try:
            chemem._refine_selected_poses = json.loads(query.value)
        except (ValueError, TypeError, AttributeError):
            chemem._refine_selected_poses = []


class GetRefinePoses(Command):
    @classmethod
    def js_code(cls, chemem):
        tracked = chemem.tracked_ligands_payload()

        docked = []
        if chemem.dock_results is not None:
            for directory, solutions in zip(chemem.dock_results.directories,
                                            chemem.dock_results.results):
                for sol in solutions:
                    docked.append({
                        "full_path": sol.full_path,
                        "display_name": sol.display_name,
                        "score": sol.score,
                        "directory": directory,
                    })

        payload = {"tracked": tracked, "docked": docked}
        return f"renderRefinePosePicker({json.dumps(payload)});"

    @classmethod
    def run(cls, chemem, query):
        chemem.run_js_code(cls.js_code(chemem))


class RunRefine(Command):
    @classmethod
    def run(cls, chemem, query):
        backend = chemem.parameters.get_value("chememBackendPath")
        if backend is None:
            return

        base_output = chemem.parameters.get_value("output")
        if not base_output:
            chemem.run_js_code('alert("Set an output directory on the Setup tab first.");')
            return

        selected = getattr(chemem, "_refine_selected_poses", None) or []

        options = [(k, v.value) for k, v in chemem.refine_parameters.parameters.items()
                   if v.value != 'protocol']
        protocols = [k for k, v in chemem.refine_parameters.parameters.items()
                     if v.value == "protocol"]

        # isolate this run in its own subdir so conf/inputs/results never clobber
        label = "slr2" if "smart-ligand-refine2" in protocols else "md"
        run_dir = make_refine_run_dir(base_output, label)

        ligand_paths = resolve_refine_ligand_paths(chemem, selected, run_dir)
        if not ligand_paths:
            chemem.run_js_code(
                'alert("Select at least one pose to refine '
                '(a tracked ligand or a docked solution).");'
            )
            return

        chemem_setup = ChemEMSetUp.from_parameters_object(
            chemem.session,
            chemem.parameters,
            backend,
            protocols,
            options,
            ligand_paths=ligand_paths,
            output_override=run_dir,
        )

        command = chemem_setup.run_command
        launch_chemem_job(chemem, command, CHEMEM_JOB, "Refine", load_type="refine")


class LoadRefineJob(Command):
    @classmethod
    def js_code(cls, results):
        payload = {
            "directories": list(results.directories),
            "results": [
                [
                    {
                        "full_path": s.full_path,
                        "display_name": s.display_name,
                        "score": s.score,
                    }
                    for s in sols
                ]
                for sols in results.results
            ],
        }
        return f"renderRefineResults({json.dumps(payload)});"

    @classmethod
    def run(cls, chemem, query):
        if query not in chemem.job_handeler.jobs:
            return

        job = chemem.job_handeler.jobs[query]
        com = job.command.split(' ')
        conf = com[1]
        options = [i for i in com if '--' in i]

        conf_data = get_output_from_conf(conf)
        if conf_data is None:
            return

        output = conf_data['output']
        results = None
        if '--smart-ligand-refine2' in options:
            results = get_refine_results(os.path.join(output, 'smart_refine'),
                                         score_prop=SR2_SCORE_PROP)
        elif '--refine' in options:
            # MD refine writes plain coordinates with no embedded score.
            results = get_refine_results(os.path.join(output, 'refine'),
                                         score_prop=SR2_SCORE_PROP)

        if results is None:
            return

        chemem.refine_results = results
        cls._auto_track_results(chemem, results)
        chemem.run_js_code(cls.js_code(results))
        # refresh the pose picker so the newly-tracked refined poses appear in it
        chemem.run_js_code(GetRefinePoses.js_code(chemem))

    @staticmethod
    def _auto_track_results(chemem, results):
        """Register each refined pose as a tracked ligand so it appears under
        'Tracked ligands' in the pose picker and can be re-refined (with any GUI
        edits captured via its atom matcher). Skips poses already tracked, and
        links the opened model into results._gui so View/Hide reuse it."""
        tracked = getattr(chemem, "_tracked_ligands", {}) or {}
        existing = {rec.get("source_sdf_path"): rec.get("model")
                    for rec in tracked.values()}
        for sols in results.results:
            for sol in sols:
                path = sol.full_path
                if path in existing:
                    if existing[path] is not None:
                        results._gui[path] = existing[path]
                    continue
                ligand_id = register_tracked_ligand(chemem, path)
                if ligand_id is None:
                    continue
                record = chemem.get_tracked_ligand_record(ligand_id)
                model = record.get("model") if record else None
                if model is not None:
                    results._gui[path] = model
                    existing[path] = model


class ViewRefineSolution(Command):
    @classmethod
    def run(cls, chemem, query):
        if chemem.refine_results is not None:
            if query.value in chemem.refine_results._gui:
                mdl = chemem.refine_results._gui.get(query.value, None)
                if mdl is not None:
                    run(chemem.session, f'show {mdl.atomspec}')
            else:
                mdl = run(chemem.session, f'open {query.value}')
                chemem.refine_results._gui[query.value] = mdl[0]


class HideRefineSolution(Command):
    @classmethod
    def run(cls, chemem, query):
        if chemem.refine_results is not None:
            mdl = chemem.refine_results._gui.get(query.value, None)
            if mdl is not None:
                run(chemem.session, f'hide {mdl.atomspec}')


class ClearRefineSolutions(Command):
    @classmethod
    def run(cls, chemem, query):
        if chemem.refine_results is not None:
            # iterate the loaded models (values), not the path keys
            for mdl in chemem.refine_results._gui.values():
                if mdl is not None:
                    run(chemem.session, f'close {mdl.atomspec}')
            chemem.refine_results = None


class AddRefineSolutionToStructure(Command):
    @classmethod
    def run(cls, chemem, query):
        from chimerax.ChemEM.simulate.tools import include_tracked_ligand_in_simulation
        if chemem.refine_results is not None:
            mdl = chemem.refine_results._gui.get(query.value, None)
            ligand_id = register_tracked_ligand(chemem, query.value, model=mdl)
            if ligand_id is not None:
                include_tracked_ligand_in_simulation(chemem, ligand_id)
