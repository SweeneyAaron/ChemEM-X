#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score-tab commands.

Wire the Score tab UI (template.html) to the scoring engine. Map scores
(CCC/MI/SCI/QScore) run in-process and return instantly; MMGBSA runs in a
background thread so the GUI stays responsive.
"""

import json
import threading

from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.score import tools as score_tools


def _push_table(chemem):
    payload = json.dumps(chemem.score_results.payload())
    chemem.run_js_code(f"renderScoreTable({payload});")


def _alert(chemem, message):
    safe = str(message).replace('"', "'").replace("\n", " ")
    chemem.run_js_code(f'alert("{safe}");')


def _get_emmap_and_resolution(chemem):
    """Returns (emmap, resolution, error_message)."""
    volume = chemem.parameters.get_parameter("current_map")
    if volume is None:
        return None, None, "Select a density map (Setup tab) before scoring map metrics."
    resolution = chemem.parameters.get_value("resolution")
    if resolution is None:
        return None, None, "Set the map resolution (Setup tab) before scoring map metrics."
    try:
        emmap = score_tools.maplike_from_volume(volume)
    except Exception as e:
        return None, None, f"Could not read the density map: {e}"
    return emmap, float(resolution), None


def _parse_payload(query):
    """The Score commands send a JSON-encoded StringParameter."""
    raw = getattr(query, "value", query)
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


class RefreshScoreTargets(Command):
    @classmethod
    def run(cls, chemem, query):
        targets = score_tools.gather_score_targets(chemem)
        payload = json.dumps(score_tools.targets_payload(targets))
        chemem.run_js_code(f"renderScoreTargets({payload});")
        # keep the comparison table in sync on refresh
        _push_table(chemem)


class RunMapScores(Command):
    @classmethod
    def run(cls, chemem, query):
        data = _parse_payload(query)
        keys = data.get("keys", []) or []
        which = [w.lower() for w in (data.get("scores", []) or [])]
        if not keys:
            _alert(chemem, "Select at least one ligand to score.")
            return
        if not which:
            _alert(chemem, "Select at least one score to compute.")
            return

        needs_map = any(w in ("ccc", "mi", "sci", "qscore") for w in which)
        emmap = resolution = None
        if needs_map:
            emmap, resolution, error = _get_emmap_and_resolution(chemem)
            if error:
                _alert(chemem, error)
                return

        targets = {t.key: t for t in score_tools.gather_score_targets(chemem)}
        scored = 0
        for key in keys:
            target = targets.get(key)
            if target is None:
                continue
            try:
                scores = score_tools.score_target(target, emmap, resolution, which)
            except Exception as e:
                scores = {"error": str(e)}
            chemem.score_results.update_scores(key, target.display_name, target.source, scores)
            scored += 1

        _push_table(chemem)
        if scored == 0:
            _alert(chemem, "No matching ligands found - try Refresh ligands.")


class ScoreCurrentSelection(Command):
    @classmethod
    def run(cls, chemem, query):
        data = _parse_payload(query)
        which = [w.lower() for w in (data.get("scores", []) or [])]
        if not which:
            _alert(chemem, "Select at least one score to compute.")
            return

        target = score_tools.selection_target(chemem)
        if target is None or not target.valid:
            _alert(chemem, "Select some atoms in ChimeraX first.")
            return

        emmap, resolution, error = _get_emmap_and_resolution(chemem)
        if error:
            _alert(chemem, error)
            return

        scores = score_tools.score_target(target, emmap, resolution, which)
        chemem.score_results.update_scores(target.key, target.display_name, target.source, scores)
        _push_table(chemem)


class RunMMGBSA(Command):
    @classmethod
    def run(cls, chemem, query):
        data = _parse_payload(query)
        keys = data.get("keys", []) or []

        # Snapshot the parametrised complex on the main thread (touches the live
        # simulation); the heavy energy evaluation runs off-thread.
        try:
            struct = score_tools.build_mmgbsa_complex(chemem)
        except Exception as e:
            _alert(chemem, f"MMGBSA setup failed: {e}")
            return
        if struct is None:
            _alert(chemem, "MMGBSA needs an active simulation containing the ligand. "
                           "Build a simulation on the Build tab first.")
            return

        # default to all simulation rows if none were explicitly selected
        sim_targets = [t for t in score_tools.gather_score_targets(chemem)
                       if t.source == "simulation"]
        if not keys:
            keys = [t.key for t in sim_targets]
        rows = {t.key: t for t in sim_targets}
        targets = [rows[k] for k in keys if k in rows]
        if not targets:
            _alert(chemem, "Select an in-simulation ligand row for MMGBSA.")
            return

        for t in targets:
            chemem.score_results.set_pending(t.key, t.display_name, t.source, "mmgbsa_dG")
        _push_table(chemem)

        platform = getattr(chemem, "platform", None)
        session = chemem.session

        def worker():
            try:
                from chimerax.ChemEM.score import scoring
                components, deltaG = scoring.compute_mmgbsa(struct, platform_name=platform)
            except Exception as e:
                def fail(msg=str(e)):
                    _alert(chemem, f"MMGBSA failed: {msg}")
                    _push_table(chemem)
                session.ui.thread_safe(fail)
                return

            def finish():
                for t in targets:
                    chemem.score_results.update_mmgbsa(
                        t.key, t.display_name, t.source, components, deltaG)
                _push_table(chemem)
            session.ui.thread_safe(finish)

        threading.Thread(target=worker, name="chemem-mmgbsa", daemon=True).start()


class ExportScores(Command):
    @classmethod
    def run(cls, chemem, query):
        table = chemem.score_results
        if not table.rows:
            _alert(chemem, "No scores to export yet.")
            return
        output_dir = chemem.parameters.get_value("output")
        if not output_dir:
            _alert(chemem, "Set an output directory (Setup tab) before exporting.")
            return
        try:
            csv_path, _ = table.export(output_dir)
        except Exception as e:
            _alert(chemem, f"Export failed: {e}")
            return
        _alert(chemem, f"Scores exported to {csv_path}")


class ClearScoreTable(Command):
    @classmethod
    def run(cls, chemem, query):
        chemem.score_results.clear()
        _push_table(chemem)
