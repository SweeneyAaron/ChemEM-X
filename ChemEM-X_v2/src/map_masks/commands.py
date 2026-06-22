#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:13:34 2025

@author: aaron.sweeney

Mask tab commands.

The mask tab drives the canonical ChemEM ``alpha_mask`` protocol by building a
``ChemEMChimera.conf`` + ``chemem <conf> --alpha-mask [args]`` command and
running it as a background job (the same machinery the Dock tab uses). The
results (.mrc maps + centroids.json) are loaded back into ChimeraX. This keeps
the GUI behaviour an exact echo of the ChemEM backend rather than a native
re-implementation.
"""

import os
import glob
import json

from chimerax.core.commands import run as cx_run
from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.dock.tools import ChemEMSetUp
from chimerax.ChemEM.core.tools import ALPHA_MASK_JOB, get_output_from_conf, launch_chemem_job


# The alpha_mask protocol CLI selector. ChemEMSetUp emits this as ``--alpha-mask``
# (ChemEM maps protocol keys to flags via ``"--" + key.replace("_", "-")``).
ALPHA_MASK_PROTOCOL = "alpha-mask"


class ClearMaskParameters(Command):
    """Reset accumulated mask parameters so each Run starts from a clean slate."""
    @classmethod
    def run(cls, chemem, query):
        chemem.mask_parameters._clear()


class AddMaskParameter(Command):
    """Add a single alpha-mask parameter (mirrors AddDockParameter)."""
    @classmethod
    def run(cls, chemem, query):
        if query is not None:
            chemem.mask_parameters.add(query)


class RunAlphaMask(Command):
    @classmethod
    def run(cls, chemem, query):
        backend = chemem.parameters.get_value("chememBackendPath")
        if backend is None:
            chemem.session.logger.warning(
                "[ChemEM] No ChemEM backend selected. Choose a ChemEM executable "
                "before running Alpha Mask."
            )
            return

        if chemem.parameters.get_value("output") is None:
            chemem.session.logger.warning(
                "[ChemEM] No output directory set. Choose an output directory "
                "before running Alpha Mask."
            )
            return

        # Split parameters exactly as RunDocking does: bare flags carry the
        # sentinel value 'protocol', everything else is a valued option.
        options = [(k, v.value) for k, v in chemem.mask_parameters.parameters.items()
                   if v.value != 'protocol']
        protocols = [k for k, v in chemem.mask_parameters.parameters.items()
                     if v.value == 'protocol']

        # Always run the alpha_mask protocol itself.
        if ALPHA_MASK_PROTOCOL not in protocols:
            protocols.append(ALPHA_MASK_PROTOCOL)

        chemem_setup = ChemEMSetUp.from_parameters_object(
            chemem.session,
            chemem.parameters,
            backend,
            protocols,
            options,
        )

        if not chemem_setup.check:
            chemem.session.logger.warning(
                "[ChemEM] Could not build the alpha-mask job (missing output "
                "directory or inputs)."
            )
            return

        launch_chemem_job(chemem, chemem_setup.run_command, ALPHA_MASK_JOB,
                          "Alpha Mask", load_type="alpha-mask")

        # Clear so the next run reflects only the current widget state.
        chemem.mask_parameters._clear()


class LoadAlphaMaskJob(Command):
    """Load the maps written by an alpha-mask job into ChimeraX."""
    @classmethod
    def run(cls, chemem, query):
        if query not in chemem.job_handeler.jobs:
            return

        job = chemem.job_handeler.jobs[query]
        com = job.command.split(' ')
        # Robustly locate the conf path (tolerates a backend path before it).
        conf = next((c for c in com if c.endswith('.conf')), com[1] if len(com) > 1 else None)
        if conf is None:
            return

        conf_data = get_output_from_conf(conf)
        if conf_data is None or 'output' not in conf_data:
            return

        results_dir = os.path.join(conf_data['output'], 'alpha_mask')
        cls.load_maps(chemem, results_dir)

    @staticmethod
    def _is_combined(path):
        return os.path.basename(path).startswith('combined_map')

    @classmethod
    def load_maps(cls, chemem, results_dir):
        if not os.path.isdir(results_dir):
            chemem.session.logger.warning(
                f"[ChemEM] Alpha-mask output not found: {results_dir}"
            )
            return

        mrc_files = glob.glob(os.path.join(results_dir, '*.mrc'))
        if not mrc_files:
            chemem.session.logger.warning(
                f"[ChemEM] No maps found in {results_dir}"
            )
            return

        # Show the combined map(s); load per-feature / per-site maps hidden.
        mrc_files.sort(key=lambda p: (not cls._is_combined(p), os.path.basename(p)))

        loaded = []
        for path in mrc_files:
            models = cx_run(chemem.session, f'open "{path}"')
            if models:
                show = cls._is_combined(path)
                for m in models:
                    m.display = show
                loaded.extend(models)

        chemem.alpha_mask_results = loaded

        centroids_path = os.path.join(results_dir, 'centroids.json')
        if os.path.exists(centroids_path):
            try:
                with open(centroids_path) as f:
                    centroids = json.load(f)
                chemem.session.logger.info(
                    f"[ChemEM] Alpha-mask found {len(centroids)} site(s): "
                    f"{list(centroids.keys())}"
                )
            except (ValueError, OSError):
                pass

        chemem.session.logger.info(
            f"[ChemEM] Loaded {len(loaded)} map(s) from {results_dir}"
        )
