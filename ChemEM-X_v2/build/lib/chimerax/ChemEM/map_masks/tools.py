#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:25:20 2025

@author: aaron.sweeney

Retired.

The mask tab used to reimplement an older, simplified confidence-map +
significant-features pipeline natively in ChimeraX (the ``AutoMask`` /
``ConfidenceMap`` / ``BindingSiteMask`` / ``SignificantFeatures`` classes that
lived here). The tab now drives the canonical ChemEM ``alpha_mask`` protocol via
the backend instead (see ``map_masks/commands.py``), so this native path has been
removed to avoid drifting from the ChemEM implementation.

The statistical/segmentation helpers that previously lived here
(``estimate_background_distribution``, ``compute_p_values``,
``benjamini_yekutieli``, ``extract_ligand_density``, ``grow_ligand_region`` ...)
remain available in ``chimerax.ChemEM.core.tools`` if ever needed.
"""
