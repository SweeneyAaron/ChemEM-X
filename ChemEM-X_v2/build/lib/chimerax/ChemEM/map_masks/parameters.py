#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 22:50:15 2025

@author: aaron.sweeney

The mask tab now drives the ChemEM ``alpha_mask`` protocol via the backend and
collects its options using the standard parameter classes in
``chimerax.ChemEM.core.parameters`` (FloatParameter / IntParameter /
StringParameter / BooleanParameter), added through the ``AddMaskParameter``
command. The previous mask-specific parameter classes (``MaskOptionsParameter``,
``ConfidenceMapParameter``) belonged to the retired native masking path and have
been removed.
"""
