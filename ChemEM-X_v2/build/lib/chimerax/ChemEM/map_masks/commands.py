#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:13:34 2025

@author: aaron.sweeney
"""

from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.map_masks.tools import AutoMask
#from chimerax.ChemEM.map_masks.protocols import ConfidenceMap,  SignificantFeatures, BindingSiteMask, AutoMask


class RunAutoMask(Command):
    @classmethod 
    def run(cls, chemem, query):
        
        
        current_model = chemem.parameters.get_parameter('current_model')
        current_map = chemem.parameters.get_parameter('current_map')
        if current_map is not None and current_model is not None:
            
            
            if chemem.parameters.get_parameter('confidence_map') is not None:
                confidence_map = chemem.parameters.get_parameter('confidence_map')
                confidence_map.value.region = confidence_map.value.full_region()
            
            
            
            automask = AutoMask(chemem.session,
                     current_model,
                     current_map,
                     protocol=query.value[0]
                     )
            
            automask.run()
        