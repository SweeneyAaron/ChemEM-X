#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:31:00 2025

@author: aaron.sweeney
"""

from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.map_masks.protocols import ConfidenceMap,  SignificantFeatures, BindingSiteMask, AutoMask


class RunAutoMask(Command):
    @classmethod 
    def run(cls, chemem, query):
        
        '''
        current_model = chemem.parameters.get_parameter('current_model')
        current_map = chemem.parameters.get_parameter('current_map')
        if current_map is not None and current_model is not None:
            
            if chemem.rendered_site:
                chemem.rendered_site.reset() 
                chemem.rendered_site = None
            
            
            if chemem.parameters.get_parameter('confidence_map') is not None:
                confidence_map = chemem.parameters.get_parameter('confidence_map')
                confidence_map.value.region = confidence_map.value.full_region()
            
            
            
            automask = AutoMask(chemem.session,
                     current_model,
                     current_map
                     )
            
            automask.run()
            '''
        print('------->>>', query)
            
        
        
        
class RunSignificantFeatures(Command):
    
    @classmethod 
    def js_error(cls):
        return f'SignificantFeaturesProtocol requires auto-style binding site'
    
    @classmethod 
    def js_code(cls, key, text):
        return f'addSigFeatListItem("{text}", {key});'
    
    @classmethod 
    def run(cls, chemem, query):
        
        
        chemem.check_q = query
        if chemem.rendered_site is not None and chemem.rendered_site.map is not None:
            
            if chemem.rendered_site.binding_site._type !=  "auto":
                chemem.run_js_code(cls.js_code_error())
                
            else:
                if query.use_confidence_map:
                    if chemem.parameters.get_parameter('confidence_map') is None:
                        confidence_map = ConfidenceMap(chemem.session, chemem.rendered_site.map).run()
                        confidence_map_volume = confidence_map.value
                        chemem.parameters.add(confidence_map)
                    
                    else:
                        confidence_map = chemem.parameters.get_parameter('confidence_map') 
                        confidence_map_volume = confidence_map.value
                        confidence_map_volume.region = chemem.rendered_site.map.region
                else:
                    #use the binding site map 
                    confidence_map_volume = chemem.rendered_site.map
                
                
                #mask the binding site
                binding_site_mask = BindingSiteMask(chemem.session, 
                                                    chemem.rendered_site,
                                                    confidence_map=confidence_map_volume)
                
                masked_map = binding_site_mask.run()
                
                significant_features =  SignificantFeatures(chemem.session,
                                                            chemem.rendered_site,
                                                            masked_map)
                features = significant_features.run()
                
                chemem.features = features
                
        
        
