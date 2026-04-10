#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:55:29 2025

@author: aaron.sweeney
"""

import numpy as np
from chimerax.ChemEM.core.tools import estimate_background_distribution, compute_p_values, benjamini_yekutieli, extract_ligand_density
from chimerax.ChemEM.map_masks.parameters import ConfidenceMapParameter
from chimerax.ChemEM.binding_site.protocols import AutoSiteChimera, RenderBindingSite
from chimerax.map_data import ArrayGridData
from chimerax.map import volume_from_grid_data


class MaskMapWithSite:
    pass 

class SignificantFeaturesProtocol:
    def __init__(self, session,
                 binding_site,
                 resolution,
                 confidence_map = None):
        
        
        
        self.session = session 
        self.binding_site = binding_site #rendered site object
        self.resolution = resolution 
        self.confidence_map = None
    
    def process_site(self):
        #first is confidence map!!
        print(self.binding_site.current_map)
        if self.confidence_map is None:
            confidence_map = ConfidenceMap(self.session, self.binding_site.map)
            self.confidence_map = confidence_map.run()
        
        
        
    @classmethod
    def mask(cls,session, rendered_site, resolution):
        instance = cls(session, rendered_site, resolution)
        return instance.process_site()


class AutoMask:
    "run the automated binding site analysis and ligand matching thing from ChemEM2"
    def __init__(self,
                 session,
                 current_model,
                 current_map,
                 use_selected = False):
        
        self.session = session 
        self.model = current_model 
        self.map = current_map
    
    def render_site_distance_map(self, binding_site):
        """
        Add the binding site's distance map to the session and render.
        """
        binding_site.add_to_session(self.session)
    
    def run(self):
        
        self.map.region = self.map.full_region() #unpack map
        confidence_map = ConfidenceMap(self.session, self.map).run()
        print('#######################')
        print(confidence_map.value)
        print('#######################')
        
        binding_sites = AutoSiteChimera(self.session,
                                        self.model,
                                        confidence_map.value) #pass conf map here?
        
        
        binding_sites = binding_sites.run()
        
        for binding_site in binding_sites:
            #need to initialise the volume object without adding it to the session
            binding_site.add_to_session(self.session, add_to_session=False)
            
            render_object = RenderBindingSite(self.session, 
                                        binding_site, 
                                        self.model,
                                        current_map=confidence_map.value, 
                                        add_to_session=True) #don't render yet
            
             
            #mask the binding site
            
            binding_site_mask = BindingSiteMask(self.session, 
                                                render_object,
                                                confidence_map=confidence_map.value).run()
            
            
            #only add binding site to session if it has something in there 
            features = SignificantFeatures(self.session,
                                           render_object,
                                           binding_site_mask).run()
            
            
            #reset 
            render_object.reset()
            self.session.models.remove([binding_site_mask])

            if features:
                binding_site.add_to_session(self.session, add_to_session=True)

        confidence_map.value.display = False
        


            

class SignificantFeatures:
    
    def __init__(self,
                 session, 
                 binding_site,
                 masked_map
                 ):
        self.session = session 
        self.binding_site = binding_site 
        self.masked_map = masked_map 
    
    def extract_features(self):
        
        ligand_densities, ligand_features = extract_ligand_density(self.grid_data_mask,
                                                          self.grid_data_binding_site_distance_map,
                                                          self.apix)
        
        site_ligands = {}
        if ligand_features:
            max_thr = np.max([i['map_amplitude_thr_no_mask'] for i in ligand_features])
            
            for num, (density, features) in enumerate(zip(ligand_densities, ligand_features)):
                features = {key: float(value) for key, value in features.items()}
                grid_data, volume = self.add_map_to_session(density)
                name = f"Feature map {num}"
                volume.name = name
                site_ligands[num] = [volume, features]
            
        self.site_ligands = site_ligands
    
    def filter_features(self):
        #TODO!!
        pass
            
            
    def add_map_to_session(self, densmap):
        grid_data = ArrayGridData(densmap,
                                  self.origin, 
                                  self.apix)
        
        
        volume = volume_from_grid_data(grid_data, self.session, style="surface")
        return grid_data, volume        
    
    def run(self):
        self.grid_data_mask = self.masked_map.matrix() 
        
        self.grid_data_binding_site_distance_map = self.binding_site.binding_site.volume.matrix() 
        self.origin, self.apix = self.binding_site.map.data_origin_and_step()
        self.extract_features()
        print('---------------------------->')
        print(self.site_ligands)
        return self.site_ligands


class BindingSiteMask:
    
    def __init__(self, session, binding_site, confidence_map=None):
        self.session = session 
        self.binding_site = binding_site 
        self.confidence_map = confidence_map
    
    def calculate_binding_site_mask(self):
        
        self.binding_site_masked = self.grid_data_densmap.copy() * (self.grid_data_binding_site > 0.1)
    
    
    def add_map_to_session(self):
        self.grid_data = ArrayGridData(self.binding_site_masked,
                                  self.origin, 
                                  self.apix)
        
        self.volume = volume_from_grid_data(self.grid_data, self.session, style="mesh")
        self.volume.name = "binding site masked"
        
    def match_regions(self):
        
        if self.binding_site.binding_site.volume.region != self.confidence_map.region:
            self.confidence_map.region = self.confidence_map.full_region()
            
            self.confidence_map.region = self.binding_site.binding_site.volume.region
    
    def run(self):
        
        if self.confidence_map is not None:
            #self.match_regions()
            self.grid_data_densmap = self.confidence_map.matrix()
        else:
            self.grid_data_densmap = self.binding_site.map.matrix()
        
        
        self.origin, self.apix = self.binding_site.map.data_origin_and_step()
        
       
        self.grid_data_binding_site = self.binding_site.binding_site.volume.matrix() 
        self.calculate_binding_site_mask()
        self.add_map_to_session()
        return self.volume
    

    
class ConfidenceMap:
    def __init__(self, session, densmap):
        
        self.densmap = densmap
        self.session = session
        self.use_model = False
        self.radius_in_angstrom = 6.0
    
    
        
    def get_full_map(self):
        
        try:
            self.full_map = self.densmap.grid_data().full_data.matrix().copy()
            self.origin = self.densmap.grid_data().full_data.origin
        except AttributeError:
            self.full_map = self.densmap.grid_data().matrix().copy()
            self.origin = self.densmap.grid_data().origin
   
    
        self.apix = self.densmap.grid_data().step
        self.region = self.densmap.region
        
    
    def estimate_background_from_cubes(self):
        self.bg_mean, self.bg_std  = estimate_background_distribution(self.full_map, cube_size=10, num_cubes=4)
    
    def compute_confidence_map(self, cube_size=10, num_cubes=4, desired_ppv=0.99):
        """
        Compute a confidence map from the given density map.
        """
      
        pvals = compute_p_values(self.full_map, self.bg_mean, self.bg_std)
        qvals = benjamini_yekutieli(pvals)
        ppvs = 1 - qvals
        thresholded_map = (ppvs >= desired_ppv).astype(np.float32)
        masked_map = self.full_map.copy() 
        masked_map = masked_map * thresholded_map
        self.confidence_map =  masked_map
        #ConfidenceMapParameter('confidence_map', masked_map)
        
    def add_map_to_session(self):
        self.grid_data = ArrayGridData(self.confidence_map,
                                  self.origin, 
                                  self.apix)
        
        self.volume = volume_from_grid_data(self.grid_data, self.session, style="mesh")
        self.volume.name = "confidence map"
        self.volume.region = self.region
    
    
    def run(self):
        self.get_full_map()
        
        if self.use_model:
            pass
        else:
            self.estimate_background_from_cubes()
        
        self.compute_confidence_map()
        self.add_map_to_session()
        #ConfidenceMapParameter('confidence_map', self.volume)
        return ConfidenceMapParameter('confidence_map', self.volume)




class DifferenceMap:
    """ 
    use old sig fieat protocol
    """
    def __init__(self):
        pass