#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:18:33 2025

@author: aaron.sweeney
"""
import numpy as np 
from chimerax.markers import MarkerSet
from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.binding_site.parameters import BindingSiteParameter
from chimerax.ChemEM.binding_site.tools import BindingSiteChemEM, RenderBindingSite
from chimerax.atomic import all_atoms

#TODO! change Rendering!!

class RenderBindingSiteFromClick(Command):
    
    @classmethod 
    def js_code(cls, site):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        js_code = f'populateFieldsFromBackendString("{site_value}");'
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        site = chemem.parameters.get_list_parameters('binding_sites', query)[0]
        if chemem.rendered_site is not None:
            chemem.rendered_site.reset()
        
        chemem.rendered_site =  RenderBindingSite(chemem.session, 
                                                  site, 
                                                  chemem.parameters.parameters['current_model'],
                                                  chemem.parameters.get_parameter('current_map'))
        chemem.run_js_code(cls.js_code(site))

class TransferSiteToConf(Command):
    
    @classmethod
    def js_code(cls, site):
        centroid_x = round(site.centroid[0], 3)
        centroid_y = round(site.centroid[1], 3)
        centroid_z = round(site.centroid[2], 3)
        box_x = int(site.box_size[0])
        box_y = int(site.box_size[1])
        box_z = int(site.box_size[2])
        
        return f"""
        document.getElementById("centroidX").value = {centroid_x};
        document.getElementById("centroidY").value = {centroid_y};
        document.getElementById("centroidZ").value = {centroid_z};
        document.getElementById("QuickboxSizeX").value = {box_x};
        document.getElementById("QuickboxSizeY").value = {box_y};
        document.getElementById("QuickboxSizeZ").value = {box_z};
        addCentroid();
        resetManualBindingSiteFields();
        """
    @classmethod 
    def run(cls, chemem, query):
        
        site = chemem.parameters.get_list_parameters('binding_sites', query)[0]
        #chemem.parameters.add_list_parameter('binding_sites_conf', site)
        chemem.run_js_code(cls.js_code(site))



class AutoSiteFinder(Command):

    @classmethod
    def js_code(cls, site):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        js_code = f'''
        addBindingSiteEntry("{site_value}", "{site.name}");
        populateFieldsFromBackendString("{site_value}");
        '''
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        
        chemem.autosites = BindingSiteChemEM.from_data(chemem.session,
                                        chemem.parameters.parameters['current_model'],
                                        chemem.parameters.get_parameter('current_map'))
        
        for site in chemem.autosites:
            if chemem.rendered_site is not None:
                chemem.rendered_site.reset()
            
            chemem.current_binding_site_id = site.name 
            #RenderAutoSite
            chemem.rendered_site = RenderBindingSite(chemem.session,
                                                         site,
                                                         chemem.parameters.parameters['current_model'],
                                                         chemem.parameters.get_parameter('current_map'))
                                                         
            
            js_code = cls.js_code(site)
            chemem.run_js_code(js_code)
            
            #don't think i need this?
            chemem.current_renderd_site_id = site.name
            
            chemem.parameters.add_list_parameter('binding_sites', site)

class BindingSiteFromMarker(Command):
    
    @classmethod
    def js_code(cls, site):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        js_code = f'''
        addBindingSiteEntry("{site_value}", "{site.name}");
        populateFieldsFromBackendString("{site_value}");
        '''
        return js_code
    
    @classmethod 
    def js_code_error(cls):
        message = "No Markers found.\\nPlace marker first to define binding site."
        return f'alert("{message}");'
    
    @classmethod 
    def run(cls, chemem, query):
        markers = []
        for model in chemem.session.models:
            if isinstance(model, MarkerSet):
                markers.append(model)
        
        if len(markers) == 0:
            chemem.run_js_code(cls.js_code_error())
        else:
            
            marker_centroids = [np.array(i.atoms[0].coord) for i in markers[0].residues]
            
            if chemem.autosites is None:
                chemem.autosites = BindingSiteChemEM.from_data(chemem.session,
                                                chemem.parameters.parameters['current_model'],
                                                chemem.parameters.get_parameter('current_map'))
            
            for centroid in marker_centroids:
                #TODO! pass back the ChemEM2Bindingsite object to make logic simpler
                site = chemem.autosites.site_from_centroid(centroid)
                
                if site is None:
                    #use manual binding site
                    site = BindingSiteParameter.get_from_centroid(centroid, chemem)
                    
                chemem.parameters.add_list_parameter('binding_sites', site)
                chemem.current_binding_site_id = site.name 
                
                chemem.rendered_site =  RenderBindingSite(chemem.session, site, 
                                                          chemem.parameters.parameters['current_model'],
                                                          chemem.parameters.get_parameter('current_map'))
                
                js_code = cls.js_code(site)
                chemem.run_js_code(js_code)
                
                #don't think i need this?
                chemem.current_renderd_site_id = site.name
                #set current working site in backend!!!
                
                
                
class RemoveBindingSite(Command):
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.remove_list_parameter('binding_sites', query)
        #TODO! check the effects of this elsewhere
        if chemem.rendered_site is not None:
            if query == chemem.rendered_site.binding_site.name:
                chemem.rendered_site.reset() 
                chemem.rendered_site = None
                
                
class SetEditBindingSiteValue(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.render_site_from_key(query)

class AssignSelectedAtomToIonSite(Command):
    @classmethod
    def js_code(cls, atom_spec):
        return f'addAtomSpecToIonFixer("{atom_spec}");'

    @classmethod
    def run(cls, chemem, query):
        
        current_model = chemem.parameters.get_parameter('current_model')
        if current_model is not None:
            atoms = current_model.atoms
            selected_atoms = atoms[atoms.selected]
            print(f'Ion setter query... {query}')

            if len(selected_atoms) != 1:
                print('bad atom selection!!')
                #TODO! return a js alert
                return

            # Use the first selected atom
            atom = selected_atoms[0]
            chain_id = atom.residue.chain.chain_id
            res_name = atom.residue.name
            res_num = atom.residue.number
            atom_name = atom.name

            # Format as a spec string compatible with the backend parser
            # e.g., /A:10@OD1
            spec_string = f"/{chain_id}:{res_num}@{atom_name}"
            print('------------------------------->',spec_string)
            
            chemem.run_js_code(cls.js_code(spec_string))

