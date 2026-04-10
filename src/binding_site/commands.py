#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:12:38 2025

@author: aaron.sweeney
"""

#binding_site_commands
import numpy as np 
from chimerax.ChemEM.core.tools import generate_unique_id
from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.binding_site.protocols import AutoSiteChimera, RenderBindingSite
from chimerax.ChemEM.binding_site.parameters import BindingSiteParameter
from chimerax.markers import MarkerSet

class SaveSite(Command):
    
    @classmethod 
    def js_code(cls, site, alt = 0):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        if alt == 0: 
            js_code = f'addBindingSiteEntry("{site_value}", "{site.name}");'
        else:
            js_code = f'updateBindingSiteEntry("{site_value}", "{site.name}");'
        
        return js_code

    @classmethod 
    def js_code_error(cls):
        return 'alert("ChemEM Error: Rendered Site in None.");'
    
    @classmethod 
    def run(cls, chemem, query):
        
        
        site = chemem.rendered_site 
        if site is None:
            chemem.run_js_code(cls.js_code_error())
            return 
        
        #choose whether to update selected site! 
        in_list = chemem.parameters.get_list_parameters('binding_sites', site.binding_site.name)
        
        if in_list:
            chemem.run_js_code(cls.js_code(site.binding_site, alt=1))
        else:
            chemem.parameters.add_list_parameter('binding_sites', site.binding_site)
            chemem.run_js_code(cls.js_code(site.binding_site))


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
        
        chemem.autosites = AutoSiteChimera.from_data(chemem.session,
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
                chemem.autosites = AutoSiteChimera.from_data(chemem.session,
                                                chemem.parameters.parameters['current_model'],
                                                chemem.parameters.get_parameter('current_map'))
            
            for centroid in marker_centroids:
                
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
            
class ResetBindingSiteFields(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.rendered_site is not None:
            chemem.rendered_site.reset() 
            chemem.render_site = None

class AddBindingSiteFromInputsOrChange(Command):
    
    @classmethod 
    def js_code(cls, site):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        js_code = f'addBindingSiteEntry("{site_value}", "{site.name}");'
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        #change or new!!
        if chemem.rendered_site is None:
            query.name = generate_unique_id()
            chemem.rendered_site =  RenderBindingSite(chemem.session, query, 
                                                      chemem.parameters.parameters['current_model'],
                                                      chemem.parameters.get_parameter('current_map'))

        else:
            
            site = chemem.rendered_site
            if query.centroid != site.binding_site.centroid :
                site.update_centroid(query.centroid)
                
            if query.box_size != site.binding_site.box_size :
                site.update_box_size(query.box_size)

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

class IncrementBindingSiteCounter(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.avalible_binding_sites += 1
        
class AddBindingSite(Command):
    
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.add_list_parameter('binding_sites', query)
        chemem.current_binding_site_id = query.name
#bindingsite101
class AddBindingSiteToConf(Command):
    
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.add_list_parameter('binding_sites_conf', query)
        chemem.current_binding_site_id = query.name        
#bindingsite101
class RemoveBindingSite(Command):
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.remove_list_parameter('binding_sites', query)
        #TODO! check the effects of this elsewhere
        if chemem.rendered_site is not None:
            if query == chemem.rendered_site.binding_site.name:
                chemem.rendered_site.reset() 
                chemem.rendered_site = None
#bindingsite101
class RemoveBindingSiteFromConf(Command):
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.remove_list_parameter('binding_sites_conf', query)

#bindingsite101
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
        
        
        