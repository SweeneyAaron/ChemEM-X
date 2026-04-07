#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:40:50 2025

@author: aaron.sweeney
"""

from . import view
import json 
from json.decoder import JSONDecodeError
from urllib.parse import parse_qs
#chimeraX imports
from chimerax.ui import HtmlToolInstance
from chimerax.core.models import ADD_MODELS, REMOVE_MODELS, MODEL_POSITION_CHANGED
from chimerax.core.tasks import ADD_TASK, REMOVE_TASK
#chememX imports
from chimerax.ChemEM.core.commands  import Command, UpdateModels, UpdateMaps
from chimerax.ChemEM.core.parameters import Parameters
from chimerax.ChemEM.core.tools import get_chemem_paths,  get_platforms, JobHandler
from chimerax.atomic.structure import AtomicStructure
from chimerax.ChemEM.simulate.tools import get_simulation


#need to import * from ligand.commands to register Command subclasses
from chimerax.ChemEM.ligand.commands import * 
from chimerax.ChemEM.binding_site.commands import * 
from chimerax.ChemEM.ligand.parameters import * 
from chimerax.ChemEM.binding_site.parameters import * 
from chimerax.ChemEM.map_masks.commands import * 
from chimerax.ChemEM.map_masks.parameters import * 
from chimerax.ChemEM.dock.commands import *
from chimerax.ChemEM.simulate.commands import *


class CHEMEM(HtmlToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = False  # No session saving for now
    CUSTOM_SCHEME = "chemem"
    display_name = "chemem" # HTML scheme for custom links
    PLACEMENT = None
    #help = "help:user/tools/chemem.html" #add this!!
    
    def __init__(self, session, tool_name):
        super().__init__(session, tool_name, size_hint=(600, 1000))
        self.parameters = Parameters()
        self.avalible_chemem_exes =  get_chemem_paths()
        self.avalible_platforms = get_platforms()
        #handelers
        self.add_model_handeler = self.session.triggers.add_handler(ADD_MODELS, self.update_models)
        self.remove_model_handeler = self.session.triggers.add_handler(REMOVE_MODELS, self.update_models)
        self.job_remove_task =  self.session.triggers.add_handler(REMOVE_TASK, self.remove_task)
        self.job_handeler = JobHandler()
        #protocol varables
        self.current_protonation_states=None
        self.autosites = None
        self.rendered_site = None
        self.dock_parameters = Parameters()
        self.dock_results = None
        self.simulation_parameters = Parameters()
        self.added_ligands = {}
        self.view = view.ChemEMView(self) #!!!
        self.view.render()
        session.metadata = self
    
    def run_js_code(self, js_code):
        self.html_view.page().runJavaScript(js_code)
    
    def remove_task(self, *args):
        job = args[1]
        if job.job_type == CHEMEM_JOB:
            if job.process.returncode == 0:
                js_code = f'updateJobStatus( {job.id}, "Completed");'
            else:
                js_code = f'updateJobStatus( {job.id}, "Failed");'
            
            self.run_js_code(js_code)
        
        if job.job_type == EXPORT_SIMULATION:
            self.current_simulation = get_simulation(self.session, 
                                                    self.simulation_parameters.get_parameter('build_params').get_parameter('output').value, 
                                                    self.platform,
                                                    self.parameters.get_parameter('current_model'),
                                                    parameters = self.simulation_parameters,
                                                    current_map=self.parameters.get_parameter('current_map'),
                                                    map_bias = None,
                                                    )
            
            self.current_simulation.setup_simulation()
            for js_code in self.current_simulation.js_code:
                self.run_js_code(js_code)
    
    def handle_scheme(self, url):
        
        command = url.path()
        query = self.extract_query(url)
        
        #print(Command.__subclasses__())
        
        for command_class in Command.__subclasses__():
            
            command_class().execute(self, command, query)
    
        print('COM:',command)
        print('QUERY', query)
        
    
    
    def extract_query(self, url):
        query = parse_qs(url.query())
        query = query['siteValue'][0]
        #print('unfilterd_query', query)
        
        try:
            query = json.loads(query)
        except JSONDecodeError:
            pass 

        if type(query) == dict:
            query = self.parameters.get_from_query(query)
            
        
        return query
    
    # functions for updating the state upon action
    def update_models(self, *args):
        
        """
        Handles adding and removing models.
        
        Parameters
        ----------
        *args : list
            - args[0] : str
                The function executed (e.g., 'remove models').
            - args[1] : list
                The list of models involved in the operation.
        """
        #TODO! refactor-this 
        
        current_map = self.parameters.get_parameter('current_map')
        current_model = self.parameters.get_parameter('current_model')
        #current_simulation_map = self.current_simulation.get_parameter('current_map')
        #current_simulation_model = self.current_simulation.get_parameter('current_model')
        #if self.current_simulation.get_parameter('AddedSolution') is not None:
        #    current_simulation_model = None #Don't need this if it was never set
        
        
        if args[0] == 'remove models':
            
            if current_model in args[1]:
                models_left = [i for i in self.session.models if isinstance(i, AtomicStructure) and i not in args[1]]
                self.clear_binding_site_tabs()
               
                if models_left:
                    #if a model still open set this as the current model 
                    UpdateModels.execute(self, 'UpdateModels', '_')
                    self.run_js_code( self.select_model_js( models_left[0] ))
                    return
                else:
                    #remove current model and cleanupTODO!         
                    del self.parameters.parameters['current_model']
                    return
                    
            
            if current_map in args[1]:
                self.clear_binding_site_tabs()
                #set current_map to None
                UpdateMaps.execute(self, 'UpdateMaps', '_')
                return
            
        
        
        if current_map is not None:
            current_map_key = UpdateMaps.get_map_id(current_map)
            js_code_maps = f'selectOptionByValue("maps", "{current_map_key}");'
        
        if current_model is not None:
            current_model_key = UpdateModels.get_model_id(current_model)
            js_code_model = f'selectOptionByValue("models", "{current_model_key}");'
        
        #Remove this in favor of simpler ux??
        '''
        if current_simulation_map is not None:
            current_simulation_map_key = UpdateSimulationMaps.get_map_id(current_simulation_map)
            js_code_simulated_map =  f'selectOptionByValue("SimulationMaps", "{current_simulation_map_key}");'
        
        if current_simulation_model is not None:
            current_simulation_model_key = UpdateModels.get_model_id(current_simulation_model)
            js_code_simulated_model =  f'selectOptionByValue("simulationModels", "{current_simulation_model_key}");'
        '''
        
        UpdateModels.execute(self, 'UpdateModels', '_')
        UpdateMaps.execute(self, 'UpdateMaps', '_')
        #UpdateSimulationMaps.execute(self, 'UpdateSimulationMaps', '_')
        #UpdateSimulationModels.execute(self, 'UpdateSimulationModels', '_')
        
        if current_map is not None:
            self.run_js_code(js_code_maps)
        
        if current_model is not None:
            self.run_js_code(js_code_model)
        
        '''
        if current_simulation_map is not None:
            self.run_js_code(js_code_simulated_map)
        
        if current_simulation_model is not None:
            self.run_js_code(js_code_simulated_model)
        '''
        
    
    
    
