#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:25:46 2025

@author: aaron.sweeney
"""

from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.dock.tools import ChemEMSetUp, get_dock_results,  get_mmgbsa_scores, add_residue_to_model,  get_atom_match_object
from chimerax.ChemEM.core.tools import ChemEMJob, CHEMEM_JOB , get_output_from_conf
from chimerax.core.commands import run 
import json
import os
DOCK_PROTOCOLS = ["dock"]
DOCK_OPTIONS = []

class RunDocking(Command):
    @classmethod 
    def run(cls, chemem, query):
        backend = chemem.parameters.get_value("chememBackendPath")
        
        if backend is not None:
            
            options = [(k,v.value) for k,v in chemem.dock_parameters.parameters.items() if v.value != 'protocol'] 
            protocols = [k for k,v in chemem.dock_parameters.parameters.items() if v.value == "protocol"]
            
            print('\n\n\n',protocols,'\n\n\n')
           
            chemem_setup = ChemEMSetUp.from_parameters_object(chemem.session, 
                                       chemem.parameters, 
                                       backend,
                                       protocols,
                                       options)
            
            #run the job
            
            
            command = chemem_setup.run_command
            job = ChemEMJob(chemem.session,
                            command,
                            CHEMEM_JOB 
                            )
            
            job_data = {"id": job.id, "status": "running"}
            job_data_json = json.dumps(job_data)
            js_code = f"addJob({job_data_json});"
            chemem.run_js_code(js_code)
            job.start() 
            chemem.job_handeler.add_job(job)
            
            
class RemoveJob(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.job_handeler.remove_job(query)    
        
class AddDockParameter(Command):
    @classmethod 
    def run(cls, chemem, query):
        #if chemem.current_simualtion_id is not None:
        chemem.dock_parameters.add(query)

class LoadDockJob(Command):
    @classmethod 
    def js_code(cls,  results):
        payload = {
                "directories": list(results.directories),
                "results": [
                    [
                        {
                            "full_path": s.full_path,
                            "display_name": s.display_name,
                            "score": s.score,
                        }
                        for s in sols
                    ]
                    for sols in results.results
                ],
            }
        js_code = f"renderDockResults({json.dumps(payload)});"
        return js_code
    @classmethod 
    def run(cls, chemem, query):
        
        if query in chemem.job_handeler.jobs:
            job = chemem.job_handeler.jobs[query]
            com = job.command.split(' ')
            conf = com[1]
            options = [i for i in com if '--' in i]
            #no type helper here currently 
            conf_data = get_output_from_conf(conf)
            if conf_data is not None:
                output = conf_data['output']
                #get the most "advanced" analysis dock -> minimization -> aggregation
                if '--dock' in options:
                    results_dir = os.path.join(output,'docking')
                if '--minimize-docking' in options:
                    results_dir = os.path.join(output,'refined')
                if '--aggregate-sites' in options:
                    results_dir = os.path.join(output,'aggregate_sites')
                
                #MMGBSA scores are calculated on the most "advanced" analysis.
  
                #dock jobs are loaded based on the results.txt file
                #this means it will only load the defined solutions
                #if multiple runs have written results to the same dir
                results = get_dock_results(results_dir)
                
                if '--rescore' in options:
                    results_dir = os.path.join(output, 'mmgbsa_rescore')
                    results = get_mmgbsa_scores(results_dir)

                js_code = cls.js_code(results)
                chemem.run_js_code(js_code)
                chemem.dock_results = results
                #keep the results somewhere so the can be deleted appropriatly
                #TODO!
                    
class ViewDockSolution(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.dock_results is not None:
            if query.value in chemem.dock_results._gui:
                mdl =  chemem.dock_results._gui.get(query.value, None)
                if mdl is not None:
                    com = f'show {mdl.atomspec}'
                    run(chemem.session, com)
            else:
                
                com = f'open {query.value}'
                mdl = run(chemem.session, com)
                chemem.dock_results._gui[query.value] = mdl[0]

class HideDockSolution(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.dock_results is not None:
            mdl = chemem.dock_results._gui.get(query.value, None)
            if mdl is not None:
                com = f'hide {mdl.atomspec}'
                run(chemem.session, com)
       
#TODO!!! clear chemem.dock_results
class ClearDockSolutions(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.dock_results is not None:
            if chemem.dock_results._gui:
                for mdl in chemem.dock_results._gui:
                    com = f'close {mdl.atomspec}'
                    run(chemem.session, com)
                
            chemem.dock_results = None

class AddDockSolutionToStructure(Command):
    @classmethod
    def run(cls, chemem, query):
        if chemem.dock_results is not None:
            mdl = chemem.dock_results._gui.get(query.value, None)
            protein = chemem.parameters.get_parameter('current_model')
            if mdl is not None and protein is not None:
                res = add_residue_to_model(protein, mdl)
                
                atom_matched_res = get_atom_match_object(res, query.value)
                chemem.added_ligands[res] = atom_matched_res
                #need to assign the atom connections, here
                
                
                    
        
        
        
            
            
            
            
    
    