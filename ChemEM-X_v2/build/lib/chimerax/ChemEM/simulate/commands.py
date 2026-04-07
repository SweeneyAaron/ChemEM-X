#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 12:30:50 2025

@author: aaron.sweeney
"""
from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.core.tools import condense_path, validate_ligand_file,  ChemEMJob,  EXPORT_SIMULATION
from chimerax.ChemEM.simulate.tools import SimulationJob, add_solution_to_structure_from_file, build_model_without_ligands, analyze_selected_atoms
from chimerax.ChemEM.core.parameters import SmilesParameter, PathParameter, Parameters
from chimerax.ChemEM.dock.tools import ChemEMSetUp
from rdkit import Chem
import os 
import tempfile

class AddSimulationParameter(Command):
    @classmethod 
    def run(cls, chemem, query):
        #if chemem.current_simualtion_id is not None:
        chemem.simulation_parameters.add(query)


#add smiles to the ligand list for simulation,
#smiles should be handeled differntly to files and the other thing, shif to com so use can manipulate manually

class AddSimulationLigandSmiles(Command):
    @classmethod 
    def js_code(cls, smiles):
        js_code = f"alert(Invalid Ligand SMILES: {smiles});"
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        valid_smiles = cls.validate_smiles(query.value)
        if valid_smiles:
            chemem.simulation_parameters.add_list_parameter('Ligands', query)
        else:
            js_code = cls.js_code(query.value)
            chemem.run_js_code(js_code)
    
    @staticmethod 
    def validate_smiles(smiles):
        smiles = Chem.MolFromSmiles(smiles)
        if smiles is not None:
            return True
        else:
            return False

def RemoveSimulationLigand(Command):
    
    @classmethod
    def js_code(cls, smiles):
        js_code = "alert(see TODO);"
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        chemem.simulation_parameters.remove_list_parameter('Ligands', query)
        #TODO! i need to check here if the ligand is in an activate simulation and remove it from there too

class AddLigandFileToSimulation(Command):
    
    @classmethod
    def js_code(cls, file):
        js_code = f"alert(Invalid Ligand File: {file});"
        return js_code

    @classmethod 
    def run(cls, chemem, query):
        #TODO! valid_smiles = cls.validate_smiles(query.value)
        valid_file = validate_ligand_file(query.value)
        #valid_file = True
        if valid_file:
            chemem.simulation_parameters.add_list_parameter('Ligands', query)
        
        else:
            js_code = cls.js_code(query.value)
            chemem.run_js_code(js_code)
            
class AddSimulationConstraintParameter(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.simulation_parameters.add_list_parameter('InitialConstraints', query)
        

class BuildSimulation(Command):
    
    @classmethod
    def js_code(cls, alert):
        js_code = f"alert({alert});"
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        chemem_executable  = chemem.parameters.get_parameter('chememBackendPath')
        if chemem_executable  is not None:
            
            #add any new ligands 
            new_ligands = chemem.simulation_parameters.get_parameter('Ligands')
            protein = chemem.parameters.get_parameter('current_model')
            if new_ligands is not None:
                for lig in new_ligands:
                    if isinstance(lig, SmilesParameter):
                        pass
                        #process like smiles!! 
                    elif isinstance(lig, PathParameter):
                        res, atom_matched_res = add_solution_to_structure_from_file(chemem.session, 
                                                                                    lig.value, 
                                                                                    protein)
                        chemem.added_ligands[res] = atom_matched_res
                        
            #collect file data for ligands and a clean protein model
            
            added_ligand_paths = [PathParameter('ligand',i.file) for i in chemem.added_ligands.values()]
            model_no_ligands = build_model_without_ligands(protein)
            chemem.session.models.add([model_no_ligands])
            
            use_selected = chemem.simulation_parameters.get_parameter('simulateSelectedAtoms')
            
            if use_selected.value:
                #create a simulation for only selcted residues
                #need to find anchor atoms and residues that are not selected 
                #so the selected protein model doesn't drift during the simulation
                #relitve to non selected model,
                #the anchor residues/atoms are constrained to their original position during simulation
                selected_residues_info = analyze_selected_atoms(protein)
                non_selected_residues = selected_residues_info['res_atoms_not_selected']
                anchor_residues =  selected_residues_info['extra_residues']
                
                for atom_array in non_selected_residues.values():
                    for atom in atom_array:
                        atom.selected = True 
                
                for residue in anchor_residues:
                    for atom in residue.atoms:
                        atom.selected = True
            
            
            output = chemem.parameters.get_parameter('output')
            if output is not None:
                
                build_parameters = Parameters() 
                build_parameters.add(output)
                build_parameters.parameters['current_model'] = model_no_ligands
                
                for lig in added_ligand_paths:
                    build_parameters.add_list_parameter('Ligands', lig)
                
                selected_atoms = chemem.simulation_parameters.get_parameter('simulateSelectedAtoms')
                if selected_atoms is not None:
                    selected_atoms= selected_atoms.value
                
                #pass it forward
                chemem.simulation_parameters.parameters['build_params'] = build_parameters
                backend = chemem.parameters.get_value("chememBackendPath")
                if backend is not None:
                    chemem_setup = ChemEMSetUp.from_parameters_object(chemem.session,
                                                                      build_parameters,
                                                                      backend,
                                                                      ['export-simulation'],
                                                                      selected_atoms=selected_atoms)
                    
                    command = chemem_setup.run_command
                    
                    
                    job = ChemEMJob(chemem.session,
                                    command,
                                    EXPORT_SIMULATION
                                    )
                    
                    
                    job.start() 
                    chemem.job_handeler.add_job(job)
                    chemem.session.models.remove([model_no_ligands])
                    
                    
                    
                else:
                    js_code = cls.js_code('Please set the ChemEM backend in the set-up tab.')
                    chemem.run_js_code(js_code)
            else:
                js_code = cls.js_code('Please set the output directory in the set-up tab.')
                chemem.run_js_code(js_code)



class RunSimulation(Command):
    
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simulation.simulation_job_id is None:
            job = SimulationJob(chemem.session, 
                                chemem.current_simulation.simulation_model,
                                chemem.current_simulation.atoms_to_position_index,
                                chemem.current_simulation.simulation,
                                )
            job.start()
            chemem.current_simulation.simulation_job_id = job.id
            chemem.job_handeler.add_job(job)
        else:
            simulation_id = chemem.current_simulation.simulation_job_id
            job = chemem.job_handeler.jobs[simulation_id]
            job._pause = False 
            


class PauseSimulation(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simulation.simulation_job_id is not None:
            simulation_id = chemem.current_simulation.simulation_job_id
            job = chemem.job_handeler.jobs[simulation_id]
            job._pause = True
            
class StopSimulation(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simulation.simulation_job_id is not None:
            simulation_id = chemem.current_simulation.simulation_job_id
            job = chemem.job_handeler.jobs[simulation_id]
            job.running = False
            

class DeleteConstrainProtein(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simulation.simulation_job_id is not None:
            chemem.current_simulation.simulation.simulation.context.setParameter('backbone_constraint_k', 0)
            chemem.current_simulation.simulation.simulation.context.setParameter('sidechain_constraint_k', 0)

class DeleteConstrainBackBone(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simulation.simulation_job_id is not None:
            chemem.current_simulation.simulation.simulation.context.setParameter('backbone_constraint_k', 0)

class DeleteConstrainSidechain(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simulation.simulation_job_id is not None:
            chemem.current_simulation.simulation.simulation.context.setParameter('sidechain_constraint_k', 0)


class DeleteLigandConstraint(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simulation.simulation_job_id is not None:
            chemem.current_simulation.simulation.simulation.context.setParameter('ligand_constraint_k', 0)

class SetSimulationMapBias(Command):
    @classmethod 
    def run(cls,chemem, query):
        if chemem.current_simulation.simulation_job_id is not None:
            simulation_id = chemem.current_simulation.simulation_job_id
            job = chemem.job_handeler.jobs[simulation_id]
            job._set_map_bias = query.value

class SetSimulationTempreture(Command):
    @classmethod 
    def run(cls,chemem, query):
        if chemem.current_simulation.simulation_job_id is not None:
            simulation_id = chemem.current_simulation.simulation_job_id
            job = chemem.job_handeler.jobs[simulation_id]
            job._set_temp = query.value


