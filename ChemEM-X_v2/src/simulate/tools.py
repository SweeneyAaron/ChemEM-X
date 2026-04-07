#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:04:21 2025

@author: aaron.sweeney
"""

from chimerax.ChemEM.dock.tools import add_residue_to_model, get_atom_match_object
from chimerax.ChemEM.core.tools import mol_from_sdf, validate_ligand_file
from chimerax.ChemEM.simulate.simulation import Simulation, SimulationContainer
from chimerax.core.commands import run 
from chimerax.core.tasks import  Task, TaskState
import datetime
import numpy as np
from chimerax.atomic import Atoms
import time

SIMULATION_JOB = 'chemem-X simulation'

def add_solution_to_structure_from_file(session, file, protein):
    if validate_ligand_file(file):
        #chimeraX model of ligand
        mdl = run(session, f'open {file}')
        if mdl:
            mdl = mdl[0]
            res = add_residue_to_model(protein, mdl)
            atom_matched_res = get_atom_match_object(res, file)
            session.models.remove([mdl])
            return res, atom_matched_res
        
    return None

def build_model_without_ligands(model):
    
    #this version allows anything other that the ligands added to be there 
    #need to ensure the chemem alows and or handels other macromolecues
    new_model = model.copy() 
    for residue in new_model.residues:
        if residue.name.startswith('LIG'):
            new_model.delete_residue(residue)
    
    return new_model
                
def analyze_selected_atoms(model):
    from chimerax.atomic import Atoms, Residues

    # Get all selected atoms in the model
    sel_atoms = model.atoms.filter(model.atoms.selected)

    # Get the unique residues that the selected atoms belong to
    sel_residues = sel_atoms.unique_residues

    # Find atoms in each residue that are not selected
    res_atoms_not_selected = {}
    for res in sel_residues:
        atoms_in_res = res.atoms
        not_selected_atoms = atoms_in_res.subtract(sel_atoms)
        res_atoms_not_selected[res] = not_selected_atoms

    # Sort residues by chain ID and residue number for sequential analysis
    sel_residues_sorted = sorted(sel_residues, key=lambda r: (r.chain_id, r.number))

    # Identify continuous segments
    segments = []
    segment_start = sel_residues_sorted[0]
    previous_res = segment_start

    for res in sel_residues_sorted[1:]:
        current_res = res

        # Check if residues are sequential and bonded
        sequential = (previous_res.chain_id == current_res.chain_id and
                      previous_res.number + 1 == current_res.number)

        bonded = False
        if sequential:
            c_atom = previous_res.find_atom('C')
            n_atom = current_res.find_atom('N')
            if c_atom and n_atom:
                bonded = any(bond.other_atom(c_atom) == n_atom for bond in c_atom.bonds)

        if not sequential or not bonded:
            # End the current segment
            segment_end = previous_res
            segments.append((segment_start, segment_end))
            # Start a new segment
            segment_start = current_res

        previous_res = current_res

    # Add the last segment
    segment_end = previous_res
    segments.append((segment_start, segment_end))

    # Collect discontinuities and extra residues
    discontinuities = []
    extra_residues = set()

    for i, segment in enumerate(segments):
        segment_start, segment_end = segment

        # Find residues before and after the segment
        # Previous residue to segment_start
        prev_res_num = segment_start.number - 1
        prev_res = get_residue(model, segment_start.chain_id, prev_res_num)
        if prev_res:
            c_atom = prev_res.find_atom('C')
            n_atom = segment_start.find_atom('N')
            if c_atom and n_atom and any(bond.other_atom(c_atom) == n_atom for bond in c_atom.bonds):
                extra_residues.add(prev_res)

        # Next residue to segment_end
        next_res_num = segment_end.number + 1
        next_res = get_residue(model, segment_end.chain_id, next_res_num)
        if next_res:
            c_atom = segment_end.find_atom('C')
            n_atom = next_res.find_atom('N')
            if c_atom and n_atom and any(bond.other_atom(c_atom) == n_atom for bond in c_atom.bonds):
                extra_residues.add(next_res)

        # Add discontinuity between this segment and the next
        if i < len(segments) - 1:
            next_segment_start = segments[i + 1][0]
            discontinuities.append((segment_end, next_segment_start))

    return {
        'selected_atoms': sel_atoms,
        'selected_residues': sel_residues,
        'res_atoms_not_selected': res_atoms_not_selected,
        'discontinuities': discontinuities,
        'extra_residues': extra_residues
    }

def get_residue(model, chain_id, res_num):
    from chimerax.atomic import Residues

    residues = model.residues.filter(
        (model.residues.chain_ids == chain_id) & (model.residues.numbers == res_num)
    )
    return residues[0] if len(residues) > 0 else None



def get_simulation(session, 
                    exported_files, 
                    platform,
                    chimerax_model,
                    parameters = None,
                    current_map=None,
                    map_bias = None, #do need to take this into account
                    ):
    
    
    
    simulation = Simulation.from_filepath(session, 
                                          exported_files, 
                                          current_map, 
                                          parameters = parameters,
                                          platform_name = platform) 
    
    temp = parameters.get_parameter('temperature')
    if temp is not None:
       
        simulation.temperature = temp.value
    
    simulation_model, atoms_to_position_index = simulation.get_model_from_complex_structure(chimerax_model)
    
    return SimulationContainer(simulation, 
                        simulation_model, 
                        atoms_to_position_index ,
                        parameters, 
                        current_map)

#self.chemem.simulation_model
#self.chemem.atoms_to_position_index_as_dic
#self.chemem.simulation
#self.chemem.update_simulation_model
#self.update_simulation_model_vectorised

class SimulationJob(Task):
    def __init__(self, session, 
                 simulation_model,
                 atoms_to_position_index,
                 simulation,
                 #need to update the model somehow!!
                 job_type = SIMULATION_JOB):
        super().__init__(session)
        self.simulation_model = simulation_model
        self.atoms_to_position_index = atoms_to_position_index
        self.atoms_to_position_index_as_dic = {a: i for a, i in atoms_to_position_index}
        self.simulation = simulation 
    
        self.job_type = job_type
        self.running = True
        self._pause = False
        self.started = False
        #set these to set a new tempreture
        self._set_temp = None
        self._set_map_bias = None
        
        self.tug = None
        self.hbond_tug = None
        self.weak_hbond_tug = None
        self.pipi_p_tug = None
        self.cation_pi_tug = None
        self.saltbridge_tug = None
        self.halogen_bond_tug = None
        self.simulated_anneling = None
        self.minimise = None
        #[15879, np.array([128.76779874, 130.55397987, 173.85587692])]
        self.step_size = 5
        
    def terminate(self):
        """Terminate this task.

        This method should be overridden to clean up
        task data structures.  This base method should be
        called as the last step of task deletion.

        """
        self.session.tasks.remove(self)
        self.end_time = datetime.datetime.now()
        
        if self._terminate is not None:
            self._terminate.set()
            
        self.state = TaskState.TERMINATING
    
   
    def run(self, *args, **kw):
        
        self.started = True
        #check that the chimerax atom positions match the simulation atom positions!
        self.simulation.chimera_x_atom_to_simulation_consistancy(self.simulation_model, 
                                                                 self.atoms_to_position_index_as_dic)
        
        print('--------------------------------------------------->>')
        print('pre min')
        print(np.array(self.simulation.get_positions())[0])
        print('--------------------------------------------------->>')
        
        self.simulation.minimise_system()
        
        print('--------------------------------------------------->>')
        print('post min')
        print(np.array(self.simulation.get_positions())[0])
        print('--------------------------------------------------->>')
        
        self.update_simulation_model()
        self.step_count = 0
        while self.running :
            #For debugging!!
            
            if self._pause:
                time.sleep(0.2)
                continue
            
            
            self.step_count += self.step_size
            
            self.simulation.step(self.step_size)
            #self.chemem.update_simulation_model()
            self.update_simulation_model_vectorised()
            
            #can update temp while paused
            if self._set_temp is not None:
                self.simulation.set_tempreture(self._set_temp)
                self.update_simulation_model()
                self._set_temp = None
            
            if self._set_map_bias is not None:
                self.simulation.set_map_bias(self._set_map_bias)
                self.update_simulation_model()
                self._set_map_bias = None
            #Tug---------------------------
            if self.tug is not None:
                
                if self.tug.atom_idx is not None:
                    
                    self.simulation.update_tug_force_for_atom(self.tug.atom_idx, self.tug.end_coord)
                    for num in range(20):
                        
                        self.simulation.step(self.step_size)
                        self.update_simulation_model()
                    self.simulation.update_tug_force_for_atom(self.tug.atom_idx, self.tug.end_coord, tug_k = 0.0)
                    self.tug.atom_idx = None
                        #set the tug stuff back to None
            
            #-----HBondTug--------------
            if self.hbond_tug is not None:
                
                self.simulation.update_hbond_tug_force_for_atom(self.hbond_tug[0],
                                                                       hbond_dist_k = self.hbond_tug[1],
                                                                       hbond_angle_k = self.hbond_tug[2])
                for num in range(20):
                    #run for a while incase the simulation is paused
                    self.simulation.step(self.step_size)
                    self.update_simulation_model()
                self.hbond_tug = None
            
            if self.weak_hbond_tug is not None:
                
                
                self.simulation.update_weak_hbond_tug_force_for_atom(self.weak_hbond_tug[0],
                                                                            distance_k = self.weak_hbond_tug[1],
                                                                            angle_k = self.weak_hbond_tug[2])
                
                self.weak_hbond_tug = None
            #-----PiPi-Tug--------------
            if self.pipi_p_tug is not None:
                self.simulation.update_pipi_p_tug(self.pipi_p_tug[0],
                                                    r0 = self.pipi_p_tug[1],
                                                    theta0 = self.pipi_p_tug[2],
                                                    offset0 = self.pipi_p_tug[3],
                                                    k_distance = self.pipi_p_tug[4],
                                                    k_angle = self.pipi_p_tug[5],
                                                    k_offset = self.pipi_p_tug[6])
                
                
                
                for num in range(20):
                    #run for a while incase the simulation is paused
                    self.simulation.step(self.step_size)
                    self.update_simulation_model()
                self.pipi_p_tug = None
            
            #-----Pi-Cation-Tug--------------
            if self.cation_pi_tug is not None:
                self.simulation.update_cation_pi_tug(self.cation_pi_tug[0],
                                                     self.cation_pi_tug[1],
                                                     self.cation_pi_tug[2],
                                                     self.cation_pi_tug[3],
                                                     self.cation_pi_tug[4]
                                                     )
                self.cation_pi_tug = None
                
            if self.saltbridge_tug is not None:
                self.simulation.update_saltbridge_tug_force_for_atom(self.saltbridge_tug[0], distance_k=self.saltbridge_tug[1])
                self.saltbridge_tug = None
            
            if self.halogen_bond_tug is not None:
                self.simulation.update_halogen_bond_tug_force_for_atom( self.halogen_bond_tug[0], 
                                                           distance_k = self.halogen_bond_tug[1], 
                                                           theta1_k = self.halogen_bond_tug[2],
                                                           theta2_k = self.halogen_bond_tug[3],
                                                           distance = self.halogen_bond_tug[4],
                                                           theta1 = self.halogen_bond_tug[5],
                                                           theta2 = self.halogen_bond_tug[6])
                self.halogen_bond_tug = None
            
            #-----minmise
            if self.minimise is not None:
                self.simulation.minimise_system()
                self.update_simulation_model()
                self.minimise = None
            
            #----simulated Anneling------
            if self.simulated_anneling is not None:

                
                #initial heating of the system!!
                for temp in range(self.simulated_anneling.startTemp, self.simulated_anneling.normTemp, self.simulated_anneling.tempStep):
                    
                    self.simulation._set_temp(temp)
                    print('TEMP:', temp)
                    for _ in range(0, self.simulated_anneling.initialHeatingInterval, self.step_size):
                        
                        self.simulation.step(self.step_size)
                        self.update_simulation_model() 
                    
                #simulation cycles!!
                for _ in range(self.simulated_anneling.simAnnCycles): #add this!!
                    
                    print('increase temp...')
                    #increase temp to top step
                    for temp in range(self.simulated_anneling.normTemp, self.simulated_anneling.topTemp, self.simulated_anneling.tempStep):#add this 
                        
                        self.simulation.set_tempreture(temp)
                        #time at each tempreture !
                        for _ in range(0, self.simulated_anneling.equilibriumTime, self.step_size):
                            
                            self.simulation.step(self.step_size)
                            self.update_simulation_model() 
                    
                    print('Hold...')
                    #hold temp at top step
                    for _ in range(0, self.simulated_anneling.holdTopTempInterval, self.step_size):
                        self.simulation.step(self.step_size)
                        self.update_simulation_model()
                    
                    print('decrese_temp...')
                    #decrease temp
                    for temp in range(self.simulated_anneling.topTemp, self.simulated_anneling.normTemp, self.simulated_anneling.tempStep):
                        self.simulation.set_tempreture(temp)
                        
                        for _ in range(0, self.simulated_anneling.equilibriumTime, self.step_size):
                            
                            self.simulation.step(self.step_size)
                            self.update_simulation_model() 
                    
                    #minimisation 
                    print('minimising')
                    if self.simulated_anneling.localMinimisation:
                        self.simulation.minimise_system()
                        
                    
                
                self.simulated_anneling = None
                

            #if self.step_count == 250:
            #    print('HBOND TUG TEST TIME')
            #    self.chemem.simulation.update_hbond_tug_force_for_atom(self.hbond_tug)

    

    def update_simulation_model_vectorised(self):
        
    
        # Obtain positions as a NumPy array directly from OpenMM
        positions = np.array(self.simulation.get_positions_as_numpy())
        
        
        #positions = np.array(self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True))
    
        # Extract atoms and corresponding indices
        atoms, indices = zip(*self.atoms_to_position_index)
        
        # Convert atoms to a ChimeraX Atoms object
        
        atoms = Atoms(atoms)
    
        # Update the coordinates in bulk
        atoms.coords = positions[np.array(indices)]
    
        

    
    
    def update_simulation_model(self):
        
        #t1 = time.perf_counter()
        positions = np.array(self.simulation.get_positions())
        for atom, index in self.atoms_to_position_index:
            atom.coord = positions[index]
    
    def on_finish(self):
        """Callback method executed after task thread terminates.

        This callback is executed in the UI thread after the
        :py:meth:`run` method returns.  By default, it does nothing.

        """
        self.terminate()
       

    