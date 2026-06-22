#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 12:30:50 2025

@author: aaron.sweeney
"""
from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.core.tools import condense_path, validate_ligand_file,  ChemEMJob,  EXPORT_SIMULATION, launch_chemem_job
from chimerax.ChemEM.simulate.tools import (SimulationJob, add_solution_to_structure_from_file,
                                            build_model_without_ligands, analyze_selected_atoms,
                                            mirror_selection_to_model, include_tracked_ligand_in_simulation,
                                            get_cation_pi_tug_indexes, get_saltbridge_tug_index,
                                            get_weak_hbond_tug_index, get_hbond_tug_index,
                                            get_halogen_bond_tug_index)
from chimerax.ChemEM.simulate.simulation import get_pipi_tug_indexes
from chimerax.ChemEM.mouse_modes import DragCoordinatesMode
from chimerax.ChemEM.core.parameters import SmilesParameter, PathParameter, Parameters
from chimerax.ChemEM.dock.tools import ChemEMSetUp
from chimerax.core.commands import run
from openmm import unit
from rdkit import Chem
import os
import tempfile


def _active_simulation_job(chemem):
    """Return the running SimulationJob and its SimulationContainer, or
    (None, None) if there is no active simulation. All interactive-restraint
    commands guard on this."""
    cs = getattr(chemem, 'current_simulation', None)
    if cs is None or cs.simulation_job_id is None:
        return None, None
    job = chemem.job_handeler.jobs.get(cs.simulation_job_id)
    if job is None:
        return None, None
    return job, cs


def _escape_js_string(s):
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')


def _trim_pdb_info(s):
    parts = s.split('.pdb ')
    return parts[1] if len(parts) > 1 else s

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

class RemoveSimulationLigand(Command):

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


class AddTrackedLigandToSimulation(Command):
    """Include an already-tracked SDF ligand (chosen from the Build-tab picker)
    in the simulation. query is the ligand_id string."""
    @classmethod
    def run(cls, chemem, query):
        include_tracked_ligand_in_simulation(chemem, query)
            
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
                #Guard: selected-residue mode is on but nothing is selected.
                #analyze_selected_atoms would otherwise IndexError on an empty
                #selection. Surface a message and abort cleanly, removing the
                #orphaned ligand-stripped copy we just added to the session.
                if len(protein.atoms.filter(protein.atoms.selected)) == 0:
                    msg = ('Selected-residue simulation is on but nothing is selected. '
                           'Select residues in the 3D view, or turn the option off.')
                    chemem.run_js_code(f'alert("{_escape_js_string(msg)}");')
                    chemem.session.models.remove([model_no_ligands])
                    return

                #create a simulation for only selcted residues
                #need to find anchor atoms and residues that are not selected
                #so the selected protein model doesn't drift during the simulation
                #relitve to non selected model,
                #the anchor residues/atoms are constrained to their original position during simulation
                selected_residues_info = analyze_selected_atoms(protein)
                #SimulationContainer.set_groups() reads this back when anchoring the
                #unselected region; without it set_anchor_residues raises KeyError.
                chemem.simulation_parameters.parameters['selected_atom_data'] = selected_residues_info
                non_selected_residues = selected_residues_info['res_atoms_not_selected']
                anchor_residues =  selected_residues_info['extra_residues']

                for atom_array in non_selected_residues.values():
                    for atom in atom_array:
                        atom.selected = True

                for residue in anchor_residues:
                    for atom in residue.atoms:
                        atom.selected = True

                #The selection above lives on `protein`; the exported model is
                #the ligand-stripped copy. Mirror it so `save ... selectedOnly
                #true` (ChemEMSetUp) captures the selected + anchor residues
                #instead of an empty PDB. Anchor data intentionally keeps
                #referencing `protein`, the model the simulation maps onto.
                mirror_selection_to_model(protein, model_no_ligands)

            
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
                                                                      ['export'],
                                                                      selected_atoms=selected_atoms)
                    
                    command = chemem_setup.run_command
                    
                    
                    launch_chemem_job(chemem, command, EXPORT_SIMULATION, "Export Simulation")
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
                                chemem = chemem,
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


#===========================================================================
# Interactive restraints / minimise / simulated annealing
#
# These drive a live simulation. Each reads the user's atom selection on the
# simulation model (via the helpers in simulate/tools.py), maps it to OpenMM
# indices, and stashes the result on the job; SimulationJob.run() applies it
# on its next loop. State lives on chemem.current_simulation (Simulation
# Container): .simulation_job_id, .simulation_model, .atoms_to_position_index
# [_as_dic], .current_restraints, .default_mouse_mode.
#===========================================================================

class EnableTugMode(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        cs.default_mouse_mode = chemem.session.ui.mouse_modes.mode(button='right')
        new_mode = DragCoordinatesMode(chemem.session, cs.atoms_to_position_index_as_dic)
        chemem.session.ui.mouse_modes.bind_mouse_mode(mouse_button="right", mode=new_mode)
        job.tug = new_mode


class DisableTugMode(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        job.tug = None
        chemem.session.ui.mouse_modes.bind_mouse_mode(mouse_button="right",
                                                      mode=cs.default_mouse_mode)
        cs.default_mouse_mode = None


class SetCationPiTTug(Command):

    @classmethod
    def js_code(cls, ring_atom, cation_atom, restraint_id):
        ring = _escape_js_string(_trim_pdb_info(ring_atom.residue.string()))
        cation = _escape_js_string(_trim_pdb_info(cation_atom.string()))
        message = f'Ring: {ring}\\nCation: {cation}'
        return f'addRestraintToList("{message}", "{restraint_id}", "CationPiTug");'

    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        ring_atoms, ring_atom_indices, cation_atom, cation_atom_index = \
            get_cation_pi_tug_indexes(cs.simulation_model, cs.atoms_to_position_index_as_dic)
        if ring_atom_indices is None:
            return
        job.cation_pi_tug = [ring_atom_indices, cation_atom_index, None, None, None]
        restraint_id = len(cs.current_restraints)
        cs.current_restraints[restraint_id] = [[ring_atom_indices, cation_atom_index],
                                               [ring_atoms[0], cation_atom]]
        chemem.run_js_code(cls.js_code(ring_atoms[0], cation_atom, restraint_id))
        run(chemem.session, f'distance {ring_atoms[0].atomspec} {cation_atom.atomspec}')


class SetPiPiTTug(Command):

    @classmethod
    def js_code(cls, ring_atoms, restraint_id):
        ring_1 = _escape_js_string(_trim_pdb_info(ring_atoms[0].residue.string()))
        ring_2 = _escape_js_string(_trim_pdb_info(ring_atoms[1].residue.string()))
        message = f'Ring 1: {ring_1}\\nRing 2: {ring_2}'
        return f'addRestraintToList("{message}", "{restraint_id}", "PiPiTug");'

    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        atoms_index_pairs = get_pipi_tug_indexes(cs.simulation_model,
                                                 cs.atoms_to_position_index_as_dic)
        if not atoms_index_pairs:
            return
        openff_index_pairs = [i[1] for i in atoms_index_pairs]
        # T-shaped: target angle pi/2 between ring normals
        job.pipi_p_tug = [openff_index_pairs, None, 1.5707963267948966, None, None, None, None]
        restraint_id = len(cs.current_restraints)
        ring_atoms = [i[0] for i in atoms_index_pairs]
        ring_atoms = [i[0] for i in ring_atoms]
        cs.current_restraints[restraint_id] = [openff_index_pairs, ring_atoms]
        chemem.run_js_code(cls.js_code(ring_atoms, restraint_id))
        run(chemem.session, f'distance {ring_atoms[0].atomspec} {ring_atoms[1].atomspec}')


class SetPiPiPTug(Command):

    @classmethod
    def js_code(cls, ring_atoms, restraint_id):
        ring_1 = _escape_js_string(_trim_pdb_info(ring_atoms[0].residue.string()))
        ring_2 = _escape_js_string(_trim_pdb_info(ring_atoms[1].residue.string()))
        message = f'Ring 1: {ring_1}\\nRing 2: {ring_2}'
        return f'addRestraintToList("{message}", "{restraint_id}", "PiPiTug");'

    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        atoms_index_pairs = get_pipi_tug_indexes(cs.simulation_model,
                                                 cs.atoms_to_position_index_as_dic)
        if not atoms_index_pairs:
            return
        openff_index_pairs = [i[1] for i in atoms_index_pairs]
        # parallel: target angle 0 between ring normals
        job.pipi_p_tug = [openff_index_pairs, None, None, None, None, None, None]
        restraint_id = len(cs.current_restraints)
        ring_atoms = [i[0] for i in atoms_index_pairs]
        ring_atoms = [i[0] for i in ring_atoms]
        cs.current_restraints[restraint_id] = [openff_index_pairs, ring_atoms]
        chemem.run_js_code(cls.js_code(ring_atoms, restraint_id))
        run(chemem.session, f'distance {ring_atoms[0].atomspec} {ring_atoms[1].atomspec}')


class SetSaltBridgeTug(Command):

    @classmethod
    def js_code(cls, selected_atoms, restraint_id):
        cation = _escape_js_string(_trim_pdb_info(selected_atoms[0].string()))
        anion = _escape_js_string(_trim_pdb_info(selected_atoms[1].string()))
        message = f'Salt Bridge: {cation} - {anion}'
        return f'addRestraintToList("{message}", "{restraint_id}", "SaltBridgeTug");'

    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        saltbridge_idx, selected_atoms = \
            get_saltbridge_tug_index(cs.simulation_model, cs.atoms_to_position_index_as_dic)
        if saltbridge_idx is None or None in selected_atoms:
            return
        job.saltbridge_tug = [saltbridge_idx, None]
        restraint_id = len(cs.current_restraints)
        chemem.run_js_code(cls.js_code(selected_atoms, restraint_id))
        run(chemem.session, f'distance {selected_atoms[0].atomspec} {selected_atoms[1].atomspec}')
        cs.current_restraints[restraint_id] = [saltbridge_idx, selected_atoms]


class SetWeakHBondTug(Command):

    @classmethod
    def js_code(cls, selected_atoms, restraint_id):
        donor = _escape_js_string(_trim_pdb_info(selected_atoms[0].string()))
        hydrogen = _escape_js_string(_trim_pdb_info(selected_atoms[1].string()))
        acceptor = _escape_js_string(_trim_pdb_info(selected_atoms[2].string()))
        message = f'Weak Hbond Donor: {donor}\\nHydrogen: {hydrogen}\\nAcceptor: {acceptor}'
        return f'addRestraintToList("{message}", "{restraint_id}", "WeakHbondTug");'

    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        weak_hbond_idx, selected_atoms = \
            get_weak_hbond_tug_index(cs.simulation_model, cs.atoms_to_position_index_as_dic)
        if weak_hbond_idx is None:
            return
        job.weak_hbond_tug = [weak_hbond_idx, None, None]
        restraint_id = len(cs.current_restraints)
        chemem.run_js_code(cls.js_code(selected_atoms, restraint_id))
        selected_atoms[1].selected = False
        run(chemem.session, 'distance sel')
        cs.current_restraints[restraint_id] = [weak_hbond_idx, selected_atoms]


class SetHBondTug(Command):

    @classmethod
    def js_code(cls, selected_atoms, restraint_id):
        donor = _escape_js_string(_trim_pdb_info(selected_atoms[0].string()))
        hydrogen = _escape_js_string(_trim_pdb_info(selected_atoms[1].string()))
        acceptor = _escape_js_string(_trim_pdb_info(selected_atoms[2].string()))
        message = f'Donor: {donor}\\nHydrogen: {hydrogen}\\nAcceptor: {acceptor}'
        return f'addRestraintToList("{message}", "{restraint_id}", "HbondTug");'

    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        # selected_atoms returned as Donor, Hydrogen, Acceptor
        hbond_idx, selected_atoms = \
            get_hbond_tug_index(cs.simulation_model, cs.atoms_to_position_index_as_dic)
        if hbond_idx is None:
            return
        job.hbond_tug = [hbond_idx, None, None]
        restraint_id = len(cs.current_restraints)
        chemem.run_js_code(cls.js_code(selected_atoms, restraint_id))
        selected_atoms[1].selected = False
        run(chemem.session, 'distance sel')
        cs.current_restraints[restraint_id] = [hbond_idx, selected_atoms]


class SetHalogenBondTug(Command):

    @classmethod
    def js_code(cls, selected_atoms, restraint_id):
        donor = _escape_js_string(_trim_pdb_info(selected_atoms[0].string()))
        acceptor = _escape_js_string(_trim_pdb_info(selected_atoms[1].string()))
        message = f'Halogen Bond Donor: {donor}\\nAcceptor: {acceptor}'
        return f'addRestraintToList("{message}", "{restraint_id}", "HalogenBondTug");'

    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        indices, selected_atoms, theta2_0, halogen_distance = \
            get_halogen_bond_tug_index(cs.simulation_model, cs.atoms_to_position_index_as_dic)
        if indices is None:
            return
        job.halogen_bond_tug = [indices, None, None, None,
                                halogen_distance * unit.angstrom, None, theta2_0 * unit.degrees]
        restraint_id = len(cs.current_restraints)
        chemem.run_js_code(cls.js_code(selected_atoms, restraint_id))
        run(chemem.session, 'distance sel')
        cs.current_restraints[restraint_id] = [indices, selected_atoms, theta2_0, halogen_distance]


class DeleteHalogenBondTug(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        indices, selected_atoms, theta2_0, halogen_distance = cs.current_restraints[int(query)]
        job.halogen_bond_tug = [indices, 0.0, 0.0, 0.0,
                                halogen_distance * unit.angstrom, None, theta2_0 * unit.degrees]
        run(chemem.session, f'distance delete {selected_atoms[0].atomspec} {selected_atoms[1].atomspec}')


class DeletePiPiTug(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        openff_ring_atoms, selected_atoms = cs.current_restraints[int(query)]
        run(chemem.session, f'distance delete {selected_atoms[0].atomspec} {selected_atoms[1].atomspec}')
        job.pipi_p_tug = [openff_ring_atoms, None, None, None, 0.0, 0.0, 0.0]
        del cs.current_restraints[int(query)]


class DeleteCationPiTug(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        indices, distance_atoms = cs.current_restraints[int(query)]
        job.cation_pi_tug = [indices[0], indices[1], 0.0, 0.0, 0.0]
        run(chemem.session, f'distance delete {distance_atoms[0].atomspec} {distance_atoms[1].atomspec}')
        del cs.current_restraints[int(query)]


class DeleteSaltBridgeTug(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        indices, selected_atoms = cs.current_restraints[int(query)]
        run(chemem.session, f'distance delete {selected_atoms[0].atomspec} {selected_atoms[1].atomspec}')
        job.saltbridge_tug = [indices, 0.0]
        del cs.current_restraints[int(query)]


class DeleteHbondTug(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        hbond_tug_idx, selected_atoms = cs.current_restraints[int(query)]
        run(chemem.session, f'distance delete {selected_atoms[0].atomspec} {selected_atoms[2].atomspec}')
        job.hbond_tug = [hbond_tug_idx, 0.0, 0.0]
        del cs.current_restraints[int(query)]


class DeleteWeakHbondTug(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        hbond_tug_idx, selected_atoms = cs.current_restraints[int(query)]
        run(chemem.session, f'distance delete {selected_atoms[0].atomspec} {selected_atoms[2].atomspec}')
        job.weak_hbond_tug = [hbond_tug_idx, 0.0, 0.0]
        del cs.current_restraints[int(query)]


class MinimiseSimulation(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        job.minimise = 1


class RunSimulatedAnneling(Command):
    @classmethod
    def run(cls, chemem, query):
        # query is a SimulatingAnnelingParameter; the job loop reads its
        # startTemp/normTemp/topTemp/tempStep/... attributes directly.
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        job.simulated_anneling = query


# =====================================================================
# Save points (Feature 1)
# =====================================================================
class SaveSimulationSnapshot(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        #the worker captures the positions + a timestamp on its next loop tick
        job.save_snapshot = 'Manual'


class RestoreSimulationSnapshot(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        try:
            job.restore_to = int(query)
        except (TypeError, ValueError):
            pass


# =====================================================================
# Torsion preferences (Feature 2)
# =====================================================================
class SyncSelectedTorsions(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        job.torsion_sync = True


class StepTorsionForward(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        job.torsion_step = (str(query), 1)


class StepTorsionBackward(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        job.torsion_step = (str(query), -1)


class ShowTorsionProfile(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        job.torsion_show = str(query)


class FlipTorsionSide(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        job.torsion_flip = str(query)


class ColorTorsionStrain(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        mode = str(query)
        job.torsion_color = mode if mode in ('local', 'global') else 'local'


class ClearTorsionStrainColor(Command):
    @classmethod
    def run(cls, chemem, query):
        job, cs = _active_simulation_job(chemem)
        if job is None:
            return
        job.torsion_color_clear = True


