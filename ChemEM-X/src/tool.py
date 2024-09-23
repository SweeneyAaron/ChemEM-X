# This file is part of the ChemEM-X software.
#
# Copyright (c) 2024 - Aaron Sweeney
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>



from . import view
import os
import subprocess
import json
import numpy as np
import copy 
from json.decoder import JSONDecodeError
from urllib.parse import parse_qs
from rdkit import Chem
import datetime
import copy 
import uuid
import threading
import time
import tempfile
import shutil
from itertools import combinations
from scipy.spatial.distance import euclidean

from openmm import app

from chimerax.ChemEM.chemem_job import  JobHandler, SimulationJob,LocalChemEMJob, SIMULATION_JOB, CHEMEM_JOB, EXPORT_SIMULATION, EXPORT_LIGAND_SIMULATION, EXPORT_COVELENT_LIGAND_SIMULATION
from chimerax.ChemEM.map_masks import  SignificantFeaturesProtocol
from chimerax.ChemEM.rdtools import Protonate, smiles_is_valid, ChemEMResult, SolutionMap, RW_mol_from_smiles,  draw_molecule_with_atom_indices, RD_PROTEIN_SMILES, save_image_temporarily, remove_temporary_file, combine_molecules_and_react, get_mol_from_output, RD_PROTEIN_HYDROGENS, hydrogen_mapping,  translate_and_rotate_molecule  
from chimerax.ChemEM.simulation import Simulation, get_complex_structure, get_complex_system,  MapBias, TugForce, HbondDistForce, HbondAngleForce, PiPiDistForce, get_position_vector_from_atom, get_model_from_complex_structure, SSE_force, SSERigidBodyBBConstraint, HelixHbondForce, PsiAngleForce, PhiAnglelForce, AnchorAtoms, get_mmpbsa_complex_system
from chimerax.ChemEM.covelent_binding import map_atoms, remove_atoms, get_bonded_atoms
from chimerax.ChemEM.mouse_modes import  DragCoordinatesMode, PickPoint
from chimerax.ChemEM.config import Config
from chimerax.mouse_modes import MouseMode
from chimerax.markers import MarkerSet
from chimerax.ui import HtmlToolInstance
from chimerax.atomic.structure import AtomicStructure
from chimerax.core.models import ADD_MODELS, REMOVE_MODELS, MODEL_POSITION_CHANGED
from chimerax.core.tasks import ADD_TASK, REMOVE_TASK
from chimerax.map import Volume
from chimerax import open_command
from chimerax.core.commands import run 
from chimerax.mouse_modes.std_modes import MovePickedModelsMouseMode
from chimerax.markers.mouse import MoveMarkersPointMouseMode

from rdkit.Chem.rdchem import BondType
from openmm import XmlSerializer
from openmm import LangevinIntegrator, Platform
from openmm import app
from openmm import unit
from openmm import MonteCarloBarostat, XmlSerializer, app, unit, CustomCompoundBondForce, Continuous3DFunction, vec3, Vec3
from scipy.ndimage import gaussian_filter
from openmm.unit.quantity import Quantity
import parmed

PLACE_LIGAND = 'place_ligand'

class CHEMEM(HtmlToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = False  # No session saving for now
    CUSTOM_SCHEME = "chemem"
    display_name = "chemem" # HTML scheme for custom links
    PLACEMENT = None
    #help = "help:user/tools/pickluster.html" #add this!!
    
    def __init__(self, session, tool_name):
        super().__init__(session, tool_name, size_hint=(600, 1000))
        self.parameters = Parameters()
        
        self.view = view.ChemEMView(self) #!!!
        
        self.add_model_handeler = self.session.triggers.add_handler(ADD_MODELS, self.update_models)
        self.remove_model_handeler = self.session.triggers.add_handler(REMOVE_MODELS, self.update_models)
        self.moved_model_handeler = self.session.triggers.add_handler(MODEL_POSITION_CHANGED, self.model_position_changed)
        #self.job_added_task = self.session.triggers.add_handler(ADD_TASK, self.add_task)
        self.job_remove_task =  self.session.triggers.add_handler(REMOVE_TASK, self.remove_task)
        
        self.job_handeler = JobHandler()
        self.view.render()
        self.current_renderd_site_id = None
        self.rendered_site = None
        self.current_protonation_states = None
        self.current_significant_features_object = None
        self.current_loaded_result = None
        self.current_simulation = Parameters()
        self._handler_references = [self.add_model_handeler, 
                                     self.remove_model_handeler, 
                                     self.moved_model_handeler,
                                     self.job_remove_task,
                                     ]
        
        self.avalible_chemem_exes =  get_chemem_paths()
        self.avalible_platforms = self.get_platforms()
        self.platforms_set = False
        self.temp_build_dir = None
        self.current_restraints = {}
        
        self.covelent_ligand = None
        self.mmgbsa = Parameters()
        #remove!!
        self.avalible_binding_sites = 0
        
        
        #enable for debugging
        #TODO!
        self.session.metadata = self
    

        
        
        
    def get_platforms(self):
        import chimerax 
        openmm_plugins_dir = os.path.join(chimerax.app_lib_dir, 'plugins')
        Platform.loadPluginsFromDirectory(openmm_plugins_dir)
        avalible_platforms = []
        for i in range(Platform.getNumPlatforms()):
            avalible_platforms.append(Platform.getPlatform(i).getName())
        return avalible_platforms
        
    def remove_task(self, *args):
        
        job = args[1]
        if job.job_type == CHEMEM_JOB:
            
            if job.success:
                js_code = f'updateJobStatus( {job.id}, "Completed");'
            else:
                js_code = f'updateJobStatus( {job.id}, "Failed");'
            self.run_js_code(js_code)
            
        elif job.job_type == EXPORT_SIMULATION:
            #self.openmm_test_run(job)
            
            if 'current_map' in self.current_simulation.parameters:
                current_map = self.current_simulation.parameters['current_map']
            else:
                current_map = None
            
            
            if 'AddedSolution' in self.current_simulation.parameters:
                chimerax_model = self.current_simulation.parameters['AddedSolution'].pdb_object
            
            else:
                #For now assumes protein only.
                chimerax_model = self.current_simulation.parameters['current_model']
            
            selected_atoms = self.current_simulation.get_parameter('selected_atoms')
            
           # simulation = Simulation(self.session, job.params.output, current_map, platform_name = self.platform)
            
            
            simulation = Simulation.from_filepath(self.session, 
                                                  job.params.output, 
                                                  current_map, 
                                                  parameters = self.current_simulation,
                                                  platform_name = self.platform) 
            #
            self.simulation = simulation
            
            simulation_model, atoms_to_position_index = simulation.get_model_from_complex_structure(chimerax_model)
            self.atoms_to_position_index = atoms_to_position_index
            self.atoms_to_position_index_as_dic = {i[0]:i[1] for i in atoms_to_position_index}
            self.simulation_model = simulation_model
            
            
            
            simulation.set_ring_groups(simulation_model, self.atoms_to_position_index_as_dic)
            simulation.set_SSE_elements(simulation_model, self.atoms_to_position_index_as_dic)
            if selected_atoms:
                simulation.set_anchor_residues(simulation_model, 
                                               self.current_simulation.parameters['selected_atom_data'],
                                               self.atoms_to_position_index_as_dic)
            
            
            
            
            simulation.set_force_groups()
            if current_map is not None:
                simulation.add_force(MapBias)
            
            
            if selected_atoms:
                simulation.add_force(AnchorAtoms)
                
            #simulation.add_force(SSE_force)
            
            #simulation.add_force(HelixHbondForce)
            #simulation.add_force(PsiAngleForce)
            #simulation.add_force(PhiAnglelForce)
            
            #simulation.add_constraint(SSERigidBodyBBConstraint)
            simulation.add_force(TugForce)
            simulation.add_force(HbondDistForce)
            simulation.add_force(HbondAngleForce)
            simulation.add_force(PiPiDistForce)
            
            simulation.set_simulation()

            #self.session.models.add([simulation_model])
            self.current_simualtion_id = None
            
            js_code = 'openTab(event, "RunSimulationTab");'
            self.run_js_code(js_code)
            self.remove_temp_directories()
        
        
        
        elif job.job_type == EXPORT_LIGAND_SIMULATION:
 
            if 'current_map' in self.current_simulation.parameters:
                current_map = self.current_simulation.parameters['current_map']
            else:
                current_map = None
            
            

            complex_ligand_structure = get_complex_structure(job.params.output)
            
            #Add residues?? 
            self.simulation_model, ligand_residue, atoms_to_position_index_as_dic = add_openff_ligand_to_chimerax_model(complex_ligand_structure, self.simulation_model)
            
            #get protein com 
            all_atom_coords = []
            for residue in self.simulation_model.residues:
                for atom in residue.atoms:
                    all_atom_coords.append(atom.coord)
            
            all_atom_coords = np.array(all_atom_coords)
            com = np.mean(all_atom_coords, axis=0)
            
            ligand_atom_coords = np.array([i.coord for i in ligand_residue.atoms])
            ligand_com = np.mean(ligand_atom_coords, axis=0)
            translation_vector =  com - ligand_com
            
            translated_coords = ligand_atom_coords + translation_vector
            for atom, crd in zip(ligand_residue.atoms, translated_coords):
                atom.coord = crd
            
            
            
            #atom = ligand_residue.atoms[0]
            print('SIM:', self.simulation, type(self.simulation.complex_structure))
            print('STRUCT:', self.simulation.complex_structure, type(self.simulation.complex_structure)) 
            
            complex_structure = self.simulation.complex_structure + complex_ligand_structure
            print('CMP:', complex_structure)
            
            #TODO!!! if solvent!!! 
            #move the atoms now!!!
            simulation_model, atoms_to_position_index = get_model_from_complex_structure(complex_structure, self.simulation_model)
            self.atoms_to_position_index = atoms_to_position_index
            self.atoms_to_position_index_as_dic = {i[0]:i[1] for i in atoms_to_position_index}
            
            coordinates = complex_structure.coordinates.copy()
            
            for crd, atom in zip(translated_coords, ligand_residue.atoms):
                atom_index = self.atoms_to_position_index_as_dic[atom]
                coordinates[atom_index] = np.array(atom.coord)
            
            
            complex_structure.coordinates = coordinates
            
            complex_system = get_complex_system(complex_structure, self.current_simulation)
            
            del self.simulation #free up memeory?
            
            simulation = Simulation(self.session, 
                                    complex_system, 
                                    complex_structure,
                                    current_map,
                                    platform_name = self.platform)
            
            
            
            self.simulation = simulation
            #-----
            simulation.set_ring_groups(simulation_model, self.atoms_to_position_index_as_dic)
            simulation.set_SSE_elements(simulation_model, self.atoms_to_position_index_as_dic)
            simulation.set_force_groups()
            if current_map is not None:
                simulation.add_force(MapBias)
            #simulation.add_force(SSE_force)
            #simulation.add_constraint(SSERigidBodyBBConstraint)
            #simulation.add_force(HelixHbondForce)
            #simulation.add_force(PsiAngleForce)
            #simulation.add_force(PhiAnglelForce)
            simulation.add_force(TugForce)
            simulation.add_force(HbondDistForce)
            simulation.add_force(HbondAngleForce)
            simulation.add_force(PiPiDistForce)
            simulation.set_simulation()
            #-----
            
            
            
            self.current_simualtion_id = None
            
            js_code = 'openTab(event, "RunSimulationTab");'
            
            self.run_js_code(js_code)
            
            self.remove_temp_directories()
            
            print('ENDED...')
            
        elif job.job_type == EXPORT_COVELENT_LIGAND_SIMULATION:

            print('@covenlent_ligand_structure')
            
            if 'current_map' in self.current_simulation.parameters:
                current_map = self.current_simulation.parameters['current_map']
            else:
                current_map = None
            
            selected_atoms = self.current_simulation.get_parameter('selected_atoms')
            
            # Retrieve molecules and structures
            map_mol = get_mol_from_output(job.params.output)
            input_mol = self.covelent_ligand.parameters['combined_rwmol']
            complex_ligand_structure = get_complex_structure(job.params.output)
            
            # Get removed protein atoms and residue indices
            removed_protein_atoms = self.covelent_ligand.get_parameter('protein_remove_atom_idx')
            protein_atom_names_to_idx = self.covelent_ligand.get_parameter('residue_idx')
            
            if removed_protein_atoms is not None:
                
                # TODO: Needs to be updated if ligand atoms have been removed
                #need to remove protein atoms from simulation, chimerax model, 
                #protein_atom_names_to_idx 
                
                
                pass
            
            # Adjust protein atom indices
            num_ligand_atoms = self.covelent_ligand.parameters['ligand_rw_mol'].GetNumHeavyAtoms()
            protein_atom_names_to_idx = {name: idx + num_ligand_atoms for name, idx in protein_atom_names_to_idx.items()}
            
            # Get current simulation complex system
            current_simulation_complex_system = self.simulation.complex_structure
            bind_protein_atom = self.covelent_ligand.get_parameter('protein_atom')
            bind_protein_atom_idx = self.atoms_to_position_index_as_dic[bind_protein_atom]
            current_simulation_atom = current_simulation_complex_system.atoms[bind_protein_atom_idx]
            
            # Map atoms between molecules
            atom_match = map_mol.GetSubstructMatch(input_mol)
            protein_atoms_conversion = {name: atom_match[idx] for name, idx in protein_atom_names_to_idx.items()}
            
            
            
            current_simulation_bonded_atoms, current_simulation_bonded_atoms_hydrogen = get_bonded_atoms(
                current_simulation_atom, protein_atoms_conversion
            )
            
            # Get bonded atoms in complex ligand structure
            cov_struct_idx = protein_atoms_conversion[bind_protein_atom.name]
            ligand_atom = complex_ligand_structure.atoms[cov_struct_idx]
            protein_keys = set(protein_atoms_conversion.values())
            current_ligand_bonded_atoms = [
                bond.atom1 if bond.atom1 != ligand_atom else bond.atom2
                for bond in ligand_atom.bonds
                if (bond.atom1.idx if bond.atom1 != ligand_atom else bond.atom2.idx) not in protein_keys
            ]
            
            # Deep copy structures
            complex_structure_copy = copy.deepcopy(current_simulation_complex_system)
            complex_structure_copy._ncopies = current_simulation_complex_system._ncopies
            
            # Update hydrogen atoms in copy
            current_simulation_bonded_atoms_hydrogen = [
                complex_structure_copy.atoms[atom.idx] for atom in current_simulation_bonded_atoms_hydrogen
            ]
            
           
            # Remove hydrogens from complex structure copy
            remove_atoms(complex_structure_copy, current_simulation_bonded_atoms_hydrogen)
            
            # Deep copy ligand structure
            complex_ligand_structure_copy = copy.deepcopy(complex_ligand_structure)
            complex_ligand_structure_copy._ncopies = complex_ligand_structure._ncopies
            
            # Identify protein atoms to remove from ligand structure
            heavy_atom_idxs = list(protein_atoms_conversion.values())
            hydrogen_idxs = []
            for idx in heavy_atom_idxs:
                atom = complex_ligand_structure_copy.atoms[idx]
                for bond in atom.bonds:
                    other_atom = bond.atom1 if bond.atom1 != atom else bond.atom2
                    if other_atom.element_name == 'H':
                        hydrogen_idxs.append(other_atom.idx)
            
            prot_atoms_to_remove = sorted(heavy_atom_idxs + hydrogen_idxs, reverse=True)
            prot_atoms_to_remove_atoms = [complex_ligand_structure_copy.atoms[idx] for idx in prot_atoms_to_remove]
            
            # Remove protein atoms from ligand structure copy
            remove_atoms(complex_ligand_structure_copy, prot_atoms_to_remove_atoms)
            
            # Remake parameters
            complex_structure_copy.remake_parm()
            complex_ligand_structure_copy.remake_parm()
            
            # Combine structures
            new_complex_structure = complex_structure_copy + complex_ligand_structure_copy
            
            
            # Create index-to-atom mapping
            index_to_atom_dic = {idx: name for name, idx in protein_atoms_conversion.items()}
            hydrogen_map, hydrogen_map_to_atom_name = hydrogen_mapping(
                Chem.AddHs(map_mol), protein_atoms_conversion, bind_protein_atom.residue.name
            )
            hydrogen_map_idx = {idx: name for name, idx in hydrogen_map.items()}
            
            # Get binding atoms
            prot_atom = new_complex_structure.atoms[bind_protein_atom_idx]
            old_prot_atom = complex_ligand_structure.atoms[protein_atoms_conversion[bind_protein_atom.name]]
            
            old_ligand_atom_idx = self.covelent_ligand.parameters['ligand_atom_bound_index'].value
            old_ligand_atom = complex_ligand_structure.atoms[atom_match[old_ligand_atom_idx]]
            ligand_atom = next(
                atom for atom in new_complex_structure.residues[-1].atoms if atom.name == old_ligand_atom.name
            )

            # Add bonds between protein and ligand
            for bond in complex_ligand_structure.bonds:
                if old_prot_atom in bond and old_ligand_atom in bond:
                    new_bond = parmed.topologyobjects.Bond(prot_atom, ligand_atom, bond.type)
                    new_complex_structure.bonds.append(new_bond)
            
            # Function to map atoms in an angle or dihedral

            # Add angles between protein and ligand
            for angle in complex_ligand_structure.angles:
                if old_prot_atom in angle and old_ligand_atom in angle:
                    mapped_atoms = map_atoms(
                        angle,
                        {'prot_atom': old_prot_atom, 'ligand_atom': old_ligand_atom},
                        {'prot_atom': prot_atom, 'ligand_atom': ligand_atom},
                        index_to_atom_dic,
                        hydrogen_map_idx,
                        new_complex_structure
                    )
                    new_angle = parmed.topologyobjects.Angle(*mapped_atoms, type=angle.type)
                    new_complex_structure.angles.append(new_angle)
            
            # Add dihedrals between protein and ligand
            for dihedral in complex_ligand_structure.dihedrals:
                if old_prot_atom in dihedral and old_ligand_atom in dihedral:
                    mapped_atoms = map_atoms(
                        dihedral,
                        {'prot_atom': old_prot_atom, 'ligand_atom': old_ligand_atom},
                        {'prot_atom': prot_atom, 'ligand_atom': ligand_atom},
                        index_to_atom_dic,
                        hydrogen_map_idx,
                        new_complex_structure
                    )
                   
                    new_dihedral = parmed.topologyobjects.Dihedral(*mapped_atoms, type=dihedral.type)
                    new_complex_structure.dihedrals.append(new_dihedral)
            
            # Remake parameters
            new_complex_structure.remake_parm()
            
            # Adjust coordinates
            coords = complex_ligand_structure.coordinates
            index_1 = old_prot_atom.idx
            target_1_name = index_to_atom_dic[index_1]
            target_1 = next(
                (np.array([atom.xx, atom.xy, atom.xz]) for atom in prot_atom.residue.atoms if atom.name == target_1_name),
                None
            )
            
            # Find second target atom
            atom_target_2 = next(
                (bond.atom1 if bond.atom1 != old_prot_atom and bond.atom1.element_name != 'H' and bond.atom1 != old_ligand_atom else bond.atom2)
                for bond in old_prot_atom.bonds
                if (bond.atom1 != old_prot_atom and bond.atom1.element_name != 'H' and bond.atom1 != old_ligand_atom) or
                   (bond.atom2 != old_prot_atom and bond.atom2.element_name != 'H' and bond.atom2 != old_ligand_atom)
            )
            index_2 = atom_target_2.idx
            target_2_name = index_to_atom_dic[index_2]
            target_2 = next(
                (np.array([atom.xx, atom.xy, atom.xz]) for atom in prot_atom.residue.atoms if atom.name == target_2_name),
                None
            )
            
            self.target_info = [target_1_name, index_1, target_1, target_2_name, index_2, target_2]
            
            new_coords = translate_and_rotate_molecule(coords, index_1, target_1, index_2, target_2)
            
            # Map new coordinates to complex structure
            new_ligand_residue_map = {
                i.idx: j.idx
                for i in old_ligand_atom.residue.atoms
                for j in ligand_atom.residue.atoms
                if i.name == j.name
            }
            
            coordinates = new_complex_structure.coordinates.copy()
            for old_idx, new_idx in new_ligand_residue_map.items():
                coordinates[new_idx] = new_coords[old_idx]
            
            new_complex_structure.coordinates = coordinates
            
            # Build simulation model
            a2p = []
            for atom, old_idx in self.atoms_to_position_index:
                old_atom_residue = self.simulation.complex_structure.atoms[old_idx].residue
                new_residue = new_complex_structure.residues[old_atom_residue.idx]
                
                for new_atom in new_residue.atoms:
                    if new_atom.name == atom.name:
                        a2p.append((atom, new_atom.idx))
            
            
                
            
            
            simulation_model, ligand_residue, self.atoms_to_position_index_as_dic = add_openff_covelent_ligand_to_chimerax_model(
                new_complex_structure,
                self.simulation_model,
                atoms_to_position_index_as_dic={i[0]: i[1] for i in a2p}
            )
            
            self.atoms_to_position_index = list(self.atoms_to_position_index_as_dic.items())
            
            # Get complex system
            complex_system = get_complex_system(new_complex_structure, self.current_simulation)

            # Clean up old simulation
            del self.simulation  # Free up memory
            
            # Set up new simulation
            simulation = Simulation(
                self.session,
                complex_system,
                new_complex_structure,
                current_map,
                platform_name=self.platform
            )
            
            self.simulation = simulation
            
            # Set up simulation forces
            simulation.set_ring_groups(simulation_model, self.atoms_to_position_index_as_dic)
            simulation.set_SSE_elements(simulation_model, self.atoms_to_position_index_as_dic)
            simulation.set_force_groups()
            if current_map is not None:
                simulation.add_force(MapBias)
            
            
            print('selected_atoms')
            print(selected_atoms)
            if selected_atoms:
                
                
                simulation.set_anchor_residues(simulation_model, 
                                               self.current_simulation.parameters['selected_atom_data'],
                                               self.atoms_to_position_index_as_dic)
                
                simulation.add_force(AnchorAtoms)
            #simulation.add_force(HelixHbondForce)
            #simulation.add_force(PsiAngleForce)
            #simulation.add_force(PhiAnglelForce)
            simulation.add_force(TugForce)
            simulation.add_force(HbondDistForce)
            simulation.add_force(HbondAngleForce)
            simulation.add_force(PiPiDistForce)
            simulation.set_simulation()
            
            self.simulation_model = simulation_model
            self.current_simualtion_id = None
            
            # Execute JavaScript code
            js_code = 'openTab(event, "RunSimulationTab");'
            self.run_js_code(js_code)
            
            #TODO!! Add back in
            self.remove_temp_directories()

            
            
            print('ENDED')
            
            
            #Adding bonds...
            #Adding angles...
            #Adding dihedrals...
            #Adding Ryckaert-Bellemans torsions...
            #Adding Urey-Bradleys...
            #Adding improper torsions...
            #Adding CMAP torsions...
            #Adding trigonal angle terms...
            #Adding out-of-plane bends...
            #Adding pi-torsions...
            #Adding stretch-bends...
            #Adding torsion-torsions...
            #Adding Nonbonded force...
            
            
            #angles! 
            #dihedrals!
            #impropers!
            #remake!
            #chimera_x_residue!
            #delete older simulations¡
            #simulate!
            
            
            #identify the ligand atoms and any atoms on the protein that have been changed.
            #identify the protein atoms in simulation and if any of them have been changed/deleted in the new structure 
            
            #remove any protein atoms that need to be removed.
            #add in the new atoms to protein.
            #add in new residue to protein. 
            #transfer the bond, angel,dihedral stuff
            #add
            
            
            
        
            
            
                
            
           
            
    def remove_temp_directories(self):
            
            if self.temp_build_dir is not None:
                try:
                    
                    shutil.rmtree(self.temp_build_dir)
                    self.temp_build_dir = None
                except Exception as e:
                    print("Unable To remove directory {self.temp_build_dir}, Error: {e}")
    
    def update_simulation_model(self):
        
        t1 = time.perf_counter()
        positions = np.array(self.simulation.get_positions())
        for atom, index in self.atoms_to_position_index:
            atom.coord = positions[index]
        t2 = time.perf_counter() - t1
        
    
    
    def add_task(self, *args):
        js_code = 'alert("JobAdded");'
        self.run_js_code(js_code)


    def update_positions(self,positions, atom_to_position_idx):
        for atom, idx in atom_to_position_idx:
            atom.coord = np.array([positions[idx].x, positions[idx].y, positions[idx].z])
    
    def run_simulation(self):
        job = SimulationJob(self.session, self)
        job.start()  
        self.current_simualtion_id = job.id
        self.job_handeler.add_job(job)
        
        
    
    def run(self, executable, job_type):
        conf_file_path = self.conf_handler.file_path 
        command = f"{executable} {conf_file_path}"
        job = LocalChemEMJob(self.session, command, self.conf_handler , job_type)
        job_data = {"id": job.id, "status": "running"}
        
        if job_type == CHEMEM_JOB:
            job_data_json = json.dumps(job_data)
            js_code = f"addJob({job_data_json});"
            self.run_js_code(js_code)
            job.start()    
            js_code = "resetQuickConfigTab();"
            self.run_js_code(js_code)
            self.job_handeler.add_job(job) 
        
        elif job_type == EXPORT_SIMULATION:
            job.start()
            self.job_handeler.add_job(job) 
            
        elif job_type == EXPORT_LIGAND_SIMULATION:
            job.start()
            self.job_handeler.add_job(job) 
        
        elif job_type == EXPORT_COVELENT_LIGAND_SIMULATION:
            job.start()
            self.job_handeler.add_job(job) 
            

    
    def build_simulation(self):
        #create a tempory directory to write files to 
        #run the export_simulation blocking?
        chemem_path = self.parameters.parameters['chememBackendPath'].value 
        
    
    def delete(self):
        for handler in self._handler_references:
            handler.remove() 
        
        self._handler_references.clear() 
        #remove tempory file create by protonation
        CleanProtonationImage().run(self, None)
        self.remove_temp_directories()
        self.remove_temp_files()
        super().delete()
    
    
    def remove_temp_files(self):
        if self.covelent_ligand is not None:
            temp_lig_path = self.covelent_ligand.get_parameter('temp_ligand_image')
            if temp_lig_path is not None:
                remove_temporary_file(temp_lig_path.name)
            temp_lig_path = self.covelent_ligand.get_parameter('temp_residue_image')
            if temp_lig_path is not None:
                remove_temporary_file(temp_lig_path.name)
            
            
            
            
            
    def mkdir(self, path):
        try:
            os.mkdir(path)
        except Exception as e:
            pass
    
    def make_conf_file(self, parameter_object):
        '''
        Make a configuration file object and save it to the state. 
        Only one configuration file is made at a time, run ChemEM will
        find the configuration file at self.conf_handeler
        

        Parameters
        ----------
        parameter_object : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        if 'output' in parameter_object.parameters:
            model_path = os.path.join(parameter_object.parameters['output'].value, 'Inputs')
        else:
            model_path = os.path.join('./', 'Inputs')
        
        self.mkdir(model_path)
        parameter_object.parameters['model_path'] = model_path
        conf_handler = Config(parameter_object.copy(), self.session)
        conf_handler.run()
        self.conf_handler = conf_handler
    
    def make_conf_file_old(self):
        
        if 'output' in self.parameters.parameters:
            model_path = os.path.join(self.parameters.parameters['output'].value, 'Inputs')
            
        else:
            model_path = os.path.join('./', 'Inputs')
        
        self.mkdir(model_path)
        self.parameters.parameters['model_path'] = model_path
        conf_handler = Config(self.parameters.copy(), self.session)
        conf_handler.run()
        self.conf_handler = conf_handler
 
        
    def run_js_code(self, js_code):
        self.html_view.page().runJavaScript(js_code)
    
    
    def select_model_js(self, model):
        model_key = UpdateModels.get_model_id(model)
        return  f'selectOptionByValue("models", "{model_key}");'
        
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
        current_simulation_map = self.current_simulation.get_parameter('current_map')
        current_simulation_model = self.current_simulation.get_parameter('current_model')
        if self.current_simulation.get_parameter('AddedSolution') is not None:
            current_simulation_model = None #Don't need this if it was never set
        
        
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
        
        
        if current_simulation_map is not None:
            current_simulation_map_key = UpdateSimulationMaps.get_map_id(current_simulation_map)
            js_code_simulated_map =  f'selectOptionByValue("SimulationMaps", "{current_simulation_map_key}");'
        
        if current_simulation_model is not None:
            current_simulation_model_key = UpdateModels.get_model_id(current_simulation_model)
            js_code_simulated_model =  f'selectOptionByValue("simulationModels", "{current_simulation_model_key}");'
            
        
        UpdateModels.execute(self, 'UpdateModels', '_')
        UpdateMaps.execute(self, 'UpdateMaps', '_')
        UpdateSimulationMaps.execute(self, 'UpdateSimulationMaps', '_')
        UpdateSimulationModels.execute(self, 'UpdateSimulationModels', '_')
        
        if current_map is not None:
            self.run_js_code(js_code_maps)
        
        if current_model is not None:
            self.run_js_code(js_code_model)
            
        if current_simulation_map is not None:
            self.run_js_code(js_code_simulated_map)
        
        if current_simulation_model is not None:
            self.run_js_code(js_code_simulated_model)
        
        
        
    def model_position_changed(self, *args):
        
        
        pass
        #if not self.job_handeler.simulation_job_running():
        #    print('POSITIONS_CHANGED:', args)
        #else:
        #    pass
        
        
        
    def set_marker_position(self, position):
        self.session.metadata = self
        
        if not self.marker_wrapper in self.session.models:
            
            self.session.models.add([self.marker_wrapper])
        
        self.marker_wrapper.residues[0].atoms[0].coord = position
        #HERERER!!!
    
    def render_site_from_key(self, key):
        if self.rendered_site is not None:
            self.rendered_site.reset()
            self.rendered_site = None 
        
        site = self.parameters.get_list_parameters('binding_sites', key)[0]
        self.rendered_site =  RenderBindingSite(self.session, site, 
                           self.parameters.parameters['current_model'],
                           self.parameters.get_parameter('current_map'))
        
        
    def handle_scheme(self, url):
        print('')
        command = url.path()
        query = self.extract_query(url)
        
        for command_class in Command.__subclasses__():
            command_class().execute(self, command, query)
    
        print('COM:',command)
        print('QUERY', query)
        try:
            print('name', query.name)
            print('value', query.value)
        except:
            pass

    def extract_query(self, url):
        query = parse_qs(url.query())
        query = query['siteValue'][0]
        print('unfilterd_query', query)
        
        try:
            query = json.loads(query)
        except JSONDecodeError:
            pass 

        if type(query) == dict:
            query = self.parameters.get_from_query(query)
            
        
        return query
    
    def clear_binding_site_tabs(self):
        self.parameters.clear_binding_site_tabs(self)
    
    


#only need one marker initially not visable, the ChemEM marker.

#Move this around with each site is simpler than marker per thing.

#╔═════════════════════════════════════════════════════════════════════════════╗
#║                                 Parameters                                  ║
#╚═════════════════════════════════════════════════════════════════════════════╝


class Parameter:
    @classmethod
    def get_all_subclasses(cls):
        """
        Recursively find all subclasses of the current class.
        """
        subclasses = set(cls.__subclasses__())
        for subclass in cls.__subclasses__():
            subclasses.update(subclass.get_all_subclasses())
        return subclasses
    
     
    def chemem_string(self):
        return f"{self.name} = {self.value}\n"


class NumericParameter(Parameter):
    _type = str  

    def __init__(self, name, value):
        super().__init__()
        self.name = name
        self.value = self._type(value)

    @classmethod
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            name = query['id']
            value = query['value']
            return cls(name, value)
    
    
    

class FloatParameter(NumericParameter):
    _type = float
    

class IntParameter(NumericParameter):
    _type = int  


class StringParameter(Parameter):
    _type = str  
    def __init__(self, name,  default ):
        self.name = name 
        self.value = default 
   
    @classmethod 
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            
            return cls(query['id'],
                       query['value'])
   
        
class PathParameter(StringParameter):
    def __init__(self, name, default):
        super().__init__(name, default)

class SmilesParameter(StringParameter):
    def __init__(self, name, default):
        super().__init__(name, default)


class BooleanParameter(Parameter):
    _type = bool
    def __init__(self, name, default):
        self.name = name 
        self.value = default 
    
    @classmethod 
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            value = cls.get_value(query['value'])
            return cls(query['id'],
                       query['value'])
    @staticmethod 
    def get_value(value):
        if value == 'false':
            value = 0
        if value == 'true':
            value = 1

class ModelParameter(Parameter):
    def __init__(self, name, default):
        self.name = name
        self.value = default 
        
    @classmethod
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            value = cls.get_value(query['value'])
            return cls(query['id'],
                       value)

    @staticmethod 
    def get_value(value):
        value = value.split('-')[0].replace(' ', '')
        value = value.split('.')
        value = tuple([int(i) for i in value])
        print('NORMAL MODEL PARAMTER')
        print(value)
        return value

class MapParameter(ModelParameter):
    def __init__(self, name, default):
        super().__init__(name, default)
    
    @classmethod
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            
            if query['id'] == 'None':
                return None
            
            value = cls.get_value(query['value'])
            return cls(query['id'],
                       value)


class BindingSiteParameter(Parameter):
    def __init__(self, name, value):
        self.name = name 
        self.value = value 
    
    @classmethod
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            value = cls.get_value(query['value'])
            c =  cls(query['id'],
                       value)
            c.centroid = c.value[0]
            c.box_size = c.value[1]
            return c
    
    @classmethod 
    def get_from_centroid(cls, centroid ,chemem):
        
        binding_site_id = generate_unique_id()
        centroid = tuple(round(i,3) for i in centroid)
        box_size = (20,20,20) #default
        value = [centroid, box_size]
        c = cls(binding_site_id,
                value)
        c.centroid = centroid
        c.box_size = box_size 
        
        return c
    
    @classmethod
    def get_value(cls, value):
        values = value.split('|')
        centroid = BindingSiteParameter.convert_to_float_tuple(values[0])
        box_size = BindingSiteParameter.convert_to_float_tuple(values[1])
        return [centroid, box_size]
    
    @staticmethod
    def convert_to_float_tuple(s):
        stripped = s.strip()[1:-1]
        split_strings = stripped.split(',')
        return tuple(float(item) for item in split_strings)
        
#get the bindinsite paramter and the currentbindingsite id

class ProtonationParameter(Parameter):
    def __init__(self, name, value):
        self.name = name 
        self.value = value 
    
    @classmethod
    def get_from_query(cls, query):
       
        if query['class'] == cls.__name__:
            value = cls.get_value(query['value'])
            c =  cls(query['id'],
                       value)
            c.max_pH = value[0]
            c.min_pH = value[1]
            c.pka_prec = value[2]
            c.smiles = query['id']
            return c
            
    @classmethod 
    def get_value(cls, value):
        
        values = value.split('|')
        #pHmax, pHmin, pka_precsion
        value = [float(i) for i in values]
        return value


class SimulatingAnnelingParameter(Parameter):
    
    def __init__(self, name, value):
        self.name =name 
        self.value = value 
        self.name_tags = ['simAnnCycles', 'startTemp', 'normTemp', 'topTemp',
                          'tempStep', 'initialHeatingInterval', 
                          'holdTopTempInterval', 'equilibriumTime', 'localMinimisation']
    
    @classmethod 
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            name = query['id']
            value = cls.get_value(query['value'])
            c = cls(name, value)
            for n,v in zip(c.name_tags, value):
                setattr(c, n, v)
            
            return c
       
    @classmethod 
    def get_value(cls, value):
        value = value.replace('[', '')
        value = value.replace(']', '')
        value = value.split(',')
        return [int(i) for i in value]
  
            
class ImplicitSolventParameter(Parameter):
    def __init__(self, name, value):
        self.name =name 
        self.value = value 
    
    @classmethod 
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            name = query['id']
            value = cls.get_value(query['value'])
            return cls(name, value)
            
    
    @classmethod 
    def get_value(cls, value):
        if value  == "None":
            return None 
        else:
            return getattr(app, value)

class RDBondParameter(Parameter):
    def __init__(self, name, value):
        self.name = name 
        self.value = value
     
    @classmethod 
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            name = query['id']
            value = cls.get_value(query['value'])
            return cls(name, value)
    
    @classmethod 
    def get_value(cls, value):
        
        value = value.split(',')
        value = [int(i) for i in value]
        values = [IntParameter('startIdx', value[0]),
                  IntParameter('endIdx', value[1]),
                  BondTypeParameter('Bond',  BondType.values[value[2]])
                  ]
        
        return values
    
    def get_start_idx(self):
        return self.values[0]
    
    def get_end_idx(self):
        return self.values[1] 
    
    def get_bond(self):
        return self.values[2]
    
    def as_tuple(self):
        return (self.get_start_idx().value,
                self.get_end_idx().value,
                self.get_bond().value)
    

class BondTypeParameter(Parameter):
    def __init__(self, name, value):
        self.name = name 
        self.value = value 
        
    
    @classmethod 
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            name = query['id']
            value = cls.get_value(query['value'])
            return cls(name, value)
    
    @classmethod 
    def get_value(cls, value):
        return BondType.values[int(value)]
            

class Parameters:
    def __init__(self):
        self.parameters = {}

    def add(self, param):
        self.parameters[param.name] = param
        return param
    
    def add_list_parameter(self, list_name, param):
        if list_name in self.parameters:
            self.parameters[list_name].append(param)
        else:
            self.parameters[list_name] = [param]
    
    def remove_list_parameter(self, list_name, param_id):
        
        if list_name in self.parameters:
            self.parameters[list_name] = [i for i in self.parameters[list_name] if  i.name != param_id]
    
    def remove_list_parameter_by_value(self, list_name, value):
        
        if list_name in self.parameters:
            self.parameters[list_name] = [i for i in self.parameters[list_name] if  i.value != value]
        
    
    def get_list_parameters(self, list_name, param_id):
        if list_name in self.parameters:
            return [i for i in self.parameters[list_name] if i.name == param_id]
        else:
            return []
    
    def get_from_query(self, query):
        
        for parameter in Parameter.get_all_subclasses():
            param = parameter.get_from_query(query)
            if param is not None:
                return param

    def get_parameter(self, parameter):
        if parameter in self.parameters:
            return self.parameters[parameter]
        else:
            return None
    
    def get_parameter_names(self):
        return self.parameters.keys()

    def get_value(self, parameter):
        return self.parameters[parameter].value
    #housekeeping functions!! 
    def _clear(self):
        self.parameters = {}
    
    def clear(self):
        retained = ['current_model', 'current_map', 'chememBackendPath']
        keys_to_delete = [key for key in self.parameters if key not in retained]
        for key in keys_to_delete:
            del self.parameters[key]
    
    def clear_binding_site_tabs(self, chemem):
        sites = self.get_parameter('binding_sites')
        if sites is not None:
            for site in sites:
                
                js_code = f'triggerDeleteButtonClickOnBindingSite("{site.name}");'
                chemem.run_js_code(js_code)
                    
                    
    
    def copy(self):
        return copy.copy(self)

#╔═════════════════════════════════════════════════════════════════════════════╗
#║                                  Commands                                   ║
#╚═════════════════════════════════════════════════════════════════════════════╝


class Command:
    @classmethod
    def js_code(cls, *args):
        """Generate JavaScript code specific to the command."""
        pass
    
    @classmethod 
    def update_chemem(cls, *args):
        """Update ChemEM object state data."""
        pass 
    
    @classmethod
    def execute(cls, chemem, command, query):
        """Execute the command with provided arguments."""
        
        if command  == cls.__name__:
            #TODO!
            if True:
            #try:
                cls.run(chemem, query)
            #except Exception as e:
            #    alert_message = f'ChemEM Error, unable to run command: {cls.__name__} - {e}'
            #    js_code = f'alert("{alert_message}");'
            #    chemem.run_js_code(js_code)
                
    
    @classmethod
    def run(cls, chemem, query):
        """Utility method to run JavaScript code on a given HTML view."""
        pass


class LoadChimeraXJob(Command):
    @classmethod 
    def run(cls, chemem, query):
        if query in chemem.job_handeler.jobs:
            job = chemem.job_handeler.jobs[query]
            path = PathParameter('loadJob', job.params.output)
            chemem.parameters.add(path)
            LoadJobFromPath().run(chemem,query)
            
            
            
class LoadJobFromPath(Command):
    @classmethod
    def run(cls, chemem, query):
        #removed currently loaded job 
        if chemem.current_loaded_result is not None:
            
            js_code =  'clearResultsLists();'
            chemem.run_js_code(js_code)
            chemem.current_loaded_result.clear()
            chemem.current_loaded_result = None
            
            
        load_path = chemem.parameters.get_parameter('loadJob')
        #check what data is avalible 
        preprocessing_path = os.path.join( load_path.value, 'preprocessing')
        fitting_path = os.path.join( load_path.value, 'fitting')
        postprocessing_path = os.path.join( load_path.value, 'post_processing')
        
        #preprocessing files 
        if os.path.exists( preprocessing_path ):

            pre_processed_map_files = [i for i in os.listdir(preprocessing_path) if i.endswith('.mrc')]
            
        else:
            pre_processed_map_files = []
        
        
        #fitting files
        if os.path.exists( fitting_path ):
            
            sdf_files = [i for i in os.listdir(fitting_path) if i.endswith('.sdf')]
            fitting_results_path = os.path.join(fitting_path, 'results.txt')
            

            if os.path.exists( os.path.join(fitting_path, 'PDB')):
                pdb_path =  os.path.join(fitting_path, 'PDB')
                pdb_files = [i for i in os.listdir(pdb_path) if i.endswith('.pdb')]
            else:
                pdb_files = None
            
            if sdf_files and pdb_files is not None:
                
                fitting_file_pairs = cls.pair_files_fitting(pdb_files, sdf_files)
        else:
            fitting_file_pairs = []
            fitting_results_path = os.path.join(fitting_path, 'results.txt')
                

        #post-processing_files
        
        if os.path.exists(postprocessing_path):
            pdb_files = [i for i in os.listdir(postprocessing_path) if i.endswith('.pdb')]
            sdf_files = [i for i in os.listdir(postprocessing_path) if i.endswith('.sdf')]
            
            if sdf_files and pdb_files:
                postproc_file_pairs = cls.pair_files_postproc(pdb_files, sdf_files)
            
            post_proc_results_path = os.path.join(postprocessing_path, 'results.txt')
            
        
        else:
            postproc_file_pairs = []
            post_proc_results_path = os.path.join(postprocessing_path, 'results.txt')
        
        results_object = ChemEMResult( chemem.session, 
                                       preprocessing_path,
                                       pre_processed_map_files,
                                       fitting_path,
                                       fitting_file_pairs,
                                       fitting_results_path,
                                       postprocessing_path,
                                       postproc_file_pairs,
                                       post_proc_results_path
                                      )
        
        chemem.current_loaded_result = results_object
        for message in chemem.current_loaded_result.messages:
            chemem.run_js_code(message)
    
            
    @classmethod 
    def pair_files_fitting(cls, pdb_files, sdf_files):
        fitting_file_pairs = []
        
        for pdb in pdb_files:
            pdb_id = pdb.split('_')[1].replace('.pdb', '')
            pair = [pdb]
            for sdf in sdf_files:
                sdf_id = sdf.split('_')[1] 
                if pdb_id == sdf_id:
                    pair.append(sdf)
            
            fitting_file_pairs.append(pair)
            
        return fitting_file_pairs
    
    @classmethod 
    def pair_files_postproc(cls, pdb_files, sdf_files):
        fitting_file_pairs = []
        
        for pdb in pdb_files:
            pdb_id = pdb.split('_')[1].replace('.pdb', '')
            pdb_cycle_id =  pdb.split('_')[3].replace('.pdb', '')
            pair = [pdb]
            for sdf in sdf_files:
                sdf_id = sdf.split('_')[1] 
                sdf_cycle_id =  sdf.split('_')[3] 
                if pdb_id == sdf_id and pdb_cycle_id == sdf_cycle_id:
                    pair.append(sdf)
            
            fitting_file_pairs.append(pair)
            
        return fitting_file_pairs


class HideSolutionMap(Command):
    @classmethod 
    def run(cls, chemem, query):
        query = str(query)
        if chemem.current_loaded_result is not None:
            
            for density_map in chemem.current_loaded_result.map_objects:
                if density_map.id == query:
                    density_map.hide()
        

class ViewSolutionMap(Command):
    @classmethod 
    def run(cls, chemem, query):
        query = str(query)
        if chemem.current_loaded_result is not None:
            
            for density_map in chemem.current_loaded_result.map_objects:
                if density_map.id == query:
                    density_map.show()
        
class ViewFittingSolutionMap(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_loaded_result is not None:
            for solution in chemem.current_loaded_result.fitting_results:
                if solution.id == query:
                    solution.show_solution()
                    return
            for solution in chemem.current_loaded_result.postprocessing_results:
                if solution.id == query:
                    solution.show_solution()
                    return

class HideFittingSolutionMap(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_loaded_result is not None:
            for solution in chemem.current_loaded_result.fitting_results:
                if solution.id == query:
                    solution.hide_solution()
                    return
        
            for solution in chemem.current_loaded_result.postprocessing_results:
                if solution.id == query:
                    solution.hide_solution()
                    return
            


class ProtonateSmiles(Command):
    @classmethod 
    def js_code(cls, smiles):
        js_code = f'addSmilesToProtonationList("{smiles}");'
        return js_code 
    
    @classmethod
    def run(cls,chemem, query):
        #need to delete the list and images at this stage TODO!
        chemem.run_js_code("removeAllProtonatedSmiles();")
        p  = Protonate.from_query(query)
        p.protonate()
        
        for state in p.protonation_states:
            js_code = cls.js_code(state)
            chemem.run_js_code(js_code)
        
        chemem.current_protonation_states = p
        

class ShowLigandSmilesImage(Command):
    
    @classmethod 
    def js_code(cls, image_path):
        js_code = f'displayChemicalStructure("{image_path}");'
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        
        if chemem.current_protonation_states is not None:
            if chemem.current_protonation_states.current_image_file is not None:
                chemem.current_protonation_states.remove_temporary_file()
            try:
                
                image_idx = chemem.current_protonation_states.protonation_states.index(query)
                chemem.current_protonation_states.save_image_temporarily(image_idx)
                file_path = chemem.current_protonation_states.current_image_file.name
                if file_path is not None:
                    
                    js_code = cls.js_code(file_path)
                    chemem.run_js_code(js_code)
                    
                    #chemem.current_protonation_states.remove_temporary_file()
            except ValueError:
                js_code = f'alert("Smiles image not found {query}");'
                chemem.run_js_code(js_code)
                
           

class CleanProtonationImage(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_protonation_states is not None:
            if chemem.current_protonation_states.current_image_file is not None:
                chemem.current_protonation_states.remove_temporary_file()

class ResetConfig(Command):
    @classmethod 
    def run(cls, chemem, query):
        
        chemem.parameters.clear_binding_site_tabs(chemem)
        chemem.parameters.clear()

class RemoveJob(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.job_handeler.remove_job(query)
        
       

class IncrementBindingSiteCounter(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.avalible_binding_sites += 1
        
class AddBindingSite(Command):
    
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.add_list_parameter('binding_sites', query)
        chemem.current_binding_site_id = query.name
   
class AddBindingSiteToConf(Command):
    
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.add_list_parameter('binding_sites_conf', query)
        chemem.current_binding_site_id = query.name        

class RemoveBindingSite(Command):
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.remove_list_parameter('binding_sites', query)
        #TODO! check the effects of this elsewhere
        if chemem.rendered_site is not None:
            if query == chemem.rendered_site.binding_site.name:
                chemem.rendered_site.reset() 
                chemem.rendered_site = None

class RemoveBindingSiteFromConf(Command):
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.remove_list_parameter('binding_sites_conf', query)
        
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

class UpdateExes(Command):
    
    @classmethod 
    def js_code(cls, exe_names):
        options_json = json.dumps(exe_names)

        js_code = f'''
        updateSelect("chememExes", {options_json});
        onExesPopulated();
        '''
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        exe_names = []
        for index, exe in enumerate(chemem.avalible_chemem_exes):
            exe_names.append( cls.get_exe_id( index, exe) )
        
        js_code = cls.js_code(exe_names)
        chemem.run_js_code(js_code)
    
    @classmethod 
    def get_exe_id(cls, index, exe ):
        return f'{exe.value}'
        
class UpdateModels(Command):
    @classmethod
    def js_code(cls, model_names):
        options_json = json.dumps(model_names)  # Ensure model names are in JSON array format
        js_code = f'''
        updateSelect("models", {options_json});
        onModelsPopulated();
        '''
        
        return js_code
      
    
    @classmethod
    def run(cls, chemem, query):
        model_names = []
        for index, model in enumerate(chemem.session.models): 
            if isinstance(model, AtomicStructure):
                model_names.append( cls.get_model_id( model ) )
        
        js_code = cls.js_code(model_names)
        chemem.run_js_code(js_code)
    
    @classmethod 
    def get_model_id(cls, model):
        return f'{".".join([str(i) for i in model.id]) } - {model.name}'


class SetCurrentExe(Command):
    
    @classmethod 
    def run(cls, chemem, query):
        path = [i for i in chemem.avalible_chemem_exes if i.value == query.value][0]
        cls.update_chemem(chemem, path)
    
    @classmethod 
    def update_chemem(cls, chemem, path):
       chemem.parameters.parameters['chememBackendPath'] = path 
            
    
class SetCurrentModel(Command):
    
    @classmethod 
    def run(cls, chemem, query):
        if chemem.session.models.have_id(query.value):
            model = [i for i in chemem.session.models if i.id == query.value][0]
            cls.update_chemem(chemem, model)
        else:
            js_code = f"alert('ChemEM can't assign model with id: {query.name}');"
            chemem.run_js_code(js_code)
    
    @classmethod 
    def update_chemem(cls, chemem, model):
        """Update ChemEM object state data."""
        if 'current_model' in chemem.parameters.parameters:
            chemem.parameters.parameters['current_model'] = model 
            
            
        else:
            chemem.parameters.parameters['current_model'] = model

class SetSimulationMap(Command):
    #TODO! add cache functions!!!
    @classmethod 
    def run(cls, chemem, query):
    
        if query is None: #change to N/A
            if 'current_map' in chemem.current_simulation.parameters:
                del chemem.current_simulation.parameters['current_map']
                
        elif chemem.session.models.have_id(query.value):
            model = [i for i in chemem.session.models if i.id == query.value][0]
            cls.update_chemem(chemem, model)
        else:
            js_code = f"alert('ChemEM can\\'t assign simulation map with id: {query.name}');"
            chemem.run_js_code(js_code)
    
    @classmethod 
    def update_chemem(cls, chemem, model):
        """Update ChemEM object state data."""
        
        chemem.current_simulation.parameters['current_map'] = model 
        
class SetSimulationModel(Command):
    @classmethod 
    def run(cls, chemem, query):
        
        
        if chemem.session.models.have_id(query.value):
            
            model = [i for i in chemem.session.models if i.id == query.value][0]
            cls.update_chemem(chemem, model)
        elif query.value == 'SelectedAtoms':
            model = cls.get_selected_model(chemem)
            if model is not None:
                chemem.current_simulation.parameters['selected_atoms'] = True
                cls.update_chemem(chemem, model)
        else:
            
            js_code = f"alert('ChemEM can\\'t assign simulation map with id: {query.name}');"
            chemem.run_js_code(js_code)
    
    @classmethod 
    def update_chemem(cls, chemem, model):
        """Update ChemEM object state data."""
        
        chemem.current_simulation.parameters['current_model'] = model 
    
    @classmethod 
    def get_selected_model(cls, chemem):
        selected_models = []
        for model in chemem.session.models:
            if model.selected:
                selected_models.append(model)
        if len(selected_models) > 1:
            chemem.run_js_code("alert('Please only select atoms from a single model');")
            return False
        else: 
            return selected_models[0]
            

class SetCurrentMap(Command):
    #TODO! add cache functions!!!
    @classmethod 
    def run(cls, chemem, query):
        
        
        if query is None: #change to N/A
            if 'current_map' in chemem.parameters.parameters:
                del chemem.parameters.parameters['current_map']
                
        elif chemem.session.models.have_id(query.value):
            model = [i for i in chemem.session.models if i.id == query.value][0]
            cls.update_chemem(chemem, model)
        else:
            js_code = f"alert('ChemEM can't assign map with id: {query.name});"
            chemem.run_js_code(js_code)
    
    @classmethod 
    def update_chemem(cls, chemem, model):
        """Update ChemEM object state data."""
        
        chemem.parameters.parameters['current_map'] = model 

class UpdateMaps(Command):
    @classmethod
    def js_code(cls, model_names):
        options_json = json.dumps(model_names)  
        js_code = f'''
        updateSelect("maps", {options_json});
        onMapsPopulated();
        ''' #can remove on models populated!!!
        return js_code
    
    @classmethod
    def run(cls, chemem, query):
        model_names = []
        for index, model in enumerate(chemem.session.models): 
            
            if isinstance(model, Volume):
                model_names.append(cls.get_map_id(model))
        
        js_code = cls.js_code(model_names)
        chemem.run_js_code(js_code)    
    
    @classmethod 
    def get_map_id(cls, model):
        return f'{".".join([str(i) for i in model.id]) } - {model.name}'

class UpdateSimulationModels(Command):
    @classmethod
    def js_code(cls, model_names):
        options_json = json.dumps(model_names)  # Ensure model names are in JSON array format
        js_code = f'''
        updateSelect("simulationModels", {options_json});
        onModelsPopulated();
        '''
        
        return js_code
      
    
    @classmethod
    def run(cls, chemem, query):
        model_names = []
        for index, model in enumerate(chemem.session.models): 
            if isinstance(model, AtomicStructure):
                model_names.append( cls.get_model_id( model ) )
        
        js_code = cls.js_code(model_names)
        chemem.run_js_code(js_code)
    
    @classmethod 
    def get_model_id(cls, model):
        return f'{".".join([str(i) for i in model.id]) } - {model.name}'



class UpdateSimulationMaps(Command):
    @classmethod 
    def js_code(cls, model_names):
        options_json = json.dumps(model_names)  
       
        js_code = f'''
        updateSelect("SimulationMaps", {options_json});
        onSimulationMapsPopulated();
        ''' #can remove on models populated!!!
        
        return js_code
    @classmethod
    def run(cls, chemem, query):
        model_names = []
        for index, model in enumerate(chemem.session.models): 
            
            if isinstance(model, Volume):
                model_names.append(cls.get_map_id(model))
        
        js_code = cls.js_code(model_names)
        chemem.run_js_code(js_code)    
    
    @classmethod 
    def get_map_id(cls, model):
        return f'{".".join([str(i) for i in model.id]) } - {model.name}'

class ReadFileAndAlterChemEM(Command):
    
    @classmethod 
    def js_code(cls):
        
        js_code = 'sendSiteValueToBackend("chemem:UpdateExes", "None");'
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        file_dialog = open_command.dialog.OpenDialog(parent=chemem.session.ui.main_window, starting_directory='' )
        file = file_dialog.get_path()
        if file is not None:
            if os.path.isfile(file):
                js_code = cls.js_code()
                cls.update_chemem(chemem, file, query)
                
            else:
                js_code= f"alert(File does not exist: {file});"
            
            chemem.run_js_code(js_code)
    
    @classmethod
    def update_chemem(cls,chemem, file, query):
        
        attribute = getattr(chemem, query)
        attribute += [PathParameter(file, file)]
        setattr(chemem, query, attribute)
     
class ReadFile(Command):
    
    @classmethod 
    def js_code(cls,  element_id, file):
        
        condensed_path = file
        #condensed_path = condense_path(file)
        js_code = f'setFilePath( "{element_id}", "{condensed_path}" );' 
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        file_dialog = open_command.dialog.OpenDialog(parent=chemem.session.ui.main_window, starting_directory='' )
        file = file_dialog.get_path()
        if file is not None:
            if os.path.isfile(file):
                js_code = cls.js_code(query, file)
                cls.update_chemem(chemem, file, query)
                
            else:
                js_code= f"alert(File does not exist: {file});"
            
            chemem.run_js_code(js_code)
    
    @classmethod
    def update_chemem(cls,chemem, file, query):
        
        chemem.parameters.add(PathParameter(query, file))

class SetDir(Command):
    
    @classmethod 
    def js_code(cls,  element_id, file):
        condensed_path = condense_path(file)
        js_code = f'setFilePath( "{element_id}", "{condensed_path}" );' 
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        file_dialog =  open_command.dialog.OpenFolderDialog(chemem.session.ui.main_window, chemem.session)
    
        file = file_dialog.get_path()
        if file is not None:
            if os.path.isdir(file):
                js_code = cls.js_code(query, file)
                cls.update_chemem(chemem, file, query)
                
            else:
                js_code= f"alert(File does not exist: {file});"
            
            chemem.run_js_code(js_code)
    
    @classmethod
    def update_chemem(cls,chemem, file, query):
        chemem.parameters.add(PathParameter(query, file))
        

class AddLigandSmiles(Command):
    @classmethod 
    def js_code(cls, smiles):
        js_code = f"alert(Invalid Ligand SMILES: {smiles});"
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        valid_smiles = cls.validate_smiles(query.value)
        if valid_smiles:
            chemem.parameters.add_list_parameter('Ligands', query)
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

class AddLigandFile(Command):
    
    @classmethod
    def js_code(cls, file):
        js_code = f"alert(Invalid Ligand File: {file});"
        return js_code

    @classmethod 
    def run(cls, chemem, query):
        #TODO! valid_smiles = cls.validate_smiles(query.value)
        valid_file = True
        if valid_file:
            chemem.parameters.add_list_parameter('Ligands', query)
        
        else:
            js_code = cls.js_code(query.value)
            chemem.run_js_code(js_code)

class RemoveLigand(Command):
    @classmethod 
    def js_code(cls):
        pass 
    @classmethod 
    def run(cls, chemem, query):
        chemem.parameters.remove_list_parameter('Ligands', query)


class UpdateChemEMParameter(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.parameters.add(query)
        
class UpdateSimulationParameter(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.current_simulation.add(query)
        
class SaveConfFile(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.make_conf_file(chemem.parameters)

class RunChemEM(Command):
    @classmethod 
    def run(cls, chemem, query):
        
        chemem.make_conf_file(chemem.parameters)
        executable = chemem.parameters.parameters['chememBackendPath'].value 
        chemem.run(executable,CHEMEM_JOB)


class SetMaunualBindingSite(Command):
    
    @classmethod 
    def js_code(cls):
        return "bindingSiteCounter++;"
    
    @classmethod 
    def run(cls, chemem, query):
        
        pass
        #update ligand bindig site id!! in js_code


        

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
            for centroid in marker_centroids:
                
                
                site  = BindingSiteParameter.get_from_centroid(centroid, chemem)
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
        
        chemem.rendered_site =  RenderBindingSite(chemem.session, site, 
                                                  chemem.parameters.parameters['current_model'],
                                                  chemem.parameters.get_parameter('current_map'))
        chemem.run_js_code(cls.js_code(site))
            
       

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



class RunSignificantFeaturesProtocol(Command):
    
    @classmethod
    def js_code_error(cls, map_id):
        return f'alert("Resolution needed for EM-map {map_id}.");'
    
    @classmethod 
    def js_code(cls, key, text):
        return f'addSigFeatListItem("{text}", {key});'
        
    
    @classmethod 
    def run(cls, chemem, query):
        if chemem.parameters.get_parameter('resolution') is not None:
            resolution = chemem.parameters.get_parameter('resolution')
            significant_features_object =  SignificantFeaturesProtocol.mask(chemem.session, chemem.rendered_site, resolution.value)
            chemem.current_significant_features_object = significant_features_object
            for key, string in significant_features_object.js_code:
                chemem.run_js_code(cls.js_code(key, string))
            
            chemem.rendered_site.map.set_display(False)
        else:
            chemem.run_js_code(cls.js_code_error())

class ShowSignificantFeature(Command):
    @classmethod 
    def run(cls, chemem, query):

        chemem.current_significant_features_object.show_feature(int(query))
        
class HideSignificantFeature(Command):
    @classmethod 
    def run(cls, chemem, query):
        
        chemem.current_significant_features_object.hide_feature(int(query))       
        
class SetEditBindingSiteValue(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.render_site_from_key(query)
                 
class AddSolutionToSimulation(Command):
    @classmethod
    def run(cls, chemem, query):
        solution = chemem.current_loaded_result.get_loaded_result_with_id(query)
        chemem.current_simulation.parameters['AddedSolution'] = solution
        #need to add an rd_ligand also
        js_code = f'setSolutionValue("{solution.string()}");'
        chemem.run_js_code(js_code)
        


class BuildSimulation(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem_executable = chemem.parameters.get_parameter('chememBackendPath')
        if chemem_executable is not None:
            if os.path.exists(f"{chemem_executable.value}.export_simulation"): 
                
                
                executable = f"{chemem_executable.value}.export_simulation"
                if 'AddedSolution' in chemem.current_simulation.parameters:
                    solution = chemem.current_simulation.parameters['AddedSolution']
                    ligand_files = extract_simulation_data_from_solution(solution)
                    
                    chemem.current_simulation.parameters['Ligands'] = ligand_files
                    new_model = build_model_without_ligands(solution.pdb_object)
                    chemem.current_simulation.parameters['current_model'] = new_model
                
                    #Need to do something with the ligands!!!
                    #ADD A CHECK FOR IF THE LIGAND FILES ARE IN THE PDB
                
                
                
                #Here is where you have a selected region id
                
                #Selected atoms!!, essentailly, ligands are always included but need to find +1 and -1 residues to pin.
                if chemem.current_simulation.get_parameter('selected_atoms') is not None:
                    selected_residues_info = analyze_selected_atoms(chemem.current_simulation.parameters['current_model'])
                    chemem.current_simulation.parameters['selected_atom_data'] = selected_residues_info
                    non_selected_residues = selected_residues_info['res_atoms_not_selected']
                    anchor_residues =  selected_residues_info['extra_residues']
                    
                    
                    for atom_array in non_selected_residues.values():
                        for atom in atom_array:
                            atom.selected = True 
                    
                    for residue in anchor_residues:
                        for atom in residue.atoms:
                            atom.selected = True
                    
                    
                #TODO!
                chemem.temp_build_dir = tempfile.mkdtemp()
                #chemem.temp_build_dir = '/Users/aaron.sweeney/Documents/ChemEM_chimera_v2/debug/test_simulate/'
                chemem.current_simulation.add(PathParameter('output', chemem.temp_build_dir))
                chemem.make_conf_file(chemem.current_simulation)
                chemem.run(executable, EXPORT_SIMULATION)
                
                #create a tempory directory to write files to 
                #run the export_simulation blocking?
                
            else:
                chemem.run_js_code('alert("Simulation requires ChemEM v0.0.4");')
        else:
            chemem.run_js_code('alert("No ChemEM executable found");')



class GetAvaliblePlatforms(Command):
    
    @classmethod 
    def js_code(cls, platform_name):
        js_code = f'addPlatformToList("{platform_name}");'
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        if chemem.platforms_set:
            pass
        else:
            for platform_name in chemem.avalible_platforms:
                js_code = cls.js_code(platform_name)
                chemem.run_js_code(js_code)
            chemem.platforms_set = True

class SetPlatform(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.platform = query


class RunSimulation(Command):
    #TODO! -- with this set up the value of current simulation needs to be set to none 
    #when a new simulation is created
    @classmethod 
    def run(cls, chemem, query):
        
        if chemem.current_simualtion_id is None:
            chemem.run_simulation()
        else:
            
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            job._pause = False 

class PauseSimulation(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            job._pause = True
            
class StopSimulation(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            job.running = False


class SetSimulationTempreture(Command):
    @classmethod 
    def run(cls,chemem, query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            job._set_temp = query.value

class EnableTugMode(Command):
    @classmethod
    def run(cls,chemem, query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            chemem.default_mouse_mode = chemem.session.ui.mouse_modes.mode(button='right')
            new_mode = DragCoordinatesMode(chemem.session, {i[0] : i[1] for i in chemem.atoms_to_position_index})
            chemem.session.ui.mouse_modes.bind_mouse_mode(mouse_button="right", mode= new_mode)
            job.tug = new_mode
        


class DisableTugMode(Command):
    @classmethod
    def run(cls,chemem, query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            job.tug = None
            chemem.session.ui.mouse_modes.bind_mouse_mode(mouse_button="right", mode= chemem.default_mouse_mode)
            chemem.default_mouse_mode = None

class SetPiPiPTug(Command):
    @classmethod 
    def run(cls, chemem, query ):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            simulation_model = chemem.simulation_model
            atoms_index_pairs = get_pipi_tug_indexes(simulation_model, chemem.atoms_to_position_index_as_dic)
            openff_index_pairs = [i[1] for i in atoms_index_pairs]
            job.pipi_p_tug = [openff_index_pairs, None, None, None]

class SetHBondTug(Command):
    @classmethod
    def js_code(cls, selected_atoms, restraint_id):
        # Function to escape JavaScript string literals
        def escape_js_string(s):
            return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')
    
        # Function to remove the part of the string before and including '.pdb '
        def trim_pdb_info(s):
            parts = s.split('.pdb ')
            return parts[1] if len(parts) > 1 else s  # returns the part after '.pdb ' if it exists
    
        # Construct the message with all special characters escaped
        donor = escape_js_string(trim_pdb_info(selected_atoms[0].string()))
        hydrogen = escape_js_string(trim_pdb_info(selected_atoms[1].string()))
        acceptor = escape_js_string(trim_pdb_info(selected_atoms[2].string()))
    
        message = f'Donor: {donor}\\nHydrogen: {hydrogen}\\nAcceptor: {acceptor}'
    
        # Create the JavaScript code string safely
        js_code = f'addRestraintToList("{message}", "{restraint_id}");'
        return js_code

    @classmethod 
    def run(cls,chemem,query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            simulation_model = chemem.simulation_model
            #selected_atoms retured as Donor, hydrogen, acceptor
            hbond_idx, selected_atoms = get_hbond_tug_index(simulation_model, chemem.atoms_to_position_index_as_dic)
            
            job.hbond_tug = [hbond_idx, None, None]
            restraint_id = len(chemem.current_restraints)
            
            js_code = cls.js_code(selected_atoms, restraint_id)
            chemem.run_js_code(js_code)
            selected_atoms[1].selected = False
            #TODO add these properly and color code them
            com = 'distance sel'
            run(chemem.session, com)
            chemem.current_restraints[restraint_id] = [hbond_idx, selected_atoms]
                    
                    
class DeleteHbondTug(Command):
    @classmethod 
    def run(cls, chemem, query):
        
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            hbond_tug_idx , selected_atoms = chemem.current_restraints[int(query)]
            com = f'distance delete {selected_atoms[0].atomspec} {selected_atoms[2].atomspec}'
            run(chemem.session, com)
            job.hbond_tug = [hbond_tug_idx, 0.0, 0.0]
            del chemem.current_restraints[int(query)]
            
        #set the tug!
        #add a distance mointor 
        #send a thing to the template


class RunSimulatedAnneling(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simualtion_id is not None:
            
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            job.simulated_anneling  = query

class MinimiseSimulation(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            job.minimise  = 1

class AddLigandToSimulation(Command):
    @classmethod 
    def js_code(self):
        return ''
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simualtion_id is not None:
            chemem_executable = chemem.parameters.get_parameter('chememBackendPath')
            if chemem_executable is not None:
                if os.path.exists(f"{chemem_executable.value}.export_ligand_simulation"): 
                    executable = f"{chemem_executable.value}.export_ligand_simulation"
                    #Validate smiles!!
                    mol = smiles_is_valid(query.value)
                    if mol:
                        #???TODO!
                        chemem.run_js_code('stopSimulation();')
                        
                        #chemem.temp_build_dir ='/Users/aaron.sweeney/Documents/ChemEM_chimera_v2/debug/test_simulate/add_lig'
                        
                        chemem.temp_build_dir = tempfile.mkdtemp()
                        #solvent???TODO!!
                        params = Parameters()
                        params.add_list_parameter('Ligands', query)
                        params.add(PathParameter('output', chemem.temp_build_dir))
                        chemem.make_conf_file(params)
                        chemem.run(executable, EXPORT_LIGAND_SIMULATION)
                        
                    else:
                        
                        error =f'Invalid SMILES string for smiles {query.value}\n Check your input and try again.\n The LIGANDS tab can be helpful for formatting SMILES correctly.'
                        chemem.run_js_code(f"alert(`{error}`);")


class AddCovelentLigandToSimulation(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simualtion_id is not None:
            mol = smiles_is_valid(query.value)
            if mol:
                chemem.covelent_ligand = Parameters() 
                chemem.covelent_ligand.add(query)
            
            else:
                error =f'Invalid SMILES string for smiles {query.value}\n Check your input and try again.\n The LIGANDS tab can be helpful for formatting SMILES correctly.'
                chemem.run_js_code(f"alert(`{error}`);")
        


class BindCovelentLigandToAtom(Command):
    
    
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simualtion_id is not None:
            if chemem.covelent_ligand.get_parameter("covelent_ligand") is not None:
                #Add the bond type of binding between protein and ligand
                
                model = chemem.simulation_model
                
                selected_atoms = []
                for residue in model.residues:
                    for atom in residue.atoms:
                        if atom.selected:
                            selected_atoms.append(atom)
                
                if len(selected_atoms) == 1:
                    chemem.covelent_ligand.parameters['protein_atom'] = selected_atoms[0]
                    chemem.covelent_ligand.add(query)
                    
                    residue_id = selected_atoms[0].residue.name 
                    
                    residue_smiles, residue_idx = RD_PROTEIN_SMILES[residue_id] 
                    
                    residue_rw_mol = RW_mol_from_smiles(residue_smiles)
                    residue_index_image = draw_molecule_with_atom_indices(residue_rw_mol)
                    chemem.covelent_ligand.parameters['residue_rw_mol'] = residue_rw_mol
                    chemem.covelent_ligand.parameters['residue_image'] = residue_index_image
                    chemem.covelent_ligand.parameters['residue_idx'] = residue_idx
                    
                    
                    
                    selected_atom_idx = residue_idx[selected_atoms[0].name]
                    chemem.covelent_ligand.parameters['selected_atom_idx'] = selected_atom_idx
                    
                    ligand_rw_mol = RW_mol_from_smiles(chemem.covelent_ligand.get_parameter("covelent_ligand").value)
                    ligand_index_image = draw_molecule_with_atom_indices(ligand_rw_mol)
                    chemem.covelent_ligand.parameters['ligand_rw_mol'] = ligand_rw_mol
                    chemem.covelent_ligand.parameters['ligand_image'] = ligand_index_image
                    
                    ligand_temp_image_path = save_image_temporarily(ligand_index_image)
                    residue_temp_image_path = save_image_temporarily(residue_index_image)
                    
                    chemem.covelent_ligand.parameters['temp_ligand_image'] = ligand_temp_image_path
                    chemem.covelent_ligand.parameters['temp_residue_image'] = residue_temp_image_path
                    
                    js_code = "resetSimulationDisplay();"
                    chemem.run_js_code(js_code)
                    js_code = f"showCovelentLigandAdvancedOptions('{residue_temp_image_path.name}', '{ligand_temp_image_path.name}');"
                    chemem.run_js_code(js_code)
                else :
                    chemem.run_js_code('alert("Please select a single protein atom for ligand binding");')
                    

class AddCovelentListParameter(Command):
    @classmethod 
    def run(cls,chemem,query):
        if chemem.current_simualtion_id is not None:
            if chemem.covelent_ligand is not None:
                chemem.covelent_ligand.add_list_parameter(query.name, query)

class AddCovelentParameter(Command):
    @classmethod 
    def run(cls,chemem,query):
        if chemem.current_simualtion_id is not None:
            if chemem.covelent_ligand is not None:
                chemem.covelent_ligand.add(query)

class RemoveCovelentListParameter(Command):
    
    @classmethod 
    def run(cls,chemem,query):
        if chemem.current_simualtion_id is not None:
            if chemem.covelent_ligand is not None:
                chemem.covelent_ligand.remove_list_parameter_by_value(query.name, query.value)
                
class BuildCovelentSimulation(Command):
    @classmethod 
    def run(cls,chemem,query):
        if chemem.current_simualtion_id is not None:
            if chemem.covelent_ligand is not None:
                ligand = chemem.covelent_ligand.get_parameter('ligand_rw_mol')
                residue = chemem.covelent_ligand.get_parameter('residue_rw_mol')
                #TODO needs to change from list!!
                
                ligand_atom_idx = chemem.covelent_ligand.get_parameter('ligand_atom_bound_index')
                residue_atom_idx = chemem.covelent_ligand.get_parameter('selected_atom_idx')
                bond_type = chemem.covelent_ligand.get_parameter('bond_type')
                
                
                protein_atoms_to_remove = chemem.covelent_ligand.get_parameter('protein_remove_atom_idx')
                if protein_atoms_to_remove is None:
                    protein_atoms_to_remove = []
                else:
                    protein_atoms_to_remove = [i.value for i in protein_atoms_to_remove]
                
                
                ligand_atoms_to_remove = chemem.covelent_ligand.get_parameter('ligand_remove_atom_idx')
                if ligand_atoms_to_remove is None:
                    ligand_atoms_to_remove = []
                else:
                    ligand_atoms_to_remove = [i.value for i in ligand_atoms_to_remove]

                ligand_bonds_to_change = chemem.covelent_ligand.get_parameter('ligand_modified_bond')
                if ligand_bonds_to_change is None:
                    ligand_bonds_to_change = []
                else:
                    ligand_bonds_to_change = [i.as_tuple() for i in ligand_bonds_to_change]
                
                
                protein_bonds_to_change = chemem.covelent_ligand.get_parameter('protein_modified_bond')
                if protein_bonds_to_change is None:
                    protein_bonds_to_change = []
                else:
                    protein_bonds_to_change = [i.as_tuple() for i in protein_bonds_to_change]
                
                #TODO! new
                ligand =  chemem.covelent_ligand.get_parameter("covelent_ligand").value
                
                residue_name = chemem.covelent_ligand.parameters['protein_atom'].residue.name

                
                product_smiles, combined_rwmol,ligand_atom_idx, residue_atom_idx = combine_molecules_and_react(ligand,
                                            residue_name,
                                            ligand_atom_idx.value,
                                            residue_atom_idx,
                                            bond_type.value,
                                            protein_bonds_to_change,
                                            ligand_bonds_to_change,
                                            protein_atoms_to_remove,
                                            ligand_atoms_to_remove
                                            )
                
                chemem.covelent_ligand.add(SmilesParameter('product_smiles', product_smiles))
                chemem.covelent_ligand.parameters['combined_rwmol'] = combined_rwmol
                
                
                chemem_executable = chemem.parameters.get_parameter('chememBackendPath')
                
                
                if chemem_executable is not None:
                    executable = f"{chemem_executable.value}.export_ligand_simulation"
                    chemem.run_js_code('stopSimulation();')
                       
                    #chemem.temp_build_dir = '/Users/aaron.sweeney/Documents/ChemEM_chimera_v2/test_covelent_ligand_2/'
                    chemem.temp_build_dir = tempfile.mkdtemp()
                    #solvent???TODO!!
                    params = Parameters()
                    params.add_list_parameter('Ligands', chemem.covelent_ligand.parameters['product_smiles'])
                    params.add(PathParameter('output', chemem.temp_build_dir))
                    params.add(IntParameter('protonation', 0))
                    
                    print(params.parameters)
                    
                    chemem.make_conf_file(params)
                    
                    chemem.run(executable, EXPORT_COVELENT_LIGAND_SIMULATION)
                #now run chemem get parameters!!

class AddMMGBSAParameter(Command):
    @classmethod 
    def run(cls, chemem, query):
        #if chemem.current_simualtion_id is not None:
        chemem.mmgbsa.add(query)


class RunMMGBSAEquilibrium(Command):
    @classmethod 
    def run(cls, chemem, query):
        #TODO! check if there is still an equlibrium!
        if chemem.simulation is not None:
            solvent_model =  chemem.mmgbsa.get_parameter('implicitSolventModel')
            complex_structure = chemem.simulation.complex_structure
            complex_system = get_mmpbsa_complex_system(complex_structure, solvent_model)
            
            simulation = Simulation(chemem.session, 
                                    complex_system, 
                                    complex_structure,
                                    None,
                                    platform_name = chemem.platform)
            
            simulation.set_SSE_elements(chemem.simulation_model, 
                                        chemem.atoms_to_position_index_as_dic)
            simulation.set_force_groups()
            if chemem.mmgbsa.get_parameter('equilibriumRestraints').value == 'SSEs':
                #helix forces 
                #TODO! reduce K
                simulation.add_force(HelixHbondForce)
                simulation.add_force(PsiAngleForce)
                simulation.add_force(PhiAnglelForce)
            
            
            simulation.set_simulation()
            chemem.mmgbsa_simulation = simulation
            
        #check if the same simulaton has been eq. if not
            #build the simulation 
        #run for eqilibrium 
        #return reporters 
        #mark as eqilirised

class RunMMGBSA(Command):
    @classmethod 
    def run(cls, chemem, query):
        pass
        #build and equlibritate system:
#╔═════════════════════════════════════════════════════════════════════════════╗
#║                             Class Helpers                                   ║
#╚═════════════════════════════════════════════════════════════════════════════╝


class RenderBindingSite:
    def __init__(self, session, 
                 binding_site, 
                 current_model,
                 current_map = None):
        self.session = session
        self.binding_site = binding_site 
        self.model = current_model 
        self.map = current_map 
        
        self.render_site()
        
        if self.map is not None:
            self.render_map()
        
    def render_map(self):
        #may have to reset region herer!!!
        self.reset_map_region()
        origin, apix = self.map.data_origin_and_step()
        grid_size = self.map.data.matrix().shape
        
        min_indices, max_indices = find_voxel_indices(self.min_coords, self.max_coords, apix, grid_size, origin)
        self.chimera_map_slice_key = ([min_indices[0], min_indices[1], min_indices[2]], 
                                      [max_indices[0], max_indices[1], max_indices[2]], 
                                      [1, 1, 1])
        self.map.region = self.chimera_map_slice_key
        self.update_display()
        
    def reset_map_region(self):
        self.map.region = self.map.full_region()
        #call this to change back the box size in chimera
        self.update_display()
        
    def update_display(self):
        self.map.display = False 
        self.map.display = True
    
    def render_site(self):
        self.min_coords, self.max_coords = get_box_vertices(self.binding_site.centroid,
                                                  self.binding_site.box_size)
        
        res_inside = []
        atoms_inside = []
        for res in self.model.residues:
            for atom in res.atoms:
                if self.is_outside(atom.coord):
                    atom.display = False 
                    atom.residue.ribbon_display = False
                else:
                    atom.display = True 
                    atoms_inside.append(atom)
                    if res not in res_inside:
                        res_inside.append(res)
        
        self.inside_atoms = atoms_inside
        for res in res_inside:
            res.ribbon_display = True
        
                    
    def is_outside(self, atom_coord):
        if np.any(np.array(atom_coord) < self.min_coords):
            return True
        elif np.any(np.array(atom_coord) > self.max_coords):
            return True 
        else:
            return False
    
    def reset(self):
        #need to check if the model is within the session
        
        #check if the model has been closed first!
        if self.model in self.session.models:
            for res in self.model.residues:
                res.ribbon_display = True
                for atom in res.atoms:
                    atom.display = False
        #check if the map has been closed!
        if self.map is not None:
            if self.map in self.session.models:
                self.reset_map_region()
                
    def update_centroid(self, new_centroid):
        self.binding_site.centroid = new_centroid 
        self.binding_site.value = [new_centroid, self.binding_site.box_size]
        self.reset()
        self.render_site()
    
    def update_box_size(self, new_box_size):
        self.binding_site.box_size = new_box_size 
        self.binding_site.value = [self.binding_site.centroid, new_box_size]
        self.reset() 
        self.render_site()
        



#╔═════════════════════════════════════════════════════════════════════════════╗
#║                             Helper functions                                ║
#╚═════════════════════════════════════════════════════════════════════════════╝

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



def add_openff_covelent_ligand_to_chimerax_model(openff_structure, model, name = 'LIG', atoms_to_position_index_as_dic = None):
    
    
    openff_resiude = openff_structure.residues[-1]
    chain_id = [i.chain_id for i in model.residues][-1]
    
    if atoms_to_position_index_as_dic is None:
        _, atoms_to_position_index = get_model_from_complex_structure(openff_structure, model)
        
        atoms_to_position_index_as_dic = {i[0]: i[1] for i in atoms_to_position_index}
    
    print('HERERERER')
    print(len(atoms_to_position_index_as_dic))
    
    next_residue = max([i.number for i in model.residues if i.chain_id == chain_id]) + 1
    new_residue = model.new_residue(name, chain_id, next_residue)
    print("New Residue:", name, chain_id, next_residue)
    atoms = []
    for atom in openff_resiude.atoms:
        new_atom = model.new_atom(atom.name, atom.element )
        atoms.append(new_atom)
        new_residue.add_atom(new_atom)
        new_atom.coord = np.array([atom.xx, atom.xy, atom.xz])
        atoms_to_position_index_as_dic[new_atom] = atom.idx
    
    #return model, new_residue
    bonds_added = []
    for atom in openff_resiude.atoms:
        for bond in atom.bonds:
            atom1 = bond.atom1.idx 
            atom2 = bond.atom2.idx 
            new_bond = sorted([atom1, atom2])
            if new_bond not in bonds_added:
                bonds_added.append(new_bond)
    
    print('11111111')
    print(len(atoms_to_position_index_as_dic))
    
    reverse_idx_to_atoms = {i:j for j,i in  atoms_to_position_index_as_dic.items()}
    print(len(reverse_idx_to_atoms))
    
    print('')
    print(atoms_to_position_index_as_dic)
    print('')
    print(reverse_idx_to_atoms)
    print('')
    print(bonds_added)
    for idx1, idx2 in bonds_added:
        a1 = reverse_idx_to_atoms[idx1]
        a2 = reverse_idx_to_atoms[idx2]
        bnd = model.new_bond(a1, a2)
    
    
    
    return model, new_residue, atoms_to_position_index_as_dic
        
    #add bonds 

def add_openff_ligand_to_chimerax_model(complex_structure, model, name = 'LIG'):
    
    chain_id = [i.chain_id for i in model.residues][-1]
    next_residue = max([i.number for i in model.residues if i.chain_id == chain_id]) + 1
    new_residue = model.new_residue(name, chain_id, next_residue)
    print("New Residue:", name, chain_id, next_residue)
    atoms = []
    #add atoms to residues
    
    for atom in complex_structure.atoms:
        new_atom = model.new_atom(atom.name,atom.element )
        atoms.append(new_atom)
        new_residue.add_atom(new_atom)
        new_atom.coord = np.array([atom.xx, atom.xy, atom.xz])
        
    #add bonds 
    
    bonds_added = []
    
    
    for atom in complex_structure.atoms:
        for bond in atom.bonds:
            atom1 = bond.atom1.idx 
            atom2 = bond.atom2.idx 
            new_bond = sorted([atom1, atom2])
            if new_bond not in bonds_added:
                bonds_added.append(new_bond)
    
    
    
    for a1, a2 in bonds_added:
        
        bnd = model.new_bond(atoms[a1], atoms[a2])
        
    
    #TODO! color, stick style!!

    return model, new_residue
    
    
    
    


def next_chain_id(current_id):
    """
    Generate the next chain ID following the sequence:
    'A' to 'Z', 'a' to 'z', and '0' to '9' using ASCII values.

    Parameters:
    - current_id (str): The current chain ID (expected to be a single character).

    Returns:
    - str: The next chain ID in the sequence.
    """
    if not current_id or len(current_id) != 1:
        raise ValueError("current_id must be a single character string.")

    # Obtain ASCII value of the current ID
    ascii_value = ord(current_id)

    # Define the transitions based on ASCII values
    if 'A' <= current_id <= 'Y' or 'a' <= current_id <= 'y' or '0' <= current_id <= '8':
        # Move to the next character in ASCII
        return chr(ascii_value + 1)
    elif current_id == 'Z':
        return 'a'
    elif current_id == 'z':
        return '0'
    elif current_id == '9':
        return 'A'
    else:
        raise ValueError("Invalid chain ID. Please provide a valid chain ID from the sequence A-Z, a-z, or 0-9.")



def get_pipi_tug_indexes(model, atoms_to_index):
    selected_atoms = [atom for residue in model.residues for atom in residue.atoms if atom.selected]
    
    if len(selected_atoms) <= 1:
        return None 
    
    #get two distinct rings!!
    rings = [i.rings() for i in selected_atoms if len(i.rings()) > 0]
    flattened_rings = list(set([i for j in rings for i in j]))
    
    if len(flattened_rings) == 2:
        
        ring_pair_atoms = []
        
        for ring in flattened_rings:
            atoms = ring.atoms
            
            openff_indexes = sorted([atoms_to_index[i] for i in atoms])
            ring_pair_atoms.append([atoms, openff_indexes])
        
        return ring_pair_atoms
    #TODO!! other things!!
            
            
        
def calculate_centroid(coords):
    """Calculate the centroid from a set of coordinates."""
    return np.mean(coords, axis=0)

def find_closest_triplet(coords):
    """Find the indices of the three atoms whose centroid is closest to the overall centroid."""
    # Calculate the overall centroid
    overall_centroid = calculate_centroid(coords)
    
    # Initialize variables to store the best (closest) triplet
    min_distance = float('inf')
    best_triplet = None

    # Generate all possible triplets of atom indices
    for triplet in combinations(range(len(coords)), 3):
        # Extract the coordinates of the current triplet
        triplet_coords = coords[list(triplet)]
        
        # Calculate the centroid of the current triplet
        triplet_centroid = calculate_centroid(triplet_coords)
        
        # Calculate the distance from the triplet centroid to the overall centroid
        distance = euclidean(triplet_centroid, overall_centroid)
        
        # If this triplet is closer than the best found so far, update the best triplet
        if distance < min_distance:
            min_distance = distance
            best_triplet = triplet

    # Return the indices of the best triplet
    return best_triplet
    


def get_hbond_tug_index(model, atoms_to_index):
    selected_atoms = [atom for residue in model.residues for atom in residue.atoms if atom.selected]

    if len(selected_atoms) != 3:
        return None
    
    hydrogen = next((atom for atom in selected_atoms if atom.element.name == 'H'), None)
    if not hydrogen:
        return None
    
    non_hydrogens = [atom for atom in selected_atoms if atom != hydrogen]
    if len(non_hydrogens) != 2:
        return None, None
    
    donor = next((atom for atom in hydrogen.neighbors if atom in non_hydrogens), None)
    if not donor:
        return None, None  

    acceptor = next((atom for atom in non_hydrogens if atom != donor), None)
    if not acceptor:
        return None, None  
    try:
        return [atoms_to_index[donor], atoms_to_index[hydrogen], atoms_to_index[acceptor]], [donor,hydrogen,acceptor]
    except KeyError:
        return None, None


def build_model_without_ligands(model):
    new_model = model.copy() 
    for residue in new_model.residues:
        if residue.standard_aa_name is None:
            new_model.delete_residue(residue)
    
    return new_model
                
        


def extract_simulation_data_from_solution(solution):
    #PDB without ligand!!, ligand files!!
    pdb_file = solution.pdb_object.filename 
    file_matcher = solution.pdb_object.name.replace('.pdb', '')
    ligand_dir = os.path.dirname(pdb_file)
    
    if ligand_dir.endswith('post_processing'):
        pass
    elif ligand_dir.endswith('PDB'):
        ligand_dir = ligand_dir = os.path.dirname(ligand_dir)
        
    ligand_matches = [i for i in os.listdir(ligand_dir) if i.endswith('.sdf') and i.startswith(file_matcher)]
    if len(ligand_matches):
        return [PathParameter(num, os.path.join(ligand_dir,i)) for num,i in enumerate(ligand_matches)]
        
    

def set_conda_environment():
    conda_path_guess = [
        os.path.expanduser("~/anaconda3/bin"),
        os.path.expanduser("~/miniconda3/bin"),
        "C:\\ProgramData\\Anaconda3\\Scripts",
        "C:\\ProgramData\\Miniconda3\\Scripts",
        "C:\\Anaconda3\\Scripts",
        "C:\\Miniconda3\\Scripts"
    ]
    
    for path in conda_path_guess:
        if os.path.exists(path):
            os.environ["PATH"] += os.pathsep + path

def find_anaconda_path():
    # Ensure conda is in the PATH
    set_conda_environment()

    # Check for CONDA_PREFIX environment variable
    conda_prefix = os.getenv('CONDA_PREFIX')
    if conda_prefix and os.path.exists(conda_prefix):
        return conda_prefix

    # Check if 'conda' command is available in the PATH
    try:
        conda_path = subprocess.check_output(['conda', 'info', '--base'], stderr=subprocess.DEVNULL).decode().strip()
        if os.path.exists(conda_path):
            return conda_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None

def get_conda_env_paths(conda_base_path):
    envs_path = os.path.join(conda_base_path, 'envs')
    if os.path.exists(envs_path):
        
        return [os.path.join(envs_path, env) for env in os.listdir(envs_path)]
    return []

def check_chemem_version(env_path):
    bin_dir = os.path.join(env_path, 'bin')
    if not os.path.exists(bin_dir):
        bin_dir = os.path.join(env_path, 'Scripts')  # For Windows compatibility
    
    #TODO!!!change to set version 0.4
    chemem_executable = os.path.join(bin_dir, 'chemem.export_ligand_simulation')
    if os.path.exists(chemem_executable):
        
        return True,  os.path.join(bin_dir, 'chemem')
        
    return False, None

def get_chemem_paths():
    conda_base_path = find_anaconda_path()
    if not conda_base_path:
        print("Anaconda installation not found.")
        return []

    env_paths = get_conda_env_paths(conda_base_path)
    
    all_paths = []
    for env_path in env_paths:
        found, chemem_executable = check_chemem_version(env_path)
        if found:
            all_paths.append(chemem_executable)
            
    return [PathParameter(i,i) for i in all_paths]

def generate_unique_id():
    return str(uuid.uuid4())

def point_to_voxel_index(point, apix, origin):
    """
    Convert a 3D point in Angstroms to voxel indices.
    
    :param point: Tuple of the 3D point coordinates (x, y, z) in Angstroms.
    :param apix: Pixel size in Angstroms.
    :param origin: Origin coordinates (x, y, z).
    :return: Tuple of voxel indices (ix, iy, iz).
    """
    ix = int((point[0] - origin[0]) / apix[0])
    iy = int((point[1] - origin[1]) / apix[1])
    iz = int((point[2] - origin[2]) / apix[2])
    return ix, iy, iz

def find_voxel_indices(min_point, max_point, apix, grid_size, origin):
    """
    Determine the minimum and maximum voxel indices within given 3D points.
    
    :param min_point: Tuple of the minimum 3D point coordinates (x, y, z) in Angstroms.
    :param max_point: Tuple of the maximum 3D point coordinates (x, y, z) in Angstroms.
    :param apix: Pixel size in Angstroms.
    :param grid_size: Array of grid sizes [nx, ny, nz].
    :param origin: Origin coordinates (x, y, z).
    :return: Two tuples representing the minimum and maximum voxel indices ((min_ix, min_iy, min_iz), (max_ix, max_iy, max_iz)).
    """
    min_indices = point_to_voxel_index(min_point, apix, origin)
    max_indices = point_to_voxel_index(max_point, apix, origin)
    
    # Ensure indices are within grid size limits
    min_indices = (
        max(0, min_indices[0]), max(0, min_indices[1]), max(0, min_indices[2])
    )
    max_indices = (
        min(grid_size[0] - 1, max_indices[0]),
        min(grid_size[1] - 1, max_indices[1]),
        min(grid_size[2] - 1, max_indices[2])
    )
    
    return min_indices, max_indices

def get_box_vertices(centroid, box_size):
    """
    Calculate the vertices of a box given its center and dimensions.

    :param center: A tuple of (x, y, z) representing the center of the box.
    :param dimensions: A tuple of (length, width, height) of the box.
    :return: A list of tuples, each representing the coordinates of a vertex.
    """
    
    x, y, z = centroid
    xb, yb, zb = box_size

    # Half dimensions
    half_xb = xb / 2
    half_yb = yb / 2
    half_zb = zb / 2

    # Calculate the coordinates of the vertices
    vertices = [
        (x - half_xb, y - half_yb, z - half_zb),
        (x - half_xb, y - half_yb, z + half_zb),
        (x - half_xb, y + half_yb, z - half_zb),
        (x - half_xb, y + half_yb, z + half_zb),
        (x + half_xb, y - half_yb, z - half_zb),
        (x + half_xb, y - half_yb, z + half_zb),
        (x + half_xb, y + half_yb, z - half_zb),
        (x + half_xb, y + half_yb, z + half_zb)
    ]
    
    min_x_vertices = min([i[0] for i in vertices])
    max_x_vertices = max([i[0] for i in vertices])
    min_y_vertices = min([i[1] for i in vertices])
    max_y_vertices = max([i[1] for i in vertices])
    min_z_vertices = min([i[2] for i in vertices])
    max_z_vertices = max([i[2] for i in vertices])

    min_coords = np.array([min_x_vertices,min_y_vertices, min_z_vertices])
    max_coords = np.array([max_x_vertices,max_y_vertices,max_z_vertices])
    
    return min_coords, max_coords

def condense_path(path, max_length=25):
    if len(path) <= max_length:
        return path  

    part_length = (max_length - 3) // 2  # 3 characters are reserved for "..."
    start_part = path[:part_length]
    end_part = path[-part_length:]
    return f"{start_part}...{end_part}"
        
        
        