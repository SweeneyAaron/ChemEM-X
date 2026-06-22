#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:04:21 2025

@author: aaron.sweeney
"""

from chimerax.ChemEM.dock.tools import add_residue_to_model, get_atom_match_object
from chimerax.ChemEM.core.tools import mol_from_sdf, validate_ligand_file
from chimerax.ChemEM.simulate.simulation import Simulation, SimulationContainer, RESIDUE_NAMES
from chimerax.core.commands import run
from chimerax.core.tasks import  Task, TaskState
from chimerax.ChemEM.core.parameters import PathParameter
from openmm import unit
import datetime
import json
import os
import numpy as np
from chimerax.atomic import Atoms
import time
from chimerax.ChemEM.simulate import torsion as _torsion


def include_tracked_ligand_in_simulation(chemem, ligand_id):
    """Add a tracked ligand to the simulation's 'Ligands' inclusion list and
    surface it in the 'Ligands in Simulation' UI list. Shared by the Build-tab
    tracked-ligand picker and the docked-solution 'Add' action so both
    pathways behave identically. No-op (returns False) if the ligand isn't
    tracked or is already included."""
    ligand_id = str(ligand_id).strip()
    record = (getattr(chemem, '_tracked_ligands', {}) or {}).get(ligand_id)
    if record is None:
        return False
    path = record.get('last_saved_path') or record.get('source_sdf_path')
    if not path:
        return False

    existing = chemem.simulation_parameters.get_parameter('Ligands') or []
    if any(getattr(p, 'name', None) == ligand_id for p in existing):
        return False  # already included

    chemem.simulation_parameters.add_list_parameter('Ligands', PathParameter(ligand_id, path))

    ligand_data = {'id': ligand_id,
                   'value': os.path.basename(path),
                   'class': 'PathParameter'}
    chemem.run_js_code(f'addLigandToSimulationList({json.dumps(ligand_data)});')
    return True

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


def mirror_selection_to_model(source_model, dest_model):
    """Copy per-atom `selected` state from source_model onto dest_model,
    matching atoms by (chain_id, residue number, insertion code, atom name).

    ChimeraX AtomicStructure.copy() does NOT preserve selection, so a
    ligand-stripped export copy starts unselected. This re-applies the source
    selection so a subsequent `save ... selectedOnly true` captures the
    selected (and anchor) residues instead of an empty PDB. Selected ligand
    atoms simply have no counterpart in the stripped copy and are excluded
    (ligands are exported separately). Returns the number of atoms selected."""
    def key(a):
        r = a.residue
        return (r.chain_id, r.number, getattr(r, 'insertion_code', '') or '', a.name)

    sel_keys = {key(a) for a in source_model.atoms if a.selected}
    if not sel_keys:
        return 0
    n = 0
    for a in dest_model.atoms:
        if key(a) in sel_keys:
            a.selected = True
            n += 1
    return n


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


#---------------------------------------------------------------------------
# Interactive-restraint selection helpers.
#
# Each reads the user's current atom selection on the simulation model and
# maps the relevant atoms to their OpenMM indices via atoms_to_index
# (chimerax atom -> openmm index). They return (indices, atoms[, ...]) on
# success and Nones on any failure so the calling command can bail cleanly.
# RESIDUE_NAMES (the 20 standard amino acids, from simulation.py) is used to
# tell protein atoms from ligand atoms.
#---------------------------------------------------------------------------

def find_cation_atom(atoms_list):
    """Return the first cationic atom in a list (protein basic groups or a
    positively-charged / quaternary ligand nitrogen), else None."""
    for atom in atoms_list:
        residue = atom.residue
        if residue.name == 'LYS' and atom.name == 'NZ':
            return atom
        elif residue.name == 'ARG' and atom.name in ['NE', 'NH1', 'NH2']:
            return atom
        elif residue.name == 'HIS' and atom.name in ['ND1', 'NE2']:
            return atom
        if atom.element.name == 'N' and atom.formal_charge == 1:
            return atom
        if atom.element.name == 'N' and len(atom.bonds) == 4:
            return atom
    return None


def get_cation_pi_tug_indexes(model, atoms_to_index):
    selected_atoms = [atom for residue in model.residues for atom in residue.atoms if atom.selected]

    if len(selected_atoms) <= 1:
        return None, None, None, None  # need at least two atoms selected

    protein_atoms = [atom for atom in selected_atoms if atom.residue.name in RESIDUE_NAMES]
    ligand_atoms = [atom for atom in selected_atoms if atom.residue.name in ['LIG', 'UNL']]

    ring_atoms = []
    cation_atom = None

    protein_rings = [i.rings() for i in protein_atoms if len(i.rings()) > 0]
    protein_rings = list(set([i for j in protein_rings for i in j]))

    ligand_rings = [i.rings() for i in ligand_atoms if len(i.rings()) > 0]
    ligand_rings = list(set([i for j in ligand_rings for i in j]))

    if protein_rings:
        # Ring is in the protein, cation is in the ligand
        ring_atoms = []
        for ring in protein_rings:
            for atom in ring.atoms:
                ring_atoms.append(atom)
        cation_atom = find_cation_atom(ligand_atoms)
        if cation_atom is None:
            return None, None, None, None
    elif ligand_rings:
        # Ring is in the ligand, cation is in the protein
        ring_atoms = []
        for ring in ligand_rings:
            for atom in ring.atoms:
                ring_atoms.append(atom)
        cation_atom = find_cation_atom(protein_atoms)
        if cation_atom is None:
            return None, None, None, None
    else:
        return None, None, None, None  # no aromatic rings in selection

    ring_atom_indices = sorted([atoms_to_index[atom] for atom in ring_atoms])
    cation_atom_index = atoms_to_index.get(cation_atom)

    if cation_atom_index is None:
        return None, None, None, None

    return ring_atoms, ring_atom_indices, cation_atom, cation_atom_index


def is_protein_cationic_atom(atom):
    if atom.residue.name == 'LYS' and atom.name == 'NZ':
        return True
    elif atom.residue.name == 'ARG' and atom.name in ['NE', 'NH1', 'NH2']:
        return True
    elif atom.residue.name == 'HIS' and atom.name in ['ND1', 'NE2']:
        return True
    return False


def is_protein_anionic_atom(atom):
    if atom.residue.name == 'ASP' and atom.name in ['OD1', 'OD2']:
        return True
    elif atom.residue.name == 'GLU' and atom.name in ['OE1', 'OE2']:
        return True
    return False


def is_ligand_cationic_atom(atom):
    if atom.element.name == 'N':
        if len(atom.bonds) >= 4:
            return True
    return False


def is_ligand_anionic_atom(atom):
    if atom.element.name == 'O':
        if len(atom.bonds) <= 1:
            return True
    return False


def get_saltbridge_tug_index(model, atoms_to_index):
    selected_atoms = [atom for residue in model.residues for atom in residue.atoms if atom.selected]
    if len(selected_atoms) != 2:
        return None, None

    cation = None
    anion = None

    protein_atoms = [atom for atom in selected_atoms if atom.residue.name in RESIDUE_NAMES]
    ligand_atoms = [atom for atom in selected_atoms if atom.residue.name in ['LIG', 'UNL']]

    for atom in protein_atoms:
        if is_protein_cationic_atom(atom):
            if cation is not None:
                return None, None
            cation = atom
        elif is_protein_anionic_atom(atom):
            if anion is not None:
                return None, None
            anion = atom

    for atom in ligand_atoms:
        if is_ligand_cationic_atom(atom):
            if cation is not None:
                return None, None
            cation = atom
        elif is_ligand_anionic_atom(atom):
            if anion is not None:
                return None, None
            anion = atom

    if anion is not None and cation is not None:
        try:
            return [[atoms_to_index[cation], atoms_to_index[anion]], [cation, anion]]
        except Exception as e:
            print('SALTBRIDGE EXCEPTION:', e)
            return None, None
    else:
        return None, None


def get_weak_hbond_tug_index(model, atoms_to_index):
    selected_atoms = [atom for residue in model.residues for atom in residue.atoms if atom.selected]

    if len(selected_atoms) != 3:
        return None, None

    hydrogen = next((atom for atom in selected_atoms if atom.element.name == 'H'), None)
    if not hydrogen:
        return None, None

    non_hydrogens = [atom for atom in selected_atoms if atom != hydrogen]
    if len(non_hydrogens) != 2:
        return None, None

    donor = next((atom for atom in hydrogen.neighbors if atom in non_hydrogens), None)
    if not donor or donor.element.name != 'C':
        return None, None  # weak H-bond donor must be carbon

    acceptor = next((atom for atom in non_hydrogens if atom != donor), None)
    if not acceptor or acceptor.element.name not in ['O', 'N']:
        return None, None

    try:
        return [atoms_to_index[donor], atoms_to_index[hydrogen], atoms_to_index[acceptor]], [donor, hydrogen, acceptor]
    except KeyError:
        return None, None


def get_hbond_tug_index(model, atoms_to_index):
    selected_atoms = [atom for residue in model.residues for atom in residue.atoms if atom.selected]

    if len(selected_atoms) != 3:
        return None, None

    hydrogen = next((atom for atom in selected_atoms if atom.element.name == 'H'), None)
    if not hydrogen:
        return None, None

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
        return [atoms_to_index[donor], atoms_to_index[hydrogen], atoms_to_index[acceptor]], [donor, hydrogen, acceptor]
    except KeyError:
        return None, None


def get_halogen_bond_tug_index(model, atoms_to_index):
    selected_atoms = [atom for residue in model.residues for atom in residue.atoms if atom.selected]

    if len(selected_atoms) != 2:
        return None, None, None, None

    halogen = next((atom for atom in selected_atoms if atom.element.name in ['F', 'I', 'Br', 'Cl']), None)
    if not halogen:
        return None, None, None, None

    acceptor = [atom for atom in selected_atoms if atom != halogen and atom.element.name in ['N', 'O', 'S']]
    if not acceptor:
        return None, None, None, None
    elif len(acceptor) > 1:
        return None, None, None, None

    acceptor = acceptor[0]

    # atom bonded to the halogen (its root)
    atoms_bonded_to_halogen = []
    for bond in halogen.bonds:
        for a in bond.atoms:
            if a != halogen:
                atoms_bonded_to_halogen.append(a)

    if len(atoms_bonded_to_halogen) != 1:
        return None, None, None, None

    halogen_root_atom = atoms_bonded_to_halogen[0]

    atoms_bonded_to_acceptor = []
    for bond in acceptor.bonds:
        for a in bond.atoms:
            if a != acceptor:
                atoms_bonded_to_acceptor.append(a)

    n_bonded_atoms = len(atoms_bonded_to_acceptor)
    element = acceptor.element.name

    # assign acceptor hybridisation -> ideal donor-acceptor-root angle
    if element == 'O':
        if n_bonded_atoms == 1:
            theta2_0 = 109.5
        elif n_bonded_atoms == 2:
            theta2_0 = 120.0
        else:
            return None, None, None, None
    elif element == 'N':
        if n_bonded_atoms == 2:
            theta2_0 = 120.0
        elif n_bonded_atoms == 3:
            theta2_0 = 109.5
        elif n_bonded_atoms == 1:
            theta2_0 = 180.0
        else:
            return None, None, None, None
    elif element == 'S':
        if n_bonded_atoms == 2:
            theta2_0 = 109.5
        elif n_bonded_atoms == 1:
            theta2_0 = 120.0
        else:
            return None, None, None, None
    else:
        return None, None, None, None

    acceptor_root_atom = atoms_bonded_to_acceptor[0]

    HALOGEN_DISTANCE = {'I': 3.0, 'Br': 2.9, 'Cl': 2.8, 'F': 2.7}
    halogen_distance = HALOGEN_DISTANCE[halogen.element.name]

    try:
        return [[atoms_to_index[halogen_root_atom],
                 atoms_to_index[halogen],
                 atoms_to_index[acceptor],
                 atoms_to_index[acceptor_root_atom]],
                [halogen_root_atom, halogen, acceptor, acceptor_root_atom],
                theta2_0,
                halogen_distance]
    except Exception as e:
        print('HALOGEN EXCEPTION: ', e)
        return None, None, None, None


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
                 chemem = None,
                 #need to update the model somehow!!
                 job_type = SIMULATION_JOB):
        super().__init__(session)
        self.simulation_model = simulation_model
        self.atoms_to_position_index = atoms_to_position_index
        self.atoms_to_position_index_as_dic = {a: i for a, i in atoms_to_position_index}
        self.simulation = simulation
        #back-ref to the CHEMEM tool instance so the worker thread can marshal
        #GUI updates via session.ui.thread_safe(self._chemem.run_js_code, js)
        self._chemem = chemem

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

        # --- Save points (Feature 1) -------------------------------------
        # In-memory only; bound to this job (and therefore this OpenMM
        # context). Captured/restored ONLY on the worker thread via flags.
        self.save_points = []          # list of snapshot dicts (see _capture_snapshot)
        self._next_save_id = 0
        self.save_snapshot = None       # set to a label str -> capture current positions
        self.restore_to = None          # set to a snapshot id -> restore those positions

        # --- Torsion analysis (Feature 2) --------------------------------
        # Per-torsion scan cache shared by minima-jumping and strain colour.
        self.torsion_cache = {}         # key -> {spec, profile, minima, current_idx}
        self._torsion_bridge = {}       # ligand_id -> bidirectional name bridge (cached)
        self.torsion_sync = None        # truthy -> (re)resolve current selection into rows
        self.torsion_step = None        # (key, direction) -> auto-scan + jump that torsion
        self.torsion_show = None        # key -> (scan-if-needed) push its profile to the plot
        self.torsion_color = None       # "local" | "global" -> colour ligands by strain
        self.torsion_color_clear = None # truthy -> restore element colours
        self.torsion_flip = None        # key -> toggle which side of the bond rotates
        self._torsion_flip = {}         # key -> bool (True = rotate the larger side)
        self.torsion_scan_step = 10     # degrees per scan increment
        self._active_specs = {}         # key -> resolved torsion spec (OpenMM index space)
        #Clean intrinsic torsion profiles exported by the ChemEM backend, indexed by
        #frozenset of the four atom names. None until _load_imported_torsion_profiles runs.
        self._imported_torsions = None
        self._imported_notice_shown = False   # one-time "using live scan" notice guard
        #key -> CustomTorsionForce term index for the restraint-drive (Part C).
        self._torsion_drive_terms = {}

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

        #Auto save-point of the positions ENTERING minimisation. Must run AFTER the
        #consistency sync above (which calls setPositions) and BEFORE minimise below,
        #so the user can always revert to the un-minimised starting configuration.
        self._capture_snapshot('auto-min', 'Pre-minimise (start)')

        #Build the RDKit<->simulation-model atom bridge for torsion analysis now,
        #while the tracked-ligand RDKit pose and the freshly-synced simulation
        #model still coincide spatially. The bridge is identity-based so it stays
        #valid for the rest of the session even as dynamics moves the atoms.
        try:
            self._build_torsion_bridges()
            self._build_torsion_drive_terms()
        except Exception:
            import traceback
            traceback.print_exc()

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

            #--- Save-point + torsion requests --------------------------------
            #These touch the OpenMM context, so they MUST run on this worker
            #thread (the context is not thread-safe). They are hoisted ABOVE the
            #pause guard so Save / Revert / torsion analysis work while paused.
            #Wrapped so a handler error cannot silently kill the loop thread.
            try:
                self._service_save_points()
                self._service_torsion_requests()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._notify_user(f'Simulation request failed: {e}')

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
                #auto save-point of the positions entering this minimisation
                self._capture_snapshot('auto-min', 'Pre-minimise')
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

    # =====================================================================
    # Worker-thread helpers shared by both features
    # =====================================================================
    def _run_js(self, js):
        """Marshal a runJavaScript call from this worker thread onto the Qt UI
        thread (runJavaScript must not be called off the GUI thread)."""
        if self._chemem is None:
            return
        try:
            self.session.ui.thread_safe(self._chemem.run_js_code, js)
        except Exception:
            import traceback
            traceback.print_exc()

    def _notify_user(self, message):
        safe = str(message).replace('\\', '\\\\').replace('"', '\\"')
        self._run_js(
            'if (typeof showSimulationStatus === "function") {{ showSimulationStatus("{0}"); }}'
            ' else {{ console.warn("{0}"); }}'.format(safe)
        )

    def _num_particles(self):
        return self.simulation.simulation.system.getNumParticles()

    # =====================================================================
    # Save points (Feature 1)
    # =====================================================================
    def _service_save_points(self):
        if self.save_snapshot is not None:
            label = self.save_snapshot
            self.save_snapshot = None
            self._capture_snapshot('manual', label)
        if self.restore_to is not None:
            snap_id = self.restore_to
            self.restore_to = None
            self._restore_snapshot(snap_id)

    def _capture_snapshot(self, kind, label):
        ctx = self.simulation.simulation.context
        state = ctx.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)   # nm Quantity (exactly what setPositions wants)
        snap = {
            'id': self._next_save_id,
            'label': label,
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S  %d-%b'),
            'kind': kind,
            'positions': positions,
            'n': self._num_particles(),
        }
        self._next_save_id += 1
        self.save_points.append(snap)
        self._notify_save_points()

    def _restore_snapshot(self, snap_id):
        snap = next((s for s in self.save_points if s['id'] == snap_id), None)
        if snap is None:
            return
        ctx = self.simulation.simulation.context
        if snap['n'] != self._num_particles():
            self._notify_user('Cannot revert: the system size changed since this '
                              'save point (e.g. a ligand was added).')
            return
        ctx.setPositions(snap['positions'])
        #Stale velocities from a different configuration explode on the next
        #step; re-seed to the live target temperature.
        try:
            ctx.setVelocitiesToTemperature(self.simulation.temperature * unit.kelvin)
        except Exception:
            import traceback
            traceback.print_exc()
        self.update_simulation_model_vectorised()
        self._notify_save_points(active=snap_id)

    def _notify_save_points(self, active=None):
        payload = [
            {'id': s['id'], 'label': s['label'],
             'timestamp': s['timestamp'], 'kind': s['kind']}
            for s in self.save_points
        ]
        active_js = 'null' if active is None else int(active)
        self._run_js('if (typeof renderSavePoints === "function") '
                     'renderSavePoints({0}, {1});'.format(json.dumps(payload), active_js))

    # =====================================================================
    # Torsion preferences (Feature 2)
    # =====================================================================
    def _build_torsion_bridges(self):
        """For each tracked ligand that is part of THIS simulation, build a
        stable identity bridge between RDKit atom indices and simulation-model
        atoms by matching coincident poses (element + nearest position). Called
        once at sim start while poses still coincide."""
        self._torsion_bridge = {}
        chemem = self._chemem
        if chemem is None:
            return
        tracked = getattr(chemem, '_tracked_ligands', None)
        if not tracked:
            return

        #Candidate atoms = everything that is NOT a standard amino acid (i.e.
        #ligands / hetero / waters / ions). The coincident-pose element+distance
        #match then picks out this ligand's own atoms exactly.
        sim_atoms = [a for a in self.simulation_model.atoms
                     if a.residue.name not in RESIDUE_NAMES]
        if not sim_atoms:
            return
        sim_coords = np.array([a.coord for a in sim_atoms])
        sim_elems = [a.element.name.upper() for a in sim_atoms]

        for ligand_id, record in tracked.items():
            matcher = record.get('atom_matcher') if isinstance(record, dict) else None
            mol = getattr(matcher, 'rdmol', None) if matcher is not None else None
            if mol is None:
                continue
            try:
                conf = mol.GetConformer()
            except Exception:
                continue

            rd_to_sim = {}
            sim_to_rd = {}
            used = set()
            ok = True
            for rd_idx in range(mol.GetNumAtoms()):
                elem = mol.GetAtomWithIdx(rd_idx).GetSymbol().upper()
                p = conf.GetAtomPosition(rd_idx)
                pv = np.array([p.x, p.y, p.z])
                best_j, best_d = None, None
                for j in range(len(sim_atoms)):
                    if j in used or sim_elems[j] != elem:
                        continue
                    d = np.linalg.norm(sim_coords[j] - pv)
                    if best_d is None or d < best_d:
                        best_d, best_j = d, j
                if best_j is None or best_d > 0.6:
                    #A heavy atom that can't be matched means this tracked
                    #ligand isn't part of THIS simulation (or the pose diverged)
                    #-> abort it. A stray hydrogen just stays unmapped (torsions
                    #that need it are dropped later); don't discard the ligand.
                    if mol.GetAtomWithIdx(rd_idx).GetAtomicNum() > 1:
                        ok = False
                        break
                    continue
                used.add(best_j)
                rd_to_sim[rd_idx] = sim_atoms[best_j]
                sim_to_rd[sim_atoms[best_j]] = rd_idx
            if not ok or not rd_to_sim:
                continue

            self._torsion_bridge[ligand_id] = {
                'rdmol': mol,
                'rd_to_sim': rd_to_sim,
                'sim_to_rd': sim_to_rd,
                'torsions': _torsion.find_rotatable_torsions(mol),
            }

        #Load the clean intrinsic torsion profiles exported by the ChemEM backend
        #(if present) now that the bridge exists. Runs on the worker thread at start.
        self._load_imported_torsion_profiles()

    def _load_imported_torsion_profiles(self):
        """Load <exported_dir>/torsion_profiles.json (written by the ChemEM backend
        export step) and index each entry by a frozenset of its four atom names for
        order-independent matching. Leaves an empty dict if the file is absent so the
        live-scan fallback kicks in."""
        self._imported_torsions = {}
        exported_dir = getattr(self.simulation, 'exported_dir', None)
        if not exported_dir:
            return
        path = os.path.join(exported_dir, 'torsion_profiles.json')
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                doc = json.load(f)
        except Exception:
            import traceback
            traceback.print_exc()
            return
        for entry in doc.get('torsions', []):
            names = entry.get('atom_names', [])
            self._imported_torsions[frozenset(str(n) for n in names)] = entry

    def _lookup_imported_profile(self, spec):
        """Return the imported profile entry for a torsion spec, or None.

        Primary match: exact set of the four atom names. Fallback: any entry whose
        atom-name set contains both central bond atoms (handles a different terminal
        neighbour choice between the backend's and the plugin's RDKit enumeration)."""
        if not self._imported_torsions:
            return None
        key = frozenset(spec.get('atom_names', ()))
        entry = self._imported_torsions.get(key)
        if entry is not None:
            return entry
        sa, sb = spec['bond_sim_atoms']
        bond_pair = {sa.name, sb.name}
        for names, entry in self._imported_torsions.items():
            if bond_pair.issubset(names):
                return entry
        return None

    def _imported_angle_offset(self, spec, entry):
        """Constant degrees offset between the backend profile's dihedral and this
        spec's dihedral (they may reference different terminal atoms), such that
        plugin_angle = backend_angle - offset. Returns 0.0 if it can't be determined.

        The exported prmtop keeps the protein+ligands concatenation order, so the
        entry's 'global_indices' live in the same OpenMM index space as spec['dihedral']
        -> both dihedrals can be read straight from the current context positions."""
        gidx = entry.get('global_indices')
        if not gidx or len(gidx) != 4:
            return 0.0
        try:
            _, xyz_A = self._read_positions_angstrom()
            n = len(xyz_A)
            g = [int(x) for x in gidx]
            if any(j < 0 or j >= n for j in g):
                return 0.0
            i, a, b, l = spec['dihedral']
            plugin_cur = _torsion.dihedral_deg(xyz_A[i], xyz_A[a], xyz_A[b], xyz_A[l])
            backend_cur = _torsion.dihedral_deg(xyz_A[g[0]], xyz_A[g[1]], xyz_A[g[2]], xyz_A[g[3]])
            if not (np.isfinite(plugin_cur) and np.isfinite(backend_cur)):
                return 0.0
            return float(backend_cur - plugin_cur)
        except Exception:
            return 0.0

    def _build_torsion_drive_terms(self):
        """Add one CustomTorsionForce term (drive_k=0) per bridgeable ligand torsion and
        map spec['key'] -> term index. Done once at job start so live stepping never has
        to add terms / reinitialize the context per torsion (just cheap parameter
        updates). The terms drive torsions during restraint-driven looping."""
        self._torsion_drive_terms = {}
        force = getattr(self.simulation, 'torsion_drive_force', None)
        if force is None:
            return
        added = False
        for ligand_id in self._torsion_bridge:
            for spec in self._torsion_specs_for_ligand(ligand_id):
                key = spec['key']
                if key in self._torsion_drive_terms:
                    continue
                i, a, b, l = spec['dihedral']
                #[k, theta0, cos_cutoff]; cos_cutoff=1.0 (cutoff 0) and k=0 => inert.
                term_idx = force.addTorsion(i, a, b, l, [0.0, 0.0, 1.0])
                self._torsion_drive_terms[key] = term_idx
                self._active_specs.setdefault(key, spec)
                added = True
        if added:
            #Adding terms changes the force layout -> reinitialize once (keep state).
            self.simulation.simulation.context.reinitialize(preserveState=True)

    def _torsion_specs_for_ligand(self, ligand_id):
        """Resolve a ligand's RDKit torsions into OpenMM-index specs. Any
        torsion with an atom that cannot be bridged to an OpenMM index is
        dropped (never push a partial set)."""
        br = self._torsion_bridge.get(ligand_id)
        if br is None:
            return []
        rd_to_sim = br['rd_to_sim']
        idx_map = self.atoms_to_position_index_as_dic
        specs = []
        for t in br['torsions']:
            i, a, b, l = t['atoms']
            try:
                sim_i, sim_a = rd_to_sim[i], rd_to_sim[a]
                sim_b, sim_l = rd_to_sim[b], rd_to_sim[l]
                omm_i, omm_a = idx_map[sim_i], idx_map[sim_a]
                omm_b, omm_l = idx_map[sim_b], idx_map[sim_l]
                moved_sim = [rd_to_sim[m] for m in t['moved_atoms']]
                moved_omm = [idx_map[s] for s in moved_sim]
            except KeyError:
                continue
            #Complementary side (the one held fixed by default). Be lenient: unbridged
            #atoms (e.g. stray hydrogens) are simply skipped - pinning the bridged
            #heavy atoms is enough to anchor the side.
            fixed_sim = [rd_to_sim[m] for m in t.get('fixed_atoms', []) if m in rd_to_sim]
            fixed_omm = [idx_map[s] for s in fixed_sim if s in idx_map]
            specs.append({
                'key': '{0}|{1}|{2}'.format(ligand_id, sim_a.name, sim_b.name),
                'ligand_id': ligand_id,
                'dihedral': (omm_i, omm_a, omm_b, omm_l),
                'moved': moved_omm,
                'fixed': fixed_omm,
                'label': '{0}–{1}'.format(sim_a.name, sim_b.name),
                'bond_sim_atoms': (sim_a, sim_b),
                'moved_sim_atoms': moved_sim,
                'fixed_sim_atoms': fixed_sim,
                #four atom names of the dihedral (i-a-b-l); robust join key with the
                #backend-exported torsion_profiles.json (which stores atom names).
                'atom_names': (sim_i.name, sim_a.name, sim_b.name, sim_l.name),
            })
        return specs

    def _service_torsion_requests(self):
        if self.torsion_sync is not None:
            self.torsion_sync = None
            rows = self._resolve_selected_torsions()
            if not rows:
                if not self._torsion_bridge:
                    self._notify_user('Torsion analysis needs a tracked SDF ligand in the simulation.')
                else:
                    self._notify_user('Select both atoms of a rotatable ligand bond, then use the selection.')
            self._push_torsion_rows(rows)
        if self.torsion_step is not None:
            key, direction = self.torsion_step
            self.torsion_step = None
            self._step_torsion(key, direction)
        if self.torsion_show is not None:
            key = self.torsion_show
            self.torsion_show = None
            self._scan_torsion(key)
            self._push_torsion_profile(key)
        if self.torsion_color is not None:
            mode = self.torsion_color
            self.torsion_color = None
            self._apply_torsion_color(mode)
        if self.torsion_color_clear is not None:
            self.torsion_color_clear = None
            self._clear_torsion_color()
        if self.torsion_flip is not None:
            key = self.torsion_flip
            self.torsion_flip = None
            self._torsion_flip[key] = not self._torsion_flip.get(key, False)
            #Re-render the row list so the button reflects the new moving side.
            self._push_torsion_rows(self._resolve_selected_torsions())

    def _moved_fixed_for(self, spec, key):
        """Return (moved_omm, fixed_omm) honouring the user's flip choice. Default:
        the smaller side (spec['moved']) rotates and the larger side is pinned; when
        flipped, the larger side rotates and the smaller side is pinned."""
        if self._torsion_flip.get(key, False):
            return spec.get('fixed', []), spec.get('moved', [])
        return spec.get('moved', []), spec.get('fixed', [])

    def _resolve_selected_torsions(self):
        selected = set(a for a in self.simulation_model.atoms if a.selected)
        try:
            for bond in self.simulation_model.bonds:
                if bond.selected:
                    selected.update(bond.atoms)
        except Exception:
            pass
        rows = []
        seen = set()
        for ligand_id in self._torsion_bridge:
            for spec in self._torsion_specs_for_ligand(ligand_id):
                sa, sb = spec['bond_sim_atoms']
                if sa in selected and sb in selected and spec['key'] not in seen:
                    seen.add(spec['key'])
                    self._active_specs[spec['key']] = spec
                    rows.append({'key': spec['key'], 'label': spec['label'],
                                 'ligand_id': str(ligand_id),
                                 'flipped': bool(self._torsion_flip.get(spec['key'], False)),
                                 'small_count': len(spec.get('moved', [])),
                                 'large_count': len(spec.get('fixed', []))})
        return rows

    def _read_positions_angstrom(self):
        """Current context positions as (nm Quantity array, angstrom ndarray)."""
        state = self.simulation.simulation.context.getState(getPositions=True)
        base_nm = state.getPositions(asNumpy=True)
        xyz_A = np.array(base_nm.value_in_unit(unit.nanometer)) * 10.0
        return base_nm, xyz_A

    def _scan_torsion(self, key):
        cache = self.torsion_cache.get(key)
        if cache and cache.get('profile'):
            return cache
        spec = self._active_specs.get(key)
        if spec is None:
            return None

        #Prefer the clean intrinsic profile exported by the ChemEM backend. The live
        #full-system scan (below) is clash-dominated and noisy; only use it as a
        #fallback for ligands with no exported profile (old exports / added ligands).
        entry = self._lookup_imported_profile(spec)
        if entry is not None:
            #The backend and the plugin may pick different terminal atoms for the same
            #rotatable bond, giving a constant angle offset between the backend profile
            #and this spec's dihedral. Shift the imported angles into the spec's
            #convention so the curve, the current-angle marker and the step targets all
            #line up (everything downstream uses spec['dihedral']).
            offset = self._imported_angle_offset(spec, entry)
            pts = []
            for a, e in entry.get('profile', []):
                pts.append(((float(a) - offset) % 360.0, float(e)))
            #A degenerate (flat) backend profile normalises to NaN/inf; fall back to the
            #live scan rather than feed non-finite energies into minima detection.
            if pts and all(np.isfinite(e) for _, e in pts):
                pts.sort(key=lambda p: p[0])
                angles = [p[0] for p in pts]
                energies = [p[1] for p in pts]
                cache = {
                    'label': spec['label'],
                    'angles': angles,
                    'energies': energies,
                    'profile': list(zip(angles, energies)),
                    'minima': _torsion.detect_minima(angles, energies),
                    'current_idx': None,
                }
                self.torsion_cache[key] = cache
                return cache

        if not self._imported_notice_shown:
            self._imported_notice_shown = True
            self._notify_user('No exported torsion profile found; using on-the-fly scan '
                              '(may look noisy). Re-export the simulation from ChemEM for '
                              'clean torsion-preference curves.')
        return self._scan_torsion_live(spec, key)

    def _scan_torsion_live(self, spec, key):
        ctx = self.simulation.simulation.context
        saved = ctx.getState(getPositions=True, getVelocities=True)
        base_nm = saved.getPositions(asNumpy=True)
        base_nm_raw = np.array(base_nm.value_in_unit(unit.nanometer))
        xyz_A = base_nm_raw * 10.0
        i, a, b, l = spec['dihedral']
        moved = spec['moved']
        step = max(1, int(self.torsion_scan_step))
        angles = list(range(0, 360, step))
        cur = _torsion.dihedral_deg(xyz_A[i], xyz_A[a], xyz_A[b], xyz_A[l])
        energies = []
        try:
            for ang in angles:
                rot_A = _torsion.rotate_points(xyz_A, a, b, moved, ang - cur)
                pos_nm = np.array(base_nm_raw)
                pos_nm[moved] = rot_A[moved] * 0.1   # angstrom -> nm
                ctx.setPositions(pos_nm * unit.nanometer)
                e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole)
                energies.append(e)
        finally:
            #restore the live trajectory bit-for-bit
            ctx.setPositions(saved.getPositions())
            ctx.setVelocities(saved.getVelocities())
        emin = min(energies)
        rel = [e - emin for e in energies]          #report RELATIVE energy (C5)
        cache = {
            'label': spec['label'],
            'angles': angles,
            'energies': rel,
            'profile': list(zip(angles, rel)),
            'minima': _torsion.detect_minima(angles, rel),
            'current_idx': None,
        }
        self.torsion_cache[key] = cache
        return cache

    @staticmethod
    def _ang_diff(x, y):
        return (x - y + 180.0) % 360.0 - 180.0

    def _next_minimum(self, minima_angles, cur, direction):
        if not minima_angles:
            return cur
        if direction > 0:
            cand = sorted(((ma - cur) % 360.0, ma) for ma in minima_angles)
        else:
            cand = sorted(((cur - ma) % 360.0, ma) for ma in minima_angles)
        for d, ma in cand:
            if d > 1e-6:
                return ma
        return cand[0][1]

    @staticmethod
    def _ang_diff_to_180(deg):
        """Map an angle in degrees to (-180, 180], matching OpenMM CustomTorsionForce
        theta range so theta0 is the SHORT-way target."""
        return (float(deg) + 180.0) % 360.0 - 180.0

    #--- Tunables for torsion flipping (one place) ----------------------------
    PIN_K = 20000.0     # kJ/mol/nm^2 ; firmly holds the fixed side (cf. 5000 restraints)
    K_LIVE = 1500.0     # kJ/mol/rad^2 ; gentle ISOLDE-ish live nudge bias
    K_HOLD = 5000.0     # kJ/mol/rad^2 ; strong hold while minimising the guaranteed flip
    MIN_ITERS = 80      # minimiser iterations for the guaranteed flip (local-ish)

    def _basin_tol_deg(self, key, target_deg, default=30.0, floor=10.0):
        """Half-width (deg) of the target minimum's basin: ~30 deg, but capped at 45%
        of the angular gap to the nearest OTHER minimum so we never accept a pose that
        is really closer to a different well."""
        cache = self.torsion_cache.get(key)
        if not cache or not cache.get('minima'):
            return default
        nearest = None
        for m in cache['minima']:
            d = abs(self._ang_diff(target_deg, m['angle']))
            if d > 1e-3 and (nearest is None or d < nearest):
                nearest = d
        if nearest is None:
            return default
        return max(floor, min(default, 0.45 * nearest))

    def _drive_torsion_to(self, spec, key, target_deg):
        """Move a torsion to target_deg while dynamics keeps running.

        Hybrid (ISOLDE-informed): pin the side the user chose to hold fixed, then try a
        short gentle live restraint nudge; if the bond doesn't reach the target BASIN in
        that window, fall back to a guaranteed micro-pause (rigid-rotate the chosen side
        to the target, hold it there with a strong restraint, energy-minimise to relieve
        the clash, reseed velocities). Either way it never snaps back and never explodes.
        Steps internally (like the tug handlers), so it works running OR paused."""
        term_idx = self._torsion_drive_terms.get(key)
        moved_omm, fixed_omm = self._moved_fixed_for(spec, key)
        if term_idx is None:
            #No pre-built driver term (e.g. ligand added after job start) -> safe fallback.
            return self._apply_torsion_rigid(spec, target_deg, moved_omm)

        theta0 = np.radians(self._ang_diff_to_180(target_deg))
        dih = spec['dihedral']
        tol_deg = self._basin_tol_deg(key, target_deg)
        cutoff_rad = np.radians(tol_deg)

        pinned = list(fixed_omm)
        try:
            if pinned:
                self.simulation.pin_atoms_to_current(pinned, self.PIN_K)
            reached = self._live_nudge_torsion(term_idx, dih, theta0, target_deg,
                                               tol_deg, cutoff_rad)
            if not reached:
                self._guaranteed_flip(spec, term_idx, theta0, target_deg, moved_omm)
        finally:
            #Always release the bias + pins so they never linger on the trajectory.
            self.simulation.release_torsion(term_idx, dih)
            if pinned:
                self.simulation.unpin_atoms(pinned)
            self.update_simulation_model_vectorised()

    def _live_nudge_torsion(self, term_idx, dih, theta0, target_deg, tol_deg, cutoff_rad):
        """Phase 1: gentle live restraint nudge. Ramp a flat-bottom cosine bias up and
        step; succeed as soon as the dihedral is within the target BASIN (tol_deg) - the
        flat-bottom means it then relaxes into the well on its own (no snap-back)."""
        i, a, b, l = dih
        RAMP_BLOCKS = 4
        STEPS_PER_BLOCK = 10
        TOTAL_BLOCKS = 30           # ~0.3 ps incl. ramp
        for nblock in range(1, TOTAL_BLOCKS + 1):
            k = self.K_LIVE * min(1.0, nblock / float(RAMP_BLOCKS))
            self.simulation.drive_torsion(term_idx, dih, theta0, k, cutoff_rad)
            self.simulation.step(STEPS_PER_BLOCK)
            self.update_simulation_model_vectorised()
            _, xyz_A = self._read_positions_angstrom()
            cur = _torsion.dihedral_deg(xyz_A[i], xyz_A[a], xyz_A[b], xyz_A[l])
            if abs(self._ang_diff(target_deg, cur)) <= tol_deg:
                return True
        return False

    def _guaranteed_flip(self, spec, term_idx, theta0, target_deg, moved_omm):
        """Phase 2: guaranteed micro-pause. Rigidly rotate the chosen side onto the
        target dihedral, hold it there with a strong restraint while the fixed side stays
        pinned, energy-minimise to clear the rigid-rotation clashes (so the next MD step
        cannot explode), then reseed velocities. Minimisation can't snap back over the
        barrier because the restraint holds the target."""
        dih = spec['dihedral']
        self._rigid_rotate_positions(spec, target_deg, moved_omm)
        self.simulation.drive_torsion(term_idx, dih, theta0, self.K_HOLD, 0.0)
        self.simulation.minimise_system(maxIterations=self.MIN_ITERS)
        self.simulation.reseed_velocities()
        self.update_simulation_model_vectorised()

    def _rigid_rotate_positions(self, spec, target_deg, moved_omm):
        """Rigidly rotate `moved_omm` about the bond axis so the dihedral becomes
        target_deg, and write the new positions (no velocity reseed / model update - the
        caller does that). The sign of the rotation depends on WHICH side moves: rotating
        the l-side changes the dihedral by +delta, the i-side by -delta."""
        ctx = self.simulation.simulation.context
        base_nm, xyz_A = self._read_positions_angstrom()
        base_nm_raw = np.array(base_nm.value_in_unit(unit.nanometer))
        i, a, b, l = spec['dihedral']
        moved = list(moved_omm) if moved_omm else list(spec['moved'])
        cur = _torsion.dihedral_deg(xyz_A[i], xyz_A[a], xyz_A[b], xyz_A[l])
        sign = 1.0 if l in moved else (-1.0 if i in moved else 1.0)
        delta = sign * self._ang_diff(target_deg, cur)   # short-way rotation
        rot_A = _torsion.rotate_points(xyz_A, a, b, moved, delta)
        pos_nm = base_nm_raw
        pos_nm[moved] = rot_A[moved] * 0.1               # angstrom -> nm
        ctx.setPositions(pos_nm * unit.nanometer)

    def _apply_torsion_rigid(self, spec, target_angle, moved_omm=None):
        """Last-resort rigid flip for torsions with no pre-built driver term (e.g. a
        ligand added mid-session). Rotates the chosen side onto the target and reseeds
        velocities. Less safe than the driver path (no minimise), but the only option
        without a driver term."""
        moved = moved_omm if moved_omm is not None else spec['moved']
        self._rigid_rotate_positions(spec, target_angle, moved)
        self.simulation.reseed_velocities()
        self.update_simulation_model_vectorised()

    def _step_torsion(self, key, direction):
        cache = self._scan_torsion(key)
        spec = self._active_specs.get(key)
        if cache is None or spec is None or not cache['minima']:
            return
        _, xyz_A = self._read_positions_angstrom()
        i, a, b, l = spec['dihedral']
        cur = _torsion.dihedral_deg(xyz_A[i], xyz_A[a], xyz_A[b], xyz_A[l])
        minima_angles = [m['angle'] for m in cache['minima']]
        target = self._next_minimum(minima_angles, cur, direction)
        self._drive_torsion_to(spec, key, target)
        cache['current_idx'] = int(np.argmin(
            [abs(self._ang_diff(target, ma)) for ma in minima_angles]))
        self._push_torsion_profile(key)

    def _strain_color(self, dev, maxdev=60.0):
        t = max(0.0, min(1.0, abs(dev) / maxdev))
        return [int(255 * t), int(255 * (1.0 - t)), 60, 255]

    def _apply_torsion_color(self, mode):
        specs = []
        for ligand_id in self._torsion_bridge:
            specs.extend(self._torsion_specs_for_ligand(ligand_id))
        if not specs:
            self._notify_user('No bridgeable ligand torsions to colour.')
            return
        _, xyz_A = self._read_positions_angstrom()
        for spec in specs:
            key = spec['key']
            self._active_specs.setdefault(key, spec)
            cache = self._scan_torsion(key)
            if cache is None or not cache['minima']:
                continue
            i, a, b, l = spec['dihedral']
            cur = _torsion.dihedral_deg(xyz_A[i], xyz_A[a], xyz_A[b], xyz_A[l])
            minima_angles = [m['angle'] for m in cache['minima']]
            if mode == 'global':
                ref = min(cache['minima'], key=lambda m: m['energy'])['angle']
                dev = abs(self._ang_diff(cur, ref))
            else:
                dev = min(abs(self._ang_diff(cur, ma)) for ma in minima_angles)
            rgba = self._strain_color(dev)
            for atom in [spec['bond_sim_atoms'][0], spec['bond_sim_atoms'][1]] + spec['moved_sim_atoms']:
                try:
                    atom.color = rgba
                except Exception:
                    pass

    def _clear_torsion_color(self):
        from chimerax.atomic.colors import element_color
        for br in self._torsion_bridge.values():
            for sim_atom in br['rd_to_sim'].values():
                try:
                    sim_atom.color = element_color(sim_atom.element.number)
                except Exception:
                    pass

    def _push_torsion_rows(self, rows):
        self._run_js('if (typeof renderSelectedTorsions === "function") '
                     'renderSelectedTorsions({0});'.format(json.dumps(rows)))

    def _push_torsion_profile(self, key):
        cache = self.torsion_cache.get(key)
        if cache is None:
            return
        current_angle = None
        spec = self._active_specs.get(key)
        if spec is not None:
            try:
                _, xyz_A = self._read_positions_angstrom()
                i, a, b, l = spec['dihedral']
                current_angle = _torsion.dihedral_deg(
                    xyz_A[i], xyz_A[a], xyz_A[b], xyz_A[l]) % 360.0
            except Exception:
                current_angle = None
        payload = {
            'key': key,
            'label': cache['label'],
            'points': [[float(an), float(en)] for an, en in zip(cache['angles'], cache['energies'])],
            'minima': cache['minima'],
            'current_idx': cache['current_idx'],
            'current_angle': current_angle,
        }
        self._run_js('if (typeof renderTorsionProfile === "function") '
                     'renderTorsionProfile({0});'.format(json.dumps(payload)))

    def on_finish(self):
        """Callback method executed after task thread terminates.

        This callback is executed in the UI thread after the
        :py:meth:`run` method returns.  By default, it does nothing.

        """
        self.terminate()
       

    