import parmed
from openmm import XmlSerializer
from openmm import LangevinIntegrator, Platform
from openmm import app
from openmm import unit
from openmm import MonteCarloBarostat, XmlSerializer, app, unit,CustomNonbondedForce, CustomCompoundBondForce, CustomBondForce, CustomAngleForce, Continuous3DFunction, vec3, Vec3, CustomCentroidBondForce, PeriodicTorsionForce, CustomTorsionForce
from openmm import LocalEnergyMinimizer
from scipy.ndimage import gaussian_filter
from openmm.unit.quantity import Quantity
import os
import numpy as np

from chimerax.atomic.structure import AtomicStructure
from chimerax.geometry import distance as get_distance
from chimerax.dssp import compute_ss
from rdkit import Chem
from openmm.app import  NoCutoff, HBonds
from chimerax.core.commands import run 
import math


from openmm import NonbondedForce
from parmed import Atom, Residue, AtomType

RESIDUE_NAMES = ['CYS','MET','GLY','ASP','ALA','VAL','PRO','PHE','ASN','THR',
                 'HIS','GLN','ARG','TRP','ILE','SER','LYS','LEU','GLU','TYR']

BACKBONE_ATOM_NAMES = ['CA', 'C', 'O', 'N']

HBOND_ELEMENTS = ['O', 'N', 'S']


class SimulationContainer:
    
    def __init__(self,
                 simulation,
                 simulation_model,
                 atoms_to_position_index,
                 parameters,
                 current_map,
                 ):
        
        self.simulation = simulation 
        self.simulation_model = simulation_model 
        self.atoms_to_position_index = atoms_to_position_index 
        #k = chimerax atom, v = openmm idx
        self.atoms_to_position_index_as_dic = {i[0]:i[1] for i in atoms_to_position_index}
        self.selected_residues = parameters.get_parameter('simulateSelectedAtoms').value
        self.current_map = current_map
        self.parameters = parameters
        self.js_code = ['openTab(event, "RunSimulationTab");']
        self.current_restraints = {}
        self.simulation_job_id = None
    
    def setup_simulation(self):
        self.set_groups()
        self.set_force_groups()
        self.set_map_force()
        self.set_initial_restraints()
        
    
    def set_groups(self):
        self.simulation.set_ring_groups(self.simulation_model, self.atoms_to_position_index_as_dic)
        self.simulation.set_SSE_elements(self.simulation_model, self.atoms_to_position_index_as_dic)
        if self.selected_residues:
            self.simulation.set_anchor_residues(self.simulation_model, 
                                                #TODO!! add selected_atom_data from thing to parameters
                                           self.parameters.parameters['selected_atom_data'],
                                           self.atoms_to_position_index_as_dic)
        
    def set_force_groups(self):
        self.simulation.set_force_groups()
    
    def set_map_force(self):
        if self.current_map is not None:
            self.simulation.add_force(MapBias)
    
    def set_initial_restraints(self):
        '''
        constrainProtein 0
        constrainProteinBackbone 0
        constrainProteinSidechain 0
        constrainLigandAtoms 0
        constrainSSE 0
        constrainHelices 0
        constrainSheets 0
        constrainProtein 0
        constrainProteinBackbone 0
        constrainProteinSidechain 0
        constrainLigandAtoms 0
        constrainSSE 0
        constrainHelices 0
        constrainSheets 0
        '''
        restraints_list = self.parameters.get_parameter('InitialConstraints')
        if restraints_list is not None:
            for rest in restraints_list:
                setattr(self, rest.name, rest.value)
        #this is not complete!!!
        if self.constrainProteinBackbone or self.constrainProtein:
            self.simulation.add_force(ConstrainBackBone) 
            restraint_id = len(self.current_restraints)
            message = 'Protein BackBone Heavy Atoms'
            js_code = f'addRestraintToList("{message}", "{restraint_id}", "ConstrainBackBone");'
            self.js_code.append(js_code)
            self.current_restraints[restraint_id] = (message, restraint_id)
        
        if self.constrainProteinSidechain or self.constrainProtein:
            self.simulation.add_force(ConstrainSideChain) 
            restraint_id = len(self.current_restraints)
            message = 'Protein Sidechain Heavy Atoms'
            js_code = f'addRestraintToList("{message}", "{restraint_id}", "ConstrainSidechain");'
            self.js_code.append(js_code)
            self.current_restraints[restraint_id] = (message, restraint_id)
        
        if self.constrainLigandAtoms:
            #TODO! check if there are any ligand atoms in the thing
            self.simulation.add_force(ConstrainSideChain) 
            restraint_id = len(self.current_restraints)
            message = 'Ligand Heavy Atoms'
            js_code = f'addRestraintToList("{message}", "{restraint_id}", "ConstrainLigands");'
            self.js_code.append(js_code)
            self.current_restraints[restraint_id] = (message, restraint_id)
        
        
        if self.selected_residues:
            self.simulation.add_force(AnchorAtoms)
        
        #need to add forces for the SSE's have them in the chemem thing 
        self.simulation.add_force(TugForce)
        self.simulation.add_force(SaltBridgeForce)
        self.simulation.add_force(HbondDistForce)
        self.simulation.add_force(HbondAngleForce)
        self.simulation.add_force(WeakHbondDistForce)
        self.simulation.add_force(WeakHbondAngleForce)
        self.simulation.add_force(HalogenBondForce)
        self.simulation.add_force(PiPiDistForce)
        self.simulation.add_force(CationPiForce)
        self.simulation.set_simulation()



class Simulation:
    def __init__(self, session, complex_system, complex_structure, densmap, platform_name = 'OpenCL'):
        #Do properly !!
        import chimerax 
        openmm_plugins_dir = os.path.join(chimerax.app_lib_dir, 'plugins')
        self.session = session
        self.complex_structure = complex_structure 
        self.complex_system = complex_system
        #self.filepath = filepath
        self.densmap = densmap
        self.platform = platform_name
        self.temperature = 100
        self.temperature_step = 1
        self.pressure = 1*unit.atmosphere
        self.heating_interval = 10
        self.tug_k = 10000
        self.saltbridge_k = 1000
        self.hbond_dist_k = 1000
        self.hbond_angle_k = 100
        self.pipi_dist_k = 1000
        self.pipi_angle_k = 100 
        self.pipi_offset_k = 100
        self.pipi_r0 = 0.35 #nanometers
        self.pipi_theta0 = 0.0 #radians
        self.pipi_offset0 = 0.0 #nanometers
        self.halogen_dist_k = 1000
        self.halogen_theta1_k = 1000 
        self.halogen_theta2_k = 1000 
        
        
        self.forces = {}
        self.force_group = 0
        self.active_constraints = []
        #hbond tugs 
        self.hbond_tug_atoms = {}
        self.weak_hbond_tug_atoms = {}
        self.saltbridge_tug_atoms = {}
        self.pipi_p_tug_rings = {}
        self.cation_pi_tugs = {}
        self.halogen_tug_atoms = {}
        #self.get_complex_structure()
        self.densmap_when_atoms_are_inside()
        
        self.debug = []
    
    
    @classmethod 
    def from_filepath(cls, session, file_path, densmap, parameters = None, platform_name = 'OpenCL'):
        
        complex_structure = get_complex_structure(file_path)
        complex_system = get_complex_system(complex_structure, parameters = parameters)
        
        
        return cls(session, 
                   complex_system,
                   complex_structure,
                   densmap,
                   platform_name = platform_name)
    
    
    def set_ions_active(self, active=True):
        val = 1.0 if active else 0.0
        try:
            self.simulation.context.setParameter("ion_switch", val)
            print(f"Ions set to {'ACTIVE' if active else 'GHOST'}")
        except Exception:
            print("No switchable ions found in system.")
    
    def chimera_x_atom_to_simulation_consistancy(self, model, atoms_to_position_index_as_dic):
        
            
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True) #in nanometers!!!
        
        for residue in model.residues:
            for atom in residue.atoms:
                
                if atom not in atoms_to_position_index_as_dic:
                    #com = f'del {atom.atomspec}'
                    #run(self.session, com)
                    continue
                
                atom_index = atoms_to_position_index_as_dic[atom]
                openmm_position = positions[atom_index]
                chimerax_positions = np.array(atom.coord) * 0.1
                openmm_array_position = np.array(openmm_position)
                self.debug.append([openmm_position,openmm_array_position, chimerax_positions])
                #if round(openmm_array_position[0],4) != round(chimerax_positions[0],4) or round(openmm_array_position[1],4) != round(chimerax_positions[1],4) or round(openmm_array_position[2],4) != round(chimerax_positions[2],4):  
                if np.all( np.round(openmm_array_position, 4) != np.round(chimerax_positions, 4) ):
                    positions[atom_index] = chimerax_positions * unit.nanometer
                    
        
        self.simulation.context.setPositions(positions)
        #self.simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
                    
        
    
    def densmap_when_atoms_are_inside(self):
        if self.densmap is not None:
            positions = self.complex_structure.positions
            positions = np.array([[i.x,i.y,i.z] for i in positions])
            
            #load all map data
            self.densmap.region = self.densmap.full_region()
            self.reload_map()
            origin, apix = self.densmap.data_origin_and_step()
            map_slice = get_chimera_slice(positions, np.array(origin), np.array(apix))
        
            self.densmap.region = map_slice 
            self.reload_map()
    
    def reload_map(self):
        if self.densmap.display:
            self.densmap.display = False 
            self.densmap.display = True
    
    
    def get_model_from_complex_structure(self, chimerax_model, distance_threshold=0.01):
        complex_structure = self.complex_structure
        atom_to_index = []
        unmatched_ligand_hydrogens = []  # List to hold unmatched ligand hydrogens
    
        # Build a mapping from (residue name, atom name) to simulation atoms
        sim_atoms_dict = {}
        for sim_atom in complex_structure.atoms:
            key = (sim_atom.residue.name, sim_atom.name)
            sim_atoms_dict.setdefault(key, []).append(sim_atom)
    
        # Build a mapping from (residue name, atom name) to ChimeraX atoms
        chimerax_atoms_dict = {}
        for atom in chimerax_model.atoms:
            key = (atom.residue.name, atom.name)
            chimerax_atoms_dict.setdefault(key, []).append(atom)
    
        # Match atoms between ChimeraX model and simulation
        for key, chimera_atoms in chimerax_atoms_dict.items():
            sim_atoms = sim_atoms_dict.get(key, [])
            if not sim_atoms:
                continue  # No matching atoms in simulation
    
            residue_name = key[0]
            atom_name = key[1]
    
            for atom in chimera_atoms:
                min_distance = None
                best_sim_atom = None
                atom_pos = np.array(atom.coord)
    
                for sim_atom in sim_atoms:
                    sim_atom_pos = np.array([sim_atom.xx, sim_atom.xy, sim_atom.xz])
                    distance = np.linalg.norm(atom_pos - sim_atom_pos)
    
                    if min_distance is None or distance < min_distance:
                        min_distance = distance
                        best_sim_atom = sim_atom
    
                if min_distance is not None and min_distance < distance_threshold:
                    atom_to_index.append([atom, best_sim_atom.idx])
                else:
                    # Collect unmatched ligand hydrogens for further processing
                    if residue_name.startswith('LIG') and atom.element.name == 'H':
                        unmatched_ligand_hydrogens.append(atom)
    
        # Build a mapping from ChimeraX heavy atoms to simulation heavy atoms within ligands
        ligand_heavy_atom_mapping = {}
    
        for atom_pair in atom_to_index:
            chimera_atom, sim_atom_idx = atom_pair
            if chimera_atom.element.name != 'H' and chimera_atom.residue.name.startswith('LIG'):
                ligand_heavy_atom_mapping[chimera_atom] = sim_atom_idx
    
        # match the unmatched ligand hydrogens
        for hydrogen_atom in unmatched_ligand_hydrogens:
            # Get the heavy atom it's bonded to
            bonded_heavy_atoms = [nbr for nbr in hydrogen_atom.neighbors if nbr.element.name != 'H']
            if not bonded_heavy_atoms:
                continue  # No bonded heavy atom found
    
            chimera_heavy_atom = bonded_heavy_atoms[0]
            sim_heavy_atom_idx = ligand_heavy_atom_mapping.get(chimera_heavy_atom)
            if sim_heavy_atom_idx is None:
                continue  # Heavy atom not matched
    
            # Get the simulation heavy atom
            sim_heavy_atom = complex_structure.atoms[sim_heavy_atom_idx]
            # Get hydrogens bonded to this heavy atom in simulation
            sim_bonded_hydrogens = [atom for atom in sim_heavy_atom.bond_partners if atom.element_name == 'H']
            if not sim_bonded_hydrogens:
                continue  # No hydrogens bonded in simulation
    
            # Match hydrogens based on minimal distance
            min_distance = None
            best_sim_hydrogen = None
            hydrogen_atom_pos = np.array(hydrogen_atom.coord)
    
            for sim_hydrogen in sim_bonded_hydrogens:
                sim_hydrogen_pos = np.array([sim_hydrogen.xx, sim_hydrogen.xy, sim_hydrogen.xz])
                distance = np.linalg.norm(hydrogen_atom_pos - sim_hydrogen_pos)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    best_sim_hydrogen = sim_hydrogen
    
            if min_distance is not None:
                atom_to_index.append([hydrogen_atom, best_sim_hydrogen.idx])
    
        return chimerax_model, atom_to_index

    
    def get_model_from_complex_structure_ori(self, chimerax_model, distance_threshold=0.01):
        complex_structure = self.complex_structure
        atom_to_index = []

        # Build a mapping from (residue name, atom name) to simulation atoms
        sim_atoms_dict = {}
        for sim_atom in complex_structure.atoms:
            key = (sim_atom.residue.name, sim_atom.name)
            sim_atoms_dict.setdefault(key, []).append(sim_atom)

        # Build a mapping from (residue name, atom name) to ChimeraX atoms
        chimerax_atoms_dict = {}
        for atom in chimerax_model.atoms:
            key = (atom.residue.name, atom.name)
            chimerax_atoms_dict.setdefault(key, []).append(atom)

        # Match atoms between ChimeraX model and simulation
        for key, chimera_atoms in chimerax_atoms_dict.items():
            sim_atoms = sim_atoms_dict.get(key, [])
            if not sim_atoms:
                
                print('NOT FOUND')
                print(key)
                print('')
                continue  # No matching atoms in simulation
            
            if key[0] == 'UNL':
                print('Doing Ligand........')
            # For each ChimeraX atom, find the closest simulation atom
            for atom in chimera_atoms:
                min_distance = None
                best_sim_atom = None
                # Get ChimeraX atom position (assuming atom.coord exists)
                
                atom_pos = np.array(atom.coord)
                

                for sim_atom in sim_atoms:
                    
                    
                    # Get simulation atom position from sim_atom.xx, sim_atom.xy, sim_atom.xz
                    sim_atom_pos = np.array([sim_atom.xx, sim_atom.xy, sim_atom.xz])
                    distance = np.linalg.norm(atom_pos - sim_atom_pos)
                    
                    
                    if min_distance is None or distance < min_distance:
                        min_distance = distance
                        best_sim_atom = sim_atom
                
                if key[0] == 'UNL':
                    print('Key', key)
                    print(min_distance, best_sim_atom)
                    print('')
                if min_distance is not None and min_distance < distance_threshold:
                    
                    atom_to_index.append([atom, best_sim_atom.idx])

        return chimerax_model, atom_to_index
    
    def set_ring_groups(self, chimerax_model, atoms_to_idx):
        atoms = [atom for residue in chimerax_model.residues for atom in residue.atoms]
        
        #get two distinct rings!!
        rings = [i.rings() for i in atoms if len(i.rings()) > 0]
        flattened_rings = list(set([i for j in rings for i in j]))

        ring_idxs = []
            
        for ring in flattened_rings:
            atoms = ring.atoms
            
            openff_indexes = sorted([atoms_to_idx[i] for i in atoms if i in atoms_to_idx])
            if len(openff_indexes) != len(atoms):
                continue
            
            elif openff_indexes not in ring_idxs:
                ring_idxs.append(openff_indexes)
            
        self.ring_idxs = ring_idxs
        #TODO!! other things!!
    
    def set_anchor_residues(self, chimerax_model, selected_atom_data, atoms_to_idx):
        residue_atoms_not_selected = []
        for atom_vector in selected_atom_data['res_atoms_not_selected'].values():
            residue_atoms_not_selected  += [atoms_to_idx[i] for i in atom_vector if i in atoms_to_idx]
        
        
        self.residue_atoms_not_selected = residue_atoms_not_selected
        
        anchor_atoms = []
        for residue in selected_atom_data['extra_residues']:
            anchor_atoms += [atoms_to_idx[i] for i in residue.atoms if i in atoms_to_idx]
        
        self.anchor_atoms = anchor_atoms
            
        
    
    def set_SSE_elements(self, chimerax_model, atoms_to_idx):
        
        def select_and_color(residues, rd):
            for residue in residues:
                for atom in residue.atoms:
                    ca = rd[atom.idx] 
                    ca.selected = True
        rd = {i : j for j,i in atoms_to_idx.items()}
        compute_ss(chimerax_model)
        
        ss_ids = {}
        ss_id_to_sse = {}
        
        ss_ids_helix = {}
        ss_id_to_sse_helix = {}
        
        select_atoms = []
        for residue in chimerax_model.residues:
            
            if residue.ss_type == 1:
                if residue.ss_id in ss_ids_helix:
                    ss_ids_helix[residue.ss_id].append(residue)
                else:
                    ss_ids_helix[residue.ss_id] = [residue]
                    ss_id_to_sse_helix[residue.ss_id] = residue.ss_type
                    
            
            if residue.ss_type > 0:
                if residue.ss_id in ss_ids:
                    ss_ids[residue.ss_id].append(residue)
                else:
                    ss_ids[residue.ss_id] = [residue]
                    ss_id_to_sse[residue.ss_id] = residue.ss_type
        
        ss_ids_openff = {}

        for ss in ss_ids:
            openff_residues = []
            residues = ss_ids[ss]
            for residue in residues:
                for atom in residue.atoms:
                    if atom in atoms_to_idx:
                        atom_idx = atoms_to_idx[atom]
                        

                        openff_residues.append(self.complex_structure.atoms[atom_idx].residue)
                        break
            
            
            ss_ids_openff[ss] = openff_residues 
        
        
        ss_ids_openff_helix = {}
        
        for ss in ss_ids_helix:
            openff_residues_helix = []
            residues = ss_ids_helix[ss]
            for residue in residues:
                for atom in residue.atoms:
                    if atom in atoms_to_idx:
                        atom_idx = atoms_to_idx[atom]
                        

                        openff_residues_helix.append(self.complex_structure.atoms[atom_idx].residue)
                        break
            
            
            ss_ids_openff_helix[ss] = openff_residues_helix
        
       
        
        def is_bonded_to(residue1, residue2):
            for atom in residue1.atoms:
                if atom.name == 'C':
                    for next_atom in atom.bond_partners:
                        if next_atom.name == 'N' and next_atom.residue.idx == residue2.idx:
                            return True
            return False
        
        helices = []
        for i in ss_ids_openff_helix:
            helices += ss_ids_openff_helix[i]
        helices = sorted(helices, key = lambda x: x.idx)
        
        sse_helices = {}
        current_helix = []
        helix_id = 1
    
        for i, residue in enumerate(helices):
            if not current_helix:
                # Start a new helix
                current_helix.append(residue)
            else:
                # Check if the current residue is bonded to the previous one
                if is_bonded_to(current_helix[-1], residue):
                    current_helix.append(residue)
                
                else:
                    # If not bonded, finalize the current helix and start a new one
                    sse_helices[f"helix_{helix_id}"] = current_helix
                    helix_id += 1
                    current_helix = [residue]
        
        if current_helix:
            sse_helices[f"helix_{helix_id}"] = current_helix
            
        
        self.sse_residues = ss_ids_openff 
        self.sse_types = ss_id_to_sse 
        #self.sse_residues_helix =  ss_ids_openff_helix
        self.sse_residues_helix = sse_helices

    def add_constraint(self, addConstraint):
        addConstraint.apply(self)
        self.active_constraints.append(addConstraint.name)
        
                
        
    
    def add_force(self, AddForce):
        
        force = AddForce().apply(self)
        force_idx = self.complex_system.addForce(force)
        self.complex_system.getForce(force_idx).setForceGroup(self.force_group)
        self.force_group += 1
        self.forces[force_idx] = force
        setattr(self, AddForce.name, force)
        print(f'{AddForce.name} added to system')
        
    def remove_force(self, force_group):
        self.complex_system.removeForce(force_group) #-1) ?
    
    def set_force_groups(self):
        
        for force in self.complex_system.getForces():
            force.setForceGroup(self.force_group)
            self.force_group += 1
        
    def set_simulation(self):
        integrator = LangevinIntegrator(self.temperature*unit.kelvin, 1.0/unit.picoseconds,
                                        1.0*unit.femtoseconds)
        self.integrator = integrator
        
        _platform = Platform.getPlatformByName(self.platform)
        simulation = app.Simulation(
            self.complex_structure.topology, self.complex_system, integrator, platform=_platform)
        
        simulation.context.setPositions(self.complex_structure.positions)
        self.simulation = simulation
    
    def minimise_system(self):
        #set temp here!!!
        #self.simulation.context.setVelocitiesToTemperature(self.temperature*unit.kelvin)
        
        self.simulation.minimizeEnergy(maxIterations=200)
    
    def get_positions(self):
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        positions = positions.value_in_unit(unit.angstrom)
        return positions 
    
    
    def get_positions_as_numpy(self):
        
        state = self.simulation.context.getState(getPositions=True)
        # Obtain positions as a NumPy array with units
        positions = state.getPositions(asNumpy=True)
        # Convert positions to angstroms and remove units
        positions_in_angstrom = positions / unit.angstrom
        return positions_in_angstrom
    
    def step(self, step):
        self.simulation.step(step)
    
    def set_map_bias(self, map_bias):
        if hasattr(self, 'map_bias'):
            self.simulation.context.setParameter('global_k', map_bias)
            print('GLOBAL K UPDATED: ', map_bias)
    
    def set_tempreture(self, new_temp, temperature_step = None):
        if temperature_step is None:
            temperature_step = self.temperature_step
        if new_temp > self.temperature:
            temps = [i+1 for i in range(int(self.temperature), int(new_temp), temperature_step)]
            
        elif new_temp  < self.temperature:
            temps = [i-1 for i in range(int(self.temperature), int(new_temp), -temperature_step)]
           
        else:
            temps = []
            
        
        for t in temps:
            self.simulation.integrator.setTemperature(t*unit.kelvin)
            #self.barostat.setDefaultTemperature(t*unit.kelvin)
            self.simulation.step(self.heating_interval)
        self.temperature = new_temp
    
    def update_tug_force_for_atom(self, atom_index, new_position, tug_k = None):
        if tug_k is None:
            tug_k = self.tug_k
            
        context = self.simulation.context
        indices, params = self.tug_force.getBondParameters(atom_index)
        
        if indices[0] == atom_index:  # Assuming atom_index is the first item in the bond
            x0, y0, z0 = Quantity(
                value=[new_position[0], new_position[1], new_position[2]], unit=unit.angstrom)
    
            new_params = [tug_k, x0,y0,z0]
            self.tug_force.setBondParameters(atom_index, indices, new_params)
            self.tug_force.updateParametersInContext(context)
    
    
    def update_halogen_bond_tug_force_for_atom(self, indices, 
                                               distance_k = None, 
                                               theta1_k = None,
                                               theta2_k = None,
                                               distance = None,
                                               theta1 = None,
                                               theta2 = None):
        
        if distance_k is None:
            distance_k = self.halogen_dist_k 
        if theta1_k is None:
            theta1_k = self.halogen_theta1_k 
        if theta2_k is None:
            theta2_k = self.halogen_theta2_k 
        
        if distance is None:
            distance = 3.0 * unit.angstrom
        if theta1 is None:
            theta1 = 180.0 * unit.degrees 
        if theta2 is None:
            theta2 = 120.0 * unit.degrees
        
        
        if str(indices) in self.halogen_tug_atoms:
            context = self.simulation.context
            halogen_int = self.halogen_tug_atoms[str(indices)]
            indices, params = self.halogen_bond_force.getBondParameters(halogen_int)
            new_params = [distance_k, theta1_k, theta2_k, distance, theta1, theta2]
            self.halogen_bond_force.setBondParameters(halogen_int, indices, new_params)
            self.salt_bridge_force.updateParametersInContext(context)
            
        else:
            halogen_int = self.halogen_bond_force.addBond(indices, [distance_k,
                                                                    theta1_k,
                                                                    theta2_k,
                                                                    distance,
                                                                    theta1,
                                                                    theta2])
            
            self.simulation.context.reinitialize(preserveState=True)
            self.halogen_tug_atoms[str(indices)] = halogen_int
        
    def update_saltbridge_tug_force_for_atom(self, indices, distance_k=None):
        
        if distance_k is None:
            distance_k = self.saltbridge_k 
        
        dist = 4.0 * unit.angstrom 
        
        if str(sorted(indices)) in self.saltbridge_tug_atoms:
            
            context = self.simulation.context
            index =  self.saltbridge_tug_atoms[str(sorted(indices))]
            dist_indices, dist_params = self.salt_bridge_force.getBondParameters(index)
            new_dist_params = [distance_k, dist]
            self.salt_bridge_force.setBondParameters(index, dist_indices, new_dist_params)
            self.salt_bridge_force.updateParametersInContext(context)
            
            
        else:
            saltbridge_int = self.salt_bridge_force.addBond(indices, [distance_k, dist])
            self.simulation.context.reinitialize(preserveState=True)
            self.saltbridge_tug_atoms[str(sorted(indices))] = saltbridge_int
            
    def update_weak_hbond_tug_force_for_atom(self, indices, distance_k = None, angle_k = None ):
        
        donor, hydrogen, acceptor = indices
        
        if distance_k is None:
            distance_k = self.hbond_dist_k 
        
        if angle_k is None:
            angle_k = self.hbond_angle_k
    
        dist = 3.4 * unit.angstrom
        angle = 180.0 * unit.degrees  
        print('Weak-HB')
        if str(indices) in self.weak_hbond_tug_atoms:
            
            context = self.simulation.context
            dist_idx, angle_idx = self.weak_hbond_tug_atoms[str(indices)]
            dist_indices, dist_params = self.weak_hbond_tug_dist_force.getBondParameters(dist_idx)
            p1, p2, p3, angle_params = self.weak_hbond_tug_angle_force.getAngleParameters(angle_idx)
            new_dist_params = [distance_k, dist]
            new_angle_params = [angle_k, angle]
            self.weak_hbond_tug_dist_force.setBondParameters(dist_idx, dist_indices, new_dist_params)
            self.weak_hbond_tug_angle_force.setAngleParameters(angle_idx, p1,p2,p3, new_angle_params)
            self.weak_hbond_tug_dist_force.updateParametersInContext(context)
            self.weak_hbond_tug_angle_force.updateParametersInContext(context)
        
        else:
            
            dist_int = self.weak_hbond_tug_dist_force.addBond([donor, acceptor], [distance_k , dist])
            angle_int = self.weak_hbond_tug_angle_force.addAngle(donor, hydrogen, acceptor, [angle_k, angle])
            self.simulation.context.reinitialize(preserveState=True)
            self.weak_hbond_tug_atoms[str(indices)] = [dist_int, angle_int]
            
            
    def update_hbond_tug_force_for_atom(self, hbond_index, hbond_dist_k = None, hbond_angle_k = None):
        
        donor, hydrogen, acceptor = hbond_index
        
        
        if hbond_dist_k is None:
            hbond_dist_k = self.hbond_dist_k 
        
        if hbond_angle_k is None:
            hbond_angle_k = self.hbond_angle_k
        
        #need to get the correct angles form here!!
        dist = 2.9 * unit.angstrom
        angle = 180.0 * unit.degrees  
        
        if str(hbond_index) in self.hbond_tug_atoms:
            
            context = self.simulation.context
            hbond_dist_idx, hbond_angle_idx = self.hbond_tug_atoms[str(hbond_index)]
            dist_indices, dist_params = self.hbond_tug_dist_force.getBondParameters(hbond_dist_idx)
            p1, p2, p3, angle_params = self.hbond_tug_angle_force.getAngleParameters(hbond_angle_idx)
            new_dist_params = [hbond_dist_k, dist]
            new_angle_params = [hbond_angle_k, angle]
            self.hbond_tug_dist_force.setBondParameters(hbond_dist_idx, dist_indices, new_dist_params)
            self.hbond_tug_angle_force.setAngleParameters(hbond_angle_idx, p1,p2,p3, new_angle_params)
            self.hbond_tug_dist_force.updateParametersInContext(context)
            self.hbond_tug_angle_force.updateParametersInContext(context)
            
        else:
            
            hbond_dist_int = self.hbond_tug_dist_force.addBond([donor, acceptor], [hbond_dist_k , dist])
            hbond_angle_int = self.hbond_tug_angle_force.addAngle(donor, hydrogen, acceptor, [hbond_angle_k, angle])
            self.simulation.context.reinitialize(preserveState=True)
            self.hbond_tug_atoms[str(hbond_index)] = [hbond_dist_int, hbond_angle_int] 
    
    def update_cation_pi_tug(self, 
                             ring_atom_indexes,
                             cation_atom_index,
                             k_distance = None,
                             k_offset = None,
                             k_angle = None,
 
                             ):
        
        if k_distance is None:
            k_distance = 1000 * unit.kilojoule_per_mole/unit.nanometer**2
        
        if k_offset is None:
            k_offset = 1000 * unit.kilojoule_per_mole
        
        if k_angle is None:
            k_angle = 1000 * unit.kilojoule_per_mole/unit.nanometer**2
        

        if str([ring_atom_indexes, cation_atom_index]) in self.cation_pi_tugs:
            context = self.simulation.context
            cation_pi_int = self.cation_pi_tugs[str([ring_atom_indexes, cation_atom_index])]
            indices, params = self.cation_pi_force.getBondParameters(cation_pi_int)
            new_params = [k_distance, k_offset, k_angle]
            self.cation_pi_force.setBondParameters(cation_pi_int, indices, new_params)
            self.cation_pi_force.updateParametersInContext(context)
            
        else:
            atoms = ring_atom_indexes[:3] + [cation_atom_index] 
            print('atoms', atoms)
            print('k_dist', k_distance)
            print('k_offset', k_offset)
            print('k_angle', k_angle)
            cation_pi_int = self.cation_pi_force.addBond(atoms, [k_distance, k_offset, k_angle])
            self.simulation.context.reinitialize(preserveState=True)
            self.cation_pi_tugs[str([ring_atom_indexes, cation_atom_index])] = cation_pi_int
    
    def update_pipi_p_tug(self,
                          ring_index_pairs, 
                          r0 = None, 
                          theta0 = None,
                          offset0 = None,
                          k_distance=None,
                          k_angle=None,
                          k_offset=None,
                          ):
  
        ring_1, ring_2 = ring_index_pairs
        
        if r0 is None:
            r0 = self.pipi_r0
        
        if theta0 is None:
            theta0 = self.pipi_theta0
        
        if offset0 is None:
            offset0 = self.pipi_offset0
        
        if k_distance is None:
            #move to self
            k_distance=1000*unit.kilojoule_per_mole/unit.nanometer**2
        
        if k_angle is None:
            k_angle = 1000*unit.kilojoule_per_mole 
        
        if k_offset is None:
            k_offset = 1000*unit.kilojoule_per_mole/unit.nanometer**2
       
        if str(sorted(ring_index_pairs)) in self.pipi_p_tug_rings:
            context = self.simulation.context
            pipi_int = self.pipi_p_tug_rings[str(sorted(ring_index_pairs))]
            indices, params = self.pipi_tug_dist_force.getBondParameters(pipi_int)
            new_params = [k_distance, k_angle, k_offset, r0, offset0, theta0]
            self.pipi_tug_dist_force.setBondParameters(pipi_int, indices, new_params)
            self.pipi_tug_dist_force.updateParametersInContext(context)
            
        else:
            #group_1_index = self.ring_idxs.index(ring_1)
            #group_2_index = self.ring_idxs.index(ring_2)
            atoms = ring_1[:3] + ring_2[:3]
            #atoms = group_1_index[:3] + group_2_index[:3]
            pipi_dist_int = self.pipi_tug_dist_force.addBond(atoms, [k_distance, k_angle, k_offset, r0, offset0, theta0])
            self.simulation.context.reinitialize(preserveState=True)
            self.pipi_p_tug_rings[str(sorted(ring_index_pairs))] = pipi_dist_int
        # Add the bonds for each ring pair
        
        

                                            
    def simulated_anneling_initial_heating(self, 
                                           start_temp=None , 
                                           norm_temp=None, 
                                           temperature_step=None,
                                           initial_heating_interval=None):
        
        if start_temp is None:
            start_temp = 0
        
        if norm_temp is None:
            norm_temp = 300 
        
        if temperature_step is None:
            temperature_step = 1
        
        if initial_heating_interval is None:
            initial_heating_interval = 10
        
        for temp in range(start_temp, norm_temp, temperature_step):
            print(temp)
            self.simulation.integrator.setTemperature(temp*unit.kelvin)
            self.simulation.step(initial_heating_interval)
    
    def _set_temp(self, temp):
        self.simulation.integrator.setTemperature(temp*unit.kelvin)
       
class Force:
    pass





class MapBias(Force):
    name = 'map_bias'
    @classmethod 
    def apply(cls, simulation_object, 
              global_k = 75.0, 
              c_level = 0.0):
        
        current_map = simulation_object.densmap 
        origin, apix = current_map.data_origin_and_step()
        origin = np.array(origin)
        apix = np.array(apix)
        copy_map = current_map.matrix()
        copy_map =  copy_map * (copy_map >= c_level)
        copy_map = copy_map / np.amax(copy_map)
        mp_force = cls.force(copy_map, origin, apix, copy_map.shape, global_k, blur=0)
        #just move ligand
        for atom in simulation_object.complex_structure.atoms:
            mp_force.addBond([atom.idx])
        return  mp_force
    
    @staticmethod 
    def force(m, origin, apix, box_size, global_k, blur=0):
        f = CustomCompoundBondForce(1, '')
        d3d_func = MapBias.compute_map_field(m,origin, apix, box_size, blur)
        f.addTabulatedFunction(name='map_potential', function=d3d_func)
        f.addGlobalParameter(name='global_k', defaultValue=global_k)
        f.setEnergyFunction('-global_k * map_potential(z1,y1,x1)')
        return f
    
    @staticmethod
    def compute_map_field(m, origin, apix, box_size, blur=0, d3d_func=None):
        morg = np.array(origin)[::-1] - apix/2
        mdim = np.array(box_size)* apix
        mmax = morg+mdim
        mod_m = gaussian_filter(m, blur)
        minmaxes = np.array(list(zip(morg/10, mmax/10))).ravel()
        if d3d_func is None:

            d3d_func = Continuous3DFunction(
                *mod_m.shape, mod_m.ravel(order="F"), *minmaxes)

        else:
            d3d_func.setFunctionParameters(
                *mod_m.shape, mod_m.ravel(order="F"), *minmaxes)
        
        return d3d_func

class SSERigidBodyBBConstraint(Force):
    name = 'rigid_body_force'
    @classmethod 
    def apply(cls, 
              simulation_object,
              ):
        for sse_id, residues in simulation_object.sse_residues.items():
            
            for residue in residues:
                
                atom_N_curr = None
                atom_C_curr = None 
                atom_CA_curr = None 
                atom_N_next = None 
                atom_C_prev = None
                for atom in residue.atoms:
                    if atom.name == 'N':
                        atom_N_curr = atom 
                    elif atom.name == 'C':
                        atom_C_curr = atom 
                    
                    elif atom.name == 'CA':
                        atom_CA_curr = atom 
                
                
                if atom_C_curr is not None:
                    for atom in atom_C_curr.bond_partners:
                        if atom.name == 'N' and atom.residue.idx != residue.idx:
                            atom_N_next = atom 
                
                if atom_N_next is not None:
                    for atom in atom_N_next.residue.atoms:
                        if atom.name == 'C':
                            atom_C_next = atom 
                        elif atom.name == 'CA':
                            atom_CA_next = atom
                
                if atom_N_curr is not None and atom_N_next is not None:
                    distance = cls.compute_distance(simulation_object.complex_structure.positions, atom_N_curr, atom_N_next)
                    simulation_object.complex_system.addConstraint(atom_N_curr.idx, atom_N_next.idx, distance)
                
                if atom_C_curr is not None and atom_C_next is not None:
                    distance = cls.compute_distance(simulation_object.complex_structure.positions, atom_C_curr, atom_C_next)
                    simulation_object.complex_system.addConstraint(atom_C_curr.idx, atom_C_next.idx, distance)
                
                if atom_CA_curr is not None and atom_CA_next is not None:
                    distance = cls.compute_distance(simulation_object.complex_structure.positions, atom_CA_curr, atom_CA_next)
                    #print('DIST', distance, atom_CA_curr, atom_CA_next)
                    simulation_object.complex_system.addConstraint(atom_CA_curr.idx, atom_CA_next.idx, distance)
        return None
            
    @staticmethod
    def compute_distance(positions, atom1, atom2):
        """
        Compute the distance between two atoms using their positions.
        """
        pos1 = positions[atom1.idx]
        pos2 = positions[atom2.idx]
        return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2) / 10


class CentroidSSEForce(Force):
    name = 'centroid_sse_force'
    @classmethod 
    def apply(cls,
              simulation_object,
              k = 1000):
        
        force = cls.get_force()
        for sse_id, residues in simulation_object.sse_residues.items():
            atom_g1_indexs = []
            atom_g2_indexes = []
            for residue in residues:
                pass
            
            
    @staticmethod 
    def get_force():
        force = CustomCentroidBondForce(2, "k*(distance(g1, g2) - r0)^2")
        force.addGlobalParameter("k", 1000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
        force.addPerBondParameter("r0")
        return force 

class HelixHbondForce(Force):
    name = 'Helix_hbond_force'
    @classmethod 
    def apply(cls, 
              simulation_object,
              k = 500 * unit.kilojoule_per_mole / unit.nanometer**2):
        force = cls.get_force(k) 
        
        for sse_id, residues in simulation_object.sse_residues_helix.items():
            for residue in residues:
                
                atom_O_curr = None
                atom_N_next = None 
                
                
                for atom in residue.atoms:
                    if atom.name == 'O':
                        atom_O_curr = atom 
                    
            
                
                plus_4_residue_index = residue.idx + 4 
                plus_4_residue = None
                try:
                    plus_4_residue = simulation_object.complex_structure.residues[plus_4_residue_index]
                
                except IndexError:
                    pass
                
                if plus_4_residue in residues:
                    for atom in plus_4_residue.atoms:
                        
                        if atom.name == 'N':
                            atom_N_next = atom 
                        
                
                if atom_O_curr is not None and atom_N_next is not None:
                    dist = cls.compute_distance(simulation_object.complex_structure.positions, atom_O_curr, atom_N_next)
                    print('HB_DIST', dist, atom_O_curr, atom_N_next)
                    force.addBond(atom_O_curr.idx, atom_N_next.idx, [dist * unit.nanometer])
        return force
    
    @staticmethod
    def compute_distance(positions, atom1, atom2):
        """
        Compute the distance between two atoms using their positions.
        """
        pos1 = positions[atom1.idx]
        pos2 = positions[atom2.idx]
        return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2) / 10
    
                
                
    @staticmethod 
    def get_force(k = 500 * unit.kilojoule_per_mole / unit.nanometer**2):
        force = CustomBondForce("0.5 * k * (r - r0)^2")
        #force.addGlobalParameter("r0", 2.9 * unit.nanometer)
        force.addGlobalParameter("k", k)
        force.addPerBondParameter("r0")
        return force




class SSE_force(Force):
    name = 'SSE_force'
    @classmethod 
    def apply(cls, 
              simulation_object,
              k = 100):
        
        sse_force = cls.get_force() 
        #TODO !! non-bonded
        for sse_id, residues in simulation_object.sse_residues.items():
            
            for residue in residues:
                
                atom_N_curr = None
                atom_C_curr = None 
                atom_CA_curr = None 
                atom_N_next = None 
                atom_C_prev = None
                for atom in residue.atoms:
                    if atom.name == 'N':
                        atom_N_curr = atom 
                    elif atom.name == 'C':
                        atom_C_curr = atom 
                    
                    elif atom.name == 'CA':
                        atom_CA_curr = atom 
                
                if atom_C_curr is not None:
                    for atom in atom_C_curr.bond_partners:
                        if atom.name == 'N' and atom.residue.idx != residue.idx:
                            atom_N_next = atom 
                
                if atom_N_curr is not None:
                    for atom in atom_N_curr.bond_partners:
                        if atom.name =='C' and atom.residue.idx != residue.idx:
                            atom_C_prev = atom 
                
                if atom_N_next is not None:
                    for atom in atom_N_next.residue.atoms:
                        if atom.name == 'CA' and atom.residue.idx != residue.idx:
                            atom_CA_next = atom
                
                # Add phi angle restraint
                if not None in [atom_C_prev, atom_N_curr, atom_CA_curr, atom_C_curr]:
                    sse_force.addTorsion(atom_C_prev.idx, atom_N_curr.idx, atom_CA_curr.idx, atom_C_curr.idx, 1, 0*unit.radians, k*unit.kilojoules_per_mole)
                
                # Add psi angle restraint
                if not None in [atom_N_curr, atom_CA_curr, atom_C_curr, atom_N_next]:
                    sse_force.addTorsion(atom_N_curr.idx, atom_CA_curr.idx, atom_C_curr.idx, atom_N_next.idx, 1, 0*unit.radians, k*unit.kilojoules_per_mole)
                
                if not None in [atom_CA_curr, atom_C_curr, atom_N_next, atom_CA_next]:
                    sse_force.addTorsion(atom_CA_curr.idx, atom_C_curr.idx, atom_N_next.idx, atom_CA_next.idx, 1, 0*unit.radians, k*unit.kilojoules_per_mole)
                
        return sse_force 
                #GetPrevious atoms 
                
                
                
              
        #sse_force.addTorsion(0, 4, 6, 9, 1, 0*radians, 100*kilojoules_per_mole)
    
    @staticmethod 
    def get_force():
        return PeriodicTorsionForce()


class PhiAnglelForce(Force):
    name = 'phi_angle_force'
    @classmethod 
    def apply(cls, 
              simulation_object,
              k = 10 * unit.kilocalories_per_mole / unit.radians**2):
        
        force = cls.get_force(k = k)
        for sse_id, residues in simulation_object.sse_residues_helix.items():
            for residue in residues:
                
                atom_N_curr = None
                atom_C_curr = None 
                atom_CA_curr = None 
                atom_C_prev = None
                for atom in residue.atoms:
                    if atom.name == 'N':
                        atom_N_curr = atom 
                    elif atom.name == 'C':
                        atom_C_curr = atom 
                    
                    elif atom.name == 'CA':
                        atom_CA_curr = atom 
                
                if atom_N_curr is not None:
                    for atom in atom_N_curr.bond_partners:
                        if atom.name =='C' and atom.residue.idx != residue.idx:
                            atom_C_prev = atom 
                
                # Add phi angle restraint
                if not None in [atom_C_prev, atom_N_curr, atom_CA_curr, atom_C_curr]:
                    
                    theta0 = cls.calculate_dihedral(atom_C_prev, 
                                                    atom_N_curr, 
                                                    atom_CA_curr, 
                                                    atom_C_curr,
                                                    simulation_object.complex_structure.positions)
                    force.addTorsion(atom_C_prev.idx, atom_N_curr.idx, atom_CA_curr.idx, atom_C_curr.idx, [theta0 * unit.radians])
        return force
            
    
    @staticmethod 
    def get_force(k = 10 * unit.kilocalories_per_mole / unit.radians**2):
        expr = "0.5* phi_k *min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0); pi = 3.1415926535"
        force = CustomTorsionForce(expr)
        force.addGlobalParameter("phi_k", k)
        force.addPerTorsionParameter("theta0")
        return force
    
    @staticmethod 
    def calculate_dihedral(a1, a2, a3, a4, positions):
        """
        Calculate the dihedral angle between four atoms in 3D space.
    
        Parameters:
        atom1, atom2, atom3, atom4: numpy arrays
            3D coordinates of the four atoms (as numpy arrays of shape (3,))
    
        Returns:
        dihedral_angle: float
            Dihedral angle in radians
        """
        atom1, atom2, atom3, atom4 = positions[a1.idx], positions[a2.idx], positions[a3.idx], positions[a4.idx]
        atom1 = np.array([atom1.x, atom1.y, atom1.z])
        atom2 = np.array([atom2.x, atom2.y, atom2.z])
        atom3 = np.array([atom3.x, atom3.y, atom3.z])
        atom4 = np.array([atom4.x, atom4.y, atom4.z])
        
        # Define vectors between atoms
        b1 = atom2 - atom1
        b2 = atom3 - atom2
        b3 = atom4 - atom3
    
        # Calculate normals to the planes defined by the vectors
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
    
        # Normalize the normal vectors
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
    
        # Calculate the vector perpendicular to b2 in the plane formed by b1 and b3
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    
        # Calculate the dihedral angle using atan2 for proper sign
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        dihedral_angle = np.arctan2(y, x)
    
        return dihedral_angle

class PsiAngleForce(PhiAnglelForce):
    name = 'psi_angle_force'
    @classmethod 
    def apply(cls, 
              simulation_object,
              k = 10 * unit.kilocalories_per_mole / unit.radians**2):

        force = cls.get_force(k = k)
        for sse_id, residues in simulation_object.sse_residues_helix.items():
            for residue in residues:
                
                atom_N_curr = None
                atom_C_curr = None 
                atom_CA_curr = None 
                atom_N_next = None 

                for atom in residue.atoms:
                    if atom.name == 'N':
                        atom_N_curr = atom 
                    elif atom.name == 'C':
                        atom_C_curr = atom 
                    
                    elif atom.name == 'CA':
                        atom_CA_curr = atom 
                
                if atom_C_curr is not None:
                    for atom in atom_C_curr.bond_partners:
                        if atom.name == 'N' and atom.residue.idx != residue.idx:
                            atom_N_next = atom 

                # Add psi angle restraint
                if not None in [atom_N_curr, atom_CA_curr, atom_C_curr, atom_N_next]:
                    
                    theta0 = cls.calculate_dihedral(atom_N_curr,
                                                    atom_CA_curr,
                                                    atom_C_curr, 
                                                    atom_N_next,
                                                    simulation_object.complex_structure.positions)
                    
                    force.addTorsion(atom_N_curr.idx, atom_CA_curr.idx, atom_C_curr.idx, atom_N_next.idx, [theta0 * unit.radians])
        return force
    
    @staticmethod 
    def get_force(k = 10 * unit.kilocalories_per_mole / unit.radians**2):
        expr = "0.5* psi_k *min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0); pi = 3.1415926535"
        force = CustomTorsionForce(expr)
        force.addGlobalParameter("psi_k", k)
        force.addPerTorsionParameter("theta0")
        return force


class ConstrainProtein(Force):
    name = 'constrain_protein'
    @classmethod 
    def apply(cls, 
              simulation_object,
              k = 5000):
        force = cls.get_force(k=k)
        for atom in simulation_object.complex_structure.atoms:
            
            if atom.residue.name in RESIDUE_NAMES and atom.element > 1:
                
            
                x0, y0, z0 = Quantity(
                    value=[atom.xx, atom.xy, atom.xz], unit=unit.angstrom)
                
                x0 = atom.xx * 0.1
                y0 = atom.xy * 0.1
                z0 = atom.xz * 0.1
                
                force.addBond([atom.idx], [x0, y0, z0])
        
        return force
    
    @staticmethod 
    def get_force(k = 1000):
        expr = "protein_constraint_k * ((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
        f = CustomCompoundBondForce(1, expr)
        f.addGlobalParameter("protein_constraint_k", 1000)
        f.addPerBondParameter("x0")
        f.addPerBondParameter("y0")
        f.addPerBondParameter("z0")
        return f


class ConstrainLigandAtoms(Force):
    name = 'constrain_ligand'
    @classmethod 
    def apply(cls,
              simulation_object,
              k = 5000):
        force = cls.get_force(k=k)
        for atom in simulation_object.complex_structure.atoms:
            if atom.residue.name.startswith('LIG') and atom.element > 1:
                x0, y0, z0 = Quantity(
                    value=[atom.xx, atom.xy, atom.xz], unit=unit.angstrom)
                
                x0 = atom.xx * 0.1
                y0 = atom.xy * 0.1
                z0 = atom.xz * 0.1
                
                force.addBond([atom.idx], [x0, y0, z0])
        
        return force
    @staticmethod 
    def get_force(k = 1000):
        expr = "sidechain_constraint_k * ((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
        f = CustomCompoundBondForce(1, expr)
        f.addGlobalParameter("ligand_constraint_k", 1000)
        f.addPerBondParameter("x0")
        f.addPerBondParameter("y0")
        f.addPerBondParameter("z0")
        return f
    
class ConstrainSideChain(Force):
    name = 'constrain_sidechain'
    @classmethod 
    def apply(cls, 
              simulation_object,
              k = 5000):
        
        force = cls.get_force(k=k)
        for atom in simulation_object.complex_structure.atoms:
            
            if atom.residue.name in RESIDUE_NAMES and atom.element > 1 and atom.name not in BACKBONE_ATOM_NAMES:
                
            
                x0, y0, z0 = Quantity(
                    value=[atom.xx, atom.xy, atom.xz], unit=unit.angstrom)
                
                x0 = atom.xx * 0.1
                y0 = atom.xy * 0.1
                z0 = atom.xz * 0.1
                
                force.addBond([atom.idx], [x0, y0, z0])
        
        return force
    
    @staticmethod 
    def get_force(k = 1000):
        expr = "sidechain_constraint_k * ((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
        f = CustomCompoundBondForce(1, expr)
        f.addGlobalParameter("sidechain_constraint_k", 1000)
        f.addPerBondParameter("x0")
        f.addPerBondParameter("y0")
        f.addPerBondParameter("z0")
        return f

class ConstrainBackBone(Force):
    name = 'constrain_backbone'
    @classmethod 
    def apply(cls, 
              simulation_object,
              k = 5000):
        
        force = cls.get_force(k=k)
        for atom in simulation_object.complex_structure.atoms:
            
            if atom.residue.name in RESIDUE_NAMES and atom.element > 1 and atom.name in BACKBONE_ATOM_NAMES:
                
            
                x0, y0, z0 = Quantity(
                    value=[atom.xx, atom.xy, atom.xz], unit=unit.angstrom)
                
                x0 = atom.xx * 0.1
                y0 = atom.xy * 0.1
                z0 = atom.xz * 0.1
                
                force.addBond([atom.idx], [x0, y0, z0])
        
        return force
    
    @staticmethod 
    def get_force(k = 1000):
        expr = "backbone_constraint_k * ((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
        f = CustomCompoundBondForce(1, expr)
        f.addGlobalParameter("backbone_constraint_k", 1000)
        f.addPerBondParameter("x0")
        f.addPerBondParameter("y0")
        f.addPerBondParameter("z0")
        return f

class AnchorAtoms(Force):
    name ='anchor_force'
    @classmethod 
    def apply(cls,
              simulation_object,
              k = 5000):
        
        anchor_force = cls.get_force(k=k)
        
        for atom_idx in simulation_object.anchor_atoms:
            atom = simulation_object.complex_structure[atom_idx]
            x0, y0, z0 = Quantity(
                value=[atom.xx, atom.xy, atom.xz], unit=unit.angstrom)
            
            x0 = atom.xx * 0.1
            y0 = atom.xy * 0.1
            z0 = atom.xz * 0.1
            
            anchor_force.addBond([atom_idx], [x0, y0, z0])
        
        return anchor_force 

    @staticmethod 
    def get_force(k = 1000):
        expr = "anchor_k * ((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
        f = CustomCompoundBondForce(1, expr)
        f.addGlobalParameter("anchor_k", 1000)
        f.addPerBondParameter("x0")
        f.addPerBondParameter("y0")
        f.addPerBondParameter("z0")
        return f



class TugForce(Force):
    name = 'tug_force'
    @classmethod 
    def apply(cls,
              simulation_object,
              k = 1000):
        
        tug_force = cls.get_force()
        for atom in simulation_object.complex_structure.atoms:
            x0, y0, z0 = Quantity(
                value=[atom.xx, atom.xy, atom.xz], unit=unit.angstrom)
            tug_force.addBond([atom.idx], [0.0, x0, y0, z0])

        return tug_force 
    @staticmethod 
    def get_force():
        expr = "tug_k * ((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
        custom_force = CustomCompoundBondForce(1, expr)  # One particle per bond
        custom_force.addPerBondParameter("tug_k")
        custom_force.addPerBondParameter("x0") #positions to be tugged to
        custom_force.addPerBondParameter("y0") 
        custom_force.addPerBondParameter("z0")
        return custom_force

class SaltBridgeForce(Force):
    name = 'salt_bridge_force'
    @classmethod 
    def apply(cls, simulation_object):
        force = cls.get_force() 
        return force
    @staticmethod
    def get_force():
        expression = 'saltbridge_k * (distance(p1, p2) - salt_bridge_r0)^2'  # p1, p2 are indices of the particles in the bond
        distance_force = CustomCompoundBondForce(2, expression)  # 2 indicates two particles are involved in each bond
        # Add parameters for force constant 'k' and target distance 'r0'
        distance_force.addPerBondParameter("saltbridge_k")   # Force constant
        distance_force.addPerBondParameter("salt_bridge_r0")  # Target distance
        return distance_force


class HalogenBondForce(Force):
    name = 'halogen_bond_force'
    @classmethod 
    def apply(cls, simulation_object):
        force = cls.get_force() 
        return force 
    
    @staticmethod 
    def get_force():
        expression = '0.5 * halogen_k_r * (r - halogen_r0)^2 + 0.5 * halogen_k_theta1 * (theta1 - halogen_theta1_0)^2 + 0.5 * halogen_k_theta2 * (theta2 - halogen_theta2_0)^2; r = distance(p2, p3); theta1 = angle(p1, p2, p3); theta2 = angle(p2, p3, p4);'
        force = CustomCompoundBondForce(4, expression) 
        force.addPerBondParameter("halogen_k_r")
        force.addPerBondParameter("halogen_k_theta1")
        force.addPerBondParameter("halogen_k_theta2")
        force.addPerBondParameter("halogen_r0")
        force.addPerBondParameter("halogen_theta1_0")
        force.addPerBondParameter("halogen_theta2_0")
        return force
        
        

class HbondDistForce(Force):
    name = 'hbond_tug_dist_force'
    @classmethod 
    def apply(cls, simulation_object):
        force = cls.get_force()
        return force
    
    @staticmethod
    def get_force():
        expression = 'k * (distance(p1, p2) - r0)^2'  # p1, p2 are indices of the particles in the bond
        distance_force = CustomCompoundBondForce(2, expression)  # 2 indicates two particles are involved in each bond
        # Add parameters for force constant 'k' and target distance 'r0'
        distance_force.addPerBondParameter("k")   # Force constant
        distance_force.addPerBondParameter("r0")  # Target distance
        return distance_force

class WeakHbondDistForce(Force):
    name = 'weak_hbond_tug_dist_force'
    @classmethod 
    def apply(cls, simulation_object):
        force = cls.get_force()
        return force
    
    @staticmethod
    def get_force():
        expression = 'weak_hb_k * (distance(p1, p2) - weak_hb_r0)^2'  # p1, p2 are indices of the particles in the bond
        distance_force = CustomCompoundBondForce(2, expression)  # 2 indicates two particles are involved in each bond
        # Add parameters for force constant 'k' and target distance 'r0'
        distance_force.addPerBondParameter("weak_hb_k")   # Force constant
        distance_force.addPerBondParameter("weak_hb_r0")  # Target distance
        return distance_force

class WeakHbondAngleForce(Force):
    name = 'weak_hbond_tug_angle_force'
    @classmethod 
    def apply(cls, simulation_object):
        force = cls.get_force()
        return force
    @staticmethod
    def get_force():
        # Define the energy expression
        expression = 'weak_hb_angle_k * (theta - weak_hb_angle_theta0)^2'
        # Create a CustomAngleForce object with the specified expression
        angle_force = CustomAngleForce(expression)
        # Add parameters for the force constant 'k' and the target angle 'theta0'
        angle_force.addPerAngleParameter("weak_hb_angle_k")     # Force constant
        angle_force.addPerAngleParameter("weak_hb_angle_theta0")  # Target angle
        return angle_force


class CationPiForce(Force):
    name = 'cation_pi_force' 
    @classmethod 
    def apply(cls, 
              simulation_object):
        force = cls.get_force() 
        return force 
    
    @staticmethod 
    def get_force():
        energy_expression = 'U_total; U_total = U_distance + U_angle + U_offset; U_distance = 0.5 * cation_pi_k_distance * (r_centroid - cation_pi_r0)^2; U_angle = 0.5 * cation_pi_k_angle * (theta - cation_pi_theta0)^2; U_offset = 0.5 * cation_pi_k_offset * (offset - cation_pi_offset0)^2; offset = sqrt(r_centroid^2 - projection^2); theta = acos(cos_theta); cos_theta = projection / r_centroid; projection = delta_cx * n1x_unit + delta_cy * n1y_unit + delta_cz * n1z_unit; r_centroid = sqrt(delta_cx^2 + delta_cy^2 + delta_cz^2); delta_cx = x4 - centroidx; delta_cy = y4 - centroidy; delta_cz = z4 - centroidz; n1x_unit = n1x / n1_norm; n1y_unit = n1y / n1_norm; n1z_unit = n1z / n1_norm; n1_norm = sqrt(n1x^2 + n1y^2 + n1z^2) + 1e-8; n1x = u1y*v1z - u1z*v1y; n1y = u1z*v1x - u1x*v1z; n1z = u1x*v1y - u1y*v1x; u1x = x2 - x1; u1y = y2 - y1; u1z = z2 - z1; v1x = x3 - x1; v1y = y3 - y1; v1z = z3 - z1; centroidx = (x1 + x2 + x3)/3; centroidy = (y1 + y2 + y3)/3; centroidz = (z1 + z2 + z3)/3;'
        force = CustomCompoundBondForce(4, energy_expression)
        force.addGlobalParameter('cation_pi_r0', 0.4 * unit.nanometer)                  # Equilibrium distance
        force.addGlobalParameter('cation_pi_theta0', 0.0 * unit.radian)          # Equilibrium angle
        force.addGlobalParameter('cation_pi_offset0', 0.0 * unit.nanometer)  
        
        force.addPerBondParameter('cation_pi_k_distance')  # Replace with your value
        force.addPerBondParameter('cation_pi_k_offset')
        force.addPerBondParameter('cation_pi_k_angle')
        
        return force
        
        
class PiPiDistForce(Force):
    name = 'pipi_tug_dist_force'
    @classmethod 
    def apply(cls,
              simulation_object
              ):
        
        force = cls.get_force()
        return force
        
    @staticmethod 
    def get_force():
        # Define the energy expression
        energy_expression = 'U_total; U_total = U_distance + U_angle + U_offset; U_distance = 0.5 * k_distance * (r_centroid - r0)^2; U_angle = 0.5 * k_angle * (theta - theta0)^2; U_offset = 0.5 * k_offset * (offset - offset0)^2; offset = sqrt(max(0, r_centroid^2 - projection^2)); projection = delta_cx * n1x_unit + delta_cy * n1y_unit + delta_cz * n1z_unit; r_centroid = sqrt(delta_cx^2 + delta_cy^2 + delta_cz^2); delta_cx = centroid2x - centroid1x; delta_cy = centroid2y - centroid1y; delta_cz = centroid2z - centroid1z; theta = acos(n1_dot_n2); n1_dot_n2 = max(-1.0, min(n1_dot_n2_raw, 1.0)); n1_dot_n2_raw = n1x_unit * n2x_unit + n1y_unit * n2y_unit + n1z_unit * n2z_unit; n1x_unit = n1x / n1_norm; n1y_unit = n1y / n1_norm; n1z_unit = n1z / n1_norm; n2x_unit = n2x / n2_norm; n2y_unit = n2y / n2_norm; n2z_unit = n2z / n2_norm; n1_norm = sqrt(n1x^2 + n1y^2 + n1z^2) + 1e-8; n2_norm = sqrt(n2x^2 + n2y^2 + n2z^2) + 1e-8; n1x = u1y*v1z - u1z*v1y; n1y = u1z*v1x - u1x*v1z; n1z = u1x*v1y - u1y*v1x; n2x = u2y*v2z - u2z*v2y; n2y = u2z*v2x - u2x*v2z; n2z = u2x*v2y - u2y*v2x; u1x = x2 - x1; u1y = y2 - y1; u1z = z2 - z1; v1x = x3 - x1; v1y = y3 - y1; v1z = z3 - z1; u2x = x5 - x4; u2y = y5 - y4; u2z = z5 - z4; v2x = x6 - x4; v2y = y6 - y4; v2z = z6 - z4; centroid1x = (x1 + x2 + x3)/3; centroid1y = (y1 + y2 + y3)/3; centroid1z = (z1 + z2 + z3)/3; centroid2x = (x4 + x5 + x6)/3; centroid2y = (y4 + y5 + y6)/3; centroid2z = (z4 + z5 + z6)/3;'


        # Create the CustomCompoundBondForce
        force = CustomCompoundBondForce(6, energy_expression)

        # Add global parameters with appropriate units
        force.addPerBondParameter('k_distance')
        force.addPerBondParameter('k_angle')
        force.addPerBondParameter('k_offset')
        
        # Add per-bond parameters for equilibrium values
        force.addPerBondParameter('r0')       # Equilibrium centroid distance (in nm)
        force.addPerBondParameter('offset0')  # Equilibrium offset distance (in nm)
        force.addPerBondParameter('theta0')   # Equilibrium angle in radians

        return force



def get_pipi_tug_indexes(model, atoms_to_index):
    selected_atoms = [atom for residue in model.residues for atom in residue.atoms if atom.selected]
    
    if len(selected_atoms) <= 1:
        return None 
    
    #get two distinct rings!!
    rings = [i.rings() for i in selected_atoms if len(i.rings) > 0]
    flattened_rings = list(set([i for j in rings for i in j]))
    
    if len(flattened_rings) == 2:
        
        ring_pair_atoms = []
        
        for ring in flattened_rings:
           
            
            openff_indexes = [atoms_to_index[i] for i in ring.atoms]
            ring_pair_atoms.append([ring.atoms, openff_indexes])
        
        print('HAVE A RING PAIR......')
        
        return ring_pair_atoms
    #TODO!! other things!!


class HbondAngleForce(Force):
    name = 'hbond_tug_angle_force'
    @classmethod 
    def apply(cls, simulation_object):
        force = cls.get_force()
        return force
    @staticmethod
    def get_force():
        # Define the energy expression
        expression = 'k * (theta - theta0)^2'
        # Create a CustomAngleForce object with the specified expression
        angle_force = CustomAngleForce(expression)
        # Add parameters for the force constant 'k' and the target angle 'theta0'
        angle_force.addPerAngleParameter("k")     # Force constant
        angle_force.addPerAngleParameter("theta0")  # Target angle
        return angle_force
    


def get_mmpbsa_complex_system(complex_structure, implicit_solvent, constraints=HBonds):
    
    complex_system = complex_structure.createSystem(nonbondedMethod=NoCutoff,
                                                     nonbondedCutoff=9.0 * unit.angstrom,
                                                     constraints=constraints, 
                                                     removeCMMotion=False, 
                                                     implicitSolvent=implicit_solvent.value)
    
    return complex_system


def get_complex_system(complex_structure, 
                       parameters = None, 
                       ions_to_add= {'NA': 4, 'CL': 4, 'MG': 4, 'CA': 4, 'MN': 4, 'FE': 4, 'CU': 4, 'ZN': 4}):
    
    if parameters is None:
        complex_system = complex_structure.createSystem(nonbondedMethod=NoCutoff,
                                                         nonbondedCutoff=9.0 * unit.angstrom,
                                                         constraints=HBonds, 
                                                         removeCMMotion=False, 
                                                         rigidWater=True)
    
    
    solvent = parameters.get_parameter('implicitSolvent')
    
    
    if solvent is not None:
        solvent = solvent.value
    
        
    if solvent is not None and solvent  != 'None':
        complex_system = complex_structure.createSystem(
            nonbondedMethod=NoCutoff,
            nonbondedCutoff=9.0 * unit.angstrom,
            constraints=HBonds,
            removeCMMotion=False,
            implicitSolvent=solvent
        )
        
    else:
        complex_system = complex_structure.createSystem(nonbondedMethod=NoCutoff,
                                                         nonbondedCutoff=9.0 * unit.angstrom,
                                                         constraints=HBonds, 
                                                         removeCMMotion=False, 
                                                         rigidWater=True)
    
    return complex_system


def get_complex_structure(path):

    complex_structue_prmtop = os.path.join(path, 'complex_structure.prmtop')
    complex_structue_inpcrd = os.path.join(path, 'complex_structure.inpcrd')
    if os.path.exists(complex_structue_prmtop) and os.path.exists(complex_structue_inpcrd ):
        complex_structure = parmed.load_file(complex_structue_prmtop, xyz=complex_structue_inpcrd)
    
    return complex_structure
    



def pin_force():
    expr = "pin_k * ((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
    f = CustomCompoundBondForce(1, expr)
    f.addPerBondParameter("pin_k")
    f.addPerBondParameter("x0")
    f.addPerBondParameter("y0")
    f.addPerBondParameter("z0")
    return f

def pin_atoms(idx_to_pin, struct, pin_k=5000):

    pin_f = pin_force()
    for atom_index in idx_to_pin:
        
        atm = struct.atoms[atom_index]
        x0, y0, z0 = Quantity(
            value=[atm.xx, atm.xy, atm.xz], unit=unit.angstrom)
        #x0, y0, z0 = atm.xx, atm.xy, atm.xz
        pin_f.addBond([atom_index], [pin_k, x0, y0, z0])
    return pin_f


def get_chimera_slice(atom_coords, origin, apix):
    
    min_coords, max_coords = calculate_box(atom_coords)
    min_indices = get_array_indices(origin, apix, min_coords)
    max_indices = get_array_indices(origin, apix, max_coords)
    
    return ([min_indices[0], min_indices[1], min_indices[2]], 
            [max_indices[0], max_indices[1], max_indices[2]], 
            [1, 1, 1])

def calculate_box( atom_coordinates, padding = 6.0):
   
    min_x, min_y, min_z = atom_coordinates[0]
    max_x, max_y, max_z = atom_coordinates[1]
    
    for coord in atom_coordinates[1:]:
        if coord[0] < min_x:
            min_x = coord[0] 
        
        if coord[1] < min_y:
            min_y = coord[1] 
        
        if coord[2] < min_z:
            min_z = coord[2] 
        
        if coord[0] > max_x:
            max_x = coord[0] 
        
        if coord[1] > max_y:
            max_y = coord[1] 
        
        if coord[2] > max_z:
            max_z = coord[2] 
        
    
    min_coords = np.array([min_x, min_y, min_z]) 
    max_coords = np.array([max_x, max_y, max_z])
    
    return min_coords, max_coords


def get_array_indices(origin, apix, coordinate):
    """
    Convert a coordinate in angstroms to array indices based on the origin and apix.

    Args:
    origin (tuple): The (x, y, z) origin coordinates in angstroms.
    apix (float): The scale of angstroms per pixel.
    coordinate (tuple): The (x, y, z) coordinate in angstroms to be converted.

    Returns:
    tuple: The (x, y, z) indices in the array corresponding to the given coordinate.
    """
    # Calculate the indices by subtracting the origin from the coordinate, dividing by apix, and rounding
    indices = tuple(int(round((coord - orig) / scale)) for coord, orig, scale in zip(coordinate, origin, apix))
    
    return indices


def get_position_vector_from_atom(atom):
    coord = atom.coord 
    n_vec = vec3.Vec3(
        x=coord[0], y=coord[0], z=coord[0])
    n_quant = unit.quantity.Quantity(value=n_vec, unit=unit.angstrom)
    return n_quant



def get_model_from_complex_structure(complex_structure, chimerax_model, distance_threshold=0.01):
    atom_to_index = []

    # Build a mapping from (residue name, atom name) to simulation atoms
    sim_atoms_dict = {}
    for sim_atom in complex_structure.atoms:
        key = (sim_atom.residue.name, sim_atom.name)
        sim_atoms_dict.setdefault(key, []).append(sim_atom)

    # Build a mapping from (residue name, atom name) to ChimeraX atoms
    chimerax_atoms_dict = {}
    for atom in chimerax_model.atoms:
        key = (atom.residue.name, atom.name)
        chimerax_atoms_dict.setdefault(key, []).append(atom)

    # Match atoms between ChimeraX model and simulation
    for key, chimera_atoms in chimerax_atoms_dict.items():
        sim_atoms = sim_atoms_dict.get(key, [])
        if not sim_atoms:
            continue  # No matching atoms in simulation

        # For each ChimeraX atom, find the closest simulation atom
        for atom in chimera_atoms:
            min_distance = None
            best_sim_atom = None
            # Get ChimeraX atom position (assuming atom.coord exists)
            
            atom_pos = np.array(atom.coord)
            

            for sim_atom in sim_atoms:
                # Get simulation atom position from sim_atom.xx, sim_atom.xy, sim_atom.xz
                sim_atom_pos = np.array([sim_atom.xx, sim_atom.xy, sim_atom.xz])
                distance = np.linalg.norm(atom_pos - sim_atom_pos)

                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    best_sim_atom = sim_atom

            if min_distance is not None and min_distance < distance_threshold:
                
                atom_to_index.append([atom, best_sim_atom.idx])

    return chimerax_model, atom_to_index




def get_model_from_complex_structure_old(complex_structure, chimerax_model):
    #new_model = AtomicStructure(self.session)
    atom_to_index = []
    
    
    for sim_residue, chimera_residue in zip(complex_structure.residues, chimerax_model.residues):

        for atom in chimera_residue.atoms:
            for sim_atom in sim_residue.atoms:
                if atom.name == sim_atom.name:
                    atom_to_index.append([atom, sim_atom.idx])
    
    #get hbond stuff
    
    return chimerax_model, atom_to_index
    