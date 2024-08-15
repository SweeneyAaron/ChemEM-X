import parmed
from openmm import XmlSerializer
from openmm import LangevinIntegrator, Platform
from openmm import app
from openmm import unit
from openmm import MonteCarloBarostat, XmlSerializer, app, unit,CustomNonbondedForce, CustomCompoundBondForce,CustomAngleForce, Continuous3DFunction, vec3, Vec3, CustomCentroidBondForce     
from openmm import LocalEnergyMinimizer
from scipy.ndimage import gaussian_filter
from openmm.unit.quantity import Quantity
import os
import numpy as np
from chimerax.atomic.structure import AtomicStructure
from chimerax.geometry import distance as get_distance
from rdkit import Chem
from openmm.app import  NoCutoff, HBonds
from chimerax.core.commands import run 

RESIDUE_NAMES = ['CYS','MET','GLY','ASP','ALA','VAL','PRO','PHE','ASN','THR',
                 'HIS','GLN','ARG','TRP','ILE','SER','LYS','LEU','GLU','TYR']

HBOND_ELEMENTS = ['O', 'N', 'S']

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
        self.temperature = 300
        self.temperature_step = 1
        self.pressure = 1*unit.atmosphere
        self.heating_interval = 10
        self.tug_k = 10000
        self.hbond_dist_k = 1000
        self.hbond_angle_k = 100
        self.pipi_dist_k = 1000
        self.pipi_angle_k = 100 
        self.pipi_offset_k = 100
        
        self.forces = {}
        self.force_group = 0
        
        #hbond tugs 
        self.hbond_tug_atoms = {}
        self.pipi_p_tug_rings = {}
        #self.get_complex_structure()
        self.densmap_when_atoms_are_inside()
        
        self.debug = []
    
    
    @classmethod 
    def from_filepath(cls, session, file_path, densmap, parameters = None, platform_name = 'OpenCL'):
        complex_structure = get_complex_structure(file_path)
        complex_system = get_complex_system(complex_structure,parameters = parameters)
        
        
        return cls(session, 
                   complex_system,
                   complex_structure,
                   densmap,
                   platform_name = platform_name)
    
    def chimera_x_atom_to_simulation_consistancy(self, model, atoms_to_position_index_as_dic):
        
            
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True) #in nanometers!!!
        
        for residue in model.residues:
            for atom in residue.atoms:
                
                if atom not in atoms_to_position_index_as_dic:
                    com = f'del {atom.atomspec}'
                    run(self.session, com)
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
    
    def get_model_from_complex_structure(self, chimerax_model):
        #new_model = AtomicStructure(self.session)
        atom_to_index = []
        
        
        for sim_residue, chimera_residue in zip(self.complex_structure.residues, chimerax_model.residues):

            for atom in chimera_residue.atoms:
                for sim_atom in sim_residue.atoms:
                    if atom.name == sim_atom.name:
                        atom_to_index.append([atom, sim_atom.idx])
        
        #get hbond stuff
        
        return chimerax_model, atom_to_index
    
    def set_ring_groups(self, chimerax_model, atoms_to_idx):
        atoms = [atom for residue in chimerax_model.residues for atom in residue.atoms]
        
        #get two distinct rings!!
        rings = [i.rings() for i in atoms if len(i.rings()) > 0]
        flattened_rings = list(set([i for j in rings for i in j]))

        ring_idxs = []
            
        for ring in flattened_rings:
            atoms = ring.atoms
            openff_indexes = sorted([atoms_to_idx[i] for i in atoms])
            if openff_indexes not in ring_idxs:
                ring_idxs.append(openff_indexes)
            
        self.ring_idxs = ring_idxs
        #TODO!! other things!!
        
    
    def add_force(self, AddForce):
        
        force = AddForce().apply(self)
        force_idx = self.complex_system.addForce(force)
        self.complex_system.getForce(force_idx).setForceGroup(self.force_group)
        self.force_group += 1
        self.forces[force_idx] = force
        setattr(self, AddForce.name, force)
        
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
        self.simulation.minimizeEnergy()
    
    def get_positions(self):
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        positions = positions.value_in_unit(unit.angstrom)
        return positions 
    
    def step(self, step):
        self.simulation.step(step)
    
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
        
     
    def update_pipi_p_tug(self,
                          ring_index_pairs, 
                          dist_k = None, 
                          angle_k = None,
                          offset_k = None):
        
        ring_1, ring_2 = ring_index_pairs
        
        if dist_k is None:
            dist_k = self.pipi_dist_k
        
        if angle_k is None:
            angle_k = self.pipi_angle_k
        
        if offset_k is None:
            offset_k = self.pipi_offset_k
            
        
        dist = 3.5 * unit.angstrom
        
        if str(sorted(ring_index_pairs)) in self.pipi_p_tug_rings:
            pass
        else:
            group_1_index = self.ring_idxs.index(ring_1)
            group_2_index = self.ring_idxs.index(ring_2)
            pipi_dist_int = self.pipi_tug_dist_force.addBond([group_1_index, group_2_index], [dist_k, dist])
            self.simulation.context.reinitialize(preserveState=True)
            self.pipi_p_tug_rings[str(sorted(ring_index_pairs))] = [pipi_dist_int]
                                            
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


class PiPiDistForce(Force):
    name = 'pipi_tug_dist_force'
    @classmethod 
    def apply(cls, simulation_object):
        force = cls.get_force() 
        for group in simulation_object.ring_idxs:
            force.addGroup(group, [1 for i in group])
        return force
    
    @staticmethod 
    def get_force():
        expression = 'k * (distance(g1,g2) - r0)^2'
        force = CustomCentroidBondForce(2, expression)
        force.addPerBondParameter('k')
        force.addPerBondParameter('r0') 
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
    


def get_complex_system(complex_structure, parameters = None):
    
    
    if parameters is None:
        complex_system = complex_structure.createSystem(nonbondedMethod=NoCutoff,
                                                         nonbondedCutoff=9.0 * unit.angstrom,
                                                         constraints=HBonds, 
                                                         removeCMMotion=False, 
                                                         rigidWater=True)
    
    
    solvent = parameters.get_parameter('solvent')
    
        
    
    if solvent is not None:
        complex_system = complex_structure.createSystem(
            nonbondedMethod=NoCutoff,
            nonbondedCutoff=9.0 * unit.angstrom,
            constraints=HBonds,
            removeCMMotion=False,
            implicitSolvent=solvent.value
        )
        
    else:
        complex_system = complex_structure.createSystem(nonbondedMethod=NoCutoff,
                                                         nonbondedCutoff=9.0 * unit.angstrom,
                                                         constraints=HBonds, 
                                                         removeCMMotion=False, 
                                                         rigidWater=True)
        
    return complex_system

def get_complex_structure(path):
    
    '''
    complex_system = os.path.join(path, 'complex_system.xml' )
    if os.path.exists(complex_system):
        with open(complex_system,'r' ) as f:
            complex_system = XmlSerializer.deserialize(f.read())
    '''
    complex_structue_prmtop = os.path.join(path, 'complex_structure.prmtop')
    complex_structue_inpcrd = os.path.join(path, 'complex_structure.inpcrd')
    if os.path.exists(complex_structue_prmtop) and os.path.exists(complex_structue_inpcrd ):
        complex_structure = parmed.load_file(complex_structue_prmtop, xyz=complex_structue_inpcrd)
    
    return complex_structure #complex_system, complex_structure
    



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

def get_model_from_complex_structure(complex_structure, chimerax_model):
    #new_model = AtomicStructure(self.session)
    atom_to_index = []
    
    
    for sim_residue, chimera_residue in zip(complex_structure.residues, chimerax_model.residues):

        for atom in chimera_residue.atoms:
            for sim_atom in sim_residue.atoms:
                if atom.name == sim_atom.name:
                    atom_to_index.append([atom, sim_atom.idx])
    
    #get hbond stuff
    
    return chimerax_model, atom_to_index
    