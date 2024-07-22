import parmed
from openmm import XmlSerializer
from openmm import LangevinIntegrator, Platform
from openmm import app
from openmm import unit
from openmm import MonteCarloBarostat, XmlSerializer, app, unit, CustomCompoundBondForce, Continuous3DFunction, vec3, Vec3
from scipy.ndimage import gaussian_filter
from openmm.unit.quantity import Quantity
import os
import numpy as np
from chimerax.atomic.structure import AtomicStructure

class Simulation:
    
    def __init__(self, session, filepath, densmap, platform_name = 'OpenCL'):
        #Do properly !!
        import chimerax 
        openmm_plugins_dir = os.path.join(chimerax.app_lib_dir, 'plugins')
        Platform.loadPluginsFromDirectory(openmm_plugins_dir)
        
        self.session = session
        self.filepath = filepath
        self.densmap = densmap
        self.platform = platform_name
        self.get_complex_structure()
        self.set_forces()
        
        
    def get_complex_structure(self):
        
        complex_system = os.path.join(self.filepath, 'complex_system.xml' )
        if os.path.exists(complex_system):
            with open(complex_system,'r' ) as f:
                self.complex_system = XmlSerializer.deserialize(f.read())
        
        complex_structue_prmtop = os.path.join(self.filepath, 'complex_structure.prmtop')
        complex_structue_inpcrd = os.path.join(self.filepath, 'complex_structure.inpcrd')
        if os.path.exists(complex_structue_prmtop) and os.path.exists(complex_structue_inpcrd ):
            self.complex_structure = parmed.load_file(complex_structue_prmtop, xyz=complex_structue_inpcrd)
    
    def get_model_from_complex_structure_old(self, chimerax_model):
        new_model = AtomicStructure(self.session)
        atom_to_index = []
        
        
        for sim_residue, chimera_residue in zip(self.complex_structure.residues, chimerax_model.residues):
            new_residue = new_model.new_residue(chimera_residue.name, 
                                                chimera_residue.chain_id, 
                                                chimera_residue.number)
            
            
            for atom in sim_residue.atoms:
                new_atom = new_model.new_atom(atom.name, atom.element_name)
                new_residue.add_atom(new_atom)
                atom_coord = np.array([atom.xx, atom.xy, atom.xz])
                new_atom.coord = atom_coord
                atom_to_index.append([new_atom, atom.idx])
                
        return new_model, atom_to_index
    
    def get_model_from_complex_structure(self, chimerax_model):
        #new_model = AtomicStructure(self.session)
        atom_to_index = []
        
        
        for sim_residue, chimera_residue in zip(self.complex_structure.residues, chimerax_model.residues):

            for atom in chimera_residue.atoms:
                for sim_atom in sim_residue.atoms:
                    if atom.name == sim_atom.name:
                        atom_to_index.append([atom, sim_atom.idx])
                
        return chimerax_model, atom_to_index

    def set_forces(self):
        force_group = 0
        for force in self.complex_system.getForces():
            force.setForceGroup(force_group)
            force_group += 1
        
        import time
        t1 = time.perf_counter()
        
        if self.densmap is not None:
            current_map = self.densmap 
            origin, apix = current_map.data_origin_and_step()
            
            origin = np.array(origin)
            apix = np.array(apix)
            copy_map = current_map.data.matrix()
            c_level = 0.0
            copy_map =  copy_map * (copy_map >= c_level)
            copy_map = copy_map / np.amax(copy_map)
   
            global_k = 75.0

            mp = map_potential_force_field(copy_map, origin, apix, copy_map.shape, global_k, blur=0)
            # add bonds to force field
            
            #just move ligand
            for idx in range(len(self.complex_structure.positions) - 32, len(self.complex_structure.positions) ):

                mp.addBond([idx], [1.0])

            # add force here potential for map
            force_idx = self.complex_system.addForce(mp)
            self.complex_system.getForce(force_idx).setForceGroup(force_group)
            force_group += 1
        
        t2 = time.perf_counter() - t1 
        print('RUNTIME:', t2)
        p_force = pin_atoms(
            [i for i in range(0, len(self.complex_structure.positions) - 32)], 
            self.complex_structure)
    
    
        pin_idx = self.complex_system.addForce(p_force)
        
        self.complex_system.getForce(pin_idx).setForceGroup(force_group)
        
        #complex_system.getForce(pin_idx).setForceGroup(force_group)
        
        integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds,
                                        1.0*unit.femtoseconds)
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
        
    
def pin_force():
    expr = "pin_k * ((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
    f = CustomCompoundBondForce(1, expr)
    f.addPerBondParameter("pin_k")
    f.addPerBondParameter("x0")
    f.addPerBondParameter("y0")
    f.addPerBondParameter("z0")
    return f

def pin_atoms(idx_to_pin, struct, pin_k=500):

    pin_f = pin_force()
    for atom_index in idx_to_pin:
        
        atm = struct.atoms[atom_index]
        x0, y0, z0 = Quantity(
            value=[atm.xx, atm.xy, atm.xz], unit=unit.angstrom)
        #x0, y0, z0 = atm.xx, atm.xy, atm.xz
        pin_f.addBond([atom_index], [pin_k, x0, y0, z0])
    return pin_f

def map_potential_force_field(m, origin, apix, box_size, global_k, blur=0):
     
    f = CustomCompoundBondForce(1, '')
    d3d_func = compute_map_field(m,origin, apix, box_size, blur)
    f.addTabulatedFunction(name='map_potential', function=d3d_func)
    f.addGlobalParameter(name='global_k', defaultValue=global_k)
    f.addPerBondParameter(name='individual_k')
    f.setEnergyFunction('-global_k * individual_k * map_potential(z1,y1,x1)')
    return f

def compute_map_field(m, origin, apix, box_size, blur=0, d3d_func=None):
    #f = CustomCompoundBondForce(1,'')
    #     d3d_func = Discrete3DFunction(*m.fullMap.shape, m.fullMap.ravel().copy())
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
    
    
    
    