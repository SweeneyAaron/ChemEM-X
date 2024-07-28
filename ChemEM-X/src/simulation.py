import parmed
from openmm import XmlSerializer
from openmm import LangevinIntegrator, Platform
from openmm import app
from openmm import unit
from openmm import MonteCarloBarostat, XmlSerializer, app, unit,CustomNonbondedForce, CustomCompoundBondForce,CustomAngleForce, Continuous3DFunction, vec3, Vec3
from openmm import LocalEnergyMinimizer
from scipy.ndimage import gaussian_filter
from openmm.unit.quantity import Quantity
import os
import numpy as np
from chimerax.atomic.structure import AtomicStructure
from chimerax.geometry import distance as get_distance
from rdkit import Chem


RESIDUE_NAMES = ['CYS','MET','GLY','ASP','ALA','VAL','PRO','PHE','ASN','THR',
                 'HIS','GLN','ARG','TRP','ILE','SER','LYS','LEU','GLU','TYR']

HBOND_ELEMENTS = ['O', 'N', 'S']


#TO SPEED UP 
#TAKE ONLY THE STRUCTURAL ELEMENTS THAT MAKE UP THE BINDING SITE
#PIN PROTEIN ATOMS 
#FORCE TO KEEP LIGANDS IN A ZONE 
#RESET ALL OF THE FORCE GROUPS 1,2,3 TO JUSt INCLUDE WHAt YOU WANT

class ChemEMSimulation:
    def __init__(self, 
                 session, 
                 filepath, 
                 densmap, 
                 centroid, 
                 atoms_to_position_idx,
                 radius = 20, 
                 platform_name = 'OpenCL'):
        
        self.session = session
        self.filepath = filepath
        self.densmap = densmap
        self.platform = platform_name
        self.centroid = centroid 
        self.radius = radius 
        self.atoms_to_position_idx = atoms_to_position_idx
        
        self.get_complex_structure()
        self.get_simualtion_hbond_pairs(self.atoms_to_position_idx)
        
        self.set_forces()
        
        #copied over 
        self.temperature = 300
        self.temperature_step = 1
        self.pressure = 1*unit.atmosphere
        self.heating_interval = 10
        self.tug_k = 10000
        self.hbond_dist_k = 1000
        self.hbond_angle_k = 100
        
        self.set_simulation()
        #self.densmap_when_atoms_are_inside()
        
        
        
    def get_complex_structure(self):
        
        complex_system = os.path.join(self.filepath, 'complex_system.xml' )
        if os.path.exists(complex_system):
            with open(complex_system,'r' ) as f:
                self.complex_system = XmlSerializer.deserialize(f.read())
        
        complex_structue_prmtop = os.path.join(self.filepath, 'complex_structure.prmtop')
        complex_structue_inpcrd = os.path.join(self.filepath, 'complex_structure.inpcrd')
        if os.path.exists(complex_structue_prmtop) and os.path.exists(complex_structue_inpcrd ):
            self.complex_structure = parmed.load_file(complex_structue_prmtop, xyz=complex_structue_inpcrd)
        
        
    
    def set_forces(self):
        force_group = 0
        for force in self.complex_system.getForces():
            force.setForceGroup(force_group)
            force_group += 1
        self.complex_system.removeForce(force_group - 1)
        
        
        pos_to_atom = {i[1] : i[0] for i in self.atoms_to_position_idx}
        #Hbond Force ChemEM
        '''
        hbond_pairs = []
        
        for donor, hydrogen, acceptor in self.hbond_idxs:
            if get_distance(pos_to_atom[donor].coord, self.centroid) <= self.radius:
                if get_distance(pos_to_atom[acceptor].coord, self.centroid) <= self.radius:
                    #Add Hbond
                    hbond_pairs.append([donor, hydrogen, acceptor])
        
        chemem_hbond_dist_force = initialize_chemem_hbond_dist_force(hbond_pairs, self.complex_structure)
        chemem_hbond_dist_force_idx = self.complex_system.addForce(chemem_hbond_dist_force)
        self.complex_system.getForce(chemem_hbond_dist_force_idx).setForceGroup(force_group)
        force_group += 1
        self.chemem_hbond_dist_force_idx = chemem_hbond_dist_force_idx
        self.chemem_hbond_dist_force  =  chemem_hbond_dist_force 
        
        #VDWForce
        protein_atoms = [] #non H
        ligand_atoms = [] # non H
        for idx, atom in enumerate(self.complex_structure.atoms):
            atom_pos = self.complex_structure.positions[idx]
            atom_pos = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
            if atom.element_name != 'H':
                if get_distance(atom_pos, self.centroid) <= self.radius:
                
                    if atom.residue.name in RESIDUE_NAMES:
                        protein_atoms.append([idx,atom])
                    else:
                        ligand_atoms.append([idx,atom])
        
        ligand_protein_atom_pairs = []
        for idx1, atom1 in ligand_atoms:
            for idx2, atom2 in protein_atoms:
                ligand_protein_atom_pairs.append([idx1, atom1, idx2, atom2])
                
        '''
        vdw_force = initilize_chemem_vdw_force(self.complex_structure)#, ligand_protein_atom_pairs)
        vdw_force_idx = self.complex_system.addForce(vdw_force)
        self.complex_system.getForce(vdw_force_idx).setForceGroup(force_group)
        force_group += 1
        self.vdw_force = vdw_force 
        self.vdw_force_idx = vdw_force_idx
        
        #pin force
        pin_idxs = []
        for idx, atom in enumerate(self.complex_structure.atoms):
            if atom.residue.name in RESIDUE_NAMES:
                pin_idxs.append(idx)
        
        pin_force = pin_atoms(pin_idxs, self.complex_structure, pin_k=5000)
        pin_force_idx = self.complex_system.addForce(pin_force)
        self.complex_system.getForce(pin_force_idx).setForceGroup(force_group)
        force_group += 1
        self.pin_force = pin_force
        self.pin_force_idx = pin_force_idx
        
        #soft barrier force
        ligand_atoms = []
        for  atom in self.complex_structure.atoms:
            if not atom.residue.name in RESIDUE_NAMES:
                ligand_atoms.append(atom.idx)
                
        soft_barrier = initialise_soft_barrier(ligand_atoms)
        soft_barrier_idx = self.complex_system.addForce(soft_barrier)
        self.complex_system.getForce(soft_barrier_idx).setForceGroup(force_group)
        force_group += 1
        self.soft_barrier = soft_barrier
        self.soft_barrier_idx = soft_barrier_idx
        

    def set_simulation(self):
        integrator = LangevinIntegrator(self.temperature*unit.kelvin, 1.0/unit.picoseconds,
                                        1.0*unit.femtoseconds)
        self.integrator = integrator
        
        
        _platform = Platform.getPlatformByName(self.platform)
        simulation = app.Simulation(
            self.complex_structure.topology, self.complex_system, integrator, platform=_platform)
        
        simulation.context.setPositions(self.complex_structure.positions)
        
        # Specify the force groups to use (e.g., only groups 0 and 1)
        #forceGroupsToUse = {0, 1, 2, 4, 5, 6, 7}
        #self.integrator.setIntegrationForceGroups(forceGroupsToUse)
        self.simulation = simulation
    
    
    def minimize_energy(self):
        # Perform energy minimization with restricted force groups
        import time
        t1 = time.perf_counter()
        LocalEnergyMinimizer.minimize(self.simulation.context, tolerance=1e-4 * unit.kilojoule / unit.mole)
        t2 = time.perf_counter() - t1 
        print('Min TIME:', t2)
        
    def write_pdb(self, file):
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions()

        # Write the minimized structure to a PDB file
        with open(file, 'w') as pdb_file:
            app.PDBFile.writeFile(self.simulation.topology, positions, pdb_file)


    def get_chemem_hbond_dist_score(self):
        context = self.simulation.context

        # Get the state of the system with energy information, specifying the group of forces
        state = context.getState(getEnergy=True, groups={self.chemem_hbond_dist_force_idx})
        energy = state.getPotentialEnergy()
        print('ChemEM_Hbond:', energy)
        return energy
        
    def get_chemem_vdw_score(self):
        context = self.simulation.context

        # Get the state of the system with energy information, specifying the group of forces
        state = context.getState(getEnergy=True, groups={self.vdw_force_idx})
        energy = state.getPotentialEnergy()
        print('ChemEM_VDW:', energy)
        return energy
        
    
    
    def get_simualtion_hbond_pairs(self, atom_to_position_idx ):
        ligand_hbond_atoms = []
        protein_hbond_atoms = []
        
        for atom, idx in atom_to_position_idx:
            if not atom.element.name in HBOND_ELEMENTS:
                continue
            
            if atom.residue.name in RESIDUE_NAMES:
                protein_hbond_atoms.append([atom, idx])
            else:
                ligand_hbond_atoms.append([atom, idx])
        
        hbond_idxs = self.get_simulation_hbond_indexes(ligand_hbond_atoms , 
                                                       protein_hbond_atoms,
                                                       {i[0] : i for i in atom_to_position_idx })
        
        self.hbond_idxs = hbond_idxs
        return hbond_idxs
    
    def get_simulation_hbond_indexes(self, ligand_hbond_atoms, protein_hbond_atoms, atom_idx, hbonding_radius = 30.0):
    
        ligand_donors = []
        ligand_acceptors = []
        protein_donors = []
        protein_acceptors = []
        
        #is donor/ acceptor 
        for ligand_atom in ligand_hbond_atoms:
            ligand_nei_atoms = ligand_atom[0].neighbors 
            ligand_nei_atom_elements = [i.element.name for i in ligand_nei_atoms]
            
            if 'H' in ligand_nei_atom_elements:
                for num, i in enumerate(ligand_nei_atom_elements):
                    if i == 'H':
                        ligand_donors.append([ligand_atom, atom_idx[ ligand_nei_atoms[num] ]])
                        

            ligand_acceptors.append(ligand_atom)
                
        for protein_atom in protein_hbond_atoms:
            #only set hbonds within the given radius for now
            distances = [get_distance(protein_atom[0].coord, i[0].coord) for i in ligand_hbond_atoms]
            gt_dist = [i for i in distances if i >= hbonding_radius]
            if not len(gt_dist) == len(distances):
                protein_nei_atoms = protein_atom[0].neighbors 
                protein_nei_atom_elements = [i.element.name for i in protein_nei_atoms]
                if 'H' in ligand_nei_atom_elements:
                    for num, i in enumerate(protein_nei_atom_elements):
                        if i == 'H':
                            protein_donors.append([protein_atom, atom_idx[ protein_nei_atoms[num] ]])
                
                protein_acceptors.append(protein_atom)
        
        
        hbond_idxs = []
        for donor in protein_donors:
            for acceptor in ligand_acceptors:
                #donor, hydrogen, acceptor
                idxs = [donor[0][1],donor[1][1], acceptor[1] ]
                hbond_idxs.append(idxs)
        
        for donor in ligand_donors:
            for acceptor in protein_acceptors:
                idxs = [donor[0][1],donor[1][1], acceptor[1] ]
                hbond_idxs.append(idxs)
        
        return hbond_idxs



def ChemEMVDWForce():
    #step(8.0 - r) *
    expression = "e0 * ((vg1 * exp(-((r - vdw1 - vdw2) / 0.5)^2)) + vg2 * exp(-((r - vdw1 - vdw2 - 3) / 2)^2) + vs * min(r - vdw1 - vdw2, 0)^2)"
    custom_force = CustomNonbondedForce(expression)
    
    custom_force.addGlobalParameter("e0", -1.0 * unit.kilojoules_per_mole)
    custom_force.addGlobalParameter("vg1", 0.008713077999999999 * unit.kilojoules_per_mole)
    custom_force.addGlobalParameter("vg2", 0.00168411182 * unit.kilojoules_per_mole)
    custom_force.addGlobalParameter("vs", -0.08191086900000001 * unit.kilojoules_per_mole)
    custom_force.addGlobalParameter("bnd", 8.0 * unit.angstrom)
    custom_force.addPerParticleParameter("vdw")
    custom_force.setCutoffDistance(0.8 * unit.nanometers)
    #custom_force.setCutoffDistance(9.0 * unit.angstroms)
    #custom_force.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    #custom_force.setUseLongRangeCorrection(False)
    return custom_force

def initilize_chemem_vdw_force(struct):
    custom_force = ChemEMVDWForce()
    peroidic_table = Chem.GetPeriodicTable()
    for atom in struct.atoms:
        vdw = peroidic_table.GetRvdw( atom.element_name )
        custom_force.addParticle([vdw * unit.angstrom])
    return custom_force
    
    
def initilize_chemem_vdw_force_old(struct,ligand_protein_atom_pairs ,cutoff = 8.0 * unit.angstrom):
    vdw_force = ChemEMVDWForceCompound()
    peroidic_table = Chem.GetPeriodicTable()
    
    print(len(ligand_protein_atom_pairs))
    
    for idx1, atom1, idx2, atom2 in ligand_protein_atom_pairs:
        ri = peroidic_table.GetRvdw( atom1.element_name )
        rj = peroidic_table.GetRvdw( atom2.element_name )
        vdw_force.addBond([idx1, idx2], [ri, rj])
    
    

    return vdw_force

def SoftBarrierPotential():
    expression = 'k * max(0, sqrt((x1-x0)^2 + (y1-y0)^2 + (z1-z0)^2) - r0)^2'
    custom_force = CustomCompoundBondForce(1, expression)
    
    custom_force.addGlobalParameter('k',  10000.0 * unit.kilocalories_per_mole )
    custom_force.addGlobalParameter('r0', 10.0 * unit.angstrom)
    custom_force.addGlobalParameter('x0', 133.538 * unit.angstrom) #this is the centroid!!
    custom_force.addGlobalParameter('y0', 132.957 * unit.angstrom)
    custom_force.addGlobalParameter('z0', 174.475 * unit.angstrom)
    
    return custom_force

def initialise_soft_barrier(ligand_idxs):
    soft_barrier = SoftBarrierPotential()
    for idx in ligand_idxs:
        soft_barrier.addBond([idx])
    return soft_barrier

def ChemEMVDWForceCompound():
    expression = "step(8.0 - distance(p1, p2)) * (e0 * ((vg1 * exp(-((distance(p1, p2) - ri - rj) / 0.5)^2)) + vg2 * exp(-((distance(p1, p2) - ri - rj - 3) / 2)^2) + vs * min(distance(p1, p2) - ri - rj, 0)^2))"
    custom_force = CustomCompoundBondForce(2, expression)  
    
    custom_force.addGlobalParameter("e0", -1.0 * unit.kilojoules_per_mole)
    custom_force.addGlobalParameter("vg1", 0.008713077999999999 * unit.kilojoules_per_mole)
    custom_force.addGlobalParameter("vg2", 0.00168411182 * unit.kilojoules_per_mole)
    custom_force.addGlobalParameter("vs", -0.08191086900000001 * unit.kilojoules_per_mole)
    custom_force.addPerBondParameter("ri") 
    custom_force.addPerBondParameter("rj") 
    return custom_force

def ChemEMHbondDistForce():
    
    expression = "(hb * e0) * max(-1, min(0, (distance(p1, p2) - ri - rj) / -0.7 * -1))"
    custom_force = CustomCompoundBondForce(2, expression)  
    custom_force.addGlobalParameter("hb", -0.07933760520000001 * unit.kilojoules_per_mole)
    custom_force.addPerBondParameter("e0") #energy unit
    custom_force.addPerBondParameter("ri")  # VdW radius for atom i
    custom_force.addPerBondParameter("rj") 

    return custom_force

def initialize_chemem_hbond_dist_force(donor_hydrogen_accpetor_indices, struct ):
    hbond_dist_force = ChemEMHbondDistForce()
    peroidic_table = Chem.GetPeriodicTable()
    for donor, hydrogen, acceptor in donor_hydrogen_accpetor_indices:
        donor_vdw = peroidic_table.GetRvdw( struct.atoms[donor].element_name )
        acceptor_vdw = peroidic_table.GetRvdw( struct.atoms[acceptor].element_name )
        hbond_dist_force.addBond([donor, acceptor], [-1.0 * unit.kilojoules_per_mole ,donor_vdw, acceptor_vdw])
        
    return hbond_dist_force


        
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
        self.temperature = 300
        self.temperature_step = 1
        self.pressure = 1*unit.atmosphere
        self.heating_interval = 10
        self.tug_k = 10000
        self.hbond_dist_k = 1000
        self.hbond_angle_k = 100
        self.get_complex_structure()
        self.densmap_when_atoms_are_inside()
        #self.set_forces()
    
    
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
            #chimera_slice = get_chimera_slice()#TODO!!
            #pass
    
    def reload_map(self):
        if self.densmap.display:
            self.densmap.display = False 
            self.densmap.display = True
    
    def get_complex_structure(self):
        
        complex_system = os.path.join(self.filepath, 'complex_system.xml' )
        if os.path.exists(complex_system):
            with open(complex_system,'r' ) as f:
                self.complex_system = XmlSerializer.deserialize(f.read())
        
        complex_structue_prmtop = os.path.join(self.filepath, 'complex_structure.prmtop')
        complex_structue_inpcrd = os.path.join(self.filepath, 'complex_structure.inpcrd')
        if os.path.exists(complex_structue_prmtop) and os.path.exists(complex_structue_inpcrd ):
            self.complex_structure = parmed.load_file(complex_structue_prmtop, xyz=complex_structue_inpcrd)
        
        
   
    
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
    
    def get_simualtion_hbond_pairs(self, atom_to_position_idx ):
        ligand_hbond_atoms = []
        protein_hbond_atoms = []
        
        for atom, idx in atom_to_position_idx:
            if not atom.element.name in HBOND_ELEMENTS:
                continue
            
            if atom.residue.name in RESIDUE_NAMES:
                protein_hbond_atoms.append([atom, idx])
            else:
                ligand_hbond_atoms.append([atom, idx])
        
        hbond_idxs = self.get_simulation_hbond_indexes(ligand_hbond_atoms , 
                                                       protein_hbond_atoms,
                                                       {i[0] : i for i in atom_to_position_idx })
        
        self.hbond_idxs = hbond_idxs
        return hbond_idxs
    
    def get_simulation_hbond_indexes(self, ligand_hbond_atoms, protein_hbond_atoms, atom_idx, hbonding_radius = 30.0):
    
        ligand_donors = []
        ligand_acceptors = []
        protein_donors = []
        protein_acceptors = []
        
        #is donor/ acceptor 
        for ligand_atom in ligand_hbond_atoms:
            ligand_nei_atoms = ligand_atom[0].neighbors 
            ligand_nei_atom_elements = [i.element.name for i in ligand_nei_atoms]
            
            if 'H' in ligand_nei_atom_elements:
                for num, i in enumerate(ligand_nei_atom_elements):
                    if i == 'H':
                        ligand_donors.append([ligand_atom, atom_idx[ ligand_nei_atoms[num] ]])
                        

            ligand_acceptors.append(ligand_atom)
                
        for protein_atom in protein_hbond_atoms:
            #only set hbonds within the given radius for now
            distances = [get_distance(protein_atom[0].coord, i[0].coord) for i in ligand_hbond_atoms]
            gt_dist = [i for i in distances if i >= hbonding_radius]
            if not len(gt_dist) == len(distances):
                protein_nei_atoms = protein_atom[0].neighbors 
                protein_nei_atom_elements = [i.element.name for i in protein_nei_atoms]
                if 'H' in ligand_nei_atom_elements:
                    for num, i in enumerate(protein_nei_atom_elements):
                        if i == 'H':
                            protein_donors.append([protein_atom, atom_idx[ protein_nei_atoms[num] ]])
                
                protein_acceptors.append(protein_atom)
        
        
        hbond_idxs = []
        for donor in protein_donors:
            for acceptor in ligand_acceptors:
                #donor, hydrogen, acceptor
                idxs = [donor[0][1],donor[1][1], acceptor[1] ]
                hbond_idxs.append(idxs)
        
        for donor in ligand_donors:
            for acceptor in protein_acceptors:
                idxs = [donor[0][1],donor[1][1], acceptor[1] ]
                hbond_idxs.append(idxs)
        
        return hbond_idxs
                
                
                
        
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
            copy_map = current_map.matrix()
            c_level = 0.0
            copy_map =  copy_map * (copy_map >= c_level)
            copy_map = copy_map / np.amax(copy_map)
   
            global_k = 75.0
            
            mp = map_potential_force_field(copy_map, origin, apix, copy_map.shape, global_k, blur=0)
            # add bonds to force field
            
            #just move ligand
            for idx in range(len(self.complex_structure.positions) ):

                mp.addBond([idx], [1.0])

            # add force here potential for map
            force_idx = self.complex_system.addForce(mp)
            self.complex_system.getForce(force_idx).setForceGroup(force_group)
            force_group += 1
        
        t2 = time.perf_counter() - t1 
        
        
        #-------------------set tug forces!!
        tug_force = initialize_tug_force([i for i in range(0, len(self.complex_structure.positions))],
                                                                             self.complex_structure)
        
        tug_idx = self.complex_system.addForce(tug_force)
        self.complex_system.getForce(tug_idx).setForceGroup(force_group)
        force_group += 1
        
        self.tug_force = tug_force 
        self.tug_idx = tug_idx
        
        #------------------set hbond force
        hbond_tug_dist_force, hbond_tug_angle_force = initialize_hbond_force(self.hbond_idxs, self.complex_structure)
        hbond_tug_dist_force_idx = self.complex_system.addForce(hbond_tug_dist_force)
        hbond_tug_angle_force_idx = self.complex_system.addForce(hbond_tug_angle_force)
        
        self.complex_system.getForce(hbond_tug_dist_force_idx).setForceGroup(force_group)
        force_group += 1
        
        self.complex_system.getForce(hbond_tug_angle_force_idx).setForceGroup(force_group)
        force_group += 1
        
        self.hbond_tug_dist_force = hbond_tug_dist_force
        self.hbond_tug_dist_force_idx = hbond_tug_dist_force_idx 
        
        self.hbond_tug_angle_force  = hbond_tug_angle_force 
        self.hbond_tug_angle_force_idx = hbond_tug_angle_force_idx 
        
        
        #------------------set up simulation
        integrator = LangevinIntegrator(self.temperature*unit.kelvin, 1.0/unit.picoseconds,
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
    
    def set_tempreture(self, new_temp):
        if new_temp > self.temperature:
            temps = [i+1 for i in range(int(self.temperature), int(new_temp), self.temperature_step)]
            
        elif new_temp  < self.temperature:
            temps = [i-1 for i in range(int(self.temperature), int(new_temp), -self.temperature_step)]
           
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
        
        
        if hbond_dist_k is None:
            hbond_dist_k = self.hbond_dist_k 
        
        if hbond_angle_k is None:
            hbond_angle_k = self.hbond_angle_k
        
        #need to get the correct angles form here!!
        dist = 2.9 * unit.angstrom
        angle = 180.0 * unit.degrees  
        
        context = self.simulation.context
        dist_indices, dist_params = self.hbond_tug_dist_force.getBondParameters(hbond_index)
        p1, p2, p3, angle_params = self.hbond_tug_angle_force.getAngleParameters(hbond_index)
        
        new_dist_params = [hbond_dist_k, dist]
        new_angle_params = [hbond_angle_k, angle]
        
        self.hbond_tug_dist_force.setBondParameters(hbond_index, dist_indices, new_dist_params)
        self.hbond_tug_angle_force.setAngleParameters(hbond_index, p1,p2,p3, new_angle_params)
        self.hbond_tug_dist_force.updateParametersInContext(context)
        self.hbond_tug_angle_force.updateParametersInContext(context)

    def get_tug_force(self):
        context = self.simulation.context

        # Get the state of the system with energy information, specifying the group of forces
        state = context.getState(getEnergy=True, groups={1 << self.tug_idx})
        energy = state.getPotentialEnergy()
        print('ENERGY OF TUG:', energy)
    
    
    
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


def update_tug_force(tug_force, dummy_atom_index, simulation, new_dummy_position, new_k = 5000):
    # Update the position of the dummy atom
    context = simulation.context
    state = context.getState(getPositions=True)
    positions = state.getPositions()
    positions[dummy_atom_index] = Vec3(*new_dummy_position) * unit.angstrom
    context.setPositions(positions)

    # Update the tugging force constant for all bonds involving the dummy atom
    for i in range(tug_force.getNumBonds()):
        atom_indices, params = tug_force.getBondParameters(i)
        # Only update if the bond involves the dummy atom
        if dummy_atom_index in atom_indices:
            params[0] = new_k  # Update the force constant
            tug_force.setBondParameters(i, atom_indices, params)

    tug_force.updateParametersInContext(context)

def TugForce():
    expr = "tug_k * ((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
    custom_force = CustomCompoundBondForce(1, expr)  # One particle per bond
    custom_force.addPerBondParameter("tug_k")
    custom_force.addPerBondParameter("x0") #positions to be tugged to
    custom_force.addPerBondParameter("y0") 
    custom_force.addPerBondParameter("z0")
    return custom_force

def initialize_tug_force(indexes,struct):
    tug_force = TugForce()
    for atom_index in indexes:
        atm = struct.atoms[atom_index]
        x0, y0, z0 = Quantity(
            value=[atm.xx, atm.xy, atm.xz], unit=unit.angstrom)
        tug_force.addBond([atom_index], [0.0, x0, y0, z0])
    return tug_force
    

#def HbondDistForce():
#    #should expand to add per bond parameter for r0
#    expression = 'k * (r - r0)^2'
#    distance_force = CustomCompoundBondForce(2, expression)
#    distance_force.addPerBondParameter("k")  # Force constant
#    distance_force.addPerBondParameter("r0") # Target distance
#    return distance_force

def HbondDistForce():
    # Correct the expression to explicitly calculate the distance
    expression = 'k * (distance(p1, p2) - r0)^2'  # p1, p2 are indices of the particles in the bond
    distance_force = CustomCompoundBondForce(2, expression)  # 2 indicates two particles are involved in each bond
    
    # Add parameters for force constant 'k' and target distance 'r0'
    distance_force.addPerBondParameter("k")   # Force constant
    distance_force.addPerBondParameter("r0")  # Target distance
    
    return distance_force

#def HbondAngleForce():
#    expression = 'k * (theta - theta0)^2'
#    angle_force = CustomCompoundBondForce(3, expression)
#    angle_force.addPerBondParameter("k") 
#    angle_force.addPerBondParameter("theta0")
#    return angle_force

def HbondAngleForce():
    # Define the energy expression
    expression = 'k * (theta - theta0)^2'

    # Create a CustomAngleForce object with the specified expression
    angle_force = CustomAngleForce(expression)
    
    # Add parameters for the force constant 'k' and the target angle 'theta0'
    angle_force.addPerAngleParameter("k")     # Force constant
    angle_force.addPerAngleParameter("theta0")  # Target angle
    return angle_force

def initialize_hbond_force(donor_hydrogen_accpetor_indices, struct ):
    hbond_dist_force = HbondDistForce()
    hbond_angle_force = HbondAngleForce()
    
    
    
    for donor, hydrogen, acceptor in donor_hydrogen_accpetor_indices:
        #GetThe distance and angle values here !!! 
        dist = 2.9 * unit.angstrom
        angle = 180.0 * unit.degrees  
        
        hbond_dist_force.addBond([donor, acceptor], [0.0, dist])
        hbond_angle_force.addAngle(donor, hydrogen, acceptor, [0.0, angle])
    return hbond_dist_force, hbond_angle_force
        
        
        
        
        


    