#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:33:23 2024

@author: aaron.sweeney
"""

from rdkit import Chem
import parmed
from chimerax.ChemEM.rdtools import get_mol_from_output, hydrogen_mapping
from chimerax.ChemEM.simulation import get_complex_structure
import copy 


def get_bonded_atoms(atom, exclude_atoms):
    bonded_atoms = []
    bonded_hydrogens = []
    for bond in atom.bonds:
        other_atom = bond.atom1 if bond.atom1 != atom else bond.atom2
        if other_atom.name not in exclude_atoms:
            if other_atom.element_name == 'H':
                bonded_hydrogens.append(other_atom)
            else:
                bonded_atoms.append(other_atom)
    return bonded_atoms, bonded_hydrogens

def remove_atoms(structure, atoms_to_remove):
    for atom in atoms_to_remove:
        
        bonds = [bond for bond in structure.bonds if atom in bond]
        for bond in bonds:
            structure.bonds.remove(bond)
        
        angles = [angle for angle in structure.angles if atom in angle]
        for angle in angles:
            structure.angles.remove(angle) 
        
        dihedrals = [dihedral for dihedral in structure.dihedrals if atom in dihedral]
        for dihedral in dihedrals:
            structure.dihedrals.remove(dihedral)
        
        impropers = [improper for improper in structure.impropers if atom in improper]
        for improper in impropers:
            structure.impropers.remove(improper)
        
        structure.atoms.remove(atom)
        structure.remake_parm() 


def map_atoms(entity, old_atoms, new_atoms, index_to_atom_dic, hydrogen_map_idx, new_complex_structure):
    mapped_atoms = []
    # Check if entity is a dihedral (has atom4), else it's an angle
    if hasattr(entity, 'atom4'):
        atom_list = [entity.atom1, entity.atom2, entity.atom3, entity.atom4]
    else:
        atom_list = [entity.atom1, entity.atom2, entity.atom3]
    for atom in atom_list:
        print(f"Attempting to map atom: {atom.name} (Index: {atom.idx})")
        if atom.idx == old_atoms['prot_atom'].idx:
            mapped_atoms.append(new_atoms['prot_atom'])
        elif atom.idx == old_atoms['ligand_atom'].idx:
            mapped_atoms.append(new_atoms['ligand_atom'])
        else:
            # Check if the atom index is in index_to_atom_dic (heavy atoms)
            if atom.idx in index_to_atom_dic:
                atom_name = index_to_atom_dic[atom.idx]
                # Search in protein residue atoms
                mapped_atom = next((a for a in new_atoms['prot_atom'].residue.atoms if a.name == atom_name), None)
                if mapped_atom is None:
                    print(f"Atom {atom_name} not found in protein residue atoms.")
                    return None
                mapped_atoms.append(mapped_atom)
            # Check if the atom index is in hydrogen_map_idx (hydrogen atoms)
            elif atom.idx in hydrogen_map_idx:
                atom_name = hydrogen_map_idx[atom.idx]
                # Search in protein residue atoms
                mapped_atom = next((a for a in new_atoms['prot_atom'].residue.atoms if a.name == atom_name), None)
                if mapped_atom is None:
                    print(f"Hydrogen atom {atom_name} not found in protein residue atoms.")
                    return None
                mapped_atoms.append(mapped_atom)
            else:
                # Assume it's a ligand atom, search in ligand residue atoms
                ligand_residue_atoms = new_complex_structure.residues[-1].atoms
                mapped_atom = next((a for a in ligand_residue_atoms if a.name == atom.name), None)
                if mapped_atom is None:
                    print(f"Atom {atom.name} not found in ligand residue atoms.")
                    return None
                mapped_atoms.append(mapped_atom)
    return mapped_atoms


class MappingObject():
    
    '''
    Just a centralised class to keep all this stuff in the same place
    '''
    
    def __init__(self, 
                 complex_ligand_structure,
                 atom_match,
                 protein_atoms_conversion,
                 index_to_atom_dic,
                 hydrogen_map, 
                 hydrogen_map_idx,
                 hydrogen_map_to_atom_name,
                 bind_protein_atom,
                 bind_protein_atom_idx,
                 current_simualtion_atom,
                 hydrogen_idxs):
    
        self.complex_ligand_structure = complex_ligand_structure
        self.atom_match = atom_match
        self.protein_atoms_conversion = protein_atoms_conversion
        self.heavy_atom_idxs = list(protein_atoms_conversion.values())
        self.index_to_atom_dic = index_to_atom_dic
        self.hydrogen_map = hydrogen_map
        self.hydrogen_map_idx = hydrogen_map_idx
        self.hydrogen_map_to_atom_name = hydrogen_map_to_atom_name
        self.bind_protein_atom = bind_protein_atom
        self.bind_protein_atom_idx = bind_protein_atom_idx
        self.current_simualtion_atom = current_simualtion_atom
        self.hydrogen_idxs = hydrogen_idxs #complex_ligand_structure protein hydrogens
        
def make_mapping_object(output, #job.params.output
                        covelent_ligand,
                        current_simulation_complex_structure,
                        atoms_to_position_index_as_dic,
                        ):
    
    #combine mol form chemem simulation
    map_mol = get_mol_from_output(output)
    
    #rdkit RWmol that was put in to the chemem simulation
    input_mol = covelent_ligand.parameters['combined_rwmol']
    
    #Parmed Amber strcutre object of ligand bound residue
    #TAKE
    complex_ligand_structure = get_complex_structure(output)
    
    #this index of the SDF used to model the protein resiude with no Hydrogens.
    protein_atom_names_to_idx = covelent_ligand.get_parameter('residue_idx')
    
    #number of ligand heavy atoms not including the protein residue 
    num_ligand_atoms = covelent_ligand.parameters['ligand_rw_mol'].GetNumHeavyAtoms()

    #this is the inital mol (input_mol) protein heavy atom mapping 
    protein_atom_names_to_idx = {i : j + num_ligand_atoms for i,j in protein_atom_names_to_idx.items()}
    
    #map from map_mol to input_mol 
    #atom n (list index) in input_mol is atom n_i (output value) in map_mol
    atom_match = map_mol.GetSubstructMatch(input_mol)

    #converts the protein_atom_names_to_index to the map_mol values!
    #TAKE!!
    protein_atoms_conversion = {i: atom_match[j] for i ,j in  protein_atom_names_to_idx.items()}
    #flip this around so its rdmol idx to names
    #TAKE!!
    index_to_atom_dic = {i:j for j,i in protein_atoms_conversion.items() }
    
    #ChimeraX atom object of defined protein binding atom!
    #TAKE
    bind_protein_atom = covelent_ligand.get_parameter('protein_atom')
    
    #the hydrogen map is the hydrogen name to the rdmol index 
    #the hydrogen_map_to_atom_name is the hydrogen_name to the heavy atom name where it is bonded 
    #TAKE!!
    hydrogen_map, hydrogen_map_to_atom_name = hydrogen_mapping(Chem.AddHs(map_mol) ,
                                    protein_atoms_conversion, 
                                    bind_protein_atom.residue.name)
    
    hydrogen_map_idx = {i:j for j,i in hydrogen_map.items()}
    #TAKE
    bind_protein_atom_idx = atoms_to_position_index_as_dic[bind_protein_atom]
    #TAKE
    current_simualtion_atom = current_simulation_complex_structure.atoms[bind_protein_atom_idx]
    
    
    #
    heavy_atom_idxs = list(protein_atoms_conversion.values())
    
    hydrogen_idxs = []
    for idx in heavy_atom_idxs:
        atom = current_simulation_complex_structure.atoms[idx]
        print('atom')
        print(atom)
        print('')
        bonds = atom.bonds 
        
        for bond in bonds:
            
            if bond.atom1.element_name == 'H':
                hydrogen_idxs.append(bond.atom1.idx)
                print('Should be hydrogen')
                print(idx)
                print(bond.atom1)
                print(bond.atom1.idx)
                
            if bond.atom2.element_name == 'H':
                print('Should be hydrogen')
                print(idx)
                print(bond.atom2)
                print(bond.atom1.idx)
                hydrogen_idxs.append(bond.atom2.idx)
    
    print(hydrogen_idxs)
    print(heavy_atom_idxs)
    
    
    mapping_object = MappingObject(complex_ligand_structure,
                                   atom_match,
                                   protein_atoms_conversion,
                                   index_to_atom_dic,
                                   hydrogen_map, 
                                   hydrogen_map_idx,
                                   hydrogen_map_to_atom_name,
                                   bind_protein_atom,
                                   bind_protein_atom_idx,
                                   current_simualtion_atom,
                                   hydrogen_idxs)#complex_ligand_structure protein hydrogens
    #TODO!!
    #REMOVE ATOMS AND BONDS!!!
    return mapping_object


def get_protein_atom_bond_partners(mapping_object):  
    
    '''
    Gets heavey atoms specified to be removed.
    Gets hydrogen atoms bonded to the protein atom of interest
    '''
    
    bonds = mapping_object.current_simualtion_atom.bonds
    
    current_simulation_bonded_atoms = []
    current_simulation_bonded_atoms_hydrogen = []
    for i in bonds:
        if i.atom1.idx != mapping_object.current_simualtion_atom.idx:
            #This assumes that the protein atom that has been remove will be removed from the dic 
            #should only leave removed atoms and hydrogens 
            if i.atom1.name not in mapping_object.protein_atoms_conversion:
                if i.atom1.element_name == 'H':
                    current_simulation_bonded_atoms_hydrogen.append(i.atom1)
                else:
                    current_simulation_bonded_atoms.append(i.atom1)
        
        #if not current atom
        if i.atom2.idx != mapping_object.current_simualtion_atom.idx:
            #if the atom is not a heavy atom in the protein residue
            if i.atom2.name not in mapping_object.protein_atoms_conversion:
                
                if i.atom1.element_name == 'H':
                    
                    current_simulation_bonded_atoms_hydrogen.append(i.atom2)
                else:
                    current_simulation_bonded_atoms.append(i.atom2)
    
    return current_simulation_bonded_atoms, current_simulation_bonded_atoms_hydrogen

def get_ligand_atom_bond_partners(mapping_object):
    '''
    Get Protein toms bonded to the ligand bond atom 
    n.b. This does not discriminate between heavy atoms and H's
    
    '''
    current_ligand_bonded_atoms = []
    cov_struct_idx =  mapping_object.protein_atoms_conversion[mapping_object.bind_protein_atom.name]
    #Get Atoms Bonded to the ligand bond atom
    bonds = mapping_object.complex_ligand_structure.atoms[cov_struct_idx].bonds 
    
    protein_keys = list(mapping_object.protein_atoms_conversion.values())
    for i in bonds:
        #if not ligand bond atom
        if i.atom1.idx != cov_struct_idx:
            #if its not a protein atom
            if i.atom1.idx not in protein_keys:
                current_ligand_bonded_atoms.append(i.atom1)
        if i.atom2.idx != cov_struct_idx:
            if i.atom2.idx not in protein_keys:
                current_ligand_bonded_atoms.append(i.atom2)

    return current_ligand_bonded_atoms

def copy_complex_structure(complex_structure):
    
    #get deepcopy of complex_structure to edit
    complex_structure_copy = copy.deepcopy(complex_structure)
    complex_structure_copy._ncopies = complex_structure._ncopies
    return complex_structure_copy 

def remove_atom_from_complex_structure(atom, complex_structure_copy):
    '''
    Removes an atom and assocaited bonds from a complex_structure

    Parameters
    ----------
    atom : TYPE
        DESCRIPTION.
    complex_structure_copy : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    bonds = [bond for bond in complex_structure_copy.bonds if atom in bond]
    for bond in bonds:
        complex_structure_copy.bonds.remove(bond)
    
    angles = [angle for angle in complex_structure_copy.angles if atom in angle]
    for angle in angles:
        complex_structure_copy.angles.remove(angle) 
    
    dihedrals = [dihedral for dihedral in complex_structure_copy.dihedrals if atom in dihedral]
    for dihedral in dihedrals:
        complex_structure_copy.dihedrals.remove(dihedral)
    
    impropers = [improper for improper in complex_structure_copy.impropers if atom in improper]
    for improper in impropers:
        complex_structure_copy.impropers.remove(improper)
    
    #add the rest!!!!
    
    complex_structure_copy.atoms.remove(atom)
    complex_structure_copy.remake_parm() 
    
    return complex_structure_copy

def add_ligand_binding_params_to_complex_structure(mapping_object,
                                                   old_prot_atom,
                                                   prot_atom,
                                                   old_ligand_atom,
                                                   ligand_atom,
                                                   new_complex_structure):
    
    index_to_atom_dic = {i:j for j,i in mapping_object.protein_atoms_conversion.items() }
    #find bonds between the protein and ligand
    bonds_to_add = [bond for bond in mapping_object.complex_ligand_structure.bonds if old_prot_atom in bond and old_ligand_atom in bond]

    for bond in bonds_to_add:
        bond_type = bond.type 
        new_bond = parmed.topologyobjects.Bond(prot_atom, ligand_atom, bond_type)
        new_complex_structure.bonds.append(new_bond)
    
    #what happens if its a hydrogen angle!!TODO! important
    angles_to_add = [angle for angle in mapping_object.complex_ligand_structure.angles if old_prot_atom in angle and old_ligand_atom in angle ]

    new_angles = []
    for angle in angles_to_add:
        a1 ,a2, a3, angle_type = angle.atom1.idx, angle.atom2.idx, angle.atom3.idx, angle.type
        # Create a list of the atoms
        atoms = [a1, a2, a3]
        found_flags = [False, False, False]
        
        # Replace old_prot_atom with prot_atom
        for i in range(len(atoms)):
            if atoms[i] == old_prot_atom.idx:
                atoms[i] = prot_atom
                found_flags[i] = True
        
        # Replace old_ligand_atom with ligand_atom
        for i in range(len(atoms)):
            if atoms[i] == old_ligand_atom.idx:
                atoms[i] = ligand_atom
                found_flags[i] = True
        
        
        find_atom = atoms[found_flags.index(False)]
        
        if find_atom in index_to_atom_dic:
            print('1...')
            name = index_to_atom_dic[find_atom]
            
            for atom in prot_atom.residue.atoms:
                if atom.name == name:
                    atoms[found_flags.index(False)] = atom
                    break
        else:
            #is a ligand atom!!
            print('2...')
            
            name = old_ligand_atom.residue.atoms[find_atom].name 
            
            print(name)
            for atom in new_complex_structure.residues[-1].atoms:
                print(atom.name)
                if atom.name == name:
                    print('Here')
                    atoms[found_flags.index(False)] = atom
            
            
            
        a1 ,a2, a3 = atoms
        print('a1', a1, 'a2', a2, 'a3', a3, 'angle_type', angle_type)
        print(found_flags)
        print('')
        new_angle = parmed.topologyobjects.Angle(a1, a2, a3, angle_type)
        new_angles.append(new_angle)
        new_complex_structure.angles.append(new_angle)

    dihedrals_to_add = [torsion for torsion in mapping_object.complex_ligand_structure.dihedrals if old_prot_atom in torsion and old_ligand_atom in torsion ]
    #create hydrogen_mapping!!! 
    new_torsions = []
    for dihedral in dihedrals_to_add:
        a1 ,a2, a3, a4, dihedral_type = dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx, dihedral.type
        ao1, ao2,ao3, ao4= dihedral.atom1, dihedral.atom2, dihedral.atom3, dihedral.atom4
        
        atoms = [a1, a2, a3, a4]
        atom_objects = [ao1,ao2,ao3, ao4]
        found_flags = [False, False, False, False]
        # Replace old_prot_atom with prot_atom
        for i in range(len(atoms)):
            if atoms[i] == old_prot_atom.idx:
                atoms[i] = prot_atom
                found_flags[i] = True
        
        # Replace old_ligand_atom with ligand_atom
        for i in range(len(atoms)):
            if atoms[i] == old_ligand_atom.idx:
                atoms[i] = ligand_atom
                found_flags[i] = True
        
        for index, flag in enumerate(found_flags):
            
            
            if flag == False:
                find_atom = atoms[index]
                find_atom_object = atom_objects[index]
                if find_atom_object.element_name != 'H':
                
                    if find_atom in index_to_atom_dic:
                        name = index_to_atom_dic[find_atom]
                        
                        for atom in prot_atom.residue.atoms:
                            if atom.name == name:
                                atoms[index] = atom
                                break
                    else:
                        name = old_ligand_atom.residue.atoms[find_atom].name 
                        
                        for atom in new_complex_structure.residues[-1].atoms:
                            if atom.name == name:
                                atoms[index] = atom
                else:
                    #find heavy atom binding partner
                    #map the heavy atoms in the ligand from the protein to their hydrogens,
                    #then map these to the new_simulation_residue!! 
                    #same for ligand!!
     
                    hydrogen_bond_atom = find_atom_object.bonds[0]
                    
                    if hydrogen_bond_atom.atom1 != find_atom:
                        hydrogen_bond_atom = hydrogen_bond_atom.atom1
                    else:
                        hydrogen_bond_atom = hydrogen_bond_atom.atom2
                    
                    if hydrogen_bond_atom.idx in  index_to_atom_dic:
                        
                        hydrogen_name = mapping_object.hydrogen_map_idx[find_atom]
              
                        for atom in prot_atom.residue.atoms:
                            if atom.name == hydrogen_name:
                                atoms[index] = atom
                                
                    
                    else:
                        name = find_atom_object.name
                    
                        for atom in new_complex_structure.residues[-1].atoms:
                            if atom.name == name:
                                atoms[index] = atom
                        
                        
                    
        
        a1 ,a2, a3,a4 = atoms
        new_torsions.append((a1,a2,a3,a4,dihedral_type))
        new_torsion = parmed.topologyobjects.Dihedral(a1, a2, a3, a4, type=dihedral_type)
        new_complex_structure.dihedrals.append(new_torsion)
        
        
        impropers_to_add = [improper for improper in mapping_object.complex_ligand_structure.impropers if old_prot_atom in improper and old_ligand_atom in improper ]
        #add the imporopers ect...
        new_complex_structure.remake_parm()
    
    return new_complex_structure
