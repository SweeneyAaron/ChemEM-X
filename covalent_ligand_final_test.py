#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:19:19 2024

@author: aaron.sweeney
"""
from rdkit import Chem 
from rdkit.Chem import AllChem

RD_PROTEIN_SMILES = {'CYS': ('N[C@H](C=O)CS', {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'SG': 5}),
 'MET': ('CSCC[C@H](N)C=O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'SD': 6, 'CE': 7}),
 'GLY': ('NCC=O', {'N': 0, 'CA': 1, 'C': 2, 'O': 3}),
 'ASP': ('N[C@H](C=O)CC(=O)O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'OD1': 6, 'OD2': 7}),
 'ALA': ('C[C@H](N)C=O', {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4}),
 'VAL': ('CC(C)[C@H](N)C=O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG1': 5, 'CG2': 6}),
 'PRO': ('O=C[C@@H]1CCCN1',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6}),
 'PHE': ('N[C@H](C=O)Cc1ccccc1',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD1': 6,
   'CD2': 7,
   'CE1': 8,
   'CE2': 9,
   'CZ': 10}),
 'ASN': ('NC(=O)C[C@H](N)C=O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'OD1': 6, 'ND2': 7}),
 'THR': ('C[C@@H](O)[C@H](N)C=O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'OG1': 5, 'CG2': 6}),
 'HIS': ('N[C@H](C=O)Cc1c[nH]c[nH+]1',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'ND1': 6,
   'CD2': 7,
   'CE1': 8,
   'NE2': 9}),
 'GLN': ('NC(=O)CC[C@H](N)C=O',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD': 6,
   'OE1': 7,
   'NE2': 8}),
 'ARG': ('NC(=[NH2+])NCCC[C@H](N)C=O',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD': 6,
   'NE': 7,
   'CZ': 8,
   'NH1': 9,
   'NH2': 10}),
 'TRP': ('N[C@H](C=O)Cc1c[nH]c2ccccc12',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD1': 6,
   'CD2': 7,
   'NE1': 8,
   'CE2': 9,
   'CE3': 10,
   'CZ2': 11,
   'CZ3': 12,
   'CH2': 13}),
 'ILE': ('CC[C@H](C)[C@H](N)C=O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG1': 5, 'CG2': 6, 'CD1': 7}),
 'SER': ('N[C@@H](C=O)CO',
  {'N': 0, 'CA': 1, 'CB': 2, 'OG': 3, 'C': 4, 'O': 5}),
 'LYS': ('N[C@H](C=O)CCCC[NH3+]',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD': 6,
   'CE': 7,
   'NZ': 8}),
 'LEU': ('CC(C)C[C@H](N)C=O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7}),
 'GLU': ('N[C@H](C=O)CCC(=O)O',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD': 6,
   'OE1': 7,
   'OE2': 8}),
 'TYR': ('N[C@H](C=O)Cc1ccc(O)cc1',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD1': 6,
   'CD2': 7,
   'CE1': 8,
   'CE2': 9,
   'CZ': 10,
   'OH': 11})}




#------WORKING RDKIT COVALENT LINKAGE----
def bond_order_value(bond_type):
    """
    Converts RDKit bond types to numerical bond order values.
    """
    bond_order_dict = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 1.5
    }
    return bond_order_dict.get(bond_type, 0)


def get_expected_valence(atom):
    """
    Calculates the expected valence of an atom based on its element and formal charge.
    
    Parameters:
    atom (rdkit.Chem.rdchem.Atom): The atom to calculate valence for.
    
    Returns:
    float: The expected valence.
    """
    periodic_table = Chem.GetPeriodicTable()
    atomic_num = atom.GetAtomicNum()
    formal_charge = atom.GetFormalCharge()
    
    # Get the default valence for the atom
    expected_valence = periodic_table.GetDefaultValence(atomic_num)
    
    # Adjust for formal charge
    
    expected_valence += formal_charge
    
    
    return expected_valence




def adjust_hydrogens(atom, s = False):
    """
    Adjusts explicit and implicit hydrogens on an atom to satisfy its valence and formal charge.
    
    Parameters:
    atom (rdkit.Chem.rdchem.Atom): The atom to adjust hydrogens for.
    """
    
    expected_valence = get_expected_valence(atom)
    
    # Calculate current bonds to heavy atoms (excluding hydrogens)
    current_bonds = 0
    for bond in atom.GetBonds():
        other_atom = bond.GetOtherAtom(atom)
        if other_atom.GetAtomicNum() > 1:
            current_bonds += bond_order_value(bond.GetBondType())
    
    # Calculate total explicit hydrogens
    explicit_h = atom.GetNumExplicitHs()
    implicit_h = atom.GetNumImplicitHs()
    # Total bonds including explicit hydrogens
    total_bonds = current_bonds + explicit_h + implicit_h
    
    
    
    # Calculate the difference between expected valence and current bonds
    h_diff = expected_valence - total_bonds
    
    if h_diff > 0:
        # Need to add hydrogens
        atom.SetNumExplicitHs(explicit_h + int(h_diff))
    elif h_diff < 0:
        # Need to remove hydrogens
        # Ensure we don't set negative hydrogens
        
        difference = int(explicit_h + h_diff )
        if difference < 0:
            atom.SetNumExplicitHs(0)
            atom.SetFormalCharge(abs(difference))
        else:
            atom.SetNumExplicitHs(max(int(explicit_h + h_diff), 0))
        
    
   
    # Recalculate total bonds after adjusting explicit hydrogens
    explicit_h = atom.GetNumExplicitHs()
    implicit_h = atom.GetNumImplicitHs()
    total_bonds = current_bonds + explicit_h + implicit_h
    implicit_h = expected_valence - total_bonds
    
    
    
    

def process_molecule(mol, bond_changes=None, atoms_to_remove=None, charges=None):
    """
    Process a molecule by applying bond changes and removing atoms,
    adjusting hydrogens as necessary based on valence and charge.

    Parameters:
    mol (rdkit.Chem.Mol): The molecule to process.
    bond_changes (list of tuples): Modifications to apply to bonds in the molecule.
        Each tuple is (idx1, idx2, new_bond_type).
    atoms_to_remove (list of ints): Atom indices to remove from the molecule.
    charges (dict): Dictionary mapping atom indices to formal charges.

    Returns:
    rdkit.Chem.Mol: The processed molecule.
    """
    # Create a mutable copy of the molecule
    mol = Chem.RWMol(mol)
    
    # Apply bond changes
    if bond_changes:
        for idx1, idx2, new_bond_type in bond_changes:
            bond = mol.GetBondBetweenAtoms(idx1, idx2)
            if bond is not None:
                old_bond_type = bond.GetBondType()
                if old_bond_type != new_bond_type:
                    bond.SetBondType(new_bond_type)
                    # Adjust hydrogens on both atoms
                    for idx in [idx1, idx2]:
                        atom = mol.GetAtomWithIdx(idx)
                        adjust_hydrogens(atom, s= True)
            else:
                # Bond does not exist; add it
                mol.AddBond(idx1, idx2, new_bond_type)
                # Adjust hydrogens on both atoms
                for idx in [idx1, idx2]:
                    atom = mol.GetAtomWithIdx(idx)
                    adjust_hydrogens(atom, s = True)
            
            
    
    # Remove atoms
    if atoms_to_remove:
        # Sort in reverse to prevent index shifting
        for idx in sorted(atoms_to_remove, reverse=True):
            if idx >= mol.GetNumAtoms():
                raise IndexError(f"Atom index {idx} is out of range for the molecule.")
            atom = mol.GetAtomWithIdx(idx)
            neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
            mol.RemoveAtom(idx)
            # After removal, adjust hydrogens on neighboring atoms
            for nbr_idx in neighbors:
                if nbr_idx >= mol.GetNumAtoms():
                    continue  # Neighbor might have been removed in earlier iterations
                nbr_atom = mol.GetAtomWithIdx(nbr_idx)
                adjust_hydrogens(nbr_atom)
    
    # Set formal charges
    if charges:
        for idx, charge in charges.items():
            if idx >= mol.GetNumAtoms():
                print('ChemEM Warning: Charged atom with idx {idx} has been removed.')
                continue
                #raise IndexError(f"Atom index {idx} is out of range for the molecule.")
            atom = mol.GetAtomWithIdx(idx)
            atom.SetFormalCharge(charge)
            adjust_hydrogens(atom)
    
    # Sanitize the molecule to ensure valence and bonding are correct
    
    
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException as e:
        raise ValueError(f"Sanitization failed: {e}")
    except Exception as e:
        raise ValueError(f"Validation failed: {e}")
    
    return mol.GetMol()


def prepare_atoms_for_binding(mol, atom_idx, bond_type):
    bond_value = bond_order_value(bond_type)
    
    if bond_value == 1.5:
        #if its aromatic
        bond_value = 1.0
    
    
    
    atom = mol.GetAtomWithIdx(atom_idx)
    explicit_h = atom.GetNumExplicitHs()
    implicit_h = atom.GetNumImplicitHs()
    
    
   
    
    for num in range(explicit_h):
        if bond_value == 0 or explicit_h == 0:
            break
        bond_value -= 1 
        explicit_h -= 1 

    
    atom.SetNumExplicitHs(explicit_h)
    
    if bond_value > 0:
        mol = Chem.AddHs(mol)
        mol = Chem.RWMol(mol)
        atom = mol.GetAtomWithIdx(atom_idx)
        nei_hs = [i for i in atom.GetNeighbors() if i.GetSymbol() == 'H']
        nei_hs = sorted(nei_hs, key = lambda x: x.GetIdx())

        for hydrogen in nei_hs:
            if bond_value == 0 or nei_hs == 0:
                break
            bond_value -= 1 
            mol.RemoveAtom(hydrogen.GetIdx())
        
            
    #need to adjust formal charge if bond_value > 1 
    if bond_value > 0:
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetFormalCharge(atom.GetFormalCharge() + bond_value)
    
    
    return mol 
    
    
    
    
    
        
    
    

def remove_hydrogens_attached_to_atom(mol, atom_idx, bond_type):
    """
    Remove hydrogens attached to a specific atom.

    Parameters:
    mol (rdkit.Chem.Mol): The molecule.
    atom_idx (int): Index of the atom from which to remove attached hydrogens.

    Returns:
    None
    """
    
    
    
    
    atom = mol.GetAtomWithIdx(atom_idx)
    
    
    hydrogens = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1]
    for h_idx in sorted(hydrogens, reverse=True):
        mol.RemoveAtom(h_idx)
    

def combine_processed_molecules(ligand, residue, ligand_atom_idx, residue_atom_idx, bond_type):
    """
    Combine two processed molecules by forming a bond between specified atoms.

    Parameters:
    ligand (rdkit.Chem.Mol): The processed ligand molecule.
    residue (rdkit.Chem.Mol): The processed residue molecule.
    ligand_atom_idx (int): Atom index in the ligand to form the bond.
    residue_atom_idx (int): Atom index in the residue to form the bond.
    bond_type (rdkit.Chem.rdchem.BondType): The bond type.

    Returns:
    rdkit.Chem.Mol: The combined molecule.
    """
    # Remove hydrogens attached to the reactive atoms
    ligand = Chem.RWMol(ligand)
    residue = Chem.RWMol(residue)
    #remove_hydrogens_attached_to_atom(ligand, ligand_atom_idx)
    ligand = prepare_atoms_for_binding(ligand, ligand_atom_idx, bond_type)
    ligand_hs = Chem.AddHs(ligand)
    p = '/Users/aaron.sweeney/Documents/ChemEM_chimera_v2/debug/ligand_prep_hs.sdf'
    write_to_sdf(ligand_hs, p)
    
    #remove_hydrogens_attached_to_atom(residue, residue_atom_idx)
    residue = prepare_atoms_for_binding(residue, residue_atom_idx, bond_type)
    residue_hs = Chem.AddHs(residue)
    p = '/Users/aaron.sweeney/Documents/ChemEM_chimera_v2/debug/residue_prep_hs.sdf'
    write_to_sdf(residue_hs, p)
    
    # Combine the molecules
    combined_mol = Chem.CombineMols(ligand, residue)
    combined_rwmol = Chem.RWMol(combined_mol)
    # Adjust residue atom indices
    residue_offset = ligand.GetNumAtoms()
    residue_atom_idx_combined = residue_atom_idx + residue_offset
    # Add bond
    

    combined_rwmol.AddBond(ligand_atom_idx, residue_atom_idx_combined, bond_type)
    combined_rwmol_hs = Chem.AddHs(combined_rwmol)

    Chem.SanitizeMol(combined_rwmol_hs)
    
    return combined_rwmol.GetMol()


def write_to_sdf( mol, file_name, conf_num=0, removeHs = False):
        '''
        write 3d mol to sdf file

        Parameters
        ----------
        mol : mol (rdkit.Chem.Mol)

        file_name : str
            Output file
        conf_num : int, optional
            conformation_number. The default is 0.
        removeHs : bool, optional
            whether to remove hydrogens when writting. The default is False.

        Returns
        -------rt
        None.

        '''
        
        writer = Chem.SDWriter(file_name)
        writer.write(mol, confId=conf_num)

def get_charges_from_molecule(mol):
    """
    Extract formal charges from an RDKit molecule.

    Parameters:
    mol (rdkit.Chem.Mol): The molecule.

    Returns:
    dict: Dictionary mapping atom indices to formal charges.
    """
    charges = {}
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge != 0:
            charges[atom.GetIdx()] = charge
    return charges

def get_residue_charges(residue, residue_name, atom_name_rd_idx):
    """
    Assign charges to a protein residue based on its name.

    Parameters:
    residue (rdkit.Chem.Mol): The residue molecule.
    residue_name (str): The three-letter code of the residue (e.g., 'LYS').

    Returns:
    dict: Dictionary mapping atom indices to formal charges.
    """
    charges = {}
    rd_idx_atom_name = {j:i for i,j in atom_name_rd_idx.items()}
    # Mapping from residue names to atom symbols and expected charges
    residue_charge_info = {
        'LYS': {'NZ': +1},   # Lysine side-chain nitrogen is positively charged
        'ARG': {'NH1': +1},   # Arginine guanidinium nitrogens are positively charged
        'ASP': {'OD1': -1},   # Aspartic acid side-chain oxygens are negatively charged
        'GLU': {'OE1': -1},   # Glutamic acid side-chain oxygens are negatively charged
        'HIS': {'ND1': +1},   # Histidine can be positively charged
        # Add more residues as needed
    }
    

    if residue_name in residue_charge_info:
        charge_info = residue_charge_info[residue_name]
        for atom in residue.GetAtoms():
            atom_idx = atom.GetIdx()
            symbol = rd_idx_atom_name[atom_idx]
            
            # Assign charge if atom symbol matches and charge is non-zero
            if symbol in charge_info and charge_info[symbol] != 0:
                charges[atom.GetIdx()] = charge_info[symbol]
    else:
        # For other residues, assume no charge
        pass

    return charges


def set_explict_hs(mol):
    mol = Chem.AddHs(mol)
    mol = Chem.RWMol(mol)
    hydrogens_to_remove = []
    for atom in mol.GetAtoms():
        
        nei_h = [i for i in atom.GetNeighbors() if i.GetSymbol() == 'H']
        
        atom.SetNumExplicitHs(len(nei_h))
        for hydrogen in nei_h:
            hydrogens_to_remove.append(hydrogen.GetIdx())
    
    for hydrogen_idx in sorted(hydrogens_to_remove, reverse = True):
        mol.RemoveAtom(hydrogen_idx)
    
        
    m2 = mol.GetMol() 
    
   
    
    return m2
    
    
    

def combine_molecules_and_react(ligand, 
                                residue, 
                                residue_name,
                                ligand_atom_idx, 
                                residue_atom_idx, 
                                bond_type,
                                residue_bond_changes=None,
                                ligand_bond_changes=None, 
                                residue_atoms_to_remove=None,
                                ligand_atoms_to_remove=None, 
                                ):
    """
    Process ligand and residue molecules, automatically detect charges, and combine them.

    Parameters:
    ligand_smiles (str): SMILES string of the ligand.
    residue_smiles (str): SMILES string of the residue.
    residue_name (str): Three-letter code of the residue (e.g., 'LYS').
    ligand_atom_idx (int): Atom index in the ligand to form the bond.
    residue_atom_idx (int): Atom index in the residue to form the bond.
    bond_type (rdkit.Chem.rdchem.BondType): The bond type.

    Other parameters are the same as before.

    Returns:
    str: SMILES string of the combined molecule.
    rdkit.Chem.Mol: The combined molecule.
    """
    # Convert SMILES to RDKit molecules
    
    
    
   # ligand = Chem.MolFromSmiles(Chem.MolToSmiles(ligand))
    #ligand = Chem.MolFromSmiles(ligand.value)
    ligand = Chem.MolFromSmiles(ligand)
    ligand = set_explict_hs(ligand)
    #residue = Chem.MolFromSmiles(Chem.MolToSmiles(residue))
    residue_smiles, name_idx = RD_PROTEIN_SMILES[residue_name]
    
    residue = Chem.MolFromSmiles(residue_smiles)
    residue = set_explict_hs(residue)
    # Get charges from molecules
    ligand_charges = get_charges_from_molecule(ligand)
    residue_charges = get_residue_charges(residue, residue_name, name_idx)

    # Process ligand
    ligand = process_molecule(
        ligand,
        bond_changes=ligand_bond_changes,
        atoms_to_remove=ligand_atoms_to_remove,
        charges=ligand_charges
    )
    
    if ligand_atoms_to_remove is not None:
        for num in ligand_atoms_to_remove:
            if num < ligand_atom_idx:
                ligand_atom_idx -= 1 

    # Process residue
    residue = process_molecule(
        residue,
        bond_changes=residue_bond_changes,
        atoms_to_remove=residue_atoms_to_remove,
        charges=residue_charges
    )
    if residue_atoms_to_remove is not None:
        for num in residue_atoms_to_remove:
            if num < residue_atom_idx:
                residue_atom_idx -= 1 
    

    # Combine the processed molecules
    combined_mol = combine_processed_molecules(ligand, residue, ligand_atom_idx, residue_atom_idx, bond_type)
    # Generate SMILES
    
    
    
    smiles = Chem.MolToSmiles(combined_mol)
    
    return smiles, combined_mol

#------WORKING RDKIT COVALENT LINKAGE END----
#ligand = 'B(c1cc(c(cc1C=O)c2c[nH]c3c2cc(cn3)c4cc(cnc4)C(=O)N(C)C)OC)(O)O'
#ligand_atoms_to_remove = [8]
ligand = 'Cc1ccc(cc1S(=O)(=O)NCCc2ccccn2)NC(=O)CN3C(=O)C(=CC=N3)Cl'
residue_name = 'CYS'


smiles, combined_mol = combine_molecules_and_react(ligand, None, residue_name, 27, 5, 
                                                   Chem.rdchem.BondType.SINGLE)





#heeerere
    