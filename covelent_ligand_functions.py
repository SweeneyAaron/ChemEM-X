from rdkit import Chem
from rdkit.Chem import rdchem, AllChem, Draw

def draw_molecule_with_atom_indices_old(mol):
    for atom in mol.GetAtoms():
        atom.SetProp('molAtomMapNumber', str(atom.GetIdx()))
    img = Draw.MolToImage(mol)
    return img

def draw_molecule_with_atom_indices(mol, size=(500, 500), dpi=300):
    # Set atom indices as labels
    for atom in mol.GetAtoms():
        atom.SetProp('molAtomMapNumber', str(atom.GetIdx()))
    
    # Draw the molecule with specified size and DPI
    img = Draw.MolToImage(mol, size=size, kekulize=True)
    
    # Convert the image to a higher resolution
    img = img.resize((size[0] * dpi // 72, size[1] * dpi // 72), resample=0)
    return img


def modify_bond_order(rwmol, bonds_to_modify):
    for atom_idx1, atom_idx2, new_bond_type in bonds_to_modify:
        bond = rwmol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
        if bond is not None:
            rwmol.RemoveBond(atom_idx1, atom_idx2)
            rwmol.AddBond(atom_idx1, atom_idx2, new_bond_type)
        else:
            print(f"No bond found between atom {atom_idx1} and {atom_idx2}")

def remove_atoms(rwmol, atom_indices):
    # Sort atom indices in reverse order to avoid index shifting issues
    for atom_idx in sorted(atom_indices, reverse=True):
        rwmol.RemoveAtom(atom_idx)

def remove_atoms(rwmol, atom_indices):
    # Sort atom indices in reverse order to avoid index shifting issues
    for atom_idx in sorted(atom_indices, reverse=True):
        rwmol.RemoveAtom(atom_idx)

def combine_molecules_and_react(ligand, 
                                residue, 
                                ligand_atom_idx, 
                                residue_atom_idx, 
                                bond_type,
                                protein_bonds_to_change,
                                ligand_bonds_to_change,
                                protein_atoms_to_remove,
                                ligand_atoms_to_remove):
    
    ligand.GetAtomWithIdx(ligand_atom_idx).SetNumExplicitHs(0)
    residue.GetAtomWithIdx(residue_atom_idx).SetNumExplicitHs(0)

    combined_mol = Chem.CombineMols(ligand, residue)
    combined_rwmol = Chem.RWMol(combined_mol)
    residue_new_zero_idx = ligand.GetNumAtoms()
    residue_bond_point = residue_atom_idx + residue_new_zero_idx
    combined_rwmol.AddBond(ligand_atom_idx, residue_bond_point , bond_type)
    

    for idx1, idx2, bond in protein_bonds_to_change:
        idx1 += residue_new_zero_idx 
        idx2 += residue_new_zero_idx 
        modify_bond_order(combined_rwmol, [(idx1,idx2, bond)])
    
   
    modify_bond_order(combined_rwmol, ligand_bonds_to_change)
    
    protein_atoms_to_remove = [i + residue_new_zero_idx for i in protein_atoms_to_remove]
    remove_atoms(combined_rwmol, protein_atoms_to_remove)
    remove_atoms(combined_rwmol, ligand_atoms_to_remove)
    
    Chem.SanitizeMol(combined_rwmol)
    # Convert to SMILES and return
    product_smiles = Chem.MolToSmiles(combined_rwmol)
    
    return product_smiles, combined_rwmol
    
    


# Example usage
ligand_smiles = "Cc1ccc(cc1S(=O)(=O)NCCc2ccccn2)NC(=O)CN3C(=O)C(=CC=N3)Cl"
cysteine_smiles = "N[C@@H](CS)C(=O)O"

ligand = Chem.RWMol(Chem.MolFromSmiles(ligand_smiles))
residue = Chem.RWMol(Chem.MolFromSmiles(cysteine_smiles))

img_protein = draw_molecule_with_atom_indices(residue)
img_ligand = draw_molecule_with_atom_indices(ligand)

img_protein.show()
img_ligand.show()


protein_bonds_to_change = []
ligand_bonds_to_change =  []
protein_atoms_to_remove = []
ligand_atoms_to_remove = []
ligand_atom_idx = 27
residue_atom_idx = 3
bond_type = rdchem.BondType.SINGLE

product_smiles, combined_rwmol = combine_molecules_and_react(ligand, 
                                residue, 
                                ligand_atom_idx, 
                                residue_atom_idx, 
                                bond_type,
                                protein_bonds_to_change,
                                ligand_bonds_to_change,
                                protein_atoms_to_remove,
                                ligand_atoms_to_remove)

img_protein = draw_molecule_with_atom_indices(combined_rwmol)
img_protein.show()



