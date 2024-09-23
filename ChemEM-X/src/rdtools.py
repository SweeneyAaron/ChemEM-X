#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:51:00 2024

@author: aaron.sweeney
"""

from dimorphite_dl import DimorphiteDL
from rdkit import Chem
from rdkit.Chem import Draw, rdchem, AllChem
from rdkit.Geometry import Point3D
from scipy.spatial.transform import Rotation as R
import tempfile
import os
import numpy as np
from chimerax.core.commands import run 
import uuid
from rdkit import Chem
from rdkit.Chem import rdFMCS



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




def smiles_is_valid(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return True 
    return False

def mol_from_sdf(sdf_file, conf_num = 0):
    suppl = Chem.SDMolSupplier(sdf_file)
    mol = suppl[conf_num]
    mol = Chem.RemoveHs(mol, sanitize=False)
    return mol

def get_mol_from_output(output):
    file = [i for i in os.listdir(output) if i.endswith('.sdf')][0]
    mol = mol_from_sdf(os.path.join(output, file))
    return mol
    


def open_pdb_model(session, file):
    com = f'open {file}'
    model = run(session, com)
    session.models.remove(model)
    return model[0]


class Protonate:
    
    def __init__(self, smiles, min_pH = 6.4, max_pH = 8.4, pka_precision=1.0):
        
        self.smiles = smiles
        self.min_pH = min_pH
        self.max_pH = max_pH 
        self.pka_precision =  pka_precision
        self.protonation_states = []
        self.images = []
        self.current_image_file = None
        
    def _protonate(self):
        
        dimorphite_dl = DimorphiteDL(

            min_ph=self.min_pH,
            max_ph=self.max_pH,
            max_variants=128,
            label_states=False,
            pka_precision=self.pka_precision
            )
        self.protonation_states = dimorphite_dl.protonate(self.smiles)
    
    def generate_2d_images(self):
        for smiles in self.protonation_states:
            img = self.draw_molecule_from_smiles(smiles)
            if img:
                self.images.append(img)
    
    def draw_molecule_from_smiles(self, smiles):
        """Generate a 2D depiction of a molecule from a SMILES string."""
        # Create a molecule object from the SMILES string
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            print("Invalid SMILES string.")
            return
        
        # Compute 2D coordinates for the molecule
        Chem.rdDepictor.Compute2DCoords(molecule)
        
        # Generate the image of the molecule
        img = Draw.MolToImage(molecule)
        return img
    
    def save_image_temporarily(self, idx):
        # Create a temporary file with the suffix '.png' to ensure the file format is correct
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    
        # Save the image to the temporary file
        self.images[idx].save(temp_file.name)
    
        # Close the file (important to release the file handle)
        temp_file.close()
    
        # Return the path to the temporary file
        self.current_image_file = temp_file
    
    def remove_temporary_file(self):
        try:
            os.remove(self.current_image_file.name)
            print(f"File {self.current_image_file.name} has been successfully removed.")
        except FileNotFoundError:
            print(f"No file found at {self.current_image_file.name}. Nothing to remove.")
        except PermissionError:
            print(f"Permission denied: cannot delete {self.current_image_file.name}.")
        except Exception as e:
            print(f"An error occurred deleting image {self.current_image_file.name}: {e}.")
    
    def protonate(self):
        self._protonate() 
        self.generate_2d_images()
    
    @classmethod 
    def from_query(cls, query):
        
        return cls( query.smiles, query.min_pH, query.max_pH, query.pka_prec)
         

class ChemEMResult:
    def __init__(self, 
                 session,
                 preprocessing_path,
                 map_files,
                 fitting_path,
                 fitting_file_pairs,
                 fitting_results_path,
                 postprocessing_path,
                 postprocessing_file_pairs,
                 postprocssing_results_path):
        self.session = session
        self.preprocessing_path = preprocessing_path
        self.map_files = map_files 
        self.fitting_path = fitting_path
        self.fitting_file_pairs = fitting_file_pairs 
        self.fitting_results_path = fitting_results_path 
        self.postprocessing_path = postprocessing_path
        self.postprocessing_file_pairs = postprocessing_file_pairs 
        self.postprocssing_results_path = postprocssing_results_path
        self.messages = []
        self.map_objects = []
        self.fitting_results = []
        self.postprocessing_results = []
        self.load_files() 
        
        
    def clear(self):
        for chemem_object in self.map_objects:
            self.session.models.remove([chemem_object.density_map ])
            del chemem_object.density_map 
        
        for chemem_object in self.fitting_results:
            if chemem_object.pdb_object in self.session.models:
                self.session.models.remove([chemem_object.pdb_object])
            del chemem_object .pdb_object
        
        for chemem_object in self.postprocessing_results:
            if chemem_object.pdb_object in self.session.models:
                self.session.models.remove([chemem_object.pdb_object])
            del chemem_object.pdb_object
            
        
        
    def load_files(self):
        self.load_map_files()
        self.load_fitting_files()
        self.load_post_processed_soltuions()
        
    def load_map_files(self):
        
        for file in self.map_files:
            file_path = os.path.join(self.preprocessing_path, file)
            com = f'open {file_path}'
            map_object = run(self.session, com)
            solution_map = SolutionMap(map_object[0])
            js_code = f'addLoadMapItem("{solution_map.name}", "{solution_map.id}");'
            self.messages.append(js_code)
            self.map_objects.append(solution_map)
            
            
    def load_fitting_files(self):
        
        if os.path.exists( self.fitting_results_path ):
            with open(self.fitting_results_path, 'r') as f:
                fitting_scores = f.read().splitlines() 
                fitting_scores = [i.split(':') for i in fitting_scores]
                fitting_scores = { i[0].replace(' ', '') : i[1].replace(' ','') for i in fitting_scores }
        
        else:
            fitting_scores = {} 
        
        self.fitting_scores = fitting_scores
        
        for result in self.fitting_file_pairs:
            score_key = result[0].replace('Ligand_', '')
            score_key = score_key.replace('.pdb', '')
            
            try:
                score = fitting_scores[score_key]
                score = round(float(score), 3)
            
            except (KeyError, ValueError) as e:
                score = 'Not Found'
            
            pdb_path =  os.path.join(self.fitting_path, 'PDB',result[0])
            ligand_paths = [os.path.join(self.fitting_path, i) for i in result[1:]]
            
            
            solution = Solution(self.session,
                                        pdb_path,
                                        ligand_paths,
                                        score
                                        )
            
            self.fitting_results.append(solution)
            
            js_code = f'addLoadFittingSolution("{solution.string()}", "{ solution.id}", "fittingMapsList");'
            self.messages.append(js_code)
    
    
    def load_post_processed_soltuions(self):
        
        post_processed_results = self.get_postprocessed_results()
        
        for result in self.postprocessing_file_pairs:
            key = result[0].replace('.pdb', '')
            key = key.split('_')
            key = str([key[1],key[3]])
            if key in post_processed_results:
                score = round(post_processed_results[key], 3)
            else:
                score = 'Not Found'
            pdb_path = os.path.join(self.postprocessing_path, result[0])
            ligand_paths = [os.path.join(self.postprocessing_path,i) for i in result[1:]]
            solution = Solution(self.session,
                                        pdb_path,
                                        ligand_paths,
                                        score
                                        )
            self.postprocessing_results.append(solution)
            js_code = f'addLoadFittingSolution("{solution.string()}", "{ solution.id}", "postprocMapsList");'
            self.messages.append(js_code)
            
            
        
        
    def get_postprocessed_results(self):
        if os.path.exists(self.postprocssing_results_path):
            with open(self.postprocssing_results_path ,'r') as f:
                results = f.read().splitlines() 
            
            current_ligand = None
            all_scores = {}
            for line in results:
                if line.startswith('Ligand_'):
                    ligand_id = line.replace('Ligand_', '')
                    ligand_id = ligand_id.replace(' ', '')
                    current_ligand = ligand_id 
                else:
                    line = line.split(':')
                    cycle_id = line[0].replace('Cycle', '')
                    cycle_id = cycle_id.replace('\t', '')
                    cycle_id = cycle_id.replace(' ','')
                    score = float(line[1])
                    
                    all_scores[str([current_ligand,cycle_id])] = score
            return all_scores
        else:
            return {}
    
    def get_loaded_result_with_id(self, solution_id):
        for solution in self.fitting_results:
            if solution.id == solution_id:
                return solution 
        for solution in self.postprocessing_results:
            if solution.id == solution_id:
                return solution
        
        return None
        
    
class SolutionMap:
    def __init__(self, density_map):
        self.density_map = density_map 
        self.name = density_map.name 
        self._id = density_map.id 
        self.id = '.'.join([str(i) for i in self._id])
    
    def show(self):
        self.density_map.display = True 
    def hide(self):
        self.density_map.display = False
    

class Solution:
    def __init__(self, session, pdb_path, ligand_paths, score=None):
        #TODO!! hydrogens will come into play here i think during simulations
        self.session = session
        self.rdkit_mol_objects = [mol_from_sdf(i) for i in ligand_paths]
        self.pdb_object = open_pdb_model(self.session, pdb_path)
        self.score = score
        self.all_matched_atoms = []
        
        self.id = str(uuid.uuid4())
        self.match_atoms_to_ligand()
        
        
    def match_atoms_to_ligand(self):
        
        for ligand in self.rdkit_mol_objects:
            ligand_positions = ligand.GetConformer().GetPositions() 
            ligand_atoms = [i.GetSymbol() for i in ligand.GetAtoms()]
   
            for residue in self.pdb_object.residues:
                #!should include Hs??
                atoms_no_hydrogen = [atom for atom in residue.atoms if atom.element.name != 'H']
                if len(atoms_no_hydrogen) == len(ligand_atoms):
                    if sorted(ligand_atoms) == sorted([atom.element.name for atom in atoms_no_hydrogen]):
                        match_atoms = self.ligand_is_match(ligand_positions, ligand_atoms, atoms_no_hydrogen)
                        if match_atoms:
                            self.all_matched_atoms.append(match_atoms)
                        else:
                            #alert!!!
                            self.all_matched_atoms.append(None)
            

    def ligand_is_match(self, ligand_positions, ligand_atoms, residue_atoms):
        
        matching_atoms = {}
        
        # Iterate through each ligand atom and its position
        for i, (lig_pos, lig_element) in enumerate(zip(ligand_positions, ligand_atoms)):
            for j, chimera_atom in enumerate(residue_atoms):
                # Check if the positions and elements match
                if np.array_equal(lig_pos, np.array(chimera_atom.coord)) and lig_element == chimera_atom.element.name:
                    # Store the match: ligand index -> residue atom
                    matching_atoms[i] = chimera_atom
                    break  # Assuming only one match is possible, break out of the loop
        
        return matching_atoms
    
    def hide_solution(self):
        self.session.models.remove([self.pdb_object])
    
    def show_solution(self):
        self.session.models.add([self.pdb_object])
        
        com = f'color #{self.pdb_object.id[0]} byhetero'
        run(self.session, com)
        com = f'style #{self.pdb_object.id[0]} stick'
        run(self.session, com)
        com = f'show  #{self.pdb_object.id[0]} cartoon'
        run(self.session, com)
        #this could get annoying
        com = 'hide sidechain target ab'
        run(self.session, com)
    
    def check_atom_positions(self):
        for idx, ligand in enumerate(self.rdkit_mol_objects):
            mol_pos = ligand.GetConformer().GetPositions()
            atom_match = self.all_matched_atoms[idx]
            if atom_match is None:
                continue
            
            new_atom_pos = []
            
            for index, atom in atom_match.items():
                if not np.array_equal(mol_pos[index], np.array(atom.coord)):
                    #set new atom position!!!
                    new_atom_pos.append( [index,np.array(atom.coord) ]  )

                if new_atom_pos:
                    self.set_new_mol_positions(new_atom_pos, ligand)
    
    def set_new_mol_positions(self, new_pos, ligand):
        conf = ligand.GetConformer()

        for index, coord in new_pos:
            
            x, y, z = coord
            conf.SetAtomPosition(index, Point3D(x, y, z))
            
    def string(self):
        return f'{self.pdb_object.name} : ChemEM-score {self.score}'


def RW_mol_from_smiles(smiles):
    return Chem.RWMol(Chem.MolFromSmiles(smiles))

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
    
    
    #remove_hydrogens_attached_to_atom(residue, residue_atom_idx)
    residue = prepare_atoms_for_binding(residue, residue_atom_idx, bond_type)
   
    
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
    
    
    
   
    ligand = Chem.MolFromSmiles(ligand)
    ligand = set_explict_hs(ligand)
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
    
    return smiles, combined_mol, ligand_atom_idx, residue_atom_idx

#------WORKING RDKIT COVALENT LINKAGE END----
    
    

def save_image_temporarily(image):
    # Create a temporary file with the suffix '.png' to ensure the file format is correct
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')

    # Save the image to the temporary file
    image.save(temp_file.name)

    # Close the file (important to release the file handle)
    temp_file.close()

    # Return the path to the temporary file
    return temp_file

def remove_temporary_file(temp_file):
    try:
        os.remove(temp_file)
        print(f"File {temp_file} has been successfully removed.")
    except FileNotFoundError:
        print(f"No file found at {temp_file}. Nothing to remove.")
    except PermissionError:
        print(f"Permission denied: cannot delete {temp_file}.")
    except Exception as e:
        print(f"An error occurred deleting image {temp_file}: {e}.")

    
def convert_parmed_to_rdkit(parm):
    
    mol = Chem.RWMol()
    # Add atoms
    atom_map = {}
    for i, atom in enumerate(parm.atoms):
        rd_atom = Chem.Atom(atom.element_name)
        atom_map[atom] = mol.AddAtom(rd_atom)


def hydrogen_mapping(mol, residue_atoms_converted, residue_name ):
    
    hydrogen_data = RD_PROTEIN_HYDROGENS[residue_name]
    hydrogen_name_to_idx = {}
    hydrogen_name_to_heavy_atom_name = {}
    for atom_name, index in  residue_atoms_converted.items():
        
        atom_hydrogens = [i for i in hydrogen_data if hydrogen_data[i] == atom_name]
        
        mol_atom = mol.GetAtomWithIdx(index)
        mol_atom_hydrogens = [i for i in mol_atom.GetNeighbors() if i.GetSymbol() == 'H']
       
        
        for hydrogen_name, hydrogen_atom in zip(atom_hydrogens, mol_atom_hydrogens):
            hydrogen_name_to_idx[hydrogen_name] = hydrogen_atom.GetIdx()
            hydrogen_name_to_heavy_atom_name[hydrogen_name] = atom_name
    return hydrogen_name_to_idx, hydrogen_name_to_heavy_atom_name
        

def translate_and_rotate_molecule(coords, index1, target1, index2, target2):
    """
    Translate and rotate a molecule so that two atoms are moved to specific target points.
    Improved to handle edge cases and use quaternions for rotations.
    """
    # Translate the molecule so that the first atom is at target1
    translation_vector = target1 - coords[index1]
    translated_coords = coords + translation_vector

    # Calculate the vector of the second atom after translation
    moved_atom_vector = translated_coords[index2] - target1
    target_vector = target2 - target1

    # Normalize vectors
    moved_atom_norm = np.linalg.norm(moved_atom_vector)
    target_norm = np.linalg.norm(target_vector)

    if moved_atom_norm == 0 or target_norm == 0:
        return translated_coords  # No rotation needed if one vector is zero

    moved_atom_vector_normalized = moved_atom_vector / moved_atom_norm
    target_vector_normalized = target_vector / target_norm

    # Calculate the axis and angle for rotation
    axis = np.cross(moved_atom_vector_normalized, target_vector_normalized)
    axis_norm = np.linalg.norm(axis)

    if axis_norm == 0:
        return translated_coords  # Vectors are parallel, no rotation needed

    angle = np.arccos(np.clip(np.dot(moved_atom_vector_normalized, target_vector_normalized), -1.0, 1.0))

    # Use quaternion for rotation to avoid potential gimbal lock issues
    rotation = R.from_quat(R.from_rotvec(axis / axis_norm * angle).as_quat())
    rotated_coords = rotation.apply(translated_coords - target1) + target1

    return rotated_coords


RD_PROTEIN_HYDROGENS = {'CYS': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HG': 'SG'},      
                        
 'MET': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HG2': 'CG',
  'HG3': 'CG',
  'HE1': 'CE',
  'HE2': 'CE',
  'HE3': 'CE'},
 
 'GLY': {'H': 'N', 'H2': 'N', 'H3': 'N', 'HA2': 'CA', 'HA3': 'CA'},
 'ASP': {'H': 'N', 'H2': 'N', 'H3': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB'},
 'ALA': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB1': 'CB',
  'HB2': 'CB',
  'HB3': 'CB'},
 
 'VAL': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB': 'CB',
  'HG11': 'CG1',
  'HG12': 'CG1',
  'HG13': 'CG1',
  'HG21': 'CG2',
  'HG22': 'CG2',
  'HG23': 'CG2'},
 
 'PRO': {'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HG2': 'CG',
  'HG3': 'CG',
  'HD2': 'CD',
  'HD3': 'CD'},
 
 'PHE': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HD1': 'CD1',
  'HD2': 'CD2',
  'HE1': 'CE1',
  'HE2': 'CE2',
  'HZ': 'CZ'},
 
 'ASN': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HD21': 'ND2',
  'HD22': 'ND2'},
 
 'THR': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB': 'CB',
  'HG1': 'OG1',
  'HG21': 'CG2',
  'HG22': 'CG2',
  'HG23': 'CG2'},
 
 'HIS': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HD2': 'CD2',
  'HD1': 'ND1',
  'HE1': 'CE1',
  'HE2': 'NE2'},
 
 'GLN': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HG2': 'CG',
  'HG3': 'CG',
  'HE21': 'NE2',
  'HE22': 'NE2'},
 
 'ARG': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HG2': 'CG',
  'HG3': 'CG',
  'HD2': 'CD',
  'HD3': 'CD',
  'HE': 'NE',
  'HH11': 'NH1',
  'HH12': 'NH2',
  'HH21': 'NH2',
  'HH22': 'NH2'},
 
 'TRP': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HD1': 'CD1',
  'HE1': 'NE1',
  'HE3': 'CE3',
  'HZ2': 'CZ2',
  'HZ3': 'CZ3',
  'HH2': 'CH2'},
 
 'ILE': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB': 'CB',
  'HG12': 'CG1',
  'HG13': 'CG1',
  'HG21': 'CG2',
  'HG22': 'CG2',
  'HG23': 'CG2',
  'HD11': 'CD1',
  'HD12': 'CD1',
  'HD13': 'CD1'},
 
 'SER': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HG': 'OG'},
 
 'LYS': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HG2': 'CG',
  'HG3': 'CG',
  'HD2': 'CD',
  'HD3': 'CD',
  'HE2': 'CE',
  'HE3': 'CE',
  'HZ1': 'NZ',
  'HZ2': 'NZ',
  'HZ3': 'NZ'},
 
 'LEU': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HG': 'CG',
  'HD11': 'CD1',
  'HD12': 'CD1',
  'HD13': 'CD1',
  'HD21': 'CD2',
  'HD22': 'CD2',
  'HD23': 'CD2'},
 
 'GLU': {'HA': 'CA',
  'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HB2': 'CB',
  'HB3': 'CB',
  'HG2': 'CG',
  'HG3': 'CG'},
 
 'TYR': {'H': 'N',
  'H2': 'N',
  'H3': 'N',
  'HA': 'CA',
  'HB2': 'CB',
  'HB3': 'CB',
  'HD1': 'CD1',
  'HD2': 'CD2',
  'HE1': 'CE1',
  'HE2': 'CE2',
  'HH': 'OH'}}
