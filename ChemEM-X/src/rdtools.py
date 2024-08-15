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

import tempfile
import os
import numpy as np
from chimerax.core.commands import run 
import uuid


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

    
    
