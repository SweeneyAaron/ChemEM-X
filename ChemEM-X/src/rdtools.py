#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:51:00 2024

@author: aaron.sweeney
"""

from dimorphite_dl import DimorphiteDL
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Geometry import Point3D
import tempfile
import os
import numpy as np
from chimerax.core.commands import run 
import uuid


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
            
        
    
    
    
    
    
    
    
