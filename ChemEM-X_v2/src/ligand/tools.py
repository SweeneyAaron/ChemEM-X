#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:47:18 2025

@author: aaron.sweeney
"""

import os
import tempfile
from rdkit import Chem 
from rdkit import Chem
from rdkit.Chem import Draw, rdchem, AllChem

try:
    from dimorphite_dl import protonate_smiles as _DIMORPHITE_PROTONATE_SMILES
except ImportError:
    _DIMORPHITE_PROTONATE_SMILES = None

try:
    from dimorphite_dl import DimorphiteDL
except ImportError:
    DimorphiteDL = None

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
        if _DIMORPHITE_PROTONATE_SMILES is not None:
            self.protonation_states = _DIMORPHITE_PROTONATE_SMILES(
                self.smiles,
                ph_min=self.min_pH,
                ph_max=self.max_pH,
                max_variants=128,
                label_states=False,
                precision=self.pka_precision,
            )
            return

        if DimorphiteDL is not None:
            dimorphite_dl = DimorphiteDL(
                min_ph=self.min_pH,
                max_ph=self.max_pH,
                max_variants=128,
                label_states=False,
                pka_precision=self.pka_precision,
            )
            self.protonation_states = dimorphite_dl.protonate(self.smiles)
            return

        raise ImportError(
            "dimorphite_dl API not found: expected protonate_smiles or DimorphiteDL"
        )
                                         
        
        
    
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

    def draw_molecule_from_mol(self, mol):
        """Generate a 2D depiction directly from an RDKit mol.

        Tracked-ligand rdmols carry explicit hydrogens and a 3D conformer, which
        would project into an ugly depiction. Strip the Hs, drop conformers, and
        compute fresh 2D coordinates before rendering.
        """
        if mol is None:
            return None

        work = Chem.Mol(mol)
        try:
            work = Chem.RemoveHs(work)
        except Exception:
            work = Chem.Mol(mol)

        work.RemoveAllConformers()
        Chem.rdDepictor.Compute2DCoords(work)
        return Draw.MolToImage(work)
    
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
         
