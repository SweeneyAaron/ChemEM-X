#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:38:59 2024

@author: aaron.sweeney
"""

import os 
from rdkit import Chem 
import json 

RESIDUE_NAMES = ['CYS','MET','GLY','ASP','ALA','VAL','PRO','PHE','ASN','THR',
                 'HIS','GLN','ARG','TRP','ILE','SER','LYS','LEU','GLU','TYR']

def mol_from_sdf(sdf_file, conf_num = 0):
    suppl = Chem.SDMolSupplier(sdf_file)
    mol = suppl[conf_num]
    mol = Chem.RemoveHs(mol, sanitize=False)
    return mol

p = '/Users/aaron.sweeney/Documents/ChemEM_anaconda_v3_new_dock/ChemEM/tools/amino_acids/'

RD_PROTEIN_SMILES = {}

for res_name in RESIDUE_NAMES:
    

    mol = os.path.join(p, f'{res_name}.sdf')
    mol = mol_from_sdf(mol)
    smi = Chem.MolToSmiles(mol)

    mol_idx = os.path.join(p, f'{res_name}.json')
    with open(mol_idx, 'r') as f:
        mol_idx = json.load(f)
    
    RD_PROTEIN_SMILES[res_name] = (smi, mol_idx)
    
    
    
    
    