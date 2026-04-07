#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:31:33 2025

@author: aaron.sweeney
"""

import os 
from chimerax.core.commands import run 
from chimerax.atomic import Element
import uuid
from rdkit import Chem 
import numpy as np 

OutputError = "OutputError"
MissingDataError = "MissingDataError"



class ChemEMSetUp:
    
    def __init__(self,
                 backend,
                 protocols,
                 options,
                 data):
        
        self.backend = backend #path variable to backend
        self.protocols = protocols #optional flags to run: List(str) 
        self.options = options #protocol flags to run: List((str, value)) ; value has the type of the value
        self.data = data #conf data List((str str)); conf key : conf value
        self.output = None
        self.conf_file_path = None 
        self.status = 0
        self._err = []
        
        self.check = self.make_conf_file()
        self.make_chemem_commands()

    def _get_data_value(self, value):
        return [i[1] for i in self.data if i[0] == value]
        
    def make_conf_file(self):
        output = self._get_data_value("output")
        
        if output:
            if os.path.exists(output[0]):
                self.output = output[0]
            else:
                self._err.append(OutputError)
                return 0
        
        else:
            self._err.append(OutputError)
            return 0
        
        file = ''
        for key, value in self.data:
            file += f'{key} = {value}\n'
        
        conf_file_path = os.path.join(self.output, 'ChemEMChimera.conf')
        
        with open(conf_file_path, 'w') as f:
            f.write(file)
        
        self.conf_file_path = conf_file_path
        return 1
            
        
        
    def make_chemem_commands(self):
        com = f'{self.backend} {self.conf_file_path} '
        
        for protocol in self.protocols:
            com += f'--{protocol} '
        
        for option, value in self.options:
            com += f'--{option} {value} ' #important trailing space
        
        
        self.run_command = com
        
        
    
    
    @classmethod 
    def from_parameters_object(cls, session, 
                               parameters_object,  #essentailly the conf file
                               backend,
                               protocols,
                               options = [],
                               selected_atoms = False):
        
        data = []
        output = parameters_object.get_value("output")
        
        
        if output is not None:
            data.append(("output", output))
        
            #-----get model input data
            model = parameters_object.get_parameter("current_model") 
            inputs_dir = os.path.join(output, 'inputs')
            mkdir(inputs_dir)
            if model is not None:
                #TODO! include Selected Atoms option!!!
                
                model_file = f"{model.name}.pdb"
                model_id = f"#{'.'.join([str(i) for i in model.id])}"
                model_path = os.path.join(inputs_dir, model_file)
                
                if selected_atoms:
                    com = f'save {model_path} format pdb models {model_id} selectedOnly true'
                else:
                    com = f'save {model_path} format pdb models {model_id}'
                
                run(session, com)
                data.append(("protein", model_path))
                
            
            #-----get ligand input data 
            ligands =  parameters_object.get_parameter("Ligands")
            #ligands are a list parameter so need to iterate through to extract data
            if ligands is not None:
                for lig in ligands:
                    data.append(("ligand", lig.value))
            
            #-----get maps input data
            density_map = parameters_object.get_parameter("current_map")
            resolution = parameters_object.get_value("resolution")
            if density_map is not None and resolution is not None:
                #no need to resave maps just get value from object path
                #TODO! may need to add a check to enure map has been saved otherwise save it
                #FYI protein models are resaved as they are subject to change
            
                data.append(("densmap", density_map.path))
                data.append(("resolution", resolution))
                
            #-----get centroids
            binding_sites = parameters_object.get_parameter("binding_sites")
            
            if binding_sites is not None:
                for site in binding_sites:
                    x,y,z = site.centroid
                    data.append(("centroid", f'({round(x,3)}, {round(y,3)}, {round(z,3)})'))
            
            
        return cls(backend, protocols, options, data)
            
                
def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


class DockSolution:
    def __init__(self,
                 full_path,
                 display_name,
                 score):
        self.full_path = full_path
        self.display_name = display_name 
        self.score = score 
        self.__id = str(uuid.uuid4)

class LoadedSolutions():
    def __init__(self):
        self.directories = []
        self.results = []
        #holds a directory of full path to mdl for hide/show soultions in the GUI
        self._gui = {}
                      

def get_dock_results(path):
    if os.path.exists(path):
        dirs = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]
        all_solutions = LoadedSolutions()
        
        for d in dirs:
            
            solutions = []
            results_path = os.path.join(path, d)
            results_file = os.path.join(results_path, 'results.txt')
            
            if os.path.exists(results_file):
                
                results = read_results_file(results_file)
                if results:
                    
                    for key, value in results.items():
                        solutions.append(DockSolution(os.path.join(results_path, f'{key}.sdf'),
                                                      key,
                                                      value))
                        
            if solutions:
                all_solutions.directories.append(d)
                all_solutions.results.append(solutions)
        return all_solutions

def get_mmgbsa_scores(path):
    if os.path.exists(path):
        dirs = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]
        all_solutions = LoadedSolutions()
        results_file = os.path.join(path, 'mmgbsa_scores.txt')
        results = read_mmgbsa_file(results_file)
        
        if results:
            
            for d in dirs:
                solutions = [] 
                results_path =  os.path.join(path, d)
                
                for key, (score, dir_name) in results.items():
                    if d == dir_name:
                        solutions.append(DockSolution(os.path.join(path , dir_name, f'{key}.sdf'),
                                                      key, 
                                                      score))
                        
                if solutions:
                    all_solutions.directories.append(d)
                    all_solutions.results.append(solutions)
        return all_solutions
                    
                
            
def read_mmgbsa_file(path):
      
    with open(path, 'r') as f:
        results_file = f.read().splitlines() 
        results = {}
        for r in results_file:
            if r.startswith(('Ligand')):
                #This is ugly as F
                #A line looks like this: "Ligand 0, pose 0:{'EEL': -18.4031, 'VDW': 23.644, 'EGB': 30.801, 'ECAV': -226.301, 'deltaG': -190.258}"
                
                key = r.split(':')[0].replace(',','').replace(' ','_')
                dir_name = key.partition('pose')[0][0:-1]
                value = r.partition('{')[2].partition('}')[0] 
                value = value.split(':')[-1]
                try:
                    value = round(float(value),3)
                    results[key] = (value, dir_name)
                except ValueError:
                    continue 
    return results
                
                
def read_results_file(file):
    
    with open(file, 'r') as f:
        results_file = f.read().splitlines() 
        results = {}
        for r in results_file:
            r = r.split(':')
            try:
                results[r[0].replace(' ', '')] = round(float(r[1]), 3) 
            except ValueError:
                #skip if there is some issue converting value to float
                continue 
    return results
                

def add_residue_to_model(model, ligand):
    #TODO! autochain id
    
    chain_id = next_unused_chain_id(model)
    
    added_ligs = [i.name for i in model.residues if i.name.startswith('LIG')]
    res_name = f'LIG{len(added_ligs)}'
    
    
    res_num =  int(max(r.number for r in model.residues)) + 1
    res = model.new_residue(res_name, chain_id, res_num)
    
    created_atoms = []
    counts = {} #use this to keep syble style nameing e.g. c1, c2 dependending on element apperance in file
    for atom in ligand.atoms:
        element_symbol = atom.element.name
        elem = Element.get_element(element_symbol)
        if element_symbol not in counts:
            counts[element_symbol] = 1
        element_name = f'{element_symbol}{counts[element_symbol]}'
        counts[element_symbol] += 1
        x,y,z = atom.coord 
       
        a = model.new_atom(element_name, elem)
        a.coord = (float(x), float(y), float(z))
        res.add_atom(a)
        created_atoms.append(a)
    
    for bond in ligand.bonds:
        atoms = bond.atoms
        idx = []
        for a in atoms:
            idx.append(ligand.atoms.index(a))
        
        b = model.new_bond(created_atoms[idx[0]], created_atoms[idx[1]])
        if hasattr(b, "order"):
            b.order = bond.order
    
    return res
        
        
        
def next_unused_chain_id(model):
    used = {c.chain_id for c in model.chains}
    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    # try single-char IDs
    for cid in chars:
        if cid not in used:
            return cid
    # fall back to two-char IDs
    for a in chars:
        for b in chars:
            cid = a + b
            if cid not in used:
                return cid
    raise RuntimeError("No available chain IDs.")    
            

def get_atom_match_object(res, lig_file):
    mol = mol_from_sdf(lig_file)
    if mol is not None:
        pos = mol.GetConformer().GetPositions() 
        cx_atoms = list(res.atoms)  # preserve order
        cx_coords = [(float(a.coord[0]), float(a.coord[1]), float(a.coord[2])) for a in cx_atoms]

        candidate = {}      # cx_atom -> (rd_idx, dist)
        rd_best = {}         # rd_idx -> (cx_atom, dist)
        
        for a, coord in zip(cx_atoms, cx_coords):
            best_idx = None
            best_d = float("inf")
            for rd_idx in range(len(pos)):
                d = _dist(coord, pos[rd_idx])
                if d < best_d:
                    best_d, best_idx = d, rd_idx
            if best_idx is not None and best_d <= 1e-4:
                candidate[a] = (best_idx, best_d)
                # resolve collisions by keeping the closest
                prev = rd_best.get(best_idx)
                if prev is None or best_d < prev[1]:
                    rd_best[best_idx] = (a, best_d)
        
        rd_to_cx = {}
        cx_to_rd = {}
        
    
        for rd_idx, (a, d) in rd_best.items():
            # ensure that 'a' still prefers this rd_idx (avoid edge cases)
            a_pref_idx, a_pref_d = candidate.get(a, (None, None))
            if a_pref_idx == rd_idx:
                rd_to_cx[rd_idx] = a
                cx_to_rd[a] = rd_idx
                
    
        return AtomMatcher(rd_to_cx, cx_to_rd, rdmol=mol, file=lig_file)



class AtomMatcher:
    
    def __init__(self,
                 rd_to_cx,
                 cx_to_rd,
                 rdmol=None,
                 file = None):
        
        self.rd_to_cx  = rd_to_cx       # RDKit idx -> ChimeraX Atom
        self.cx_to_rd  = cx_to_rd       # ChimeraX Atom -> RDKit idx
        self.rdmol = rdmol
        self.file = file
    

    def get_by_rd_idx(self, rd_idx):
        """Return ChimeraX Atom for a given RDKit atom index (or None)."""
        return self.rd_to_cx.get(rd_idx)

    def get_by_atom(self, cx_atom):
        """Return RDKit atom index for a given ChimeraX Atom (or None)."""
        return self.cx_to_rd.get(cx_atom)

    def __getitem__(self, key) :
        """Convenience: matcher[rd_idx] -> Atom, matcher[cx_atom] -> rd_idx."""
        if isinstance(key, int):
            return self.get_by_rd_idx(key)
        return self.get_by_atom(key)

    def update_positions(self):
        # TODO : update positions if moved before writting
        pass


def _dist(a,b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
        
    
        

def mol_from_sdf(sdf_file, conf_num = 0):
    suppl = Chem.SDMolSupplier(sdf_file)
    mol = suppl[conf_num]
    
    return mol

    