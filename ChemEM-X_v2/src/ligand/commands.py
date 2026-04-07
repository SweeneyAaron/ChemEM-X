#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:43:40 2025

@author: aaron.sweeney
"""

from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.ligand.tools import Protonate

class ProtonateSmiles(Command):
    @classmethod 
    def js_code(cls, smiles):
        js_code = f'addSmilesToProtonationList("{smiles}");'
        return js_code 
    
    @classmethod
    def run(cls,chemem, query):
        #need to delete the list and images at this stage TODO!
        print('---------HERERERERER')
        chemem.run_js_code("removeAllProtonatedSmiles();")
        p  = Protonate.from_query(query)
        p.protonate()
        
        for state in p.protonation_states:
            js_code = cls.js_code(state)
            chemem.run_js_code(js_code)
        
        chemem.current_protonation_states = p
        
class CleanProtonationImage(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_protonation_states is not None:
            if chemem.current_protonation_states.current_image_file is not None:
                chemem.current_protonation_states.remove_temporary_file()

class ShowLigandSmilesImage(Command):
    
    @classmethod 
    def js_code(cls, image_path):
        js_code = f'displayChemicalStructure("{image_path}");'
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        
        if chemem.current_protonation_states is not None:
            if chemem.current_protonation_states.current_image_file is not None:
                chemem.current_protonation_states.remove_temporary_file()
            try:
                
                image_idx = chemem.current_protonation_states.protonation_states.index(query)
                chemem.current_protonation_states.save_image_temporarily(image_idx)
                file_path = chemem.current_protonation_states.current_image_file.name
                if file_path is not None:
                    
                    js_code = cls.js_code(file_path)
                    chemem.run_js_code(js_code)
                    
                    #chemem.current_protonation_states.remove_temporary_file()
            except ValueError:
                js_code = f'alert("Smiles image not found {query}");'
                chemem.run_js_code(js_code)
                
           