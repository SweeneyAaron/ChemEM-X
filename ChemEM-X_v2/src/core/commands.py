#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:28:39 2025

@author: aaron.sweeney
"""
import os
import json
from chimerax.ChemEM.core.tools import condense_path, validate_ligand_file, _build_atom_matcher_from_model_and_sdf, register_tracked_ligand
from chimerax.ChemEM.core.parameters import PathParameter
from chimerax import open_command
from chimerax.atomic.structure import AtomicStructure
from chimerax.map import Volume
from rdkit import Chem
from chimerax.core.commands import run as cx_run


def _normalise_ligand_id(ligand_id):
    return str(ligand_id).strip()


def _pop_tracked_ligand(chemem, ligand_id):
    tracked = getattr(chemem, "_tracked_ligands", None)
    if not tracked:
        return None

    if ligand_id in tracked:
        return tracked.pop(ligand_id)

    ligand_id_key = _normalise_ligand_id(ligand_id)
    if ligand_id_key in tracked:
        return tracked.pop(ligand_id_key)

    for tracked_id in list(tracked.keys()):
        if _normalise_ligand_id(tracked_id) == ligand_id_key:
            return tracked.pop(tracked_id)

    return None


class Command:
    @classmethod
    def js_code(cls, *args):
        """Generate JavaScript code specific to the command."""
        pass
    
    @classmethod 
    def update_chemem(cls, *args):
        """Update ChemEM object state data."""
        pass 
    
    @classmethod
    def execute(cls, chemem, command, query):
        """Execute the command with provided arguments."""
        
        if command  == cls.__name__:
            #TODO!
            if True:
            #try:
                cls.run(chemem, query)
            #except Exception as e:
            #    alert_message = f'ChemEM Error, unable to run command: {cls.__name__} - {e}'
            #    js_code = f'alert("{alert_message}");'
            #    chemem.run_js_code(js_code)
                
    
    @classmethod
    def run(cls, chemem, query):
        """Utility method to run JavaScript code on a given HTML view."""
        pass


class SetDir(Command):
    
    @classmethod 
    def js_code(cls,  element_id, file):
        condensed_path = condense_path(file)
        js_code = f'setFilePath( "{element_id}", "{condensed_path}" );' 
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        file_dialog =  open_command.dialog.OpenFolderDialog(chemem.session.ui.main_window, chemem.session)
    
        file = file_dialog.get_path()
        if file is not None:
            if os.path.isdir(file):
                js_code = cls.js_code(query, file)
                cls.update_chemem(chemem, file, query)
                
            else:
                js_code= f"alert(File does not exist: {file});"
            
            chemem.run_js_code(js_code)
    
    @classmethod
    def update_chemem(cls,chemem, file, query):
        chemem.parameters.add(PathParameter(query, file))


class ReadFile(Command):
    
    @classmethod 
    def js_code(cls,  element_id, file):
        
        condensed_path = file
        #condensed_path = condense_path(file)
        js_code = f'setFilePath( "{element_id}", "{condensed_path}" );' 
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        file_dialog = open_command.dialog.OpenDialog(parent=chemem.session.ui.main_window, starting_directory='' )
        file = file_dialog.get_path()
        if file is not None:
            if os.path.isfile(file):
                js_code = cls.js_code(query, file)
                cls.update_chemem(chemem, file, query)
                
            else:
                js_code= f"alert(File does not exist: {file});"
            
            chemem.run_js_code(js_code)
    
    @classmethod
    def update_chemem(cls,chemem, file, query):
        
        chemem.parameters.add(PathParameter(query, file))

class ReadFileAndAlterChemEM(Command):
    
    @classmethod 
    def js_code(cls):
        
        js_code = 'sendSiteValueToBackend("chemem:UpdateExes", "None");'
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        file_dialog = open_command.dialog.OpenDialog(parent=chemem.session.ui.main_window, starting_directory='' )
        file = file_dialog.get_path()
        if file is not None:
            if os.path.isfile(file):
                js_code = cls.js_code()
                cls.update_chemem(chemem, file, query)
                
            else:
                js_code= f"alert(File does not exist: {file});"
            
            chemem.run_js_code(js_code)
    
    @classmethod
    def update_chemem(cls,chemem, file, query):
        
        attribute = getattr(chemem, query)
        attribute += [PathParameter(file, file)]
        setattr(chemem, query, attribute)

class UpdateExes(Command):
    
    @classmethod 
    def js_code(cls, exe_names):
        options_json = json.dumps(exe_names)

        js_code = f'''
        updateSelect("chememExes", {options_json});
        onExesPopulated();
        '''
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        exe_names = []
        for index, exe in enumerate(chemem.avalible_chemem_exes):
            exe_names.append( cls.get_exe_id( index, exe) )
        
        js_code = cls.js_code(exe_names)
        chemem.run_js_code(js_code)
    
    @classmethod 
    def get_exe_id(cls, index, exe ):
        return f'{exe.value}'


class SetCurrentExe(Command):
    
    @classmethod 
    def run(cls, chemem, query):
        path = [i for i in chemem.avalible_chemem_exes if i.value == query.value][0]
        cls.update_chemem(chemem, path)
    
    @classmethod 
    def update_chemem(cls, chemem, path):
       chemem.parameters.parameters['chememBackendPath'] = path 
            
    

class GetAvaliblePlatforms(Command):
    
    @classmethod 
    def js_code(cls, platform_name):
        js_code = f'addPlatformToList("{platform_name}");'
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
    
        for platform_name in chemem.avalible_platforms:
            js_code = cls.js_code(platform_name)
            chemem.run_js_code(js_code)
        chemem.platforms_set = True

class SetPlatform(Command):
    @classmethod
    def run(cls, chemem, query):
        #store the platform name string, not the Parameter wrapper, so it can be
        #passed straight to Platform.getPlatformByName when the simulation builds.
        chemem.platform = getattr(query, 'value', query)


class UpdateModels(Command):
    @classmethod
    def js_code(cls, model_names):
        options_json = json.dumps(model_names)  # Ensure model names are in JSON array format
        js_code = f'''
        updateSelect("models", {options_json});
        onModelsPopulated();
        '''
        
        return js_code
      
    @classmethod
    def _tracked_setup_ligand_model_ids(cls, chemem):
        tracked = set()
        for rec in getattr(chemem, "setup_ligands", {}).values():
            mdl = rec.get("model")
            if mdl is not None and getattr(mdl, "id", None) is not None:
                tracked.add(tuple(mdl.id))

            mid = rec.get("model_id")
            if mid is not None:
                tracked.add(tuple(mid))
        return tracked

    @classmethod
    def run(cls, chemem, query):
        tracked_model_ids = cls._tracked_setup_ligand_model_ids(chemem)
        model_names = []
        for model in chemem.session.models:
            if not isinstance(model, AtomicStructure):
                continue
            if tuple(model.id) in tracked_model_ids:
                continue  # skip tracked SDF ligand helper models
            model_names.append(cls.get_model_id(model))

        
        js_code = cls.js_code(model_names)
        chemem.run_js_code(js_code)
    


    @classmethod 
    def get_model_id(cls, model):
        return f'{".".join([str(i) for i in model.id]) } - {model.name}'

class SetCurrentModel(Command):
    
    @classmethod 
    def run(cls, chemem, query):
        
        if chemem.session.models.have_id(query.value):
            model = [i for i in chemem.session.models if i.id == query.value][0]
            cls.update_chemem(chemem, model)
        else:
            js_code = f"alert('ChemEM can't assign model with id: {query.name}');"
            chemem.run_js_code(js_code)
    
    @classmethod 
    def update_chemem(cls, chemem, model):
        """Update ChemEM object state data."""
        if 'current_model' in chemem.parameters.parameters:
            chemem.parameters.parameters['current_model'] = model 
        
        else:
            chemem.parameters.parameters['current_model'] = model

class UpdateMaps(Command):
    @classmethod
    def js_code(cls, model_names):
        options_json = json.dumps(model_names)  
        js_code = f'''
        updateSelect("maps", {options_json});
        onMapsPopulated();
        ''' #can remove on models populated!!!
        return js_code
    
    @classmethod
    def run(cls, chemem, query):
        model_names = []
        for index, model in enumerate(chemem.session.models): 
            
            if isinstance(model, Volume):
                model_names.append(cls.get_map_id(model))
        
        js_code = cls.js_code(model_names)
        chemem.run_js_code(js_code)    
    
    @classmethod 
    def get_map_id(cls, model):
        return f'{".".join([str(i) for i in model.id]) } - {model.name}'


class UpdateChemEMParameter(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.parameters.add(query)

class SetCurrentMap(Command):
    #TODO! add cache functions!!!
    @classmethod 
    def run(cls, chemem, query):
        
        
        if query is None: #change to N/A
            if 'current_map' in chemem.parameters.parameters:
                del chemem.parameters.parameters['current_map']
                
        elif chemem.session.models.have_id(query.value):
            model = [i for i in chemem.session.models if i.id == query.value][0]
            cls.update_chemem(chemem, model)
        else:
            js_code = f"alert('ChemEM can't assign map with id: {query.name});"
            chemem.run_js_code(js_code)
    
    @classmethod 
    def update_chemem(cls, chemem, model):
        """Update ChemEM object state data."""
        
        chemem.parameters.parameters['current_map'] = model 
        

class AddLigandSmiles(Command):
    @classmethod 
    def js_code(cls, smiles):
        js_code = f"alert(Invalid Ligand SMILES: {smiles});"
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        valid_smiles = cls.validate_smiles(query.value)
        if valid_smiles:
            chemem.parameters.add_list_parameter('Ligands', query)
        else:
            js_code = cls.js_code(query.value)
            chemem.run_js_code(js_code)
    
    @staticmethod 
    def validate_smiles(smiles):
        smiles = Chem.MolFromSmiles(smiles)
        if smiles is not None:
            return True
        else:
            return False


class AddLigandFile(Command):
    
    @classmethod
    def js_code(cls, file):
        js_code = f"alert(Invalid Ligand File: {file});"
        return js_code

    @classmethod 
    def run(cls, chemem, query):
        
        valid_file = validate_ligand_file(query.value)
        #valid_file = True
        if valid_file:

            chemem.parameters.add_list_parameter('Ligands', query)
            #tracking ligand models opened by sdf file (shared registration)
            register_tracked_ligand(chemem, query.value,
                                    ligand_id=_normalise_ligand_id(query.name))
        else:
            js_code = cls.js_code(query.value)
            chemem.run_js_code(js_code)
            


class RemoveLigand(Command):
    @classmethod 
    def js_code(cls):
        pass 
    @classmethod 
    def run(cls, chemem, query):
        chemem.parameters.remove_list_parameter('Ligands', query)
        if getattr(chemem, '_tracked_ligands') is not None:
            record = _pop_tracked_ligand(chemem, query)
            if record is None:
                if hasattr(chemem, "push_tracked_ligands_to_ui"):
                    chemem.push_tracked_ligands_to_ui()
                return
            model = record.get("model")
            if (
                model is not None
                and getattr(model, "id", None) is not None
                and chemem.session.models.have_id(tuple(model.id))
            ):
                chemem.session.models.remove([model])
            if hasattr(chemem, "push_tracked_ligands_to_ui"):
                chemem.push_tracked_ligands_to_ui()




