# This file is part of the ChemEM-X software.
#
# Copyright (c) 2024 - Aaron Sweeney
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>



from . import view
import os
import subprocess
import json
import numpy as np
from json.decoder import JSONDecodeError
from urllib.parse import parse_qs
from rdkit import Chem
import datetime
import copy 
import uuid
import threading
import time
import tempfile
import shutil

from chimerax.ChemEM.chemem_job import  JobHandler, LocalChemEMJob, SimulationJob, SIMULATION_JOB, CHEMEM_JOB, EXPORT_SIMULATION
from chimerax.ChemEM.map_masks import  SignificantFeaturesProtocol
from chimerax.ChemEM.rdtools import Protonate, smiles_is_valid, ChemEMResult, SolutionMap
from chimerax.ChemEM.simulation import Simulation
from chimerax.ChemEM.mouse_modes import  DragCoordinatesMode
from chimerax.mouse_modes import MouseMode
from chimerax.markers import MarkerSet
from chimerax.ui import HtmlToolInstance
from chimerax.atomic.structure import AtomicStructure
from chimerax.core.models import ADD_MODELS, REMOVE_MODELS, MODEL_POSITION_CHANGED
from chimerax.core.tasks import ADD_TASK, REMOVE_TASK
from chimerax.map import Volume
from chimerax import open_command
from chimerax.core.commands import run 
from chimerax.mouse_modes.std_modes import MovePickedModelsMouseMode
from chimerax.markers.mouse import MoveMarkersPointMouseMode
from chimerax.geometry import distance as get_distance
import parmed
from openmm import XmlSerializer
from openmm import LangevinIntegrator, Platform
from openmm import app
from openmm import unit
from openmm import MonteCarloBarostat, XmlSerializer, app, unit, CustomCompoundBondForce, Continuous3DFunction, vec3, Vec3
from scipy.ndimage import gaussian_filter
from openmm.unit.quantity import Quantity



def pin_force():
    expr = "pin_k * ((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
    f = CustomCompoundBondForce(1, expr)
    f.addPerBondParameter("pin_k")
    f.addPerBondParameter("x0")
    f.addPerBondParameter("y0")
    f.addPerBondParameter("z0")
    return f

def pin_atoms(idx_to_pin, struct, pin_k=500):

    pin_f = pin_force()
    for atom_index in idx_to_pin:
        
        atm = struct.atoms[atom_index]
        x0, y0, z0 = Quantity(
            value=[atm.xx, atm.xy, atm.xz], unit=unit.angstrom)
        #x0, y0, z0 = atm.xx, atm.xy, atm.xz
        pin_f.addBond([atom_index], [pin_k, x0, y0, z0])
    return pin_f

def map_potential_force_field(m, origin, apix, box_size, global_k, blur=0):
     
    f = CustomCompoundBondForce(1, '')
    d3d_func = compute_map_field(m,origin, apix, box_size, blur)
    f.addTabulatedFunction(name='map_potential', function=d3d_func)
    f.addGlobalParameter(name='global_k', defaultValue=global_k)
    f.addPerBondParameter(name='individual_k')
    f.setEnergyFunction('-global_k * individual_k * map_potential(z1,y1,x1)')
    return f

def compute_map_field(m, origin, apix, box_size, blur=0, d3d_func=None):
    #f = CustomCompoundBondForce(1,'')
    #     d3d_func = Discrete3DFunction(*m.fullMap.shape, m.fullMap.ravel().copy())
    morg = np.array(origin)[::-1] - apix/2

    mdim = np.array(box_size)* apix

    mmax = morg+mdim

    mod_m = gaussian_filter(m, blur)
    minmaxes = np.array(list(zip(morg/10, mmax/10))).ravel()
    

    if d3d_func is None:

        d3d_func = Continuous3DFunction(
            *mod_m.shape, mod_m.ravel(order="F"), *minmaxes)

    else:
        d3d_func.setFunctionParameters(
            *mod_m.shape, mod_m.ravel(order="F"), *minmaxes)
    
    return d3d_func


import numpy as np

def get_inc_index(positions, map_origin, box_size, apix):
    """
    Determine which atoms are inside a specified box.

    Parameters:
    - positions (numpy.ndarray): Array of atom positions (shape: num_atoms x 3).
    - map_origin (tuple): The (x, y, z) coordinates of the box origin.
    - box_size (tuple): The size of the box in pixels (x, y, z).
    - apix (float): Angstroms per pixel, the scale factor for box dimensions.

    Returns:
    - Tuple of lists: (indices_inside, indices_outside)
    """
    # Calculate the actual dimensions in angstroms
    box_dimensions = np.array(box_size) * apix

    # Initialize lists to store indices
    indices_inside = []
    indices_outside = []

    # Iterate through all positions and determine if inside or outside the box
    for index, position in enumerate(positions):
        # Convert position to be relative to the origin of the map
        rel_position = position - np.array(map_origin)
        
        # Check if the position is within the bounds in all three dimensions
        if np.all(rel_position >= 0) and np.all(rel_position <= box_dimensions):
            indices_inside.append(index)
        else:
            indices_outside.append(index)

    return indices_inside, indices_outside

#TODO! move this to another file
class Config:
    
    def __init__(self, paramter_object, session,):
        
        self.parameters = paramter_object 
        self.file = f"""
#╔═════════════════════════════════════════════════════════════════════════════╗
#║                          ChemEM-X Configuration File                        ║
#╚═════════════════════════════════════════════════════════════════════════════╝
#File created:  {datetime.datetime.now().strftime("%B %d, %Y, %I:%M %p")}
"""
        self.session = session
        self.output = './'
        
        
    
    def get_output(self):
        if self.check_variable('output'):
            
            if os.path.exists(self.parameters.parameters['output'].value):
                self.output = self.parameters.parameters['output'].value
        #self.file += f'output = {self.output}\n'
    
    def get_protein(self):
        if self.check_variable('current_model'):
            model = self.parameters.parameters['current_model']
            model_file = f"{model.name}.pdb"
            
            rm_model = False
            if model.id is None:
                self.session.models.add([model])
                rm_model = True
                
            model_id = f"#{'.'.join([str(i) for i in model.id])}"
            model_file = os.path.join(self.parameters.parameters['model_path'], model_file)
            com = f'save {model_file} format pdb models {model_id}'
            run(self.session, com)
            if rm_model:
                self.session.models.remove([model])
            
            self.file += '\n'
            self.file += f"protein = {model_file}\n"
        else:
            #TODO! change to alert make more specific
            print('Model Variable Needed')
        
        #write the file
    
    def get_ligand(self):
        if self.check_variable('Ligands'):
            
            for ligand in self.parameters.parameters['Ligands']:
                self.file += f"ligand = {ligand.value}\n" 
        else:
            #TODO! change to alert make more specific
            print('Ligand Varable Needed!!')
    
    def get_map(self):
        if self.check_variable('current_map'):
            map_path = self.parameters.parameters['current_map'].path
            if self.check_variable('resolution'):
                self.file += f"densmap = {map_path}\n"
                #self.file += f"resolution = {self.parameters.parameters['resolution'].value}\n"
            else:
                print('Map Entered with no resolution')
                return
        else:
            if self.check_variable('resolution'):
                print('Resolution Entered with No map !!!')
    
    def get_binding_sites(self):
        if self.check_variable('binding_sites_conf'):
            for site in self.parameters.parameters['binding_sites_conf']:
                self.file += f"centroid = {site.centroid}\n"
                self.file += f"segment_dimensions = {site.box_size}\n"
        else:
            #TODO!
            print('No Value entered for binding site!!')
    
    def get_parameters(self):
        self.file += '\n'
        for param in  self.parameters.parameters.values():
            if hasattr(param, 'chemem_string'):
                self.file += param.chemem_string()
                
    def check_variable(self, var):
        if var in self.parameters.parameters:
            return True 
        else:
            return False

    def write_file(self):
        file_name = os.path.join(self.output, 'ChemEM-X_conf.conf')
        with open(file_name, 'w') as f:
            f.write(self.file)
        self.file_path = file_name
        
            
    def run(self):
        self.get_output()
        self.get_protein()
        self.get_ligand()
        self.get_map()
        self.get_binding_sites()
        self.get_parameters()
        self.write_file()
        print(self.file) #TODO! for debuging!!
        

class CHEMEM(HtmlToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = False  # No session saving for now
    CUSTOM_SCHEME = "chemem"
    display_name = "chemem" # HTML scheme for custom links
    PLACEMENT = None
    #help = "help:user/tools/pickluster.html" #add this!!
    
    def __init__(self, session, tool_name):
        super().__init__(session, tool_name, size_hint=(600, 1000))
        self.parameters = Parameters()
        
        self.view = view.ChemEMView(self) #!!!
        
        self.add_model_handeler = self.session.triggers.add_handler(ADD_MODELS, self.update_models)
        self.remove_model_handeler = self.session.triggers.add_handler(REMOVE_MODELS, self.update_models)
        self.moved_model_handeler = self.session.triggers.add_handler(MODEL_POSITION_CHANGED, self.model_position_changed)
        #self.job_added_task = self.session.triggers.add_handler(ADD_TASK, self.add_task)
        self.job_remove_task =  self.session.triggers.add_handler(REMOVE_TASK, self.remove_task)
        self.job_handeler = JobHandler()
        self.view.render()
        self.current_renderd_site_id = None
        self.rendered_site = None
        self.current_protonation_states = None
        self.current_significant_features_object = None
        self.current_loaded_result = None
        self.current_simulation = Parameters()
        self._handler_references = [self.add_model_handeler, 
                                     self.remove_model_handeler, 
                                     self.moved_model_handeler,
                                     self.job_remove_task]
        
        self.avalible_chemem_exes =  get_chemem_paths()
        self.avalible_platforms = self.get_platforms()
        self.platforms_set = False
        self.temp_build_dir = None
        self.current_restraints = {}
        #remove!!
        self.avalible_binding_sites = 0
        
        
        #enable for debugging
        #TODO!
        self.session.metadata = self
        
    def get_platforms(self):
        import chimerax 
        openmm_plugins_dir = os.path.join(chimerax.app_lib_dir, 'plugins')
        Platform.loadPluginsFromDirectory(openmm_plugins_dir)
        avalible_platforms = []
        for i in range(Platform.getNumPlatforms()):
            avalible_platforms.append(Platform.getPlatform(i).getName())
        return avalible_platforms
        
    def remove_task(self, *args):
        
        job = args[1]
        if job.job_type == CHEMEM_JOB:
            
            if job.success:
                js_code = f'updateJobStatus( {job.id}, "Completed");'
            else:
                js_code = f'updateJobStatus( {job.id}, "Failed");'
            self.run_js_code(js_code)
            
        elif job.job_type == EXPORT_SIMULATION:
            #self.openmm_test_run(job)
            
            if 'current_map' in self.current_simulation.parameters:
                current_map = self.current_simulation.parameters['current_map']
            else:
                current_map = None
            
            chimerax_model = self.current_simulation.parameters['AddedSolution'].pdb_object
            
            simulation = Simulation(self.session, job.params.output, current_map, platform_name = self.platform)
            self.simulation = simulation
            simulation_model, atoms_to_position_index = simulation.get_model_from_complex_structure(chimerax_model)
            self.hbond_idxs  = simulation.get_simualtion_hbond_pairs(atoms_to_position_index)
            simulation.set_forces()
            self.simulation_model = simulation_model
            self.atoms_to_position_index = atoms_to_position_index
            self.atoms_to_position_index_as_dic = {i[0]:i[1] for i in atoms_to_position_index}
            #self.session.models.add([simulation_model])
            self.current_simualtion_id = None
            
            js_code = 'openTab(event, "RunSimulationTab");'
            self.run_js_code(js_code)
            self.remove_temp_directories()
            

            
            #do this properly on init!!
            
    def remove_temp_directories(self):
        
            if self.temp_build_dir is not None:
                try:
                    shutil.rmtree(self.temp_build_dir)
                    self.temp_build_dir = None
                except Exception as e:
                    print("Unable To remove directory {self.temp_build_dir}, Error: {e}")
    
    def update_simulation_model(self):
        
        t1 = time.perf_counter()
        positions = np.array(self.simulation.get_positions())
        for atom, index in self.atoms_to_position_index:
            atom.coord = positions[index]
        t2 = time.perf_counter() - t1
        #print('UpdateModleTime', t2)
    
    
    def add_task(self, *args):
        js_code = 'alert("JobAdded");'
        self.run_js_code(js_code)


    def update_positions(self,positions, atom_to_position_idx):
        for atom, idx in atom_to_position_idx:
            atom.coord = np.array([positions[idx].x, positions[idx].y, positions[idx].z])
    
    def run_simulation(self):
        job = SimulationJob(self.session, self)
        job.start()  
        self.current_simualtion_id = job.id
        self.job_handeler.add_job(job)
        
        
    
    def run(self, executable, job_type):
        conf_file_path = self.conf_handler.file_path 
        command = f"{executable} {conf_file_path}"
        job = LocalChemEMJob(self.session, command, self.conf_handler , job_type)
        job_data = {"id": job.id, "status": "running"}
        
        if job_type == CHEMEM_JOB:
            job_data_json = json.dumps(job_data)
            js_code = f"addJob({job_data_json});"
            self.run_js_code(js_code)
            job.start()    
            js_code = "resetQuickConfigTab();"
            self.run_js_code(js_code)
            self.job_handeler.add_job(job) 
        
        elif job_type == EXPORT_SIMULATION:
            job.start()
            self.job_handeler.add_job(job) 
            
        

    
    def build_simulation(self):
        #create a tempory directory to write files to 
        #run the export_simulation blocking?
        chemem_path = self.parameters.parameters['chememBackendPath'].value 
        
    
    def delete(self):
        for handler in self._handler_references:
            handler.remove() 
        
        self._handler_references.clear() 
        #remove tempory file create by protonation
        CleanProtonationImage().run(self, None)
        self.remove_temp_directories()
        super().delete()
    
    def mkdir(self, path):
        try:
            os.mkdir(path)
        except Exception as e:
            pass
    
    def make_conf_file(self, parameter_object):
        '''
        Make a configuration file object and save it to the state. 
        Only one configuration file is made at a time, run ChemEM will
        find the configuration file at self.conf_handeler
        

        Parameters
        ----------
        parameter_object : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if 'output' in parameter_object.parameters:
            model_path = os.path.join(parameter_object.parameters['output'].value, 'Inputs')
        else:
            model_path = os.path.join('./', 'Inputs')
        
        self.mkdir(model_path)
        parameter_object.parameters['model_path'] = model_path
        conf_handler = Config(parameter_object.copy(), self.session)
        conf_handler.run()
        self.conf_handler = conf_handler
    
    def make_conf_file_old(self):
        
        if 'output' in self.parameters.parameters:
            model_path = os.path.join(self.parameters.parameters['output'].value, 'Inputs')
            
        else:
            model_path = os.path.join('./', 'Inputs')
        
        self.mkdir(model_path)
        self.parameters.parameters['model_path'] = model_path
        conf_handler = Config(self.parameters.copy(), self.session)
        conf_handler.run()
        self.conf_handler = conf_handler
 
        
    def run_js_code(self, js_code):
        self.html_view.page().runJavaScript(js_code)
    
    
    def select_model_js(self, model):
        model_key = UpdateModels.get_model_id(model)
        return  f'selectOptionByValue("models", "{model_key}");'
        
    def update_models(self, *args):
        
        """
        Handles adding and removing models.
        
        Parameters
        ----------
        *args : list
            - args[0] : str
                The function executed (e.g., 'remove models').
            - args[1] : list
                The list of models involved in the operation.
        """
        #TODO! refactor-this 
        
        current_map = self.parameters.get_parameter('current_map')
        current_model = self.parameters.get_parameter('current_model')
        current_simulation_map = self.current_simulation.get_parameter('current_map')
        
        if args[0] == 'remove models':
            
            if current_model in args[1]:
                models_left = [i for i in self.session.models if isinstance(i, AtomicStructure) and i not in args[1]]
                self.clear_binding_site_tabs()
               
                if models_left:
                    #if a model still open set this as the current model 
                    UpdateModels.execute(self, 'UpdateModels', '_')
                    self.run_js_code( self.select_model_js( models_left[0] ))
                    return
                else:
                    #remove current model and cleanupTODO!         
                    del self.parameters.parameters['current_model']
                    return
                    

            if current_map in args[1]:
                self.clear_binding_site_tabs()
                #set current_map to None
                UpdateMaps.execute(self, 'UpdateMaps', '_')
                
                
                return
        
        if current_map is not None:
            current_map_key = UpdateMaps.get_map_id(current_map)
            js_code_maps = f'selectOptionByValue("maps", "{current_map_key}");'

        if current_model is not None:
            current_model_key = UpdateModels.get_model_id(current_model)
            js_code_model = f'selectOptionByValue("models", "{current_model_key}");'
        
        
        if current_simulation_map is not None:
            current_simulation_map_key = UpdateSimulationMaps.get_map_id(current_simulation_map)
            js_code_simulated_map =  f'selectOptionByValue("SimulationMaps", "{current_simulation_map_key}");'
        
        
        
        UpdateModels.execute(self, 'UpdateModels', '_')
        UpdateMaps.execute(self, 'UpdateMaps', '_')
        UpdateSimulationMaps.execute(self, 'UpdateSimulationMaps', '_')
        
        if current_map is not None:
            self.run_js_code(js_code_maps)
        
        if current_model is not None:
            self.run_js_code(js_code_model)
        if current_simulation_map is not None:
            self.run_js_code(js_code_simulated_map)
        
        
    def model_position_changed(self, *args):
        
        self.args = args
        if isinstance(args[1], ChemEMMarker):
            position = args[1].residues[0].atoms[0].coord 
            self.set_marker_position(position)
        
    def set_marker_position(self, position):
        self.session.metadata = self
        
        if not self.marker_wrapper in self.session.models:
            
            self.session.models.add([self.marker_wrapper])
        
        self.marker_wrapper.residues[0].atoms[0].coord = position
        #HERERER!!!
    
    def render_site_from_key(self, key):
        if self.rendered_site is not None:
            self.rendered_site.reset()
            self.rendered_site = None 
        
        site = self.parameters.get_list_parameters('binding_sites', key)[0]
        self.rendered_site =  RenderBindingSite(self.session, site, 
                           self.parameters.parameters['current_model'],
                           self.parameters.get_parameter('current_map'))
        
        
    def handle_scheme(self, url):
        print('')
        command = url.path()
        query = self.extract_query(url)
        
        for command_class in Command.__subclasses__():
            command_class().execute(self, command, query)
    
        print('COM:',command)
        print('QUERY', query)
        try:
            print('name', query.name)
            print('value', query.value)
        except:
            pass

    def extract_query(self, url):
        query = parse_qs(url.query())
        query = query['siteValue'][0]
        print('unfilterd_query', query)
        
        try:
            query = json.loads(query)
        except JSONDecodeError:
            pass 

        if type(query) == dict:
            query = self.parameters.get_from_query(query)
            
        
        return query
    
    def clear_binding_site_tabs(self):
        self.parameters.clear_binding_site_tabs(self)
    
    


#only need one marker initially not visable, the ChemEM marker.

#Move this around with each site is simpler than marker per thing.

#╔═════════════════════════════════════════════════════════════════════════════╗
#║                                 Parameters                                  ║
#╚═════════════════════════════════════════════════════════════════════════════╝


class Parameter:
    @classmethod
    def get_all_subclasses(cls):
        """
        Recursively find all subclasses of the current class.
        """
        subclasses = set(cls.__subclasses__())
        for subclass in cls.__subclasses__():
            subclasses.update(subclass.get_all_subclasses())
        return subclasses
    
     
    def chemem_string(self):
        return f"{self.name} = {self.value}\n"


class NumericParameter(Parameter):
    _type = str  

    def __init__(self, name, value):
        super().__init__()
        self.name = name
        self.value = self._type(value)

    @classmethod
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            name = query['id']
            value = query['value']
            return cls(name, value)
    
    
    

class FloatParameter(NumericParameter):
    _type = float
    

class IntParameter(NumericParameter):
    _type = int  


class StringParameter(Parameter):
    _type = str  
    def __init__(self, name,  default ):
        self.name = name 
        self.value = default 
   
    @classmethod 
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            
            return cls(query['id'],
                       query['value'])
   
        
class PathParameter(StringParameter):
    def __init__(self, name, default):
        super().__init__(name, default)

class SmilesParameter(StringParameter):
    def __init__(self, name, default):
        super().__init__(name, default)


class BooleanParameter(Parameter):
    _type = bool
    def __init__(self, name, default):
        self.name = name 
        self.value = default 
    
    @classmethod 
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            value = cls.get_value(query['value'])
            return cls(query['id'],
                       query['value'])
    @staticmethod 
    def get_value(value):
        if value == 'false':
            value = 0
        if value == 'true':
            value = 1

class ModelParameter(Parameter):
    def __init__(self, name, default):
        self.name = name
        self.value = default 
        
    @classmethod
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            value = cls.get_value(query['value'])
            return cls(query['id'],
                       value)

    @staticmethod 
    def get_value(value):
        value = value.split('-')[0].replace(' ', '')
        value = value.split('.')
        value = tuple([int(i) for i in value])
        return value

class MapParameter(ModelParameter):
    def __init__(self, name, default):
        super().__init__(name, default)
    
    @classmethod
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            
            if query['id'] == 'None':
                return None
            
            value = cls.get_value(query['value'])
            return cls(query['id'],
                       value)


class BindingSiteParameter(Parameter):
    def __init__(self, name, value):
        self.name = name 
        self.value = value 
    
    @classmethod
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            value = cls.get_value(query['value'])
            c =  cls(query['id'],
                       value)
            c.centroid = c.value[0]
            c.box_size = c.value[1]
            return c
    
    @classmethod 
    def get_from_centroid(cls, centroid ,chemem):
        
        binding_site_id = generate_unique_id()
        centroid = tuple(round(i,3) for i in centroid)
        box_size = (20,20,20) #default
        value = [centroid, box_size]
        c = cls(binding_site_id,
                value)
        c.centroid = centroid
        c.box_size = box_size 
        
        return c
    
    @classmethod
    def get_value(cls, value):
        values = value.split('|')
        centroid = BindingSiteParameter.convert_to_float_tuple(values[0])
        box_size = BindingSiteParameter.convert_to_float_tuple(values[1])
        return [centroid, box_size]
    
    @staticmethod
    def convert_to_float_tuple(s):
        stripped = s.strip()[1:-1]
        split_strings = stripped.split(',')
        return tuple(float(item) for item in split_strings)
        
#get the bindinsite paramter and the currentbindingsite id

class ProtonationParameter(Parameter):
    def __init__(self, name, value):
        self.name = name 
        self.value = value 
    
    @classmethod
    def get_from_query(cls, query):
       
        if query['class'] == cls.__name__:
            value = cls.get_value(query['value'])
            c =  cls(query['id'],
                       value)
            c.max_pH = value[0]
            c.min_pH = value[1]
            c.pka_prec = value[2]
            c.smiles = query['id']
            return c
            
    @classmethod 
    def get_value(cls, value):
        
        values = value.split('|')
        #pHmax, pHmin, pka_precsion
        value = [float(i) for i in values]
        return value
        

class Parameters:
    def __init__(self):
        self.parameters = {}

    def add(self, param):
        self.parameters[param.name] = param
        return param
    
    def add_list_parameter(self, list_name, param):
        if list_name in self.parameters:
            self.parameters[list_name].append(param)
        else:
            self.parameters[list_name] = [param]
    
    def remove_list_parameter(self, list_name, param_id):
        
        if list_name in self.parameters:
            self.parameters[list_name] = [i for i in self.parameters[list_name] if  i.name != param_id]
        
    def get_list_parameters(self, list_name, param_id):
        if list_name in self.parameters:
            return [i for i in self.parameters[list_name] if i.name == param_id]
        else:
            return []
    
    def get_from_query(self, query):
        
        for parameter in Parameter.get_all_subclasses():
            param = parameter.get_from_query(query)
            if param is not None:
                return param

    def get_parameter(self, parameter):
        if parameter in self.parameters:
            return self.parameters[parameter]
        else:
            return None
    
    def get_parameter_names(self):
        return self.parameters.keys()

    def get_value(self, parameter):
        return self.parameters[parameter].value
    #housekeeping functions!! 
    def _clear(self):
        self.parameters = {}
    
    def clear(self):
        retained = ['current_model', 'current_map', 'chememBackendPath']
        keys_to_delete = [key for key in self.parameters if key not in retained]
        for key in keys_to_delete:
            del self.parameters[key]
    
    def clear_binding_site_tabs(self, chemem):
        sites = self.get_parameter('binding_sites')
        if sites is not None:
            for site in sites:
                
                js_code = f'triggerDeleteButtonClickOnBindingSite("{site.name}");'
                print(js_code)
                chemem.run_js_code(js_code)
                    
                    
    
    def copy(self):
        return copy.copy(self)

#╔═════════════════════════════════════════════════════════════════════════════╗
#║                                  Commands                                   ║
#╚═════════════════════════════════════════════════════════════════════════════╝


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
            #if True:
            try:
                cls.run(chemem, query)
            except Exception as e:
                alert_message = f'ChemEM Error, unable to run command: {cls.__name__} - {e}'
                print(alert_message)
                js_code = f'alert("{alert_message}");'
                chemem.run_js_code(js_code)
                
    
    @classmethod
    def run(cls, chemem, query):
        """Utility method to run JavaScript code on a given HTML view."""
        pass


class LoadChimeraXJob(Command):
    @classmethod 
    def run(cls, chemem, query):
        if query in chemem.job_handeler.jobs:
            job = chemem.job_handeler.jobs[query]
            path = PathParameter('loadJob', job.params.output)
            chemem.parameters.add(path)
            LoadJobFromPath().run(chemem,query)
            
            
            
class LoadJobFromPath(Command):
    @classmethod
    def run(cls, chemem, query):
        #removed currently loaded job 
        if chemem.current_loaded_result is not None:
            
            js_code =  'clearResultsLists();'
            chemem.run_js_code(js_code)
            chemem.current_loaded_result.clear()
            chemem.current_loaded_result = None
            
            
        load_path = chemem.parameters.get_parameter('loadJob')
        #check what data is avalible 
        preprocessing_path = os.path.join( load_path.value, 'preprocessing')
        fitting_path = os.path.join( load_path.value, 'fitting')
        postprocessing_path = os.path.join( load_path.value, 'post_processing')
        
        #preprocessing files 
        if os.path.exists( preprocessing_path ):

            pre_processed_map_files = [i for i in os.listdir(preprocessing_path) if i.endswith('.mrc')]
            
        else:
            pre_processed_map_files = []
        
        
        #fitting files
        if os.path.exists( fitting_path ):
            
            sdf_files = [i for i in os.listdir(fitting_path) if i.endswith('.sdf')]
            fitting_results_path = os.path.join(fitting_path, 'results.txt')
            

            if os.path.exists( os.path.join(fitting_path, 'PDB')):
                pdb_path =  os.path.join(fitting_path, 'PDB')
                pdb_files = [i for i in os.listdir(pdb_path) if i.endswith('.pdb')]
            else:
                pdb_files = None
            
            if sdf_files and pdb_files is not None:
                
                fitting_file_pairs = cls.pair_files_fitting(pdb_files, sdf_files)
        else:
            fitting_file_pairs = []
            fitting_results_path = os.path.join(fitting_path, 'results.txt')
                

        #post-processing_files
        
        if os.path.exists(postprocessing_path):
            pdb_files = [i for i in os.listdir(postprocessing_path) if i.endswith('.pdb')]
            sdf_files = [i for i in os.listdir(postprocessing_path) if i.endswith('.sdf')]
            
            if sdf_files and pdb_files:
                postproc_file_pairs = cls.pair_files_postproc(pdb_files, sdf_files)
            
            post_proc_results_path = os.path.join(postprocessing_path, 'results.txt')
            
        
        else:
            postproc_file_pairs = []
            post_proc_results_path = os.path.join(postprocessing_path, 'results.txt')
        
        results_object = ChemEMResult( chemem.session, 
                                       preprocessing_path,
                                       pre_processed_map_files,
                                       fitting_path,
                                       fitting_file_pairs,
                                       fitting_results_path,
                                       postprocessing_path,
                                       postproc_file_pairs,
                                       post_proc_results_path
                                      )
        
        chemem.current_loaded_result = results_object
        for message in chemem.current_loaded_result.messages:
            chemem.run_js_code(message)
    
            
    @classmethod 
    def pair_files_fitting(cls, pdb_files, sdf_files):
        fitting_file_pairs = []
        
        for pdb in pdb_files:
            pdb_id = pdb.split('_')[1].replace('.pdb', '')
            pair = [pdb]
            for sdf in sdf_files:
                sdf_id = sdf.split('_')[1] 
                if pdb_id == sdf_id:
                    pair.append(sdf)
            
            fitting_file_pairs.append(pair)
            
        return fitting_file_pairs
    
    @classmethod 
    def pair_files_postproc(cls, pdb_files, sdf_files):
        fitting_file_pairs = []
        
        for pdb in pdb_files:
            pdb_id = pdb.split('_')[1].replace('.pdb', '')
            pdb_cycle_id =  pdb.split('_')[3].replace('.pdb', '')
            pair = [pdb]
            for sdf in sdf_files:
                sdf_id = sdf.split('_')[1] 
                sdf_cycle_id =  sdf.split('_')[3] 
                if pdb_id == sdf_id and pdb_cycle_id == sdf_cycle_id:
                    pair.append(sdf)
            
            fitting_file_pairs.append(pair)
            
        return fitting_file_pairs


class HideSolutionMap(Command):
    @classmethod 
    def run(cls, chemem, query):
        query = str(query)
        if chemem.current_loaded_result is not None:
            
            for density_map in chemem.current_loaded_result.map_objects:
                if density_map.id == query:
                    density_map.hide()
        

class ViewSolutionMap(Command):
    @classmethod 
    def run(cls, chemem, query):
        query = str(query)
        if chemem.current_loaded_result is not None:
            
            for density_map in chemem.current_loaded_result.map_objects:
                if density_map.id == query:
                    density_map.show()
        
class ViewFittingSolutionMap(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_loaded_result is not None:
            for solution in chemem.current_loaded_result.fitting_results:
                if solution.id == query:
                    solution.show_solution()
                    return
            for solution in chemem.current_loaded_result.postprocessing_results:
                if solution.id == query:
                    solution.show_solution()
                    return

class HideFittingSolutionMap(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_loaded_result is not None:
            for solution in chemem.current_loaded_result.fitting_results:
                if solution.id == query:
                    solution.hide_solution()
                    return
        
            for solution in chemem.current_loaded_result.postprocessing_results:
                if solution.id == query:
                    solution.hide_solution()
                    return
            


class ProtonateSmiles(Command):
    @classmethod 
    def js_code(cls, smiles):
        js_code = f'addSmilesToProtonationList("{smiles}");'
        return js_code 
    
    @classmethod
    def run(cls,chemem, query):
        #need to delete the list and images at this stage TODO!
        chemem.run_js_code("removeAllProtonatedSmiles();")
        p  = Protonate.from_query(query)
        p.protonate()
        
        for state in p.protonation_states:
            js_code = cls.js_code(state)
            chemem.run_js_code(js_code)
        
        chemem.current_protonation_states = p
        

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
                    print('file, ', file_path)
                    
                    js_code = cls.js_code(file_path)
                    chemem.run_js_code(js_code)
                    
                    #chemem.current_protonation_states.remove_temporary_file()
            except ValueError:
                js_code = f'alert("Smiles image not found {query}");'
                chemem.run_js_code(js_code)
                
           

class CleanProtonationImage(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_protonation_states is not None:
            if chemem.current_protonation_states.current_image_file is not None:
                chemem.current_protonation_states.remove_temporary_file()

class ResetConfig(Command):
    @classmethod 
    def run(cls, chemem, query):
        
        chemem.parameters.clear_binding_site_tabs(chemem)
        chemem.parameters.clear()

class RemoveJob(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.job_handeler.remove_job(query)
        
       

class IncrementBindingSiteCounter(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.avalible_binding_sites += 1
        
class AddBindingSite(Command):
    
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.add_list_parameter('binding_sites', query)
        chemem.current_binding_site_id = query.name
   
class AddBindingSiteToConf(Command):
    
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.add_list_parameter('binding_sites_conf', query)
        chemem.current_binding_site_id = query.name        

class RemoveBindingSite(Command):
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.remove_list_parameter('binding_sites', query)
        #TODO! check the effects of this elsewhere
        if chemem.rendered_site is not None:
            if query == chemem.rendered_site.binding_site.name:
                chemem.rendered_site.reset() 
                chemem.rendered_site = None

class RemoveBindingSiteFromConf(Command):
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.remove_list_parameter('binding_sites_conf', query)
        
class TransferSiteToConf(Command):
    
    @classmethod
    def js_code(cls, site):
        centroid_x = round(site.centroid[0], 3)
        centroid_y = round(site.centroid[1], 3)
        centroid_z = round(site.centroid[2], 3)
        box_x = int(site.box_size[0])
        box_y = int(site.box_size[1])
        box_z = int(site.box_size[2])
        
        return f"""
        document.getElementById("centroidX").value = {centroid_x};
        document.getElementById("centroidY").value = {centroid_y};
        document.getElementById("centroidZ").value = {centroid_z};
        document.getElementById("QuickboxSizeX").value = {box_x};
        document.getElementById("QuickboxSizeY").value = {box_y};
        document.getElementById("QuickboxSizeZ").value = {box_z};
        addCentroid();
        resetManualBindingSiteFields();
        """
    @classmethod 
    def run(cls, chemem, query):
        
        site = chemem.parameters.get_list_parameters('binding_sites', query)[0]
        #chemem.parameters.add_list_parameter('binding_sites_conf', site)
        chemem.run_js_code(cls.js_code(site))

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
    def run(cls, chemem, query):
        model_names = []
        for index, model in enumerate(chemem.session.models): 
            if isinstance(model, AtomicStructure):
                model_names.append( cls.get_model_id( model ) )
        
        js_code = cls.js_code(model_names)
        chemem.run_js_code(js_code)
    
    @classmethod 
    def get_model_id(cls, model):
        return f'{".".join([str(i) for i in model.id]) } - {model.name}'


class SetCurrentExe(Command):
    
    @classmethod 
    def run(cls, chemem, query):
        path = [i for i in chemem.avalible_chemem_exes if i.value == query.value][0]
        cls.update_chemem(chemem, path)
    
    @classmethod 
    def update_chemem(cls, chemem, path):
       chemem.parameters.parameters['chememBackendPath'] = path 
            
    
class SetCurrentModel(Command):
    
    @classmethod 
    def run(cls, chemem, query):
        if chemem.session.models.have_id(query.value):
            model = [i for i in chemem.session.models if i.id == query.value][0]
            cls.update_chemem(chemem, model)
        else:
            js_code = f"alert('ChemEM can't assign model with id: {query.name});"
            chemem.run_js_code(js_code)
    
    @classmethod 
    def update_chemem(cls, chemem, model):
        """Update ChemEM object state data."""
        if 'current_model' in chemem.parameters.parameters:
            chemem.parameters.parameters['current_model'] = model 
            
            
        else:
            chemem.parameters.parameters['current_model'] = model

class SetSimulationMap(Command):
    #TODO! add cache functions!!!
    @classmethod 
    def run(cls, chemem, query):
    
        if query is None: #change to N/A
            if 'current_map' in chemem.current_simulation.parameters:
                del chemem.current_simulation.parameters['current_map']
                
        elif chemem.session.models.have_id(query.value):
            model = [i for i in chemem.session.models if i.id == query.value][0]
            cls.update_chemem(chemem, model)
        else:
            js_code = f"alert('ChemEM can't assign simulation map with id: {query.name});"
            chemem.run_js_code(js_code)
    
    @classmethod 
    def update_chemem(cls, chemem, model):
        """Update ChemEM object state data."""
        
        chemem.current_simulation.parameters['current_map'] = model 
        
            

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

class UpdateSimulationMaps(Command):
    @classmethod 
    def js_code(cls, model_names):
        options_json = json.dumps(model_names)  
       
        js_code = f'''
        updateSelect("SimulationMaps", {options_json});
        onSimulationMapsPopulated();
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
        #TODO! valid_smiles = cls.validate_smiles(query.value)
        valid_file = True
        if valid_file:
            chemem.parameters.add_list_parameter('Ligands', query)
        
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


class UpdateChemEMParameter(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.parameters.add(query)
        
class UpdateSimulationParameter(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.current_simulation.add(query)
        
class SaveConfFile(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.make_conf_file()

class RunChemEM(Command):
    @classmethod 
    def run(cls, chemem, query):
        
        chemem.make_conf_file(chemem.parameters)
        executable = chemem.parameters.parameters['chememBackendPath'].value 
        chemem.run(executable,CHEMEM_JOB)


class SetMaunualBindingSite(Command):
    
    @classmethod 
    def js_code(cls):
        return "bindingSiteCounter++;"
    
    @classmethod 
    def run(cls, chemem, query):
        
        pass
        #update ligand bindig site id!! in js_code

class ActivateMarkerPlacement(Command):
    @classmethod 
    def run(cls, chemem, query):
        activate_marker_placement(chemem, BindingSiteParameter)
        

class BindingSiteFromMarker(Command):
    
    @classmethod
    def js_code(cls, site):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        js_code = f'''
        addBindingSiteEntry("{site_value}", "{site.name}");
        populateFieldsFromBackendString("{site_value}");
        '''
        return js_code
    
    @classmethod 
    def js_code_error(cls):
        message = "No Markers found.\\nPlace marker first to define binding site."
        return f'alert("{message}");'
    
    @classmethod 
    def run(cls, chemem, query):
        markers = []
        for model in chemem.session.models:
            if isinstance(model, MarkerSet):
                markers.append(model)
        
        if len(markers) == 0:
            chemem.run_js_code(cls.js_code_error())
        else:
            marker_centroids = [np.array(i.atoms[0].coord) for i in markers[0].residues]
            for centroid in marker_centroids:
                
                
                site  = BindingSiteParameter.get_from_centroid(centroid, chemem)
                chemem.parameters.add_list_parameter('binding_sites', site)
                chemem.current_binding_site_id = site.name 
                
                chemem.rendered_site =  RenderBindingSite(chemem.session, site, 
                                                          chemem.parameters.parameters['current_model'],
                                                          chemem.parameters.get_parameter('current_map'))
                
                js_code = cls.js_code(site)
                chemem.run_js_code(js_code)
                
                #don't think i need this?
                chemem.current_renderd_site_id = site.name
                #set current working site in backend!!!
    
class ResetBindingSiteFields(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.rendered_site is not None:
            chemem.rendered_site.reset() 
            chemem.render_site = None
            


class AddBindingSiteFromInputsOrChange(Command):
    
    @classmethod 
    def js_code(cls, site):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        js_code = f'addBindingSiteEntry("{site_value}", "{site.name}");'
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        #change or new!!
        if chemem.rendered_site is None:
            query.name = generate_unique_id()
            chemem.rendered_site =  RenderBindingSite(chemem.session, query, 
                                                      chemem.parameters.parameters['current_model'],
                                                      chemem.parameters.get_parameter('current_map'))

        else:
            
            site = chemem.rendered_site
            if query.centroid != site.binding_site.centroid :
                site.update_centroid(query.centroid)
                
            if query.box_size != site.binding_site.box_size :
                site.update_box_size(query.box_size)

            
            
class RenderBindingSiteFromClick(Command):
    
    @classmethod 
    def js_code(cls, site):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        js_code = f'populateFieldsFromBackendString("{site_value}");'
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        site = chemem.parameters.get_list_parameters('binding_sites', query)[0]
        if chemem.rendered_site is not None:
            chemem.rendered_site.reset()
        
        chemem.rendered_site =  RenderBindingSite(chemem.session, site, 
                                                  chemem.parameters.parameters['current_model'],
                                                  chemem.parameters.get_parameter('current_map'))
        chemem.run_js_code(cls.js_code(site))
            
       

class SaveSite(Command):
    
    @classmethod 
    def js_code(cls, site, alt = 0):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        if alt == 0: 
            js_code = f'addBindingSiteEntry("{site_value}", "{site.name}");'
        else:
            js_code = f'updateBindingSiteEntry("{site_value}", "{site.name}");'
        
        return js_code

    @classmethod 
    def js_code_error(cls):
        return 'alert("ChemEM Error: Rendered Site in None.");'
    
    @classmethod 
    def run(cls, chemem, query):
        
        
        site = chemem.rendered_site 
        if site is None:
            chemem.run_js_code(cls.js_code_error())
            return 
        
        #choose whether to update selected site! 
        in_list = chemem.parameters.get_list_parameters('binding_sites', site.binding_site.name)
        
        if in_list:
            chemem.run_js_code(cls.js_code(site.binding_site, alt=1))
        else:
            chemem.parameters.add_list_parameter('binding_sites', site.binding_site)
            chemem.run_js_code(cls.js_code(site.binding_site))



class RunSignificantFeaturesProtocol(Command):
    
    @classmethod
    def js_code_error(cls, map_id):
        return f'alert("Resolution needed for EM-map {map_id}.");'
    
    @classmethod 
    def js_code(cls, key, text):
        return f'addSigFeatListItem("{text}", {key});'
        
    
    @classmethod 
    def run(cls, chemem, query):
        if chemem.parameters.get_parameter('resolution') is not None:
            resolution = chemem.parameters.get_parameter('resolution')
            significant_features_object =  SignificantFeaturesProtocol.mask(chemem.session, chemem.rendered_site, resolution.value)
            chemem.current_significant_features_object = significant_features_object
            for key, string in significant_features_object.js_code:
                chemem.run_js_code(cls.js_code(key, string))
            
            chemem.rendered_site.map.set_display(False)
        else:
            chemem.run_js_code(cls.js_code_error())

class ShowSignificantFeature(Command):
    @classmethod 
    def run(cls, chemem, query):
        
        print(f'Show Sig feature {int(query)}')
        chemem.current_significant_features_object.show_feature(int(query))
        
class HideSignificantFeature(Command):
    @classmethod 
    def run(cls, chemem, query):
        
        chemem.current_significant_features_object.hide_feature(int(query))       
        
class SetEditBindingSiteValue(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.render_site_from_key(query)
                 


class AddSolutionToSimulation(Command):
    @classmethod
    def run(cls, chemem, query):
        solution = chemem.current_loaded_result.get_loaded_result_with_id(query)
        chemem.current_simulation.parameters['AddedSolution'] = solution
        js_code = f'setSolutionValue("{solution.string()}");'
        chemem.run_js_code(js_code)
        

class BuildSimulation(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem_executable = chemem.parameters.get_parameter('chememBackendPath')
        if chemem_executable is not None:
            if os.path.exists(f"{chemem_executable.value}.export_simulation"): 
                
                
                executable = f"{chemem_executable.value}.export_simulation"
                if 'AddedSolution' in chemem.current_simulation.parameters:
                    solution = chemem.current_simulation.parameters['AddedSolution']
                    ligand_files = extract_simulation_data_from_solution(solution)
                    
                    chemem.current_simulation.parameters['Ligands'] = ligand_files
                    new_model = build_model_without_ligands(solution.pdb_object)
                    chemem.current_simulation.parameters['current_model'] = new_model
                    
                #ADD A CHECK FOR IF THE LIGAND FILES ARE IN THE PDB
                
                
                chemem.temp_build_dir = tempfile.mkdtemp()
                #chemem.temp_build_dir = '/Users/aaron.sweeney/Documents/ChemEM_chimera_v2/debug/test_simulate/'
                chemem.current_simulation.add(PathParameter('output', chemem.temp_build_dir))
                chemem.make_conf_file(chemem.current_simulation)
                chemem.run(executable, EXPORT_SIMULATION)
                
                #create a tempory directory to write files to 
                #run the export_simulation blocking?
                
            else:
                chemem.run_js_code('alert("Simulation requires ChemEM v0.0.4");')
        else:
            chemem.run_js_code('alert("No ChemEM executable found");')


class GetAvaliblePlatforms(Command):
    
    @classmethod 
    def js_code(cls, platform_name):
        js_code = f'addPlatformToList("{platform_name}");'
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        if chemem.platforms_set:
            pass
        else:
            for platform_name in chemem.avalible_platforms:
                js_code = cls.js_code(platform_name)
                chemem.run_js_code(js_code)
            chemem.platforms_set = True

class SetPlatform(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.platform = query


class RunSimulation(Command):
    #TODO! -- with this set up the value of current simulation needs to be set to none 
    #when a new simulation is created
    @classmethod 
    def run(cls, chemem, query):
        
        if chemem.current_simualtion_id is None:
            chemem.run_simulation()
        else:
            
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            job._pause = False 

class PauseSimulation(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            job._pause = True
            
class StopSimulation(Command):
    @classmethod 
    def run(cls, chemem, query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            job.running = False


class SetSimulationTempreture(Command):
    @classmethod 
    def run(cls,chemem, query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            job._set_temp = query.value

class EnableTugMode(Command):
    @classmethod
    def run(cls,chemem, query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            chemem.default_mouse_mode = chemem.session.ui.mouse_modes.mode(button='right')
            new_mode = DragCoordinatesMode(chemem.session, {i[0] : i[1] for i in chemem.atoms_to_position_index})
            chemem.session.ui.mouse_modes.bind_mouse_mode(mouse_button="right", mode= new_mode)
            job.tug = new_mode
        
        

class DisableTugMode(Command):
    @classmethod
    def run(cls,chemem, query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            job.tug = None
            chemem.session.ui.mouse_modes.bind_mouse_mode(mouse_button="right", mode= chemem.default_mouse_mode)
            chemem.default_mouse_mode = None


class SetHBondTug(Command):
    @classmethod
    def js_code(cls, selected_atoms, restraint_id):
        # Function to escape JavaScript string literals
        def escape_js_string(s):
            return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')
    
        # Function to remove the part of the string before and including '.pdb '
        def trim_pdb_info(s):
            parts = s.split('.pdb ')
            return parts[1] if len(parts) > 1 else s  # returns the part after '.pdb ' if it exists
    
        # Construct the message with all special characters escaped
        donor = escape_js_string(trim_pdb_info(selected_atoms[0].string()))
        hydrogen = escape_js_string(trim_pdb_info(selected_atoms[1].string()))
        acceptor = escape_js_string(trim_pdb_info(selected_atoms[2].string()))
    
        message = f'Donor: {donor}\\nHydrogen: {hydrogen}\\nAcceptor: {acceptor}'
    
        # Create the JavaScript code string safely
        js_code = f'addRestraintToList("{message}", "{restraint_id}");'
        return js_code

    @classmethod 
    def run(cls,chemem,query):
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            simulation_model = chemem.simulation_model
            #selected_atoms retured as Donor, hydrogen, acceptor
            hbond_idx, selected_atoms = get_hbond_tug_index(simulation_model, chemem.atoms_to_position_index_as_dic)
            if hbond_idx is not None:
                if hbond_idx in chemem.hbond_idxs:
                    hbond_tug_idx = chemem.hbond_idxs.index(hbond_idx)
                    job.hbond_tug = [hbond_tug_idx, None, None]
                    restraint_id = len(chemem.current_restraints)
                    
                    js_code = cls.js_code(selected_atoms, restraint_id)
                    chemem.run_js_code(js_code)
                    selected_atoms[1].selected = False
                    #TODO add these properly and color code them
                    com = 'distance sel'
                    run(chemem.session, com)
                    chemem.current_restraints[restraint_id] = [hbond_tug_idx, selected_atoms]
                    
                    
class DeleteHbondTug(Command):
    @classmethod 
    def run(cls, chemem, query):
        
        if chemem.current_simualtion_id is not None:
            simulation_id = chemem.current_simualtion_id
            job = chemem.job_handeler.jobs[simulation_id]
            hbond_tug_idx , selected_atoms = chemem.current_restraints[int(query)]
            com = f'distance delete {selected_atoms[0].atomspec} {selected_atoms[2].atomspec}'
            run(chemem.session, com)
            job.hbond_tug = [hbond_tug_idx, 0.0, 0.0]
            del chemem.current_restraints[int(query)]
            
        #set the tug!
        #add a distance mointor 
        #send a thing to the template

def get_hbond_tug_index(model, atoms_to_index):
    selected_atoms = [atom for residue in model.residues for atom in residue.atoms if atom.selected]

    if len(selected_atoms) != 3:
        return None
    
    hydrogen = next((atom for atom in selected_atoms if atom.element.name == 'H'), None)
    if not hydrogen:
        return None  
    
    non_hydrogens = [atom for atom in selected_atoms if atom != hydrogen]
    if len(non_hydrogens) != 2:
        return None, None
    
    donor = next((atom for atom in hydrogen.neighbors if atom in non_hydrogens), None)
    if not donor:
        return None, None  

    acceptor = next((atom for atom in non_hydrogens if atom != donor), None)
    if not acceptor:
        return None, None  
    try:
        return [atoms_to_index[donor], atoms_to_index[hydrogen], atoms_to_index[acceptor]], [donor,hydrogen,acceptor]
    except KeyError:
        return None, None




#╔═════════════════════════════════════════════════════════════════════════════╗
#║                             Class wrappers                                  ║
#╚═════════════════════════════════════════════════════════════════════════════╝


class RenderBindingSite:
    def __init__(self, session, 
                 binding_site, 
                 current_model,
                 current_map = None):
        self.session = session
        self.binding_site = binding_site 
        self.model = current_model 
        self.map = current_map 
        
        self.render_site()
        
        if self.map is not None:
            self.render_map()
        
    def render_map(self):
        #may have to reset region herer!!!
        self.reset_map_region()
        origin, apix = self.map.data_origin_and_step()
        grid_size = self.map.data.matrix().shape
        
        min_indices, max_indices = find_voxel_indices(self.min_coords, self.max_coords, apix, grid_size, origin)
        self.chimera_map_slice_key = ([min_indices[0], min_indices[1], min_indices[2]], 
                                      [max_indices[0], max_indices[1], max_indices[2]], 
                                      [1, 1, 1])
        self.map.region = self.chimera_map_slice_key
        self.update_display()
        
    def reset_map_region(self):
        self.map.region = self.map.full_region()
        #call this to change back the box size in chimera
        self.update_display()
        
    def update_display(self):
        self.map.display = False 
        self.map.display = True
    
    def render_site(self):
        self.min_coords, self.max_coords = get_box_vertices(self.binding_site.centroid,
                                                  self.binding_site.box_size)
        
        res_inside = []
        atoms_inside = []
        for res in self.model.residues:
            for atom in res.atoms:
                if self.is_outside(atom.coord):
                    atom.display = False 
                    atom.residue.ribbon_display = False
                else:
                    atom.display = True 
                    atoms_inside.append(atom)
                    if res not in res_inside:
                        res_inside.append(res)
        
        self.inside_atoms = atoms_inside
        for res in res_inside:
            res.ribbon_display = True
        
                    
    def is_outside(self, atom_coord):
        if np.any(np.array(atom_coord) < self.min_coords):
            return True
        elif np.any(np.array(atom_coord) > self.max_coords):
            return True 
        else:
            return False
    
    def reset(self):
        #need to check if the model is within the session
        
        #check if the model has been closed first!
        if self.model in self.session.models:
            for res in self.model.residues:
                res.ribbon_display = True
                for atom in res.atoms:
                    atom.display = False
        #check if the map has been closed!
        if self.map is not None:
            if self.map in self.session.models:
                self.reset_map_region()
                
    def update_centroid(self, new_centroid):
        self.binding_site.centroid = new_centroid 
        self.binding_site.value = [new_centroid, self.binding_site.box_size]
        self.reset()
        self.render_site()
    
    def update_box_size(self, new_box_size):
        self.binding_site.box_size = new_box_size 
        self.binding_site.value = [self.binding_site.centroid, new_box_size]
        self.reset() 
        self.render_site()
        

class ChemEMMarkerMouseMode(MouseMode):
    def __init__(self,  chemem):
        super().__init__(chemem.session)
        self._marker_placement_active = True
        self.name = 'chemem_marker'
        self.original_mode = MoveMarkersPointMouseMode(self.session)
        #self.marker_wrapper = ChemEMMarker(self.session)
        self.chemem = chemem
        
        
    def enable(self):
        self._marker_placement_active = True
        self.session.logger.info("Marker placement mode activated. Right-click to place a marker.")

    def disable(self):
        self._marker_placement_active = False
        self.session.logger.info("Marker placement mode deactivated.")

    def mouse_down(self, event):
        if self._marker_placement_active:
            x, y = event.position()
            self.place_marker(x, y, event)
            self._marker_placement_active = False  # Deactivate after placing the marker
            self.session.ui.mouse_modes.bind_mouse_mode(mouse_button="right", mode=self.original_mode)
            
    
    def place_marker(self, x, y, event):
        # Convert screen coordinates to scene coordinates
        view = self.session.main_view
        xyz1, xyz2 = view.clip_plane_points(x, y)
        xyz = .5 * (xyz1 + xyz2)
        print('MRK1')
        self.chemem.set_marker_position(xyz)
        #self.session.models.add([self.marker_wrapper])
        #self.marker_wrapper.place_marker(xyz)
        #self.session.metadata = self.marker_wrapper
        #parameter = BindingSiteParameter.get_from_marker(self.marker_wrapper,self.chemem)
        #pass to chemem 
        #self.chemem.site_from_marker(parameter)
        

        
class ChemEMMarker(MarkerSet):
    def __init__(self, session):
        super().__init__(session)

    def place_marker(self, position):
        print(position)
        marker = self.create_marker(position, (136, 179, 198, 255), 1.0)  # Red color, size 1.0
        
        print(type(marker))
        #self.add_marker(marker)
        

def activate_marker_placement(chemem, parameter):
    #original_mode = chemem.session.ui.mouse_modes.mode(button="right")
    mode = ChemEMMarkerMouseMode(chemem)
    chemem.session.ui.mouse_modes.bind_mouse_mode(mouse_button="right", mode=mode)   

#╔═════════════════════════════════════════════════════════════════════════════╗
#║                             Helper functions                                ║
#╚═════════════════════════════════════════════════════════════════════════════╝


def build_model_without_ligands(model):
    new_model = model.copy() 
    for residue in new_model.residues:
        if residue.standard_aa_name is None:
            new_model.delete_residue(residue)
    
    return new_model
                
        


def extract_simulation_data_from_solution(solution):
    #PDB without ligand!!, ligand files!!
    pdb_file = solution.pdb_object.filename 
    file_matcher = solution.pdb_object.name.replace('.pdb', '')
    ligand_dir = os.path.dirname(pdb_file)
    
    if ligand_dir.endswith('post_processing'):
        pass
    elif ligand_dir.endswith('PDB'):
        ligand_dir = ligand_dir = os.path.dirname(ligand_dir)
        
    ligand_matches = [i for i in os.listdir(ligand_dir) if i.endswith('.sdf') and i.startswith(file_matcher)]
    if len(ligand_matches):
        return [PathParameter(num, os.path.join(ligand_dir,i)) for num,i in enumerate(ligand_matches)]
        
    

def set_conda_environment():
    conda_path_guess = [
        os.path.expanduser("~/anaconda3/bin"),
        os.path.expanduser("~/miniconda3/bin"),
        "C:\\ProgramData\\Anaconda3\\Scripts",
        "C:\\ProgramData\\Miniconda3\\Scripts",
        "C:\\Anaconda3\\Scripts",
        "C:\\Miniconda3\\Scripts"
    ]
    
    for path in conda_path_guess:
        if os.path.exists(path):
            os.environ["PATH"] += os.pathsep + path

def find_anaconda_path():
    # Ensure conda is in the PATH
    set_conda_environment()

    # Check for CONDA_PREFIX environment variable
    conda_prefix = os.getenv('CONDA_PREFIX')
    if conda_prefix and os.path.exists(conda_prefix):
        return conda_prefix

    # Check if 'conda' command is available in the PATH
    try:
        conda_path = subprocess.check_output(['conda', 'info', '--base'], stderr=subprocess.DEVNULL).decode().strip()
        if os.path.exists(conda_path):
            return conda_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None

def get_conda_env_paths(conda_base_path):
    envs_path = os.path.join(conda_base_path, 'envs')
    if os.path.exists(envs_path):
        
        return [os.path.join(envs_path, env) for env in os.listdir(envs_path)]
    return []

def check_chemem_version(env_path):
    bin_dir = os.path.join(env_path, 'bin')
    if not os.path.exists(bin_dir):
        bin_dir = os.path.join(env_path, 'Scripts')  # For Windows compatibility
        
    chemem_executable = os.path.join(bin_dir, 'chemem.chemem_path')
    if os.path.exists(chemem_executable):
        
        return True,  os.path.join(bin_dir, 'chemem')
        
    return False, None

def get_chemem_paths():
    conda_base_path = find_anaconda_path()
    if not conda_base_path:
        print("Anaconda installation not found.")
        return []

    env_paths = get_conda_env_paths(conda_base_path)
    
    all_paths = []
    for env_path in env_paths:
        found, chemem_executable = check_chemem_version(env_path)
        if found:
            all_paths.append(chemem_executable)
            
    return [PathParameter(i,i) for i in all_paths]

# Example usage
anaconda_path = find_anaconda_path()
if anaconda_path:
    print(f"Anaconda is installed at: {anaconda_path}")
else:
    print("Anaconda installation not found.")


def generate_unique_id():
    return str(uuid.uuid4())

def point_to_voxel_index(point, apix, origin):
    """
    Convert a 3D point in Angstroms to voxel indices.
    
    :param point: Tuple of the 3D point coordinates (x, y, z) in Angstroms.
    :param apix: Pixel size in Angstroms.
    :param origin: Origin coordinates (x, y, z).
    :return: Tuple of voxel indices (ix, iy, iz).
    """
    ix = int((point[0] - origin[0]) / apix[0])
    iy = int((point[1] - origin[1]) / apix[1])
    iz = int((point[2] - origin[2]) / apix[2])
    return ix, iy, iz

def find_voxel_indices(min_point, max_point, apix, grid_size, origin):
    """
    Determine the minimum and maximum voxel indices within given 3D points.
    
    :param min_point: Tuple of the minimum 3D point coordinates (x, y, z) in Angstroms.
    :param max_point: Tuple of the maximum 3D point coordinates (x, y, z) in Angstroms.
    :param apix: Pixel size in Angstroms.
    :param grid_size: Array of grid sizes [nx, ny, nz].
    :param origin: Origin coordinates (x, y, z).
    :return: Two tuples representing the minimum and maximum voxel indices ((min_ix, min_iy, min_iz), (max_ix, max_iy, max_iz)).
    """
    min_indices = point_to_voxel_index(min_point, apix, origin)
    max_indices = point_to_voxel_index(max_point, apix, origin)
    
    # Ensure indices are within grid size limits
    min_indices = (
        max(0, min_indices[0]), max(0, min_indices[1]), max(0, min_indices[2])
    )
    max_indices = (
        min(grid_size[0] - 1, max_indices[0]),
        min(grid_size[1] - 1, max_indices[1]),
        min(grid_size[2] - 1, max_indices[2])
    )
    
    return min_indices, max_indices

def get_box_vertices(centroid, box_size):
    """
    Calculate the vertices of a box given its center and dimensions.

    :param center: A tuple of (x, y, z) representing the center of the box.
    :param dimensions: A tuple of (length, width, height) of the box.
    :return: A list of tuples, each representing the coordinates of a vertex.
    """
    
    x, y, z = centroid
    xb, yb, zb = box_size

    # Half dimensions
    half_xb = xb / 2
    half_yb = yb / 2
    half_zb = zb / 2

    # Calculate the coordinates of the vertices
    vertices = [
        (x - half_xb, y - half_yb, z - half_zb),
        (x - half_xb, y - half_yb, z + half_zb),
        (x - half_xb, y + half_yb, z - half_zb),
        (x - half_xb, y + half_yb, z + half_zb),
        (x + half_xb, y - half_yb, z - half_zb),
        (x + half_xb, y - half_yb, z + half_zb),
        (x + half_xb, y + half_yb, z - half_zb),
        (x + half_xb, y + half_yb, z + half_zb)
    ]
    
    min_x_vertices = min([i[0] for i in vertices])
    max_x_vertices = max([i[0] for i in vertices])
    min_y_vertices = min([i[1] for i in vertices])
    max_y_vertices = max([i[1] for i in vertices])
    min_z_vertices = min([i[2] for i in vertices])
    max_z_vertices = max([i[2] for i in vertices])

    min_coords = np.array([min_x_vertices,min_y_vertices, min_z_vertices])
    max_coords = np.array([max_x_vertices,max_y_vertices,max_z_vertices])
    
    return min_coords, max_coords

def condense_path(path, max_length=25):
    if len(path) <= max_length:
        return path  

    part_length = (max_length - 3) // 2  # 3 characters are reserved for "..."
    start_part = path[:part_length]
    end_part = path[-part_length:]
    return f"{start_part}...{end_part}"
        
        
        