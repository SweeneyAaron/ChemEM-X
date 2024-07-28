import os
import datetime
from chimerax.core.commands import run 

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