from chimerax.core.toolshed import BundleAPI
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))


class _MyAPI(BundleAPI):

    api_version = 1
    
    
    
    '''

    @staticmethod
    def register_command(bi, ci, logger):
        from . import ChemEM
        
        func = ChemEM.proplid
        desc = ChemEM.proplid_desc
        
 
        from chimerax.core.commands import register
 
        register(ci.name, desc, func)
        
    '''
    @staticmethod
    def start_tool(session, bi, ti):
        from . import tool
        
        return tool.CHEMEM(session, ti.name)


bundle_api = _MyAPI()
