#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:28:39 2025

@author: aaron.sweeney
"""

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