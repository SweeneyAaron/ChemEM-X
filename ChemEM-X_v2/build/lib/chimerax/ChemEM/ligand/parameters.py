#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 21:03:25 2025

@author: aaron.sweeney
"""

from chimerax.ChemEM.core.parameters import Parameter

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