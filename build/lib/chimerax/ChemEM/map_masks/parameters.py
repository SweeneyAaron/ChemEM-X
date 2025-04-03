#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 22:50:15 2025

@author: aaron.sweeney
"""

from chimerax.ChemEM.core.parameters import Parameter 


class ConfidenceMapParameter(Parameter):
    def __init__(self, name, value):
        self.name = name 
        self.value = value 
    
    @classmethod 
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            name = query['id']
            value = cls.get_value(query['value'])
            return cls(name, value)

class MaskOptionsParameter(Parameter):
    def __init__(self, name, value):
        self.name = name 
        self.value = value 
    
    @classmethod 
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            value = cls.get_value(query['value'])
            c =  cls(query['id'],
                       value)
            c.use_confidence_map = value[0]

            return c
     
    @classmethod 
    def get_value(cls, value):
        values = value.split('|')
        value = [float(i) for i in values]
        return value
    


    
    