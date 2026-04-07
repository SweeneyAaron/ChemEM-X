#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:13:17 2025

@author: aaron.sweeney
"""

#bindingsite parameters
from chimerax.ChemEM.core.parameters import Parameter 
from chimerax.ChemEM.core.tools import generate_unique_id 

class BindingSiteParameter(Parameter):
    def __init__(self, name, value):
        self.name = name 
        self.value = value 
        self._type = 'Manual'
    
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
        