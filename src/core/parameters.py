#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:40:53 2025

@author: aaron.sweeney
"""

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
                       value)
    @staticmethod 
    def get_value(value):
        value = None
        
        if value == 'false':
            value = 0
        if value == 'true':
            value = 1
        return value


class PathParameter(StringParameter):
    def __init__(self, name, default):
        super().__init__(name, default)

class SmilesParameter(StringParameter):
    def __init__(self, name, default):
        super().__init__(name, default)


