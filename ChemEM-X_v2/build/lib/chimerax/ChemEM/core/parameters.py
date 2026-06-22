#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:40:53 2025

@author: aaron.sweeney
"""
import copy


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

#----------Complex params-----

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


class SimulatingAnnelingParameter(Parameter):
    """Simulated-annealing schedule sent from the GUI as a list like
    '[cycles, startTemp, normTemp, topTemp, tempStep, initialHeatingInterval,
    holdTopTempInterval, equilibriumTime, localMinimisation]'. Each value is
    unpacked onto a named attribute so SimulationJob can read them directly
    (job.simulated_anneling.startTemp, .normTemp, ...)."""

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.name_tags = ['simAnnCycles', 'startTemp', 'normTemp', 'topTemp',
                          'tempStep', 'initialHeatingInterval',
                          'holdTopTempInterval', 'equilibriumTime', 'localMinimisation']

    @classmethod
    def get_from_query(cls, query):
        if query['class'] == cls.__name__:
            c = cls(query['id'], cls.get_value(query['value']))
            for n, v in zip(c.name_tags, c.value):
                setattr(c, n, v)
            return c

    @classmethod
    def get_value(cls, value):
        value = value.replace('[', '').replace(']', '').split(',')
        return [int(i) for i in value]

#---class to hold ChemEM State

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
    
    def remove_list_parameter_by_value(self, list_name, value):
        
        if list_name in self.parameters:
            self.parameters[list_name] = [i for i in self.parameters[list_name] if  i.value != value]
        
    
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
        
        if parameter in self.parameters:
            return self.parameters[parameter].value
        else:
            return None
    #housekeeping functions!! 
    def _clear(self):
        self.parameters = {}
    
    def clear(self):
        retained = ['current_model', 'current_map', 'chememBackendPath']
        keys_to_delete = [key for key in self.parameters if key not in retained]
        for key in keys_to_delete:
            del self.parameters[key]
    
    def clear_binding_site_tabs(self, chemem):
        # Drop both the candidate and conf binding-site stores and clear their UI lists.
        self.parameters.pop('binding_sites', None)
        self.parameters.pop('binding_sites_conf', None)
        if getattr(chemem, 'rendered_site', None) is not None:
            try:
                chemem.rendered_site.reset()
            except Exception:
                pass
            chemem.rendered_site = None
        chemem.run_js_code('if (typeof clearBindingSiteLists === "function") clearBindingSiteLists();')

    def copy(self):
        return copy.copy(self)
    
    
    