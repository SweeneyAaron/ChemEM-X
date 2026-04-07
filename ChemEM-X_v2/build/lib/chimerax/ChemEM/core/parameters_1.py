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
