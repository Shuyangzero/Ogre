# -*- coding: utf-8 -*-


import numpy as np

from ase.data import atomic_numbers,atomic_masses_iupac2016

from ibslib.driver import BaseDriver_


def density(struct, volume_key=""):
    """
    Passing in a structure returns the density of the structure. Density will
    be in units of g / cm^3.
    
    Arguments
    ---------
    struct: ibslib.Structure
        Structure object to calculate density
    volume_key: str
        Key to use for the volume of the Structure. If the default empty string
        is used, then the volume will be calculated.
    
    """
    if len(volume_key) == 0:
        volume = struct.get_unit_cell_volume()
    else:
        volume = struct.properties[volume_key]
       
    mass = np.sum([atomic_masses_iupac2016[atomic_numbers[x]]
                   for x in struct.geometry["element"]])
    
    ## Conversion factor for converting amu/angstrom^3 to g/cm^3
    ## Want to just apply factor to avoid any numerical errors to due float 
    factor = 1.66053907
    
    return (mass / volume)*factor


class Density(BaseDriver_):
    """
    Density calculator using API for ibslib Driver. 
    
    Arguments
    ---------
    masses = np.array
        Array of atomic masses indexed by atomic number
    numbers = dict
        Dictionary with atomic symbols as the key and atomic number as value.
    
    """
    def __init__(self, 
                 masses=atomic_masses_iupac2016, 
                 numbers=atomic_numbers,
                 volume_key=""):
        self.masses = masses
        self.numbers = numbers
        self.volume_key = volume_key
    
    
    def calc_struct(self, struct):
        struct_density = self.density(struct)
        struct.properties["density"] = struct_density
        
    
    def density(self, struct):
        if len(self.volume_key) == 0:
            volume = struct.get_unit_cell_volume()
        else:
            volume = struct.properties[self.volume_key]
           
        mass = np.sum([self.masses[self.numbers[x]]
                       for x in struct.geometry["element"]])
        factor = 1.66053907
        
        return (mass / volume)*factor
    
        
    

        

        