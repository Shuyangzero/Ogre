# -*- coding: utf-8 -*-

import numpy as np


def no_k_grid(struct):
    return []

def const_333(struct):
    return [3,3,3]


def k_grid_24(struct):
    """
    Takes in the structure and returns the appropriate k_grid. 
    
    """
    lattice_vectors = struct.get_lattice_vectors()
    norms = np.linalg.norm(lattice_vectors, axis=1)
    norms = norms.reshape(-1)
    k_grid = 24 / norms
    k_grid = [int(np.ceil(x)) for x in k_grid]
    return k_grid
    
    
    
def k_grid_40(struct):
    """
    Takes in the structure and returns the appropriate k_grid. 
    
    """
    lattice_vectors = struct.get_lattice_vectors()
    norms = np.linalg.norm(lattice_vectors, axis=1)
    norms = norms.reshape(-1)
    k_grid = 40 / norms
    k_grid = [int(np.ceil(x)) for x in k_grid]
    return k_grid
