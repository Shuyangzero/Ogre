# -*- coding: utf-8 -*-

import numpy as np

from ase.data.vdw import vdw_radii
from ase.data import atomic_numbers

from ibslib import Structure

def construct_pair_keys(elements):
    """
    Constructs pair_key matrix for every element in the argument
    """
    # Incase a Structure object is passed instead of element array
    if type(elements) == Structure:
        elements = elements.geometry["element"]
    atom_keys = np.array(elements)
    pair_keys = np.char.add(atom_keys,np.array(["-"],dtype="<U2"))
    pair_keys = np.char.add(pair_keys,atom_keys[:,None])
    return pair_keys


def construct_h_bond_key(donor,acceptor):
    """
    Constructs h_bond key
    """
    return "{}H-{}".format(donor,acceptor)


intermolecular_dist = \
    {
        "vdw": {}, 
        
        # These are cutoff distances for specific 3 body hydrogen bonds
        "h_bond": \
            {
              "OH-O": 1.5,
              "OH-N": 1.6,
              "NH-O": 1.6,
              "NH-N": 1.75
            }
    }
        
# Construct vdw contact distances for pairwise_dist dictionary
pair_keys = construct_pair_keys([x for x in atomic_numbers.keys()])
pair_keys = pair_keys.ravel()

for pair in pair_keys:
    a1 = atomic_numbers[pair.split("-")[0]]
    a2 = atomic_numbers[pair.split("-")[1]]
    try:
        r1 = vdw_radii[a1]
        r2 = vdw_radii[a2]
    except:
        continue
    if np.isnan(r1) or np.isnan(r2):
        continue
    intermolecular_dist["vdw"][pair] = r1 + r2