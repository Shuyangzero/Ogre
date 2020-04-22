# -*- coding: utf-8 -*-

import numpy as np

def get_chemical_formula(struct):
    """
    Return chemical formula as a dictionary sorted in alphabetial 
    order.
    """
    ele = struct.geometry["element"]
    e,c = np.unique(ele, return_counts=True)
    
    # Alphabetical Order
    sort_idx = np.argsort(e)
    e = e[sort_idx]
    c = c[sort_idx]
    
    formula_dict = {}
    for i,element in enumerate(e):
        formula_dict[element] = c[i]
    
    return formula_dict


def D_to_H(struct):
    """
    Modifies structure obejct so that all Deuterium atoms 
    are changed to Hydrogens.
    """
    ele = struct.geometry["element"]
    d_idx = np.where(ele == "D")[0]
    ele[d_idx] = "H"
    struct.geometry["element"] = ele
