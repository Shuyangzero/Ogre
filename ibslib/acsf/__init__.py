# -*- coding: utf-8 -*-



from .acsf import *
from .combine import *
from .driver import *

def get_unique_ele(struct_dict):
    base = BaseACSF(struct_dict, 1)
    return base.unique_ele

def get_driver(struct_dict):
    """
    Function for easy creation of the standard RSF driver.
    
    """
    RSF_kw = \
    {
        "struct_dict": struct_dict,
        "cutoff": 12,
        "unique_ele": [], 
        "force": False,
        "n_D_inter": 12, 
        "init_scheme": "shifted",
        "eta_range": [0.05,0.5], 
        "Rs_range": [1,10],
        "del_neighbor": True
    }
    driver_kw = \
    {
        "file_format": "struct",
        "cutoff": 12,
        "prop_name": "",
    }
    rsf = RSF(**RSF_kw)
    driver = Driver(rsf, **driver_kw)
    return driver
    

__all__ = ["BaseACSF", 
           "RSF", 
           "Driver", 
           "combine", 
           "get_fast_kw_ele",
           "get_unique_ele",
           "get_driver"]