# -*- coding: utf-8 -*-

import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as SGA
from ibslib.io import import_structures
from ibslib.io import import_geo_dir


def pymatgen_sg_analysis(struct_dir, file_type='json', symprec=0.001,
                         return_struct_dict = False):
    '''
    Purpose:
        Takes a directory of structure files as an input and checks the 
          space group of each structure with the pymatgen analyzer.
    Arguments:
        file_type: Can be json or geometry. Used to change the type of 
                     file in the struct dir. 
        symprec: Precision used in space group analysis
    Returns:
        Pandas dataframe of the results and optionally the struct_dict of the 
          directory.
    '''
    
    if file_type == 'json':
        struct_dict = import_structures(struct_dir)
        results = pymatgen_sg_dict_match(struct_dict,symprec=symprec)
    elif file_type == 'geometry':
        struct_dict = import_geo_dir(struct_dir)
        results = pymatgen_sg_dict_determine(struct_dict,symprec=symprec)
    
    if return_struct_dict == True:
        return results, struct_dict
    else:
        return results

def pymatgen_sg_dict_match(struct_dict, symprec=0.001):
    '''
    Purpose:
        Takes a dictionary of structure objects and checks the space group 
          of each structure with the pymatgen analyzer and compares to the 
          space group stored in the json file.
    Returns:
        Pandas dataframe of the results.
    '''
    results = pd.DataFrame(columns=['struct_id','Match','Space Group', 
                                    'Pymatgen Space Group'])
    for row,struct_id in enumerate(struct_dict):
        match = False
        
        struct = struct_dict[struct_id]
        sg = struct.get_property('space_group')
        psg = pymatgen_sg_struct(struct,symprec=symprec)
        
        if sg == psg:
            match = True
        
        # This is by far the best way I found to append to a pandas dataframe
        #   row wise.
        results.loc[row] = [struct_id, match, sg, psg]
        
    return results

def pymatgen_sg_dict_determine(struct_dict, symprec=0.001):
    '''
    Purpose:
        Takes a dictionary of structure objects and determines the space group
        of each structure with the pymatgen analyzer. This space group is 
        added to the structure object. 
    Note:
        Doesn't need to return the struct_dict back out because the dictionary
          is pass by reference. 
    '''
    results = pd.DataFrame(columns=['struct_id','Pymatgen Space Group'])
    
    for row,struct_id in enumerate(struct_dict):
        struct = struct_dict[struct_id]
        psg = pymatgen_sg_struct(struct,symprec=symprec)
        struct.set_property('space_group', psg)
        results.loc[row] = [struct_id, psg]
    return results

def pymatgen_sg_struct(struct,symprec=0.001):
    '''
    Purpose:
        Takes a single structure and checks is space group with the pymatgen
          space group analyzer.
    '''
    structp = struct.get_pymatgen_structure()
    psg = SGA(structp,symprec=symprec).get_space_group_number()
    return psg