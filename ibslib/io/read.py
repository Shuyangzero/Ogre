# -*- coding: utf-8 -*-


import os

## torch is optional
try: 
    import torch
except:
    pass

import ase
from ase.io.formats import all_formats as ase_all_formats

### Pymatgen CifParser is not necessary but can work much better than ASE
try:
    from pymatgen.io.cif import CifParser
except:
    pass

from ibslib import Structure 
from ibslib.io.check import format2extension,check_format,check_file_type,check_ext

implemented_file_formats = ["geo", "geometry", "aims", "json", "cif", 
                            "ase"]

def read(struct_path, file_format='', recursive=False):
    """ 
    Read a single structure file or entire structure directory. The file format
    by default is automatically detected by using the file extension. 
    Alternatively, the user can specify a file format. 
    
    Arguments
    ---------
    struct_path: str
        Path to a structure file or a directory of structure files. 
    file_format: str
        File format to use. Can be any accepted ase file format. 
    recursive: bool
        Controls whether the entire folder tree will be explored to read in
        all structures files that are found. file_format will control which 
        file types are read in. For example, if file_format is json, then only
        json files will be read.
    
    Returns
    -------
    Structure or StructDict
        Returns a Structure object if the struct_path was pointed at a single 
        file.
        Returns a StructDict if the struct_path was pointed at a directory.
        
        
    """
    # If file format is used, check if it's an accepted value
    if len(file_format) > 0:
        check_format(file_format)
    
    if os.path.isdir(struct_path):
        if not recursive: 
            return read_dir(struct_path, file_format)
        else:
            return read_recursive(struct_path, file_format)
    elif os.path.isfile(struct_path):
        return read_file(struct_path, file_format)
    else:
        raise Exception("Input {} ".format(struct_path) +
                "was not recoginized as a file or a directory")


def read_dir(struct_dir, file_format=''):
    """
    Import any type of structure from directory to the structure class and 
    returns all structure objects as a dictionary, the best python data 
    structure due to hashing for O(1) lookup time.
    """
    file_list = os.listdir(struct_dir)
    struct_dict = {}
    for file_name in file_list:
        file_path = os.path.join(struct_dir,file_name)
        struct = read_file(file_path, file_format)
        if struct == None:
            continue
        struct_dict[struct.struct_id] = struct
            
    return struct_dict


def read_recursive(struct_dir, file_format=""):
    """
    Looks through entire file structure below struct_dir to read in all 
    Structure files with the desired file format. 
    
    Arguments
    ---------
    struct_dir: str
        Path to the folder to search 
    file_format: str
        If a file_format is not specified, it will read in all files 
        identified to be structure files.
    
    """
    struct_dict = {}
    
    if len(file_format) > 0:
        desired_ext = format2extension[file_format]
    else:
        desired_ext = ""
    
    
    for root, dirnames, filenames in os.walk(struct_dir):
        ## No files in directory
        if len(filenames) == 0:
            continue
        for file_name in filenames:
            file_path = os.path.join(root, file_name)
            
            ## Check if the file should be read
            ext = check_ext(file_name)
            if len(desired_ext) > 0:
                if ext != desired_ext:
                    continue
                else:
                    struct = read_file(file_path, file_format)
                    if struct == None:
                        continue
                    struct_dict[struct.struct_id] = struct
            ## Try to read in all files
            else:
                temp_file_format = check_file_type(file_name)
                struct = read_file(file_path, temp_file_format)
                if struct == None:
                    continue
                struct_dict[struct.struct_id] = struct
    return struct_dict
            


def read_file(file_path, file_format=''):
    """
    Imports a single file to a structure object.
    """
    if len(file_format) > 0:
        if file_format in ["geometry","geo", "aims"]:
            struct = import_geo(file_path)
        elif file_format == "json":
            struct = import_json(file_path)
        elif file_format == "cif":
            struct = import_cif(file_path)
        elif file_format == "ase":
            struct = import_ase(file_path)
        elif file_format == "torch":
            struct = import_torch(file_path)
            
    elif '.json' == file_path[-5:]:
            struct = import_json(file_path)
    elif '.cif' == file_path[-4:]:
        struct = import_cif(file_path)
    elif '.in' == file_path[-3:] or file_path.endswith('.next_step'):
        struct = import_geo(file_path)
    elif '.pt' == file_path[-3:]:
        struct = import_torch(file_path)
    else:
        try: struct = import_ase(file_path)
        except: 
            print("Could not load file {}. Check file_format argument."
                  .format(file_path))
            return None
        
    return struct


def import_json(file_path):
    struct = Structure()
    struct.build_struct_from_json_path(file_path)
    if struct.struct_id == None:
        file_name = os.path.basename(file_path)
        struct.struct_id = file_name.replace('.json','')
    return struct


def import_geo(file_path, struct_id=''):
    """
    Import single geometry file. 
    
    Arguments
    ---------
    file_path: str
        Path to the geometry file to be loaded
    struct_id: str
        Option to specify a struct_id. The default is an empty string which 
        specifies the behavior to use the file name as the struct_id. 
    """
    struct = Structure()
    struct.build_geo_from_atom_file(file_path)
    if len(struct_id) == 0:
        file_name = os.path.basename(file_path)
        struct.struct_id = file_name.replace('.in','')
    else:
        struct.struct_id = struct_id
    return struct


def import_cif(file_path,occupancy_tolerance=100):
    """
    Import cif with pymatgen. This is the current default.
    
    Aruments
    --------
    occupancy_tolerance: int
        Occupancy tolerance for pymatgen.io.cif.CifParser. Files from the 
        CSD can contain multiple atoms which sit on the same site when 
        PBC are taken into account. Pymatgen correctly recognizes these as the
        same atom, however, it has a tolerance for the number of atoms which it
        will assume are the same. This is the occupancy tolerance. I don't 
        believe this value should be limited and thus it's set at 100.
    """
    try: pycif = CifParser(file_path, occupancy_tolerance=occupancy_tolerance)
    except: return(import_cif_ase(file_path))
    pstruct_list = pycif.get_structures()
    struct = Structure.from_pymatgen(pstruct_list[0])
    file_name = os.path.basename(file_path)
    struct.struct_id = file_name.replace('.cif','')
    return struct


def import_cif_ase(file_path):
    """
    Import a single cif file using ase.io
    """
    atoms = ase.io.read(file_path,format='cif')
    struct = Structure.from_ase(atoms)
    file_name = os.path.basename(file_path)
    struct.struct_id = file_name.replace('.cif','')
    return struct


def import_ase(file_path):
    """
    Import general file using ase.io
    """
    atoms = ase.io.read(file_path)
    struct = Structure.from_ase(atoms)
    file_name = os.path.basename(file_path)
    # Struct ID is filename before decimal 
    struct.struct_id = file_name.split('.')[0]
    return struct


def import_torch(file_path):
    """
    Import Structure.json file saved as a Pytorch file.
    
    """
    struct_ = torch.load(file_path)
    struct = Structure()
    struct.from_dict(struct_)
    return struct



