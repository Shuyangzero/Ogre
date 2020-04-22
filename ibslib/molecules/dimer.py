# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as R

from ibslib import Structure
from ibslib.molecules import check_molecule,FindMolecules
from ibslib.molecules.utils import com,moit,orientation,show_axes,align


def check_dimer(struct, fm=FindMolecules(), exception=True):
    """
    Check if a structure is a dimer. 
    
    Arguments
    ---------
    struct: Structure
        Checks if struct is a dimer of identical molecules. 
    exception: bool 
        Controls if Excpetion will be raised if the structure is not a dimer. 
    
    """
    ## Check if has lattice vectors
    check_molecule(struct)
    
    ## Check if fm has already been calculated
    if len(fm.molecule_struct_list) == 0:
        fm.calc(struct)
        
    if len(fm.molecule_struct_list) == 2:
        if len(fm.unique_molecule_list) != 1:
            if not exception:
                return False
            
            raise Exception("Structure not a dimer. " +
                            "Structure {} has two molecules "
                            .format(struct.struct_id)
                            + "but they are not identical.")
        else:
            return True
        

def show_dimer_axes(struct, fm=FindMolecules(), ele="He"):
    if len(fm.molecule_struct_list) == 0:
        fm.calc(struct)
    check_dimer(struct, fm=fm)
    for molecule in fm.molecule_struct_list:
        com_pos = com(molecule) 
        axes = moit(molecule)
        struct.append(com_pos[0],com_pos[1],com_pos[2],ele)
        for row in axes:
            row += com_pos
            struct.append(row[0],row[1],row[2],ele)
            

def center_dimer(struct, fm=FindMolecules()):
    """
    Centers the system for one the molecules in the dimer such that the COM is 
    at the origin and the axes defined by the moment of inertia tensor are  
    oriented with the origin. This is done by translating and rotation the 
    entire dimer system.
    
    Arguments
    ---------
    struct: Structure
        Structure to adjust
    fm: FindMolecules
        FindMolecules object that has been calculated using Structure. This will
        save time. Otherwise, default will calculate the molecules in the 
        Structure. 
    
    """
    if len(fm.molecule_struct_list) == 0:
        fm.calc(struct)
        
    check_dimer(struct, fm=fm)
    
    trans = com(fm.molecule_struct_list[0])
    rot = orientation(fm.molecule_struct_list[0])
    
    geo = struct.get_geo_array()
    geo = geo - trans
    geo = np.dot(geo, rot.T)
    struct.from_geo_array(geo, struct.geometry["element"])
    
#    ## fm should also be modified
#    align(fm.unique_molecule_list[0])
#    for idx,molecule in enumerate(fm.molecule_struct_list):
#        geo = molecule.get_geo_array()
#        geo = geo - trans
#        geo = np.dot(geo, rot.T)
#        molecule.from_geo_array(geo, molecule.geometry["element"])    
#        fm.molecule_struct_list[idx] = molecule
        

def generate(molecule, trans, rot, align_molecule=True):
    """
    Generates a dimer system for the molecule by placing the molecule oriented 
    at the origin and then translating the COM by the trans vector and then 
    rotating the molecule. 
    
    Arguments
    ---------
    molecule: Structure
        Molecule to use in the construction of the dimer.
    trans: np.array
        Array of translation vector to apply the COM of the molecule. 
    rot: np.array
        Either a 3x3 rotation matrix or (3,1) or (3,) vector of euler angles. 
    align_molecule: bool
        If true, will automatically align molecule with the origin. 
    
    """
    check_molecule(molecule)
    
    ## Align molecule with the origin
    if align_molecule: 
        align(molecule)
    
    ## Now compute what to do with the rotation matrix
    if type(rot) != np.array:
        rot = np.array(rot)
    
    if rot.shape == (3,) or rot.shape == (3,1):
        r = R.from_euler('xyz', rot, degrees=True)
        rot = r.as_matrix()
        
    if rot.shape == (3,3):
        ## Check for valid rotation matrix
        det = np.linalg.det(rot)
        delta = np.abs(np.abs(det) - 1)
        if delta > 0.001:
            raise Exception("Determinant of rotation matrix is not equal to 1.")
    else:
        raise Exception("Rotation matrix {} ".format(rot)+
                    "Does not have a 3,3 shape.")
    
    ## Store Euler Angles for regeneration
    r = R.from_matrix(rot)
    euler = r.as_euler('xyz',degrees=True)
    
    ele = molecule.geometry["element"]
    geo = molecule.get_geo_array()
    geo = np.dot(geo, rot)
    geo = geo + trans
    
    dimer = Structure()
    dimer.struct_id = molecule.struct_id+"_dimer_{:.2f},{:.2f},{:.2f}_{:.2f},{:.2f},{:.2f}".format(trans[0],trans[1],trans[2],euler[0],euler[1], euler[2])
    dimer.from_geo_array(molecule.get_geo_array(), ele)
    
    for idx,row in enumerate(geo):
        dimer.append(row[0], row[1], row[2], ele[idx])
        
    return dimer
    
    
    
    
    
    