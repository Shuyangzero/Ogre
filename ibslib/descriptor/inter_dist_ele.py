# -*- coding: utf-8 -*-

import numpy as np

from ase.data.vdw import vdw_radii
from ase.data import atomic_numbers

from ibslib.motif.utils import construct_supercell_by_molecule


def inter_dist_ele(struct,supercell=3):
    """
    For computing all intermolecular distances in the specified supercell size. 
    Structure must have the num_molecules stored as a property and the atoms
    labaled uniformly for each molecule.
    """
    
    geo = struct.get_geo_array()
    nmpc = struct.get_property("num_molecules")
    napm = int(geo.shape[0] / nmpc)
    if geo.shape[0] % napm != 0:
        raise Exception("Number of molecules is not compatable with the "+
                "number of atoms of Structure {}.".format(struct.struct_id))
    supercell_struct = construct_supercell_by_molecule(struct,supercell=supercell)
    geo = supercell_struct.get_geo_array()
    
    # 85% of time of calculation is spent here
    dist = np.linalg.norm(geo - geo[:,None],axis=-1)
    
    nmpc = int(geo.shape[0]/napm)
    
    # From first molecule to second to last molecule
    inter_mask = []
    for i in range(1,nmpc):
        # Offset starting value as the napm multiplied by which molecule is 
        # being indexed currently
        temp_mask = np.arange(napm*i,napm*nmpc)
        # One row for each atom in the molecule
        rows = np.arange(0,napm)*dist.shape[1]
        # Broadcast over rows of atoms in the molecule and the intermolecular 
        # contacts of interest
        temp_mask = rows[:,None] + temp_mask
        flat_mask = temp_mask.ravel()
        inter_mask.append(flat_mask)
    
    # Make element array
    ele = supercell_struct.geometry["element"]
    ele_pairs = np.char.add(ele,"-")
    ele_pairs = np.char.add(ele,ele[:,None])
    
        
    inter_mask_array = np.hstack(inter_mask)
    result_dist = np.take(dist,inter_mask_array)
    result_ele = np.take(ele_pairs, inter_mask_array)
    
    return result_dist,result_ele


def inter_dist_ele_cutoff_matrix(struct,cutoff_matrix,supercell=3):
    """
    Computes intermolecular distances and masks the cutoff_matrix for all 
    intermolecular distances.
    """
    geo = struct.get_geo_array()
    nmpc = struct.get_property("num_molecules")
    napm = int(geo.shape[0] / nmpc)
    if geo.shape[0] % napm != 0:
        raise Exception("Number of molecules is not compatable with the "+
                "number of atoms of Structure {}.".format(struct.struct_id))
    supercell_struct = construct_supercell_by_molecule(struct,supercell=supercell)
    geo = supercell_struct.get_geo_array()
    
    # 85% of time of calculation is spent here
    dist = np.linalg.norm(geo - geo[:,None],axis=-1)
    
    nmpc = int(geo.shape[0]/napm)
    cutoff_matrix = np.tile(cutoff_matrix,(nmpc,nmpc))
    
    if dist.shape != cutoff_matrix.shape:
        raise Exception("Distance matrix and cutoff matrix are not"+
                        " the same shape.")
    
    # From first molecule to second to last molecule
    inter_mask = []
    for i in range(1,nmpc):
        # Offset starting value as the napm multiplied by which molecule is 
        # being indexed currently
        temp_mask = np.arange(napm*i,napm*nmpc)
        # One row for each atom in the molecule
        rows = np.arange(0,napm)*dist.shape[1]
        # Broadcast over rows of atoms in the molecule and the intermolecular 
        # contacts of interest
        temp_mask = rows[:,None] + temp_mask
        flat_mask = temp_mask.ravel()
        inter_mask.append(flat_mask)
    
        
    inter_mask_array = np.hstack(inter_mask)
    result_dist = np.take(dist,inter_mask_array)
    cutoff_dist = np.take(cutoff_matrix, inter_mask_array)
    
    return result_dist,cutoff_dist


class TrackDistances():
    """
    Class for tracking distances between elements found in a struct_dict
    """
    def __init__(self):
        self.dist_dict  = {}
        
    
    def calc(self,struct_dict=None):
        if struct_dict == None:
            raise Exception("No StructDict was passed to TrackDistances.calc")
        else:
            self.struct_dict = struct_dict
            
        for struct_id,struct in self.struct_dict.items():
            self.calc_struct(struct)
    
    
    def calc_struct(self,struct):
        result_dist,result_ele = inter_dist_ele(struct,supercell=2)
        unique_ele = np.unique(result_ele)
        keys = [x for x in self.dist_dict.keys()]
        for ele_pair in unique_ele:
            idx = np.where(result_ele == ele_pair)[0]
            dist = result_dist[idx]
            
            if ele_pair in keys:
                temp = self.dist_dict[ele_pair]
                temp = np.concatenate((temp, dist))
            else:
                self.dist_dict[ele_pair] = dist
    
    def hist(self,ele_pair,mult=1):
        vdW_sum = vdw_radii[atomic_numbers[ele_pair[0]]] + \
                  vdw_radii[atomic_numbers[ele_pair[1]]]
        vdW_cutoff = vdW_sum*mult
        dist = self.dist_dict[ele_pair]
        idx = np.where(dist < vdW_cutoff)[0]
        hist_dist = dist[idx]
        plt.hist(hist_dist)