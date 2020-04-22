# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist

from ase.data import atomic_numbers,vdw_radii

from ibslib.driver import BaseDriver_
from ibslib import Structure
from ibslib.structures.utils import get_molecules
from ibslib.descriptor.lebedev import lebedev_5



class InterMolecularDistance(BaseDriver_):
    """
    Calculates all pair-wise atomic intermolecular distances up to a cutoff 
    radius.
    
    Arguments
    ---------
    cutoff: float
        Intermolecular distances outside ther cutoff radius will not be stored. 
    bonds_kw: 
        Arguments fed into Structure.get_bonds. 
    sr: bool
        If False, returns absolute distances. 
        If True, returns the distance divided by the sum of the two vdW radii. 
    
    """
    def __init__(self, cutoff=6, 
                 bonds_kw={"mult": 1.20, "skin": 0.0, "update": False},
                 sr=True):
        
        self.cutoff = cutoff
        self.bonds_kw = bonds_kw
        self.sr = sr
    
    
    def calc_struct(self, struct):
        """
        Algorithm is as follows: 
            1. For each molecule in the system, translate the system so the 
            molecule sits at the origin. 
            2. Then, a supercell will be calculated 
            with all possible lattice points within the cutoff radius plus the 
            maximum length of the molecule. This will guarentee the correct 
            sphere around the molecule. 
            3. Then, the pairwise distances for each 
            atom in the molecule will be calculated with every atom in the 
            supercell. Distances that are greater than the cutoff will be 
            discarded. 
            4. Store results and repeat for every molecule in the system. 
        
        """
        self.struct = struct
        molecule_idx_list = get_molecules(struct, self.bonds_kw)
        
        correction = self.get_correction()
        supercell = self.construct_supercell(self.struct, 
                                             correction)
        supercell_geo = supercell.get_geo_array()
        supercell_ele = supercell.geometry["element"]
        
        dist_dict = {}
        unique_ele = np.unique(struct.geometry["element"])
        unique_list = []
        reverse_unique_list = []
        vdw_sum = []
        
        ### Get all unqiue types of element pairs
        for ele1 in unique_ele:
            for ele2 in unique_ele:
                key = "".join([ele1,ele2])
                dist_dict[key] = []
                
                reverse_key = "".join([ele2,ele1])
                if key in unique_list:
                    continue
                if reverse_key in unique_list:
                    continue
                else:
                    vdw_sum.append(vdw_radii[atomic_numbers[ele1]]+
                                   vdw_radii[atomic_numbers[ele2]])
                    unique_list.append(key)
                    reverse_unique_list.append(reverse_key)
        
        ## Can safely assume the first entry in the supercell is the unit 
        ## cell at 0,0,0 due to the way that self.prep_idx has been defined
        for molecule_idx in molecule_idx_list:
            mol_pos = supercell_geo[molecule_idx]
            mol_ele = supercell_ele[molecule_idx]
            
            ## Get all other positions
            mask = np.ones(supercell_geo.shape[0], np.bool)
            mask[molecule_idx] = 0
            other_pos = supercell_geo[mask]
            other_ele = supercell_ele[mask]
            
            dist = cdist(mol_pos, other_pos)
            dist = dist.ravel()
            ele_pair = np.char.add(mol_ele[:,None], other_ele)
            ele_pair = ele_pair.ravel()
            
            ele_sort_idx = np.argsort(ele_pair)
            ele_pair = ele_pair[ele_sort_idx]
            dist = dist[ele_sort_idx]
            ele_key,count = np.unique(ele_pair, return_counts=True)
            
            dist_idx = 0
            for temp_idx,key in enumerate(ele_key):
                temp_count = count[temp_idx]
                temp_start = dist_idx
                temp_end = dist_idx + temp_count
                dist_dict[key].append(dist[temp_start:temp_end])
                ### Move the start to the end of the previous
                dist_idx = temp_end
                
        for idx,key in enumerate(unique_list):
            if key != reverse_unique_list[idx]:
                dist_dict[key] = np.hstack(dist_dict[key])
                temp_stacked = np.hstack(dist_dict[reverse_unique_list[idx]])
                dist_dict[key] = np.hstack([dist_dict[key],
                                           temp_stacked])
                del(dist_dict[reverse_unique_list[idx]])
            else:
                dist_dict[key] = np.hstack(dist_dict[key])
            dist_dict[key].sort()
        
        if self.sr:
            idx = 0
            for key,value in dist_dict.items():
                dist_dict[key] = value / vdw_sum[idx]
                idx += 1
        
        return dist_dict
            
            
        
    def get_correction(self, struct=None):
        """
        Some atoms of moleceules may be outside the unit cell. This atom that
        is furthest outside the unitcell needs to be used as a correction to 
        the cutoff distance to guarantee proper construction of unit cell 
        sphere. 
        
        """
        if struct == None:
            struct = self.struct
        
        geo = struct.get_geo_array()
        lv = np.vstack(struct.get_lattice_vectors())
        lv_inv = np.linalg.inv(lv.T)
        
        ## Get all fractional coordinates
        frac = np.dot(lv_inv, geo.T).T
        
        ## Find largest magnitude
        frac = np.abs(frac)
        max_frac = np.max(frac)
        
        if max_frac < 1:
            return 0
        
        ## Compute cartesian correction to radius
        idx = np.argmax(frac)
        vec_idx = idx % 3
        correction = np.linalg.norm(lv[vec_idx,:])*(max_frac - 1)
        
        return correction
        
        
    def construct_supercell(self, struct=None, correction=0):
        """
        Uses points on Lebedev grid to construct points on the surface of the 
        desired cutoff radius. Then, the nearest lattice point to each of these
        points which is greater than the cutoff distance away is used. Once
        we know the points furtherst away for the supercell, all intermediary 
        lattice points may be added easily. Translation vectors for the 
        molecules in the unit cell may be applied to fill out the supercell 
        sphere. 
        
        """
        if struct == None:
            struct = self.struct
        
        geo = struct.get_geo_array()
        ele = struct.geometry["element"]
            
        ## Get 50 coordinates on the Levedev grid by using n=5
        ## This is a reasonable number because the lattice sites will be very 
        ## corse grained compared to the grid
        coords = np.vstack(lebedev_5["coords"])
        
        radius = self.cutoff + correction
        ## Modify coords for unit sphere for cutoff radius
        coords *= radius
        
        ## Convert coords to fractional representation
        lv = np.vstack(struct.get_lattice_vectors())
        lv_inv = np.linalg.inv(lv.T)
        frac_coords = np.dot(lv_inv, coords.T).T
        
        
        ## Round all fractional coordinates up to the nearest integer 
        ## While taking care of negative values to round down
        sign = np.sign(frac_coords)
        frac_coords = np.ceil(np.abs(frac_coords))
        frac_coords *= sign
        
        unique = np.unique(frac_coords, axis=0)
        unique_cart = np.dot(unique, lv)
        
        ## Need to fill in grid
        max_norm = np.max(np.linalg.norm(unique_cart, axis=-1))
        max_idx = np.max(np.abs(frac_coords))
        all_idx = self.prep_idx(max_idx, -max_idx)
        all_idx_cart = np.dot(all_idx, lv)
        all_norm = np.linalg.norm(all_idx_cart, axis=-1)
        take_idx = np.where(all_norm <= max_norm)[0]
        
        all_cart_points = all_idx_cart[take_idx]
        
        ## For visualization
        self.lattice_points = all_cart_points
        
        ## Constructure supercell structure
        supercell_geo = np.zeros((geo.shape[0]*all_cart_points.shape[0],3))
        supercell_ele = np.repeat(ele[:,None], all_cart_points.shape[0],axis=-1)
        supercell_ele = supercell_ele.T.reshape(-1)
        for idx,trans in enumerate(all_cart_points):
            start = idx*geo.shape[0]
            end = (idx+1)*geo.shape[0]
            supercell_geo[start:end,:] = geo+trans
        
        supercell_struct = Structure.from_geo(supercell_geo, supercell_ele)
            
        return supercell_struct
        
    
    def prep_idx(self, max_idx, min_idx):
        """
        Return a list of all index permutations between a max index and 
        minimum index. 
        
        """
        if int(max_idx) != max_idx or \
           int(min_idx) != min_idx:
               raise Exception("Max index {} and min index {} "
                    .format(max_idx,min_idx)+
                    "cannot safely be converted to integers.")
        
        idx_range = np.arange(min_idx, max_idx+1)[::-1]
        
        ## Sort idx_range array so the final list is sorted by magnitude 
        ## so that lower index, and positive index, planes are given preference
        sort_idx = np.argsort(np.abs(idx_range))
        idx_range = idx_range[sort_idx]
        return np.array(
                np.meshgrid(idx_range,idx_range,idx_range)).T.reshape(-1,3)
        
    
    def visualize(self, grid=None, element="He"):
        if grid == None:
            grid = self.lattice_points
        
        ele = np.repeat(element, grid.shape[0])
        
        return Structure.from_geo(grid, ele)
        
        
        

class IMD(InterMolecularDistance):
    """
    Wrapper for shorter API
    """

        
if __name__ == "__main__":
    pass
