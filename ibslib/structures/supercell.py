# -*- coding: utf-8 -*-

import copy
import numpy as np

from ibslib import Structure
from ibslib.driver import BaseDriver_
from ibslib.structures.lebedev import lebedev_5
from ibslib.structures.utils import get_molecules
from ibslib.molecules.utils import com

from ibslib.io import write



class SupercellSphere(BaseDriver_):
    """
    Builds a spherical supercell using the provided distance as the radius of 
    the sphere. Accounts for atoms outside of the unit cell by computing a 
    correction to add to the radius radius. 
    
    Arguments
    ---------
    radius: float
        By default, will use this radius value. Will construct of sphere of 
        unit cells greater than or equal to this radius value. 
    mult: int
        If multiplicity is greater than zero, then the radius will be 
        calculated as the largest lattice vector multiplied by mult. 
    correction: bool
        If True, a correction will be automatically calculated for atoms
        outside the unit cell.
        If False, no correction will be used. 
    
    """
    def __init__(self, radius=10, mult=0, com=False, correction=True):
        self.radius = radius
        self.mult = mult
        self.correction=correction

    
    def calc_struct(self,struct):
        self.struct = struct
        
        ## Calculate radius based on mult
        if self.mult > 0:
            self.radius_by_mult()
        
        correction = self.get_correction(struct)
        if not self.correction:
            correction = 0
            
        self.supercell = self.construct_supercell(struct, correction)
        self.supercell.struct_id = "{}_Supercell_{}".format(
                self.struct.struct_id,self.radius)
        self.supercell.properties["Radius"] = self.radius
        self.supercell.properties["Mult"] = self.mult
        return self.supercell
    
    
    def write(self, output_dir, file_format="json", overwrite=False):
        temp_dict = {self.supercell.struct_id: self.supercell}
        write(output_dir, 
              temp_dict, 
              file_format=file_format, 
              overwrite=overwrite)
    
    
    def radius_by_mult(self):
        """
        Calculates radius as the largest lattice vector norm multiplied by the 
        multiplicity.
        
        """
        lat = np.vstack(self.struct.get_lattice_vectors())
        norms = np.linalg.norm(lat, axis=-1)
        max_norm = np.max(norms)
        self.radius = max_norm*self.mult
    
    
    def get_correction(self, struct=None):
        """
        Some atoms of moleceules may be outside the unit cell. This atom that
        is furthest outside the unitcell needs to be used as a correction to 
        the radius distance to guarantee proper construction of unit cell 
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
        desired radius radius. Then, the nearest lattice point to each of these
        points which is greater than the radius distance away is used. Once
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
        
        radius = self.radius + correction
        ## Modify coords for unit sphere for radius
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
        
        
def com_supercell(struct, radius=25, mult=0, ele="Cu", com_only=False,
                  bonds_kw={"mult":1.20, "skin":0.0, "update":False}):
    """
    Will construct the supercell of the COM of the molecules in the system. 
    
    Argumnets 
    ---------
    struct: Structure
        Structure object to build COM supercell with
    radius: float
        Used for supercell_sphere class
    mult: int
        Used for supercell_sphere class
    ele: str
        Name of element to use to visualize the COM
    com_only: bool
        If False, will overlay COM on top of molecules. 
        If True, will only have COM positions in the supercell.
        
    """
    supercell = copy.deepcopy(struct)
    molecules = get_molecules(struct, ret="struct",
                              bonds_kw=bonds_kw)
    com_list = [com(mol) for mol in molecules]
    
    if not com_only:
        for coords in com_list:
            supercell.append(coords[0],coords[1],coords[2],ele)
    else:
        com_list = np.vstack(com_list)
        com_ele = np.repeat(ele, com_list.shape[0])
        supercell.from_geo_array(com_list, com_ele)
        
    sphere = SupercellSphere(radius=radius, mult=mult, correction=False)
    supercell = sphere.calc_struct(supercell)
    supercell.struct_id = "{}_COM".format(supercell.struct_id)
    
    return supercell