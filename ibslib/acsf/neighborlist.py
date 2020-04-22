# -*- coding: utf-8 -*-



"""
Compute environments of each atom in the molecule.

"""

import numpy as np

from scipy.spatial.distance import squareform

from ase.neighborlist import NeighborList, \
                             PrimitiveNeighborList
from ase.data import chemical_symbols

from ibslib import Structure,StructDict


def init_cutoffs(struct, radius=12):
    """
    Initializes cutoffs for structures. In this case, want to the cutoffs
    to be equal to the cutoff radius. 
    
    Arguments
    ---------
    struct: ibslib.Structure
        Uses the atoms present in struct to return list of cutoffs.

    """
    return [radius for x in struct.geometry]
    

class NeighborList(NeighborList):
    """
    Wrapper around ASE neighborlist class to implement for Structure obejcts.
    
    Arguments
    ---------
    cutoffs: list of float
        List of cutoff radii - one for each atom. If the spheres (defined by
        their cutoff radii) of two atoms overlap, they will be counted as
        neighbors.
    skin: float
        If no atom has moved more than the skin-distance since the
        last call to the :meth:`~ase.neighborlist.NewPrimitiveNeighborList.update()`
        method, then the neighbor list can be reused. This will save
        some expensive rebuilds of the list, but extra neighbors outside
        the cutoff will be returned.
    sorted: bool
        Sort neighbor list.
    self_interaction: bool
        Should an atom return itself as a neighbor?
    bothways: bool
        Return all neighbors.  Default is to return only "half" of
        the neighbors.
        
    """
    def __init__(self, cutoffs, skin=0.1, 
                 sorted=False, self_interaction=False,
                 bothways=True, primitive=PrimitiveNeighborList):
        
        super(NeighborList, self).__init__(cutoffs, skin=skin, 
                 sorted=sorted, self_interaction=self_interaction,
                 bothways=bothways, primitive=primitive)
        
    
    
    def calc(self, struct_obj):
        if type(struct_obj) == dict or \
           type(struct_obj) == StructDict:
            self.calc_dict(struct_obj)
        elif type(struct_obj) == Structure:
            self.calc_struct(struct_obj)
    
    
    def calc_dict(self, struct_dict):
        for struct_id,struct in struct_dict.items():
            self.calc_struct(struct)
    
    
    def calc_struct(self, struct, return_atom_struct=False,
                    central_atom=""):
        """
        Description of datastructure which is created:
        
        neighborlist: array
            This is an array but it's save as a list to be json serializable.
            This is the position of all neighbors for each atom in the 
            structure. The index of the list is the same as the atoms in the 
            geometry of the structure.
        
        neighborele: list of dict
            This is a list of dictionaries. Each dictionary has a key for each 
            unique element in the given system. NOTE, this may not include
            all unique elements for a systems in a structure dictionary.
            Each value for each key in the dictionary is a list of indices
            corresponding to the given element. This way, the neighborlist
            for an atom may be separated into interactions between the atom 
            and another specific type of element. This is necessary for the 
            atom symmetry function representations and thus is used there.
            (NOTE: This could also be used to do some interesting visualization
            where only a central atom and all of its neighbors of one type
            are displayed.)
        
        return_atom_struct: bool
            If an Structure for the atomic environment should be returned. 
            Otherwise, returns None.
        
        central_atom: str
            If return_atom_struct is True, then the central atom is the 
            provided element string. If the length of the string is 0, then 
            the default behavior is to use the atom's actual specie. This is 
            useful for visualization purposes to set the middle atom to Cl
            or another highly visable atom. 
        
        """
        # Store the neighborlist positions in Structures
        struct.properties["neighborlist"] = []
        struct.properties["neighborele"] = []
        unique_ele = np.unique(struct.geometry["element"])
        atoms = struct.get_ase_atoms()
        self.update(atoms)
        
        if return_atom_struct:
            atom_struct_dict = {}
    
        for i,atom in enumerate(atoms):
            position_list = []
            ele_dict = {}
            for x in unique_ele:
                ele_dict[x] = []
            
            if return_atom_struct:
                atom_struct = Structure()
                if len(central_atom) > 0:
                    identity = central_atom
                else:
                    identity = chemical_symbols[atom.number]
                identity = np.array([identity])
                atom_struct.from_geo_array(atom.position[None,:],identity)
            
            indices,offsets = self.get_neighbors(i)
            idx = 0
            for j, offset in zip(indices, offsets):
                
                # Calculate absolute position for atom
                position = atoms.positions[j] + np.dot(offset,
                                                       atoms.get_cell())
                position_list.append(position)
                
                # Add the current 
                ele_dict[chemical_symbols[atoms.numbers[j]]].append(idx)
                
                if return_atom_struct:
                    atom_struct.append(position[0],position[1],position[2],
                            chemical_symbols[atoms.numbers[j]])
                
                idx += 1
                    
                
            position_array = np.array(position_list)
            
            struct.properties["neighborlist"].append(position_array.tolist())
            struct.properties["neighborele"].append(ele_dict)
            
            if return_atom_struct:
                atom_struct_dict["{}_{}"
                        .format(struct.struct_id,i)] = atom_struct
            
        if return_atom_struct:
            return atom_struct_dict    
    

if __name__ == "__main__":
    from ibslib.io import read,write
    
    
    
    