


"""
File for structure checks:
    - Duplicates
    - Physical Structure
    - Molecule in structure checks

"""

import numpy as np
import json
from ibslib import Structure,StructDict

from pymatgen.analysis.structure_matcher import (StructureMatcher,
                                                ElementComparator,
                                                SpeciesComparator,
                                                FrameworkComparator)

class pymatgen_compare():
    def __init__(self, pymatgen_kw=
                 {
                        "ltol": 0.2,                                    
                        "stol": 0.3,
                        "angle_tol": 5,
                        "primitive_cell": True,
                        "scale": True,
                        "attempt_supercell": False
                 }):
        self.kw = pymatgen_kw
    
    
    def __call__(self, struct1, struct2):
        sm =  StructureMatcher(
                    **self.kw,                                 
                    comparator=SpeciesComparator())                           
                                                                               
        pstruct1 = struct1.get_pymatgen_structure()                                      
        pstruct2 = struct2.get_pymatgen_structure() 
        
        return sm.fit(pstruct1, pstruct2)
    
    
    
class DuplicateCheck():
    """
    Checks if there are duplicate structures in an entire structure dictionary. 
    Should probably implement a couple different options for structure
    checking. Implement standard ibslib API. To this, the duplicate check
    probably needs to be initialized with the structure dictionary such 
    that duplicate calculations are never performed. Should probably also
    include a report option...
    
    Arguments
    ---------
    struct_dict: StructDict
        Dictionary containing all structures that should be checked for 
        duplicates.
    mode: str
        Mode of operation. Can be one of pair or complete. Pair will do the 
        minimal number of comparisons (n chose 2). Complete will always compare
        each structure to every other structure in the struct_dict.
    compare_fn: callable
        Callable that performs the comparison. This function can be arbitrary,
        but it must take in two structures as an argument and return True if
        the structures are duplicates, or False if they are not. See the 
        pymatgen_compare class for an example of what this might look like.
        
    """
    def __init__(self, 
                 struct_dict, 
                 mode="pair", 
                 compare_fn=pymatgen_compare()):
        
        self.struct_dict = struct_dict
        self.compare_fn = compare_fn
        self._check_mode(mode)
        self._comparisons()
        
        self.duplicates_dict = {}
        for key in self.struct_dict.keys():
            self.duplicates_dict[key] = []
        
    
    def _check_mode(self, mode):
        self.modes = ["pair", "complete"]
        if mode not in self.modes:
            raise Exception("DuplicateCheck mode {} is not available. "
                            .format(mode) +
                            "Please use one of {}.".format(self.modess))
        else:
            self.mode = mode
    
    
    def _comparisons(self):
        """
        Get the dictionary of comparisons that have to be made for pair mode
        esxecution.
        
        """
#        keys = np.array([x for x in self.struct_dict.keys()])
#        square = np.char.add(np.char.add(keys, "_"), keys[:,None])
        
        ### Just doing this to get the correct shape and index values for a
        ## pairwise comparison
        temp = np.arange(0,len(self.struct_dict),1)
        square = temp + temp[:,None]
        idx = np.triu_indices(n=square.shape[0],
                              k=1,
                              m=square.shape[1])
        
        #### Build the dictionary of indicies that each structure must be 
        ## compared to for pairwise comparison.
        keys = [x for x in self.struct_dict.keys()]
        comparison_dict = {}
        for key in keys:
            comparison_dict[key] = []
            
        for idx_pair in zip(idx[0],idx[1]):
            key = keys[idx_pair[0]]
            comparison_dict[key].append(idx_pair[1])
        
        self.comparisons = comparison_dict
        
    
    def calc(self, struct_obj):
        """
        General calc wrapper
        """
        if type(struct_obj) == StructDict or \
           type(struct_obj) == dict:
            self.calc_dict(struct_obj)
        elif type(struct_obj) == Structure:
            self.calc_struct(struct_obj)
            
    
    def calc_dict(self, struct_dict):
        for struct_id,struct in struct_dict.items():
            self.calc_struct(struct)
            
    
    def calc_struct(self, struct):
        eval("self.calc_struct_{}(struct)".format(self.mode))
        
        
    
    def calc_struct_pair(self, struct):
        """
        Pair mode implementation. 
        
        """
        keys = [x for x in self.struct_dict.keys()]
        if struct.struct_id not in keys:
            raise Exception("Structure ID {} was not found "
                    .format(struct.struct_id)+
                    "in the DuplicateCheck.struct_dict.")
        
        struct_dup_pool = [struct.struct_id]
        for idx in self.comparisons[struct.struct_id]:
            struct2 = self.struct_dict[keys[idx]]
            
            if self.compare(struct, struct2):
                struct_dup_pool.append(struct2.struct_id)
                print(struct.struct_id, struct2.struct_id)
            
        
        ## Now update the duplicates dict of all found duplicates with the 
        # same values
        for struct_id in struct_dup_pool:
            self.duplicates_dict[struct_id] += struct_dup_pool
            # Only use unique values
            self.duplicates_dict[struct_id] = \
                np.unique(self.duplicates_dict[struct_id]).tolist()
    
    
    def calc_struct_complete(self, struct):
        """
        Compare structure to all other structures in the structure dictionary.
        
        """
        raise Exception("Complete mode is not implemented yet.")
        
    
    def compare(self, struct1, struct2):      
        """
        Compare structures using pymatgen's StructureMatcher
        
        """                                                           
        return self.compare_fn(struct1, struct2)
    
    
    def write(self, file_name="duplicates.json"):
        """
        Output a format for the duplicates. Can outputs all the duplciates
        for each structure and a section for only the unique list of 
        duplicates found in the duplicates_dict.
        
        """
        ### First find unique duplicates in the duplicates_dict
        id_used = []
        unique = []
        for name,duplicates in self.duplicates_dict.items():
            if len(duplicates) == 1:
                continue
            elif name in id_used:
                continue
            else:
                unique.append(duplicates)
                [id_used.append(x) for x in duplicates]
        
        output_dict = {}
        output_dict["struct"] = self.duplicates_dict
        output_dict["dups"] = unique
        
        with open(file_name,"w") as f:
            f.write(json.dumps(output_dict, indent=4))
        




if __name__ == "__main__":     
    from ibslib.io import read,write
    test_dir = "/Users/ibier/Research/Results/Hab_Project/genarris-runs/GIYHUR/20191103_Full_Relaxation/GIYHUR_Relaxed_spg"
    
    s = read(test_dir)
    dc = DuplicateCheck(s)
    
    