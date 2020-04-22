# -*- coding: utf-8 -*-


"""
Uniquely label the atoms of an arbitrary molecule.
"""

import numpy as np

from pymatgen import Molecule
from pymatgen.analysis.molecule_matcher import InchiMolAtomMapper
from ase.data import atomic_numbers

import openbabel as ob


class LabelAtoms():
    def __init__(self):
        self.ima = InchiMolAtomMapper()
    
    
    def calc(self, struct):
        pstruct = struct.get_pymatgen_structure()
        self._pymatgen_site_float(pstruct)
        return self.ima.get_molecule_hash(pstruct)
    
    
    def get_inchi(self, struct):
        mol = get_openbabel(struct)
        
        obconv = ob.OBConversion()
        obconv.SetOutFormat("inchi")
        print(obconv.WriteString(mol))
    
    
    def uniform_labels(self, molecule_struct_1, molecule_struct_2,
                       hold_first_order=False):
        """
        Uses Pymatgen and Openbabel to find a uniform labeling of atoms of 
        molecules one and two.
        
        Arguments
        ---------
        molecule_struct: Structure
            Structure object of a molecule. 
        hold_order: bool
            True: The atomic ordering of molecule_struct_1 will be held const. 
                  This requires slight reordering of the orderings fround from 
                  pymatgen. This is taken care of by _reorder_atoms.
                  If you're identifying the ordering of multiple molecules,
                  it's advisable that on the first ordering, this is False, 
                  and on all further orderings this is True. 
            False: Change order of molecule_struct_1 and molecule_struct_2
                   using the orderings identified by Pymatgen. 
        """
        pstruct_1 = molecule_struct_1.get_pymatgen_structure()
        pstruct_2 = molecule_struct_2.get_pymatgen_structure()
        self._pymatgen_site_float(pstruct_1)
        self._pymatgen_site_float(pstruct_2)
        clabel1,clabel2 = self.ima.uniform_labels(pstruct_1,pstruct_2)
        if clabel1 == None or clabel2 == None:
            raise Exception("No uniform label found by Pymatgen.")
        clabel1 = np.array(clabel1) - 1
        clabel2 = np.array(clabel2) - 1
        self._reorder_atoms(molecule_struct_1, molecule_struct_2,
                           clabel1, clabel2, hold_first_order=hold_first_order)
        return clabel1,clabel2
    
    
    def _pymatgen_site_float(self, pstruct):
        """
        Ensures that all sites are saved as type float. If this is not
        performed, the current versions of pymatgen/openbabel throw an error
        because the default type for pymatgen is a double and not a quad.
        """
        for site in pstruct.sites:
            site.coords = site.coords.astype(float)
        return
    
    
    def _reorder_atoms(self, molecule_struct_1, molecule_struct_2, 
                       clabel1, clabel2, hold_first_order=False):
        """
        Performs reordering of the molecules to match the uniform labeling
        provided by Pymatgen. However, there's an import difference. The 
        order of the orignal atoms in molecule_struct_1 can be kept constant.  
        """
        if hold_first_order == True:
            # If hold_order of 1, then idx into clabel arrays is rearranged 
            idx = np.argsort(clabel1)
        else:
            idx = np.arange(0,clabel1.shape[0])
        
        self._reorder_geo(molecule_struct_1, idx, clabel1)
        self._reorder_geo(molecule_struct_2, idx, clabel2)
        self._reorder_elements(molecule_struct_1, idx, clabel1)
        self._reorder_elements(molecule_struct_2, idx, clabel2)
    
    
    def _reorder_geo(self, molecule_struct, idx, label):
        """
        idx indexes into the label array
        """
        geo = molecule_struct.get_geo_array()
        geo = geo[label[idx],:]
        molecule_struct.geometry["x"] = geo[:,0]
        molecule_struct.geometry["y"] = geo[:,1]
        molecule_struct.geometry["z"] = geo[:,2]
    
    
    def _reorder_elements(self, molecule_struct, idx, label):
        """
        idx indexes into the label array
        """
        elements = molecule_struct.geometry["element"]
        elements = elements[label[idx]]
        molecule_struct.geometry["element"] = elements
        
        
        
def get_openbabel(molecule_struct):
    from openbabel import OBMol,OBAtom
    geo = molecule_struct.get_geo_array()
    geo = geo.astype(float)
    elements = molecule_struct.geometry["element"]
    
    # Build OBmol
    mol = OBMol()
    for i,ele in enumerate(elements):
        num = atomic_numbers[ele]
        a = mol.NewAtom()
        a.SetAtomicNum(num)
        a.SetVector(geo[i][0], geo[i][1], geo[i][2])
    
    return mol
        
    


if __name__ == "__main__":
    pass
#    from ibslib.io import read,write
#    from ibslib.molecules import UniqueMolecules
#    
#    struct_path = "/Users/ibier/Research/Results/Molecule_Volume_Estimation/PAHs/PAHs_crystal/ABECAL/ABECAL.cif"
#    struct_test = read(struct_path)
#    
#    um = UniqueMolecules(struct_test)
#    molecule_struct = um.unique_molecule_list[0]
#    
#    la = LabelAtoms()
#    
#    temp1,temp2 = la.uniform_labels(um.molecule_struct_list[0], 
#                                    um.molecule_struct_list[1])
    
    
    
    