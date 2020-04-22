
import os,copy
import numpy as np

from ibslib import Structure,StructDict
from ibslib.io import read,write
from ibslib.driver import BaseDriver_
from ibslib.descriptor.bond_neighborhood import BondNeighborhood
from ibslib.descriptor.R_descriptor import calc_R,ele_R,struct_R
from ibslib.motif.utils import get_molecules,reconstruct_with_whole_molecules, \
                               construct_smallest_molecule
from ibslib.molecules.utils import align
try: 
    from ibslib.molecules.label_molecule_atoms import LabelAtoms
except:
    pass


class FindMolecules(BaseDriver_):
    """
    Find molecular units in periodic and non-periodic Structures. 
    
    Arguments
    ---------
    folder: str
        Path to the parent folder for all molecules.
    mult: float
        Multiplicative factor to multiply ase.neighborlist.natural_cutoff.
    output_rstruct: bool
        Decides if the reconstructed structures should also be output.
    rstruct_folder: path
        Path for reconstructed structures if they are to be output. 
    residues: int
        Number of unqiue molecules the user would like to find. If the value is 
        zero, then default settings are used. However, if ean integer value is 
        supplied, then the mult parameter will be varied to try to achieve an 
        identical number of unique residues in the system. 
    conformation: bool
        If True, will account for conformational degrees of freedom when
        identifying unique molecules. If False, then molecules will be 
        duplicates if they have the exact same covalent bonds present. 
    mult_range: iterable
        Values for mult parameter to search over if residues is fixed nubmer.
    
    """
    def __init__(self, 
                 folder="Molecules", 
                 rstruct_folder="Structures",
                 mult=1.05, 
                 conformation=True, 
                 residues=0,
                 output_rstruct=False,
                 file_format="json",
                 overwrite=False,
                 mult_range=np.arange(1.05, 1.25, 0.005),
                 verbose=True):
        """
        Takes in all settings for the FindMolecules class. 
        
        """
        self.folder = folder
        self.mult = mult
        self.mult_range = mult_range
        self.residues=residues
        self.conformation = conformation
        if type(residues) != int or residues < 0:
            raise Exception("Residues argument can only be a positive integer.")
            
        self.output_rstruct = output_rstruct
        self.rstruct_folder = rstruct_folder
        self.overwrite = overwrite
        self.file_format = file_format
        self.verbose=verbose
        self.molecules = []
        self.unique = []
        
    
    def calc(self, struct_obj, folder=""):
        if len(folder) > 0:
            self.folder = folder
        obj_type = type(struct_obj)
        if obj_type == dict or obj_type == StructDict:
            return self.calc_dict(struct_obj)
        elif obj_type == Structure:
            return self.calc_struct(struct_obj)
    
    
    def calc_dict(self, struct_dict):
        for struct_id,struct in struct_dict.items():
            self.calc_struct(struct)
            self.write(file_format=self.file_format,overwrite=self.overwrite)
    
    
    def calc_struct(self, struct):
        ## Reset internal storage
        self.molecules = []
        self.unique = []

        # Reconstruct with whole molecules because it will make all further 
        # calculations faster
        self.struct = struct
        
        if self.residues == 0:
            self._calc_struct(struct, self.mult)
        else:
            ## Otherwise search over mult range
            for mult in self.mult_range:
                if self.verbose:
                    print("{}: Trying mult={:.3f}".format(self.struct.struct_id,
                                                      mult))
                self._calc_struct(struct, mult)
                ## Now check if obtained the correct number of unique molecules
                if len(self.unique) == self.residues:
                    if self.verbose:
                        print("{}: Success finding {} unique residues"
                              .format(self.struct.struct_id, self.residues))
                    break
                else:
                    ## If not correct, reset calculation. 
                    ## If using none of the mult values work, then this will 
                    ## leave the len of unique_molecule_list as zero indicating
                    ## failure. 
                    self.molecules = []
                    self.unique = []
                
                
        if self.output_rstruct:
            self.rstruct = self._reconstruct_structure()
    
    
    def _calc_struct(self, struct, mult=1.05):
        """
        Wrapper for performing calculation using specific mult parameter.
        
        """           
        
        ## Wrap in try/except because it's may not work every time
        try: 
            self.molecules = get_molecules(self.struct, 
                                              mult=mult)
        except:
            if self.verbose: 
                print("FindMolecules failed for {} "
                      .format(self.struct.struct_id)+ 
                      "with mult parameter {:.3f}".format(mult))
        
        for idx,molecule in enumerate(self.molecules):
            molecule.struct_id = "{}_molecule_{}".format(struct.struct_id,idx)
        
        if self.conformation: 
            self.unique = find_unique_molecules(self.molecules)
        else:
            self.unique = find_unique_molecules_no_conformation(self.molecules)
        
        for idx,molecule in enumerate(self.unique):
            molecule.struct_id = "{}_unique_molecule_{}".format(
                                       struct.struct_id,idx)

    
    def write(self, folder="", file_format="json", overwrite=False):
        if len(folder) > 0:
            self.folder = folder
        
        self.write_unique(self.folder, file_format=file_format, 
                          overwrite=overwrite)
        
        if self.output_rstruct:
            self.write_rstruct(self.rstruct_folder, file_format=file_format,
                               overwrite=overwrite)
        
    
    def write_unique(self,folder,file_format="json",overwrite=False):
        """
        Writes unique single molecules found for the structure to a single 
        folder.
        """
        
        if len(self.struct.get_lattice_vectors()) > 0:
            volume = self.struct.get_unit_cell_volume() 
            molecule_volume = volume / len(self.molecules)
        else:
            volume = 0
            molecule_volume = 0
        
        if len(self.unique) == 1:
            molecule_struct = self.unique[0]
            if molecule_volume > 0:
                molecule_struct.set_property("molecule_volume", 
                                              molecule_volume)
            temp_dict = {}
            temp_dict[molecule_struct.struct_id] = molecule_struct
            write(folder, temp_dict, file_format=file_format,
                  overwrite=overwrite)
            return
        
        for i,molecule_struct in enumerate(self.unique):
            if len(self.unique) == 1:
                # Add molecular volume if there's only one type of molecule
                if molecule_volume > 0:
                    molecule_struct.set_property("molecule_volume", 
                                                 molecule_volume)
            temp_dict = {}
            temp_dict[molecule_struct.struct_id] = molecule_struct
            write(folder, temp_dict, file_format=file_format,
                  overwrite=overwrite)
           
    
    def write_rstruct(self,folder,file_format="json",overwrite=False):
        """
        Writes the reconstructed geometry from the smallest molecule
        representation.
        """
        struct_id = self.struct.struct_id
        self.rstruct.set_property("num_molecules", 
                                  len(self.molecules))
        self.rstruct.set_property("num_unique_molecules", 
                                  len(self.unique))
        path = os.path.join(folder,struct_id)
        write(path,self.rstruct,file_format=file_format,
              overwrite=overwrite)
        
    
    def _reconstruct_structure(self):
        """
        Reconstructs the system using the smallest molecule representation.
        If there's only one unique molecule, then reorders the molecules to 
        all have the same atom indexing.

        """
        rstruct = Structure()
        rstruct.set_lattice_vectors(self.struct.get_lattice_vectors())
        
        ## Prepare properties to store
        num_atoms = 0
        for molecule_struct in self.molecules:
            num_atoms += molecule_struct.geometry.shape[0]
        rstruct.properties["bonds"] = [[] for x in range(num_atoms)]
        rstruct.properties["molecule_idx"] = [[] for x in
                                               range(len(self.molecules))]
        
        if len(self.unique) == 1:
            try: 
                self._reorder_atoms()
            except:
                print("Was not able to reorder the atom labels using "+
                  "openbabel. Please note, the atom orderings in "+
                  "rstruct may not be consistent for different molecules.")
        
        atom_idx = 0
        for idx,molecule_struct in enumerate(self.molecules):
            geo_array = molecule_struct.get_geo_array()
            ele = molecule_struct.geometry['element']
            
            ## Add correct bonds to rstruct
            molecule_bonds = molecule_struct.get_bonds()
            ## Adjust to correct idx
            correct_idx_bonds = []
            for bond_list in molecule_bonds:
                temp_list = []
                for value in bond_list:
                    temp_list.append(value+atom_idx)
                correct_idx_bonds.append(temp_list)
            
            for bond_idx,value in enumerate(correct_idx_bonds):
                bond_idx += atom_idx
                rstruct.properties["bonds"][bond_idx] = value
            
            for i,coord in enumerate(geo_array):
                rstruct.append(coord[0],coord[1],coord[2],ele[i])
                rstruct.properties["molecule_idx"][idx].append(atom_idx)
                atom_idx += 1
                
        return rstruct
    
    
    def _reorder_atoms(self):
        """
        Reorders atoms in each molecule based off the ordering of the unique 
        molecule using LabelAtoms from ibslib using Pymatgen functions.
        """
        unique_molecule = self.unique[0]
        la  = LabelAtoms()
        for molecule in self.molecules:
            la.uniform_labels(unique_molecule, molecule, 
                              hold_first_order=True)

            
class MoleculesByIndex(BaseDriver_):
    """
    Creates molecule lists that are identical to FindMolecules. Molecule lists
    are constructed by using the number of atoms per molecule. This Driver 
    assumes that the molecules are ordered in the structure.
    
    Arguments
    ---------
    napm: int
        Number of atoms per molecule for crystal structures. 
    fix_pbc: bool
        If True, will fix cases where the molecules have been split across a 
        periodic boundary 
    
    """
    def __init__(self, napm, fix_pbc=False):
        self.napm = int(napm)
        self.fix_pbc = fix_pbc
        self.molecules = []
        self.unique = []
    
    
    def calc_struct(self, struct):
        self.molecules = []
        self.unique = []
        
        geo = struct.get_geo_array()
        ele = struct.geometry["element"]
        
        ## Check that struct and self.napm are compatible
        err = geo.shape[0] % self.napm
        if err != 0:
            raise Exception("Number of atoms per molecule {}".format(self.napm)+
                            " is not compatible with the number of atoms "+
                            "in the structure {}.".format(geo.shape[0]))
        
        num_molecules = int(geo.shape[0] / self.napm)
        idx = np.arange(0,self.napm)
        
        for num in range(num_molecules):
            temp_idx = idx + num*self.napm
            molecule = Structure.from_geo(geo[temp_idx],
                                          ele[temp_idx])
            
            if self.fix_pbc:
                molecule = construct_smallest_molecule(struct, 
                                                       molecule)
            
            self.molecules.append(molecule)
        
        ## Assume that all molecules are identical so any of them can be 
        ## taken to be unique. 
        self.unique.append(self.molecules[0])
        
    
    def write(self, output_dir, file_format="json", overwrite=False):
        """
        Write the molecule list.
        
        """
        temp_dict = StructDict()
        for molecule in self.molecules:
            temp_dict.update(molecule)
            
        write(output_dir, 
              temp_dict, 
              file_format=file_format, 
              overwrite=overwrite)
        

def find_unique_molecules_no_conformation(molecule_struct_list):
    """
    Will need to clean up this unique molecule finding in the future. 
    In additon, this method for finding unique molecules will not require 
    Pytorch which is advantageous for installation on systems running CentOS 6
    like Hippolyta. 
    
    """
    
    ## First test that formulas are the same
    same_formula = []
    added_list = []
    for idx1,molecule1 in enumerate(molecule_struct_list):
        temp_list = []
        ## Check if molecule has already been added
        if idx1 in added_list:
            continue
        else:
            added_list.append(idx1)
            
        formula1 = molecule1.formula()
        temp_list.append(molecule1)
        
        ## Compare with all others
        for idx2,molecule2 in enumerate(molecule_struct_list[idx1+1:]):
            ## Make sure to offset idx2 correctly
            idx2 += idx1+1
            formula2 = molecule2.formula()
            if formula1 == formula2:
                temp_list.append(molecule2)
                added_list.append(idx2)
        
        ## Now add in groups of molecules
        same_formula.append(temp_list)
    
    
    ## There's a range of physically relevant mult
    ## ranges to search over for the bond neighborhood calculation. 
    ## This is physical issue caused by describing all covalent bonds by a 
    ## single number for every bonding environment.
    ## This method, while hack-ish, is physically motivated and more robust.
    bn_unique = []
    for mult in [1.1, 1.15, 1.2, 1.25, 1.3]:
        temp_unique = []
        ## Now check bonding information for each group
        bn = BondNeighborhood(bond_kw={"mult": mult,
                                       "skin": 0.0})
        for molecule_group in same_formula:
            if len(molecule_group) == 1:
                ## Must be unique
                temp_unique.append(molecule_group[0])
                continue
            
            ## Otherwise, we have to compare bonding information
            bonding_info = []
            for molecule in molecule_group:
                fragments,count = bn.calc(molecule)
                bonding_info.append((fragments,list(count)))
                
            added_list = []
            for idx1,bonding1 in enumerate(bonding_info):
                if idx1 in added_list:
                    continue
                else:
                    added_list.append(idx1)
                    ## Must be unique at this point
                    temp_unique.append(molecule_group[idx1])
                    
                frag1,count1 = bonding1
                
                for idx2,bonding2 in enumerate(bonding_info[idx1+1:]):
                    idx2 += idx1+1
                    frag2,count2 = bonding2
                    
                    ## Compare
                    if frag1 == frag2 and count1 == count2:
                        added_list.append(idx2)
                    else:
                        ## Don't have to do anything because if the molecule is 
                        ## unique then it will be put in the unique list in 
                        ## the loop just outside of this one. 
                        pass
        
        bn_unique.append(temp_unique)
    
    ## Return smallest from bn unique since we have searched over all 
    ## physically relevant values. This should give a more robust result. 
    length = [len(x) for x in bn_unique]
    min_idx = np.argmin(length)
    
    return bn_unique[min_idx]


def find_unique_molecules(molecule_struct_list, tol=1):
    """
    Bad wrapper for now
    
    """
    if len(molecule_struct_list) == 0:
        return []
    molecule_groups = find_unique_groups(molecule_struct_list,tol=1)
    unique_molecule_idx = [x[0] for x in molecule_groups]
    return [copy.deepcopy(molecule_struct_list[x]) for x in unique_molecule_idx]


def reconstruct_from_unique_molecules(molecule_struct_list, tol=1):
    """
    Reconstruct the geometry from unique molecules identified in the structure
    """
    molecule_groups = find_unique_groups(molecule_struct_list,tol=1)
    unique_molecule_idx = [x[0] for x in molecule_groups]
    raise Exception("Not Implemented")


def find_unique_groups(molecule_struct_list, tol=1):
    """
    Returns a list of indices for unique molecule groups where every molecule 
    in each group are indentical within the tolerance.
    
    Arguments
    ---------
    struct: Structure
    tol: float
        Tolerance for identifying individual molecules
        
    """   
    difference_matrix = p_molecule_distance(molecule_struct_list)
    molecule_groups = unique_groups(difference_matrix, tol=tol)
    return molecule_groups
    

def unique_groups(difference_matrix, tol=1):
    """ 
    Breaks difference matrix into groups of similar molecules.
    Returns list molecules which make up unique groups for indexing into the 
    original molecule_struct_list.
    """
    # List of molecules which need to be sorted
    unsorted = np.array([x for x in range(difference_matrix.shape[0])])
    # List of groups of the same molecules
    molecule_groups = []
    
    while len(unsorted) != 0:
        # Pick first molecule which hasn't been sorted
        current_idx = unsorted[0]
        # Take its row of the difference matrix
        row = difference_matrix[current_idx,:]
        # Take positions of other molecules yet to be sorted
        row = row[unsorted]
        # Find those same by tolerance value
        same_tol = row < tol
        # Must be greater than 0 so value cannot be -1
        same_nonzero = row >= 0
        # Combine
        same = np.logical_and(same_tol, same_nonzero)
        # Gather results
        same_idx = np.where(same == True)[0]
        # Reference original molecule index which is obtained from the 
        # unsorted list for the final unique groups
        molecule_groups.append(unsorted[same_idx])
        # Delete indexs in unsorted which have now been sorted
        unsorted = np.delete(unsorted, same_idx)              
    
    return molecule_groups


def p_molecule_distance(molecule_list):
    """ 
    Finds minimum pairwise distance between all molecules in molecule list.
    Pairwise distance computed by taking the difference between the internal 
    coordinates of each molecule in the molecule list.
    
    Returns -1 if the elements or number of elements are not the same.
    Otherwise, returns positive value of the difference between internal 
    coordinates of the two different molecules. 
    
    """
    ## Moved to here to make torch optional
    import torch
    
    num_mol = len(molecule_list)
    
    # Initialize to zeros
    difference_matrix = np.zeros((num_mol,num_mol))
    
    # Loop over upper diagonal of pairwise matrix
    for i in range(num_mol):
        for j in range(i+1,num_mol):
            molecule_0 = molecule_list[i]
            molecule_1 = molecule_list[j]
            
            # Compute internal coordinates of 1
            R_0 = calc_R(molecule_0)
            R_0 = ele_R(R_0)
            R_0 = struct_R(R_0)
            
            # Compute internal coordinates of 2
            R_1 = calc_R(molecule_1)
            R_1 = ele_R(R_1)
            R_1 = struct_R(R_1)
            
            # Check if elements are the same
            if [key for key in R_0.keys()] != [key for key in R_1.keys()]:
                 # If elements are not all the same then the structures
                 # are certainly different so return -1
                 difference_matrix[i,j] = -1
                 continue
            
            # Computer difference w.r.t. different interactions in system
            difference = 0
            diff = 0
            for element,inter_dict in R_0.items():
                for inter,value_tensor in inter_dict.items():
                    R_1_value_tensor = R_1[element][inter]
                    
                    # Check if number of interactions are the same
                    if len(value_tensor) != len(R_1_value_tensor):
                         # If iteractions are not the same length then they
                         # are certainly different so return -1
                         diff = -1
                         continue
                    
                    temp = torch.sum(value_tensor - R_1_value_tensor)
                    difference += torch.abs(temp) / len(value_tensor.view(-1))
            
            # Put value in difference matrix symmetric across diagonal
            if diff < 0:
                difference_matrix[i,j] = -1
                difference_matrix[j,i] = -1
            else:
                difference_matrix[i,j] = difference
                difference_matrix[j,i] = difference
    
    return difference_matrix                        
        
        


if __name__ == "__main__":
    pass
