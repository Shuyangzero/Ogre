# -*- coding: utf-8 -*-


import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from spglib import niggli_reduce
from ase.data import atomic_numbers,atomic_masses_iupac2016

from ibslib import Structure
from ibslib.molecules import FindMolecules



def get_molecules(struct, 
                  bonds_kw={"mult":1.20, "skin":0.0, "update":False},
                  ret="idx"):
    """
    Returns the index of atoms belonging to each molecule in the Structure. 
    
    """
    bonds = struct.get_bonds(**bonds_kw)
    
    ## Build connectivity matrix
    graph = np.zeros((struct.geometry.shape[0],struct.geometry.shape[0]))
    for atom_idx,bonded_idx_list in enumerate(bonds):
        for bond_idx in bonded_idx_list:
            graph[atom_idx][bonded_idx_list] = 1
    
    graph = csr_matrix(graph)
    n_components, component_list = connected_components(graph)
    molecule_idx_list = [np.where(component_list == x)[0] 
                            for x in range(n_components)]
    
    if ret == "idx":
        return molecule_idx_list
    else:
        geo = struct.get_geo_array()
        ele = struct.geometry["element"]
        
        molecule_struct_list = []
        for idx,entry in enumerate(molecule_idx_list):
            mol_geo = geo[entry]
            mol_ele = ele[entry]
            
            mol = Structure.from_geo(mol_geo,mol_ele)
            mol.struct_id = "{}_molecule_{}".format(struct.struct_id,
                                                    idx)
            molecule_struct_list.append(mol)
        
        return molecule_struct_list


def reduce(struct):
    """
    Compute reduced lattice of the input Structure. This will modify input 
    structure. 
    
    """
    lv = struct.get_lattice_vectors()
    rcell = niggli_reduce(lv)
    struct.reset_lattice_vectors(list(rcell))
    return struct


def whole_molecules(struct, 
                    bonds_kw={"mult":1.20, "skin":0.0, "update":False},
                    move_in=True):
    """
    Contruct whole molecules for the Structure.
    
    """
    fm = FindMolecules(mult=bonds_kw["mult"],
                       residues=0,
                       output_rstruct=True,
                       mult_range=np.arange(0.85, 1.25, 0.005))
    fm.calc_struct(struct)
    
    if len(fm.rstruct.get_geo_array()) > 0:
        struct.from_geo_array(fm.rstruct.get_geo_array(),
                              fm.rstruct.geometry["element"])
        struct.properties["molecule_idx"] = fm.rstruct.properties["molecule_idx"]
    else:
        raise Exception("Reconstruction Failed")
    
    return struct


def move_com(struct, fm=FindMolecules(output_rstruct=True)):
    """
    Move center of mass of molecules in Structure inside the unit cell. User 
    may also specify a fm argument that is MoleculesByIndex if the system's
    molecules are ordered by index. 
    
    Arguments
    ---------
    struct: Structure
        Structure object to move molecules into the unit cell
    fm: FindMolecules
        Object used to find the molecules in the Structure. 
    
    """
    ## Use pre-computed molecule idx from FindMolecules.rstruct if available
    if "molecule_idx" in struct.properties:
        molecule_idx  = struct.properties["molecule_idx"]
    else:
        fm.calc_struct(struct)
        struct.from_geo_array(fm.rstruct.get_geo_array(),
                              fm.rstruct.geometry["element"])
        struct.properties["molecule_idx"] = fm.rstruct.properties["molecule_idx"]
        molecule_idx = struct.properties["molecule_idx"]
    
    geo = struct.get_geo_array()
    ele = struct.geometry["element"]
    lv = np.vstack(struct.get_lattice_vectors())
    lv_inv = np.linalg.inv(lv.T)
    
    for indices in molecule_idx:
        molecule_geo = geo[indices]
        molecule_ele = ele[indices]
        
        ## Calculate COM of molecule
        mass = np.array([atomic_masses_iupac2016[atomic_numbers[x]] 
                     for x in molecule_ele]).reshape(-1)
        total = np.sum(mass)
        com = np.sum(molecule_geo*mass[:,None], axis=0) / total
        
        ## Calculate how COM needs to be moved 
        pos = np.dot(lv_inv, com)
        pos = pos.reshape(-1)
        
        trans = np.zeros((3,))
        lv_trans = np.zeros((3,))
        for lv_idx,value in enumerate(pos):
                
            ## Check numerical tolerance around zero
            if np.abs(value) < 1e-3:
                continue
            
            if value < 0:
                ## Move COM in if value is negative
                mult = np.abs(int(value)) + 1
                trans += lv[lv_idx,:]*mult
                lv_trans += mult
            else:
                ## Can assume value is positive
                delta = value - 1
                
                if delta > 1e-3:
                    mult = int(value) + 1
                    trans -= lv[lv_idx,:]*mult
                    lv_trans -= mult
        
#        print(lv_trans, pos)
        geo[indices] += trans
        
    struct.from_geo_array(geo, ele)
    
    return struct
        
        
def preprocess(struct, bonds_kw={"mult":1.20, "skin":0.0, "update":False}):
    """
    Perform a set of rational preprocessing operations to safeguard the quality 
    of later numerical analysis performed on the molecular crysatl Structure. 
    
    """
    struct.get_bonds(**bonds_kw)
    struct = whole_molecules(struct)
    #### Niggli reduce can do very weird things where it makes the lattice 
    #### parameter definitions negative. Why would it do that? For now, 
    #### turning this feature off because it's detrimental. Maybe look at 
    #### pymatgen for better solution. 
#    struct = reduce(struct)
    struct = move_com(struct)
    bonds_kw["update"] = True
    struct.get_bonds(**bonds_kw)
    return struct


def center_on_molecule(struct, center_idx=0, pre=True, 
                       bonds_kw={"mult":1.20, "skin":0.0, "update":False}):
    """
    Will center the structure on molecule indexed by center_idx. Useful for 
    defining different atomic positions in the unit cells for physically the 
    same system. First use of this is for generating images of overlapping
    systems from the CSD and from relaxed structures. Sometimes, the molecule
    the system is centered on can change during relaxation.
    
    Arguments
    --------
    struct: Structure
        Structure to center 
    center_idx: int
        Index of molecule to center the structure on
    pre: bool
        If True, will perform preprocessing. If False, make sure preprocessing 
        is performed before calling this utility. 
    bonds_kw: dict
        Dictionary of keyword arguments to pass to struct.get_bonds
    
    """
    if pre:
        temp_bonds_kw = bonds_kw.copy()
        temp_bonds_kw["update"] = True
        struct = preprocess(struct, bonds_kw=temp_bonds_kw)
    
    geo = struct.get_geo_array()
    ele = struct.geometry["element"]
    molecule_idx = get_molecules(struct, bonds_kw=bonds_kw)
    
    ### Check that the one you want to center is not already the center
    com_list = []
    for entry in molecule_idx:
        molecule_geo = geo[entry]
        molecule_ele = ele[entry]
        mass = np.array([atomic_masses_iupac2016[atomic_numbers[x]] 
                     for x in molecule_ele]).reshape(-1)
        total = np.sum(mass)
        com = np.sum(molecule_geo*mass[:,None], axis=0) / total
        com_list.append(com)
    dist = cdist(np.array([[0,0,0]]), np.vstack(com_list))[0]
    min_to_center = np.argmin(dist)
    if min_to_center == center_idx:
        print("Structure already centered on molecule {}".format(center_idx))
        ## But we still perform all calculations below just in case
        
    ### Center on Molecule
    center_molecule_idx = molecule_idx[center_idx]
    center_molecule_geo = geo[center_molecule_idx]
    center_molecule_ele = ele[center_molecule_idx]
    
    ## Calculate molecule COM
    mass = np.array([atomic_masses_iupac2016[atomic_numbers[x]] 
                     for x in center_molecule_ele]).reshape(-1)
    total = np.sum(mass)
    com = np.sum(center_molecule_geo*mass[:,None], axis=0) / total
    
    ## Now translate system by negative com
    geo -= com
    struct.geometry["x"] = geo[:,0]
    struct.geometry["y"] = geo[:,1]
    struct.geometry["z"] = geo[:,2]
    
    ## Make sure all molecules COM inside unit cell
    struct.properties["molecule_idx"] = molecule_idx
    struct = move_com(struct)
    
    return struct
    



if __name__ == "__main__":
    from ibslib import SDS
    from ibslib.io import read,write 
    outstream = SDS("/Users/ibier/Desktop/Temp/Struct_Reduce", 
                    file_format="geo",
                    overwrite=True)
    
    s = read("/Users/ibier/Research/Results/Hab_Project/genarris-runs/FUQJIK/8mpc/20191210_Experimental_Volume/acsf_report/acsf/experimental/relaxation.json")
    outstream.update(s)
    
    preprocess(s)
    s.struct_id = "PREPROCESSES"
    
    outstream.update(s)
    
    
    
    
    
    
    
    
    
    
    
    
    