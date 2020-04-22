# -*- coding: utf-8 -*-

__author__ = 'Manny Bier'

import os, itertools,copy
import numpy as np

from scipy import sparse
from scipy.spatial.distance import cdist

from ase.neighborlist import NeighborList,natural_cutoffs
from ase.data import atomic_masses_iupac2016,atomic_numbers
from pymatgen.symmetry.analyzer import PointGroupAnalyzer as pga

from ibslib import Structure
from ibslib.io import read,write,output_geo



"""
It will be better overall to break these complex operations into classes 
which will be easier to use and have a more intuitive API.
"""

def get_molecules(struct, mult=1.05):
    """ 
    
    Arguments
    ---------
    mult: float
        Multiplicative factor to use for natural_cutoffs
    
    Returns
    -------
    List of Structure objects for each molecule identified using the smallest
    molecule representation.
    
    """
    molecule_list = find_molecules(struct, mult=mult)
    molecule_struct_list = extract_molecules(struct, molecule_list,
                                             mult=mult)
    if len(molecule_struct_list) == 0:
        raise Exception("No molecules found for structure {}."
                        .format(struct.struct_id))
    return molecule_struct_list


def find_molecules(struct, mult=1.05):
    """ 
    Identify molecular fragments in struct
    
    Returns
    -------
    List of lists for atom index of each molecule
    
    """
    atoms = struct.get_ase_atoms()
    cutOff = natural_cutoffs(atoms, mult=mult)
    ## Skin=0.3 is not a great parameter, but it seems to work pretty well
    ## for mulicule identification. In addition, it's not the same affect as 
    ## change the mult value because it's a constant addition to all 
    ## covalent bonds.
    neighborList = NeighborList(cutOff, skin=0.3)
    neighborList.update(atoms)
    matrix = neighborList.get_connectivity_matrix()
    n_components, component_list = sparse.csgraph.connected_components(matrix)
    molecule_list = [np.where(component_list == x)[0] 
                    for x in range(n_components)]
    return molecule_list


def extract_molecules(struct, molecule_list, whole_molecules=True, 
                      mult=1.05):
    """ Converts list of list of coordinates to Structures """
    # Information from original structure
    geo = struct.get_geo_array()
    elements = struct.geometry['element']
    
    # Extract molecule geometries from original
    molecule_geo_list = [geo[x,:] for x in molecule_list]
    molecule_ele_list = [elements[x] for x in molecule_list]
    
    # Convert geometry to Structure
    molecule_struct_list = [Structure() for x in range(len(molecule_list))]
    [x.from_geo_array(molecule_geo_list[i],molecule_ele_list[i])
      for i,x in enumerate(molecule_struct_list)]
    
    # Construct whole molecule representations
    if whole_molecules:
        molecule_struct_list = [construct_smallest_molecule(struct,x,mult=mult) 
                                 for x in molecule_struct_list]
    return molecule_struct_list


def construct_smallest_molecule(struct,molecule_struct,mult=1.05):
    """ Make molecule smallest possible w.r.t. structure pbc
    
    Purpose
    -------
    Sometimes structures are given where molecules are not fully connected 
    because all atomic coordinates have been brought into the cell. This 
    function minimizes the distance between coordinates in the molecule 
    w.r.t. the pbc of the input structure such that the molecule's final
    geometric coordinates are fully connected. 
      
    """
    # Check if molecule is fully connected
    mol_index_list = find_molecules(molecule_struct,mult=mult)
    if len(mol_index_list) == 1:
        return molecule_struct
    
    # New method: construct supercell and pick molecule fragment with the same
    # length as the input. Doesn't matter which one is chosen because they are 
    # all images of the same molecule. The final positions will be augmented
    # to have COM inside the cell.
    temp = copy.deepcopy(molecule_struct)
    temp.set_lattice_vectors(struct.get_lattice_vectors())
    
    # Loop through creating supercells of increasing size to find smallest
    # molecule representation efficiently.
    success = False
    for i in range(2,9):
        # Construct ixixi supercell about the origin
        supercell = construct_supercell_by_molecule(temp, supercell=i, 
                                                    include_negative=False)
        # Get atom index for molecules from supercell
        result = find_molecules(supercell, mult=mult)
        # Get molecule structure
        molecule_list = extract_molecules(supercell, result, 
                                          mult=mult,
                                          whole_molecules=False)
        # Identify whole molecule in non-periodic cell
        frag_list = [len(find_molecules(x, mult=mult)) for x in molecule_list]
        try:
            whole_molecule_idx = frag_list.index(1)
            success = True
            break
        except:
            pass
        
    if success == False:
        raise Exception('No whole represenation was found for the molecule '+
            'without periodic boundary conditions. Please check the ' +
            'structure for any irregularities. If none are found, then '+
            'improvements probably need to be made to the source code to '+
            'work for this structure.')
    
    whole_molecule = molecule_list[whole_molecule_idx]
    geo = whole_molecule.get_geo_array()
    
    # Ensure COM of final molecule is inside cell and smallest possibe
    # w.r.t. lattice sites.
    COM = com(whole_molecule)
    
    # Lattice vector array as columns
    lattice_vectors = np.array(struct.get_lattice_vectors()).T
    lattice_vectors_i = np.linalg.inv(lattice_vectors)
    relative_COM = np.dot(lattice_vectors_i, COM)
    
    # First make COM all positive w.r.t. lattice vectors 
    trans_idx = relative_COM < -0.0001
    trans_vector = np.dot(lattice_vectors, trans_idx[:,None])
    geo = geo + trans_vector.T
    
    # Recompute COM then move inside of cell
    temp_molecule = Structure()
    temp_molecule.from_geo_array(geo, whole_molecule.geometry['element'])
    COM = com(temp_molecule)
    relative_COM = np.dot(lattice_vectors_i, COM)
    
    trans_idx = relative_COM > 0.99
    trans_vector = np.dot(lattice_vectors, trans_idx[:,None])
    geo = geo - trans_vector.T
    
    # Set final molecule
    final_molecule = Structure()
    final_molecule.from_geo_array(geo, whole_molecule.geometry['element'])
    final_molecule.struct_id = molecule_struct.struct_id

    return final_molecule
    

def reconstruct_with_whole_molecules(struct):
    """ Build smallest molecule representation of struct. 
    
    """
    rstruct = Structure()
    rstruct.set_lattice_vectors(struct.get_lattice_vectors())
    molecule_struct_list = get_molecules(struct)
    for molecule_struct in molecule_struct_list:
        geo_array = molecule_struct.get_geo_array()
        ele = molecule_struct.geometry['element']
        for i,coord in enumerate(geo_array):
            rstruct.append(coord[0],coord[1],coord[2],ele[i])
    return rstruct


def com(struct):
    """
    Calculates center of mass of the system.
    """
    geo_array = struct.get_geo_array()
    element_list = struct.geometry['element']
    mass = np.array([atomic_masses_iupac2016[atomic_numbers[x]]
                     for x in element_list]).reshape(-1)
    total = np.sum(mass)
    com = np.sum(geo_array*mass[:,None], axis=0)
    com = com / total
    return com
        

def find_translation_vector(f1, f2, lattice_vectors):
    """
    From a set of a lattice vectors, find the lattice vector that minimizes the
      distance between fragment 1, f1, and fragment 2, f2.
    """
    base_atom = len(f1)
    full_geo = np.concatenate([f1, f2], axis=0)
    x_dist = np.min(calc_euclidean_dist_vectorized(full_geo[:,0][:,None]
                    )[0:base_atom,base_atom:])
    y_dist = np.min(calc_euclidean_dist_vectorized(full_geo[:,1][:,None]
                    )[0:base_atom,base_atom:])
    z_dist = np.min(calc_euclidean_dist_vectorized(full_geo[:,2][:,None]
                    )[0:base_atom,base_atom:])

    min_dist = np.array([[x_dist,y_dist,z_dist]])
    closest_vector = np.argmin(cdist(min_dist, lattice_vectors))
    
    # Decide to add or subtract lattice vector
    sign = 1
    f1_mean = np.mean(f1,axis=0)
    f2_mean = np.mean(f2,axis=0)
    mean_dist = f2_mean - f1_mean
    plus = mean_dist + lattice_vectors[closest_vector,:]
    minus = mean_dist - lattice_vectors[closest_vector,:]
    if np.sum(np.abs(plus)) > np.sum(np.abs(minus)):
        sign = -1
    return closest_vector,sign
    

def get_molecule_orientation(molecule_struct):
    """
    Not quite sure what this function needs to do yet, but the indexing for 
      pymatgen principal axes is shown correctly
    
    Arguments
    ---------
    Pymatgen molecule object
    
    """
    molp = molecule_struct.get_pymatgen_structure()
    PGA = pga(molp)
    pa = PGA.principal_axes
#    axes = np.zeros(3,3)
#    for i,row in enumerate(pa):
#        axes[i,:] = row
    return pa

def get_orr_tensor(struct):
    """ Gets orientation of all molecules in the struct """
    molecule_list = get_molecules(struct)
    orr_tensor = np.zeros((len(molecule_list),3,3))
    for i,molecule_struct in enumerate(molecule_list):
        orr_tensor[i,:,:] = get_molecule_orientation(molecule_struct)
    return orr_tensor


def get_COM(struct):
    """ Gets COM positions for all molecules in the structure 
    
    Returns
    -------
    List of all COM positions in the structure
    """
    molecule_list = get_molecules(struct)
    COM_array = np.zeros((len(molecule_list),3))
    for i,molecule_struct in enumerate(molecule_list):
        COM_array[i,:] = calc_COM(molecule_struct)
    return COM_array
        
        
def calc_COM(molecule_struct):
    """ COM calculation for Molecule Structure """ 
    geometry = molecule_struct.get_geo_array()
    elements = molecule_struct.geometry['element']
    element_numbers = [atomic_numbers[x] for x in elements]
    element_masses = np.array([atomic_masses_iupac2016[x] 
                                 for x in element_numbers])[:,None]
    weighted_geometry = geometry*element_masses
    return np.sum(weighted_geometry,axis=0) / np.sum(element_masses)


def construct_supercell_by_molecule(struct,supercell=3,include_negative=False):
    """ Construct supercell w.r.t. the molecules in the current structure

    Arguments
    ---------
    struct: Structure 
        Structure object that was used to construct the molecules argument,
        Must have lattice parameters.
    supercell: int
        Dimension of supercell (int x int x int)
        
    """
    if supercell <= 0:
        raise Exception('Input to construct_supercell must be larger than 0')
    
    lattice_vectors = struct.get_lattice_vectors()
    if lattice_vectors == False:
        raise Exception('Input Structure object to function '+
                'construct_supercell must have lattice parameters.')
    lattice_vectors = np.array(lattice_vectors)
    
    # Array for translations to construct supercell
    translation_vectors = get_translation_vectors(supercell, lattice_vectors,
                                                  include_negative=include_negative)
    
    # Initialize supercell
    supercell_struct = Structure()
    supercell_struct.set_lattice_vectors(lattice_vectors*supercell)
    geo_array = struct.get_geo_array()
    
    # Broadcast geometry with translation vectors
    supercell_geo = geo_array[:,None,:] + translation_vectors
    num_atoms,num_tr,dim = supercell_geo.shape
    
    # Getting correct indexing for supercell tensor
    # Indexing scheme for molecules in first unit cell
    depth_index = num_tr*dim*np.arange(num_atoms)
    # Broadcast across three dimensions
    column_values = np.arange(3)
    unit_cell_index = depth_index[:,None] + column_values
    # Index scheme for the next unit cells in supercell
    molecule_index = np.arange(num_tr)*3
    # Broadcast initial molecule across next moleculess
    supercell_index = molecule_index[:,None,None] + unit_cell_index
    supercell_index = supercell_index.reshape(num_tr*num_atoms, 3)
    
    supercell_geo = np.take(supercell_geo, supercell_index)
    
    ###########################################################################
    # For example, this gets the original geometry                            #
    ###########################################################################
#    depth_index = num_tr*dim*np.arange(num_atoms)
#    column_values = np.arange(3)
#    broadcasted_index = depth_index[:,None] + column_values
    ###########################################################################
    
    num_ele = translation_vectors.shape[0]
    supercell_elements = np.tile(struct.geometry['element'],num_ele)
    supercell_struct.from_geo_array(supercell_geo, supercell_elements)
    
    return supercell_struct

def construct_molecular_index_for_supercell(num_atoms, num_tr,
                                            combine_mol=True):
    '''
    
    Arguments
    ---------
    num_atoms: int
        Number of atoms in the original structure
    num_tr: int
        Number of translation vectors applied to construct supercell
    combine_mol: bool
        True: molecules should be combined, such is the case when the desired 
                output is a single supercell strcutre
        False: molecules should not be combined, such is the case when trying
                 to identify the smallest representation of the molecule w/o
                 pbc
    
    '''
    # Cartesian space
    dim = 3
    # Getting correct indexing for supercell tensor
    # Indexing scheme for molecules in first unit cell
    depth_index = num_tr*dim*np.arange(num_atoms)
    # Broadcast across three dimensions
    column_values = np.arange(3)
    unit_cell_index = depth_index[:,None] + column_values
    # Index scheme for the next unit cells in supercell
    molecule_index = np.arange(num_tr)*3
    # Broadcast initial molecule across next moleculess
    supercell_index = molecule_index[:,None,None] + unit_cell_index
    
    if combine_mol == True:
        return supercell_index.reshape(num_tr*num_atoms, 3)
    
    return supercell_index


def construct_orientation_supercell(struct,supercell,include_negative=False,
                                    molecule_list=[]):
    """ Construct supercell of only molecular orientations 
    
    Arguments
    ---------
    struct: Structure 
        Structure object that was used to construct the molecules argument,
        Must have lattice parameters.
    supercell: int
        Dimension of supercell (int x int x int)
    molecule_lsit: list of Structures
        Can pass in argument if molecule_list was pre-computed
        
    """
    if supercell <= 0:
        raise Exception('Input to construct_supercell must be larger than 0')
        
    lattice_vectors = struct.get_lattice_vectors()
    if lattice_vectors == False:
        raise Exception('Input Structure object to function '+
                'construct_supercell must have lattice parameters.')
        
    lattice_vectors = np.array(lattice_vectors)
    translation_vectors = get_translation_vectors(supercell, lattice_vectors,
                                                  include_negative)
    if len(molecule_list) == 0:
        molecule_list = get_molecules(struct)
    
    num_atoms = struct.get_geo_array().shape[0]
    num_molecules = len(molecule_list)
    num_tr = len(translation_vectors)
    
    COM_array = np.array([calc_COM(molecule_struct) 
                         for molecule_struct in molecule_list])
    orientation_tensor = np.array([get_molecule_orientation(mol) 
                                     for mol in molecule_list])
    orientation_tensor = orientation_tensor + COM_array[:,None,:]
    orientation_tensor = orientation_tensor[:,None,:] + \
                             translation_vectors[:,None,:]
    orientation_tensor = orientation_tensor.reshape(num_molecules*num_tr,3,3)
    
    COM_tensor = COM_array[:,None,:] + translation_vectors
    COM_tensor = COM_tensor.reshape(num_molecules*num_tr,3)
    
    return orientation_tensor,COM_tensor
    
    

def get_translation_vectors(supercell, lattice_vectors, include_negative=False):
    ''' Returns all translation vectors for a given supercell size
    
    Arguments
    ---------
    supercell: int
        Value of the supercell dimension. Example 3x3x3
    lattice_vectors: Numpy array
        Lattice vectors in row format where each lattice vector is one row.
    include_negative: bool
        False: Only supercells in the positive direction will be constructed
        True: Supercells in the positive and negative direction will be 
                constructed. 
        If true, constructs the supercell about the origin of the original 
    
    Returns: Numpy array of all translation vectors in row format.
    '''
    if include_negative:
        list_range = [x for x in range(-supercell+1,supercell,1)]
    else:
        list_range = [x for x in range(supercell)]
    tr = list(itertools.product(list_range,list_range,list_range))
    translation_vectors = np.dot(tr,lattice_vectors)
    return translation_vectors


def compute_motif(struct, supercell=3, include_negative=True, num_mol=12):
    """ Computes deg_array which is translated into specific packing motifs
    
    Arguments
    ---------
    supercell: int
        Value of the supercell dimension. Example 3x3x3
    include_negative: bool
        False: Only supercells in the positive direction will be constructed
        True: Supercells in the positive and negative direction will be 
                constructed. This will double the number constructed. 
    num_mol: int >= 4
        Number of nearest neighbor molecules to be used for motif 
          identification. Should be at least four.
          
    """
    deg_array,plane_deg_min = compute_deg_array(struct, supercell, 
                                        include_negative, num_mol=num_mol)
    return motif_definitions(deg_array,plane_deg_min)


def compute_deg_array(struct, supercell=3, include_negative=True, num_mol=12):
    molecule_list = get_molecules(struct)
    orientation_tensor,COM_array = construct_orientation_supercell(struct, 
                                            supercell,include_negative,
                                            molecule_list)
    deg_array,plane_deg_min = compute_orientation_difference(orientation_tensor,COM_array,
                                             molecule_list,num_mol=num_mol)
    return deg_array,plane_deg_min


def motif_definitions(deg_array,plane_deg_min):
    """ Defines how motifs are identified from the deg_array
    
    Arguments
    ---------
    deg_array: np.array (n,)
        Vector of orientation differences found 
    plane_deg_min: np.array
        Vector of orientation differences found for molecules that were 
        co-planar to the reference molecule
    """
    num_mol = 4
    if len(deg_array) < num_mol:
        raise Exception("For proper motif identification, the input array "+
                        "must have a length of at least 6. "+
                        "Input was {}.".format(deg_array))
    # Only use first for neighbors
    def_deg = deg_array[0:num_mol]
    
    sheet_like = def_deg < 9
    # Be more stringent for sheet classification
    if np.sum(deg_array < 9) == len(deg_array):
            return 'Sheet'
    else:
        if sheet_like[0] == True:
            if sheet_like[1] != True:
                if np.sum(plane_deg_min < 9) == len(plane_deg_min):
                    return 'Sheet'
                return 'Sandwich'
            else:
#                if np.sum(plane_deg_min < 9) == len(plane_deg_min):
#                    return 'Gamma'
                return 'Gamma'
        else:
            # Have at least 1 co-planar in first 4 neighbors
            if np.sum(sheet_like) == 1:
                if np.sum(plane_deg_min < 9) == len(plane_deg_min):
                    return 'Sheet'
            return 'Herringbone'
    

def compute_orientation_difference(orientation_tensor,COM_array,molecule_list,
                                   num_mol=12):
    """ 
    
    Computes difference between molecular orientation bewteen the molecular 
      plane and the principal axes of num_mol from the supercell closest to 
      the molecule in molecule_list closest to the origin.
    
    Arguments
    ---------
    num_mol: int
        Should be approximately equal to the number of molecules per unit cell 
          multiplied by supercell
    """
    centerd_orientation_tensor = orientation_tensor - COM_array[:,None]
    
    index_min,index_dist_min = find_nearest_COM(COM_array, 
                                    reference=molecule_list, num_mol=num_mol)
    
    molecule_struct_min = molecule_list[index_min]
    origin_orientation = centerd_orientation_tensor[index_dist_min]
    # Compute norm to molecular plane of original molecule
    plane_norm = get_molecule_orientation(molecule_struct_min)[0,:]
    
    original_COM_array = np.array([calc_COM(x) for x in molecule_list])
    COM_test = original_COM_array[index_min,:]
    origin_COM = COM_array[index_dist_min]
    dist_vector = COM_test - origin_COM 
    dist_vector = dist_vector / np.linalg.norm(dist_vector,axis=1)[:,None]
    COM_test = COM_test/np.linalg.norm(COM_test)
    COM_angles = np.dot(dist_vector, COM_test)
    np.minimum(COM_angles, 1.0, out=COM_angles)
    molecule_planes = np.rad2deg(np.arccos(COM_angles))
    np.around(molecule_planes,decimals=1, out=molecule_planes)
    
    # Identify if there are any molecular planes
    index_plane = np.where((np.abs(molecule_planes-90) < 11) | 
                    (np.abs(molecule_planes-180) < 11) | 
                    (molecule_planes < 11))     
    
    orr_diff_array = np.zeros((num_mol,3))
    for i,orr2 in enumerate(origin_orientation):
        orr_diff_array[i,:] = np.dot(plane_norm,orr2.T)
    
    # Small numerical errors
    np.minimum(orr_diff_array, 1.0, out=orr_diff_array)
    deg = np.rad2deg(np.arccos(orr_diff_array))
    deg_min = np.min(deg,axis=1)
    np.around(deg_min,decimals=1, out=deg_min)
    
    return deg_min,deg_min[index_plane]

def find_nearest_COM(COM_array, reference=[], num_mol=12):
    """ Find index of nearest num_mol to origin with optional reference list
    
    Arguments
    ---------
    COM_array: np.array nx3
        2D matrix of COM positions of all molecules to be indexed
    reference: list of Structures
        If provided, a list of Structures to be used as the reference molecule.
        The molecule closest to the origin of this list is identified and 
        the num_mol nearest in the COM_array will be indexed.
    num_mol: int
        Number of nearest neighbors to identify
    
    Returns
    -------
    index_min: int
        Index of the COM found nearest to the origin w.r.t the COM_array or 
        the reference list if the reference list is provided
    index_dist_min: np.array
        Vector of index for nearests neighbors to the min COMs
    
    """
    # Get COM of molecule struct closest to origin
    if len(reference) > 1:
        original_COM_array = np.array([calc_COM(x) for x in reference])
        COM_dist = np.linalg.norm(original_COM_array,axis=1)        
        index_min = np.argmin(COM_dist)
        COM_min = original_COM_array[index_min,:][None,:]
    elif len(reference) == 1:
        COM_min = calc_COM(reference[0])
        index_min = 0
    else:
        COM_dist = np.linalg.norm(COM_array, axis=1)
        index_min = np.argmin(COM_dist)
        COM_min = COM_array[index_min,:][None,:]
    # Get index of num_mol closest to this initial molecule
    dist_from_min = np.linalg.norm(COM_array-COM_min,
                                   axis=1)
    index_dist_min = np.argsort(dist_from_min)
    same_mol = np.where(dist_from_min < 0.1)[0]
    remove_index = np.where(index_dist_min == same_mol)[0]
    index_dist_min = np.delete(index_dist_min, remove_index)
    return index_min,index_dist_min[0:num_mol] 
    

def add_orientation_to_struct(struct,orientation_tensor,COM_array,
                              num_mol=-1,supercell=0,include_negative=False,
                              ele='S'):
    """ For visualizing orientations in the Structure
    """
    reference = []
    if supercell > 0:
        reference = get_molecules(struct)
        struct = construct_supercell_by_molecule(struct,supercell=supercell,
                                include_negative=include_negative)  
    if num_mol > 0:
        index_min,index_dist_min = find_nearest_COM(COM_array, reference, 
                                                    num_mol = num_mol)
        orientation_tensor = orientation_tensor[index_dist_min]              
        COM_array = COM_array[index_dist_min]
        
    if num_mol == 0:
        return struct
    
    for i,orientation in enumerate(orientation_tensor):
        for vector in orientation:
            struct.append(vector[0],vector[1],vector[2],ele)
    for vector in COM_array:
        struct.append(vector[0],vector[1],vector[2],'I')
    return struct


def implemented_motifs():
    """ 
    Returns
    -------
    List strings of all implemented motif definitions
    """
    return ['Sheet', 'Gamma', 'Herringbone',
            'Sandwich']
#            'Slipped Sheet', 'Slipped Gamma', 


def eval_motif(struct_dict, supercell=3, include_negative=True, num_mol=8):
    """
    kwargs: struct, supercell=3, include_negative=True, num_mol=12
    
    num_mol should be equal to nmpc*(supercell - 1) so as to not include a 
      molecule and its own image
    """
    motif_list = []
    for struct_id,struct in struct_dict.items():
        motif_list.append(compute_motif(struct, 
                  supercell=supercell, include_negative=include_negative,
                  num_mol=num_mol))
    return motif_list


def calc_and_separate(struct_dir, output_dir, motif_kwargs={}, file_format='json',
                      overwrite=False):
    """ 
    
    Calculates the motif for each structure in the directory and then
      separates structures into sub-directories of the output_dir based on
      the structural motif.
    
    """
    struct_dict = read(struct_dir)
    motif_list = implemented_motifs()
    motif_dict = {}
    for i,motif in enumerate(motif_list):
        motif_dict[motif] = i
    output_dicts = [{} for x in range(len(motif_list))]
    for struct_id,struct in struct_dict.items():
        motif = compute_motif(struct, **motif_kwargs)
        index = motif_dict[motif]
        output_dicts[index][struct_id] = struct
    
    for i,motif in enumerate(motif_list):
        motif_struct_dict = output_dicts[i]
        output_motif_dir = os.path.join(output_dir, motif)
        write(output_motif_dir, motif_struct_dict, 
                       file_format=file_format, overwrite=overwrite)


def calc_save_property(struct_dir, motif_kwargs={}, output_dir='', overwrite=False):
    """
    
    Adds motif to the property information of each Structure
    
    Arguments
    ---------
    struct_dir: file path
    motif_kwargs: see compute_motif
    output_dir: file path
        If no output_dir is provided than Structures will be saved in 
          the struct_dir
    
    """
    struct_dict = read(struct_dir)
    for struct_id,struct in struct_dict.items():
        motif = compute_motif(struct, **motif_kwargs)
        struct.set_property('motif', motif)
    
    if len(output_dir) == 0:
        output_dir = struct_dir
        overwrite = True
    write(output_dir, struct_dict, file_format='json',
                       overwrite=overwrite)


class MoleculesByIndex():
    """
    Gets molecule_structs simply using the index of the atoms and the 
    number of molecules per cell. 

    Arguments 
    ---------
    napm: int
        Number of atoms per molecule
    """
    def __init__(self, napm):
        self.napm = napm
    

    def __call__(self, struct):
        geo = struct.get_geo_array()
        elements = struct.geometry["element"]
        nmpc = int(len(geo) / self.napm)
        if nmpc * self.napm != len(geo):
            raise Exception("MoleculeByIndex failed because Structure {}"
                .format(struct.struct_id) + "has {} atoms which is not "
                .format(len(geo)) + "divisible by napm {}.".format(self.napm))
        molecule_struct_list = []
        for i in range(nmpc):
            # Offset index by molecule number
            idx = i*self.napm
            idx = np.arange(0,self.napm) + idx
            temp_geo = geo[idx,:]
            temp_ele = elements[idx]
            molecule_struct = Structure()
            molecule_struct.from_geo_array(temp_geo,temp_ele)
            molecule_struct_list.append(molecule_struct)
        return molecule_struct_list
            
        



if __name__ == '__main__':
    pass
    
