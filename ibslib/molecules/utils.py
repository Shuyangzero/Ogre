
import numpy as np

from ase.data import atomic_numbers,atomic_masses_iupac2016
from pymatgen.symmetry.analyzer import PointGroupAnalyzer as PGA


def check_molecule(struct, exception=True):
    # Check for valid molecule_struct
    if len(struct.get_lattice_vectors()) > 0:
        if exception:
            raise Exception("Structure with lattice vectors {} was passed "
                .format(struct.get_lattice_vectors_better())+
                "into MoleculeBonding class. Molecule structure "+
                "without lattice vectors should be passed to "+
                "MoleculeBonding.")
        else:
            return False
    else:
        return True
    

def com(struct):
    """
    Calculates center of mass of the system. 

    """
    check_molecule(struct)
    geo_array = struct.get_geo_array()
    element_list = struct.geometry['element'] 
    mass = np.array([atomic_masses_iupac2016[atomic_numbers[x]] 
                     for x in element_list]).reshape(-1)
    total = np.sum(mass)
    com = np.sum(geo_array*mass[:,None], axis=0)
    com = com / total
    return com


def moit(struct):
    """
    Calculates the moment of inertia tensor for the system.

    """
    check_molecule(struct)
    mol = struct.get_pymatgen_structure()
    pga = PGA(mol)
    ax1,ax2,ax3 = pga.principal_axes
    return np.vstack([ax1,ax2,ax3])


def orientation(struct):
    """
    Returns a rotation matrix for the moment of inertial tensor 
    for the given molecule Structure. 

    """
    check_molecule(struct)
    axes = moit(struct)
    return np.linalg.inv(axes.T)


def align(struct):
    """
    Aligns the molecule such that the COM is at the origin and the axes defined 
    by the moment of inertia tensor are oriented with the origin. 
    
    """
    check_molecule(struct)
    trans = com(struct)
    rot = orientation(struct)
    
    geo = struct.get_geo_array()
    geo = geo - trans
    geo = np.dot(geo, rot.T)
    struct.from_geo_array(geo, struct.geometry["element"])
    return struct
    


def show_axes(struct, ele="He"):
    """
    Visualize the COM of the molecule and the axes defined
    by the moment of inertial tensor of the molecule by adding
    an atom of type ele to the structure.

    """
    check_molecule(struct)
    com_pos = com(struct) 
    axes = moit(struct)
    struct.append(com_pos[0],com_pos[1],com_pos[2],ele)
    for row in axes:
        row += com_pos
        struct.append(row[0],row[1],row[2],ele)


    
