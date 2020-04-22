"""                                                                            
If any part of this module is used for a publication please cite:              
                                                                               
F. Curtis, X. Li, T. Rose, A. Vazquez-Mayagoitia, S. Bhattacharya,             
L. M. Ghiringhelli, and N. Marom "GAtor: A First-Principles Genetic            
Algorithm for Molecular Crystal Structure Prediction",                         
J. Chem. Theory Comput., DOI: 10.1021/acs.jctc.7b01152;                        
arXiv 1802.08602 (2018)                                                        
"""


from copy import deepcopy
from math import cos, sin
import math
import numpy as np
import random
from core import user_input, output
from structures.structure import StoicDict, Structure
from structures import structure_handling
from pymatgen import Molecule, Lattice
from pymatgen.symmetry.analyzer import PointGroupAnalyzer as pga
from functools import reduce

__author__ = "Farren Curtis, Xiayue Li, and Timothy Rose"
__copyright__ = "Copyright 2018, Carnegie Mellon University and "+\
                "Fritz-Haber-Institut der Max-Planck-Gessellschaft"
__credits__ = ["Farren Curtis", "Xiayue Li", "Timothy Rose",
               "Alvaro Vazquez-Mayagoita", "Saswata Bhattacharya",
               "Luca M. Ghiringhelli", "Noa Marom"]
__license__ = "BSD-3"
__version__ = "1.0"
__maintainer__ = "Timothy Rose"
__email__ = "trose@andrew.cmu.edu"
__url__ = "http://www.noamarom.com"

def main(list_of_structures, replica):
    '''
    Args: list of 2 Structures() to crossover, 
    the replica name running the crossover instance.
    Returns: A single Structure() if crossover 
    is successful or False if crossover fails 
    '''
    ui = user_input.get_config()
    num_mols = ui.get_eval('run_settings', 'num_molecules')
    parent_a = list_of_structures[0]
    parent_b = list_of_structures[1]
    output_parent_properties(parent_a, parent_b, replica)
    cross_obj = Crossover(parent_a, parent_b, num_mols, replica)
    child_struct = cross_obj.cross()
    return child_struct

class Crossover(object):
    ''' 
    Takes 2 parent structures and combines their 
    lattice vectors and moelcular orientations'''

    def __init__(self, parent_a, parent_b, num_mols, replica):
        '''__init__ will always run when a class is initialized. '''
        self.ui = user_input.get_config()
        self.replica = replica
        self.parent_a = deepcopy(structure_handling.cell_lower_triangular(parent_a))
        self.parent_b = deepcopy(structure_handling.cell_lower_triangular(parent_b))
        self.geometry_a = self.parent_a.geometry
        self.geometry_b = self.parent_b.geometry
        self.latA = deepcopy(self.parent_a.get_lattice_vectors())
        self.latB = deepcopy(self.parent_b.get_lattice_vectors())
        self.num_mols = num_mols
        self.num_atom_per_mol = int(len(self.parent_a.geometry)/self.num_mols)
        self.random_orientation(self.parent_a)
        self.random_orientation(self.parent_b)
        self.orientation_info_a = []
        self.orientation_info_b = []

    def cross(self):
        '''
        Crossover function which combines 
        (cell lattice angles) alpha, beta, gamma 
        (lattice vector magnitudes) a, b, c
        (molecule COM's) x, y, z
        (molecule orientations) thetax, thety, thetaz
        '''
        #shell debugging
        #print "parent a \n"
        #print self.parent_a.get_geometry_atom_format()
        #print "parent b"
        #print self.parent_b.get_geometry_atom_format()

        # Get lattice information from each parent 
        A_a, B_a, C_a = self.parent_a.get_lattice_magnitudes()
        A_b, B_b, C_b = self.parent_b.get_lattice_magnitudes()
        alpha_a, beta_a, gamma_a = self.parent_a.get_lattice_angles()
        alpha_b, beta_b, gamma_b = self.parent_b.get_lattice_angles() 
        lattice_info_a = [A_a, B_a, C_a, alpha_a, beta_a, gamma_a]
        lattice_info_b = [A_b, B_b, C_b, alpha_b, beta_b, gamma_b]

        #Molecule information for each parent
        self.get_orientation_info()

        #Create Child From combining 'genes' of parents
        atom_types = [self.geometry_a[i][3] for i in range(len(self.geometry_a))]
        child_lattice_info = self.combine_lattice_info(lattice_info_a, 
                                                       lattice_info_b)
        child_orientation_info = self.combine_orientation_info_dimers(self.orientation_info_a, 
                                                       self.orientation_info_b) 
        lattice, child_coords = self.reconstruct_child(child_lattice_info, 
                                                       child_orientation_info)
        child_struct = self.create_child_struct(child_coords, 
                                                lattice, 
                                                atom_types)
        return child_struct


    def get_mol_list(self, geo):
        ''' Returns input geometry grouped into arrays of molecules '''
        mol_list = [geo[x:x+self.num_atom_per_mol] 
                    for x in range(0, len(geo), self.num_atom_per_mol)]
        return mol_list

    def get_COM_frac(self, mol, lattice_vectors):
        ''' Returns array of COM for a given molecule '''
        atoms = []
        types = []
        for atom in mol:
            atoms.append([float(atom[0]), float(atom[1]), float(atom[2])])
            types.append(atom[3])
        molp = Molecule(types, atoms)
        COM = molp.center_of_mass #cartesian
        latinv = np.linalg.inv(lattice_vectors)
        frac_COM = np.dot(latinv, COM)
        return frac_COM

    def get_centered_molecule(self, mol):
        ''' Returns array of COM and centered geometry for each molecule '''
        atoms = []
        types = []
        centered_mol = []
        for atom in mol:
            atoms.append([float(atom[0]), float(atom[1]), float(atom[2])])
            types.append(atom[3])
        molp = Molecule(types, atoms)
        centered_molp = molp.get_centered_molecule()
        for site in centered_molp.sites:
            centered_mol.append(list(site._coords))
        return centered_mol, types

    def get_orientation_info(self):
        mol_list_a = self.get_mol_list(self.geometry_a)
        mol_list_b = self.get_mol_list(self.geometry_b)
        for mol in mol_list_a:
            COM = self.get_COM_frac(mol, self.latA)
            centered_mol, types = self.get_centered_molecule(mol)
            z, y, x, aligned_mol = self.get_orientation_angles(centered_mol, types)
            self.orientation_info_a.append([z, y, x, COM, aligned_mol])
        for mol in mol_list_b:
            COM = self.get_COM_frac(mol, self.latB)
            centered_mol, types = self.get_centered_molecule(mol)
            z, y, x, aligned_mol = self.get_orientation_angles(centered_mol, types)
            self.orientation_info_b.append([z, y, x, COM, aligned_mol])

    def get_orientation_angles(self, mol, types):
        ''' Computes the principal axes for each molecule and the corresponding
            rotation matrices.
            Returns: rotations about z, y, x axis and molecule with principal axes 
            aligned. '''
        atoms = []
        before = ""
        for atom in mol:
            atoms.append([float(atom[0]), float(atom[1]), float(atom[2])])

        #Visualize in shell for debugging
        #print "beforei\n"
        #for i in range(len(types)):
        #    st =  "atom " + str(mol[i][0])+" "+str(mol[i][1])+" "+str(mol[i][2])+" "+types[i]
        #    before += st + "\n"
        #print before

        molp = Molecule(types, atoms)
        centered_molp = molp.get_centered_molecule()

        #compute principal axes and eignvalues
        PGA = pga(centered_molp)
        ax1, ax2, ax3 = PGA.principal_axes
        eig1, eig2, eig3 = PGA.eigvals
        eigen_sys = [[eig1, ax1],[eig2, ax2],[eig3, ax3]]
        sort_eig = sorted(eigen_sys)

        #Construct rotation matrix and its inverse
        rot = np.column_stack((sort_eig[0][1], sort_eig[1][1], sort_eig[2][1]))
        rot_trans = np.linalg.inv(np.array(rot))

        #Aligned geometry
        centered_sites = []
        aligned_mol = []
        for site in centered_molp.sites:
            centered_sites.append(list(site._coords))
        for atom in centered_sites:
            new_atom = np.dot(rot_trans, np.array(atom))
            new_atom = [new_atom[0], new_atom[1], new_atom[2]]
            aligned_mol.append(new_atom)

        #Visualize
        #print "after"
        #atom_out = ""
        #for i in range(len(centered_sites)):
        #    st =  "atom " + str(aligned_mol[i][0])+" "\
        #    +str(aligned_mol[i][1])+" "+str(aligned_mol[i][2])+" "+types[i]
        #    atom_out += st + "\n"
        #self.output(atom_out)
        #print atom_out

        
        #Euler angles 
        z, y, x =  self.mat2euler(rot)
        return z, y, x, aligned_mol

    def combine_lattice_info(self, lattice_info_a, lattice_info_b):
        ''' combines a, b, c, alpha, beta, gamma from parent lattices '''
        child_lattice_info = []
        rand_vec = [random.uniform(0.25,0.75) for i in range(6)]
 
        for i in range(6):
            if i < 3:
                rand_scale = random.uniform(0.8,1.1)
                new_info = rand_scale * (rand_vec[i]*lattice_info_a[i] 
                                      + (1-rand_vec[i]) * lattice_info_b[i])
                child_lattice_info.append(new_info)
            elif i >= 3:
                rand_scale = random.uniform(0.97, 1.03)
                new_info = rand_scale * (rand_vec[i] * lattice_info_a[i] 
                                      + (1-rand_vec[i]) * lattice_info_b[i])
                if 88.5 <= new_info <= 91.5:
                    self.output("--Setting angle close to 90 degrees")
                    new_info = 90.0
                child_lattice_info.append(new_info)
        #debugging
        #child_lattice_info = lattice_info_a
        return child_lattice_info


    def combine_orientation_info(self, orientation_info_a, orientation_info_b): 
        '''Returns child orientation info in the form
                            info = [z, y, x, COM, centered_mol]'''
        self.output("Parent A orientation info: %s" % (orientation_info_a[0][:3]))
        self.output("Parent B orientation info: %s" % (orientation_info_b[0][:3]))
    
        # Choose which Parent's COM the child will inherit
        COM_choice = random.random()
        if COM_choice < 0.5:
            parent_a_COM = True
        else: 
            parent_a_COM = False

        # Choose which Parent's conformer the child will inherit
        #mol_geo_choice = random.random()
        #if mol_geo_choice < 0.5:
        #    parent_a_mol = True
        #else:
        #    parent_a_mol = False

        # Randomly permute indices of molecules paired for mating
        n = len(orientation_info_a); a = list(range(n))
        perm = [[a[i - j] for i in range(n)] for j in range(n)]
        p_i = np.random.choice(a)

        # Create child by combining Parents' orientation info
        orientation_info_child = []
        rand_cut = random.uniform(0.15,0.85)
        j = 0
        for i in perm[p_i]:
            child_z, child_y, child_x  = ( 
                np.array(orientation_info_a[j][:3])*rand_cut + 
                np.array(orientation_info_b[i][:3])*(1-rand_cut))
            if parent_a_COM: 
                COM = orientation_info_a[j][3]
            else:
                COM = orientation_info_b[i][3]
            #if parent_a_mol:
            centered_mol = orientation_info_a[j][4]
            #else:
            #    centered_mol = orientation_info_b[i][4]
            orientation_info_child.append([child_z, child_y, child_x, COM, centered_mol])   

        # Return Childs orientation info 
        self.output("Child orientation info: %s" % (orientation_info_child[0][:3]))
        return orientation_info_child

    def combine_orientation_info_dimers(self, orientation_info_a, orientation_info_b):
        '''Returns child orientation info in the form
                            info = [z, y, x, COM, centered_mol]'''
        self.output("Parent A orientation info: "+ str(orientation_info_a[0][:3]))
        self.output("Parent B orientation info: "+ str(orientation_info_b[0][:3]))


        choice_a = random.sample(set(range(self.num_mols)), 2)
        choice_b = random.sample(set(range(self.num_mols)), 2)

        orientation_info_child = []
        for i in range(len(orientation_info_a)):

            if i == choice_a[0]:
                child_z, child_y, child_x =(
                np.array(orientation_info_b[choice_b[0]][:3]))
                centered_mol = orientation_info_b[choice_b[0]][4]
                COM = orientation_info_a[i][3]
            elif i == choice_a[1]:
                child_z, child_y, child_x =(
                np.array(orientation_info_b[choice_b[1]][:3]))
                centered_mol = orientation_info_b[choice_b[1]][4]
                COM = orientation_info_a[i][3]
            else:
                child_z, child_y, child_x =(
                np.array(orientation_info_a[i][:3]))
                COM = orientation_info_a[i][3]
                centered_mol = orientation_info_a[i][4]
            orientation_info_child.append([child_z, child_y, child_x, COM, centered_mol])

        self.output("Child orientation info: " + str(orientation_info_child[0][:3]))
        return orientation_info_child
    def reconstruct_child(self, child_lattice_info, child_orientation_info):
        '''
        Reconstructs the child's atomic positions and lattice
        from its inherited genes 
        '''
        lattice= self.lattice_lower_triangular(self.lattice_from_info(child_lattice_info))
        A, B, C = lattice
        child_coordinates = []

        for mol_info in child_orientation_info:
            mol_coords = []
            z, y, x, COM, centered_mol = mol_info
            rot_from_euler = self.euler2mat(z, y, x)
            COM_xyz = np.dot(lattice, COM)
            for atom in centered_mol:
                mol_coords.append(np.dot(rot_from_euler, np.array(atom).reshape((3,1))).tolist())
            for coord in mol_coords:
                new_coords = [coord[0][0] + COM_xyz[0], coord[1][0] + COM_xyz[1], coord[2][0]+COM_xyz[2]]
                child_coordinates.append(new_coords)
        lattice = [A, B, C]
        return lattice, np.array(child_coordinates)

    def return_lattice_trans(self, a, b, c):
        lattice_vectors = np.transpose([a,b,c])
        latinv = np.linalg.inv(lattice_vectors)
        return latinv

    def random_orientation(self, struct):
        rand = random.random()
        a = struct.get_property("lattice_vector_a")
        b = struct.get_property("lattice_vector_b")
        c = struct.get_property("lattice_vector_c")
        old_lattice = [a, b, c]
        if rand < 0.5:
            i = 0
            choice = np.random.choice(["bca","cab"])
            choice = "cab"
            if choice == "cab":
                new_lattice = [c, a, b]
                trans_mat = np.dot(np.transpose(new_lattice),
                                   np.linalg.inv(np.transpose(old_lattice)))
                for atom in struct.geometry:
                    frac = struct.get_frac_data()[0][i]
                    new_frac = [frac[2], frac[0], frac[1]]
                    new_atom = np.dot(new_frac, new_lattice)
                    atom['x'], atom['y'], atom['z'] = new_atom
                    i = i +1
                struct.set_property("lattice_vector_a", c)
                struct.set_property("lattice_vector_b", a)
                struct.set_property("lattice_vector_c", b)
                structure_handling.move_molecule_in(struct, create_duplicate=False)
                struct = structure_handling.cell_lower_triangular(struct)
                self.output("-- cab orientation")
                self.output(struct.get_geometry_atom_format())
            elif choice == "bca":
                new_lattice = [b, c, a]
                trans_mat = np.dot(np.transpose(new_lattice),
                                   np.linalg.inv(np.transpose(old_lattice)))
                for atom in struct.geometry:
                    frac = struct.get_frac_data()[0][i]
                    new_frac = [frac[1], frac[2], frac[0]]
                    new_atom = np.dot(new_frac, new_lattice)
                    atom['x'], atom['y'], atom['z'] = new_atom
                    i = i +1
                struct.set_property("lattice_vector_a", b)
                struct.set_property("lattice_vector_b", c)
                struct.set_property("lattice_vector_c", a)
                structure_handling.move_molecule_in(struct, create_duplicate=False)
                struct = structure_handling.cell_lower_triangular(struct)
                self.output("-- bca orientation")
                self.output(struct.get_geometry_atom_format())
 
    def create_child_struct(self, child_geometry, child_lattice, atom_types):
        ''' Creates a Structure() for the child to be added to the collection'''
        child_A, child_B, child_C = child_lattice 
        temp_vol = np.dot(np.cross(child_A, child_B), child_C)
        alpha = self.angle(child_B, child_C)*180./np.pi
        beta = self.angle(child_C, child_A)*180./np.pi
        gamma = self.angle(child_A, child_B)*180./np.pi
        new_struct = Structure() 
        for i in range(len(child_geometry)):
            new_struct.build_geo_by_atom(float(child_geometry[i][0]), float(child_geometry[i][1]),
                                     float(child_geometry[i][2]), atom_types[i])
        new_struct.set_property('lattice_vector_a', child_A)
        new_struct.set_property('lattice_vector_b', child_B)
        new_struct.set_property('lattice_vector_c', child_C)
        new_struct.set_property('a', np.linalg.norm(child_A))
        new_struct.set_property('b', np.linalg.norm(child_B))
        new_struct.set_property('c', np.linalg.norm(child_C))
        new_struct.set_property('cell_vol', temp_vol)
  #      new_struct.set_property('crossover_type', cross_type)
        new_struct.set_property('alpha',alpha)
        new_struct.set_property('beta', beta)
        new_struct.set_property('gamma', gamma)
        return new_struct

    def lattice_lower_triangular(self, lattice):
        '''
        Returns a list of lattice vectors that corresponds to the 
        a, b, c, alpha, beta, gamma as specified by struct
        ! In row vector form !
        '''
        A, B, C = lattice
        a = np.linalg.norm(A)
        b = np.linalg.norm(B)
        c = np.linalg.norm(C)
        alpha = self.angle(B, C)
        beta = self.angle(C,A)
        gamma = self.angle(A, B)
        lattice[0][0] = a
        lattice[0][1] = 0; lattice[0][2] = 0
        lattice[1][0] = np.cos(gamma)*b
        lattice[1][1] = np.sin(gamma)*b
        lattice[1][2] = 0
        lattice[2][0] = np.cos(beta)*c
        lattice[2][1] = (b*c*np.cos(alpha) - lattice[1][0]*lattice[2][0])/lattice[1][1]
        lattice[2][2] = (c**2 - lattice[2][0]**2 - lattice[2][1]**2)**0.5
        return np.array(lattice)

    def lattice_from_info(self, lattice_info):
        a, b, c, alpha, beta, gamma = lattice_info 
        lattice = Lattice.from_parameters(a, 
                                          b, 
                                          c, 
                                          alpha,
                                          beta, 
                                          gamma)
        a, b, c = lattice._lengths
        alpha, beta, gamma = lattice._angles
        self.output(str(a)+" "+str(b)+" "+str(c)+" "+str(alpha)+" "+str(beta)+" "+str(gamma))
        return lattice.matrix


    def angle(self,v1,v2):
        numdot = np.dot(v1,v2)
        anglerad = np.arccos(numdot/(self.leng(v1)*self.leng(v2)))
        angledeg = anglerad*180/np.pi
        return (anglerad)

    def leng(self,v):
        length = np.linalg.norm(v)
        return length

    def output_child_properties(self, child):
        child_vecs = np.array(list(map(float, child.get_lattice_magnitudes())))
        child_angs = np.array(list(map(float, child.get_lattice_angles())))
        message = ('-- Child lattice vectors: ' + str(child_vecs).strip('[').strip(']') +
                 '\n-- Child lattice angles: ' + str(child_angs).strip('[').strip(']'))
        self.output(message)

    def output(self, message): output.local_message(message, self.replica)

    def mat2euler(self, M, cy_thresh=None):
        ''' 
        Discover Euler angle vector from 3x3 matrix

        Uses the following conventions.

        Parameters
        ----------
        M : array-like, shape (3,3)
        cy_thresh : None or scalar, optional
        threshold below which to give up on straightforward arctan for
        estimating x rotation.  If None (default), estimate from
        precision of input.

        Returns
        -------
        z : scalar
        y : scalar
        x : scalar
        Rotations in radians around z, y, x axes, respectively

        Notes
        -----
        If there was no numerical error, the routine could be derived using
        Sympy expression for z then y then x rotation matrix, which is::

        [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
        [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
        [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

        with the obvious derivations for z, y, and x

        z = atan2(-r12, r11)
        y = asin(r13)
        x = atan2(-r23, r33)

        Problems arise when cos(y) is close to zero, because both of::

        z = atan2(cos(y)*sin(z), cos(y)*cos(z))
        x = atan2(cos(y)*sin(x), cos(x)*cos(y))

        will be close to atan2(0, 0), and highly unstable.

        The ``cy`` fix for numerical instability below is from: *Graphics
        Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
        0123361559.  Specifically it comes from EulerAngles.c by Ken
        Shoemake, and deals with the case where cos(y) is close to zero:

        See: http://www.graphicsgems.org/

        The code appears to be licensed (from the website) as "can be used
        without restrictions".
        '''
        M = np.asarray(M)
        if cy_thresh is None:
            try:
                cy_thresh = np.finfo(M.dtype).eps * 4
            except ValueError:
                cy_thresh = _FLOAT_EPS_4
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
        cy = math.sqrt(r33 * r33 + r23 * r23)
        if cy > cy_thresh: # cos(y) not close to zero, standard form
            z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else: # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21,  r22)
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = 0.0
        return z, y, x

    def euler2mat(self, z=0, y=0, x=0):
        ''' Return matrix for rotations around z, y and x axes

        Uses the z, then y, then x convention above

        Parameters
        ----------
        z : scalar
        Rotation angle in radians around z-axis (performed first)
        y : scalar
        Rotation angle in radians around y-axis
        x : scalar
        Rotation angle in radians around x-axis (performed last)

        Returns
        -------
        M : array shape (3,3)
        Rotation matrix giving same rotation as for given angles

    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True

    The output rotation matrix is equal to the composition of the
    individual rotations

    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True

    You can specify rotations by named arguments

    >>> np.all(M3 == euler2mat(x=xrot))
    True

    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.

    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)

    Rotations are counter-clockwise.

    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True

    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
        '''
        Ms = []
        if z:
            cosz = math.cos(z)
            sinz = math.sin(z)
            Ms.append(np.array(
                 [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
        if y:
            cosy = math.cos(y)
            siny = math.sin(y)
            Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
        if x:
            cosx = math.cos(x)
            sinx = math.sin(x)
            Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
        if Ms:
            return reduce(np.dot, Ms[::-1])
        return np.eye(3)

def rotation_matrix(theta, psi, phi):
    Rxyz = np.matrix([ ((np.cos(theta) * np.cos(psi)),
                (-np.cos(phi) * np.sin(psi)) + (np.sin(phi) * np.sin(theta) * np.cos(psi)),
                (np.sin(phi) * np.sin(psi)) + (np.cos(phi) * np.sin(theta) * np.cos(psi))),

                ((np.cos(theta) * np.sin(psi)),
                (np.cos(phi) * np.cos(psi)) + (np.sin(phi) * np.sin(theta) * np.sin(psi)),
                (-np.sin(phi) * np.cos(psi)) + (np.cos(phi) * np.sin(theta) * np.sin(psi))),

                ((-np.sin(theta)),
                (np.sin(phi) * np.cos(theta)),
                (np.cos(phi) * np.cos(theta)))])
    return Rxyz
def output_parent_properties(parent_a, parent_b, replica):
    parent_a_vecs = np.array(list(map(float, parent_a.get_lattice_magnitudes())))
    parent_b_vecs = np.array(list(map(float, parent_b.get_lattice_magnitudes())))
    parent_a_angs = np.array(list(map(float, parent_a.get_lattice_angles())))
    parent_b_angs = np.array(list(map(float, parent_b.get_lattice_angles())))
    message = ('-- Parent A lattice vectors: ' + str(parent_a_vecs).strip('[').strip(']') +
               '\n-- Parent A lattice angles: ' + str(parent_a_angs).strip('[').strip(']') +
               '\n-- Parent B lattice vectors: ' + str(parent_b_vecs).strip('[').strip(']') +
               '\n-- Parent B lattice angles: ' + str(parent_b_angs).strip('[').strip(']'))
    output.local_message(message, replica)
