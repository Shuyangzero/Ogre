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
from utilities.misc import random_rotation_matrix
from utilities.space_group_utils import reduce_by_symmetry, rebuild_by_symmetry,\
                                        are_symmops_compatible

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
	Args: list of 2 Structures() to crossover, the replica name running the crossover instance.
	Returns: A single Structure() if crossover is successful or False if crossover fails 
	'''
	parent_a = list_of_structures[0]
	parent_b = list_of_structures[1]
	cross_obj = Symmetric_Crossover(parent_a,parent_b,replica)
	child_struct = cross_obj.crossover()
	return child_struct

def fix_method():
	ui = user_input.get_config()
	swap_sym = ui.get_eval(sn,"swap_sym_prob")
	crossover_methods = []
	if random.random() < swap_sym:
		crossover_methods.append("swap_sym")
	return crossover_method

class Symmetric_Crossover(object):
    '''
    Takes 2 parent structures and combines them via different crossover options.
    '''
    def __init__(self,parent_a,parent_b, replica):
        self.ui = user_input.get_config()
        sn = "crossover"
        self.swap_sym = self.ui.get_eval(sn,"swap_sym_prob")
        self.swap_sym_tol = self.ui.get_eval(sn,"swap_sym_tol")

        self.blend_lat = self.ui.get_eval(sn,"blend_lat_prob")
        self.blend_lat_tol = self.ui.get_eval(sn,"blend_lat_tol")
        self.blend_lat_cent = self.ui.get_eval(sn,"blend_lat_cent")
        self.blend_lat_std = self.ui.get_eval(sn,"blend_lat_std")
        self.blend_lat_ext = self.ui.get_boolean(sn,
                     "blend_lat_ext")

        self.blend_COM = self.ui.get_eval(sn,"blend_mol_COM_prob")
        self.blend_COM_cent = self.ui.get_eval(sn,"blend_mol_COM_cent")
        self.blend_COM_std = self.ui.get_eval(sn,"blend_mol_COM_std")
        self.blend_COM_ext = self.ui.get_boolean(sn,
                     "blend_mol_COM_ext")

        self.swap_geo = self.ui.get_eval(sn,"swap_mol_geo_prob")
        self.swap_geo_tol = self.ui.get_eval(sn,"swap_mol_geo_tol")
        self.swap_geo_attempts = \
        self.ui.get_eval(sn,"swap_mol_geo_orient_attempts")

        self.blend_orien = \
        self.ui.get_eval(sn,"blend_mol_orien_prob")
        self.blend_orien_cent = \
        self.ui.get_eval(sn,"blend_mol_orien_cent")
        self.blend_orien_std = \
        self.ui.get_eval(sn,"blend_mol_orien_std")
        self.blend_orien_ext = \
        self.ui.get_boolean(sn,"blend_mol_orien_ext")
        self.blend_orien_tol = \
        self.ui.get_eval(sn,"blend_mol_orien_tol")
        self.blend_orien_ref = \
        self.ui.get_eval(sn,"blend_mol_orien_ref_prob")
        self.blend_orien_attempts = \
        self.ui.get_eval(sn,"blend_mol_orien_orient_attempts")

        self.allow_no_cross = self.ui.get_boolean(sn,"allow_no_crossover")

        
        self.parent_a = structure_handling.cell_lower_triangular(parent_a)
        self.parent_b = structure_handling.cell_lower_triangular(parent_b)
        self.replica = replica
        self.verbose = self.ui.verbose()
        self.all_geo = self.ui.all_geo()
        self.num_mol = self.ui.get_eval("run_settings","num_molecules")
        self.napm = int(len(parent_a.geometry)/self.num_mol)

    def crossover(self):
        '''
        Main crossover procedure
        '''
        self.parent_a_reduced = reduce_by_symmetry(self.parent_a,
							   create_duplicate=True)
        if len(self.parent_a_reduced.geometry) % self.napm != 0:
            message = "-- Symmetry reduction of parent a failed"
            output.local_message(message,self.replica)
            return False
        
        if self.verbose:
            output.local_message("--Parent a symmetry operations--",
					     self.replica)
            output.local_message("\n".join(map(str,
            self.parent_a_reduced.properties["symmetry_operations"])),
					     self.replica)

        if self.verbose and self.all_geo:
            output.local_message("--Parent a reduced geometry--", self.replica)
            output.local_message(self.parent_a_reduced.get_geometry_atom_format(), self.replica)

        #Check to avoid deadloop
        if not self.check_probs():
            raise ValueError("All symmetric crossover probability set to extremely small, but allow_no_crossover is not set to TRUE; exiting due to concern of deadloop")

        while True:
            operations = []
            self.child_struct = False
			
            #Reduce parent a by reference to reduced parent b
			#If swap_sym is called
            if self.allow_operation(self.swap_sym):
                success = self.reduce_by_reference()
                if success:
                    operations.append("swap_symmetry")
                else:
                    self.swap_sym = 0
                    if not self.check_probs():
                        return False

            #Use symmetrically reduced parent a
			#When swap_sym is not called
            if self.child_struct == False:
                self.child_struct = self.parent_a_reduced

            ############# Blend Lattices #################
            if self.allow_operation(self.blend_lat):
                success = self.blend_lattice_vectors()
                if not success:
                    self.blend_lat = 0
                    if not self.check_probs():
                        return False
                else:
                    operations.append("blend_lattices")

            ############# Blend Molecule COM #############
            if self.allow_operation(self.blend_COM):
                self.blend_molecule_COM()
                operations.append("blend_molecule_COM")

            ############# Swap Molecule Geometry #########
            if self.allow_operation(self.swap_geo):
                success = self.swap_molecule_geometry()
                if not success:
                    self.swap_geo = 0
                    if not self.check_probs():
                        return False
                else:
                    operations.append("swap_molecule_geometry")

            ############# Blend Molecule Orientation #####
            if self.allow_operation(self.blend_orien):
                self.blend_molecule_orientation()
                operations.append("blend_molecule_orientation")

            ############# Post Processing ################
            if self.allow_no_cross or len(operations) > 0:
                if self.verbose and self.all_geo: 
                    message = "-- Child reduced --\n"
                    message += self.child_struct.get_geometry_atom_format()
                    message += "-- Child sym operations for reconstruction\n"
                    message +="\n".join(map(str, self.child_struct.properties["symmetry_operations"]))
                    output.local_message(message, self.replica)

                final_child_struct = rebuild_by_symmetry(self.child_struct,
                                                         napm = self.napm,
                                                         create_duplicate=True)
                if final_child_struct == False:
                    message = "-- Reconstruction of child failed"
                    output.local_message(message, self.replica)
                    return False

                if final_child_struct.geometry.size != self.parent_a.geometry.size:
                    message = "-- Reconstructed child structure has smaller Z than parents"
                    output.local_message(message, self.replica)
                    return False

                message = "-- Summary of Crossover --\n"
                if len(operations) == 0:
                    message += "No crossover operation called"
                else:
                    message += "Crossover operation(s) called: "
                    message += " ".join(map(str,operations))

                output.local_message(message,self.replica)
                final_child_struct.set_property("crossover_type", " ".join(map(str,operations)))

                return final_child_struct

    def reduce_by_reference(self):
        '''
        Reduce the parent_b by symmetry, 
        and reduce parent_a by selecting molecules
        closest to the seed molecules of parent_b
        '''
        ref_struct = reduce_by_symmetry(self.parent_b,
						create_duplicate=True)
        
        if not are_symmops_compatible(self.parent_a.get_lattice_vectors(),
                    ref_struct.properties["symmetry_operations"],
                    self.swap_sym_tol):
            message = "-- Symmetry of parent b incomptaible with "
            message += "parent a's lattice vectors"
            output.local_message(message,self.replica)
            return False
        if len(ref_struct.geometry) % self.napm != 0:
            #Reduction failure; potential overlap 
            message = "-- Symmetry reduction of parent b failed"
            output.local_message(message,self.replica)
            return False

        selected = [False for x in range(self.num_mol)]
        lat_mat = np.transpose(ref_struct.get_lattice_vectors())
        trans = ([[x,y,z] for z in range(-2,3) 
				  for y in range(-2,3)
				  for x in range(-2,3)])
        napm = self.napm
        coms = [structure_handling.cm_calculation(self.parent_a,
                list(range(x*napm,(x+1)*napm)))\
                for x in range(self.num_mol)]

        #Select molecules from parent_a 
		#closest to the seed molecles of parent_b
		#Translation by parent_b's lattice vectors
        for i in range(int(len(ref_struct.geometry)/self.napm)):
            min_dist = None; choose = None
            ref_com = structure_handling.cm_calculation(
                ref_struct,list(range(i*self.napm,(i+1)*self.napm)))

            for j in [x for x in range(self.num_mol) if not selected[x]]:
                ref_coms = [np.add(ref_com,
                            np.dot(lat_mat,t))\
                            for t in trans]
                diffs = [np.subtract(coms[i],x)\
                    for x in ref_coms]
                dist = min([np.linalg.norm(x) for x in diffs])
                if min_dist == None or dist < min_dist:
                    choose = j
                    min_dist = dist
            selected[choose] = True

        #Construct reduced geometry from parent_a
        struct = Structure()
        struct.set_lattice_vectors(self.parent_a.get_lattice_vectors())
        geo = self.parent_a.geometry
        for i in [x for x in range(self.num_mol) if selected[x]]:
            for j in range(i*napm,(i+1)*napm):
                struct.build_geo_by_atom(*geo[j])
        
        struct.properties["symmetry_operations"] = \
        ref_struct.properties["symmetry_operations"]
        self.child_struct = struct

        if self.verbose:
            message = "\n--Symmetry is swapped; new operations:\n"
            message += "\n".join(map(str,
                        self.child_struct.properties["symmetry_operations"]))
            output.local_message(message,self.replica)

        if self.verbose and self.all_geo:
            output.local_message("--New reduced geometry--",
					     self.replica)
            output.local_message(self.child_struct.get_geometry_atom_format(),
                                 self.replica)
        return True

    def blend_lattice_vectors(self):
        lata = self.child_struct.get_lattice_vectors()
        latb = self.parent_b.get_lattice_vectors()
        symmops = self.child_struct.properties["symmetry_operations"]
        for i in range (5):
            #5 times because some combination may not be valid
            b = self.get_blending_parameter(self.blend_lat_cent,
							self.blend_lat_std,
							self.blend_lat_ext)
            new_lats = np.add(np.multiply(lata,(1-b)),
					  np.multiply(latb,b))
            if not are_symmops_compatible(new_lats,
						      symmops,
						      self.blend_lat_tol):
                continue
            self.child_struct.reset_lattice_vectors(new_lats)
            if self.verbose:
                message = "\n-- Lattices are blended; "
                message += "blending parameter: " + str(b)
                output.local_message(message,self.replica)
            return True

        if self.verbose:
            message = "\n-- Lattice blending is not valid for this pair"
            output.local_message(message,self.replica)
        return False

    def blend_molecule_COM(self):
        reduced_nmpc = int(self.child_struct.get_n_atoms()/self.napm)
        
        if self.verbose:
            message = "\n-- Molecule COM's are blended"
            output.local_message(message,self.replica)
        for i in range (reduced_nmpc):
            COM = structure_handling.cm_calculation(self.child_struct,
				list(range(i*self.napm,(i+1)*self.napm)))
            com = self.pair_molecule(i,False)
            
            b = self.get_blending_parameter(self.blend_COM_cent,
							self.blend_COM_std,
							self.blend_COM_ext)
            new_COM = np.add(np.multiply(COM,(1-b)),
                      np.multiply(com,b))
            trans = np.subtract(new_COM,COM)
            structure_handling.mole_translation(self.child_struct,
							    i,
							    self.napm,
							    trans,
							    create_duplicate=False)
            if self.verbose:
                message = "------ Molecule No. %i \n" % i
                message += "-- Blending parameter: %f \n" % b
                message += "-- Original COM:        " + \
                       ", ".join(map(str,COM))
                message += "\n-- Paired molecule COM: " + \
                       ", ".join(map(str,com))
                message += "\n-- Final COM:           " + \
                       ", ".join(map(str,new_COM))
                message += "\n-- Translation vector:  " + \
                       ", ".join(map(str,trans))
                output.local_message(message,self.replica)

    def swap_molecule_geometry(self):
        reduced_nmpc = int(self.child_struct.get_n_atoms()/self.napm)
        success = False
        for i in range (reduced_nmpc):
            COM = structure_handling.cm_calculation(self.child_struct,
                list(range(i*self.napm,(i+1)*self.napm)))
            paired = self.pair_molecule(i,True)
            com = structure_handling.cm_calculation(paired,
                list(range(0,self.napm)))
            structure_handling.cell_translation(paired, 
                                [-x for x in com],
                                create_duplicate=False)
            result = structure_handling.mole_get_orientation(
                    self.child_struct,
                    list(range(i*self.napm,(i+1)*self.napm)),
                    paired.geometry,
                    COM,
                    self.swap_geo_tol,
                    create_duplicate=True)
            if result != False:
            #The two molecules have similar geometry
            #Aborting swap geometry
                continue
            
            success = True
            mol = get_molecule(self.child_struct, self.napm, i)
            structure_handling.cell_translation(mol, 
							    [-x for x in COM],
							    create_duplicate=False)

            #Find an orientation of the second geometry
            #That has a small residual with the first
            adjusted = get_closest_orientation(paired,
                               mol,
                               self.swap_geo_attempts)

            #Move the COM of the adjusted geometry to original COM
            structure_handling.cell_translation(adjusted,
                                COM,
                                create_duplicate=False)
            for x in range (self.napm):
            #Replace the old geometry with new one)
                self.child_struct.geometry[i*self.napm+x] = \
                deepcopy(adjusted.geometry[x])
        if self.verbose:
            message = "-- Molecule %i geometry swapped" % i
            output.local_message(message,self.replica)
        
        if success and self.all_geo and self.verbose:
            message = "--Child structure after swap_molecule_geometry--\n"
            message+= self.child_struct.get_geometry_atom_format()
            output.local_message(message,self.replica)

        if not success and self.verbose:
            message = "-- swap_molecule_geometry not valid; "
            message += "all molecule geometries identical"
            output.local_message(message,self.replica)
        
        return success	

    def blend_molecule_orientation(self):

        if self.verbose:
            message = "\n-- Molecule orientations are blended"
            output.local_message(message,self.replica)

        reduced_nmpc = int(self.child_struct.get_n_atoms()/self.napm)
        
        for i in range (reduced_nmpc):
            COM = structure_handling.cm_calculation(self.child_struct,
                list(range(i*self.napm,(i+1)*self.napm)))
            mol = get_molecule(self.child_struct, self.napm, i)
            structure_handling.cell_translation(mol, 
                                [-x for x in COM],
                                create_duplicate=False)

            paired = self.pair_molecule(i,True)
            com = structure_handling.cm_calculation(paired,
                list(range(0,self.napm)))
            structure_handling.cell_translation(paired, 
                                [-x for x in com],
                                create_duplicate=False)

            mapping = structure_handling.mole_get_orientation(
                    self.child_struct,
                    list(range(i*self.napm,(i+1)*self.napm)),
                    paired.geometry,
                    COM,
                    tol = self.blend_orien_tol,
                    create_duplicate=True)

            v = [[5,0,0],[0,5,0],[0,0,5]]
            mol.reset_lattice_vectors(v)
            paired.reset_lattice_vectors(v)

            if mapping == False:
            #Geometry different
                allow_ref = \
                self.allow_operation(self.blend_orien_ref)
                message = "------ Molecule No. %i\n" % i
                message += "-- Different from paired molecule\n"
                message += "-- Blind orientation blend; "
                if allow_ref:
                    message += "reflection allowed"
                else:
                    message += "reflection not allowed"
                output.local_message(message,self.replica)

                result = self.blend_orien_blind(mol,
                                paired,
                                allow_ref)

            else:
                allow_ref = \
                self.allow_operation(self.blend_orien_ref)
                message = "------ Molecule No. %i\n" % i
                message = "-- Same geometry from paired molecule\n"
                if not allow_ref and mapping[0]:
                #Mirror reflection is involved
                #But reflection is not alloweed for this blend
                    message += "-- Reflection not used;  "
                    message += "using blind orientation blend" 
                    if self.verbose:
                        output.local_message(message,
                                     self.replica)

                    result = self.blend_orien_blind(mol,
                                    paired,
                                    False)
                else:
                    message += "-- Blending orientation "
                    message += "based on found transfommation"
                    if self.verbose:
                        output.local_message(message,
                                     self.replica)
                    result = self.blend_orientation(mol,
                                    paired,
                                    mapping)
                    if result == False:
                        result = self.blend_orien_blind(
                            mol, paired, allow_ref)

            if self.all_geo and self.verbose:
                v = [[5,0,0],[0,5,0],[0,0,5]]
                mol.reset_lattice_vectors(v)
                paired.reset_lattice_vectors(v)
                result.reset_lattice_vectors(v)
                message = "-- Original orientation --\n"
                message += mol.get_geometry_atom_format()
                message += "-- Paired molecule orientation --\n"
                message += paired.get_geometry_atom_format()
                message += "-- New orientation --\n"
                message += result.get_geometry_atom_format()
                output.local_message(message,self.replica)

            #Move the COM of the adjusted geometry to original COM
            structure_handling.cell_translation(result,
                                COM,
                                create_duplicate=False)

            for x in range (self.napm):
            #Replace the old geometry with new one)
                self.child_struct.geometry[i*self.napm+x] = \
                deepcopy(result.geometry[x])


    def blend_orien_blind(self, s1, s2, allow_reflection):
        '''
        Blends the orientation of s1 and s2
        by applying random rotation to s1
        and finding the smallest weighted residual average
        '''
        if s1.get_n_atoms() != s2.get_n_atoms():
            raise ValueError("Two structures do not have same amount of atoms")
        
        min_resi = None
        b = self.get_blending_parameter(self.blend_orien_cent,
                        self.blend_orien_std,
                        self.blend_orien_ext)
        if self.verbose:
            message = "-- Blind blending parameter: " + str(b)
            output.local_message(message,self.replica)

        for i in range (self.blend_orien_attempts):
            rot = random_rotation_matrix()
            k = structure_handling.\
                 cell_transform_mat(s1,
                        rot,
                        create_duplicate=True)

            if allow_reflection and self.allow_operation(0.5):
            #Half of the time use reflection
                structure_handling.\
                    cell_reflection_z(k,
                              create_duplicate=True)

            #Now calculate the coordinates residual
            resi = 0
            for i in range(s1.get_n_atoms()):
                resi += b*np.linalg.norm([k.geometry[i][x]
                            -s2.geometry[i][x]
                            for x in range(3)])
                resi += (1-b)*np.linalg.norm([k.geometry[i][x]
                            -s1.geometry[i][x]
                            for x in range(3)])

            if min_resi == None or resi < min_resi:
                min_resi = resi
                best = k
        return best

    def blend_orientation(self, s1, s2, transformation):
        '''
        Blend s1 and s2 by applying partial of the transformation provided
        '''
        if transformation[0]:
        #A mirror reflection is involved
            result = structure_handling.\
                 cell_reflection_z(s1,create_duplicate=True)

            #Returns reflected molecule to close to original
            result = get_closest_orientation(result,
                             s1,
                             self.blend_orien_attempts)

            result.reset_lattice_vectors([[5,0,0],[0,5,0],[0,0,5]])


            transformation = structure_handling.\
            mole_get_orientation(result,
                         list(range(result.get_n_atoms())),
                         s2.geometry,
                         tol=self.blend_orien_tol,
                         create_duplicate=True)
            message = "-- Mirror reflection applied first\n"
        else:
            result = deepcopy(s1)
            message = "-- No mirror reflection\n"

        b = self.get_blending_parameter(self.blend_orien_cent,
                        self.blend_orien_std,
                        self.blend_orien_ext)
        if transformation == False:
            message += "-- After mirror reflection, mapping failed\n"
            message += "-- Proceeds to blind blending"
            if self.verbose:
                output.local_message(message,self.replica)
            return False

        vec = transformation[1:4]
        ang = transformation[4] * b
        structure_handling.cell_rotation(result, 
                         vec=vec,
                         deg=ang,
                         create_duplicate=False)
        if self.verbose:
            message += "-- Blending parameter: " + str(b) + "\n"
            message += "-- Rotation vector: " + " ".join(map(str,vec))
            message += "\n-- Rotation degrees: " + str(ang) 
            output.local_message(message,self.replica)

        return result


    def pair_molecule(self,mol_index,return_molecule=True):
        '''
        Pairs up the mol_index'th molecule in self.child_struct
        With a molecule in parent b whose COM is closest
        Note: Indexing starts from 0

        if return_molecule=True,
        Returns a Structure() object of the molecule

        if False, returns the COM of the molecule
        '''
        COM = structure_handling.cm_calculation(self.child_struct,
            list(range(mol_index*self.napm,(mol_index+1)*self.napm)))

        nmpc = int(self.parent_b.get_n_atoms()/self.napm)
        trans = ([[x,y,z] for z in range(-2,3) 
                  for y in range(-2,3)
                  for x in range(-2,3)])
        
        min_dist = None
        lat_mat = np.transpose(self.parent_b.get_lattice_vectors())
        for i in range (nmpc):
            com = structure_handling.cm_calculation(self.parent_b,
                list(range(i*self.napm,(i+1)*self.napm)))

            for t in trans:
                new_com = np.add(com,np.dot(lat_mat,t))
                diff = np.subtract(COM,new_com)
                dist = np.linalg.norm(diff)
                if min_dist == None or dist < min_dist:
                    ind = i
                    tt = t[:]
                    result = new_com[:]
                    min_dist = dist
        
        if not return_molecule:
            return result
        mol = get_molecule(self.parent_b, self.napm, ind)
        structure_handling.cell_translation(mol, 
                            np.dot(lat_mat,tt),
                            create_duplicate=False)
        return mol

    def get_blending_parameter(self,cent,std,ext):
        if ext:
            return np.random.normal(loc=cent,scale=std)
        while True:
            k = np.random.normal(loc=cent,scale=std)
            if (k>=0) and (k<=1):
                return k

    def check_probs(self):
        '''
        Check that not all probabilities are too small
        '''
        probs = ([self.swap_sym, self.blend_lat, self.blend_COM,
              self.swap_geo, self.blend_orien])
        if max(probs)<0.005 and not self.allow_no_cross:
            return False
        else:
            return True
    
    def allow_operation(self,prob):
        if random.random() < prob:
            return True
        else:
            return False

def get_molecule(struct, napm, mol_index):
	'''
	Returns the mol_index'th molecule in the structure
	as a Structure() object
	'''
	result = Structure()
	result.properties = deepcopy(struct.properties)
	for i in range (mol_index*napm, mol_index*napm+napm):
		result.build_geo_by_atom(*struct.geometry[i])
	return result


def get_closest_orientation(s1, s2, attempts=20):
	'''
	Conducts a number of random rotations (attempts) on s1
	Returns the one rotation with the smallest residual with s2
	'''
	if s1.get_n_atoms() != s2.get_n_atoms():
		raise ValueError("Two structures do not have same amount of atoms")

	min_resi = None
	for i in range (attempts):
		rot = random_rotation_matrix()
		k = structure_handling.cell_transform_mat(s1,
							  rot,
							  create_duplicate=True)

		#Now calculate the coordinates residual
		resi = 0
		for i in range(s1.get_n_atoms()):
			resi += np.linalg.norm([k.geometry[i][x]-s2.geometry[i][x]
						for x in range(3)])
		if min_resi == None or resi < min_resi:
			min_resi = resi
			best = k

	return best

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
