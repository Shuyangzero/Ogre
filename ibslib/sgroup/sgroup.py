"""
If any part of this module is used for a publication please cite:

X. Li, F. Curtis, T. Rose, C. Schober, A. Vazquez-Mayagoitia, K. Reuter,
H. Oberhofer, and N. Marom "Genarris: Random Generation of Molecular Crystal
Structures and Fast Screening with a Harris Approximation, ",
J. Chem. Phys., DOI: 10.1063/1.5014038; arXiv 1803.02145 (2018)
"""
"""
Created on Wed Apr 01 14:55:55 2015

@author: Patrick Kilecdi
This module stores information for the space group
sgroup.wycgen generate a random fractional coordinate sitting on a Wyckoff position
sgroup.wycarr produces a list of Wyckoff position arrangements that add up to nmpc of molecules per cell
"""
from ibslib.sgroup.spglib_database import spglib_database
from spglib import get_symmetry_from_database
import random, glob, os, json
import numpy as np

__author__ = "Xiayue Li, Timothy Rose, Christoph Schober, and Farren Curtis"
__copyright__ = "Copyright 2018, Carnegie Mellon University and "+\
                "Fritz-Haber-Institut der Max-Planck-Gessellschaft"
__credits__ = ["Xiayue Li", "Luca Ghiringhelli", "Farren Curtis", "Tim Rose",
               "Christoph Schober", "Alvaro Vazquez-Mayagoita",
               "Karsten Reuter", "Harald Oberhofer", "Noa Marom"]
__license__ = "BSD-3"
__version__ = "1.0"
__maintainer__ = "Timothy Rose"
__email__ = "trose@andrew.cmu.edu"
__url__ = "http://www.noamarom.com"


setting={0:random.random(),1:0.25,2:0.75}
enantiomorphic=set([3,4,5,6,7,8,9,10,11,12,13,14,15])

MAX_AVAILABLE_SPACE_GROUP = 230
CHIRAL_SPACE_GROUPS = \
    ([1] + list(range(3,6)) + list(range(16,25)) + list(range(75,81))
    + list(range(89,99)) + list(range(143,147)) + list(range(149,156))
    + list(range(168,174)) + list(range(177,183)) + list(range(195,200))
    + list(range(207,215)))
RACEMIC_SPACE_GROUPS = \
    [x for x in range(1, 231) if not x in CHIRAL_SPACE_GROUPS]

class SpaceGroupManager():
    def __init__(
        self, nmpc, is_chiral, is_racemic, wyckoff_list=[0],
        space_groups_allowed=None):
        '''
        nmpc: Number of molecule per cell; allowed space group must have Wyckoff
            position combo that can match nmpc
        is_chiral: Allowed space groups must be either chiral or racemic
        wyckoff_list: Constrain the Wyckoff position selection.
            Default [0], molecules must be placed on a general Wyckoff position.
            Set this to None to allow any wyckoff position combo
        space_groups_allowed: A list of int with allowed space groups; will be
            further pruned to be compatible with the above

        Raises ValueError if no valid space groups are allowed
        '''
        self._maximum_space_group = MAX_AVAILABLE_SPACE_GROUP
        self._chiral_space_groups = CHIRAL_SPACE_GROUPS
        self._racemic_space_groups = RACEMIC_SPACE_GROUPS
        self._nmpc = nmpc
        self._is_chiral = is_chiral
        self._is_racemic = is_racemic
        self._wyckoff_list = wyckoff_list
        self._space_groups_allowed = space_groups_allowed
        self._deduce_allowed_space_groups()

        self._space_group = None
        self._space_group_selected_counter = {}
        self._space_group_user_counter = {}
        for sg in range(1, MAX_AVAILABLE_SPACE_GROUP + 1):
            self._space_group_selected_counter[sg] = 0
            self._space_group_user_counter[sg] = 0

        # For keeping counts of the spacegroups in the structure pool quickly
        self.structures_in_pool = {}
        self.sg_in_pool = [0 for x in range(len(self._space_group_range))]
        self.sg_lookup_dict = {}
        # Dictionary to convert a space group to an index for sg_in_pool
        for i,space_group in enumerate(self._space_group_range):
            self.sg_lookup_dict[space_group] = i

    def _deduce_allowed_space_groups(self):
        space_group_range = []
        if self._space_groups_allowed != None:
            space_group_range = self._space_groups_allowed
        else:
            space_group_range = range(0, self._maximum_space_group + 1)

            if self._is_chiral:
                space_group_range = \
                    [sg for sg in space_group_range
                        if sg in self._chiral_space_groups]
            elif self._is_racemic:
                space_group_range = \
                    [sg for sg in space_group_range
                        if sg in self._racemic_space_groups]
            else:
                # No limitations placed on space groups
                space_group_range = \
                    [sg for sg in range(1,MAX_AVAILABLE_SPACE_GROUP+1)]

        self._space_group_range = []
        if (self._wyckoff_list != None):
            self._is_wyckoff_list_fixed = True
            for sg in space_group_range:
                space_group = Sgroup(sg)
                if (space_group.wyckoff_counter(self._wyckoff_list)
                    == self._nmpc):
                    self._space_group_range.append(sg)
        else:
            self._is_wyckoff_list_fixed = False
            for sg in space_group_range:
                space_group = Sgroup(sg)
                if sg.wyckoff_preparation(self._nmpc):
                    self._space_group_range.append(sg)

#        if len(self._space_group_range) == 0:
#            raise ValueError(
#                "No available space group matches the requirement of nmpc=%i,"
#                " is_chiral=%s, wyckoff_list=%s, space_groups_allowed"
#                "=%s" % (self._nmpc, str(self.is_chiral),
#                         str(self._wyckoff_list),
#                         str(self._space_groups_allowed)))
#        print_time_log("Space group range from input: " +
#                " ".join(map(str, self._space_group_range)))

    def get_space_group_uniformly(self, output_dir):
        '''
        output_dir: str
            path to structures being generated and outputted.
        return: int
            space group to try next.
        Purpose: Get an under-represented space group number in the current
            pool to promote uniform sampling of space groups.
        '''
        #First, let there be a small probability of selecting a random
        # space group rather than making sure a least-represented
        # space group is selected
        if random.random() < 0.10:
            self._space_group = self.get_space_group_randomly()
            return self._space_group

        #Get the space group distribution so far in the pool.
        struct_flist = glob.glob(os.path.join(output_dir, '*.json'))
        #Get list of space groups so far in the pool.
        sg_list_in_pool = []
        for struct_file in struct_flist:
            with open(struct_file) as f:
                struct_data = json.load(f)
                sg_list_in_pool.append(int(struct_data['properties']['space_group']))
        #Get counts of each space group
        sg_counts_list = [sg_list_in_pool.count(sg_allowed) for sg_allowed in self._space_group_range]
        #Choose randomly from one of the least represented space groups.
        least_represented_sg_count = min(sg_counts_list)
        indicies_of_least_represented_sgs = [i for i, count in enumerate(sg_counts_list) if count == least_represented_sg_count]
        sg_group_range_index = indicies_of_least_represented_sgs[random.randint(0, len(indicies_of_least_represented_sgs) - 1)]
        sg = self._space_group_range[sg_group_range_index]

        self._space_group = Sgroup(sg)
        self._space_group.wyckoff_preparation(self._nmpc)
        if not self._is_wyckoff_list_fixed:
            self._wyckoff_list = \
                self._space_group.wyckoff_selection(self._nmpc)

        self._space_group_selected_counter[sg] += 1
        return self._space_group

    def get_space_group_uniformly_manny(self, output_dir, 
                                        struct_list=None):
        '''
        Saves structures it has already seen in a dictionary for fast lookup
            and saves their space groups to a ordered list
        '''

        if random.random() < 0.01:
            self._space_group = self.get_space_group_randomly()
            return self._space_group

        # start_time = time.time()
        if not struct_list:
             struct_list = glob.glob(os.path.join(output_dir, '*.json'))
        
        for struct_file in struct_list:
            if struct_file not in self.structures_in_pool:
                self.structures_in_pool[struct_file] = {}
                struct_file_path = os.path.join(output_dir,struct_file)
                with open(struct_file_path,'r') as f:
                    struct_dict = json.load(f)
                    sg = int(struct_dict['properties']['space_group'])
                index = self.sg_lookup_dict[sg] 
                self.sg_in_pool[index] += 1

        
        # np.where returns tuple
        least_represented_index = np.where(self.sg_in_pool == 
                                           np.min(self.sg_in_pool))
        next_sg_index = np.random.choice(least_represented_index[0])
        
        next_sg = self._space_group_range[next_sg_index]
        
        self._space_group = Sgroup(next_sg)
        self._space_group.wyckoff_preparation(self._nmpc)
        if not self._is_wyckoff_list_fixed:
            self._wyckoff_list = \
                self._space_group.wyckoff_selection(self._nmpc)
        
        # end_time = time.time()
        
        # test = os.listdir(output_dir)
        # print_time_log('Getting uniformly manny has ended by selecting {}. '
        #               'It took this much time: {} '
        #               'and it loaded this many files: {} '
        #               'from this directory: {}'
        #      .format(next_sg, end_time-start_time, len(test), output_dir))
        
        self._space_group_selected_counter[next_sg] += 1
        return self._space_group


    def get_space_group_randomly(self):
        sg = self._space_group_range[
            int(random.uniform(0, len(self._space_group_range)))]

        self._space_group = Sgroup(sg)
        self._space_group.wyckoff_preparation(self._nmpc)
        if not self._is_wyckoff_list_fixed:
            self._wyckoff_list = \
                self._space_group.wyckoff_selection(self._nmpc)

        self._space_group_selected_counter[sg] += 1
        return self._space_group

    def get_wyckoff_list_randomly(self):
        if not self._is_wyckoff_list_fixed:
            self._wyckoff_list = \
                self._space_group.wyckoff_selection(self._nmpc)

        return self._wyckoff_list

    def increment_space_group_counter(self):
        self._space_group_user_counter[
            self._space_group.space_group_number] += 1

class Sgroup(object):

    '''
    Wyckoff.csv was parsed to generate spglib_database.py, a Python dictionary
    recognizing all 230 space group coordinate settings, tabulated by 530 Hall
    number keys. Provided a given space group number as an object of Sgroup,
    the appropriate Hall number is indexed from a list of predetermined hall
    number selections for all 230 space groups, converted to a string, and
    used to access its Hall number settings from spglib_database.py.
    '''

    def __init__(self, space_group_number):
        self.spacegroup_to_hall_number = \
        [1,   2,   3,   6,   9,  18,  21,  30,  39,  57,
         60,  63,  72,  81,  90, 108, 109, 112, 115, 116,
         119, 122, 123, 124, 125, 128, 134, 137, 143, 149,
         155, 161, 164, 170, 173, 176, 182, 185, 191, 197,
         203, 209, 212, 215, 218, 221, 227, 228, 230, 233,
         239, 245, 251, 257, 263, 266, 269, 275, 278, 284,
         290, 292, 298, 304, 310, 313, 316, 322, 334, 335,
         337, 338, 341, 343, 349, 350, 351, 352, 353, 354,
         355, 356, 357, 358, 359, 361, 363, 364, 366, 367,
         368, 369, 370, 371, 372, 373, 374, 375, 376, 377,
         378, 379, 380, 381, 382, 383, 384, 385, 386, 387,
         388, 389, 390, 391, 392, 393, 394, 395, 396, 397,
         398, 399, 400, 401, 402, 404, 406, 407, 408, 410,
         412, 413, 414, 416, 418, 419, 420, 422, 424, 425,
         426, 428, 430, 431, 432, 433, 435, 436, 438, 439,
         440, 441, 442, 443, 444, 446, 447, 448, 449, 450,
         452, 454, 455, 456, 457, 458, 460, 462, 463, 464,
         465, 466, 467, 468, 469, 470, 471, 472, 473, 474,
         475, 476, 477, 478, 479, 480, 481, 482, 483, 484,
         485, 486, 487, 488, 489, 490, 491, 492, 493, 494,
         495, 497, 498, 500, 501, 502, 503, 504, 505, 506,
         507, 508, 509, 510, 511, 512, 513, 514, 515, 516,
         517, 518, 520, 521, 523, 524, 525, 527, 529, 530]
        self.space_group_number = space_group_number
        self.hall_number_str = self.get_hall_number_str()
        self.hall_number_int = self.get_hall_number_int()
        self.name = spglib_database[self.hall_number_str]['Space Group']
        self.multiplicity = spglib_database[self.hall_number_str]['Multiplicity']
        self.site_symmetries = spglib_database[self.hall_number_str]['Site Symmetry']
        self.coordinates = spglib_database[self.hall_number_str]['Coordinates']
        self.op = get_symmetry_from_database(self.hall_number_int)['rotations']
        self.trans = get_symmetry_from_database(self.hall_number_int)['translations']
        self.unique_site_symmetries = self.get_uss()
        self.wtran = self.get_wtran()
        self.wmult = self.get_wmult()
        self.wop = self.get_wop()
        self.nwyc = self.get_nwyc()
        self.sname = self.unique_site_symmetries
        self.swycn = self.get_swycn()
        self.swyc_mult = self.get_swyc_mult()
        self.nsg = self.get_nsg()
        self.mlist = []
        self.new_wop = []
        self.indices = []
        self.new_wtran = []
        self.selected_positions()
        self.bravais_system_type_list = [None]*230
        # 2 triclinc systems
        for i in range(2):
            self.bravais_system_type_list[i] = 'Triclinic'
        for j in range(2,15):
        # 13 monoclinic systems
            self.bravais_system_type_list[j] = 'Monoclinic'
        # 59 orthorhombic systems
        for k in range(15,74):
            self.bravais_system_type_list[k] = 'Orthorhombic'
        # 68 tetragonal systems
        for l in range(74,142):
            self.bravais_system_type_list[l] = 'Tetragonal'
        # 25 trigonal systems
        for m in range(142,167):
            self.bravais_system_type_list[m] = 'Trigonal'
        # 27 hexagonal systems
        for n in range(167,194):
            self.bravais_system_type_list[n] = 'Hexagonal'
        # 36 cubic systems
        for o in range(194,229):
            self.bravais_system_type_list[o] = 'Cubic'
        self.blt = self.get_bravais_system_type()

    def get_bravais_system_type(self):
        '''
        Returns Bravais system type for a space group
        '''
        return self.bravais_system_type_list[self.space_group_number-1]

    def get_hall_number_str(self):
        '''
        String Hall number for accessing spglib_database (dictionary)
        '''
        return str(self.spacegroup_to_hall_number[self.space_group_number-1])

    def get_hall_number_int(self):
        '''
        Integer Hall number for accessing get_symmetry_from_database
        '''
        return self.spacegroup_to_hall_number[self.space_group_number-1]

    def get_wtran(self):
        '''
        - Translation of the Wyckoff positions
        - Returns a list of lists containing numpy arrays
        '''
        translation_vectors = []
        for mult in self.coordinates:
            tmp_translation_vectors = []
            for vector in mult:
                vector = vector.replace('(','').replace(')','')
                coordinate = np.array([])
                for num in vector.split(','):
                    if all((char.isalpha() or char=='-' or char=='+')\
                           for char in num):
                        coordinate = np.append(coordinate,0)
                    elif any(char=='+' for char in num):
                        for char in num:
                            if char.isalpha() or char=='+' or char=='-':
                                num = num.replace(char,'')
                        coordinate = np.append(coordinate,eval(num))
                    elif all(char.isdigit() or char=='/' for char in num):
                        coordinate = np.append(coordinate,eval(num))
                tmp_translation_vectors.append(coordinate)
            translation_vectors.append(tmp_translation_vectors)
        return translation_vectors

    def get_wop(self):
        '''
        - Constraints on Wyckoff positions
        - Returns a list of lists containing numpy arrays
        '''
        constraint_matrices = []
        for mult in self.coordinates:
            tmp_constraint_matrices = []
            for vector in mult:
                vector = vector.replace('(','').replace(')','')
                matrix = []
                for num in vector.split(','):
                    if all(char.isdigit() or char=='/' for char in num):
                        matrix.append(np.array([0,0,0]))
                    else:
                        for char in num:
                            if not (char.isalpha() or char=='-'):
                                num = num.replace(char,'')
                        if num=='x':
                            matrix.append(np.array([1,0,0]))
                        elif num=='-x':
                            matrix.append(np.array([-1,0,0]))
                        elif num=='y':
                            matrix.append(np.array([0,1,0]))
                        elif num=='-y':
                            matrix.append(np.array([0,-1,0]))
                        elif num=='z':
                            matrix.append(np.array([0,0,1]))
                        elif num=='-z':
                            matrix.append(np.array([0,0,-1]))
                tmp_constraint_matrices.append(np.vstack([matrix]))
            constraint_matrices.append(tmp_constraint_matrices)
        return constraint_matrices

    def get_nwyc(self):
        '''
        - Number of Wyckoff positions
        - Returns an integer
        '''
        return len(self.wop)

    def get_wmult(self):
        '''
        - Multiplicity of Wyckoff positions
        - Returns a list of integers
        '''
        multiplicity = []
        for mult in self.multiplicity:
            multiplicity.append(int(mult))
        return multiplicity

    def get_uss(self):
        '''
        - Unique site symmetry groups
        - Returns a list of strings
        '''
        unique_sites = []
        for site in self.site_symmetries:
            if site not in unique_sites:
                unique_sites.append(site)
        return unique_sites

    def get_nsg(self):
        '''
        - Number of unique site symmetry groups
        - Returns an integer
        '''
        return len(self.unique_site_symmetries)

    def get_swycn(self):
        '''
        - Number of Wyckoff positions for each unique site symmetry
        - Returns a list of integers
        '''
        number_wyckoff_positions = []
        for site in self.unique_site_symmetries:
            number_wyckoff_positions.append(self.site_symmetries.count(site))
        return number_wyckoff_positions

    def get_swyc_mult(self):
        '''
        - Multiplicity of Wyckoff positions for each unique site symmetry
        - Returns a list of integers
        '''
        wyckoff_position_multiplicity = []
        for site in self.unique_site_symmetries:
            wyckoff_position_multiplicity.append(int(self.multiplicity
                                [self.site_symmetries.index(site)]))
        return wyckoff_position_multiplicity

    def selected_positions(self):
        # FOR TESTING # - only general positions
        self.new_wop = []
        for mult in self.wop:
            self.new_wop.append(mult[0])
        assert(self.get_nwyc()==len(self.wop)==len(self.new_wop))
        self.new_wtran = []
        for trans in self.wtran:
            self.new_wtran.append(trans[0])
        assert(self.get_nwyc()==len(self.new_wtran))

# =============================================================================
#     def selected_positions(self):
#         '''
#         - Creates a new_wop, new_wtran and indices
#         - For multiplicities with more than one wyckoff position, a random
#           coordinate is selected - the index of which is stored in order to access
#           the corresponding translation vector
#         - Returns None
#         '''
#         self.new_wop = []
#         self.indices = []
#         # randomly select a coordinate if there are more than 1
#         for mult in self.wop:
#             if len(mult)==1:
#                 self.new_wop.append(mult[0])
#             else:
#                 indx = random.randint(0,len(mult)-1)
#                 self.new_wop.append(mult[indx])
#                 self.indices.append(indx)
#             assert(self.get_nwyc()==len(self.wop)==len(self.new_wop))
#             # get the trans vector that corresponds to the coord
#             self.new_wtran = []
#             count = 0
#             for trans in self.wtran:
#                 if len(trans)==1:
#                     self.new_wtran.append(trans[0])
#                 else:
#                     self.new_wtran.append(trans[self.indices[count]])
#                     count+=1
#             assert(self.get_nwyc()==len(self.new_wtran))
# =============================================================================

    def wycgen(self,wn):
        '''
        - Produces a position of wyckoff position no. wn
        - Called in generation_util
        - Returns list of numpy.float64s
        '''
        assert(wn<len(self.wmult))
        assert(wn<len(self.new_wop))
        assert(wn<len(self.new_wtran))
        v=np.array([random.random(),random.random(),random.random()])
        nxyz = np.array([0,0,0])
        nxyz = np.dot(v,self.new_wop[wn])
        nxyz = nxyz + self.new_wtran[wn]
        return nxyz
        
#        for i in range (0,3):
#            for j in range (0,3):
#                nxyz[i]+=self.new_wop[wn][i][j]*v[j]
#        for i in range (0,3):
#            nxyz[i]+=self.new_wtran[wn][i]
#        return nxyz

    def wycarr(self,nmpc):
        '''
        - Produces a list of Wyckoff position arrangements that add up to nmpc of molecules per cell
        '''
        self.mlist=[]
        for i in range (0,nmpc+1):
            self.mlist.append([])
        self.mlist[0]=[[]]
        for i in range (0,self.nwyc):
            for j in range (0,nmpc+1-self.wmult[i]):
                for k in range (0,len(self.mlist[j])):
                    self.mlist[j+self.wmult[i]].append(self.mlist[j][k]+[i])
        return self.mlist[nmpc]


    def wyckoff_preparation (self,nmpc):
        '''
         -  Prepare a list of Wyckoff position for placing the molecule
         -  wyckoff_preparation creates all possible combinations of Wyckoff
            positions to achieve a certain number of molecules in the cell for a given space group.
        '''
        self.selected_positions()
        self.mlist=[]
        for i in range (0,nmpc+1):
            self.mlist.append([])
        self.mlist[0]=[[]]
        current_wyckoff = 0
        for i in range (0,self.nsg):
            # Change from enumerating by unique symmetry to standard enumeration
            # Start at zero then add swycn for each unique symmetry to get standard index
            if i == 0:
                current_wyckoff = 0
            else:
                current_wyckoff = self.swycn[i]+current_wyckoff
            wyckoff_constraints = np.array(self.new_wop[current_wyckoff])
            # If trace is zero there are no constraints and can't be repeated
            constraint_trace = abs(np.trace(wyckoff_constraints))
            if constraint_trace > 0:
            #A full knapsack for the repeating Wyckoff Positions
                for j in range (0,nmpc+1-self.swyc_mult[i]):
                    for k in range (0,len(self.mlist[j])):
                        self.mlist[j+self.swyc_mult[i]].append(self.mlist[j][k]+[i])
            else:
            #A limited knapsack for the non-repeating Wyckoff positions
                for j in range (nmpc-self.swyc_mult[i],-1,-1):
                    for l in range (1,1+int(min(self.swycn[i],(nmpc-j)/(self.swyc_mult[i])))):
                        for k in range (0,len(self.mlist[j])):
                            self.mlist[j+l*self.swyc_mult[i]].append(self.mlist[j][k]+[i for ll in range (0,l)])
        return not (len(self.mlist[nmpc])==0)

    def wyckoff_selection_trace (self,nmpc):
        '''
        '''
        ll=int(random.uniform(0,len(self.mlist[nmpc])))
        slist=self.mlist[nmpc][ll]
        wlist=[]
        for i in range (0,len(slist)):
            # Change index from unique symmetry to standard enumeration
            current_wyckoff = 0
            for j in range(0,slist[i]):
                current_wyckoff += self.swycn[j]
            current_trace = abs(np.trace(self.new_wop[current_wyckoff]))
            if current_trace > 0:
                wn=int(random.uniform(0,self.swycn[slist[i]]))
                for j in range (0,slist[i]):
                    wn+=self.swycn[j]
                wlist.append(wn)
            else:
                # This is if the Wyckoff position cannot be repeated
                # While loop is a slow way to do this
                while True:
                    wn=int(random.uniform(0,self.swycn[slist[i]]))
                    for j in range (0,slist[i]):
                        wn+=self.swycn[j]
                    if not (wn in wlist):
                        break
                wlist.append(wn)
        return(wlist)

    def wyckoff_counter(self,wyckoff_list):
        '''
        Count the sum of molecules required by the wyckoff list specified
        '''
        counter = 0
        for wyc in wyckoff_list:
            if wyc >= len(self.wmult): #Unsuitable list
                return -1
            counter += self.wmult[wyc]
        return counter

def allowed_sg_wyckoff_list(nmpc,wyckoff_list):
    '''
    Given a list of Wyckoff position number
    Returns a list of space group that gives the desirable nmpc
    '''
    result = []
    for i in range (0,MAX_AVAILABLE_SPACE_GROUP):
        sg = Sgroup(i)
        if nmpc==sg.wyckoff_counter(wyckoff_list):
            result.append(i)
    return result

def allowed_sg_nmpc(nmpc):
    '''
    Given nmpc, returns the list of space group that can yield the number
    '''
    result = []
    for i in range (0,MAX_AVAILABLE_SPACE_GROUP):
        sg = Sgroup(i)
        if sg.wyckoff_preparation(nmpc):
            result.append(i)
    return result

def select_chiral_sg (sg_list):
    '''
    Select and return the chiral space groups in the given sg_list
    sg_list should be a list of integers
    list of chiral space groups:
    http://www-chimie.u-strasbg.fr/csd.doc/ConQuest/PortableHTML/conquest_portable-3-325.html
    '''
    result = []
    for sg in sg_list:
        if sg in [1]+range(3,6)+range(16,25)+range(75,81)+range(89,99)\
            +range(143,147)+range(149,156)+range(168,174)+range(177,183)\
            +range(195,200)+range(207,215):
            result.append(sg)
    return result

def select_racemic_sg (sg_list):

    '''
    Select and return the racemic space groups in the given sg_list
    sg_list should be a list of integers
    list of chiral space groups:
    http://www-chimie.u-strasbg.fr/csd.doc/ConQuest/PortableHTML/conquest_portable-3-325.html
    '''
    result = []
    for sg in sg_list:
        if not sg in [1]+range(3,6)+range(16,25)+range(75,81)+range(89,99)\
            +range(143,147)+range(149,156)+range(168,174)+range(177,183)\
            +range(195,200)+range(207,215):
            result.append(sg)
    return result


if __name__ == '__main__':
    pass
