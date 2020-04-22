
from collections import defaultdict
import json
import math
import numpy as np
import os
import datetime

import ase
from ase import Atoms
from ase.data import atomic_numbers,atomic_masses_iupac2016
from ase.formula import Formula
from ase.neighborlist import NeighborList,natural_cutoffs

from pymatgen import Lattice as LatticeP
from pymatgen import Structure as StructureP
from pymatgen import Molecule


class Structure(object):
    """
    Core object for describing geometry and properties of Structures. There are 
    many construction methods for the Structure object from Pymatgen, ASE, or
    from a dictionary. 
    
    """
    def __init__(self):
        # initialize settings and empty geometry
        self.struct_id = ""
        self.input_ref = None        
        self.properties = {}
        self.geometry = np.zeros(0, dtype=[('x', float), ('y', float), 
                    ('z', float), ('element', 'U13'), ('spin', float), 
                    ('charge', float), ('fixed', 'bool')])
    
    
    def __str__(self):
        if len(self.struct_id) == 0:
            self.get_struct_id(universal=True)
        if len(self.get_lattice_vectors()) > 0:
            struct_type = "Crystal"
        return "{}: {} {}".format(self.struct_id, 
                                  struct_type,
                                  self.formula())
    
    def __repr__(self):
        if len(self.get_lattice_vectors()) > 0:
            struct_type = "Crystal"
        else:
            struct_type = "Molecule"
        return "{} {}".format(struct_type, self.formula())
    
    
    def append(self, x, y, z, element, spin=None, charge=None, fixed=False):
        # increase the size of the array
        size = self.geometry.size
        self.geometry.resize(size + 1)
        # assign values
        self.geometry[size]['x'] = x
        self.geometry[size]['y'] = y
        self.geometry[size]['z'] = z
        self.geometry[size]['element'] = element
        self.geometry[size]['spin'] = spin 
        # test for non-assigned spin with math.isnan(a[i]['spin'])
        self.geometry[size]['charge'] = charge 
        # test for non-assigned charge with math.isnan(a[i]['charge'])
        self.geometry[size]['fixed'] = fixed 


    def reset_lattice_vectors(self, vectors):
        if "lattice_vector_a" in self.properties:
            del(self.properties["lattice_vector_a"])
        if "lattice_vector_b" in self.properties:
            del(self.properties["lattice_vector_b"])
        if "lattice_vector_c" in self.properties:
            del(self.properties["lattice_vector_c"])
        self.set_lattice_vectors(vectors)


    def set_lattice_vectors(self, vectors):
        if vectors is None or vectors is False: return False
        
        if len(vectors) != 3:
            raise Exception("set_lattice_vectors got {}".format(vectors) +
                        "This is supposed to be a list of three " + 
                        "lattice vectors.")
            
        for vector in vectors: 
            self.add_lattice_vector(vector)
            
        # After setting lattice vectors, recalculate the volume of system
        lattice = self.get_lattice_vectors_better()
        self.properties["cell_vol"] = np.linalg.det(lattice)


    def set_lattice_angles(self):
        alpha, beta, gamma = self.get_lattice_angles() 
        self.set_property("alpha", alpha)
        self.set_property("beta", beta)
        self.set_property("gamma", gamma)
        
    
    def from_geo_array(self, array, elements):
        """  Set geometry of structure to the input array and elements
        
        Arguments
        ---------
        Array: np.matrix of numbers
          Atom coordinates should be stored row-wise
        Elements: np.matrix of strings
          Element strings using shorthand notations of same number of rows 
          as the array argument
        """
        size = array.shape[0]
        if len(elements) != size:
            raise Exception('Dimension of array and element arguments to '+
                    'Structure.from_geo_array are not equal.')
        self.geometry = np.full(size,None,
                     dtype=[('x', 'float32'),('y', 'float32'), ('z', 'float32'), 
                            ('element', 'U13'), ('spin', 'float32'), 
                            ('charge', 'float32'), ('fixed', 'bool')])
        self.geometry['x'] = array[:,0]
        self.geometry['y'] = array[:,1]
        self.geometry['z'] = array[:,2]
        self.geometry['element'] = np.array(elements)
    
    
    @classmethod 
    def from_geo(cls, array, elements):
        """
        Construction method of Structure object. 
        
        """
        struct = cls()
        struct.from_geo_array(array, elements)
        struct.get_struct_id(universal=True)
        return struct


    def add_lattice_vector(self, vector):
        lattice_vector_name = 'lattice_vector_a'
        if 'lattice_vector_a' in self.properties: lattice_vector_name = 'lattice_vector_b'
        if 'lattice_vector_b' in self.properties: lattice_vector_name = 'lattice_vector_c'
        if 'lattice_vector_c' in self.properties: raise Exception  # lattice vectors are full, 
        self.set_property(lattice_vector_name, vector)

    """ These functions should just be called: from_json, from_aims, etc. 
        The names of these functions are terrible.
    """
    def build_geo_from_atom_file(self, filepath): self.build_geo_whole_atom_format(read_data(filepath))
    def build_struct_from_json_path(self, filepath): self.loads(read_data(filepath))	
    
    
    def build_geo_whole_atom_format(self, atom_string):
        """
        Constructs relevant geometry properties from an FHI-aims geometry 
          file. 
        """
        def add_previous_atom(atom):
            try: spin = atom.get('spin')
            except: spin = None
            try: charge = atom.get('charge')
            except: charge = None
            try: fixed = atom.get('fixed')
            except: fixed = False
            self.append(atom['x'], atom['y'], atom['z'],
                        atom['element'], spin, charge, fixed)
        lines_iter = iter(atom_string.split('\n'))
        atom = {} 
        while True:
            try: line = next(lines_iter).split()  # read each line
            
            # THIS COULD HAVE BEEN WRITTEN IN A CLEARER WAY TO EXIT WHILE LOOP
            #   AND SAVE THE LAST ATOM TO THE GEOMETRY.
            except: add_previous_atom(atom); return self.geometry
            if len(line) == 0: continue
            if '#' in line[0]: continue  # skip commented lines
            
            if line[0] == 'lattice_vector': 
                self.add_lattice_vector((float(line[1]), float(line[2]), float(line[3])))
                continue
            
            if line[0] == 'atom':
                if not len(atom) == 0: add_previous_atom(atom)
                atom = {}
                atom['x'] = float(line[1])
                atom['y'] = float(line[2])
                atom['z'] = float(line[3])
                atom['element'] = str(line[4])
                
            if line[0] == 'atom_frac':
                if not len(atom) == 0: add_previous_atom(atom)
                atom = {}
                frac_coord = np.array([float(line[1]),
                                       float(line[2]),
                                       float(line[3])])[:,None]
                lv = np.array(self.get_lattice_vectors_better())
                cart_coord = np.dot(lv.T,
                                    frac_coord)
                atom['x'] = cart_coord[0]
                atom['y'] = cart_coord[1]
                atom['z'] = cart_coord[2]
                atom['element'] = str(line[4])
                
            # only affects previous atom
            if 'initial_spin' in line[0]: atom['spin'] = float(line[1])
            if 'initial_charge' in line[0]: atom['charge'] = float(line[1]) 
            if any('constrain_relaxation' in s for s in line) and any('true' in s for s in line): 
                atom['fixed'] = True
        
    def set_input_ref(self, input_ref): self.input_ref = input_ref   
    
    def set_property(self, key, value):
        self.properties[key] = value
#        try: self.properties[key] = ast.literal_eval(value)
#        except: self.properties[key] = value
        
    def delete_property(self, key):
        try: self.properties.pop(key)
        except: pass
    
    # getters
    def get_geometry(self): return self.geometry
    def pack_geometry(self): return adapt_array(self.geometry)
    def get_n_atoms(self): return self.geometry.size

    def get_n_atoms_per_mol(self, num_mols): return self.geometry.size/num_mols

    def get_atom_types(self):
        element_list = []
        for i in range(self.geometry.size):
            element_list.append(self.geometry[i]['element'])
        return element_list

    def get_input_ref(self): return  self.input_ref
    def get_stoic(self): return  calc_stoic(self.geometry)
    def get_stoic_str(self): return self.get_stoic().get_string()
    def get_path(self): return self.get_stoic_str() + '/' + str(self.get_input_ref()) + '/' +str(self.get_struct_id()) 
    def get_property(self, key):
        try: return self.properties.get(key)
        except:
            try: self.reload_structure()  # may not have properly read property
            except Exception as e: print(e); return None
            
    def get_lattice_vectors(self):
        """ Always returns list as data type """
        if 'lattice_vector_a' not in self.properties: return []
        return_list = []
        return_list.append(self.get_property('lattice_vector_a'))
        return_list.append(self.get_property('lattice_vector_b'))
        return_list.append(self.get_property('lattice_vector_c'))
        return return_list


    def get_lattice_vectors_better(self):
        """
        This function should be deprecated from source code 
        during clean-up of Structure class.

        """
        return self.get_lattice_vectors()
    
    
    def get_geometry_atom_format(self): 
        """
        Should be renamed to: convert/get_aims()
        There should be a master convert or get function that accepts str 
            argument for: 'aims', 'json', 'pymatgen', 'ase'
        
        Takes a np.ndarry with standard "geometry" format.
        Returns a string with structure in standard aims format.
        If atom's spin is spedcified, it's value is located on the line below the atom's coordinates.
        similarly with charge and relaxation constraint.
        
        MODIFIED TO WORK WITH MOLECULES: It's ridiculous that this didn't work 
          if there were no lattice vectors. 
        """
        lattice_vectors = self.get_lattice_vectors()
        atom_string = ''
        if lattice_vectors is not False:
            for vector in lattice_vectors:
                atom_string += 'lattice_vector ' + ' '.join(map(str, vector)) + '\n'
        for item in self.geometry:
            atom_string += 'atom ' + "%.5f" % item['x'] + ' ' + "%.5f" % item['y'] + ' ' + "%.5f" % item['z'] + ' ' + str(item['element']) + '\n'
            if not math.isnan(item['spin']): atom_string += 'initial_moment ' + "%.5f" % item['spin'] + '\n'
            if not math.isnan(item['charge']): atom_string += 'initial_charge ' + "%.5f" % item['charge'] + '\n'
            if item['fixed'] == True: atom_string += 'constrain_relaxation    .true.\n'
        return atom_string
    
    
    def get_aims(self):
        return self.get_geometry_atom_format()
    
    
    def get_geo_array(self):
        """ Return np.array of (x,y,z) geometry """
        num_atoms = self.geometry.shape[0]
        x_array = self.geometry['x'].reshape(num_atoms,1)
        y_array = self.geometry['y'].reshape(num_atoms,1)
        z_array = self.geometry['z'].reshape(num_atoms,1)
        
        current_geometry = np.concatenate((x_array,y_array,z_array),axis=1)
    
        return current_geometry
    
    
    def get_ase_atoms(self):
        """ Works for periodic and non-periodic systems
        
        Purpose: Returns ase atoms object
        """
        symbols = self.geometry['element']
        positions = np.stack([self.geometry['x'],
                              self.geometry['y'],
                              self.geometry['z']],axis=1)
        cell = np.array(self.get_lattice_vectors())
        if len(cell) == 3:
            pbc = (1,1,1)
            return ase.Atoms(symbols=symbols, positions=positions,
                         cell=cell, pbc=pbc)
        else:
            pbc = (0,0,0)
            return ase.Atoms(symbols=symbols, positions=positions)
    
    @classmethod
    def from_ase(cls, atoms):
        """ 
        Construction classmethod for Structure from ase Atoms object. 
        
        """
        symbols = atoms.get_chemical_symbols()
        geo_array = atoms.get_positions()
        pbc = atoms.get_pbc()

        struct = cls()
        
        if pbc.any() == True:
            cell = atoms.get_cell()
            struct.properties["lattice_vector_a"] = cell[0]
            struct.properties["lattice_vector_b"] = cell[1]
            struct.properties["lattice_vector_c"] = cell[2]
                
        struct.from_geo_array(geo_array,symbols)

        return struct
  
    
    def get_pymatgen_structure(self):
        """
        Inputs: A np.ndarry structure with standard "geometry" format
        Outputs: A pymatgen core structure object with basic geometric properties
        """
        if self.get_lattice_vectors():
            frac_data = self.get_frac_data()
            coords = frac_data[0] # frac coordinates
            atoms = frac_data[1] # site labels
            lattice = LatticeP.from_parameters(a=frac_data[2],
                                               b=frac_data[3], 
                                               c=frac_data[4], 
                                               alpha=frac_data[5],
                                               beta=frac_data[6], 
                                               gamma=frac_data[7])
            structp = StructureP(lattice, atoms, coords)
            return structp
            
        else:
            coords = self.get_geo_array()
            symbols = self.geometry['element']
            molp = Molecule(symbols, coords)
            return molp


    def get_frac_data(self):
        """
        Inputs: A np.ndarry structure with standard "geometry" format
        Outputs:  Fractional coordinate data in the form of positions (list), 
        atom_types (list), lattice vector a magnitude, lattice vector b magnitude, 
        lattice vector c magnitude, alpha beta, gamma.
        """
        geo = self.geometry
        A = self.get_property('lattice_vector_a')
        B = self.get_property('lattice_vector_b')
        C = self.get_property('lattice_vector_c')
        alpha, beta, gamma = self.get_lattice_angles()
        a, b, c = self.get_lattice_magnitudes()
        atoms = [i for i in range(len(geo))]
        lattice_vector = np.transpose([A,B,C])
        latinv = np.linalg.inv(lattice_vector)
        coords = []
        for i in range(len(geo)):
            atoms[i] = geo[i][3]
            coords.append(np.dot(latinv,[geo[i][0],geo[i][1],geo[i][2]]))
        return coords, atoms, a, b, c, alpha, beta, gamma
    

    @classmethod
    def from_pymatgen(cls,pymatgen_obj):
        """ 
        Construction classmethod for Structure by converting pymatgen 
        Lattice/Molecule to Structure.
        
        """
        struct = cls()
        
        geometry = np.array([site.coords for site in pymatgen_obj])
        species = np.array([site.specie for site in pymatgen_obj])
        if type(pymatgen_obj) == Molecule:
            struct.from_geo_array(geometry,species)
            
        elif type(pymatgen_obj) == LatticeP:
            raise Exception('Lattice conversion not implemented yet')
        
        elif type(pymatgen_obj) == StructureP:
            struct.from_geo_array(geometry,species)
            struct.set_lattice_vectors(pymatgen_obj.lattice.matrix)
        
        return struct


    def formula(self):
        formula_dict = {}
        ele_list,count = np.unique(self.geometry["element"], return_counts=True)
        for idx,ele in enumerate(ele_list):
            ## Conversion to int to be JSON serializable
            formula_dict[ele] = int(count[idx])
        self.properties["formula"] = formula_dict
        return formula_dict


    def density(self):
        volume = self.get_unit_cell_volume()
        mass = np.sum([atomic_masses_iupac2016[atomic_numbers[x]]
                       for x in self.geometry["element"]])
    
        ## Conversion factor for converting amu/angstrom^3 to g/cm^3
        ## Want to just apply factor to avoid any numerical errors to due float 
        factor = 1.66053907
    
        self.properties["density"] = (mass / volume)*factor
        return self.properties["density"]


    def get_struct_id(self, universal=False):
        """
        Get the id for the structure. If a struct_id has already been stored, 
        this will be returned. Otherwise, a universal struct_id will be 
        constructed. If universal argument is True, then the current struct_id
        will be discarded and a universal struct_id will be constructed.

        """
        if len(self.struct_id) > 0 and universal == False:
            return self.struct_id
        else:
            ## Get type 
            name = ""
            if len(self.get_lattice_vectors_better()) > 0:
                name = "Structure"
            else:
                name = "Molecule"
            
            ## Get formula
            formula = Formula.from_list(self.geometry["element"]) 
            ## Reduce formula, which returns formula object
            formula = formula.format("hill")
            ## Then get string representation stored in formula._formula
            formula = str(formula)
            ## Add formula to name
            name += "_{}".format(formula)

            ## Add Date
            today = datetime.date.today()
            name += "_{}{}{}".format(today.year,today.month,today.year)

            ## Add random string
            name += "_{}".format(rand_str(10))

            self.struct_id = name
            return self.struct_id


    def get_space_group(self, update=True):
        if "space_group" in self.properties:
            return self.properties["space_group"]
        if "spg" in self.properties:
            self.properties["space_group"] = self.properties["spg"]
            return self.properties["spg"]
        
        pstruct = self.get_pymatgen_structure()
        spg_symbol,spg_internation_number = pstruct.get_space_group_info()

        self.properties["space_group"] = spg_internation_number
        return self.properties["space_group"]


    def get_lattice_angles(self):
        A = self.get_property('lattice_vector_a')
        B = self.get_property('lattice_vector_b')
        C = self.get_property('lattice_vector_c')
        alpha = self.angle(B, C)
        beta = self.angle(C, A)
        gamma = self.angle(A, B)
        return alpha, beta, gamma


    def get_lattice_magnitudes(self):
        A = self.get_property('lattice_vector_a')
        B = self.get_property('lattice_vector_b')
        C = self.get_property('lattice_vector_c')
        a = np.linalg.norm(A)
        b = np.linalg.norm(B)
        c = np.linalg.norm(C)
        return a, b, c


    def get_unit_cell_volume(self):
        if "cell_vol" in self.properties:
            self.properties["unit_cell_volume"] = self.properties["cell_vol"]
            return self.properties["cell_vol"]
        if "unit_cell_volume" in self.properties:
            return self.properties["unit_cell_volume"]
        A = self.get_property('lattice_vector_a')
        B = self.get_property('lattice_vector_b')
        C = self.get_property('lattice_vector_c')
        self.properties["unit_cell_volume"] = np.linalg.det([A,B,C])
        return self.properties["unit_cell_volume"]
        

    def get_atom_distance(self,a1,a2):
        return np.linalg.norm([self.geometry[a1][k]-self.geometry[a2][k] for k in range(3)])


    def angle(self, v1, v2):
        numdot = np.dot(v1,v2)
        anglerad = np.arccos(numdot/(np.linalg.norm(v1)*np.linalg.norm(v2)))
        angledeg = anglerad*180/np.pi
        return angledeg
    
    
    def document(self, _id=""):
        """
        Turn Structure object into a document for MongoDB.
        
        Arguments
        ---------
        _id: str
           The _id for the document in the MongoDB. Default behavior is to
           use the struct_id as the _id.
        """
        struct_doc = self.__dict__.copy()
        struct_doc["geometry"] = self.geometry.tolist()
        if len(_id) == 0:
            struct_doc["_id"] = self.struct_id
        else:
            struct_doc["_id"] = _id
        return struct_doc
    

    # json data handling packing
    def dumps(self):
        if len(self.get_lattice_vectors_better()) > 0: 
            self.properties["lattice_vector_a"]=list(self.properties["lattice_vector_a"])
            self.properties["lattice_vector_b"]=list(self.properties["lattice_vector_b"])
            self.properties["lattice_vector_c"]=list(self.properties["lattice_vector_c"])
        data_dictionary = {}
        data_dictionary['properties'] = self.properties
        data_dictionary['struct_id'] = self.struct_id
        data_dictionary['input_ref'] = self.input_ref
        data_dictionary['geometry'] = self.geometry.tolist()
        
        return json.dumps(data_dictionary, indent=4)
        
    
    def loads(self, json_string):
        data_dictionary = json.loads(json_string)
        self.properties = data_dictionary['properties']
        try: self.struct_id = data_dictionary['struct_id']
        except: pass
        try: self.input_ref = data_dictionary['input_ref']
        except: pass # if input reference from initial pool then skip this part
        self.geometry = convert_array(data_dictionary['geometry'])
    
    
    @classmethod
    def from_dict(cls,dictionary):
        struct = cls()
        struct.properties = dictionary["properties"]
        struct.struct_id = dictionary["struct_id"]
        struct.geometry = convert_array(dictionary["geometry"])
        
    
    def get_bonds(self, mult=1.20, skin=0.0, update=False):
        """
        Returns array of covalent bonds in the molecule. In addition, these
        are stored in the Structure properties for future reference. 
        
        Arguments
        ---------
        mult: float
            For ASE neighborlist
        skin: float
            For ASE neighborlist
        update: bool
            If True, will force an update of bond information.
            
        Returns
        -------
        list
            The index of the list corresponds to the atom the bonds are 
            describing. Inside each index is another list. This is the indices
            of the atoms the atom is bonded. Please keep in mind that Python
            iterables are zero indexed whereas most visualization softwares
            will label atoms starting with 1. 
        
        """
        if update == False and "bonds" in self.properties:
            pass
        else:
            atoms = self.get_ase_atoms()
            cutOff = natural_cutoffs(atoms, mult=mult)
            neighborList = NeighborList(cutOff, self_interaction=False,
                                        bothways=True, skin=skin)
            neighborList.update(atoms)
            
            # Construct bonding list indexed by atom in struct
            bonding_list = [[] for x in range(self.geometry.shape[0])]
            for i in range(self.geometry.shape[0]):
                temp_list = list(neighborList.get_neighbors(i)[0])
                if len(temp_list) > 0:
                    temp_list.sort()
                bonding_list[i] = [int(x) for x in temp_list]
            
            self.properties["bonds"] = bonding_list
        
        return self.properties["bonds"]
        
        
class StoicDict(defaultdict):    
    
    def __hash__(self):
        return str(self).__hash__()    
    
    def get_string(self):
        keys = list(self.keys())
        keys.sort()

#        keys.sort()
        stoic_string = ''
        for item in keys:
            stoic_string += str(item) + ':' + str(self[item]) + '_'
        stoic_string = stoic_string[:-1]  # remove last underscore
        return stoic_string
    
def calc_stoic(geo):
    """
    returns a dictionary representing the stoichiometries
    """
    stoic = StoicDict(int)
    for item in geo:
        stoic[item['element']] += 1
    return stoic

def get_geo_from_file(file_name):
    """ 
    given the path to a geometry-style file, returns the geometry in proper format
    """
    tmp_struct = Structure()
    atom_file = open(file_name, 'r')
    geo = tmp_struct.build_geo_whole_atom_format(atom_file.read())
    atom_file.close()
    return geo

def adapt_array(arr):
    return json.dumps(arr.tolist())

def convert_array(list_of_list):
    """
    takes the array stored in json format and return it to a np array with 
    proper dtype
    """
    geometry = np.zeros(len(list_of_list), dtype=[('x', 'float32'), 
                ('y', 'float32'), ('z', 'float32'), ('element', 'U13'), 
                ('spin', 'float32'), ('charge', 'float32'), ('fixed', 'bool')])
            
    for i in range(len(list_of_list)): 
        geometry[i]['x'] = list_of_list[i][0]
        geometry[i]['y'] = list_of_list[i][1]
        geometry[i]['z'] = list_of_list[i][2]
        geometry[i]['element'] = str(list_of_list[i][3])
        try:
            geometry[i]['spin'] = list_of_list[i][4]
        except: geometry[i]['spin'] = None
        try:
            geometry[i]['charge'] = list_of_list[i][5]
        except: geometry[i]['charge'] = None
        try:
            geometry[i]['fixed'] = list_of_list[i][6]
        except: geometry[i]['fixed'] = None
    return geometry

def read_data(filepath):
    full_filepath = filepath
    d_file = open(full_filepath, 'r')
    contents_string = d_file.read()
    d_file.close()
    return contents_string


def rand_str(length):
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
    np_alphabet = np.array([x for x in alphabet])
    rand = np.random.choice(np_alphabet, size=(length,), replace=True)
    return "".join(rand)


if __name__ == '__main__':
    Structure = Structure
