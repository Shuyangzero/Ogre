import os
import sys
from ase.io import *
from pymatgen.io.vasp.inputs import *
sys.path.append('../')
import slab_generator
from pymatgen.io.vasp.inputs import Poscar

struct = '~/Downloads/aspirin_form_1.POSCAR.vasp'
struct_ase = read(struct)
miller_index = [1, 0, 0]
layers = 2
vacuum = 15
working_dir = os.path.abspath('.')
slab_list = slab_generator.repair_organic_slab_generator_move(struct_ase,
                                                              miller_index,
                                                              layers, vacuum,
                                                              working_dir, None)
Poscar(slab_list[0]).write_file("Ogre_Surface.POSCAR.vasp")
