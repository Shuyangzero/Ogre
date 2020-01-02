import os
import sys
from ase.io import *
from pymatgen.io.vasp.inputs import *
sys.path.append('../')
import slab_generator
from pymatgen.io.vasp.inputs import Poscar

struct = './PENCEN.POSCAR.vasp'
struct_ase = read(struct)
miller_index = [1, 1, 2]
layers = 3
vacuum = 15
working_dir = os.path.abspath('.')
slab_list = slab_generator.repair_organic_slab_generator_move(struct_ase,
                                                              miller_index,
                                                              layers, vacuum,
                                                              working_dir, [3,
                                                                            3,
                                                                            1])
Poscar(slab_list[0]).write_file("Ogre_Surface.POSCAR.vasp")
