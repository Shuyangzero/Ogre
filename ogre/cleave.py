import os
import sys
from ase.io import read, write
from ogre import slab_generator
from pymatgen.io.vasp.inputs import Poscar

struct = '../example_POSCARs/PENCEN.POSCAR.vasp'
struct_ase = read(struct)
miller_index = [1, 0, 0]
layers = 4
vacuum = 100
working_dir = os.path.abspath('.')
slab_list = slab_generator.repair_organic_slab_generator_move(struct_ase,
                                                              miller_index,
                                                              layers, vacuum,
                                                              working_dir, None)
Poscar(slab_list[0]).write_file("Ogre_Surface.POSCAR.vasp")
