import os
from ase.io import read, write
from ogre import slab_generator
from pymatgen.io.vasp.inputs import Poscar
from ogre.slab_generator import task
import os

name = 'TETCEN'
struc = read('./structures/relaxed_structures/{}.cif'.format(name))
miller_index = [-1, -1, 1]
layers = list(range(10, 16, 1))
vacuum = 40
if not os.path.isdir(name):
    os.mkdir(name)
task(name, struc, miller_index, layers, vacuum, super_cell=None)
