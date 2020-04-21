import os
from ase.io import read, write
from pymatgen.io.vasp.inputs import Poscar
from ogre.generators import task
import os

name = 'aspirin'
struc = read('./structures/relaxed_structures/{}.cif'.format(name))
miller_index = [1,0,0]
layers = [2]
vacuum = 5
if not os.path.isdir(name):
    os.mkdir(name)
task(name, struc, miller_index, layers, vacuum, None, 'VASP')
