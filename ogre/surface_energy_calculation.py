from ogre import slab_generator
from ogre.utils.unique_planes import UniquePlanes
from pymatgen.io.vasp.inputs import Poscar
import os
from ase.io import write, read
from pymatgen.io.vasp.inputs import *
import numpy
path = '../example_POSCARs/PENCEN.POSCAR.vasp'
name = 'PENCEN'
bulk = read(path)
vacuum = 15
working_dir = os.path.abspath('.')
up = UniquePlanes(bulk, index=2, verbose=False)
print("{} unique planes are found".format(len(up.unique_idx)))
os.mkdir('output')
for miller_index in up.unique_idx:
    for layers in range(2,5,1):
        slab_list = slab_generator.repair_organic_slab_generator_move(bulk,
                                                              miller_index,
                                                              layers, vacuum,
                                                              working_dir, None)
        Poscar(slab_list[0]).write_file("output/POSCAR")
        slab_ase = read('output/POSCAR')
        os.remove("output/POSCAR")
        write("output/{}.{}.{}.in".format(name, "".join(str(int(x)) for x in miller_index), str(layers)), slab_ase)