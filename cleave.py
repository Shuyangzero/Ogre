import os
from ase.io import read, write
from ogre import slab_generator
from pymatgen.io.vasp.inputs import Poscar


struct = './structures/aspirin.POSCAR.vasp'
name = 'aspirin'
struct_ase = read(struct)
miller_index = [0, 1, 2]
layers = list(range(1, 10, 1))
vacuum = 40
working_dir = os.path.abspath('.')

slablists = slab_generator.orgslab_generator(struct_ase, miller_index, layers,
                                             vacuum, working_dir, super_cell=None,
                                             users_defind_layers=None,
                                             based_on_onelayer=True)

for layer, slablist in zip(layers, slablists):
    for i, slab in enumerate(slablist):
        Poscar(slab).write_file("POSCAR")
        slab_ase = read("POSCAR")
        os.remove("POSCAR")
        write("{}.{}.{}.{}.in".format(name, "".join(str(int(x))
                                                    for x in miller_index), layer, i), slab_ase)
