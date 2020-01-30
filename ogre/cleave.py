import os
from ase.io import read
from ogre import slab_generator
from pymatgen.io.vasp.inputs import Poscar


struct = '../example_POSCARs/PENCEN.POSCAR.vasp'
struct_ase = read(struct)
miller_index = [1, 0, 0]
layers = [2, 3, 4]
vacuum = 15
working_dir = os.path.abspath('.')

slablists = slab_generator.orgslab_generator(struct_ase, miller_index, layers,
                                             vacuum, working_dir, super_cell=None,
                                             users_defind_layers=None,
                                             based_on_onelayer=True)

for index_layer, slablist in enumerate(slablists):
    for index, slab in enumerate(slablist):
        Poscar(slab).write_file("Ogre_Surface_" + str(layers[index_layer]) + "_layers_" + str(index) + ".POSCAR.vasp")

