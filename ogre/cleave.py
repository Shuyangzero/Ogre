import os
from ase.io import read
from ogre import slab_generator
from pymatgen.io.vasp.inputs import Poscar
from ogre.utils.utils import different_single_layer

struct = '../example_POSCARs/PENCEN.POSCAR.vasp'
struct_ase = read(struct)
miller_index = [1, 0, 0]
layers = 4
vacuum = 15
working_dir = os.path.abspath('.')
# slab_one_layer_list[0] is the structure of single layer
# (z direction is not perpendicular to xy face, no vacuum is added).
slab_one_layer_list = slab_generator.repair_organic_slab_generator_move_onelayer(struct_ase,
                                                                                 miller_index,
                                                                                 layers, vacuum,
                                                                                 working_dir, None)
# Poscar(slab_one_layer_list[0]).write_file("one_layer_incline.POSCAR.vasp").
slab_one_layer = slab_one_layer_list[0]
different_one_layer_list = different_single_layer(slab_one_layer, users_define_layers=None)

# Change one layer structure to "layers" layers structure, add vacuum, and build
# the super cell.
slab_list = slab_generator.change_layers_and_supercell(different_one_layer_list,
                                                       layers, vacuum, working_dir,
                                                       super_cell=None)

for index, slab in enumerate(slab_list):
    Poscar(slab).write_file("Ogre_Surface_" + str(index) + ".POSCAR.vasp")
