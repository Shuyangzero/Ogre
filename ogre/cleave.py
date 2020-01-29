import os
from ase.io import read
from ogre import slab_generator
from pymatgen.io.vasp.inputs import Poscar
from ogre.utils.utils import different_single_layer
from ogre.utils.utils import different_target_surfaces
from ogre.utils.utils import surface_self_defined

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

slab_one_layer = slab_one_layer_list[0]
# # Method one: Find possible one_layer, then generate the target surface.
# # Poscar(slab_one_layer_list[0]).write_file("one_layer_incline.POSCAR.vasp").
# different_one_layer_list = different_single_layer(slab_one_layer, users_define_layers=None)
#
# # Change one layer structure to "layers" layers structure, add vacuum, and build
# # the super cell.
# slab_list_one = slab_generator.change_layers_and_supercell(different_one_layer_list,
#                                                            layers, vacuum, working_dir,
#                                                            super_cell=None)
#
# for index, slab in enumerate(slab_list_one):
#     Poscar(slab).write_file("Ogre_Surface_" + str(index) + ".POSCAR.vasp")

# Method two: Generate one target surface, then find more possible surfaces
slab_list_two = slab_generator.change_layers_and_supercell(slab_one_layer_list, layers, vacuum,
                                                           working_dir, super_cell=None,
                                                           c_perpendicular=False)
delta_move = surface_self_defined(struct_ase, miller_index, layers).cell[2, :]
slab_list_two_more = different_target_surfaces(slab_list_two[0], vacuum, working_dir, delta_move,
                                               super_cell=None, users_define_layers=None,
                                               c_perpendicular=True)

for index, slab in enumerate(slab_list_two_more):
    Poscar(slab).write_file("Ogre_Surface_" + str(index) + ".POSCAR.vasp")
