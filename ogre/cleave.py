import os
from ase.io import read
from ogre import slab_generator
from pymatgen.io.vasp.inputs import Poscar

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
# Change one layer structure to "layers" layers structure, add vacuum, and build
# the super cell.
slab_list = slab_generator.change_layers_and_supercell(slab_one_layer_list[0],
                                                       layers, vacuum, working_dir,
                                                       super_cell=None)

Poscar(slab_list[0]).write_file("Ogre_Surface.POSCAR.vasp")
