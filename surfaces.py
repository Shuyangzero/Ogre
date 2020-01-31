from ogre import slab_generator

struc_path = './example_POSCARs/aspirin.POSCAR.vasp'
name = 'aspirin'
layers = [2, 3, 4]
highest_index = 1
vacuum = 15

slab_generator.cleav_planes(struc_path, name, vacuum, layers, highest_index)