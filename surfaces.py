from ogre import slab_generator

name = 'HOJCOB'
struc_path = './structures/relaxed_structures/{}.cif'.format(name)
layers = list(range(10,16))
highest_index = 1
vacuum = 40

slab_generator.cleave_planes(struc_path, name, vacuum, layers, highest_index)
