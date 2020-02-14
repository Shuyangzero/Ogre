from ogre import slab_generator
name = ''
struc_path = './structures/relaxed_structures/{}.cif'.format(name)
layers = list(range(1,16))
highest_index = 1
vacuum = 40
slab_generator.cleave_planes(struc_path, name, vacuum, layers, highest_index)
