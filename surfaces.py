from ogre import generators
name = 'aspirin'
struc_path = './structures/relaxed_structures/{}.cif'.format(name)
layers = list(range(1,16))
highest_index = 1
vacuum = 5
generators.cleave_planes(struc_path, name, vacuum, layers, highest_index)
