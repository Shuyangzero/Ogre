from ogre import slab_generator

struc_path = './structures/relaxed_structures/.cif'
name = 'QQQCIG04'
layers = list(range(1,10))
highest_index = 1
vacuum = 40

slab_generator.cleave_planes(struc_path, name, vacuum, layers, highest_index)
