from ogre import slab_generator

struc_path = './example_POSCARs/aspirin.POSCAR.vasp'
name = 'aspirin'
miller_index = [1, 0, 0]
layers = [2, 3, 4]
highest_index = 1
vacuum = 15

slab_generator.cleave(struc_path, name, vacuum, layers, highest_index)