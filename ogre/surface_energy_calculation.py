from ogre import slab_generator
from ogre.utils.unique_planes import UniquePlanes
from pymatgen.io.vasp.inputs import Poscar
import os
from ase.io import write, read
import numpy
from multiprocessing import Pool
import shutil


def task(name, struc, miller_index, layers, vacuum,
         super_cell):
    dir_name = '{}_{}_{}'.format(name, "".join(str(int(x))
                                               for x in miller_index), str(layers))
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    working_dir = os.path.abspath('./{}'.format(dir_name))
    print("start {} {}".format("".join(str(int(x))
                                       for x in miller_index), str(layers)))
    slab_list = slab_generator.repair_organic_slab_generator_move(struc, miller_index, layers, vacuum, working_dir,
                                                                  super_cell)
    poscar_str = "output/POSCAR.{}.{}.{}".format(name, "".join(str(int(x))
                                                               for x in miller_index), str(layers))
    print(poscar_str)
    Poscar(slab_list[0]).write_file(poscar_str)
    slab_ase = read(poscar_str)
    os.remove(poscar_str)
    write("output/{}.{}.{}.in".format(name, "".join(str(int(x))
                                                    for x in miller_index), str(layers)), slab_ase)
    print("finish {} {}".format("".join(str(int(x))
                                        for x in miller_index), str(layers)))
    shutil.rmtree(working_dir)


path = '../example_POSCARs/PENCEN.POSCAR.vasp'
name = 'PENCEN'
bulk = read(path)
vacuum = 100
if not os.path.isdir('output'):
    os.mkdir('output')

up = UniquePlanes(bulk, index=2, verbose=False)
p = Pool()
print("{} unique planes are found".format(len(up.unique_idx)))
for miller_index in up.unique_idx:
    for layers in range(2, 5, 1):
        p.apply_async(task, args=(name, bulk, miller_index,
                                  layers, vacuum, None,))
p.close()
p.join()
