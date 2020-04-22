from ogre import generators
import argparse
from configparser import ConfigParser
from ase.io import read, write
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', dest='filename', type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()
    filename = args.filename
    config = ConfigParser()
    config.read(filename, encoding='UTF-8')
    io = config['io']
    parameters = config['parameters']
    methods = config['methods']
    structure_path = io['structure_path']
    structure_name = io['structure_name']
    format_string = io['format']
    cleave_option = int(methods['cleave_option'])
    layers_string = parameters['layers']
    miller_index = [int(x) for x in parameters['miller_index'].split(" ")]
    list_of_layers = []
    for item in layers_string.split(' '):
        if item:
            if '-' in item:
                start, end = item.split('-')
                list_of_layers.extend(list(range(int(start), int(end) + 1)))
            else:
                list_of_layers.append(int(item))
    highest_index = int(parameters['highest_index'])
    vacuum_size = int(parameters['vacuum_size'])
    supercell_size = parameters['supercell_size'].split(' ')
    supercell_size = None if len(supercell_size) < 3 else [
        int(x) for x in supercell_size]
    if not os.path.isdir(structure_name):
        os.mkdir(structure_name)
    initial_structure = read(structure_path)

    if cleave_option == 0:
        print("Cleave single surface")
        generators.atomic_task(structure_name, initial_structure, miller_index,
                               list_of_layers, vacuum_size, supercell_size, format_string)
    elif cleave_option == 1:
        print("Cleave surfaces for surface energy calculations")
        generators.cleave_for_surface_energies(
            structure_path, structure_name, vacuum_size, list_of_layers, highest_index, supercell_size, format_string)
if __name__ == "__main__":
    main()