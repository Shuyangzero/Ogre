from ase.io import read, write
from ase.lattice.surface import *
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.vasp.inputs import Poscar
from ogre.utils.utils import *


@timeTest
def inorganic_slab_generator(struc, miller_index, no_layers, vacuum, working_dir,
                             super_cell):
    """
    Generate inorganic surface by using ase library.

    Parameters
    ----------
    struc : Atoms structure or list of atoms structures
        The original bulk structure.
    miller_index : list of int, [h, k, l]
        The Miller Index of target surface.
    no_layers : int
        Number of surface's layers.
    vacuum : double
        Height of vacuum layer, unit: Angstrom. Notice that the vacuum layer
        would be added to both the bottom and the top of surface.
    working_dir : string
        The path of original bulk's file.
    super_cell : list of int, [a, b, 1]
        Make a (a * b * 1) supercell.
    """
    surfaces_list = []
    run = True
    term = 0
    while run:
        try:
            slab = surface(struc, miller_index, no_layers,
                           vacuum=vacuum, termination=term)
            write(working_dir + 'slab_raw', slab, format="vasp")
            f1 = open(working_dir + 'slab_raw', 'r')
            f2 = open(working_dir + 'tmp', 'w')
            index = 0
            for line in f1:
                if index == 0:
                    tmp = line
                    f2.write('slab\n')
                elif index == 5:
                    f2.write(tmp)
                    f2.write(line)
                else:
                    f2.write(line)
                index += 1
            f1.close()
            f2.close()

            os.rename(working_dir + 'tmp', working_dir + 'POSCAR_slab')
            slab = mg.Structure.from_file(working_dir + 'POSCAR_slab')
            # os.rename(working_dir + 'tmp', working_dir + 'POSCAR_slab_' + str(term) )
            # slab = mg.Structure.from_file(working_dir + 'POSCAR_slab_' + str(term))
            # slab = slab.get_reduced_structure()
            slab = slab.get_sorted_structure()
            surfaces_list.append(slab)
            os.remove(working_dir + 'POSCAR_slab')
            os.remove(working_dir + "slab_raw")
            term += 1
        except:
            run = False

    # Further filters out any surfaces made that might be the same
    tol = 0.01
    m = StructureMatcher(ltol=tol, stol=tol, primitive_cell=False,
                         scale=False)
    new_slabs = [g[0] for g in m.group_structures(surfaces_list)]

    match = StructureMatcher(ltol=tol, stol=tol, primitive_cell=False,
                             scale=False)
    new_slabs = [g[0] for g in match.group_structures(new_slabs)]
    if super_cell is not None:
        if super_cell[-1] != 1:
            print("Warning: Please extend c direction by cleaving more layers "
                  "rather than make supercell! The supercell is aotumatically "
                  "set to [" + str(super_cell[0]) + ", " + str(super_cell[1]) + ", " +
                  "1]!")
        super_cell_copy = deepcopy(super_cell)
        super_cell_copy[-1] = 1
        new_slabs = [slab.make_supercell(super_cell_copy) for slab in new_slabs]
    return new_slabs


@timeTest
def organic_slab_generator(struc, miller_index, no_layers, vacuum, working_dir,
                           super_cell):
    """
    Generate organic surface by deleting all those broken molecules.

    Parameters
    ----------
    struc : Atoms structure or list of atoms structures
        The original bulk structure.
    miller_index : list of int, [h, k, l]
        The Miller Index of target surface.
    no_layers : int
        Number of surface's layers.
    vacuum : double
        Height of vacuum layer, unit: Angstrom. Notice that the vacuum layer
        would be added to both the bottom and the top of surface.
    working_dir : string
        The path of original bulk's file.
    super_cell : list of int, [a, b, 1]
        Make a (a * b * 1) supercell.
    """
    tmp = None
    write(working_dir + 'bulk.POSCAR.vasp', struc, format="vasp")
    modify_poscar(working_dir + 'bulk.POSCAR.vasp')
    slab = surface(struc, miller_index, no_layers,
                   vacuum=vacuum)
    write(working_dir + 'slab_raw', slab, format="vasp")
    f1 = open(working_dir + 'slab_raw', 'r')
    f2 = open(working_dir + 'tmp', 'w')
    index = 0
    for line in f1:
        if index == 0:
            tmp = line
            f2.write('slab\n')
        elif index == 5:
            f2.write(tmp)
            f2.write(line)
        else:
            f2.write(line)
        index += 1
    f1.close()
    f2.close()

    os.rename(working_dir + 'tmp', working_dir + 'POSCAR_slab')
    slab = mg.Structure.from_file(working_dir + 'POSCAR_slab')
    slab = slab.get_sorted_structure()
    crystal = mg.Structure.from_file(working_dir + 'bulk.POSCAR.vasp')
    os.remove(working_dir + 'POSCAR_slab')
    os.remove(working_dir + "slab_raw")
    os.remove(working_dir + 'bulk.POSCAR.vasp')
    bulk_sg = StructureGraph.with_local_env_strategy(crystal, JmolNN())
    print("the bulk_sg is:", type(bulk_sg))
    # structure with bonding information. use method with_local_env_strategy,
    # which takes an structure oject and jmolNN oject which define the near-neigbors sites
    slab_sg = StructureGraph.with_local_env_strategy(slab, JmolNN())
    # Construct undirected graph for bulk
    # return a list of networkx graph objects
    _, bulk_graphs = get_bulk_molecules(bulk_sg)
    print("the unique molecule in bulk is:", _)
    # retrieve subgraphs as unique Molecule objects
    sg = bulk_sg.get_subgraphs_as_molecules()
    # Construct undirected graph for slab
    # and eliminate those presenting in bulk to get broken molecules
    slab_molecules = get_broken_molecules(slab_sg, bulk_graphs)
    slab_molecules = double_screen(slab_molecules, sg)
    delete_sites = reduced_sites(slab_molecules, slab)
    delete_list = []
    # Obtain delete_sites
    # but those sites may be periodic_images in the slab
    for delete_site in delete_sites:
        for i, atom in enumerate(slab):
            if atom.is_periodic_image(delete_site):
                delete_list.append(i)
                break
    slab.remove_sites(delete_list)
    if super_cell is not None:
        if super_cell[-1] != 1:
            print("Warning: Please extend c direction by cleaving more layers "
                  "rather than make supercell! The supercell is aotumatically "
                  "set to [" + str(super_cell[0]) + ", " + str(super_cell[1]) + ", " +
                  "1]!")
        super_cell_copy = deepcopy(super_cell)
        super_cell_copy[-1] = 1
        slab.make_supercell(super_cell_copy)
    return [slab.get_sorted_structure()]


@timeTest
def repair_organic_slab_generator_graph(struc, miller_index,
                                        no_layers, vacuum,
                                        working_dir, super_cell=None):
    """
    Repair the broken molecules by graph_repair method. The basic idea is
    finding the corresponding intact molecules and then replacing those broken
    molecules with the intact molecules. This is an alternative method to repair
    broken molecules since it has less restrictions than
    repair_organic_slab_generator_move. So, when move_method is not working,
    please try this method.

    For example: Cleaving a miller index surface ([2, 3, 1]) with a number of
    layers (1, 2, ...).

    Parameters
    ----------
    struc : Atoms structure or list of atoms structures
        The original bulk structure.
    miller_index : list of int, [h, k, l]
        The Miller Index of target surface.
    no_layers : int
        Number of surface's layers.
    vacuum : double
        Height of vacuum layer, unit: Angstrom. Notice that the vacuum layer
        would be added to both the bottom and the top of surface.
    working_dir : string
        The path of original bulk's file.
    super_cell : list of int, [a, b, 1]
        Make a (a * b * 1) supercell.
    """
    write(working_dir + '/bulk.POSCAR.vasp', struc, format="vasp")
    modify_poscar(working_dir + '/bulk.POSCAR.vasp')
    bulk = mg.Structure.from_file(working_dir + '/bulk.POSCAR.vasp')
    super_structure_sg = StructureGraph.with_local_env_strategy(bulk,
                                                                JmolNN())
    bulk_structure_sg = super_structure_sg * (3, 3, 3)
    unique_bulk_subgraphs, molecules = \
        get_bulk_subgraphs_unique(bulk_structure_sg)
    print("There would be {} different molecules in bulk".format(str(len(molecules))))
    slab = surface(struc, miller_index, layers=no_layers, vacuum=vacuum)
    output_file = working_dir + '/slab_original.POSCAR.vasp'
    format_ = 'vasp'
    write(output_file, format=format_, images=slab)
    modify_poscar(output_file)
    slab = mg.Structure.from_file(output_file)
    os.remove(output_file)
    os.remove(working_dir + '/bulk.POSCAR.vasp')
    slab_sg = StructureGraph.with_local_env_strategy(slab, JmolNN())
    slab_supercell_sg = slab_sg * (3, 3, 1)
    different_subgraphs_in_slab, slab_molecules = \
        get_slab_different_subgraphs(slab_supercell_sg, unique_bulk_subgraphs)
    # sg = super_structure_sg.get_subgraphs_as_molecules()
    sg = molecules
    slab_molecules = double_screen(slab_molecules, sg)
    new_different_subgraphs = []
    less_new_different_subgraphs = []
    for subgraph in different_subgraphs_in_slab:
        one_neribs = []
        for n, nbrs in subgraph.adjacency():
            neribs = []
            for nbr, _ in nbrs.items():
                neribs.append(nbr)
            one_neribs.append(neribs)
        tmp = [len(one_nerib) == 3 for one_nerib in one_neribs]
        if len(subgraph.nodes()) >= 4:
            if any(tmp):
                new_different_subgraphs.append(subgraph)
            else:
                less_new_different_subgraphs.append(subgraph)
    broken_subgraphs, intact_subgraphs = \
        brokenMolecules_and_corresspoundingIntactMolecules(new_different_subgraphs,
                                                           unique_bulk_subgraphs)
    less_broken_subgraphs, less_intact_subgraphs = \
        brokenMolecules_and_corresspoundingIntactMolecules(less_new_different_subgraphs,
                                                           unique_bulk_subgraphs)
    delete_sites = reduced_sites(slab_molecules, slab)
    delete_list = []
    c_fracs = np.array(slab.frac_coords[:, 2])
    c_frac_min = min(c_fracs) - 0.03
    # Obtain delete_sites
    # but those sites may be periodic_images in the slab
    for delete_site in delete_sites:
        for i, atom in enumerate(slab):
            if atom.is_periodic_image(delete_site):
                delete_list.append(i)
                break
    slab.remove_sites(delete_list)
    fix_c_negative = False
    slab = fix_broken_molecules(broken_subgraphs, intact_subgraphs,
                                bulk_structure_sg, slab_supercell_sg, slab,
                                c_frac_min, fixed_c_negative=fix_c_negative)
    slab = less_fix_broken_molecules(less_broken_subgraphs, less_intact_subgraphs,
                                     bulk_structure_sg, slab_supercell_sg, slab,
                                     c_frac_min, fixed_c_negative=fix_c_negative)
    slab = put_everyatom_into_cell(slab)
    if super_cell is not None:
        if super_cell[-1] != 1:
            print("Warning: Please extend c direction by cleaving more layers "
                  "rather than make supercell! The supercell is aotumatically "
                  "set to [" + str(super_cell[0]) + ", " + str(super_cell[1]) + ", " +
                  "1]!")
        super_cell_copy = deepcopy(super_cell)
        super_cell_copy[-1] = 1
        slab.make_supercell(super_cell_copy)
    return [slab.get_sorted_structure()]


@timeTest
def repair_organic_slab_generator_move_onelayer(struc, miller_index,
                                                no_layers, vacuum, working_dir,
                                                super_cell=None):
    """
    Repair the broken molecules by move_repair method. The idea is based on the
    periodicity of original bulk, and use the unchanged periodicity to repair those
    broken molecoles. Attention: This method might be not working well while cleaving
    a high index miller surface with a small number of layer. However, a bigger number
    of layers could assure that this method is working. Please try
    repair_organic_slab_generator_graph once this method fails!

    For example: Cleaving a low miller index ([2, 1, 1]) surface with an
    appropriate number (3, 4 ...) of layer

    Parameters
    ----------
    struc : Atoms structure or list of atoms structures
        The original bulk structure.
    miller_index : list of int, [h, k, l]
        The Miller Index of target surface.
    no_layers : int
        Number of surface's layers.
    vacuum : double
        Height of vacuum layer, unit: Angstrom. Notice that the vacuum layer
        would be added to both the bottom and the top of surface.
    working_dir : string
        The path of original bulk's file.
    super_cell : list of int, [a, b, 1]
        Make a (a * b * 1) supercell.
    """
    write(working_dir + '/bulk.POSCAR.vasp', struc, format="vasp")
    modify_poscar(working_dir + '/bulk.POSCAR.vasp')
    bulk = mg.Structure.from_file(working_dir + '/bulk.POSCAR.vasp')
    super_structure_sg = StructureGraph.with_local_env_strategy(bulk,
                                                                JmolNN())
    bulk_structure_sg = super_structure_sg * (3, 3, 3)
    unique_bulk_subgraphs, molecules = \
        get_bulk_subgraphs_unique(bulk_structure_sg)
    print("There would be {} different molecules in bulk".format(str(len(molecules))))
    # get the slab via ase and deal with it via pymatgen
    os.remove(working_dir + '/bulk.POSCAR.vasp')
    slab = surface(struc, miller_index, layers=no_layers, vacuum=vacuum)
    file_name = working_dir + "/ASE_surface.POSCAR.vasp"
    format_ = 'vasp'
    write(file_name, format=format_, images=slab)
    modify_poscar(file_name)
    slab_temp = mg.Structure.from_file(file_name)
    # attention! the slab is assigned to a new object

    virtual_layers = 4
    slab = surface_self_defined(struc, miller_index, layers=virtual_layers)
    delta = np.array(slab.cell)[2, :]
    # if vacuum is not None:
    slab.center(vacuum=1000, axis=2)

    file_name = working_dir + '/slab_before.POSCAR.vasp'
    write(file_name, format=format_, images=slab)
    modify_poscar(file_name)
    slab_move = mg.Structure.from_file(file_name)
    os.remove(file_name)
    slab_move = handle_with_molecules(slab_move, delta, down=True)
    Poscar(slab_move.get_sorted_structure()).write_file(working_dir + "/AlreadyMove.POSCAR.vasp")
    # delete intact molecule in slab_move
    slab = slab_move
    species_intact, coords_intact = [], []
    # os.remove(output_file)
    # super_structure_sg = StructureGraph.with_local_env_strategy(bulk,
    #                                                             JmolNN())
    # sg = super_structure_sg.get_subgraphs_as_molecules()
    sg = molecules
    Find_Broken_Molecules(slab, sg, species_intact, coords_intact, unique_bulk_subgraphs)
    # find the broken molecules for the first minor movement and delete the intact molecules
    try:
        slab = put_everyatom_into_cell(slab)
        Poscar(slab.get_sorted_structure()).write_file(working_dir + "/POSCAR_Broken.POSCAR.vasp")
        os.remove(working_dir + "/POSCAR_Broken.POSCAR.vasp")
        slab = handle_with_molecules(slab, delta, down=False)
    except ValueError:
        # No broken molecules anymore. So, return the slab_move
        slab_move = get_one_layer(slab_move, layers_virtual=virtual_layers)
        # slab_move = read(working_dir + "/AlreadyMove.POSCAR.vasp")
        # slab_move.center(vacuum=vacuum, axis=2)
        os.remove(working_dir + "/AlreadyMove.POSCAR.vasp")
        # slab_move = modify_cell(slab_move)
        temp_file_name = working_dir + "/temp.POSCAR.vasp"
        write(temp_file_name, slab_move)
        modify_poscar(temp_file_name)
        slab_move = mg.Structure.from_file(temp_file_name)
        os.remove(temp_file_name)
        print("No Broken molecules!")
        # if super_cell is not None:
        #     if super_cell[-1] != 1:
        #         print("Warning: Please extend c direction by cleaving more layers "
        #               "rather than make supercell! The supercell is aotumatically "
        #               "set to [" + str(super_cell[0]) + ", " + str(super_cell[1]) + ", " +
        #               "1]!")
        #     super_cell_copy = deepcopy(super_cell)
        #     super_cell_copy[-1] = 1
        #     slab_move.make_supercell(super_cell_copy)
        return [slab_move.get_sorted_structure()]
    os.remove(working_dir + "/AlreadyMove.POSCAR.vasp")

    Find_Broken_Molecules(slab, sg, species_intact, coords_intact, unique_bulk_subgraphs)
    try:
        slab = put_everyatom_into_cell(slab)
        Poscar(slab.get_sorted_structure()).write_file(working_dir + "/POSCAR_Broken_two.POSCAR.vasp")
        os.remove(working_dir + "/POSCAR_Broken_two.POSCAR.vasp")
    except ValueError:
        for i in range(len(species_intact)):
            slab.append(species_intact[i], coords_intact[i], coords_are_cartesian=True)
        slab = get_one_layer(slab, virtual_layers)
        temp_file_name = working_dir + "/temp.POSCAR.vasp"
        write(temp_file_name, slab)
        modify_poscar(temp_file_name)
        slab = mg.Structure.from_file(temp_file_name)
        os.remove(temp_file_name)
        print("No Broken molecules!")
        # temp_file_name = working_dir + "/temp.POSCAR.vasp"
        # Poscar(slab.get_sorted_structure()).write_file(temp_file_name)
        # slab = read(temp_file_name)
        # slab.center(vacuum=vacuum, axis=2)
        # os.remove(temp_file_name)
        # slab = modify_cell(slab)
        # write(temp_file_name, slab)
        # modify_poscar(temp_file_name)
        # slab = mg.Structure.from_file(temp_file_name)
        # os.remove(temp_file_name)
        # print("No Broken molecules!")
        # if super_cell is not None:
        #     if super_cell[-1] != 1:
        #         print("Warning: Please extend c direction by cleaving more layers "
        #               "rather than make supercell! The supercell is aotumatically "
        #               "set to [" + str(super_cell[0]) + ", " + str(super_cell[1]) + ", " +
        #               "1]!")
        #     super_cell_copy = deepcopy(super_cell)
        #     super_cell_copy[-1] = 1
        #     slab.make_supercell(super_cell_copy)
        return [slab.get_sorted_structure()]

    speices = slab.species
    slab_coords = slab.frac_coords
    slab_coords_cart = slab.cart_coords

    for i, coord in enumerate(slab_coords):
        new_cart_coords = np.array(slab_coords_cart[i]) + delta
        # move the slab to match broken molecules
        slab.append(speices[i], coords=new_cart_coords, coords_are_cartesian=True)

    try:
        for i in range(len(species_intact)):
            slab.append(species_intact[i], coords_intact[i], coords_are_cartesian=True)
        file_name = working_dir + '/POSCAR_move.vasp'
        Poscar(slab.get_sorted_structure()).write_file(file_name)
        slab = mg.Structure.from_file(file_name)
        os.remove(file_name)

        slab_sg = StructureGraph.with_local_env_strategy(slab, JmolNN())
        super_structure_sg = StructureGraph.with_local_env_strategy(bulk,
                                                                    JmolNN())
        bulk_structure_sg = super_structure_sg * (3, 3, 3)
        unique_bulk_subgraphs, molecules = \
            get_bulk_subgraphs_unique(bulk_structure_sg)

        slab_supercell_sg = slab_sg * (3, 3, 1)
        different_subgraphs_in_slab, slab_molecules = \
            get_slab_different_subgraphs(slab_supercell_sg, unique_bulk_subgraphs)
        # sg = super_structure_sg.get_subgraphs_as_molecules()
        sg = molecules
        slab_molecules = double_screen(slab_molecules, sg)
        print("The number of molecules that need to be fixed : ", len(slab_molecules))
        # slab_molecules are the molecules that are broken and need to be fixed
        delete_sites = reduced_sites(slab_molecules, slab)
        delete_list = []

        for delete_site in delete_sites:
            for i, atom in enumerate(slab):
                if atom.is_periodic_image(delete_site):
                    delete_list.append(i)
                    break
        slab.remove_sites(delete_list)
    except ValueError:
        print("No Broken molecules!")

    # for i in range(len(species_intact)):
    #     slab.append(species_intact[i], coords_intact[i], coords_are_cartesian=True)

    file_name = working_dir + "/POSCAR_move_final.vasp"
    os.remove(working_dir + "/ASE_surface.POSCAR.vasp")
    try:
        slab = get_one_layer(slab, virtual_layers)
        output_file = working_dir + "/Orge_surface.POSCAR.vasp"
        write(output_file, slab)
        modify_poscar(output_file)
        slab = mg.Structure.from_file(output_file)
        os.remove(output_file)
        # Poscar(slab.get_sorted_structure()).write_file(file_name)
        # structure = read(file_name, format=format_)
        # structure.center(vacuum=vacuum, axis=2)
        # os.remove(file_name)
        # slab = modify_cell(structure)
        # output_file = working_dir + "/Orge_surface.POSCAR.vasp"
        # write(output_file, slab, format=format_)
        # modify_poscar(output_file)
        # slab = mg.Structure.from_file(output_file)
        # os.remove(output_file)
        # if super_cell is not None:
        #     if super_cell[-1] != 1:
        #         print("Warning: Please extend c direction by cleaving more layers "
        #               "rather than make supercell! The supercell is aotumatically "
        #               "set to [" + str(super_cell[0]) + ", " + str(super_cell[1]) + ", " +
        #               "1]!")
        #     super_cell_copy = deepcopy(super_cell)
        #     super_cell_copy[-1] = 1
        #     slab.make_supercell(super_cell_copy)
        return [slab.get_sorted_structure()]
    except ValueError:
        print("The {} slab with {} layers can not be reconstructed. And the result refers to ASE's surfaces. Please "
              "try the graph_repair method!".format(miller_index, no_layers))
        # if super_cell is not None:
        #     if super_cell[-1] != 1:
        #         print("Warning: Please extend c direction by cleaving more layers "
        #               "rather than make supercell! The supercell is aotumatically "
        #               "set to [" + str(super_cell[0]) + ", " + str(super_cell[1]) + ", " +
        #               "1]!")
        #     super_cell_copy = deepcopy(super_cell)
        #     super_cell_copy[-1] = 1
        #     slab_temp.make_supercell(super_cell_copy)
        return [slab_temp.get_sorted_structure()]


def change_layers_and_supercell(slab_list, no_layers, vacuum, working_dir, super_cell=None):
    surface_list = []
    slab_list = list(slab_list)
    for slab in slab_list:
        slab_one_layer_incline = deepcopy(slab)
        file_name = working_dir + "/one_layer.POSCAR.vasp"
        Poscar(slab_one_layer_incline.get_sorted_structure()).write_file(file_name)
        slab_one_layer_incline = read(file_name)
        os.remove(file_name)
        slab_several_layers = slab_one_layer_incline * (1, 1, no_layers)
        if vacuum is not None:
            slab_several_layers.center(vacuum=vacuum, axis=2)
        slab_several_layers = modify_cell(slab_several_layers)
        write(file_name, images=slab_several_layers)
        modify_poscar(file_name)
        slab_several_layers = mg.Structure.from_file(file_name)
        os.remove(file_name)
        if super_cell is not None:
            if super_cell[-1] != 1:
                print("Warning: Please extend c direction by cleaving more layers "
                      "rather than make supercell! The supercell is aotumatically "
                      "set to [" + str(super_cell[0]) + ", " + str(super_cell[1]) + ", " +
                      "1]!")
            super_cell_copy = deepcopy(super_cell)
            super_cell_copy[-1] = 1
            slab_several_layers.make_supercell(super_cell_copy)
        surface_list.append(slab_several_layers.get_sorted_structure())
    return surface_list
