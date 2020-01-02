from ase import io
from pymatgen.analysis.structure_matcher import StructureMatcher
from utils.utils import *
from pymatgen.io.vasp.inputs import Poscar
from ase.lattice.surface import *
from ase.io import *


@timeTest
def inorganic_slab_generator(struc, miller_index, no_layers, vacuum, working_dir):
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

    return new_slabs


@timeTest
def organic_slab_generator(struc, miller_index, no_layers, vacuum, working_dir):
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

    return [slab]


@timeTest
def repair_organic_slab_generator_graph(struc, miller_index,
                                        no_layers, vacuum,
                                        working_dir, super_cell=None):
    write(working_dir + '/bulk.POSCAR.vasp', struc, format="vasp")
    modify_poscar(working_dir + '/bulk.POSCAR.vasp')
    bulk = mg.Structure.from_file(working_dir + '/bulk.POSCAR.vasp')
    super_structure_sg = StructureGraph.with_local_env_strategy(bulk,
                                                                JmolNN())
    bulk_structure_sg = super_structure_sg * (3, 3, 3)
    unique_bulk_subgraphs, molecules = \
        get_bulk_subgraphs(bulk_structure_sg)
    print("There would be {} different molecules in bulk".format(str(len(molecules))))
    # get the slab via ase and deal with it via pymatgen
    slab = surface(struc, miller_index, layers=no_layers, vacuum=vacuum)
    output_file = working_dir + '/slab_original.POSCAR.vasp'
    format_ = 'vasp'
    write(output_file, format=format_, images=slab)
    # if format_ == 'vasp':
    #     updatePOSCAR(output_file)
    modify_poscar(output_file)
    slab = mg.Structure.from_file(output_file)
    os.remove(output_file)
    os.remove(working_dir + '/bulk.POSCAR.vasp')
    slab_sg = StructureGraph.with_local_env_strategy(slab, JmolNN())
    slab_supercell_sg = slab_sg * (3, 3, 1)
    different_subgraphs_in_slab, slab_molecules = \
        get_slab_different_subgraphs(slab_supercell_sg, unique_bulk_subgraphs)
    sg = super_structure_sg.get_subgraphs_as_molecules()
    slab_molecules = double_screen(slab_molecules, sg)
    # slab_molecules are the molecules that are broken and need to be fixed

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
    # c_frac_min = 0
    # try:
    #     c_fracs = np.array(slab.frac_coords[:, 2])
    #     c_frac_min = min(c_fracs)
    #     fix_c_negative = False
    # except IndexError:
    #     fix_c_negative = True
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
def repair_organic_slab_generator_move(struc, miller_index,
                                       no_layers, vacuum, working_dir,
                                       super_cell=None):
    write(working_dir + '/bulk.POSCAR.vasp', struc, format="vasp")
    modify_poscar(working_dir + '/bulk.POSCAR.vasp')
    bulk = mg.Structure.from_file(working_dir + '/bulk.POSCAR.vasp')
    super_structure_sg = StructureGraph.with_local_env_strategy(bulk,
                                                                JmolNN())
    bulk_structure_sg = super_structure_sg * (3, 3, 3)
    unique_bulk_subgraphs, molecules = \
        get_bulk_subgraphs(bulk_structure_sg)
    print("There would be {} different molecules in bulk".format(str(len(molecules))))
    # get the slab via ase and deal with it via pymatgen
    os.remove(working_dir + '/bulk.POSCAR.vasp')
    slab = surface(struc, miller_index, layers=no_layers, vacuum=vacuum)
    file_name = "ASE_surface.POSCAR.vasp"
    format_ = 'vasp'
    write(file_name, format=format_, images=slab)
    modify_poscar(file_name)
    # attention! the slab is assigned to a new object

    slab = surface_self_defined(struc, miller_index, layers=no_layers, vacuum=vacuum)
    delta = np.array(slab.cell)[2, :]
    if vacuum is not None:
        slab.center(vacuum=vacuum, axis=2)

    file_name = 'slab_before.POSCAR.vasp'
    write(file_name, format=format_, images=slab)
    modify_poscar(file_name)
    slab_move = mg.Structure.from_file(file_name)
    os.remove(file_name)
    slab_move = handle_with_molecules(slab_move, delta, down=True)
    Poscar(slab_move.get_sorted_structure()).write_file("AlreadyMove.POSCAR.vasp")
    os.remove("AlreadyMove.POSCAR.vasp")

    # delete intact molecule in slab_move
    slab = slab_move
    species_intact, coords_intact = [], []
    # os.remove(output_file)
    super_structure_sg = StructureGraph.with_local_env_strategy(bulk,
                                                                JmolNN())
    sg = super_structure_sg.get_subgraphs_as_molecules()
    Find_Broken_Molecules(slab, sg, species_intact, coords_intact, unique_bulk_subgraphs)
    try:
        slab = put_everyatom_into_cell(slab)
        Poscar(slab.get_sorted_structure()).write_file("POSCAR_Broken.POSCAR.vasp")
        os.remove("POSCAR_Broken.POSCAR.vasp")
        slab = handle_with_molecules(slab, delta, down=False)
    except ValueError:
        print("No Broken molecules!")

    Find_Broken_Molecules(slab, sg, species_intact, coords_intact, unique_bulk_subgraphs)
    try:
        slab = put_everyatom_into_cell(slab)
        Poscar(slab.get_sorted_structure()).write_file("POSCAR_Broken_two.POSCAR.vasp")
        os.remove("POSCAR_Broken_two.POSCAR.vasp")
    except ValueError:
        print("No Broken molecules!")

    speices = slab.species
    slab_coords = slab.frac_coords
    slab_coords_cart = slab.cart_coords
    # remove_sites = []

    for i, coord in enumerate(slab_coords):
        new_cart_coords = np.array(slab_coords_cart[i]) + delta
        # move the slab to match broken molecules
        slab.append(speices[i], coords=new_cart_coords, coords_are_cartesian=True)

    try:
        file_name = 'POSCAR_move.vasp'
        Poscar(slab.get_sorted_structure()).write_file(file_name)
        slab = mg.Structure.from_file(file_name)
        os.remove(file_name)

        slab_sg = StructureGraph.with_local_env_strategy(slab, JmolNN())
        super_structure_sg = StructureGraph.with_local_env_strategy(bulk,
                                                                    JmolNN())
        bulk_structure_sg = super_structure_sg * (3, 3, 3)
        unique_bulk_subgraphs, molecules = \
            get_bulk_subgraphs(bulk_structure_sg)

        slab_supercell_sg = slab_sg * (3, 3, 1)
        different_subgraphs_in_slab, slab_molecules = \
            get_slab_different_subgraphs(slab_supercell_sg, unique_bulk_subgraphs)
        sg = super_structure_sg.get_subgraphs_as_molecules()
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

    for i in range(len(species_intact)):
        slab.append(species_intact[i], coords_intact[i], coords_are_cartesian=True)

    file_name = 'POSCAR_move_final.vasp'
    try:
        Poscar(slab.get_sorted_structure()).write_file(file_name)
        structure = read(file_name, format=format_)
        os.remove(file_name)
        slab = modify_cell(structure)
        output_file = "Orge_surface.POSCAR.vasp"
        io.write(output_file, slab, format=format_)
        modify_poscar(output_file)
        slab = mg.Structure.from_file(output_file)
        os.remove(output_file)
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
    except ValueError:
        print("The slab can not be reconstructed, please refer to ASE_surface.POSCAR.vasp "
              "or try the graph_repair method!")
