import os
from core.interface_generator import *
from core.slab_generator import *
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--path', dest='path', type=str, default='./')
    # parser.add_argument('--isorganic', dest='isorganic',
    #                     type=bool, default=True)
    parser.add_argument('--repair', dest='repair', type=bool, default=False)
    return parser.parse_args()


'''
Usage: 
python main.py --path /test/organic/
python main.py --path /test/organic/ --repair True
python main.py --path /test/inorganic/
'''


def main():
    args = parse_arguments()
    working_dir = args.path
    ZSLGenerator.working_dir = working_dir
    ZSLGenerator.get_equiv_transformations = get_equiv_transformations_Sam
    Input_dat = open(working_dir+"Input.dat")
    Dat_list = []
    for i in Input_dat.readlines():
        if "------" in i:
            break
        Dat_list.append(i.split()[1])
    sub_name = working_dir + Dat_list[0]
    film_name = working_dir + Dat_list[1]
    isorganic = Dat_list[-1]
    if 'T' in isorganic or 't' in isorganic:
        if args.repair:
            Possible_surfaces_generator = repair_organic_slab_generator
        else:
            Possible_surfaces_generator = organic_slab_generator
    else:
        Possible_surfaces_generator = inorganic_slab_generator
    sub_miller = Dat_list[2]
    sub_miller = sub_miller.replace("(", "")
    sub_miller = sub_miller.replace(")", "")
    sub_miller = sub_miller.split(",")
    sub_miller = (int(sub_miller[0]), int(sub_miller[1]), int(sub_miller[2]))
    film_miller = Dat_list[3]
    film_miller = film_miller.replace("(", "")
    film_miller = film_miller.replace(")", "")
    film_miller = film_miller.split(",")
    film_miller = (int(film_miller[0]), int(
        film_miller[1]), int(film_miller[2]))
    sub_layers = int(Dat_list[4])
    film_layers = int(Dat_list[5])
    vacuum = int(Dat_list[6])
    max_area = int(Dat_list[7])
    max_area_ratio_tol = float(Dat_list[8])
    max_length_tol = float(Dat_list[9])
    max_angle_tol = float(Dat_list[10])
    distance = float(Dat_list[11])
    nStruc = int(Dat_list[12])
    shift_gen = bool(Dat_list[13])
    xy_range = Dat_list[14]
    z_range = Dat_list[16]
    try:
        if xy_range != "None":
            xy_range = xy_range.replace("[", "")
            xy_range = xy_range.replace("]", "")
            xy_range = xy_range.split(",")
            xy_range = [float(xy_range[0]), float(xy_range[1])]
        if z_range != "None":
            z_range = z_range.replace("[", "")
            z_range = z_range.replace("]", "")
            z_range = z_range.split(",")
            z_range = [float(z_range[0]), float(z_range[1])]
    except:
        pass

    xy_steps = int(Dat_list[15])
    z_steps = int(Dat_list[17])
    fparam = 0.5

    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    # start
    ####################################################################
    ase_substrate = read(sub_name)
    ase_film = read(film_name)
    pmg_substrate = Structure.from_file(sub_name, primitive=False)
    pmg_film = Structure.from_file(film_name, primitive=False)
    # Finding possible superlattice tranformation matrices

    ZSL = ZSLGenerator(max_area_ratio_tol=max_area_ratio_tol, max_area=max_area,
                       max_length_tol=max_length_tol, max_angle_tol=max_angle_tol)
    Sub_analyzer = SubstrateAnalyzer(ZSL, film_max_miller=10)
    Match_finder = Sub_analyzer.calculate(film=pmg_film, substrate=pmg_substrate, substrate_millers=[sub_miller], film_millers=[film_miller])
    Match_list = list(Match_finder)

    # Cleave slabs
    substrate_slabs = Possible_surfaces_generator(
        ase_substrate, sub_miller, sub_layers, vacuum, working_dir)
    film_slabs = Possible_surfaces_generator(
        ase_film, film_miller, film_layers, vacuum, working_dir)
    ##################################
    # importing generated matrices

    sets = open(working_dir + "film_sub_sets")
    film_sub_matices = []
    sets = sets.readlines()
    for x in sets:
        x = x.replace("[", "")
        x = x.replace("]", "")
        x = x.split(",")
        film_mat = [[float(x[0]), float(x[1])], [float(x[2]), float(x[3])]]
        sub_mat = [[float(x[4]), float(x[5])], [float(x[6]), float(x[7])]]
        film_sub_matices.append([film_mat, sub_mat])

    if len(film_sub_matices) <= nStruc:
        nStruc = len(film_sub_matices)

    for i in range(len(substrate_slabs)):
        for j in range(len(film_slabs)):
            ext = "Interfaces_Sub_" + str(i) + "_Film_" + str(j)
            ext_Dir = os.path.join(working_dir, ext)
            if(os.path.isdir(ext_Dir) == False):
                os.mkdir(ext_Dir)
            for k in range(nStruc):
                Interface = Interface_generator(substrate_slabs[i], film_slabs[j], film_sub_matices[k][1],
                                                film_sub_matices[k][0], distance, fparam)
                Iface = Interface[0]
                sub_coords = Interface[2]
                film_coords = Interface[3]
                Poscar(Iface).write_file(
                    ext_Dir + "/POSCAR_Iface_" + str(k), direct=False)
                if shift_gen:
                    if xy_range != "None":
                        x_grid_range = np.linspace(
                            xy_range[0], xy_range[1], xy_steps)
                        y_grid_range = np.linspace(
                            xy_range[0], xy_range[1], xy_steps)
                        x_ref = film_coords[:, 0].copy()
                        y_ref = film_coords[:, 1].copy()
                    else:
                        x_grid_range = [0]
                        y_grid_range = [0]
                    if z_range != "None":
                        z_grid_range = np.linspace(
                            z_range[0], z_range[1], xy_steps)
                        z_ref = film_coords[:, 2].copy()
                    else:
                        z_grid_range = [0]

                ext2 = "Interfaces_Sub_" + \
                    str(i) + "_Film_" + str(j)+"/Shifts_Iface_"+str(k)
                ext_Dir2 = os.path.join(working_dir, ext2)
                if (os.path.isdir(ext_Dir2) == False):
                    os.mkdir(ext_Dir2)

                if shift_gen:
                    for ii in range(len(x_grid_range)):
                        for jj in range(len(y_grid_range)):
                            for kk in range(len(z_grid_range)):
                                if xy_range != "None":
                                    film_coords[:, 0] = x_ref + \
                                        x_grid_range[ii]
                                    film_coords[:, 1] = y_ref + \
                                        y_grid_range[jj]
                                if z_range != "None":
                                    film_coords[:, 2] = z_ref + \
                                        z_grid_range[kk]
                                int_coords = np.concatenate(
                                    (sub_coords, film_coords), axis=0)
                                new_int = Structure(
                                    Iface.lattice.matrix, Iface.species, int_coords, coords_are_cartesian=True)
                                reduced_int = new_int.get_reduced_structure()
                                Poscar(reduced_int).write_file(ext_Dir2 + "/POSCAR_Iface_" + str(k)
                                                               + "_x_" + str(int(x_grid_range[ii] * 10)) + "_y_" + str(
                                    int(y_grid_range[jj] * 10)) + "_z_" + str(int(z_grid_range[kk]*10)), direct=False)

    os.remove(working_dir + "film_sub_sets")


if __name__ == "__main__":
    main()
