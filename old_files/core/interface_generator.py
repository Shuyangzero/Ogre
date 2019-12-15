import os
from itertools import product
import numpy as np
import math
from pymatgen.io.cif import CifWriter
from ase.build import general_surface
from ase.spacegroup import crystal
from ase.visualize import view
from ase.lattice.surface import *
from ase.io import *
import pymatgen as mg
from pymatgen.io.vasp.inputs import Poscar
import argparse
import pymatgen as mg
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.surface import Slab, SlabGenerator, ReconstructionGenerator
from pymatgen.analysis.substrate_analyzer import SubstrateAnalyzer, ZSLGenerator
from core.utils.utils import *

def get_equiv_transformations_Sam(self, transformation_sets, film_vectors,
                                  substrate_vectors):
    # Monkey-patching the original function of pymatgen to generate the transformation matrices

    text_file = open(self.working_dir + "film_sub_sets", "w")
    film_sub_sets = []
    for (film_transformations, substrate_transformations) in \
            transformation_sets:

        # Apply transformations and reduce using Zur reduce methodology
        films = [reduce_vectors(*np.dot(f, film_vectors))
                 for f in film_transformations]
        Sam_films = []
        for i in films:
            Sam_films.append(mat_clean(i))

        substrates = [reduce_vectors(*np.dot(s, substrate_vectors))
                      for s in substrate_transformations]

        sam_substrates = []
        for i in substrates:
            sam_substrates.append(mat_clean(i))
        # Check if equivelant super lattices

        for f, s in product(films, substrates):
            if self.is_same_vectors(f, s):
                f_index = Sam_films.index(mat_clean(f))
                s_index = sam_substrates.index(mat_clean(s))

                print([film_transformations[f_index].tolist(), substrate_transformations[s_index].tolist()],
                      file=text_file)

                film_sub_sets.append(
                    [film_transformations[f_index].tolist(), substrate_transformations[s_index].tolist()])
                yield [f, s]

    text_file.close()


def Interface_generator(Ini_sub_slab, Ini_film_slab, sub_tr_mat, film_tr_mat, distance, fparam):

    raw_ini_sub_slab_mat = np.array(Ini_sub_slab.lattice.matrix)
    raw_ini_film_slab_mat = np.array(Ini_film_slab.lattice.matrix)
    sub_reduction = reduce_vectors(
        raw_ini_sub_slab_mat[0], raw_ini_sub_slab_mat[1])
    film_reduction = reduce_vectors(
        raw_ini_film_slab_mat[0], raw_ini_film_slab_mat[1])
    reduced_sub_mat = np.array(
        [sub_reduction[0], sub_reduction[1], raw_ini_sub_slab_mat[2]])
    reduced_film_mat = np.array(
        [film_reduction[0], film_reduction[1], raw_ini_film_slab_mat[2]])
    red_Ini_sub_slab = Structure(mg.Lattice(reduced_sub_mat), Ini_sub_slab.species, Ini_sub_slab.cart_coords,
                                 coords_are_cartesian=True)
    red_Ini_film_slab = Structure(mg.Lattice(reduced_film_mat), Ini_film_slab.species, Ini_film_slab.cart_coords,
                                  coords_are_cartesian=True)
    red_Ini_sub_slab.make_supercell(scaling_matrix=scale_mat(sub_tr_mat))
    red_Ini_film_slab.make_supercell(scaling_matrix=scale_mat(film_tr_mat))
    Ini_sub_mat = red_Ini_sub_slab.lattice.matrix
    Ini_film_mat = red_Ini_film_slab.lattice.matrix
    sub_r_vecs = reduce_vectors(Ini_sub_mat[0], Ini_sub_mat[1])
    film_r_vecs = reduce_vectors(Ini_film_mat[0], Ini_film_mat[1])
    sub_mat = np.array([sub_r_vecs[0], sub_r_vecs[1], Ini_sub_mat[2]])
    film_mat = np.array([film_r_vecs[0], film_r_vecs[1], Ini_film_mat[2]])
    modif_sub_struc = mg.Structure(mg.Lattice(sub_mat), red_Ini_sub_slab.species, red_Ini_sub_slab.cart_coords,
                                   coords_are_cartesian=True)
    modif_film_struc = mg.Structure(mg.Lattice(film_mat), red_Ini_film_slab.species, red_Ini_film_slab.cart_coords,
                                    coords_are_cartesian=True)
    sub_sl_vecs = [modif_sub_struc.lattice.matrix[0],
                   modif_sub_struc.lattice.matrix[1]]
    film_sl_vecs = [modif_film_struc.lattice.matrix[0],
                    modif_film_struc.lattice.matrix[1]]
    film_angel = angle(film_sl_vecs[0], film_sl_vecs[1])
    sub_angel = angle(sub_sl_vecs[0], sub_sl_vecs[1])
    u_size = fparam * \
        (np.linalg.norm(sub_sl_vecs[0])) + (1 -
                                            fparam) * (np.linalg.norm(film_sl_vecs[0]))
    v_size = fparam * \
        (np.linalg.norm(sub_sl_vecs[1])) + (1 -
                                            fparam) * (np.linalg.norm(film_sl_vecs[1]))
    mean_angle = fparam * sub_angel + (1 - fparam) * film_angel
    sub_rot_mat = [[u_size, 0, 0], [v_size * math.cos(mean_angle), v_size * math.sin(mean_angle), 0],
                   [0, 0, np.linalg.norm(modif_sub_struc.lattice.matrix[2])]]
    film_rot_mat = [[u_size, 0, 0], [v_size * math.cos(mean_angle), v_size * math.sin(mean_angle), 0],
                    [0, 0, -np.linalg.norm(modif_film_struc.lattice.matrix[2])]]
    film_normal = np.cross(film_sl_vecs[0], film_sl_vecs[1])
    sub_normal = np.cross(sub_sl_vecs[0], sub_sl_vecs[1])
    film_un = film_normal / np.linalg.norm(film_normal)
    sub_un = sub_normal / np.linalg.norm(sub_normal)
    film_sl_vecs.append(film_un)
    L1_mat = np.transpose(film_sl_vecs)
    L1_res = [[u_size, v_size * math.cos(mean_angle), 0],
              [0, v_size * math.sin(mean_angle), 0], [0, 0, 1]]
    L1_mat_inv = np.linalg.inv(L1_mat)
    L1 = np.matmul(L1_res, L1_mat_inv)
    sub_sl_vecs.append(sub_un)
    L2_mat = np.transpose(sub_sl_vecs)
    L2_res = [[u_size, v_size * math.cos(mean_angle), 0],
              [0, v_size * math.sin(mean_angle), 0], [0, 0, -1]]
    L2_mat_inv = np.linalg.inv(L2_mat)
    L2 = np.matmul(L2_res, L2_mat_inv)
    sub_rot_lattice = mg.Lattice(sub_rot_mat)
    film_rot_lattice = mg.Lattice(film_rot_mat)
    r_sub_coords = np.array(modif_sub_struc.cart_coords)
    r_film_coords = np.array(modif_film_struc.cart_coords)

    for ii in range(len(r_sub_coords)):
        r_sub_coords[ii] = np.matmul(L2, r_sub_coords[ii])
    for ii in range(len(r_film_coords)):
        r_film_coords[ii] = np.matmul(L1, r_film_coords[ii])

    sub_slab = mg.Structure(
        sub_rot_lattice, modif_sub_struc.species, r_sub_coords, coords_are_cartesian=True)
    film_slab = mg.Structure(
        film_rot_lattice, modif_film_struc.species, r_film_coords, coords_are_cartesian=True)
    # SP_text = open(working_dir+ "SP_typ" , "w")
    sub_sp_num = len(sub_slab.types_of_specie)
    film_sp_num = len(film_slab.types_of_specie)
    # SP_text.close()
    sub_slab_mat = np.array(sub_slab.lattice.matrix)
    film_slab_mat = np.array(film_slab.lattice.matrix)
    sub_slab_coords = sub_slab.cart_coords
    film_slab_coords = film_slab.cart_coords
    sub_slab_zmat = sub_slab_coords[:, [2]]
    film_slab_zmat = film_slab_coords[:, [2]]
    sub_slab_zmat = sub_slab_zmat - min(sub_slab_zmat)
    film_slab_zmat = film_slab_zmat - min(film_slab_zmat)
    sub_max_z = max(sub_slab_zmat)
    sub_min_z = min(sub_slab_zmat)
    modif_film_slab_zmat = film_slab_zmat + sub_max_z - sub_min_z + distance
    film_slab_coords[:, [2]] = modif_film_slab_zmat
    sub_slab_coords[:, [2]] = sub_slab_zmat

    sub_max_z = max(sub_slab_zmat)
    film_min_z = min(modif_film_slab_zmat)
    sub_max_list = coords_sperator(sub_slab_zmat, sub_sp_num, True)
    film_min_list = coords_sperator(modif_film_slab_zmat, film_sp_num, False)

    interface_coords = np.concatenate(
        (sub_slab_coords, film_slab_coords), axis=0)
    interface_species = sub_slab.species + film_slab.species
    interface_latt = sub_slab_mat
    interface_latt[2][2] = abs(sub_slab_mat[2][2]) + \
        abs(film_slab_mat[2][2]) + distance

    Adding_val = 0.5 * (interface_latt[2][2] - max(interface_coords[:, [2]]))
    sub_max_list += Adding_val
    film_min_list += Adding_val
    sub_max_z += Adding_val
    film_min_z += Adding_val

    interface_coords[:, [2]] += 0.5 * \
        (interface_latt[2][2] - max(interface_coords[:, [2]]))
    sub_slab_coords[:, [2]] += Adding_val
    film_slab_coords[:, [2]] += Adding_val
    # sub_slab_coords[:, [2]] += 0.5 * \
    #     (interface_latt[2][2] - max(sub_slab_coords[:, [2]]))
    # film_slab_coords[:, [2]] += 0.5 * \
    #     (interface_latt[2][2] - max(film_slab_coords[:, [2]]))
    interface_lattice = mg.Lattice(interface_latt)
    interface_struc = mg.Structure(
        interface_lattice, interface_species, interface_coords, coords_are_cartesian=True)
    interface_struc = interface_struc.get_reduced_structure()
    # Poscar(interface_struc.get_reduced_structure()).write_file(working_dir +  "POSCAR_interface", direct=False )

    ###################
    # Seperate the first two layers

    surf_int_species = []
    surf_int_coords = []

    for k in range(len(interface_coords)):
        for k1 in sub_max_list:
            if (round(interface_coords[k, 2], 8) == round(k1[0], 8)):
                surf_int_coords.append(interface_coords[k, :])
                surf_int_species.append(interface_species[k])
        for k2 in film_min_list:
            if (round(interface_coords[k, 2], 8) == round(k2[0], 8)):
                surf_int_coords.append(interface_coords[k, :])
                surf_int_species.append(interface_species[k])

    surf_int_coords = np.array(surf_int_coords)
    surf_int_coords[:, 2] -= min(surf_int_coords[:, 2])
    surf_int_lat = np.array([interface_latt[0], interface_latt[1], [
                            0, 0, 15*max(surf_int_coords[:, 2])]])
    surf_int_coords[:, 2] += 7 * max(surf_int_coords[:, 2])
    surf_struc = Structure(surf_int_lat, surf_int_species,
                           surf_int_coords, coords_are_cartesian=True)
    # surf_cif = CifWriter(surf_struc)
    # surf_cif.write_file(working_dir+"surf.cif")
    surf_struc = surf_struc.get_reduced_structure()
    # Poscar(surf_struc.get_reduced_structure()).write_file(working_dir + "POSCAR_Surf_int", direct = False)

    return [interface_struc, surf_struc, sub_slab_coords, film_slab_coords]