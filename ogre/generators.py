# -*- coding: UTF-8 -*-
from abc import ABC, abstractmethod
from ogre.utils.utils import from_ASE_to_pymatgen
import sys
from ase.io import read, write
from ase.build import surface
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.vasp.inputs import Poscar
from ogre.utils import utils
from ogre.utils.unique_planes import UniquePlanes
from multiprocessing import Pool
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from copy import deepcopy
from pymatgen.analysis.local_env import JmolNN
import shutil
from ase import io
import os
import numpy as np
import pymatgen as mg
from tqdm import tqdm
import networkx as nx


class SlabGenerator(ABC):
    def __init__(self, initial_structure, miller_index, list_of_layers,
                 vacuum_size, supercell_size, working_directory,
                 desired_num_of_molecules_oneLayer):

        self.initial_structure = initial_structure
        self.miller_index = miller_index
        self.list_of_layers = list_of_layers
        self.vacuum_size = vacuum_size
        self.supercell_size = supercell_size
        self.working_directory = working_directory
        self.desired_num_of_molecules_oneLayer = desired_num_of_molecules_oneLayer

    @abstractmethod
    def cleave(self):
        None

# TODO: create an inorganic slab generator.


class OrganicSlabGenerator(SlabGenerator):
    """
    Initialize the organic slab generator to cleave the surfaces for 
    certain number of layers and Miller index.

    Parameters
    ----------
    initial_structure: ASE Atoms structure.
        The initial bulk structure to be cleaved.
    miller_index: List[int]: [h, k, l]
        The Miller index of the surface plane.
    list_of_layers: List[int]. 
        A list of layers to cleave.
    vacuum_size: float
        Height of vacuum size, unit: Angstrom. Note that the vacuum size
        would be added to both the bottom and the top of surface.
    supercell_size: List[int]: [a, b, 1]
        Make a (a * b * 1) supercell.
    working_directory: str
        The path to save the resulting slab structures.
    """
    def __init__(self, 
                 initial_structure, 
                 miller_index, 
                 list_of_layers, 
                 vacuum_size, 
                 supercell_size, 
                 working_directory,
                 desired_num_of_molecules_oneLayer):
        super().__init__(initial_structure, miller_index, list_of_layers,
                         vacuum_size, supercell_size, working_directory,
                         desired_num_of_molecules_oneLayer)


    def cleave(self):
        """
        Cleave the slab and repair the broken molecules.

        Returns:
        --------
        List[List[structures]]
            list of list of slabs for the required list of layers. Each list 
            contains one or multiple terminations.
        """
        one_layer_slab, delta_cart = self._cleave_one_layer_v3()
        slab_list = []
        one_layer_slab = one_layer_slab[0]
        deletes = [0] * len(self.list_of_layers)
        if self.desired_num_of_molecules_oneLayer > 0:
            number = utils.number_of_molecules(one_layer_slab)
            new_layers = [(self.desired_num_of_molecules_oneLayer * layer - 1) // number + 1 for layer in self.list_of_layers]
            deletes = [new_layer * number - self.desired_num_of_molecules_oneLayer * layer
                       for new_layer, layer in zip(new_layers, self.list_of_layers)]
            self.list_of_layers = new_layers

        termination_list = self._surface_termination( one_layer_slab, delta_cart, None)

        for delete, layer in zip(deletes, self.list_of_layers):
            slabs_with_different_terminations = self._pile_to(
                termination_list, delta_cart, layer, c_perpendicular = True)
            slabs_with_different_terminations = [utils.delete_molecules(slab,
                                                                        self.working_directory,
                                                                        self.vacuum_size, delete)
                                                 for slab in slabs_with_different_terminations]
            slabs_with_different_terminations = self._supercell(slabs_with_different_terminations)
            slab_list.append(slabs_with_different_terminations)
        return slab_list


    def _supercell(self, slabs_with_different_terminations):
        """
        Make supercell based on reconstructed slab.

        Parameters:
        -----------
        slabs_with_different_terminations: List[pymatgen structure].
        
        Returns:
        --------
        slabs: List[pymatgen structure].
        """
        if self.supercell_size is not None:
            if self.supercell_size[-1] != 1:
                print("Warning: Please extend c direction by cleaving more layers "
                      "rather than make supercell! The supercell is automatically "
                      "set to [" + str(self.supercell_size[0]) + ", " + str(self.supercell_size[1]) + ", " +
                      "1]!")
            supercell_size_copy = deepcopy(self.supercell_size)
            supercell_size_copy[-1] = 1
            slabs = [slab.make_supercell(supercell_size_copy)
                     for slab in slabs_with_different_terminations]
            return slabs
        else:
            return slabs_with_different_terminations


    def _pile_to(self, termination_list, delta_cart, layer,
                 c_perpendicular=True, set_vacuum=True):
        """
        Generate multiple-layer slabs by piling up one-layer slabs.

        Parameters:
        -----------
        termination_list: List[pymatgen structure].
            One-layer slabs with different terminations.
        delta_cart: List[double].
            The differences between two adjacent layers in Cartesian
            Coordinates.
        layer: int.
            The number of layers.
        c_perpendicular: bool.
            c_perpendicular is set to True if c direction would be
            perpendicular to a-b side. Otherwise it is false. The defaule
            option if True.  

        Returns:
        --------
        surface_list: List[pymatgen structures]
            List of generated slabs with one specific number of layer.
        """
        surface_list = []
        slab_list = list(termination_list)
        for slab in slab_list:
            slab_one_layer_incline = deepcopy(slab)
            file_name = os.path.join(
                self.working_directory, "one_layer.POSCAR.vasp")
            Poscar(slab_one_layer_incline.get_sorted_structure()
                   ).write_file(file_name)
            slab_one_layer_incline = read(file_name,parallel=False)
            slab_one_layer_incline.center(vacuum=1000, axis=2)
            os.remove(file_name)
            write(file_name, images=slab_one_layer_incline,parallel=False)
            utils.modify_poscar(file_name)
            slab_one_layer_incline = mg.Structure.from_file(file_name)
            os.remove(file_name)
            # slab_several_layers = slab_one_layer_incline * (1, 1, layer)
            cart_coords = deepcopy(slab_one_layer_incline.cart_coords)
            species = deepcopy(slab_one_layer_incline.species)
            for i in range(layer - 1):
                for index, coord in enumerate(cart_coords):
                    new_coord = np.array(coord) + (i + 1) * \
                        np.array(delta_cart)
                    slab_one_layer_incline.append(species[index], new_coord,
                                                  coords_are_cartesian=True)
            Poscar(slab_one_layer_incline.get_sorted_structure()
                   ).write_file(file_name)
            slab_several_layers = read(file_name,parallel=False)
            os.remove(file_name)
            if set_vacuum == True:
                if self.vacuum_size is not None:
                    slab_several_layers.center(vacuum=self.vacuum_size, axis=2)
            if c_perpendicular is True:
                slab_several_layers = utils.modify_cell(slab_several_layers)
            write(file_name, images=slab_several_layers,parallel=False)
            utils.modify_poscar(file_name)
            slab_several_layers = mg.Structure.from_file(file_name)
            os.remove(file_name)
            # if self.supercell_size is not None:
            #     if self.supercell_size[-1] != 1:
            #         print("Warning: Please extend c direction by cleaving more layers "
            #               "rather than make supercell! The supercell is automatically "
            #               "set to [" + str(self.supercell_size[0]) + ", " + str(self.supercell_size[1]) + ", " +
            #               "1]!")
            #     supercell_size_copy = deepcopy(self.supercell_size)
            #     supercell_size_copy[-1] = 1
            #     slab_several_layers.make_supercell(supercell_size_copy)
            surface_list.append(slab_several_layers.get_sorted_structure())
        return surface_list

    def _cleave_one_layer(self, virtual_layers=4, virtual_vacuum=1000):
        """
        Main process to generate one-layer slab from bulk. The approach can be
        briefly decribed as below:
            1. Find all unique intact molecules from the supercell (3 x 3 x 3) of
               original bulk. We onlly need to analyze the molecules that within that boundary
               of original bulks or crosses the boundary due to periodicity.
            2. Use ASE to cleave raw slabs with "virtual_layers" number of
               layer.
            3. After comparing all (slab) molecules with intact molecules, copy
               all broken molecules and translate by v3 to repair broken molecules
               that in the upper side. 
            4. Identify redisual fragments and delete broken molecules.
            5. Extract one-layer slab from new slabs based on periodicity.
               There are "virtual_layers" identical layers in the slab and we just
               need one layer of them.
        More details about the approach could refer to paper: "Ogre: A Python Package 
        for Molecular Crystal Surface Generation with Applications to Surface Energy and
        Crystal Habit Prediction", FIG 3. and section B: Broken Molecule Reconstruction

        Parameters:
        -----------
        virtual_layers: int.
            The number of layers of raw slabs. The virtual_layers is set
            to 4 to avoid that some molecules are cut by both upper and lower
            boundary for more than one time. ATTENTION: This parameter should
            be optimized to be self-adaptive to very high Miller index.
            Otherwise, please moderately increase this number, i.e, to 8, when we
            need to cleave high Miller index slabs.
        virtual_vacuum: float.
            Height of vacuum size of raw slabs, unit: Angstrom. Note that the vacuum size
            would be added to both the bottom and the top of surface.

        Returns:
        --------
        List[slab], delta_cart:
        List[slab]:
            one-layer slab list. The list actually just contain only one slab.
        delta_cart: List[double].
            The differences between two adjacent layers in Cartesian
            Coordinates.
        """
        # ATTENTION! input parameter virtual_layers could be increased, i.e, to 8, 
        # if current setting doesn't work for higher Miller index surfaces. The same for virtual_vacuum

        write(os.path.join(self.working_directory, 'bulk.POSCAR.vasp'),
              self.initial_structure, format="vasp",parallel=False)
        utils.modify_poscar(os.path.join(
            self.working_directory, 'bulk.POSCAR.vasp'))
        bulk = mg.Structure.from_file(
            os.path.join(self.working_directory, 'bulk.POSCAR.vasp'))
        super_structure_sg = StructureGraph.with_local_env_strategy(bulk,
                                                                    JmolNN())
        bulk_structure_sg = super_structure_sg * (3, 3, 3)
        unique_bulk_subgraphs, molecules = \
            utils.get_bulk_subgraphs_unique(bulk_structure_sg)
        # print("There would be {} different molecules in bulk".format(
        #    str(len(molecules))))
        # get the slab via ase and deal with it via pymatgen
        os.remove(os.path.join(self.working_directory, 'bulk.POSCAR.vasp'))
        slab = surface(self.initial_structure,
                       self.miller_index, layers=1, vacuum=15)
        file_name = os.path.join(
            self.working_directory, "ASE_surface.POSCAR.vasp")
        format_ = 'vasp'
        write(file_name, format=format_, images=slab,parallel=False)
        utils.modify_poscar(file_name)
        slab_temp = mg.Structure.from_file(file_name)

        slab = utils.surface(
            self.initial_structure, self.miller_index, layers=virtual_layers)
        delta = np.array(slab.cell)[2, :]
        # if self.vacuum_size is not None:
        slab.center(vacuum=virtual_vacuum, axis=2)

        file_name = os.path.join(
            self.working_directory, 'slab_before.POSCAR.vasp')
        write(file_name, format=format_, images=slab,parallel=False)
        utils.modify_poscar(file_name)
        slab_move = mg.Structure.from_file(file_name)
        os.remove(file_name)
        slab_move = utils.handle_with_molecules(slab_move, delta, down=True)
        Poscar(slab_move.get_sorted_structure()).write_file(
            os.path.join(self.working_directory, "AlreadyMove.POSCAR.vasp"))
        # delete intact molecule in slab_move
        slab = slab_move
        species_intact, coords_intact = [], []
        # os.remove(output_file)
        sg = molecules
        utils.Find_Broken_Molecules(slab, sg, species_intact,
                                    coords_intact, unique_bulk_subgraphs)
        # find the broken molecules for the first minor movement and delete the intact molecules
        try:
            slab = utils.put_everyatom_into_cell(slab)
            Poscar(slab.get_sorted_structure()).write_file(
                os.path.join(self.working_directory, "POSCAR_Broken.POSCAR.vasp"))
            os.remove(os.path.join(self.working_directory,
                                   "POSCAR_Broken.POSCAR.vasp"))
            slab = utils.handle_with_molecules(slab, delta, down=False)
        except ValueError:
            # No broken molecules anymore. So, return the slab_move
            slab_move = mg.Structure.from_file(os.path.join(
                self.working_directory, "AlreadyMove.POSCAR.vasp"))
            slab_move, delta_cart = self._extract_layer(
                slab_move, layers_virtual=virtual_layers)
            os.remove(os.path.join(self.working_directory,
                                   "AlreadyMove.POSCAR.vasp"))
            temp_file_name = os.path.join(
                self.working_directory, "temp.POSCAR.vasp")
            write(temp_file_name, slab_move,parallel=False)
            utils.modify_poscar(temp_file_name)
            slab_move = mg.Structure.from_file(temp_file_name)
            os.remove(temp_file_name)
            #print("No Broken molecules!")
            try:
                os.remove(os.path.join(self.working_directory,
                                       "ASE_surface.POSCAR.vasp"))
            except FileNotFoundError:
                print("Already delete!")
            return [slab_move.get_sorted_structure()], delta_cart
        os.remove(os.path.join(self.working_directory,
                               "AlreadyMove.POSCAR.vasp"))

        utils.Find_Broken_Molecules(slab, sg, species_intact,
                                    coords_intact, unique_bulk_subgraphs)
        try:
            slab = utils.put_everyatom_into_cell(slab)
            Poscar(slab.get_sorted_structure()).write_file(
                os.path.join(self.working_directory, "POSCAR_Broken_two.POSCAR.vasp"))
            os.remove(os.path.join(self.working_directory,
                                   "POSCAR_Broken_two.POSCAR.vasp"))
        except ValueError:
            for i in range(len(species_intact)):
                slab.append(
                    species_intact[i], coords_intact[i], coords_are_cartesian=True)
            slab, delta_cart = self._extract_layer(slab, virtual_layers)
            temp_file_name = os.path.join(
                self.working_directory, "temp.POSCAR.vasp")
            write(temp_file_name, slab,parallel=False)
            utils.modify_poscar(temp_file_name)
            slab = mg.Structure.from_file(temp_file_name)
            os.remove(temp_file_name)
            #print("No Broken molecules!")
            try:
                os.remove(os.path.join(self.working_directory,
                                       "ASE_surface.POSCAR.vasp"))
            except FileNotFoundError:
                print("Already delete!")
            return [slab.get_sorted_structure()], delta_cart

        speices = slab.species
        slab_coords = slab.frac_coords
        slab_coords_cart = slab.cart_coords

        for i, coord in enumerate(slab_coords):
            new_cart_coords = np.array(slab_coords_cart[i]) + delta
            # move the slab to match broken molecules
            slab.append(speices[i], coords=new_cart_coords,
                        coords_are_cartesian=True)

        try:
            for i in range(len(species_intact)):
                slab.append(
                    species_intact[i], coords_intact[i], coords_are_cartesian=True)
            file_name = os.path.join(
                self.working_directory, 'POSCAR_move.vasp')
            Poscar(slab.get_sorted_structure()).write_file(file_name)
            slab = mg.Structure.from_file(file_name)
            os.remove(file_name)

            slab_sg = StructureGraph.with_local_env_strategy(slab, JmolNN())
            super_structure_sg = StructureGraph.with_local_env_strategy(bulk,
                                                                        JmolNN())
            bulk_structure_sg = super_structure_sg * (3, 3, 3)
            unique_bulk_subgraphs, molecules = \
                utils.get_bulk_subgraphs_unique(bulk_structure_sg)

            slab_supercell_sg = slab_sg * (3, 3, 1)
            different_subgraphs_in_slab, slab_molecules = \
                utils.get_slab_different_subgraphs(
                    slab_supercell_sg, unique_bulk_subgraphs)
            sg = molecules
            slab_molecules = utils.double_screen(slab_molecules, sg)
            # print("The number of molecules that need to be fixed : ",
            #      len(slab_molecules))
            # slab_molecules are the molecules that are broken and need to be fixed
            delete_sites = utils.reduced_sites(slab_molecules, slab)
            delete_list = []

            for delete_site in delete_sites:
                for i, atom in enumerate(slab):
                    if atom.is_periodic_image(delete_site):
                        delete_list.append(i)
                        break
            slab.remove_sites(delete_list)
        except ValueError:
            print("No Broken molecules!")

        file_name = os.path.join(
            self.working_directory, "POSCAR_move_final.vasp")
        os.remove(os.path.join(self.working_directory,
                               "ASE_surface.POSCAR.vasp"))
        delta_cart = 0
        try:
            slab, delta_cart = self._extract_layer(slab, virtual_layers)
            output_file = os.path.join(
                self.working_directory, "Orge_surface.POSCAR.vasp")
            write(output_file, slab,parallel=False)
            utils.modify_poscar(output_file)
            slab = mg.Structure.from_file(output_file)
            os.remove(output_file)
            return [slab.get_sorted_structure()], delta_cart
        except ValueError:
            print("The {} slab with {} layers can not be reconstructed. And the result refers to ASE's surfaces. Please "
                  "try the graph_repair method!".format(self.miller_index, 1))
            return [slab_temp.get_sorted_structure()], delta_cart


    def _cleave_one_layer_v2(self, virtual_layers=4, virtual_vacuum=1000):
        """
        Main process (version 2) to generate one-layer slab from bulk. This
        process doesn't compare two graphs to check whether they are
        isomorphic. So, this approach would be faster and more accurate.

        Parameters:
        -----------
        virtual_layers: int.
            The number of layers of raw slabs. The virtual_layers is set
            to 4 to avoid that some molecules are cut by both upper and lower
            boundary for more than one time. ATTENTION: This parameter should
            be optimized to be self-adaptive to very high Miller index.
            Otherwise, please moderately increase this number, i.e, to 8, when we
            need to cleave high Miller index slabs.
        virtual_vacuum: float.
            Height of vacuum size of raw slabs, unit: Angstrom. Note that the vacuum size
            would be added to both the bottom and the top of surface.

        Returns:
        --------
        List[slab], delta_cart:
        List[slab]:
            one-layer slab list. The list actually just contain only one slab.
        delta_cart: List[double].
            The differences between two adjacent layers in Cartesian
            Coordinates.
        """
        while True:
            virtual_slab = utils.surface(self.initial_structure, self.miller_index, virtual_layers)
            double_virtual_slab = utils.surface(self.initial_structure, self.miller_index, 2 * virtual_layers)

            slab_1 = from_ASE_to_pymatgen(self.working_directory, virtual_slab)
            slab_2 = from_ASE_to_pymatgen(self.working_directory, double_virtual_slab)

            # Delete all molecules in slab_2
            delete_list = range(len(slab_2))
            slab_2.remove_sites(delete_list)

            slab_sg = StructureGraph.with_local_env_strategy(slab_1, JmolNN())
            slab_sg_extend = slab_sg * (1, 1, 1)

            slab_sg_graph = nx.Graph(slab_sg_extend.graph)
            super_graphs = list(nx.connected_component_subgraphs(slab_sg_graph))

            # subgraph_that needed to be added!!
            subgraph_added = []
            print("There're {} molecules in the super bulk".format(len(super_graphs)))
            ii = 0
            while ii < len(super_graphs):
                subgraph = super_graphs[ii]
                out_boundary = all([list(d['to_jimage'])[2] == 0  for u, v, d in
                                   subgraph.edges(data=True)])
                ii += 1
                if not out_boundary:
                    ii -= 1
                    super_graphs.remove(subgraph)
                    subgraph_added.append(subgraph)

            print("{} molecules are moved.".format(len(subgraph_added)))
            if len(super_graphs) >= 1.5 * len(subgraph_added):
                break
            else:
                virtual_layers *= 2


        for subgraph in super_graphs:
            for n in subgraph:
                subgraph.add_node(n, specie = str(slab_sg_extend.structure[n].specie))

        for subgraph in subgraph_added:
            for n in subgraph:
                subgraph.add_node(n, specie = str(slab_sg_extend.structure[n].specie))

        molecules_old = []
        for subgraph in super_graphs:
            coords = [slab_sg_extend.structure[n].coords
                      for n in subgraph.nodes()]
            species = [slab_sg_extend.structure[n].specie
                       for n in subgraph.nodes()]
            molecule = mg.Molecule(species=species, coords=coords)
            molecules_old.append(molecule)


        delta = np.array(slab_1.lattice.matrix[-1])

        molecules_added = []
        for subgraph in subgraph_added:
            coords = [np.array(slab_sg_extend.structure[n].coords) + delta
                      for n in subgraph.nodes()]
            species = [slab_sg_extend.structure[n].specie
                       for n in subgraph.nodes()]
            molecule = mg.Molecule(species=species, coords=coords)
            molecules_added.append(molecule)

        broken_molecules_num = len(molecules_added)

        # add broken molecules to slab_2
        sites = []
        for molecule in molecules_added:
            for curr_site in molecule:
                curr_site = mg.PeriodicSite(curr_site.specie,
                                            curr_site.coords,
                                            slab_2.lattice,
                                            coords_are_cartesian=True)
                tmp = [curr_site.is_periodic_image(site) for site in sites]
                if not any(tmp):
                    sites.append(curr_site)

        for site in sites:
            slab_2.append(species=site.specie, coords=site.coords, coords_are_cartesian=True)

        # move upper broken molecules to lower
        one_layer_sg = StructureGraph.with_local_env_strategy(
            slab_2, JmolNN())
        bulk_sg = one_layer_sg * (1, 1, 1)
        subgraphs, molecules = utils.get_bulk_subgraphs_v2(
            bulk_structure_sg=bulk_sg)

        number_of_molecules_to_move = len(molecules) - broken_molecules_num
        while(number_of_molecules_to_move != 0):
            highest_z_locations = [np.max(np.array(slab_2.lattice.get_fractional_coords(molecule.cart_coords))[:, 2])
                                   for molecule in molecules]

            [highest_z_locations, molecules] = list(
                zip(*(sorted(zip(highest_z_locations, molecules), key=lambda a: a[0], reverse=True))))
            print(highest_z_locations[0])

            slab_2 = utils.move_molecule(molecules[:(number_of_molecules_to_move // 2 if number_of_molecules_to_move // 2 > 1 else 1)], slab_2, delta)
            one_layer_sg = StructureGraph.with_local_env_strategy(
                slab_2, JmolNN())
            bulk_sg = one_layer_sg * (1, 1, 1)
            subgraphs, molecules = utils.get_bulk_subgraphs_v2(
                bulk_structure_sg=bulk_sg)

            print("The number of molecules that are broken after moved is {}".format(len(molecules)))
            number_of_molecules_to_move = len(molecules) - broken_molecules_num

        sites = []
        for molecule in molecules_old:
            for curr_site in molecule:
                curr_site = mg.PeriodicSite(curr_site.specie,
                                            curr_site.coords,
                                            slab_2.lattice,
                                            coords_are_cartesian=True)
                tmp = [curr_site.is_periodic_image(site) for site in sites]
                if not any(tmp):
                    sites.append(curr_site)

        for site in sites:
            slab_2.append(species=site.specie, coords=site.coords, coords_are_cartesian=True)

        file_name = os.path.join(
            self.working_directory, 'slab_2.POSCAR.vasp')
        Poscar(slab_2.get_sorted_structure()).write_file(file_name)
        slab_2 = read(file_name,parallel=False)
        os.remove(file_name)
        slab_2.center(vacuum=virtual_vacuum, axis = 2)
        slab, delta_cart = self._extract_layer(from_ASE_to_pymatgen(self.working_directory, slab_2), virtual_layers)
        slab = from_ASE_to_pymatgen(self.working_directory, slab)
        return [slab.get_sorted_structure()], delta_cart


    def _cleave_one_layer_v3(self, virtual_vacuum=15):
        """
        Main process (version 3) to generate one-layer slab from bulk. This 
        approach would be faster and more accurate than v1 and v2.

        Parameters:
        -----------
        virtual_vacuum: float.
            Height of vacuum size of raw slabs, unit: Angstrom. Note that the vacuum size
            would be added to both the bottom and the top of surface.

        Returns:
        --------
        List[slab], delta_cart:

        List[slab]:
            one-layer slab list. The list actually just contain only one slab.
        delta_cart: List[double].
            The differences between two adjacent layers in Cartesian
            Coordinates.
        """
        slab_first = utils.surface(self.initial_structure, self.miller_index, layers=1)
        slab_first = from_ASE_to_pymatgen(self.working_directory, slab_first)
        slab_first_sg = StructureGraph.with_local_env_strategy(slab_first, JmolNN())
        slab_first_sg = slab_first_sg * (1, 1, 1)
        delta_cart, _, molecules = utils.get_bulk_subgraphs_v3(slab_first,
                                                               bulk_structure_sg=slab_first_sg) 
        delete_list = range(len(slab_first))
        slab_first.remove_sites(delete_list)
        sites = []
        for molecule in molecules:
            for curr_site in molecule:
                curr_site = mg.PeriodicSite(curr_site.specie,
                                            curr_site.coords,
                                            slab_first.lattice,
                                            coords_are_cartesian=True)
                tmp = [curr_site.is_periodic_image(site) for site in sites]
                if not any(tmp):
                    sites.append(curr_site)

        for site in sites:
            slab_first.append(species=site.specie, coords=site.coords, coords_are_cartesian=True)

        file_name = os.path.join(
            self.working_directory, 'slab_first.POSCAR.vasp')
        Poscar(slab_first.get_sorted_structure()).write_file(file_name)
        slab_ASE = read(file_name,parallel=False)
        os.remove(file_name)
        slab_ASE.center(vacuum=virtual_vacuum, axis = 2)
        slab = from_ASE_to_pymatgen(self.working_directory, slab_ASE)
        return [slab.get_sorted_structure()], delta_cart



    def _extract_layer(self, slab, layers_virtual=4):
        """
        Step 5 in _cleave_one_layer function:
            Extract one-layer slab from new slabs based on periodicity.
            There are "virtual_layers" identical layers in the slab and we just
            need one layer of them.

        Parameters:
        -----------
        slab: pymatgen structure.
            "virtual_layers"-layer slabs after repairing.
        layers_virtual: int.
            The number of layer of slab.

        returns:
        --------
        structure, delta_cart:
        structure: ASE structure.
            one-layer slab.
        delta_cart: List[double].
            The differences between two adjacent layers in Cartesian
            Coordinates.
        """
        slab_incline = deepcopy(slab)
        slab_incline = utils.put_everyatom_into_cell(slab_incline)
        super_structure_sg = StructureGraph.with_local_env_strategy(slab_incline,
                                                                    JmolNN())
        bulk_structure_sg = super_structure_sg * (1, 1, 1)
        super_subgraphs, molecules = utils.get_bulk_subgraphs(
            bulk_structure_sg)
        account_list = [0] * len(super_subgraphs)
        c_frac_gap = []
        for index_one in range(len(super_subgraphs) - 1):
            for index_two in range(index_one + 1, len(super_subgraphs)):
                if nx.is_isomorphic(super_subgraphs[index_one], super_subgraphs[index_two],
                                    node_match=utils.node_match):
                    species_one = molecules[index_one].species
                    coords_one = slab_incline.lattice.get_fractional_coords(
                        molecules[index_one].cart_coords)
                    species_two = molecules[index_two].species
                    coords_two = slab_incline.lattice.get_fractional_coords(
                        molecules[index_two].cart_coords)

                    account = 0
                    for item_a, coord_a in enumerate(coords_one):
                        for item_b, coord_b in enumerate(coords_two):
                            if species_one[item_a] == species_two[item_b] and abs(coord_a[0] - coord_b[0]) <= 1e-4 and abs(
                                    coord_a[1] - coord_b[1]) <= 1e-4:
                                c_frac_gap.append(abs(coord_a[2] - coord_b[2]))
                                if coord_a[2] < coord_b[2]:
                                    account += 1
                                    break
                                else:
                                    account -= 1
                                    break

                    if account >= 0.5 * len(coords_one):
                        account_list[index_one] += 1
                    elif account <= - 0.5 * len(coords_two):
                        account_list[index_two] += 1
        delta_cart = slab_incline.lattice.get_cartesian_coords(
            [0, 0, min(c_frac_gap)])
        slab_molecules = [molecule for item, molecule in enumerate(
            molecules) if account_list[item] != layers_virtual - 1]
        delete_sites = utils.reduced_sites(slab_molecules, slab_incline)
        delete_list = []

        for delete_site in delete_sites:
            for i, atom in enumerate(slab_incline):
                if atom.is_periodic_image(delete_site):
                    delete_list.append(i)
                    break
        slab_incline.remove_sites(delete_list)
        file_name = os.path.join(
            self.working_directory, 'one_layer.POSCAR.vasp')
        Poscar(slab_incline.get_sorted_structure()).write_file(file_name)

        # find the structure, next we need to find the periodicity
        format_ = 'vasp'
        structure = io.read(file_name, format=format_,parallel=False)
        os.remove(file_name)
        structure.center(vacuum=15, axis=2)
        return structure, delta_cart

    def _surface_termination(self, one_layer_slab, delta_move, users_define_layers=None):
        """
        Determine all alternative one-layer slab by analyzing terminations and
        moving molecules from one side to another (upper side to lower side).
        The approach is counting the species and numbers of the
        highest c-direction atoms. 

        More details could refer to paper "Ogre...", Fig 4. and
        section C: Surface Terminations.

        Parameters:
        -----------
        one_layer_slab: pymatgen structure.
            One-layer slab that is generated by _cleave_one_layer() and needs to
            be analyzed to produce other alternative one-layer slab with
            different terminations.
        delta_move: List[double].
            The distance from the upper side of one-layer slab to the lower
            side in Cartesian Coordinates.
        users_define_layers: int.
            Possible number of groups in one-layer slab. Once the goups are
            defined, molecules in one group would move from one side to another
            simultaneously. The default number is None, means that each
            molecule makes up a group. 

        Returns:
        --------
        slab_list: List[pymatgen structure].
            List of one-layer slab with different terminations.
        """
        file_name = os.path.join(
            self.working_directory, "one_layer_temp.POSCAR.vasp")
        Poscar(one_layer_slab.get_sorted_structure()).write_file(file_name)
        one_layer_temp = io.read(file_name,parallel=False)
        os.remove(file_name)
        one_layer_temp.center(vacuum=15, axis=2)
        if delta_move is None:
            delta = np.array(one_layer_temp.cell)[2, :]
        else:
            delta = delta_move
        one_layer_temp.center(vacuum=1000, axis=2)
        file_name = os.path.join(
            self.working_directory, " one_layer_temp.POSCAR.vasp")
        io.write(file_name, images=one_layer_temp,parallel=False)
        utils.modify_poscar(file_name)
        one_layer = mg.Structure.from_file(file_name)
        os.remove(file_name)
        one_layer = utils.put_everyatom_into_cell(one_layer)
        one_layer_sg = StructureGraph.with_local_env_strategy(
            one_layer, JmolNN())
        bulk_sg = one_layer_sg * (1, 1, 1)
        subgraphs, molecules = utils.get_bulk_subgraphs(
            bulk_structure_sg=bulk_sg)
        highest_z_locations = [np.max(np.array(one_layer.lattice.get_fractional_coords(molecule.cart_coords))[:, 2])
                               for molecule in molecules]
        highest_species = [
            str(molecule.species[np.argmax(np.array(one_layer.lattice.get_fractional_coords(molecule.cart_coords))[:,
                                                                                                                   2])]) for molecule in molecules]

        [highest_z_locations, highest_species, molecules] = list(
            zip(*(sorted(zip(highest_z_locations, highest_species, molecules), key=lambda a: a[0], reverse=True))))

        gaps = [highest_z_locations[i] - highest_z_locations[i + 1]
                for i in range(len(highest_species) - 1)]

        small_gap, big_gap = 0, 1
        if users_define_layers is None:
            users_define_layers = len(molecules)

        medium_gap = utils.double_find_the_gap(
            small_gap, big_gap, gaps, users_define_layers, 1e-4)
        if medium_gap != -1:
            qualified = [medium_gap < gap for gap in gaps]
        else:
            qualified = [1] * len(gaps)

        slab_list = [one_layer]

        one_layer_temp = deepcopy(one_layer)
        highest_specie = [highest_species[0]]
        for index, molecule in enumerate(molecules[: -1]):
            if qualified[index] == 1 and highest_species[index + 1] not in highest_specie:
                one_layer_temp_two = utils.move_molecule(
                    molecules[: index + 1], one_layer_temp, delta)
                slab_list.append(one_layer_temp_two)
                highest_specie.append(highest_species[index + 1])

        slab_temp_list = []
        for index, slab in enumerate(slab_list):
            file_name = os.path.join(
                self.working_directory, "primitive_onelayer_" + str(index) + ".POSCAR.vasp")
            Poscar(slab.get_sorted_structure()).write_file(file_name)
            slab_temp = io.read(file_name,parallel=False)
            os.remove(file_name)
            slab_temp.center(vacuum=15, axis=2)
            slab_temp_list.append(slab_temp)

        for index, slab_temp in enumerate(slab_temp_list):
            file_name = os.path.join(
                self.working_directory, "primitive_onelayer_" + str(index) + ".POSCAR.vasp")
            # slab_temp.set_cell(cell)
            io.write(file_name, images=slab_temp,parallel=False)
            utils.modify_poscar(file_name)
            slab_list[index] = mg.Structure.from_file(file_name)
            os.remove(file_name)
        return slab_list


def atomic_task(name, 
                initial_structure, 
                miller_index, 
                list_of_layers, 
                vacuum_size,
                supercell_size, 
                format_string,
                desired_num_of_molecules_oneLayer):
    """
    Atomic task to cleave a surface plane with certain Miller index for 
    different layers.

    Parameters
    ----------
    initial_structure: ASE Atoms structure.
        The initial bulk structure to be cleaved.
    miller_index: List[int]: [h, k, l]. 
        The Miller index of the surface plane.
    list_of_layers: List[int]. 
        A list of layers to cleave.
    vacuum_size: float
        Height of vacuum size, unit: Angstrom. Note that the vacuum size
        would be added to both the bottom and the top of surface.
    supercell_size: List[int]: [a, b, c]
        Make a (a * b * 1) supercell.
    working_directory: str
        The path to save the resulting slab structures.
    """
    format_dict = {'FHI': 'in', 'VASP': 'POSCAR', 'CIF': 'cif'}
    # Create working directory to isolate the workflow
    dir_name = '{}_{}'.format(name, "".join(str(int(x))
                                            for x in miller_index))
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    working_dir = os.path.abspath('./{}'.format(dir_name))

    # print("start {}".format("".join(str(int(x))
    #                                for x in miller_index)))

    generator = OrganicSlabGenerator(
        initial_structure, 
        miller_index, 
        list_of_layers, 
        vacuum_size, 
        supercell_size, 
        working_dir,
        desired_num_of_molecules_oneLayer)
    
    slab_lists = generator.cleave()

    for layers, slab_list in zip(list_of_layers, slab_lists):
        for i, slab in enumerate(slab_list):
            poscar_str = "{}/POSCAR.{}.{}.{}.{}".format(
                            name, 
                            name, 
                            "".join(str(int(x)) for x in miller_index), 
                            layers, 
                            i)
            Poscar(slab).write_file(poscar_str)
            
            slab_ase = read(poscar_str,parallel=False)
            os.remove(poscar_str)
            write("{}/{}.{}.{}.{}.{}"
                  .format(name, 
                          name, 
                          "".join(str(int(x)) for x in miller_index), 
                          layers, 
                          i, 
                          format_dict[format_string]), 
                          slab_ase,parallel=False)
                  
    # This try-except is used to avoid FileNotFoundError when Ogre is called by MPI processes.
    try:
        shutil.rmtree(working_dir)
    except FileNotFoundError:
        print(f"Directory {working_dir} was already deleted.")


def cleave_for_surface_energies(structure_path, 
                                structure_name, 
                                vacuum_size, 
                                list_of_layers, 
                                highest_index, 
                                supercell_size, 
                                format_string,
                                desired_num_of_molecules_oneLayer):
    """
    Multiprocess launcher to cleave a surface plane with certain Miller index 
    for different layers.

    Parameters
    ----------
    structure_path: str
        The path of initial bulk structure.
    structure_name: str
        The structure's name, used to create the directory.
    highest_index: int
        The highest value of Miller index used to calculate the Wulff shape.
    miller_index: List[int]: [h, k, l].
        The Miller index of the surface plane.
    list_of_layers: List[int].
        A list of layers to cleave.
    vacuum_size: float
        Height of vacuum size, unit: Angstrom. Note that the vacuum size
        would be added to both the bottom and the top of surface.
    supercell_size: List[int]: [a, b, c]
        Make a (a * b * 1) supercell.
    working_directory: str
        The path to save the resulting slab structures.
    format_string: str
        The format of output file, could be "VASP", "FHI" or "CIF".
        
    """
    initial_structure = read(structure_path,parallel=False)
    if not os.path.isdir(structure_name):
        os.mkdir(structure_name)
    up = UniquePlanes(initial_structure, index=highest_index, verbose=False)
    # p = Pool(processes=1)
    p = Pool()
    unique_idx = set()
    for x  in up.unique_idx:
        i = np.gcd.reduce(x)
        inverse_x = (tuple(-int(y/i) for y in x))
        if inverse_x not in unique_idx:
            unique_idx.add(tuple(int(y/i) for y in x))

    print("{} unique planes are found".format(len(unique_idx)))
    pbar = tqdm(total=len(unique_idx))
    def update(*a):
        nonlocal pbar
        pbar.update()
    for miller_index in unique_idx:
        p.apply_async(atomic_task, args=(
            structure_name, 
            initial_structure, 
            list(miller_index), 
            list_of_layers, 
            vacuum_size, 
            supercell_size, 
            format_string,
            desired_num_of_molecules_oneLayer), 
            callback=update)
    
    p.close()
    p.join()
