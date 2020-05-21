# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ibslib.io import check_dir
from ibslib.io import read, write
from pymatgen.analysis.wulff import WulffShape
from pymatgen import Lattice
from pymatgen.io.cif import CifParser
from tqdm import tqdm
from ibslib.io import read
import argparse
import numpy as np
from ibslib.analysis import get
from scipy import stats
import matplotlib
import collections
from configparser import ConfigParser
from pymatgen.core.structure import Structure

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', dest='filename', default='ogreSWAMP.config',
                        type=str)
    parser.add_argument('--fontsize', dest='fontsize', default=25,
                        type=int)
    return parser.parse_args()


def Boettger(ax, layers, energies, area, tag):
    x, y = [], []
    c = {"pbe":"orange", "ts":"g", "mbd":"r"}
    for i in range(2, len(layers)):
        bulk = (energies[i] - energies[i-1]) / (layers[i] - layers[i-1])
        x.append(layers[i])
        y.append(16000*(energies[i] - layers[i] * bulk)/(2*area))
    ax.plot(x, y, c=c[tag], label="{} Boettger".format(
        tag.upper()), linewidth=7.0)
    return x, y


def Linear(ax, layers, energies, area, tag):
    x, y = [], []
    c = {"pbe":"yellow", "ts":"brown", "mbd":"blue"}
    for i in range(2, len(layers)):
        if i < 2:
            slope, intercept, r_value, p_value, std_err = \
                stats.linregress(layers[0:i+1], energies[0:i+1])
        else:
            slope, intercept, r_value, p_value, std_err = \
                stats.linregress(layers[i-2:i+1], energies[i-2:i+1])
        x.append(layers[i])
        y.append(16000*intercept/(2*area))
    ax.plot(x, y, c=c[tag], label="{} Linear".format(
        tag.upper()), linewidth=7.0, linestyle='dashed')
    return x, y


def convergence_plot(structure_name, scf_path, threshhold, consecutive_step, fontsize):
    """Plot the surface convergence plots and calculate the surface energy for each surface.

    Parameters:
    ----------
        structure_name: str
            Structure's name.
        scf_path: str
            The path of SCF data in json format, stored by scripts in scripts/
        threshhold: float
            Threshhold that determines the convergence.
        consecutive_step: int
            The relative difference should be within the threshhold up to the number of consecutive steps.
        fontsize: int
            Font size to plot the convergence plots.

    Returns:
    --------
    : dict
        Dictionary that contains the surface energy values for TS and MBD with linear and Boettger method.
    """
    if not os.path.isdir(structure_name):
        os.mkdir(structure_name)
    s = read(scf_path)
    for struct_id, struct in s.items():
        split_name = struct_id.split(".")
        index = split_name[1]
        layers = split_name[2]
        termination = split_name[3]
        struct.properties["index"] = index
        struct.properties["layers"] = int(layers)
        struct.properties["termination"] = int(termination)
        a, b = struct.properties["lattice_vector_a"], struct.properties["lattice_vector_b"]
        area = abs(np.cross(a, b)[-1])
        struct.properties["area"] = area
    results = get(s, "prop", ["mbd_energy", "energy", "index", "layers", "area",
                              "termination", "vdw_energy"])
    indexes = results["index"].unique()
    num_images = 0
    for index in indexes:
        tot_data = results[results["index"] ==
                           index].sort_values(by=["layers"])
        terms = tot_data["termination"].unique()
        num_images += len(terms)

    energy_results = collections.defaultdict(list)

    for index in tqdm(indexes):
        tot_data = results[results["index"] ==
                           index].sort_values(by=["layers"])
        terms = tot_data["termination"].unique()
        for term in terms:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.tick_params(labelsize=fontsize)
            data = tot_data[tot_data["termination"] == term]
            area = data["area"][0]
            layers = data["layers"]
            energies = data["energy"]
            mbd_energies = data["mbd_energy"]
            pbe_energies = data["energy"] - data["vdw_energy"]
            mbd_energies = data["mbd_energy"] + pbe_energies
            ts_energies = data["energy"]
            layers = list(layers)
            maxy = 0
            miny = float('inf')
            for energies, tag in [(pbe_energies,"pbe"), (ts_energies, "ts"), (mbd_energies, "mbd")]:
                energies = list(energies)
                bx, by = Boettger(ax, layers, energies, area, tag)
                lx, ly = Linear(ax, layers, energies, area, tag)
                diff = []
                if tag in ["ts", "mbd"]:
                    for i in range(1, len(ly)):
                        diff.append(abs(ly[i] - ly[i-1]) / ly[i] * 100)
                    stop_index = 0
                    while stop_index < len(diff):
                        for i in range(consecutive_step):
                            if diff[stop_index - i] >= threshhold:
                                break
                        else:
                            break
                        stop_index += 1
                        continue
                    if stop_index == len(diff):
                        print(
                            "The surface calculation with {} is not converged for {}".format(tag, index))
                        exit()
                else:
                    stop_index = len(ly) - 1
                ly = ly[:stop_index + 1]
                by = by[:stop_index + 1]
                maxy = max(max(by[-1], ly[-1]), maxy)
                miny = min(min(by[-1], ly[-1]), miny)
                energy_results[tag].append((index, term, by[-1], ly[-1]))
            ax.set_xticks(range(3, layers[-1] + 1, 2))
            if len(terms) == 2:
                ax.set_title("({}) Type {}".format(
                    index, 'I' if term == 0 else 'II'), fontsize=fontsize)
            else:
                ax.set_title("({})".format(index), fontsize=fontsize)
            ax.set_ylim(
                miny - 5, maxy + 2)
            ax.set_xlabel("Number of Layers", fontsize=fontsize)
            ax.set_ylabel("Surface Energy ($mJ/m^{2}$)", fontsize=fontsize)
            plt.savefig("{}/{}_{}.png".format(structure_name, index, term),
                        dpi=400, bbox_inches="tight")
            plt.close()
    return energy_results


def Wulff_plot(structure_name, structure_path, projected_direction, fitting_method, energy_results, threshhold):
    """Plot Wulff shape by pymatgen.

    Parameters:
    ----------
        structure_name: str
            Structure's name.
        structure_path: str
            The path of initial bulk structure.
        projected_direction: List[int]
            The projected direction for the Wulff shape.
        fitting_method: int
            0: linear method, 1: Boettger method.
        energy_results: dict
            Dictionary that contains the surface energy values for TS and MBD with linear and Boettger method.
        threshhold: float
            Threshhold that determines the convergence.
    """
    structure = Structure.from_file(structure_path)
    lattice = structure.lattice
    print("Lattice parameters are [{}, {}, {}], please check whether they are in the same order as the input.".format(
        lattice.a, lattice.b, lattice.c))
    for tag in ["ts", "mbd"]:
        energy_result = energy_results[tag]
        data = {}
        for index, term, by, ly in energy_result:
            idx = []
            temp_idx = ""
            for char in index:
                if char == "-":
                    temp_idx += char
                else:
                    temp_idx += char
                    idx.append(temp_idx)
                    temp_idx = ""
            idx = tuple([int(x) for x in idx])
            if fitting_method == 1:
                if idx not in data or data[idx] > by:
                    data[idx] = float(by)
            else:
                if idx not in data or data[idx] > ly:
                    data[idx] = float(ly)
        with open("{}/surface_energy_{}_threshhold={}.csv".format(structure_name, tag, threshhold),'w') as f:
            f.write("Miller index\tSurface energy($mJ/m^{2}$)\n")
            for idx, energy in data.items():
                f.write("{}\t{}\n".format("".join(str(x) for x in idx), energy))
        plt.figure()
        w_r = WulffShape(lattice, data.keys(), data.values())
        w_r.get_plot(direction=projected_direction)
        plt.savefig("{}/Wulff_{}_threshhold={}.png".format(structure_name, tag, threshhold),
                    dpi=400, bbox_inches="tight")
        

if __name__ == "__main__":
    """
    Performs all actions for the post-calculation analysis of the surface 
    energy values by parsing data and producing figures. 
    
    """
    
    ## Parse Arguments from command line and 
    args = parse_arguments()
    filename = args.filename
    fontsize = args.fontsize
    
    ## Parse settings from the configuration file
    config = ConfigParser()
    config.read(filename, encoding='UTF-8')
    io, Wulff, methods, convergence = config['io'], config['Wulff'], \
                                      config['methods'], config['convergence']
    
    ## Prepare settings and convert to correct datatype
    scf_path = io['scf_path']
    structure_name = io['structure_name']
    structure_path = io['structure_path']
    plot = 't' in Wulff['Wulff_plot'].lower()
    projected_direction = [int(x) for x in 
                           Wulff['projected_direction'].split(" ")]
    threshhold = float(convergence['threshhold'])
    consecutive_step = int(convergence['consecutive_step'])
    fitting_method = methods['fitting_method']

    ## Plot convergence curves
    energy_results = convergence_plot(
        structure_name, scf_path, threshhold, consecutive_step, fontsize)
    
    ## Plot Wulff shapes
    if plot:
        Wulff_plot(structure_name, structure_path,
                   projected_direction, fitting_method, energy_results, threshhold)
