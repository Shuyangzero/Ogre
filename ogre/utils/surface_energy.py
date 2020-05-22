

import os
import collections

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

from ibslib.io import read
from ibslib.analysis import get



def Boettger(ax, layers, energies, area, tag):
    """
    Function that produces the surface energy result from the Boettger Method. 
    Call function for each relevant surface. 
    
    Arguments
    ---------
    ax: matplotlib.axes
        Figure axes to add the linear line to. 
    layers: iterable
        Sorted iterable of integers corresponding to the number of layers that 
        will make up the x-axis. 
    energies: iterable
        Sorted w.r.t. the number of layers and will make up the y-axis. 
    area: float
        Surface area of the relevant surface. 
    tag: str
        String indicating the method. Must be one of: "pbe", "ts", "mbd"
    
    """
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
    """
    Function that produces the surface energy result from the Linear Fit Method. 
    Call function for each relevant surface. 
    
    Arguments
    ---------
    ax: matplotlib.axes
        Figure axes to add the linear line to. 
    layers: iterable
        Sorted iterable of integers corresponding to the number of layers that 
        will make up the x-axis. 
    energies: iterable
        Sorted w.r.t. the number of layers and will make up the y-axis. 
    area: float
        Surface area of the relevant surface. 
    tag: str
        String indicating the method. Must be one of: "pbe", "ts", "mbd"
    
    """
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


def convergence_plots(structure_name, 
                      scf_path, 
                      threshold, 
                      consecutive_step, 
                      fontsize):
    """
    Plot the surface convergence plots and calculate the surface energy for each 
    surface.

    Parameters
    ----------
    structure_name: str
        Structure's name.
    scf_path: str
        The path of SCF data in json format, stored by scripts in scripts/
    threshold: float
        Threshold that determines the convergence. The threshold should be given
        as a percent. 
    consecutive_step: int
        The relative difference should be within the threshold up to the 
        number of consecutive steps.
    fontsize: int
        Font size to plot the convergence plots.

    Returns
    -------
    dict
        Dictionary that contains the surface energy values for TS and MBD from 
        the Linear and Boettger surface slab energy fitting methods. 
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


    for index in indexes:
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
                            if diff[stop_index - i] >= threshold:
                                break
                        else:
                            break
                        stop_index += 1
                        continue
                    if stop_index == len(diff):
                        raise Exception(
        "The surface calculation with the {} method was not converged for surface {}"
        .format(tag, index) +
        " to the user defined threshold of {}.".format(threshold))
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