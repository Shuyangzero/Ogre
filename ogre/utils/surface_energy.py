

import os
import collections

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


def add_line(ax, x, y, method, tag):
    """
    For adding lines only up to a converged number of layers. 
    
    Arguments
    ---------
    ax: matplotlib.axes
        Figure axes to add the linear line to.
    x: iterable 
        Values for number of layers 
    y: iterable
        Values for surfac energy from Boettger or Linear
    method: str
        One of "Boettger" or "Linear"
    tag: str
        String indicating the method. Must be one of: "pbe", "ts", "mbd"
    """
    if method == "Boettger":
        c = {"pbe":"orange", "ts":"g", "mbd":"r"}
        linestyle = "dashed"
    elif method == "Linear":
        c = {"pbe":"yellow", "ts":"brown", "mbd":"blue"}
        linestyle = "solid"
    else:
        raise Exception("Method not recognized")
    
    ax.plot(x, y, c=c[tag], label="{} Linear".format(
        tag.upper()), linewidth=7.0, linestyle=linestyle)


def convergence_plots(structure_name, 
                      scf_path, 
                      threshold=5, 
                      max_layers=-1, 
                      fontsize=16,
                      pbe=False,
                      mbd=True,
                      boettger=True,
                      combined_figure=True):
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
    max_layers: int
        Maximum number of layers to use before an error is raised that the 
        convergence tolerance was not reached. Default value of -1 indicates
        that all layers will be used. 
    fontsize: int
        Font size to plot the convergence plots.
    pbe: bool
        If True, PBE results will be included in the final plots.
    mbd: bool
        If True, MBD results will be included in the final plots.
    boettger: bool
        If True, Boettger results will be included in the final plots. 
    combined_figure: bool
        If True, will create one large combined figure. 
        If False, figures are save separately. 

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

    ## Building combined figure
    if combined_figure:
        ## Decide if three or four columns should be used
        r3 = num_images % 3
        r4 = num_images % 4
        
        if r3 <= r4:
            columns = 3
        else:
            columns = 4
        
        rows = int(num_images / columns)+1

        axes_height = 5
        axes_width = 6
        fig,ax_list = plt.subplots(
                ncols=columns,
                nrows=rows,
                figsize=(axes_width*columns,axes_height*rows))
        ax_list = ax_list.ravel()
    
    axes_idx = 0
    for index in indexes:
        tot_data = results[results["index"] ==
                           index].sort_values(by=["layers"])
        terms = tot_data["termination"].unique()
        
        for term in terms:
            
            if not combined_figure:
                ## Prepare figure
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
            else:
                ax = ax_list[axes_idx]
                axes_idx += 1
                
            ax.tick_params(labelsize=fontsize)
            
            ## Prepare data
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
            
            maxy_list = []
            miny_list = []
            max_layer_list = []
            min_layer_list = []
            not_converged_list = []
            
            for energies, tag in [(pbe_energies,"pbe"), 
                                  (ts_energies, "ts"), 
                                  (mbd_energies, "mbd")]:
                
                if tag == "pbe":
                    if not pbe:
                        continue
                if tag == "mbd":
                    if not mbd:
                        continue
                
                energies = list(energies)
                
                ### Calculate energies using a dummy axes
                temp_fig = plt.figure()
                temp_ax = temp_fig.add_subplot(111)
                bx, by = Boettger(temp_ax, layers, energies, area, tag)
                lx, ly = Linear(temp_ax, layers, energies, area, tag)
                plt.close(temp_fig)
                
                diff = []
                if tag in ["ts", "mbd", "pbe"]:
                    for i in range(1, len(ly)):
                        diff.append(abs(ly[i] - ly[i-1]) / ly[i] * 100)
                    
                    ## Get last layer number for which convergence is achieveds
                    converged = False
                    keep_idx = -1
                    for idx,value in enumerate(diff):
                        ## Plus one because that's how diff list is built
                        idx += 1
                        last_layer_number = layers[idx]
                        
                        if value <= threshold:
                            converged = True
                            keep_idx = idx
                            break
                    
                    ## Check convergence occured at all
                    if not converged:
                        print(
                            "The surface calculation with the {} method "
                            .format(tag) +
                            "was not converged for surface {} " 
                            .format(index) +
                            "to the user defined threshold of {} "
                            .format(threshold) +
                            "after {} layers."
                            .format(layers[-1])
                            )
                        
                    ## Check if convergence occurs within max_layers
                    elif max_layers > 0:
                        if last_layer_number > max_layers:
                            print(
                                "The surface calculation with the {} method "
                                .format(tag) +
                                "was not converged for surface {} " 
                                .format(index) +
                                "to the user defined threshold of {} "
                                .format(threshold)+
                                "within max layers value of {}. "
                                .format(max_layers) +
                                "Convergence occured at {} layers."
                                .format(last_layer_number)
                                )
                            converged = False
                            
                        else:
                            ## Everything is fine
                            pass
                            ## Store values to be used for plotting
                            
                    else:
                        ## Max layers is -1 which means that any last layer value
                        ## is allowed.
                        ## Everything is fine. 
                        pass
                
                if max_layers > 0:
                    
                    if keep_idx > 0:
                        ## Add results to final figure
                        temp_layers = layers[:keep_idx+1]
                        
                        bx = bx[:keep_idx+1]
                        by = by[:keep_idx+1]
                        
                        lx = lx[:keep_idx+1]
                        ly = ly[:keep_idx+1]
                    else:
                        temp_layers = layers.copy()
                        
                    add_line(ax, lx, ly, "Linear", tag)
                    if boettger: 
                        add_line(ax, bx, by, "Boettger", tag)
                    
                else:
                    temp_layers = layers.copy()
                    add_line(ax, lx, ly, "Linear", tag)
                    if boettger: 
                        add_line(ax, bx, by, "Boettger", tag)
                        
                ## Storage for figure formatting
                maxy = max(by+ly)
                miny = min(by+ly)
                energy_results[tag].append((index, term, by[-1], ly[-1]))
                
                maxy_list.append(maxy)
                miny_list.append(miny)
                max_layer_list.append(max(temp_layers)+2)
                min_layer_list.append(min(temp_layers)+2)
                
                if converged == False:
                    not_converged_list.append(tag)
            
            maxy = max(maxy_list)
            miny = min(miny_list)
            
            ## Set xticks
            xtick_values = range(min(min_layer_list), max(max_layer_list) + 1)
            if len(xtick_values) > 8:
                 xtick_values = range(min(min_layer_list), 
                                      max(max_layer_list) + 1,
                                      2)
            ax.set_xticks(xtick_values)
            ## Set y limites
            ax.set_ylim(
                miny - 5, maxy + 2)
            
            ## Set title
            if len(terms) == 2:
                ax.set_title("({}) Type {}".format(
                    index, 'I' if term == 0 else 'II'), fontsize=fontsize)
            else:
                ax.set_title("({})".format(index), fontsize=fontsize)
            
            ## Set labels 
            ax.set_xlabel("Number of Layers", fontsize=fontsize)
            ax.set_ylabel("Surface Energy ($mJ/m^{2}$)", fontsize=fontsize)
            
            ## Add text if the surface is not converged
            if len(not_converged_list) > 0:
                text = "NOT CONVERGED: \n"
                for entry in not_converged_list[:-1]:
                    text += "{}, ".format(entry)
                text += "{}".format(not_converged_list[-1])
                
                ax.text(0.0, 1.02,
                        text,
                        color="r",
                        fontsize=12,
                        transform=ax.transAxes)
            
            if not combined_figure:
                plt.savefig("{}/{}_{}.png".format(structure_name, index, term),
                            dpi=400, bbox_inches="tight")
                plt.close()
    
    
    if combined_figure:
        
        ### Adding legend
        legend_lines = [
                Line2D([0], [0], color="orange", lw=7),
                Line2D([0], [0], color="g", lw=7),
                Line2D([0], [0], color="r", lw=7),
                Line2D([0], [0], color="yellow", lw=7),
                Line2D([0], [0], color="brown", lw=7),
                Line2D([0], [0], color="blue", lw=7)]
        legend_labels = ["Boettger PBE",
                         "Boettger PBE+TS",
                         "Boettger PBE+MBD",
                         "Linear PBE",
                         "Linear PBE+TS",
                         "Linear PBE+MBD"]
        
        if not pbe:
            keep_idx = [1,2,4,5]
            if not boettger:
                keep_idx = [4,5]    
                if not mbd:
                    keep_idx = [4]        
            elif not mbd:
                keep_idx = [1,4]
                
            legend_lines = [legend_lines[x] for x in keep_idx]
            legend_labels = [legend_labels[x] for x in keep_idx]
            
        elif not boettger:
            keep_idx = [3,4,5]
            
            if not mbd:
                keep_idx = [3,4]
            
            legend_lines = [legend_lines[x] for x in keep_idx]
            legend_labels = [legend_labels[x] for x in keep_idx]
        
        elif not mbd:
            keep_idx = [0,1,3,4]
            legend_lines = [legend_lines[x] for x in keep_idx]
            legend_labels = [legend_labels[x] for x in keep_idx]
        
    
        ## Can place in the bottom right corner because there's an open
        ## plot there. 
        ax = ax_list[-1]
        ax.axis("off")
        ax.legend(legend_lines,
                  legend_labels,
                  fontsize=fontsize,
                  loc="center")
        ax.text(0.0, 0.9,
                "CONVERGENCE THRESHOLD: {}%".format(threshold),
                color="r",
                fontsize=fontsize,
                transform=ax.transAxes)

        ## Turn other axes off
        for ax in ax_list[axes_idx:-1]:
            ax.axis("off")
        
        plt.tight_layout()
        fig.savefig("{}_Convergence_Plots.pdf".format(structure_name),
                    )
        
        plt.close()
    
    return energy_results