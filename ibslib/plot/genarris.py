# -*- coding: utf-8 -*-



"""
Plotting functions specific for Genarris. These functions always return a 
dictionary of values that can be fed back into the funciton to create exactly 
the same graph.

"""

import copy
import numpy as np
import matplotlib.pyplot as plt 
from ibslib.analysis.pltlib import format_ticks
from ibslib.plot import labels_and_ticks
from ibslib.plot.ml import pca
from matplotlib.ticker import FormatStrFormatter


def plot_volume_hist(
    volume_values,
    pred_volume=-1,
    exp_volume=-1,
    ax=None,
    figname="",
    hist_kw = 
        {
          "facecolor": "tab:blue",
          "edgecolor": "k"
        },
    pred_axvline_kw =
        {
            "linestyle": "dashed",
            "linewidth": 3,
            "c": "tab:green"
        },
    exp_axvline_kw =
        {
            "linestyle": "dashed",
            "linewidth": 3,
            "c": "tab:red",
        },
    xlabel_kw = 
        {
            "xlabel": "Unit Cell Volume, $\AA^3$",
            "fontsize": 16,
            "labelpad": 0,
        },
    ylabel_kw = 
        {
            "ylabel": "Number of Structures",
            "fontsize": 16,
            "labelpad": 10, 
        },
    xticks = 
        {
            "xlim": [],
            "xticks_kw":
                {
                    "ticks": [],
                },
            "xticklabels_kw": 
                {
                    "labels": [],
                    "fontsize": 12,
                },
        
        },
    yticks = 
        {
            "ylim": [],
            "yticks_kw":
                {
                    "ticks": [],
                },
            "yticklabels_kw": 
                {
                    "labels": [],
                    "fontsize": 12,
                },
        
        },
    ):
    """
    Plots a Genarris volume histogram 
    
    Arguments
    ---------
    volume_values: list
        List or array of volume values from the pool of structures. It's easy 
        to obtain this by reading in the directory using ibslib.io.read and
        then getting values using ibslib.analysis.get.
    pred_volume: float
        Provide a value if the user wishes to draw the Genarris predicted 
        volume on the graph. 
    exp_volume: float
        Provide a vlue if the user wishes to draw an experimental volume 
        on the graph. 
    ax: matplotlib.pyplot.Axes 
        Provide if this plot is supposed to be drawn on an already existing
        Axes. If ax is None, a new figure is created. This gives the user
        the most flexibility. If you would like the figure to be a certain
        size, please initialize the figure first yourself and feed in the 
        Axes object. For example:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
        
    """
    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "volume_values":
            arguments_copy[key] = value
        elif key == "ax":
            arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    
    arguments["volume_values"] = list(arguments["volume_values"])
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    ax.hist(volume_values, **hist_kw)
    format_ticks(ax)
    
    labels_and_ticks(ax, arguments["xlabel_kw"], arguments["ylabel_kw"], 
                     arguments["xticks"], arguments["yticks"])
        
    if pred_volume > 0:
        ylim = ax.get_ylim()
        ax.axvline(pred_volume, ymin=ylim[0], ymax=ylim[1], **pred_axvline_kw)
    
    if exp_volume > 0:
        ylim = ax.get_ylim()
        ax.axvline(exp_volume, ymin=ylim[0], ymax=ylim[1], **exp_axvline_kw)
    
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    
    if len(figname) > 0:
        fig.savefig(figname)    
    
    ## Cannot return ax object as argument
    del(arguments["ax"])
    # Return arguments that would recreate this graph
    return arguments
        


def plot_spg_hist(
    spg_values,
    general_spg_values=[],
    special_spg_values=[],
    ax=None,
    general_bar_kw = 
        {
            "width": 0.8,
            "color": "tab:blue",
            "edgecolor": "k"
        },
    special_bar_kw = 
        {
            "width": 0.8,
            "color": "tab:orange",
            "edgecolor": "k",
        },
    exp_spg_arrow =
        {
        
        },
    xlabel_kw = 
        {
            "xlabel": "Space Group",
            "fontsize": 16,
            "labelpad": 0,
        },
    ylabel_kw = 
        {
            "ylabel": "Number of Structures",
            "fontsize": 16,
            "labelpad": 10, 
        },
    xticks = 
        {
            "xlim": [],
            "xticks_kw":
                {
                    "ticks": [],
                },
            "xticklabels_kw": 
                {
                    "labels": [],
                    "fontsize": 8,
                    "rotation": 90,
                },
        
        },
    yticks = 
        {
            "ylim": [],
            "yticks_kw":
                {
                    "ticks": [],
                },
            "yticklabels_kw": 
                {
                    "labels": [],
                    "fontsize": 12,
                },
        
        },    
    ):
    """
    Plots a Genarris space group histogram
    
    Arguments
    ---------
    spg_values: list
        List or array of volume values from the pool of structures. It's easy 
        to obtain this by reading in the directory using ibslib.io.read and
        then getting values using ibslib.analysis.get.
    general_spg_values: list
        List or array of space group numbers for which the molecule would sit 
        on a general position. If neither general_spg_values or 
        special_spg_values are provided, all spg_values are assumed to 
        be general positions.
    special_spg_values: list
        List or array of space group numbers for which the molecule would sit 
        on a specical position.  
    ax: matplotlib.pyplot.Axes 
        Provide if this plot is supposed to be drawn on an already existing
        Axes. If ax is None, a new figure is created. This gives the user
        the most flexibility. If you would like the figure to be a certain
        size, please initialize the figure first yourself and feed in the 
        Axes object. For example:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
    exp_spg_arrow: dict
        Dictionary of arguments to be use to draw an arrow pointing towards
        the experimental space group.
        
    """
    arguments = locals()
    arguments_copy = {}
    
    for key,value in arguments.items():
        if key == "spg_values":
            arguments_copy[key] = value
        elif key == "ax":
            arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    arguments["spg_values"] = list(arguments["spg_values"])
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    if len(general_spg_values) > 0 or len(special_spg_values) > 0:
        original_bins = general_spg_values + special_spg_values
        min_spg = min(original_bins)
        max_spg = max(original_bins)
        original_bins_final = [min_spg-1]
        original_bins_final += original_bins
        original_bins_final.append(max_spg+1)
        original_bins_final.sort()
    else:
        min_spg = min(spg_values)
        max_spg = max(spg_values)
        original_bins_final = np.arange(min_spg-1, max_spg+2, 1).tolist()
        
    ## Run hist the first time to get histogram setting values
    temp_fig = plt.figure()
    temp_ax = temp_fig.add_subplot(111)
    hist = temp_ax.hist(spg_values, bins=original_bins_final, edgecolor="k",
                   align="mid")
    
    #### Create maping from spg values to consecutive integers so that the 
    ## plot only shows all allowed spg values consecutively.
    all_spg = hist[1][1:-1]
    int_map = np.arange(0,len(all_spg))
    
    ### Now we're going to make a bar graph of the values we obtained, but 
    ## using different formatting for general and special positions
    if len(special_spg_values) > 0:
        special_height = []
        for value in special_spg_values:
            idx = np.where(hist[1] == value)[0]
            special_height.append(hist[0][idx][0])
        
        ## Refer to int_map to get x values
        special_int_idx = np.searchsorted(all_spg, special_spg_values)
        special_int_map = int_map[special_int_idx]
        
        ax.bar(special_int_map, special_height, **special_bar_kw)

    
    if len(general_spg_values) > 0:
        general_height = []
        for value in general_spg_values:
            idx = np.where(hist[1] == value)[0]
            general_height.append(hist[0][idx][0])
            
        ## Refer to int_map to get x values
        general_int_idx = np.searchsorted(all_spg, general_spg_values)
        general_int_map = int_map[general_int_idx]
        
        ax.bar(general_int_map, general_height, **general_bar_kw)
    
    # Handle behavior when both values are zero
    if len(special_spg_values) == 0 and len(general_spg_values) == 0:
        ax.bar(int_map, hist[0], **general_bar_kw)
    
    
    # Default for xtick values is over the observed range of tick values
    if len(arguments["xticks"]["xticks_kw"]["ticks"]) == 0 and \
        len(general_spg_values) == 0 and \
        len(special_spg_values) == 0:
        min_spg = min(spg_values)
        max_spg = max(spg_values)
        arguments["xticks"]["xticks_kw"]["ticks"] = np.arange(min_spg,max_spg+1,1)
    elif len(arguments["xticks"]["xticks_kw"]["ticks"]) == 0:
        ### Set xticks to int map
        arguments["xticks"]["xticklabels_kw"]["labels"] = [str(x) for x in all_spg] 
        arguments["xticks"]["xticks_kw"]["ticks"] = int_map.tolist()
        
    format_ticks(ax)
    labels_and_ticks(ax, arguments["xlabel_kw"], arguments["ylabel_kw"], 
                     arguments["xticks"], arguments["yticks"])
    
    
    if len(general_spg_values) > 0 or len(special_spg_values) > 0:
        for i,xtick in enumerate(ax.get_xticklabels()):
            if i in general_int_idx:
                xtick.set_color("tab:blue")
            elif i in special_int_idx:
                xtick.set_color("tab:orange")
    else:
        pass

#    ax.set_xticklabels(np.array(arguments["xticks"]["xticklabels_kw"]["labels"]).astype(int))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    
    ### Cannot return ax object as an argument
    del(arguments["ax"])
    return arguments


def scatter_3D():
    """
    Plots a scatter plot of the structures with respect to the a,b,c lattice 
    vectors and, if provided, the energy with a colorbar.
    
    """
    raise Exception("Not Implemented.")
    
    

def pca_exp(
    matrix,
    exp_data = 
        {
            "features": [],
            "energy": 0,
            "norm": True,
        },
    exp_scatter_kw = 
        {
            "marker": "x",
            "facecolor": "red",
            "edgecolor": "k",
            "linewidth": 3,
            "s": 50
        },
    ax = None,
    norm=True, 
    pca_components = [],
    targets = [],
    colormap = 
        {
            "cmap": "hot",
            "truncate":
                {
                    "minval": 0.0,
                    "maxval": 1.0,
                    "n": 10000,
                },
            "ylabel": "Total Energy, eV",
            "ylabel_kw":
                {
                    "rotation": 270,
                    "labelpad": 20,
                    "fontsize": 16
                },
            "ticks": [],
            "ticklabels": []
        },
    scatter_kw = 
        {
            "facecolor": "tab:blue",
            "edgecolor": "k",
            "linewidth": 1,
        },
    xlabel_kw = 
        {
            "xlabel": "Principle Component 1",
            "fontsize": 16,
            "labelpad": 0,
        },
    ylabel_kw = 
        {
            "ylabel": "Principle Component 2",
            "fontsize": 16,
            "labelpad": 10, 
        },
    xticks = 
        {
            "xlim": [],
            "xticks_kw":
                {
                    "ticks": [],
                },
            "xticklabels_kw": 
                {
                    "labels": [],
                    "fontsize": 12,
                },
            "FormatStrFormatter": "%.0f"
        },
    yticks = 
        {
            "ylim": [],
            "yticks_kw":
                {
                    "ticks": [],
                },
            "yticklabels_kw": 
                {
                    "labels": [],
                    "fontsize": 12,
                },
            "FormatStrFormatter": "%.2f"
        }    
    ):
    """
    Plots PCA with experimental datapoint. Expands on arguments for the 
    ibslib.plot.ml.pca with an experimental path argument. The projection 
    will be constructed using all of the features in the matrix argument. 
    Then, the features of the experimental will be projected using the fit 
    PCA components.
    
    Arguments
    ---------
    exp: dict
        Dictionary with entry with the experimental features and the energy 
        value for the experimental structure.
    
    """
    # Standard arguments augmentations to make json writtable
    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "matrix":
            if type(value) != list:
                value = value.tolist()
            arguments_copy[key] = value
        elif key == "ax":
            arguments_copy[key] = value
        elif key == "exp_data":
            if type(exp_data["features"]) != list:
                temp_features = list(value["features"])
                arguments_copy[key] = {"features": temp_features,
                                       "energy": value["energy"]}
            else:
                arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    pca_args = pca(matrix, 
                 ax=ax, 
                 norm=norm, 
                 pca_components=pca_components, 
                 targets=targets, 
                 colormap=colormap, 
                 scatter_kw=scatter_kw, 
                 xlabel_kw=xlabel_kw, 
                 ylabel_kw=ylabel_kw, 
                 xticks=xticks, 
                 yticks=yticks)
    
    ## Perform projection on experimental structure using the PCA components
    ## returned from pca plotting.
    
    ## But first, have to normalize exp features using features from projection
    if len(exp_data["features"]) > 0:
        if exp_data["norm"] == True:
            exp_norm = np.array(exp_data["features"]) - np.mean(np.array(matrix), 
                                                                axis=0)
        else:
            exp_norm = np.array(exp_data["features"])
        exp_projection = np.dot(exp_norm, 
                                np.array(pca_args["pca_components"]).T)
        ax.scatter(exp_projection[0,0], exp_projection[0,1], 
                   **exp_scatter_kw)
    
    ## Transfer results from pca_args to local copy of the arguments
    arguments["pca_components"] = pca_args["pca_components"]
    arguments["colormap"] = pca_args["colormap"]
    arguments["xticks"] = pca_args["xticks"]
    arguments["yticks"] = pca_args["yticks"]
    
    del(arguments["ax"])
    return arguments
      
    

    
if __name__ == "__main__":
    from ibslib.io import read,write
    from ibslib.analysis import get
    
        
    
#    struct_dict = read("/Users/ibier/Research/Results/Hab_Project/genarris-runs/BZOXZT/20191029_Calculations/BZOXZT_2mpc_raw_jsons")
#    results = get(struct_dict, "prop", ["unit_cell_volume","spg"])
#    volume_values = results["unit_cell_volume"].values
#    spg_values = results["spg"]
#    
#    temp = plot_volume_hist(volume_values, pred_volume=375, exp_volume=350)
#    temp = plot_spg_hist(spg_values, general_spg_values=[2,3,4,5], 
#                         special_spg_values=[7])
#    
#    temp = plot_dist_mat(dist_matrix)
    
    
    
    