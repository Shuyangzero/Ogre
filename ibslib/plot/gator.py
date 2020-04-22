# -*- coding: utf-8 -*-

import copy
import numpy as np 
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors 
from matplotlib.lines import Line2D

from ibslib.analysis.pltlib import format_ticks
from ibslib.plot.ml import pca
from ibslib.plot.utils import labels_and_ticks,\
                              truncate_colormap,\
                              colors_from_colormap

"""
Common GAtor plots.
"""


def energy_ranking():
    """
    Energy hierarchy type plot.
    
    """
    raise Exception("Not Implemented Yet.")
    

def max_it():
    """
    Max of a parameter, such as Hab, as a function of GA iteration.
    
    """
    raise Exception("Not Implemented Yet.")

def pca_exp(
    matrix,
    targets = [],
    exp_data = 
        {
            "features": [],
            "energy": 0,
            "norm": True,
        },
    exp_scatter_kw = 
        {
            "marker": "x",
            "facecolor": "tab:green",
            "edgecolor": "k",
            "alpha": 0.5,
            "linewidth": 3,
            "s": 100
        },
    plot_IP=False,
    IP_scatter_kw = 
        {
            "facecolor": "tab:pink",
            "edgecolor": "k"
        },
    ax = None,
    norm=True, 
    pca_components = [],
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
    Plots PCA for initial pool and GA.  Expands on arguments for the 
    ibslib.plot.ml.pca with an experimental path argument. The projection 
    will be constructed using all of the features in the IP and the GA pool. 
    Then, the features of the experimental will be projected using the fit 
    PCA components. Allows two modes of plotting. The first is to plot the 
    energy values as a colormap. The second mode of plotting is to plot the 
    colormap as a function of the order the structure was added. 
    
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
    
    ## ASSUMES THAT TARGETS IS THE ITERATION VALUE
    ## Should probably make seperate argument, but I don't have to for now
    if plot_IP:
        matrix = np.array(matrix)
        targets = np.array(targets)

        ## Get IP idx
        idx = np.where(targets == 0)[0]
        ip_matrix = matrix[idx]
        
        ## Perfom normalizatoin
        ip_matrix = ip_matrix - np.mean(matrix, axis=0)

        ip_proj = np.dot(ip_matrix,
                         np.array(pca_args["pca_components"]).T)
        
        ax.scatter(ip_proj[:,0], ip_proj[:,1],
                  **IP_scatter_kw)

        facecolor = IP_scatter_kw["facecolor"]
        ax.legend(handles=[Line2D([0], [0], marker='o', color='w', 
                    label='Initial Pool',
                    markerfacecolor=facecolor, 
                    markeredgecolor="k",
                    markersize=15)])

    ## Transfer results from pca_args to local copy of the arguments
    arguments["pca_components"] = pca_args["pca_components"]
    arguments["colormap"] = pca_args["colormap"]
    arguments["xticks"] = pca_args["xticks"]
    arguments["yticks"] = pca_args["yticks"]
    
    del(arguments["ax"])
    return arguments


def hist(
    value_list,
    bins=10, 
    density=False,
    ax=None,
    figname="",
    fig_kw = {},
    bar_kw = 
        {
          "edgecolor": "k",
          "color_list": ["tab:blue", "tab:red"],
          "alpha_list": [],
        },   
    legend_kw = 
        {
            "labels": ["Final GA Pool", "Initial Pool"],
            "loc": "upper right",
        },
    colormap = 
        {
            "cmap": "viridis",
            "truncate":
                {
                    "minval": 0.0,
                    "maxval": 1.0,
                    "n": 10000,
                }
        },            
    xlabel_kw = 
        {
            "xlabel": "",
            "fontsize": 16,
            "labelpad": 0,
        },
    ylabel_kw = 
        {
            "ylabel": "",
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
            "FormatStrFormatter": "%.2f"
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
        }): 
    """
    Plots a histogram of a property by overlaying the ip_values on top
    of the ga_values. 

    """
    if len(value_list) > 2:
        raise Exception("A list of two values must be paseed to GAtor hist.")


    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "values":
            arguments_copy[key] = list(value)
        elif key == "ax":
            arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    
    fig = None
    if ax == None:
        fig = plt.figure(**arguments["fig_kw"])
        ax = fig.add_subplot(111)
    
    ## Check for xlim to determine bin range behavior
    if len(arguments["xticks"]["xlim"]) != 0:
        pass
    else:
        ## Need to get min and max to have bins in these locations
        min_val = 0
        max_val = 0
        for values in value_list:
            temp_min = np.min(values)
            if temp_min < min_val:
                min_val = temp_min
            
            temp_max = np.max(values)
            if temp_max > max_val:
                max_val = temp_max
        
        arguments["xticks"]["xlim"] = [min_val, max_val]
    
    ## Set this at bottom so it's obvious how it's being used
    hist_range = arguments["xticks"]["xlim"]
    
    ## First compute the correct location of bin edges 
    bin_edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
    centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
    heights = np.diff(bin_edges)
    
    ## Now bin the data
    binned_data_sets = [
        np.histogram(d, range=hist_range, bins=bins,
                     density=arguments["density"])[0]
        for d in value_list]
    
    ## Prepare colors and alpha
    cmap = eval("cm.{}".format(colormap["cmap"]))
    cmap = truncate_colormap(cmap, **colormap["truncate"])
    color_list = colors_from_colormap(len(value_list), cmap)
    
    if len(arguments["bar_kw"]["color_list"]) == 0:
        arguments["bar_kw"]["color_list"] = color_list
    
    if len(arguments["bar_kw"]["alpha_list"]) == 0:
        arguments["bar_kw"]["alpha_list"] = [1 for x in
                                             range(len(binned_data_sets))]
    
    ## Copy for use below. This will be directly fed into ax.bar() calls
    bar_kw = arguments["bar_kw"].copy()
    del(bar_kw["color_list"])
    del(bar_kw["alpha_list"])
    
    ## We are ready to complete plotting
    for i,binned_data in enumerate(binned_data_sets):
        
        ## Remove numerical noise
        idx = np.where(binned_data > 0.001)[0]
        temp_centers = centers[idx]
        binned_data = binned_data[idx]
        temp_height = heights[idx]
        
        color = arguments["bar_kw"]["color_list"][i]
        alpha = arguments["bar_kw"]["alpha_list"][i]
        
        ax.bar(temp_centers, binned_data, width=temp_height, 
               color=color,
               alpha=alpha,
               **bar_kw)
    
    format_ticks(ax)
    labels_and_ticks(ax, 
                     arguments["xlabel_kw"], 
                     arguments["ylabel_kw"], 
                     arguments["xticks"], 
                     arguments["yticks"])

    ax.legend(**legend_kw)
    
    ## Save if figname provided
    if len(figname) > 0:
        if fig == None:
            fig = plt.figure(**arguments["fig_kw"])
            temp_ax = fig.add_subplot(111)
            temp_ax = ax
        fig.savefig(figname) 
    
    ## Cannot return ax object as argument
    del(arguments["ax"])
    # Return arguments that would recreate this graph
    return arguments
    


def avg_it(
    avg_values,
    all_values=[],
    ax=None,
    figname="",
    fig_kw = {},
    avg_plot_kw =
        {
            "color": "tab:blue",
            "linewidth": 2,
        },
    all_plot_kw =
        {
            "color": "tab:red",
            "marker": "o",
            "linestyle": "dashed",
        },
    xlabel_kw = 
        {
            "xlabel": "GA Iteration",
            "fontsize": 16,
            "labelpad": 10,
        },
    ylabel_kw = 
        {
            "ylabel": "Average Energy, kJ$\cdot$mol$^{-1}$",
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
            "FormatStrFormatter": "%i"
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
    Average of a parameter, such as energy, as a function of GA iteration.
    
    Arguments
    ---------
    avg_values: iterable
        Iterable containing the average values
    all_values: iterable,optional
        Iterable containing the actual value at each iteration. 

    """
    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "x" or key == "y":
            arguments_copy[key] = list(value)
        elif key == "ax":
            arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    
    fig = None
    if ax == None:
        fig = plt.figure(**arguments["fig_kw"])
        ax = fig.add_subplot(111)
    
    iterations = [x for x in range(len(avg_values))]
    
    ## Plot average
    ax.plot(iterations, avg_values, 
            **avg_plot_kw)

    ## Store ylim for average
    ylim = ax.get_ylim()

    ## Plot scatter of each point
    if len(all_values) > 0:
        ax.plot(iterations, all_values, **all_plot_kw)
        ax.set_ylim(ylim)
        # if len(arguments["yticks"]["ylim"]) == 0:
        #     arguments["yticks"]["ylim"] = ylim
    
    ## Standard Formatting
    format_ticks(ax)
    labels_and_ticks(ax, 
                     arguments["xlabel_kw"], 
                     arguments["ylabel_kw"], 
                     arguments["xticks"], 
                     arguments["yticks"])
    
    ## Save if figname provided
    if len(figname) > 0:
        if fig == None:
            fig = plt.figure(**arguments["fig_kw"])
            temp_ax = fig.add_subplot(111)
            temp_ax = ax
        fig.savefig(figname)    
    
    ## Cannot return ax object as argument
    del(arguments["ax"])
    # Return arguments that would recreate this graph
    return arguments


def min_it(
    values ,   
    ax=None,
    figname="",
    fig_kw = {},
    scatter_kw =
        {
            "color": "tab:blue",
        },
    line_kw =
        {
            "color": "tab:red",
        },
    xlabel_kw = 
        {
            "xlabel": "GA Iteration",
            "fontsize": 16,
            "labelpad": 10,
        },
    ylabel_kw = 
        {
            "ylabel": "Minimum Energy, kJ$\cdot$mol$^{-1}$",
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
            "FormatStrFormatter": "%i"
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
    Min of a parameter, such as energy, as a function of GA iteration.
    
    Arguments
    ---------
    values: iteratable
        Value for each iteration. 
    
    """
    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "x" or key == "y":
            arguments_copy[key] = list(value)
        elif key == "ax":
            arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    
    fig = None
    if ax == None:
        fig = plt.figure(**arguments["fig_kw"])
        ax = fig.add_subplot(111)
    
    iterations = [x for x in range(len(values))]
    
    ## Plot scatter
    ax.scatter(iterations, values, **scatter_kw)
    
    ## Now compute minimum at each iteration
    min_list = []
    for it_idx in range(len(values)):
        it_idx = it_idx+1
        min_list.append(min(values[:it_idx]))
        
    ax.plot(iterations, min_list, **line_kw)
    
    ## Standard Formatting
    format_ticks(ax)
    labels_and_ticks(ax, 
                     arguments["xlabel_kw"], 
                     arguments["ylabel_kw"], 
                     arguments["xticks"], 
                     arguments["yticks"])
    
    ## Save if figname provided
    if len(figname) > 0:
        if fig == None:
            fig = plt.figure(**arguments["fig_kw"])
            temp_ax = fig.add_subplot(111)
            temp_ax = ax
        fig.savefig(figname)    
    
    ## Cannot return ax object as argument
    del(arguments["ax"])
    # Return arguments that would recreate this graph
    return arguments
    