# -*- coding: utf-8 -*-


import copy
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors 

from ibslib.analysis.pltlib import format_ticks
from ibslib.plot.utils import labels_and_ticks,\
                              truncate_colormap,\
                              colors_from_colormap


def text_plot(
        text="",
        text_kw =
            {
                "x": 0.025,
                "y": 0.5,
                "horizontalalignment": "left",
                "fontsize": 12,
                "wrap": True,
            },
        ax=None,
        figname="",
        fig_kw={}):
    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "ax":
            arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    
    fig = None
    if ax == None:
        fig = plt.figure(**arguments["fig_kw"])
        ax = fig.add_subplot(111)
    
    text_kw["s"] = text
    ax.text(**text_kw)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
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
    


def line(
    x,
    y = [],
    ax=None,
    figname="",
    fig_kw = {},
    plot_kw =
        {
            "color": "tab:blue",
            "linewidth": 1,
        },
    xlabel_kw = 
        {
            "xlabel": "Targets",
            "fontsize": 16,
            "labelpad": 0,
        },
    ylabel_kw = 
        {
            "ylabel": "Predicted",
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
        }
    ):
    """
    Create a basic line plot using potentially multiple lines.
    
    Arguments:
    x: list/array
        Can be a single 
    
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
        
    if len(arguments["y"]) > 0:
        ax.plot(x,y, **plot_kw)
    else:
        ax.plot(x, **plot_kw)
    
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
    
    

def lines(
    values,
    ax=None,
    figname="",
    fig_kw = {},
    legend_kw = 
        {
            "legend": True,
            "fontsize": 16,
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
    plot_kw = 
        {
            "linewidth": 2,
            "colors": [],
            "labels": [],
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
        }
    ):
    """
    Create a basic line plot composed of multiple lines.
    
    Arguments:
    values: list of lists/arrays
        Values for the lines you would like to plot. These must be formatted 
        such that each entry in the list has two lists/arrays, one for the 
        x values and one for the y values.
    color_map: matplotlib.cm object
        Name of a colormap from matplotlib you would like to use. This is by
        far the easiest way to color multiple lines. If you would like more 
        control over coloring, use plot_kw["colors"].
    
        
    """
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
    
    ### Build color mapping for each line
    cmap = eval("cm.{}".format(colormap["cmap"]))
    cmap = truncate_colormap(cmap, **colormap["truncate"])
    color_list = colors_from_colormap(len(values), cmap)
    
    if len(arguments["plot_kw"]["colors"]) == 0:
        arguments["plot_kw"]["colors"] = color_list
    
    ## Temporarily delete to use only the rest of plot_kw in following loop
    ## because colors will be set individually
    temp_colors = arguments["plot_kw"]["colors"].copy()
    temp_labels = arguments["plot_kw"]["labels"].copy()
    del(arguments["plot_kw"]["colors"])
    del(arguments["plot_kw"]["labels"])
    
    for idx,val in enumerate(values):
        ## Check if there will be out of index issue. If there will be, then 
        ## correct it by looping back around. 
        num_colors = len(temp_colors)
        if idx >= num_colors:
            loop = int(idx / num_colors) + 1
            idx = idx - loop*num_colors
            
        color = temp_colors[idx]
        if len(temp_labels) > 0:
            label = temp_labels[idx]
        
        if len(val) == 1:
            ax.plot(val, c=color, label=label, **arguments["plot_kw"])
        elif len(val) == 2:
            ax.plot(val[0], val[1], c=color, label=label, 
                    **arguments["plot_kw"])
        else:
            raise Exception("Values passed into lines must be a list of "+
                    "lists of lists containing either one or two entries " +
                    "to plot as the x and y coordinates.")
            
    format_ticks(ax)
    labels_and_ticks(ax, 
                     arguments["xlabel_kw"], 
                     arguments["ylabel_kw"], 
                     arguments["xticks"], 
                     arguments["yticks"])
    
    
    ## Restore colors to arguments
    arguments["plot_kw"]["colors"] = temp_colors
    arguments["plot_kw"]["labels"] = temp_labels

    if len(temp_labels) > 0 and arguments["legend_kw"]["legend"] == True:
        temp_legend_kw = arguments["legend_kw"].copy()
        del(temp_legend_kw["legend"])
        ax.legend(**temp_legend_kw)
    
    ## Cannot return ax object as argument
    del(arguments["ax"])
    # Return arguments that would recreate this graph
    return arguments
    
    

def hist(
    values,
    ax=None,
    figname="",
    fig_kw = {},
    hist_kw = 
        {
          "facecolor": "tab:blue",
          "edgecolor": "k"
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
    Create a basic histogram. 
    
    """
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
        
    ax.hist(values, **hist_kw)     
    
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


def ahist(
    value_list,
    bins=10, 
    density=False,
    ax=None,
    figname="",
    fig_kw = {},
    bar_kw = 
        {
          "edgecolor": "k",
          "color_list": [],
          "alpha_list": [],
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
    Prefered method for plotting multiple histograms on a single plot. The
    location of the bars for each plot are will be identical making cleaner
    looking comparisons of histograms.
    
    """
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
    
    
def hline():
    """
    Puts horizontal line on graph. 
    
    """
    raise Exception("Not Implemented")
    

def vline():
    """
    Puts vertical line on graph. 
    
    """
    raise Exception("Not Implemented")
    
    
    
    
if __name__ == "__main__":
    from ibslib.io import read,write 
    from ibslib.analysis import get
    
#    s = read("/Users/ibier/Research/Results/Hab_Project/genarris-runs/IBUZIP/20191207_/acsf_report/acsf/relaxed")
#    results = get(s, "prop", ["unit_cell_volume"])
#    values = results["unit_cell_volume"].values
#    
#    hist_kw = \
#        {
#          "facecolor": "tab:blue",
#          "edgecolor": "k",
#          "density": True,
#        }
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    
#    hist(values, hist_kw=hist_kw, ax=ax)
#    
#    hist_kw["facecolor"] = "lightgray"
#    hist(values*np.random.normal(1.0,0.1,size=values.shape), 
#         hist_kw=hist_kw, 
#         ax=ax)
    
    
    