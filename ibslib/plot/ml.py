# -*- coding: utf-8 -*-



"""
Plotting scripts typically used with Machine Learning applications implemented
using an API that is compatible with the ibslib.report functionality.

"""

import copy
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ibslib.analysis.pltlib import format_ticks
from ibslib.plot import labels_and_ticks
from ibslib.plot import colorbar

def pvt(
    predicted, 
    targets, 
    ax=None,
    figname="",
    fig_kw = {},
    target_line = 
        {
            # min and max for line
            "lim": [],
            "plot_kw":
                {
                    "color": "tab:red",
                    "linewidth": 1,
                }
        },
    scatter_kw = 
        {
            "facecolor": "tab:blue",
            "edgecolor": "k",
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
    Scatter plot of predicted versus target with a line representing perfect
    prediction added. 
    
    """
    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "predicted" or key == "targets":
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
        
    ax.scatter(targets, predicted, **scatter_kw)
    
    format_ticks(ax)
    
    labels_and_ticks(ax, 
                     arguments["xlabel_kw"], 
                     arguments["ylabel_kw"], 
                     arguments["xticks"], 
                     arguments["yticks"])
    
    
    if len(arguments["target_line"]["lim"]) == 0:
        xlim = arguments["xticks"]["xlim"]
        ylim = arguments["yticks"]["ylim"]
        lower = min([xlim[0], ylim[0]])
        upper = max(xlim[1], ylim[1])
        arguments["target_line"]["lim"] = [lower, upper]
        
    ax.plot(arguments["target_line"]["lim"], 
            arguments["target_line"]["lim"],
            **arguments["target_line"]["plot_kw"])
    
    ax.set_xlim(arguments["xticks"]["xlim"])
    ax.set_ylim(arguments["yticks"]["ylim"])
    
    
    ## Save if figname provided
    if len(figname) > 0:
        if fig == None:
            fig = plt.figure()
        fig.savefig(figname)    
    
    ## Cannot return ax object as argument
    del(arguments["ax"])
    # Return arguments that would recreate this graph
    
    return arguments


def err_hist():
    """
    Histogram of the errors from the model.
    
    """
    raise Exception("Not Implemented Yet")
    

def confusion():
    """
    Plotting function for the confusion or similarity matrix. 
    """
    raise Exception("Not Implemented Yet")
    

def paired_err_hist():
    """
    Paired histogram of the errors from the model.
    """
    raise Exception("Not Implemented Yet")
    

def pca(
    matrix,
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
    Projection of input matrix using PCA to create a two dimensional plot. 
    Can pass in the pca vectors the user would like to use to project data 
    instead of fitting PCA. Can pass in targets to correlate unsupervised 
    projection with the target of interest. This will build colormap for 
    the targets. Additionally, any projection could be used 
    instead of PCA, but this would require copying the entirety of this 
    function and changing the projection method.
    
    Arguments
    ---------
    matrix: np.array
        Matrix to be projected down. 
    ax: matplotlib.pyplot.Axes
        Pass in if you would like to add the figure to an existing Axes.
    norm: bool
        If you would like to include the normalization that PCA performs
        if you include your own pca_components
    pca_components: np.array
        Principle axes from a previous PCA calculation. This can be used to 
        project the distance matrix in the same was as other previous 
        projects, such as during downselection. One would want the view the 
        downselection process using the same projection. 
    targets: 1D np.array
        Array of targets to correlate with PCA projection. If used, will 
        build colormap for the targets.
    colormap: dict
        Dictionary of arguments associated with the creation of the targets
        colormap. 
    
    """
    # Standard arguments augmentations to make json writtable
    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "matrix":
            if type(value) != list:
                value = value.copy().tolist()
            arguments_copy[key] = value
        elif key == "ax":
            arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    
    if type(matrix) == list:
        matrix = np.array(matrix)
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    from sklearn.decomposition import PCA
    
    ## If PCA components have not been provided
    if len(pca_components) == 0:
        pca_obj = PCA(n_components=2)
        results = pca_obj.fit_transform(matrix)
        arguments["pca_components"] = pca_obj.components_.tolist()
    else:
        if type(arguments["pca_components"]) == list:
            pca_components = np.array(arguments["pca_components"])
        
        if norm:
            norm_matrix = matrix - np.mean(matrix, axis=0)
        else:
            norm_matrix = matrix
            
        results = np.dot(norm_matrix, pca_components.T)

    if len(targets) == 0:
        ax.scatter(results[:,0], results[:,1], **scatter_kw)
    else:
        ## Perform colormapping 
        cmap = eval("cm.{}".format(colormap["cmap"]))
        p = ax.scatter(results[:,0], results[:,1],
                       c=targets,
                       cmap=cmap,
                       **scatter_kw)
        cbar = colorbar(p)
        cbar.ax.set_ylabel(colormap["ylabel"], **colormap["ylabel_kw"])
        
        ## Now handle cbar ticks
        if len(colormap["ticks"]) == 0 and \
           len(colormap["ticklabels"]) == 0:
            arguments["colormap"]["ticks"] = cbar.get_ticks().tolist()
        else:
            cbar.set_ticks(arguments["colormap"]["ticks"])
            if len(colormap["ticklabels"]) != 0:
                cbar.set_ticklabels(colormap["ticklabels"])
        
        ## Now save the tick values and labels to arguments as they have
        ## have been added. 
        locator, formatter = cbar._get_ticker_locator_formatter()
        ticks, ticklabels, _ = cbar._ticker(locator, formatter)
        arguments["colormap"]["ticks"] = ticks.tolist()
        arguments["colormap"]["ticklabels"] = ticklabels
            
    format_ticks(ax)
    labels_and_ticks(ax, 
                     arguments["xlabel_kw"], 
                     arguments["ylabel_kw"], 
                     arguments["xticks"], 
                     arguments["yticks"])
    
    del(arguments["ax"])
    return arguments
    
    
if __name__ == "__main__":
    pass
#    from ibslib.io import read,write
#    from ibslib.analysis import get
#    
#    s = read("/Users/ibier/Research/Volume_Estimation/Huang_Paper/Huang_Predicted_Results/json")
#    results = get(s,"prop",["density", "pred_density"])
#    
#    args = pvt(results["pred_density"], results["density"])
#    
#    args["xticks"]["xlim"] = (1,2.5)
#    args["xticks"]["xticks_kw"]["ticks"] = []
#    args["xticks"]["xticks_kw"]["xticklabels_kw"] = []
#    args["xticks"]["FormatStrFormatter"] = "%.1f"
#    
#    args["yticks"]["ylim"] = (1,2.5)
#    args["yticks"]["yticks_kw"]["ticks"] = []
#    args["yticks"]["yticks_kw"]["yticklabels_kw"] = []
#    args["yticks"]["FormatStrFormatter"] = "%.1f"
#    
#    args["target_line"]["lim"] = (0,3)
#    args["target_line"]["plot_kw"]["linewidth"] = 2
#    
#    
#    pvt(**args)
    
    
