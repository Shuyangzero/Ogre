

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D


def sort_keys(miller_index):
    """
    Sort miller indices in a human-rational way
    
    Arguments
    ---------
    miller_index: iterable
        Any iterable that holds the miller indices as a iteratable of length 3. 
        
        
    Returns
    -------
    list
        Sorted list of miller indices where each entry is a numpy array. 
    """
    array = np.vstack(miller_index)
    pos_idx = np.where((array >= 0).all(axis=-1))[0]
    pos_array = array[pos_idx]
    pos_sum = np.sum(pos_array, axis=-1)
    pos_sort_idx = np.argsort(pos_sum, axis=-1)
    
    neg_idx = np.where((array < 0).any(axis=-1))[0]
    neg_array = array[neg_idx]
    neg_sum = np.sum(neg_array, axis=-1)
    neg_sort_idx = np.argsort(neg_sum, axis=-1)[::-1]
    
    final_list = [miller_index[x] for x in pos_idx[pos_sort_idx]]
    final_list += [miller_index[x] for x in neg_idx[neg_sort_idx]]
    
    return final_list


def color_wheel(miller_index):
    """
    Construct colors for miller incides from color wheel
    
    Argumnets
    ---------
    miller_index: iterable
        Any iterable that holds the miller indices as a iteratable of length 3.
    
    Returns
    -------
    list
        List of matplotlib rgb entries. 
        
    """
    color_wheel = {}
    color_wheel[(0,0,0)] = (1,1,1)
    color_wheel[(1,0,0)] = matplotlib.colors.to_rgb("red")
    color_wheel[(0,1,0)] = matplotlib.colors.to_rgb("green")
    color_wheel[(0,0,1)] = matplotlib.colors.to_rgb("blue")
    color_wheel[(-1,0,0)] = matplotlib.colors.to_rgb("yellow")
    color_wheel[(0,-1,0)] = matplotlib.colors.to_rgb("tab:pink")
    color_wheel[(0,0,-1)] = matplotlib.colors.to_rgb("tab:cyan")
    
    colors = []
    for key in miller_index:
        combinations = []
        
        if key in color_wheel:
            combinations.append(color_wheel[key])
        else:
            for entry_idx,entry in enumerate(key):
                temp_tuple = [0,0,0]
                
                if entry >= 1:
                    temp_tuple[entry_idx] = 1
                    if entry > 1:
                        combinations.append((0,0,0))
                        combinations.append((0,0,0))
                        combinations.append((0,0,0))
                elif entry <= -1:
                    temp_tuple[entry_idx] = -1
                    if entry < -1:
                        combinations.append((1,1,1))
                        combinations.append((1,1,1))
                        combinations.append((1,1,1))
                        
                temp_tuple = tuple(temp_tuple)
                
#                if temp_tuple == (0,0,0):
#                    continue
                
                combinations.append(color_wheel[temp_tuple])
        
        combinations = np.array(combinations)
        mean = np.mean(combinations, axis=0)
        
        colors.append(tuple(mean))
        
    return colors


def str2tuple(idx_str):
    """
    Convert a miller index string to a tuple. 
    
    Arguments
    ---------
    idx_str: str
        String for miller index such as "1,1,1"
        
    Returns
    -------
    tuple
        Returns integer tuple such as (1,1,1)
    
    """
    idx = []
    temp_idx = ""
    for char in idx_str:
        if char == "-":
            temp_idx += char
        else:
            temp_idx += char
            idx.append(temp_idx)
            temp_idx = ""
    idx = tuple([int(x) for x in idx])
    return idx


def wulffmaker_index(miller_index):
    """
    Returns the string to be used for the Wulffmaker default index. 
    
    Arguments
    ---------
    miller_index: iterable
        Any iterable that holds the miller indices as a iteratable of length 3. 
        
    Returns
    -------
    str
        String to be copied to wulffmaker for the miller indices. 
    
    """
    index_string = "pickIndex[i_, j_] :=\n"
    index_string += "Which[\n"
    
    for idx,index in enumerate(miller_index):
        ## Mathematica indexing starts at 1
        idx += 1
        
        temp_string = "i=={},\n".format(idx)
        temp_string += "Which[j==1,{},j==2,{},j==3,{},True,1],\n".format(
                        index[0], index[1], index[2])
        
        index_string += temp_string
        
    index_string += "True,\n"
    index_string += "RandomInteger[{-3, 3}]]"
        
    return index_string


def wulffmaker_gamma(energy):
    """
    Returns the string to be used for the Wulffmaker default gamma values. 
    
    Arguments
    ---------
    energy: iterable
        Any iterable that holds the surface energies
    
    Returns
    -------
    str
        String to be copied to wulffmaker for the surface energies. 
        
    """
    gamma_string = "pickGamma[i_] :=\n"
    gamma_string += "Which[\n"
    
    idx = 1
    for idx,value in enumerate(energy):
        idx += 1
        gamma_string += "i=={},\n".format(idx)
        gamma_string += "{:.4f},\n".format(value)
        
    gamma_string += "True,\n"
    gamma_string += "1]"
        
    return gamma_string
    

def wulffmaker_color(miller_index):
    """
    Returns the string to be used as the Wulffmaker pickColor default display
    settings. 
    
    Arguments
    ---------
    miller_index: iterable
        Any iterable that holds the miller indices as a iteratable of length 3. 
    
    Returns
    -------
    str
        String to be copied to wulffmaker for the surface colors. 
        
    """
    color_string = "pickColor[i_] :=\n"
    color_string += "Which[\n"
    
    colors = color_wheel(miller_index)
    for idx,_ in enumerate(miller_index):
        color = colors[idx]
        color_string += "i=={},\n".format(idx+1)
        color_string += "RGBColor[{:.6f},{:.6f},{:.6f}],\n".format(
                color[0], color[1], color[2])
        
    
    color_string += "True,\n"
    color_string += "Hue[RandomReal[], 1, 2/3]]"
        
    return color_string


def miller_index_legend(
        miller_index, 
        figname="legend.pdf",
        savefig_kw =
            {
                "dpi": 400
            },
        legend_kw = 
            {
                "loc": 'center',
                "ncol": 9,
                "labelspacing": 1,
                "columnspacing": 1,
                "handletextpad": 0.05,
                "borderpad": 1,
                "fontsize": 22,
                "frameon": False    
            },
        Line2D_kw=
            {
                "marker": "s",
                "color": "w",
                "markersize": 40,
            },
        figure_kw=
            {
                "figsize": (40,10),
            }
        ):
    """
    Create a legend for the miller indices. 
    
    Arguments
    ---------
    miller_index: iterable
        Any iterable that holds the miller indices as a iteratable of length 3. 
    figname: str
        File name to save the figure as. 
    savefig_kw: dict
        Key-word arguments to be passed to fig.savefig
    legend_kw: dict
        Key-word arguments passed into the matplotlib.axes.legend command
    Line2D_kw: dict
        Key-word arguments passed into the Line2D command which creates the 
        shape used in the plot for each miller index. Should not use the 
        markerfacecolor or label arguments because these are used by function. 
    figure_kw: dict
        Key-word arguments passed to matplotlib.pyplot.figure command. 
        
    """
    colors = color_wheel(miller_index)
    legend_elements = []
    for idx,miller in enumerate(miller_index):
        miller_str = "("
        miller_str += "".join([str(x) for x in miller])
        miller_str += ")"
        temp_legend_elements = [
            Line2D([0],[0], 
                   markerfacecolor=colors[idx],
                   label=miller_str,
                   **Line2D_kw)]
        legend_elements += temp_legend_elements
    
    fig = plt.figure(**figure_kw)
    ax = fig.add_subplot(111)
    
    ax.legend(handles=legend_elements,
              **legend_kw)
    
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(figname,
                **savefig_kw)
    plt.close()