# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.colors as clr
from mpl_toolkits.mplot3d import Axes3D

from textwrap import wrap

from ibslib.analysis import pltlib

from ibslib.io import read,write
from ibslib.motif.utils import eval_motif,implemented_motifs
from ibslib.analysis.pltlib import format_ticks,format_axis
from ibslib.analysis.dictops import get

###############################################################################
# Plotting Utilities                                                          #
###############################################################################


def standard_formatting(ax):
    ax.ticklabel_format(useOffset=False,linewidth=3)
    ax.spines['top'].set_linewidth(tick_width)
    ax.spines['right'].set_linewidth(tick_width)
    ax.spines['bottom'].set_linewidth(tick_width)
    ax.spines['left'].set_linewidth(tick_width)
    ax.tick_params(axis='both', which='major', labelsize=tick_size,
                   width=2, length=7)
    ax.grid(True, axis='both', which='both')


def add_legend(plot_list, ax):
    labs = [p.get_label() for p in plot_list]
    ax.legend(plot_list, labs, loc=0)


def get_marker_list():
    """ Returns custom marker list of most usable markers
    
    """
    marker_dict = Line2D.markers
    marker_list = []
    for marker_id in marker_dict:
        if marker_id == None or marker_id == 'None' or marker_id == '' \
           or marker_id == ' ' or marker_id == '.' or marker_id == ',':
               continue
        marker_list.append(marker_id)
    return marker_list


def get_marker_dict(input_list):
    """ Returns a dictionary of markers for input_list
    
    """
    if len(input_list) <= 6:
        marker_list = ['o','^','s','h','D','*']
    else:
        marker_list = get_marker_list()
    marker_dict = {}
    for i,value in enumerate(input_list):
        marker_dict[value] = marker_list[i]
    return marker_dict


def get_color_iter():
    colors = ['tab:orange', 'g', 'r', 'b', 'm', 'y', 
          'c']
    return colors


def get_iter_dict(input_list, iter_list):
    """
    Makes a dictionary which combines the input list and the iter list
    """
    iter_dict = {}
    for i,input_value in enumerate(input_list):
        iter_dict[input_value] = iter_list[i]
    return iter_dict


def marker_plot(ax, GAtor_Hab_values,GAtor_energy_values, 
                s=12, c='tab:grey', facecolors='none', label='GA',
                marker_list=[]):
    """
    For plotting by iterating over colors and markers
    """
    for i,Hab in enumerate(GAtor_Hab_values):
        energy = GAtor_energy_values[i]
        marker = marker_list[i]
        if type(c) == list:
            color = c[i]
        else:
            color = c
        if facecolors == 'none':
            ax.scatter(Hab,energy, s=s, edgecolor=color,
                       facecolors=facecolors,
                       marker=marker, linewidth=2)
        else:
            ax.scatter(Hab,energy, s=s, c=color,
                       marker=marker)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=10000):
    new_cmap = clr.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
  

###############################################################################
# Data Utilities                                                              #
###############################################################################


def prepare_prop_from_dict(input_dict, key):
    '''
    Returns a list of all values from the key of each member in the input_dict 
    '''
    value = []
    for member in input_dict:
        value.append(input_dict[member].get_property(key))
    return value


def correct_energy(energy_list, nmpc=4, global_min=None):
    '''
    Purpose:
        Corrects all energies values in the energy list relative to the global
          min and converts eV to kJ/mol/molecule.
    '''
    if global_min == None:
        global_min = min(energy_list)
    corrected_list = []
    for energy_value in energy_list:
        corrected_energy = energy_value - global_min
        corrected_energy = corrected_energy/(0.010364*float(nmpc))
        corrected_list.append(corrected_energy)
    return corrected_list


###############################################################################
# For plotting histograms of initial pool properties, primarily the space     #
#   group distribution                                                        #
###############################################################################

def plot_IP_hist(user_input, kwargs, 
                 hist_kwargs={"density": False,
                              "bins": "auto",
                              "edgecolor": "k"
                              }
                 ):
    '''
    Purpose:
        Plots the intial pool histogram for a path to a structure directory
          or the user can specify a structure dictionary as the input.
    Arguments:
        user_input: Can be either a path to a structure directory or a 
                      structure dictionary.
        prop: String of the property of interest.              
   '''
    if type(user_input) == dict:
        return plot_IP_hist_dict(user_input, hist_kwargs, **kwargs)
    elif type(user_input) == str:
        s = read(user_input)
        return plot_IP_hist_dict(s, hist_kwargs, **kwargs)
    
def plot_IP_hist_dir(struct_dir, **kwargs):
    '''
    Purpose:
        Plots for a directory of structures.
    '''
    struct_dict = read(struct_dir)
    return plot_IP_hist_dict(struct_dict, **kwargs)
    

def plot_IP_hist_dict(struct_dict,
                      hist_kwargs={"density": False,
                                   "bins": "auto",
                                   "edgecolor": "k"
                                   },
                      prop='space_group',nmpc=2,
                      deduce_sg=False,
                      xlabel='Space Group', ylabel='Observed Distribution',
                      figname=None,  
                      GAtor_IP=False,
                      label_size=24, tick_size=18, figure_size=(12,8),
                      tick_width=3):
    '''
    Purpose:
        Plots histogram of prop from the struct_dict.
    Arguments:
        GAtor_IP: Will print the property value in the histogram from just the 
                    initial pool along with the entire population. 
    '''
    values = prepare_prop_from_dict(struct_dict, prop)

    if any(elem is None for elem in values):
        if all(elem is None for elem in values):
            raise Exception('Structures are missing the property {}.'
                            ' Please check that the property exists.'
                            .format(prop))
        else:
            print('Not all structures have the desired property.')
            none_list = list_struct_without_prop(struct_dict, prop)
            print('Structures without {}: {}'.format(prop,none_list))
            values = [x for x in values if x is not None]
        
    
    if type(deduce_sg) == bool:
        if deduce_sg:
            allowed_sg = deduce_allowed_sg(nmpc)
            bins = max(allowed_sg) - min(allowed_sg) + 1
    elif type(deduce_sg) == list:
        allowed_sg = deduce_sg
    
    
    # Standard formatting 
    fig = plt.figure(figsize=figure_size)
    ax1 = fig.add_subplot(111)
    ax1 = fig.gca()

    ax1.spines['top'].set_linewidth(tick_width)
    ax1.spines['right'].set_linewidth(tick_width)
    ax1.spines['bottom'].set_linewidth(tick_width)
    ax1.spines['left'].set_linewidth(tick_width)
    ax1.tick_params(axis='both', which='major', labelsize=tick_size)
    ax1.tick_params(which='both', width=4, length=7)
    
    if GAtor_IP == True:
        IP_dict = {}
        for struct_id in struct_dict:
            if 'init' == struct_id[0:4]:
                IP_dict[struct_id] = struct_dict[struct_id]
        IP_values = prepare_prop_from_dict(IP_dict, prop)
        plt.hist([values,IP_values], bins='auto', density=[density,density], 
                 color = ['b','tab:orange'], label=['GAtor','Initial Pool'])
        plt.legend(loc='upper right', fontsize=tick_size)
        ax1.set_ylabel('Frequency', fontsize=label_size, labelpad=25)
        ax1.set_xlabel('Space Group', fontsize=label_size)
        return

    hist1 = plt.hist(values, **hist_kwargs)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size, labelpad=25)
    
    if deduce_sg and GAtor_IP==False:
    # Easy way to make plots that share an x axis with a new y-axis
        ax2 = ax1.twinx()
        # Example of useful tick params
#        ax2.tick_params(
#            axis='x',          # changes apply to the x-axis
#            which='both',      # both major and minor ticks are affected
#            bottom=False,      # ticks along the bottom edge are off
#            top=False,         # ticks along the top edge are off
#            labelbottom=False) # labels along the bottom edge are off
        ax2.tick_params(axis='both', which='major', labelsize=tick_size)
        ax2.tick_params(which='both', width=4, length=7)
        if type(deduce_sg) == bool:
            hist2 = plt.hist(allowed_sg, alpha=0.33,
                                 color='tab:orange',
                                 **hist_kwargs)
            ax2.set_yticks([])
            ax2.set_ylabel('All Possible Space Groups', rotation=270,
                       labelpad=35, fontsize = label_size)
            
        elif type(deduce_sg) == list:
            hist2 = ax2.hist(allowed_sg[0], alpha=0.33,
                                 color='tab:orange',
                                 **hist_kwargs)
            hist3 = ax2.hist(allowed_sg[1], alpha=0.33,
                                 color='tab:red',
                                 **hist_kwargs)
            ax2.set_yticks([])
            ax2.set_ylabel('All Possible Space Groups', rotation=270,
                       labelpad=35, fontsize = label_size)
        

    
    fig.tight_layout()
    if figname != None:
        fig.savefig(figname)
    
    plt.show()
    return ax1


def deduce_allowed_sg(nmpc, is_chiral=None, is_racemic=None):
    '''
    Purpose:
        Deduces the allowed space groups using the general Wyckoff posittion
          and the number of molecules per cell. 
    '''
    SGM = SpaceGroupManager(nmpc, is_chiral, is_racemic)
    SGM._deduce_allowed_space_groups()
    sgs = SGM._space_group_range
    
    return sgs


def list_struct_without_prop(struct_dict, key):
    none_list = []
    for struct_id in struct_dict:
        value = struct_dict[struct_id].get_property(key)
        if value == None:
            none_list.append(struct_id)
    return none_list

###############################################################################
# For plotting GAtor Hab plots with initial pools as a different color        #
###############################################################################
    

def plot_GA_prop(struct_dir, prop_key, energy_key, nmpc, 
                   init_str = 'init',
                   experimental_parameters=[], 
                   experimental_path='',
                   figname=None,
                   motif=False, color_ip=True, 
                   use_ip=True, ip_only=False):
    '''
    General function for relative energy versus property for Structure pools 
      from GAtor. There are many arugments used to contorl the plotting 
      behavior.
     
    Arguments
    ---------
    struct_dir: path
        Path to structure directory which is to be used for plotting
    prop_key: str
        String of desired stored property in the Structure file
    energy_key: str
        String of desired energy to be used in the Structure file
    nmpc: int
        Number of molecules per cell
    init_str: str
        String used to determine which structures were from the initial pool
    experimental_parameters: list [name, energy (eV), prop_value]
        Pass in if the experimental properties are to be plotted on the graph
    experimental_struct_path: path
        Path to the experimental Structure file to use. 
        Will not use if length of path str is 0
    figname: path
        If provided, the file will be saved at the provided path
    motif: bool
        True: Indicates to use motif property displayed as plot markers.
    color_ip: bool
        True:  Motifs are colored by either IP or GAtor
        False: Motifs are colored for best clarity
    use_ip: bool
        True:  IP structures are used
        False: IP structures are first separated and not in final plot
    ip_only: bool
        True:  Only plots IP
        False: No change to plot
    
    '''
    struct_dict = read(struct_dir)
    
    struct_dict = clean_struct_dict_prop(struct_dict, prop_key)
    GAtor_struct_dict, IP_struct_dict = separate_IP(struct_dict, init_str)
    
    if not use_ip:
        IP_struct_dict = {}
        struct_dict = GAtor_struct_dict
    
    if ip_only:
        if not use_ip:
            raise Exception("Arguments use_ip and ip_only to function "+
                            "plot_GA_prop cannot both be False.")
        struct_dict = IP_struct_dict
    
    if color_ip == True:
        plot_GAtor_Hab_dict(GAtor_struct_dict, IP_struct_dict,
                            prop_key, energy_key, nmpc, 
                            experimental_parameters,
                            motif=motif, figname=figname)
    else:
        plot_energy_property_motif(struct_dict, 
                                   prop_key, energy_key, nmpc, 
                                   experimental_parameters,
                                   figname=figname)


def clean_struct_dict_prop(struct_dict, Hab_key):
    '''
    Purpose:
        Remove any structures from the struct dict that don't have the Hab key.
    '''
    temp_struct_dict = {}
    for struct_id in struct_dict:
        struct = struct_dict[struct_id]
        Hab_value = struct.get_property(Hab_key)
        if Hab_value == None:
            continue
        else:
            temp_struct_dict[struct_id] = struct_dict[struct_id]
    return temp_struct_dict

def clean_struct_dict_energy(struct_dict, energy_key):
    '''
    Purpose:
        Removes any structures from the struct dict that have an energy value 
          of 0 which would generally indicate that the energy evaluation did 
          not successfully complete. 
    '''
    temp_struct_dict = {}
    for struct_id in struct_dict:
        struct = struct_dict[struct_id]
        energy_value = struct.get_property(energy_key)
        if energy_value == 0:
            continue
        else:
            temp_struct_dict[struct_id] = struct_dict[struct_id]
    return temp_struct_dict

def separate_IP(struct_dict, init_str = 'init'):
    """
    init_str: str
        String used to determine which structures were from the initial pool
    """
    GAtor_struct_dict = {}
    IP_struct_dict = {}
    for struct_id in struct_dict:
        if init_str == struct_id[0:4]:
            IP_struct_dict[struct_id] = struct_dict[struct_id]
        elif init_str in struct_id:
            IP_struct_dict[struct_id] = struct_dict[struct_id]
        else:
            GAtor_struct_dict[struct_id] = struct_dict[struct_id]
    return GAtor_struct_dict, IP_struct_dict

def plot_GAtor_Hab_dict(GAtor_struct_dict, IP_struct_dict, Hab_key, energy_key,
                        nmpc, experimental_parameters = [], 
                        motif=False, figname=None):
    '''
    Arguments
    ---------
        experimetnal_parameters: [name, energy (eV), Hab_value]
        motif: Bool
           True: Evaluate motifs of GAtor_struct_dict and use markers to 
                   distinguish between motifs
           False: Do not evaluate motifs of structures
    '''
    IP_Hab_values = prepare_prop_from_dict(IP_struct_dict, Hab_key)
    IP_energy_values = prepare_prop_from_dict(IP_struct_dict, energy_key)
    GAtor_Hab_values = prepare_prop_from_dict(GAtor_struct_dict, Hab_key)
    GAtor_energy_values = prepare_prop_from_dict(GAtor_struct_dict, energy_key)
    
    if motif:
        motif_list = implemented_motifs()
        marker_dict = get_marker_dict(motif_list)
        IP_motifs = prepare_prop_from_dict(IP_struct_dict,'motif')
        if None in IP_motifs:
            IP_motifs = eval_motif(IP_struct_dict)
        IP_markers = []
        for motif in IP_motifs:
            IP_markers.append(marker_dict[motif])
        
        GAtor_motifs = prepare_prop_from_dict(GAtor_struct_dict, 'motif')
        if None in GAtor_motifs:
            GAtor_motifs = eval_motif(GAtor_struct_dict)
        GAtor_markers = []
        for motif in GAtor_motifs:
            GAtor_markers.append(marker_dict[motif])
    else:
        IP_markers = 'o'
        GAtor_markers = 'o'
    
    # Not a great way to do this but just combining lists
    total_energy_list = []
    total_energy_list += IP_energy_values[:]
    total_energy_list += GAtor_energy_values[:]
    
    if len(experimental_parameters) == 3:
        total_energy_list.append(experimental_parameters[1])
    
    global_min = min(total_energy_list)
    
    IP_energy_values = correct_energy(IP_energy_values, nmpc,
                                      global_min=global_min)
    GAtor_energy_values = correct_energy(GAtor_energy_values, nmpc,
                                         global_min=global_min)

    # Standard formatting values
    label_size = 24
    tick_size = 24
    figure_size=(12,8)
    width = 6
    point_size = 75
    
    fig = plt.figure(figsize=figure_size)
    ax = fig.gca()
    
    if motif:
        marker_plot(ax, IP_Hab_values, IP_energy_values, 
                    s=point_size, c='b', facecolors='none',
                    label='Initial Pool',
                    marker_list=IP_markers)
        marker_plot(ax, GAtor_Hab_values, GAtor_energy_values, 
                    s=point_size, c='r', facecolors='none',
                    label='Initial Pool',
                    marker_list=GAtor_markers)
    else:
        plt.scatter(GAtor_Hab_values,GAtor_energy_values, 
                    s=point_size, c='tab:grey', label='GA')
        plt.plot(IP_Hab_values, IP_energy_values, 
                    s=point_size, c='b', label='Initial Pool')
    
    if len(experimental_parameters)==3:
        experimental_energy = correct_energy([experimental_parameters[1]],
                                             nmpc=nmpc,global_min=global_min)
        plt.scatter(experimental_parameters[2],experimental_energy,
                    s=point_size, c='tab:orange', 
                    label=experimental_parameters[0])
    
    plt.xlabel("$H^{max}_{ab}$ (meV)", fontsize=34)

    ax.set_ylabel("\n"
                  .join(wrap('Relative Energy per Molecule (kJ mol$^{-1}$ molecule$^{-1}$)', 31)),
                  fontsize=32,labelpad=-2)
    
    ax.ticklabel_format(useOffset=False,linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    # Border Thickness
    ax.spines['top'].set_linewidth(width)
    ax.spines['right'].set_linewidth(width)
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['left'].set_linewidth(width)
    
#    ax.set_xlim([0,220])
#    ax.set_ylim([0,50])
    
    if motif == False:
        plt.legend(fontsize=18)
    else:
        marker_list = [x for _,x in marker_dict.items()]
        handle_list = []
        for i,marker in enumerate(marker_list):
            handle_list.append(mlines.Line2D([], [], color='blue', 
                          marker=marker, linestyle='None',
                          markersize=10, label=motif_list[i]))
        plt.legend(handles=handle_list)
    
    plt.tight_layout()
    plt.show()
    
    if figname != None:
        fig.savefig(figname)

    return ax

        
###############################################################################
# For plotting general property versus energy plots with one or more entries  #
###############################################################################


def plot_many_property_vs_energy(struct_dir_list, prop_key_list, 
                                 energy_key_list, legend_list,
                                 nmpc_list, **kwargs):
    '''
    Purpose:
        On a single plot, plot property versus energy for one or more structure
          directories. Primarily developed for plotting Habmax versus energy, 
          but should work for other scalar properties as well.
    '''
    struct_dict_list = []
    for struct_dir in struct_dir_list:
        struct_dict = read(struct_dir)
        struct_dict_list.append(struct_dict)   
    plot_many_property_vs_energy_dict(struct_dict_list, prop_key_list, 
                                      energy_key_list, legend_list,
                                      nmpc_list, **kwargs)
    return struct_dict_list


def plot_many_property_vs_energy_dict(struct_dict_list, prop_key_list, 
                                      energy_key_list, legend_list,
                                      nmpc_list, 
                                      experimental_data=[],
                                      figure_size=(12,8),
                                      figname=None,
                                      xlabel='',
                                      ylabel='',
                                      label_size = 24,
                                      tick_size = 24,
                                      tick_width = 3,
                                      border_width = 3,
                                      point_size = 10,
                                      point_width=2,
                                      marker_list=None,
                                      color_list=None,
                                      legend_columns=1,
                                      legend_font_size=16,
                                      labelpad_x=0,
                                      labelpad_y=-5):
    
    fig = plt.figure(figsize=figure_size)
    ax = fig.gca()     
    
    ax.tick_params(axis='both', which='major', labelsize=tick_size,
                   width=tick_width)
    # Border Thickness
    ax.spines['top'].set_linewidth(border_width)
    ax.spines['right'].set_linewidth(border_width)
    ax.spines['bottom'].set_linewidth(border_width)
    ax.spines['left'].set_linewidth(border_width)
    
    
    if color_list == None:
        color_list = cm.rainbow(np.linspace(0,1,len(struct_dict_list)))
    
    if marker_list == None:
        marker_dict = Line2D.markers
        for marker_id in marker_dict:
            if marker_id == None or marker_id == 'None' or marker_id == '' \
               or marker_id == ' ' or marker_id == '.' or marker_id == ',':
                   continue
            marker_list.append(marker_id)
    
    for i,struct_dict in enumerate(struct_dict_list):
        struct_dict = clean_struct_dict_energy(struct_dict, energy_key_list[i])
        energy_values = prepare_prop_from_dict(struct_dict, energy_key_list[i])
        prop_values = prepare_prop_from_dict(struct_dict, prop_key_list[i])
        
        if len(experimental_data) != 0:
            energy_values.append(experimental_data[i][0])
            prop_values.append(experimental_data[i][1])
        
        energy_values = correct_energy(energy_values, nmpc_list[i])
        plt.plot(prop_values, energy_values, marker_list[i], c=color_list[i], 
                    label=legend_list[i], fillstyle='none', 
                    markersize=point_size, markeredgewidth=point_width)
        
        if len(experimental_data) != 0:
            plt.plot(prop_values[-1],energy_values[-1],
                     marker_list[i], c=color_list[i], fillstyle='full', 
                     markersize=point_size+5, markeredgewidth=point_width,
                     markeredgecolor='k')
        
    plt.xlabel(xlabel, fontsize=label_size,labelpad=labelpad_x)
    if len(ylabel) == 0:
        plt.ylabel("\n"
                  .join(wrap('Relative Energy per Molecule (kJ mol$^{-1}$ molecule$^{-1}$)', 31)),
                  fontsize=label_size,labelpad=labelpad_y)
    plt.legend(loc='upper right', fontsize=legend_font_size,
               ncol=legend_columns)
    plt.tight_layout()
    plt.show()
    if figname != None:
        fig.savefig(figname)        


###############################################################################
# Plotting lattice+property distribution plots                                #
###############################################################################
        
def plot_lattice(struct_dict, prop="", 
                 figname="",
                 colormap=cm.hot,
                 colormap_lim=(0.1,0.6),
                 relative_energy=False,
                 nmpc=4,
                 experimental_cell=[],
                 experimental_plot_text_kwargs = 
                     {
                        "text": "X",
                        'family': 'serif',
                        'color':  'green',
                        'weight': 'normal',
                        'size': 28,
                     },
                 figsize=(8,8),
                 scatter_kwargs=
                     {
                        "s": 100, 
                     },
                 xlim=[], ylim=[], zlim=[],
                 xlabel_kwargs=
                     {
                       "xlabel": 'a ($\AA$)',
                       "fontsize": 12,
                       "labelpad": 10
                     },
                 ylabel_kwargs=
                     {
                       "ylabel": 'b ($\AA$)',
                       "fontsize": 12,
                       "labelpad": 10
                     },
                 zlabel_kwargs=
                     {
                       "zlabel": 'c ($\AA$)',
                       "fontsize": 12,
                       "labelpad": 10
                     },
                 cbar_title_kwargs=
                     {
                       "ylabel": 'Relative Energy (kJ mol$^{-1}$)',
                       "fontsize": 12,
                       "rotation": 270,
                       "labelpad": 30
                     },
                 tick_params_kwargs=
                     {
                        "axis": "both",
                        "labelsize": 12,
                        "width": 3,
                        "length": 5,
                     },
                 cbar_tick_params_kwargs=
                     {
                        "axis": "both",
                        "labelsize": 12,
                        "width": 2,
                        "length": 2,
                     },
                 cbar_ticks=[],
                 xticks=[],
                 yticks=[],
                 zticks=[]
                  ):
    """
    Creates a 3D plot of the lattice parameters, each colored according to a 
    input property value. If no input property is provided, then the scatter
    plot will not include a colorbar. 
    
    Arguments
    ---------
    struct_dict: StructDict
        Dictionary of structure objects. 
    prop: str
        String of property to obtain from each structure. Default bevavior 
        is to not include a property. 
    figname: str
        If figname is provided, the plot will be saved at the provided string. 
    colormap: matplotlib.cm.cmap object
        Matplotlib colormap to use.
    colormap_lim: (float,float)
        If provided, truncates colormap to be between the provided limits. 
        This removes colors which are too dark or too light.
    relative_energy: bool
        If true, corrects all energies values in the prop list relative to 
        the global min and converts eV to kJ/mol/molecule.
    nmpc: int
        If using relative energy, nmpc is required to get lattice energy
        per mol.
    experimental_cell: [float, float, float]
        The a,b,c lattice parameters of the experimental structure. If 
        the user provides this argument, then the experimental will be added
        to the graph as a separate shape. 
    experimental_plot_text_kwargs:
        Controls how the experimental will be added to the plot. Kwargs 
        are passed to ax.text function.
    figsize: (int, int)
        Figure size argument for plt.figure
    scatter_kwargs: dict
        Keyword arguments to use in plt.scatter.
    xlim,ylim,zlim: [float, float]
        If lists are provided, these are use to set the range of the 
        x,y, or z axes. 
    xlabel_kwargs,ylabel_kwargs,zlabel_kwargs: dict
        Keyword argument dictionaries to pass to plt.set_xlabel. Can include
        axis labels, parameters for the font, etc. 
    cbar_title_kwargs: str
        Keyword argument dictionary for the color map. 
        Used if a property string is provided.  
    tick_params_kwargs: dict of keyword arguments
        Keyword arguments passed to ax.tick_params.
    xticks,yticks,ztickss: dict of keyword arguments
        Manytimes, it is necessary to specify the specific ticks to use in
        order for the tick numbers to not overlap poorly.
    
    
    """
    
    prop_list = ["a","b","c"]
    if len(prop) > 0:
        prop_list.append(prop)
    
    results_df = get(struct_dict,"prop",prop_list)
    
    if relative_energy and len(prop) > 0:
        results_df[prop] = correct_energy(results_df[prop],nmpc)
        
    if len(colormap_lim) > 0:
        colormap = truncate_colormap(colormap, colormap_lim[0], 
                                     colormap_lim[1])
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if len(experimental_cell) != 0:
        ax.text(experimental_cell[0], 
                experimental_cell[1], 
                experimental_cell[2],  
                **experimental_plot_text_kwargs) 
    
    if len(prop) > 0:
        p = ax.scatter(results_df["a"],results_df["b"], 
                       results_df["c"], c=results_df[prop], 
                        cmap=colormap, **scatter_kwargs) 
        cbar = plt.colorbar(p)
        cbar.ax.set_ylabel(**cbar_title_kwargs)
        cbar.ax.tick_params(**cbar_tick_params_kwargs)
    else:
         p = ax.scatter(results_df["a"],results_df["b"], 
                       results_df["c"], **scatter_kwargs) 
         
    ax.set_xlabel(**xlabel_kwargs)
    ax.set_ylabel(**ylabel_kwargs)
    ax.set_zlabel(**zlabel_kwargs)
    
    if len(xlim) > 0:
        ax.set_xlim(xlim[0], xlim[1])
    if len(ylim) > 0:
        ax.set_ylim(ylim[0], ylim[1])
    if len(zlim) > 0:
        ax.set_zlim(zlim[0], zlim[1])
    
    ax.tick_params(**tick_params_kwargs)
    if len(xticks) > 0:
        ax.set_xticks(xticks)
    if len(yticks) > 0:
        ax.set_yticks(yticks)
    if len(zticks) > 0:
        ax.set_zticks(zticks)
    
    plt.tight_layout()
    if len(figname) > 0:
        fig.savefig(figname)
    plt.show()


###############################################################################
# More motif                                                                  #
###############################################################################
        
def motif_prop_plot(struct_dir, property_key, error=True, label_kwargs={},
                    figname=''):
    """ Bar plot of property w.r.t. motifs
    
    """
    
    struct_dict = read(struct_dir)
    struct_dict = clean_struct_dict_prop(struct_dict, property_key)
    
    motif_list = np.array(implemented_motifs())
    num_motif = len(motif_list)
    motif_index_dict = {}
    for i,motif in enumerate(motif_list):
        motif_index_dict[motif] = i
        
    struct_motif_list = prepare_prop_from_dict(struct_dict, 'motif')
    property_list = prepare_prop_from_dict(struct_dict, property_key)
    if 'energy' in property_key:
        property_list = correct_energy(property_list)
    
    hist_list = [[] for x in range(len(motif_list))]
    for i,motif in enumerate(struct_motif_list):
        motif_index = motif_index_dict[motif]
        hist_list[motif_index].append(property_list[i])
        
    
    hist_mean = np.zeros(num_motif)
    hist_error = np.zeros(num_motif)
    for i,hist in enumerate(hist_list):
        if len(hist) == 0:
            hist_error[i] = 0
            hist_mean[i] = 0
        else:
            mean = np.mean(hist)
            hist_error[i] = np.std(hist)
            hist_mean[i] = mean
    
    motif_index_sort = np.flip(np.argsort(hist_mean), 0)
    hist_mean = hist_mean[motif_index_sort]
    hist_error = hist_error[motif_index_sort]
    motif_list = motif_list[motif_index_sort]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    format_ticks(ax)
    format_axis(ax, **label_kwargs)
    if error == True:
        ax.bar(motif_list, hist_mean, yerr=hist_error, ecolor='black', 
               capsize=5,edgecolor='k')
    else:
        ax.bar(motif_list, hist_mean, edgecolor='k')
    
    plt.tight_layout()
    if len(figname) > 0:
        fig.savefig(figname)
    
    return hist_mean, hist_error


def motif_dist_plot(struct_dir, label_kwargs={},
                    figname='', ip_only=False, init_str='init'):
    """ Bar plot of number of observed motifs
    
    """
    
    struct_dict = read(struct_dir)
    struct_dict = clean_struct_dict_prop(struct_dict, 'motif')
    
    if len(struct_dict) == 0:
        raise Exception("It appears that the structure directory {} "
                "has no structures with motif evaluated".format(struct_dir))
    
    if ip_only:
        GAtor_struct_dict, IP_struct_dict = separate_IP(struct_dict, init_str)
        struct_dict = IP_struct_dict
    
    motif_list = np.array(implemented_motifs())
    motif_index_dict = {}
    for i,motif in enumerate(motif_list):
        motif_index_dict[motif] = i
        
    struct_motif_list = prepare_prop_from_dict(struct_dict, 'motif')
    
    hist_list = [0 for x in range(len(motif_list))]
    for i,motif in enumerate(struct_motif_list):
        motif_index = motif_index_dict[motif]
        hist_list[motif_index] += 1
    
    total = np.sum(hist_list)
    hist_list = hist_list / total
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    format_ticks(ax)
    format_axis(ax, **label_kwargs)
    ax.bar(motif_list, hist_list, edgecolor='k')
    
    plt.tight_layout()
    if len(figname) > 0:
        fig.savefig(figname)
    
    return None

def plot_energy_property_motif(struct_dict, property_key, energy_key, 
                               nmpc, experimental_parameters = [], 
                               figname=None):
    prop_values = prepare_prop_from_dict(struct_dict, property_key)
    energy_values = prepare_prop_from_dict(struct_dict, energy_key)
    motifs_values = prepare_prop_from_dict(struct_dict,'motif')
    if None in motifs_values:
        motifs_values = eval_motif(struct_dict)
    

    motif_list = implemented_motifs()
    marker_dict = get_marker_dict(motif_list)
    motif_markers = []
    for motif in motifs_values:
        if motif == 'Harringbone':
            motif = 'Herringbone'
        motif_markers.append(marker_dict[motif])
    
    color_list = get_color_iter()
    color_dict = get_iter_dict(motif_list, color_list)
    motif_colors = []
    for motif in motifs_values:
        if motif == 'Harringbone':
            motif = 'Herringbone'
        motif_colors.append(color_dict[motif])
    
    total_energy_list = energy_values
    
    if len(experimental_parameters) == 3:
        total_energy_list.append(experimental_parameters[1])
    
    global_min = min(total_energy_list)
    
    energy_values = correct_energy(energy_values, nmpc,
                                   global_min=global_min)

    # Standard formatting values
    label_size = 24
    tick_size = 24
    figure_size=(12,8)
    width = 6
    point_size = 125
    
    fig = plt.figure(figsize=figure_size)
    ax = fig.gca()
    
    marker_plot(ax, prop_values, energy_values, 
                s=point_size, c=motif_colors, facecolors='none',
                label='Initial Pool',
                marker_list=motif_markers)
    
    if len(experimental_parameters)==3:
        experimental_energy = correct_energy([experimental_parameters[1]],
                                             nmpc=nmpc,global_min=global_min)
        ax.scatter(experimental_parameters[2],experimental_energy,
                    s=point_size, c='tab:orange', 
                    label=experimental_parameters[0])
    
    plt.xlabel("$H^{max}_{ab}$ (meV)", fontsize=34)

    ax.set_ylabel("\n"
                  .join(wrap('Relative Energy per Molecule (kJ mol$^{-1}$ molecule$^{-1}$)', 31)),
                  fontsize=32,labelpad=-2)
    
    ax.ticklabel_format(useOffset=False,linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    # Border Thickness
    ax.spines['top'].set_linewidth(width)
    ax.spines['right'].set_linewidth(width)
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['left'].set_linewidth(width)
    
#    ax.set_xlim([0,220])
#    ax.set_ylim([-2.5,100])
    
    if motif == False:
        plt.legend(fontsize=18)
    else:
        marker_list = [x for _,x in marker_dict.items()]
        handle_list = []
        for i,marker in enumerate(marker_list):
            handle_list.append(mlines.Line2D([], [], linewidth=2,
                          markeredgecolor=color_dict[motif_list[i]], 
                          markerfacecolor='none',
                          marker=marker, linestyle='None',
                          markersize=16, label=motif_list[i],
                          ))
        plt.legend(handles=handle_list,
                   fontsize=22)
    
    plt.tight_layout()
    plt.show()
    
    if figname != None:
        fig.savefig(figname)

    return ax
    

    

if __name__ == '__main__':
    struct_dir = '/Users/ibier/Research/Results/Hab_Project/FUQJIK/4_mpc/Property_Based/GAtor_075/motif'
#    struct_dir = '/Users/ibier/Research/Results/Hab_Project/FUQJIK/4_mpc/Energy_Based/cross_025/Hab_Calcs/database_json_files'
#    struct_dir = '/Users/ibier/Research/Results/Hab_Project/FUQJIK/4_mpc/Energy_Based/cross_075/Hab_Calcs/database_json_files'
#    struct_dir = '/Users/ibier/Research/Results/Hab_Project/FUQJIK/8_mpc/Arjuna_RCD_025_testrun/database_json_files'
#    struct_dir = '/Users/ibier/Research/Results/Hab_Project/FUQJIK/4_mpc/original_GA_runs/rcd_025/database_json_files_Hab'
#    hab_key = 'Habmax'
#    plot_GAtor_Hab(struct_dir, 
#                   hab_key, 'energy', 4, motif=True, IP=False)
#    label_kwargs = \
#    {
#        'xlabel': '',
#        'ylabel': 'Habmax'
#    }
#    motif_mean,motif_std = motif_prop_plot(struct_dir,'Habmax', error=False, label_kwargs=label_kwargs)
#    label_kwargs = \
#        {
#            'xlabel': '',
#            'ylabel': "\n".join(wrap('Relative Energy per Molecule (kJ mol$^{-1}$ molecule$^{-1}$)', 31))
#        }
#    motif_prop_plot(struct_dir,'energy', error=False, label_kwargs=label_kwargs)
