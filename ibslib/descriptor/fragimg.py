
import numpy as np
import networkx as nx

from ase.data import vdw_radii,atomic_numbers
from ase.data.colors import jmol_colors

from ibslib import Structure
from ibslib.io import read,write
from ibslib.analysis.pltlib import format_ticks
from ibslib.descriptor.bond_neighborhood import BondNeighborhood,\
                                              construct_bond_neighborhood_model
                                              
import matplotlib.pyplot as plt



class FragmentImage():
    def __init__(self, BondNeighborhood_object=BondNeighborhood()):
        self.bn = BondNeighborhood_object
    
    
    def calc(self, struct_dict, name_list=[], figname="",
             separating_lines=False):
        """
        Plots all structures in the struct_dict. Will first have to pass over
        through all of the structures and find the one with the largest
        number of unique fragments. This wil be done in order to align the 
        subplots correctly. There's also the issue of aligning all fragments
        correctly. Maybe it's easier for all boxes to be the same and only 
        rescale if the user says to.
        
        
        For scaling the size of the name subplot. If there's molecule name
        that is long, add a line break using `\n` to the molecule name. 
            
        """
        keys = [x for x in struct_dict.keys()]
        if len(name_list) == 0:
            name_list = keys
        
        fragment_struct_list = []
        f_list = []
        c_list = []
        num_fragments_list = []
        for struct_id,struct in struct_dict.items():
            temp_fragment_struct_list,f,c = self._find_fragments(struct)
            fragment_struct_list.append(temp_fragment_struct_list)
            f_list.append(f)
            c_list.append(c)
            num_fragments_list.append(len(f))
        
        max_fragments = np.max(num_fragments_list)
        
        fig,ax_list = self._create_figure([], num_columns=max_fragments+1,
                                          num_rows=len(struct_dict),
                                          scale_widths=False,
                                          name="use name")
        
        for i,row in enumerate(ax_list):
            key = keys[i]
            name = name_list[i]
            struct = struct_dict[key]
            self.fragments(struct, name=name,
                           figure=fig, ax_list=row,
                           separating_lines=separating_lines)
            
        if len(figname) > 0:
            fig.savefig(figname)
        
    
    def fragments(self, struct, figname="", name="",
                  figure=None, ax_list=[], scale_widths=False,
                  separating_lines=False):
        """
        Plots all fragments of a structure together
        
        Arguments
        ---------
        struct: Structure
            Structure object to make the fragment plot for.
        figname: str 
            Path to the location the figure should be saved
        name: str
            Name of the molecule to be plotted on the image. Default is a null
            string which implies that no name will be plotted.
        figure: matplotlib.pyplot.figure
            Figure object. Can be provided to edit an existing figure.
        ax_list: list
            If Figure object is provided, this ax_list is used for the 
            current structure.
        separating_lines: bool
            Whether to draw lines around the entries from each structure. 
        
        """
        fragment_struct_list,f,c = self._find_fragments(struct)
        if figure == None:
            fig,ax_list = self._create_figure(fragment_struct_list,
                                              name=name,
                                              scale_widths=scale_widths)
        else:
            fig = figure
        
        for i,f_struct in enumerate(fragment_struct_list):
            if len(name) > 0:
                self.struct(f_struct, ax=ax_list[i+1])
            else:
                self.struct(f_struct, ax=ax_list[i])
            
        for ax in ax_list:
            ax.set_ylim([-1.75,1.75])
            ax.set_xlim([-1,1])
            if separating_lines:
                ax.set_axis_on()
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_axis_off()
        
        for i,fragment_struct in enumerate(fragment_struct_list):
            fragment_idx = i
            if len(name) > 0:
                i += 1
                if i == len(ax_list):
                    break
            ax = ax_list[i]
            
            fragment_name = f[fragment_idx]
            counts = c[fragment_idx]
            ax.text(0,1.25,"{}".format(fragment_name),
                    ha="center",
                    fontweight="bold",
                    fontsize=16)
            ax.text(0,-1.5,"{}".format(counts),
                    ha="center",
                    fontweight="bold",
                    fontsize=16)
        
        if len(name) > 0:
            ax_list[0].text(0,0, "{}".format(name),
                   ha="center",
                   fontweight="bold",
                   fontsize=22)
            
        # Do not save yet if figure was passed into function
        if figure == None:
            plt.tight_layout()
            
            if len(figname) > 0:
                fig.savefig(figname)
    
    
    def _create_figure(self, fragment_struct_list, scale_widths=False, name="",
                       num_columns=-1, num_rows=1):
        """
        Creates figure space for the structure fragments to be plotted. 
        
        """
        if num_columns < 0:
            num_columns = len(fragment_struct_list)
            if len(name) > 0:
                num_columns += 1
            
        gridspec_kw = \
            {
              "wspace": 0,
              "hspace": 0
            }
        
        if scale_widths:
            if len(name) > 0:
                gridspec_kw["width_ratios"] = [4]
                gridspec_kw["width_ratios"] += [struct.geometry.shape[0] 
                                                for struct in 
                                                fragment_struct_list]
            else:
                gridspec_kw["width_ratios"] = [struct.geometry.shape[0] 
                                                for struct in 
                                                fragment_struct_list]
            
            
        figsize=(2*num_columns, 2*num_rows)
        
        fig,ax_list = plt.subplots(nrows=num_rows, ncols=num_columns,
                                   gridspec_kw=gridspec_kw,
                                   figsize=figsize, 
                                   sharey="col",
                                   sharex="row")
        
        return fig,ax_list
        
        
    def struct(self,struct,ax=None):
        """
        Plots the structure. Could be plotting an entire structure or a
        fragment structure. The function is agnostic to the size of the 
        structure. 
        
        """
        g = self.bn._build_graph(struct)
        if struct.geometry.shape[0] == 2:
            pos = nx.bipartite_layout(g, nodes=[x for x in g.nodes])
        else:
            pos = nx.spring_layout(g)
        
        # Scale such that all points are within the image
        for i,array in pos.items():
            if (1 - np.max(np.abs(array))) < 0.25:
                pos[i] = array*0.75
        
        color_list = self._get_colors(struct.geometry["element"])
        labels = {}
        for i,ele in enumerate(struct.geometry["element"]):
            labels[i] = ele
        if ax == None:
            nx.draw(g,pos=pos,labels=labels,with_labels=True,
                    node_color=color_list,
                    font_weight="bold",
                    font_size=14)
        else:
            nx.draw(g,pos=pos,labels=labels,with_labels=True,
                    node_color=color_list,
                    font_weight="bold",
                    font_size=14,
                    ax=ax)
            
    
    def _find_fragments(self, struct):
        """
        Finds and returns fragments structures for each unique fragment in the
        Structure object.
        
        """
        fragments,counts = self.bn.calc(struct)
        g = self.bn._build_graph(struct)
        n = self.bn._calc_neighbors(g)
        sorted_neighborhood = self.bn._sort(g,n)
        neighborhood_names = self.bn._construct_fragment_list(sorted_neighborhood)
        geo = struct.get_geo_array()
        ele = struct.geometry["element"]
        
        fragment_struct_list = []
        for fragment in fragments:
            fragment_idx = neighborhood_names.index(fragment)
            atom_idx = sorted_neighborhood[fragment_idx][0]
            
            temp_geo = geo[atom_idx,:]
            temp_ele = ele[atom_idx]
            
            struct = Structure()
            struct.from_geo_array(temp_geo, temp_ele)
            
            fragment_struct_list.append(struct)
        
        return fragment_struct_list,fragments,counts
        
    
    def _get_colors(self, elements):
        idx = [atomic_numbers[x] for x in elements]
        return jmol_colors[idx]
    


if __name__ == "__main__":
    pass
#    bn = BondNeighborhood()
#    mfi = MoleculeFragmentImage(bn)
#    mfi.calc(test_dict)
