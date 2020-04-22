# -*- coding: utf-8 -*-

import torch

from ibslib import Structure,StructDict
from ibslib.io import read,write
from ibslib.descriptor.R_descriptor import AtomicPairDistance
from ibslib.descriptor.rdf import ACSF


class DiversityAnalysis():
    def __init__(self,device=torch.device("cpu"), 
                 acsf_kwargs = 
                     {
                        "n_D_inter": 12, 
                        "init_scheme": "shifted", 
                        "cutoff": 12,
                        "eta_range": [0.05,0.5], 
                        "Rs_range": [0.1,12], 
                        "learn_rep": False
                     }
                ):
        """
        Initialize calculator
        """
        self.apd = AtomicPairDistance(p_type="element")
        self.acsf = ACSF(device, **acsf_kwargs)
    
    
    def calc(self, struct_obj, prop=''):
        """
        Calculate acsf for Structure or StructDict. Please note, for now these return 
        different types.
        """
        obj_type = type(struct_obj)
        if obj_type == dict or obj_type == StructDict:
            return self._calculate_acsf_dict(struct_obj,prop=prop)
        elif obj_type == Structure:
            return self._calculate_acsf_struct(struct_obj,prop=prop)
        
    
    def calc_struct(self,struct):
        all_rdf,_ = self._calculate_acsf_struct(struct)
        struct.properties = {"RSF": all_rdf.detach().numpy().tolist()}
        # Store for writing 
        self.struct = struct
    
    
    def write(self, output_dir, file_format="json", overwrite=True):
        if len(file_format) == 0:
            file_format = "json"
        temp_dict = {self.struct.struct_id: self.struct}
        write(output_dir,temp_dict,file_format=file_format, 
              overwrite=overwrite)
        
    
    def _calculate_acsf_dict(self,struct_dict,prop=''):
        all_rdf = torch.tensor([])[:,None]
        prop_values = []
        for struct_id,struct in struct_dict.items():
            if len(prop) > 0:
                prop_values.append(struct.get_property(prop))
            
            self.apd.calculate(struct)
            ele_R = struct.get_property("R")
            rdf = torch.tensor([])
            for ele,inter_dict in ele_R.items():
                for inter_name,inter in inter_dict.items():
                    temp = self.acsf.eval_struct(inter[None,:])
                    rdf = torch.cat((rdf,temp.view(-1)))
            if all_rdf.nelement() == 0:
                all_rdf = rdf[None,:]
            else:
                all_rdf = torch.cat((all_rdf, rdf[None,:]), dim=0)
        
        return all_rdf, prop_values
    

    def _calculate_acsf_struct(self,struct,prop=''):
        if len(prop) > 0:
            prop_value = struct.get_property(prop)
        else:
            prop_value = 0
        
        self.apd.calculate(struct)
        ele_R = struct.get_property("R")
        rdf = torch.tensor([])
        for ele,inter_dict in ele_R.items():
            for inter_name,inter in inter_dict.items():
                temp = self.acsf.eval_struct(inter[None,:])
                rdf = torch.cat((rdf,temp.view(-1)))
        
        return rdf,prop_value
    
    
    def reduce_dim(self, all_rdf, dim_reduction_obj):
        """
        Reduce dimension for all_rdf tensor using dim_reduction_obj
        """
        return dim_reduction_obj.fit_transform(all_rdf.detach().numpy())
    
    
    def calc_diff(self, all_rdf):
        """
        Calculate Euclidean distance between rdf 
        """
        difference = all_rdf[:,None] - all_rdf
        difference = difference.pow(2)
        difference = torch.sum(difference, dim=-1)
        return difference
        


if __name__ == "__main__":
    pass
#    original_genarris = "/Users/ibier/Research/Results/FUQJIK/8_mpc/INITIAL_POOL/final_initial_pool_cleaned"
#    new_genarris = "/Users/ibier/Research/Results/FUQJIK/8_mpc/Genarris/Building_Final_IP/Arjuna_initial_pool"
#    
#    struct_dict_old = read(original_genarris)
#    struct_dict_new = read(new_genarris)
#    
#    da = DiversityAnalysis()
#    old_rdf,energy_old = da.calculate_acsf(struct_dict_old, "energy_tier1")
#    new_rdf,energy_new = da.calculate_acsf(struct_dict_new, "energy_tier1_relaxed")
#    
#    all_rdf = torch.cat((old_rdf,new_rdf))
#    all_energy = energy_old
#    all_energy += energy_new
#    
#    from sklearn.decomposition import PCA
#    pca = PCA(n_components=2)
#    results = da.reduce_dim(all_rdf,pca)
#    
#    colors = []
#    markers = []
#    for _ in struct_dict_old:
#        colors.append("tab:blue")
#        markers.append('o')
#    for _ in struct_dict_new:
#        colors.append("tab:orange")
#        markers.append('^')
#        
#    idx_old = np.arange(0,len(struct_dict_old))
#    idx_new = np.arange(0,len(struct_dict_new)) + len(struct_dict_old)
#    
#    import matplotlib.pyplot as plt
#    from matplotlib.pyplot import cm
#    fig = plt.figure(figsize=(12,8))
#    plt.scatter(results[idx_old,0], results[idx_old,1], edgecolor="tab:blue",
#                marker='o', s=100, facecolors="none")
#    plt.scatter(results[idx_new,0], results[idx_new,1], edgecolor="tab:orange",
#                marker='*', s=100, facecolors="none")
#    plt.show()
#    
#    fig = plt.figure(figsize=(12,8))
#    cmap = cm.hot
#    a = plt.scatter(results[idx_old,0], results[idx_old,1], c=energy_old[0:46], cmap=cmap,
#                marker='o', s=100, lw=2)
#    a.set_facecolor('none') 
#    a = plt.scatter(results[idx_new,0], results[idx_new,1], c=energy_new, cmap=cmap,
#                marker='*', s=100, lw=2)
#    a.set_facecolor('none') 
#    plt.colorbar()
#    
#    fig.savefig("IP_Energy_Plot.pdf")
#    
#    
##    from ibslib.motif import eval_motif
##    
##    motif_old = eval_motif(struct_dict_old)
##    motif_new = eval_motif(struct_dict_new)
##    motif_all = motif_old
##    motif_all += motif_new
#    
#    from ibslib.analysis.plotting import marker_plot,get_marker_dict,get_color_iter
#    from ibslib.motif import implemented_motifs
#    
#    from matplotlib.lines import Line2D
#    import matplotlib.lines as mlines
#    
#    motif_list = implemented_motifs()
#    marker_dict = get_marker_dict(motif_list)
#    markers = []  
#    colors = []
#    
#    color_list = get_color_iter()
#    color_dict = {}
#    for i,motif in enumerate(motif_list):
#        color_dict[motif] = color_list[i]
#        
#    for motif in motif_old:
#        markers.append(marker_dict[motif])  
#        colors.append(color_dict[motif])
#    
#    fig = plt.figure(figsize=(12,8))
#    ax = fig.add_subplot(111)
#    marker_plot(ax, results[:,0], results[:,1], s=100, marker_list=markers,
#                c=colors)
#    
#    marker_list = [x for _,x in marker_dict.items()]
#    handle_list = []
#    for i,marker in enumerate(marker_list):
#            handle_list.append(mlines.Line2D([], [], linewidth=2,
#                          markeredgecolor=color_dict[motif_list[i]], 
#                          markerfacecolor='none',
#                          marker=marker, linestyle='None',
#                          markersize=16, label=motif_list[i],
#                          ))
#    plt.legend(handles=handle_list,
#               fontsize=22)
#    
#    fig.savefig("IP_Motif_Plot.pdf")
        
    
    
        
    
    
    
