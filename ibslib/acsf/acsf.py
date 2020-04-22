# -*- coding: utf-8 -*-


import torch
import numpy as np
from scipy.spatial.distance import cdist

from ibslib import Structure
    

class BaseACSF():
    """
    Uses struct_dict in order to identify all unique elements in the dataset
    to construct the ACSF. 
    
    BaseACSF serves as a model for any ACSF implementations. 
    """
    def __init__(self,struct_dict,cutoff,force=False,unique_ele=[]):
        """
        Initialization of the ACSF based on the structures present in the 
        struct_dict.
        
        Arguments
        ---------
        struct_dict: dict,ibslib.StructDict
            Structure dictionary
        cutoff: float
            Cutoff radius for all calculations. 
        force: bool
            If force training will be used. If force is True, then the 
            atom centered symmetry functions need to be constructure w.r.t.
            the direction of the system. 
        unique_ele: list
            List of the unique elements for all the structures in the 
            struct_dict. If an empty list is provided, then the unique 
            elements will be automatically determined. This, however, is not
            preferable if the dataset is very large.
            
        """
        self.struct_dict = struct_dict
        self.cutoff = cutoff
        self.unique_ele = np.array(unique_ele)
        self._unique_ele()
        self._initialize()
        pass
    
    
    def _unique_ele(self):
        if len(self.unique_ele) > 0:
            return
        unique_ele = np.array([])
        for struct_id,struct in self.struct_dict.items():
            ele = struct.geometry["element"]
            unique = np.unique(ele)
            unique_ele = np.concatenate([unique_ele, unique])
            unique_ele = np.unique(unique_ele)
        self.unique_ele = unique_ele
        
    
    def _initialize(self):
        pass
        
    
    def forward(struct):
        """
        Compute the ACSF for the entire structure by looping over atoms.
        """
        for atom in struct.geometry:
            pass


class RSF(BaseACSF):
    """
    Class for constructing two-body radial distribution functions.
    """
    def __init__(self, 
                 struct_dict={},
                 cutoff=6,
                 unique_ele=[], 
                 force=False,
                 n_D_inter=50, 
                 init_scheme="shifted",
                 eta_range=[0.05,0.5], 
                 Rs_range=[0.1,12],
                 del_neighbor=True):
        """ 
        Radial symmetry funciton for both energies and forces. 

        Arguments
        ---------
        struct_dict: StructDict
            Structure dictionary is optional to be used if you would like the 
            unique elements to be automatically determined for you. 
            Otherwise, you must pass in the unique elements yourself.
        n_D_inter: int
            Number of symmetry functions to use for each interaction.
        init_scheme: str 
            One of "centered","shifted","random". Defines the initialization
              scheme which is used for the eta and Rs parameters. 
        cutoff: float
            Cutoff distance used in cutoff function.
        eta_range: list [start, end]
            Used in random initialization
        Rs_range: list [start, end]
            Used in random initialization
        del_neighbor: bool
            If True, the neighborlist and neighborele are deleted from the 
            structure object This is important if working with a very large 
            dataset for two reasons. 
            1) Want to keep the in-memory representation as small as possible.
               After the ACSF descriptors are done being computed, don't
               need the neighborlists anymore. 
            2) Want to keep the amount written to drive as small as possible. 
               Don't want huge datasets, saving all the neighborhood information 
               when all that's required in the ACSF vector. 
        
        """
        self.struct_dict = struct_dict
        self.cutoff = cutoff
        self.unique_ele = np.array(unique_ele)
        self.force = force
                
        # Initialize RDF specific parameters
        self.n_D_inter = n_D_inter
        self.init_scheme = init_scheme
        self.cutoff = cutoff
        self.eta_range = eta_range
        self.Rs_range = Rs_range
        self.del_neighbor = del_neighbor
        
        self.avail_init_schemes = ["centered","shifted","random"]
        if init_scheme not in self.avail_init_schemes:
            raise Exception("Initializaiton scheme for representation layer "+
            "is not available. Available schemes are {}, however input was {}."
            .format(self.avail_init_schemes, init_scheme))
            
            
        super(RSF, self).__init__(struct_dict,cutoff,unique_ele=self.unique_ele, 
             force=force)    
        
    
    def _initialize(self):
        # Performing parameter initialization
        eval("self.init_{}()".format(self.init_scheme))
        self._lookup_dict()
        self.eta = self.eta.numpy()
        self.Rs = self.Rs.numpy()
    
    
    def init_centered(self):
        """
        Initialization scheme used for centered radial symmetry funciton as 
          defined in Gastegger et al. 2018. 
        """
        if self.n_D_inter > 1:
            delta_R = (self.cutoff - 1.5) / (self.n_D_inter - 1)
            r = torch.arange(start=1.0, end=(self.cutoff-0.25), step=delta_R)
        else:
            delta_R = self.cutoff/2
            r = torch.tensor([self.cutoff / 2])

        eta = 1 / (2*r.pow(2))
        self.eta = eta.clone().detach().requires_grad_(False)
        
        Rs = torch.zeros(eta.size())
        self.Rs = Rs.clone().detach().requires_grad_(False)

    
    def init_shifted(self):
        """
        Initialization scheme used for shifted radial symmetry funciton as 
          defined in Gastegger et al. 2018. 
        """
        if self.n_D_inter > 1:
            delta_R = (self.cutoff-0.5) / (self.n_D_inter-1)
#            r = torch.arange(start=self.Rs_range[0], 
#                             end=(self.cutoff-0.5+delta_R), step=delta_R)
            r = np.linspace(start=self.Rs_range[0],
                            stop=self.Rs_range[1],
                            num=self.n_D_inter,
                            endpoint=True)
            r = torch.tensor(r)
        else:
            delta_R = self.cutoff/2
            r = torch.tensor([self.cutoff / 2])

        self.Rs = r.clone().detach().requires_grad_(False)
        eta = torch.zeros(self.Rs.size()) + 1 / (2*delta_R**2)
        self.eta = eta.clone().detach().requires_grad_(False)

    
    def init_random(self):
        """
        Random initialization of eta and Rs
        """
        self.eta = torch.FloatTensor(self.n_D_inter,
                        requires_grad=False).uniform_(
                        self.eta_range[0],self.eta_range[1])
        self.Rs = torch.FloatTensor(self.n_D_inter,
                            requires_grad=False).uniform_(
                                    self.Rs_range[0],self.Rs_range[1])


    def cutoff_fn(self,R):
        """
        Computes the cutoff function for a batch of examples. Due to the 
        pre-construction of the neighborlist, all atoms in R are assumed to
        be within the cutoff radius.

        cutoff = {
                    (1/2)[cos(pi*rij / cutoff) + 1] if rij <= rc
                    0
                 }

        Returns result from cutoff fn. This will be a torch tensor 
          with dimension [n_samples, n_atoms_per_system, R]
        """
        mask = np.where(R > self.cutoff)[0]
        result = (1/2)*(np.cos(np.pi * R / self.cutoff) + 1)
        result[mask] = 0
        return result
    
    
    def forward(self,struct):
        """
        Implements forward calculation for atomic type radial symmetry 
        function.
        
        I think that force evaluation only changes the calculations which 
        go on here. Also, need to implement element specific stuff initialization
        and calculation.
        """
        struct.properties["acsf"] = []
        geo_array = struct.get_geo_array()
        for i,pos in enumerate(geo_array):
            atom_result = self._empty_vector()
            atom_type = struct.geometry["element"][i]
            
            # Get neighborlist and neighborele for atom i
            neighborlist = np.array(struct.properties["neighborlist"][i])
            neighborele = struct.properties["neighborele"][i]
            
            # Now, calculate all distances for neighborlist
            dist = cdist(pos[None,:], neighborlist)
            
            # Looping through all interaction types
            for key,idx in neighborele.items():
                name = atom_type+key
                
                # Collect the distances for on interaction type
                dist_ele = dist[0,idx]
                
                ##### Compute Symmetry Function
                
                # Add direction components if force training
                if self.force == True:
                    directions = ["x","y","z"]
                    for j,direction in enumerate(directions):
                        name_dist = name+direction
                        
                        # Get length is the direction of x,y, or z
                        direction_dist = neighborlist[idx,j] - pos[j]
                        # Project norm distance along direction to get cosine
                        # of angle between. This is used to calculate density
                        # of descriptor along x,y,z directions.
                        direction_fraction = direction_dist / dist_ele
                        
                        G = self._compute(dist_ele, direction_fraction)
                        atom_result[self.lookup_dict[name_dist]] = G
                else:
                    G = self._compute(dist_ele, [])
                    atom_result[self.lookup_dict[name]] = G
            
            struct.properties["acsf"].append(atom_result.tolist())
        
        if self.del_neighbor == True:
            del(struct.properties["neighborlist"])
            del(struct.properties["neighborele"])
    
    
    def _compute(self, dist, direction_fraction=[]):
        """
        Computes the actual symmetry function from distances.
        
        """
        G = np.exp(-self.eta[:,None] * np.square(dist - self.Rs[:,None]))
        
        # Multiply by direction fraction for force implementation
        if len(direction_fraction) > 0:
            G = direction_fraction * G
            
        G = G * self.cutoff_fn(dist)
        G = np.sum(G, axis=-1)
        return G
    
    
    def _empty_vector(self):
        """
        Based on the settings of the RDF, constructs an empty vector of the 
        correct size for a single atom.
        
        """
        dim = 0
        # All pairs
        dim = len(self.unique_ele) * len(self.unique_ele)
        # With dimension of interaction
        dim = dim*self.n_D_inter
        
        if self.force == True:
            dim = dim*3
        
        return np.zeros([dim,])
    
    
    def _lookup_dict(self):
        """
        Constructs a dictionary used for indexing into the final atomic feature
        vector.
        
        """
        self.lookup_dict = {}
        keys = np.char.add(self.unique_ele[:,None], self.unique_ele)
        keys = keys.ravel()
        
        if self.force:
            keys = np.char.add(keys, np.array(["x","y","z"])[:,None])
            keys = keys.ravel()
        
        for i,key in enumerate(keys):
            idx = np.arange(0,self.n_D_inter)
            idx = idx + self.n_D_inter*i
            self.lookup_dict[key] = idx
    
    
    def plot_rsf(self, figsize=(6,5), figname="", legend=True):
        """
        Plot the radial symmetry functions that the user has defined in the 
        initialization of the RDF class.
        
        """
        import matplotlib.pyplot as plt
        from matplotlib import colors 
        from matplotlib.pyplot import cm
        
        x = np.arange(0,self.cutoff, 0.001)
        cutoff = self.cutoff_fn(x)
        y_array = cutoff*np.exp(-self.eta[:,None]*np.square(x - self.Rs[:,None]))
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        #### Mapping colors for every symmetry function
        ints = np.arange(0,y_array.shape[0]+1,1)
        norm = colors.Normalize(vmin=ints[0], vmax=ints[-1], clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
        color_list = []
        for i in ints:
            color_list.append(mapper.to_rgba(i))
        
        
        for i,entry in enumerate(y_array):
            ax.plot(x,entry,label="$\eta$={:.2f}, $R_s$={:.2f}"
                       .format(self.eta[i], self.Rs[i]),
                       c=color_list[i])
        
        ax.plot(x,cutoff,label="Cutoff Function")
        
        if legend:
            ax.legend()
        
        if len(figname) > 0:
            fig.savefig(figname)
            
        return ax



if __name__ == "__main__":
    pass
