

import numpy as np
from ase.data import vdw_radii,atomic_numbers,covalent_radii
from ase.data.colors import jmol_colors

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ibslib.driver import BaseDriver_
from ibslib.descriptor.bond_neighborhood import BondNeighborhood


all_radii = []
for idx,value in enumerate(vdw_radii):
    if np.isnan(value):
        value = covalent_radii[idx]
    all_radii.append(value)
all_radii = np.array(all_radii)


class MoleculeVolumeEstimator(BaseDriver_):
    """
    Class for performing molecular volume estimation using vdW radii
    
    Arguments
    ---------
    tol: float
        Tolerance to converge the estimation.
    iterations: int
        Maximum number of iterations to converge.
    batch: int
        Number of samples to process in MC method at a single time. 
        This is also the number for which convergence is tested. 
    vdW: numpy.array
        Array of vdW volumes index by the atomic number.
    bn: bool
        If the bond neighborhood model should be used. 
        
    """
    def __init__(self, tol=1e-3, iterations=1e8, batch=200000, vdW=all_radii,
                 bn=True, update=True):
        # Change vdW radii
        self.iterations = int(iterations)
        self.batch = int(batch)
        self.vdW = vdW
        self.tol = tol
        self.bn = bn
        self.update = update
        
    
    def calc_struct(self, molecule_struct):
        """
        Calculates the predicted of the molecular volume for a Structure object
        of a molecule. 
        
        """
        self.struct = molecule_struct
        
        if "mc_volume" in self.struct.properties:
            if self.update:
                pass
            else:
                return self.struct.properties["mc_volume"]
        
        volume = self.MC(molecule_struct)
        if self.bn:
            # Adjust volume based on CSD analysis
            volume = self.bond_neighberhood_model(molecule_struct,volume)
            self.struct.properties["pred_volume"] = volume
        else:
            self.struct.properties["mc_volume"] = volume
        return volume
    
    
    def bond_neighberhood_model(self,molecule_struct,MC_volume):
        """
        Applies linear correction to the input volume from CSD analysis.
        """
        ### Model definition
        required_fragments = np.array(
        ['MC_volume', 'HC', 'HOC', 'OOCC', 'HHHCC', 'ON', 'OC', 'CCCC', 'OCCO',
       'HCCC', 'ClC', 'NC', 'CCCN', 'OONC', 'NPP', 'HHNC', 'HNCC', 'HCNN',
       'FC', 'BrC', 'NCN', 'ICCC', 'CCC', 'HHHCN', 'HNCS', 'HCCS', 'HHHCSi',
       'HB', 'NCC', 'CCNN', 'SC', 'HHHCS', 'NCO', 'HNCN', 'ClP', 'IC', 'OCP',
       'NNN', 'CNNN', 'HHHNC', 'HHCCS', 'NCCC', 'HHCC', 'NCCN', 'CNNS', 'SP',
       'SCN', 'HCCNO', 'SCS', 'HHCOO', 'HHCNN', 'OCSi', 'CCCSe', 'CCCTe',
       'ONN', 'HHCCN', 'ONCC', 'OONN', 'HOCC', 'HO', 'HHCCP', 'HCCOO', 'CCCS',
       'CCCSi'],
         dtype=object)
        coef = np.array(
        [1.38732781,  1.09273374,  0.01984596, -5.83779431,  3.03098325,
        -2.44819463, -3.19448159,  0.60461945,  5.70153497, -0.22730391,
         2.68279888,  0.34512353, -0.11118681,  3.00057338,  4.59332838,
        -0.23272109,  2.78659772, -2.87304195,  0.89258333,  2.23787589,
         1.93557232,  5.16266751,  1.84458424,  2.74767878,  5.54271523,
         0.10956046,  4.38654368,  1.03917423,  2.81809621, -2.52236071,
         1.09844288,  0.4967574 ,  2.73880443, -0.49803522,  1.68784377,
        -0.27383809,  3.54043547,  0.63538187, -3.11453747, -5.11258648,
        -1.01933284,  1.5332678 ,  0.32195421,  1.79623041,  1.83284527,
         6.25311953, -4.88660802, -3.14142807, -1.63444334, -7.94330151,
        -2.25539996, -0.41931587, -1.11906979, -2.25206549, -1.55124899,
        -0.4613776 ,  0.5797792 , -0.8123834 ,  2.7653278 , -3.49700741,
         1.37890689, -0.02064372,  1.74777657,  7.82082655,  0])        
        
        # Initialize feature vector for structure
        neighborhood_vector = np.zeros(required_fragments.shape[0]+1)
        # Add MC Volume
        neighborhood_vector[0] = MC_volume
        neighborhood_vector[-1] = 1
        
        # Calc structure fragments and counts
        bn = BondNeighborhood()
        f,c = bn.calc_struct(molecule_struct)
        # Check which fragments in struct are in fragment list and return
        # the index of their location
        f_idx,c_idx = np.nonzero(required_fragments[:,None] == f)

        c = np.array(c)
        f = np.array(f)
        
        # Populate feature vector 
        neighborhood_vector[f_idx] = c[c_idx]
        volume = np.sum(np.dot(neighborhood_vector,coef))
        
        return volume
        
    
    def MC(self, molecule_struct):
        """
        Monte Carlo method for volume estimation using vdW radii
        
        """
        self.struct = molecule_struct
        self.geo = molecule_struct.get_geo_array()
        
        # Get radii array for each element in geometry
        self.radii = np.array([self.vdW[atomic_numbers[ele]] for ele in 
                      molecule_struct.geometry["element"]])[:,None]
        
        # Get simulation sample region
        self._set_region()
        
        # Building iteration counter
        n_batchs = int(self.iterations / self.batch) 
        last_batch = self.iterations - n_batchs*self.batch
        batchs = [self.batch for x in range(n_batchs)]
        # Add a last batch if necessary to reach exactly self.iterations
        if last_batch != 0:
            batchs.append(last_batch)
        
        # Keep track of in an out
        self.vdW_in = 0
        self.total_samples = 0
        self.volume_tracking = []
        self.var_tracking = []
        self.points_inside = []
        self.c = []
        for batch_size in batchs:
            points = self._sample_points(batch_size)
            # Pairwise distance between points and all atoms in geometry
            distances = points - self.geo[:,None]
            distances = np.linalg.norm(distances, axis=-1)
            
            # Find which samples were within a vdW radii from any atom
            vdW_dist = distances - self.radii
            
            self.vdW_dist = vdW_dist
            r_idx,in_idx = np.where(vdW_dist <= 0)
            self.points_inside.append(points[in_idx])
            self.c.append(self._define_colors(self.struct.geometry["element"][r_idx]))
            
            vdW_in = vdW_dist <= 0
            vdW_in = np.sum(vdW_in, axis=0)
            vdW_in = np.where(vdW_in >= 1)[0]
            vdW_in = vdW_in.shape[0]
            self.vdW_in += vdW_in
            self.total_samples += batch_size
            self.volume_tracking.append(
                    (self.vdW_in / self.total_samples) * self.region_volume)
            self.var_tracking.append(self.volume_tracking[-1] * \
                            (self.region_volume - self.volume_tracking[-1]) \
                            / self.total_samples)
            
            # Check for convergence
            if len(self.volume_tracking) > 2:
                converged = self._check_convergence()
                if converged == False:
                    pass
                else:
                    break
        
        self.molecule_volume = (self.vdW_in / self.total_samples) * self.region_volume
        
        # Saving points and colors for plotting 
        self.points_inside = np.vstack(self.points_inside).reshape(-1,3)
        self.c = np.vstack(self.c)
        
        return self.molecule_volume

    
    def _set_region(self):
        """ 
        Sets the sample region for the simulation
        
        """
        # Minimum and Max in each direction bias of 5
        min_region = np.min(self.geo,axis=0) - 5
        max_region = np.max(self.geo,axis=0) + 5
        self.region = np.array(list(zip(min_region,max_region)))
        
        self.region_volume = (self.region[0][1] - self.region[0][0])* \
                             (self.region[1][1] - self.region[1][0])* \
                             (self.region[2][1] - self.region[2][0])
        
    
    def _sample_points(self, batch_size):
        """
        Returns a sampling of the MC region for the given batch_size
        """
        x = np.random.uniform(self.region[0][0], self.region[0][1],
                              size=(batch_size,1))
        y = np.random.uniform(self.region[1][0], self.region[1][1],
                              size=(batch_size,1))
        z = np.random.uniform(self.region[2][0], self.region[2][1],
                              size=(batch_size,1))
        return np.concatenate((x,y,z),axis=1)
    
    
    def _check_convergence(self):
        """
        Checks is calculation has converged the volume of the system
        """
        # Difference of last two volume predictions
        err = abs(self.volume_tracking[-1] - self.volume_tracking[-2])
        if err > self.tol:
            return False
        else:
            return True
        

    def plot_MC(self, num_points=-1, elev=0, azim=0, figname=""):
        """
        Plot the points which were found to be inside the molecule. Useful
        for visualizing the volume which is found by MC method.
        
        Arguments
        ---------
        num_points: int
            Number of points to use to make plot. If -1, then use all points.
        """
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.w_xaxis.set_pane_color((0., 0., 0., 1.0))
        ax.w_yaxis.set_pane_color((0., 0., 0., 1.0))
        ax.w_zaxis.set_pane_color((0., 0., 0., 1.0))
        if num_points < 0:
            ax.scatter(self.points_inside[:,0],
                    self.points_inside[:,1],
                    self.points_inside[:,2], alpha=0.1,
                    c = self.c)
        else:
            num_points = int(num_points)
            ax.scatter(self.points_inside[:num_points, 0],
                        self.points_inside[:num_points, 1],
                        self.points_inside[:num_points, 2], alpha=0.1,
                        c = self.c[:num_points])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
#        for ii in xrange(0,360,10):
        ax.view_init(elev=elev, azim=azim) 
        plt.tight_layout()
        if len(figname) > 0:
            fig.savefig(figname, bbox_inches='tight',
                        transparent=True, pad_inches=0)
        
    
    def _define_colors(self,elements):
        idx = [atomic_numbers[x] for x in elements]
        return jmol_colors[idx]
    
    
    
if __name__ == "__main__":
    pass
#    from ibslib.io import read,write
#    import matplotlib.pyplot as plt
#    struct_dict = read("Single_Molecules/json")
#    
#    x = []
#    y = []
#    mve = MoleculeVolumeEstimator()
#    for struct_id,struct in struct_dict.items():
#        exp_volume = struct.get_property("molecule_volume")
#        pred_volume = mve.calc(struct)
#        print("{} {} {}".format(exp_volume, pred_volume, exp_volume-pred_volume))
#        x.append(exp_volume)
#        y.append(pred_volume)
#    
#    plt.scatter(x,y)
