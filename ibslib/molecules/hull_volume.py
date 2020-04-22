# -*- coding: utf-8 -*-


"""
Compute the molecule accessible surface area and the molecule 
un-accessible volume. 
    1. Compute the points along the surface that are accessible. 
        - This can be done by constructing a lebedev quadrature grid and then 
          using scipy ConvexHull. 
    2. Use this information to integrate the volume of the system such that 
        volumes inside the molecule are not-accessible and count towards 
        the molecule un-accessible. 
        - Once SCIPY convex hull is constructed, I can easily compute the 
          volume! 
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist,pdist
import spaudiopy as spa

from ase.data import vdw_radii,atomic_numbers,covalent_radii

from ibslib import Structure
from ibslib.driver import BaseDriver_


all_radii = []
for idx,value in enumerate(vdw_radii):
    if np.isnan(value):
        value = covalent_radii[idx]
    all_radii.append(value)
all_radii = np.array(all_radii)


class HullVolume(BaseDriver_):
    
    def __init__(self, lebedev_n=10, vdw=all_radii,
                 update=True):
        self.lebedev_n = lebedev_n
        self.vdw = vdw
        self.update = update
        self.struct = None
        self.vdw_neigh = []
        
        ## For visualization
        self.elements_for_visualization = []
    
    
    def calc_struct(self, struct):
        self.elements_for_visualization = []
        self.struct = struct
        
        if "hull_volume" in self.struct.properties:
            if self.update:
                pass
            else:
                return
        
        self.vdw_neigh = self._get_vdw_neighbors(self.struct)
        self.grid = self._generate_grid(self.struct)
        self.hull = ConvexHull(self.grid)
        self.struct.properties["hull_volume"] = self.hull.volume
        self.struct.properties["hull_area"] = self.hull.area
        
    
    def _generate_grid(self, struct=None):
        if struct == None:
            struct = Structure.from_geo(np.array([[0,0,0]]), np.array(["H"]))
        
        
        geo = struct.get_geo_array()
        ele = struct.geometry["element"]
        
        points_per_atom,_,_ = spa.grids.lebedev(n=self.lebedev_n)
        points_per_atom = points_per_atom.shape[0]
        integration_grid = np.zeros((points_per_atom*geo.shape[0],3))
        for idx,pos in enumerate(geo):
            radius = self.vdw[atomic_numbers[ele[idx]]]
            
            start_row = idx*points_per_atom
            end_row = (idx+1)*points_per_atom
            integration_grid[start_row:end_row,:] = self.atom_grid(pos,radius)
            
            for i in range(end_row-start_row):
                self.elements_for_visualization.append(ele[idx])
            
        return integration_grid
    
    
    def _surface_grid(self, struct=None, neighbor_list=None):
        """
        Takes the Lebedev grid from all atoms and keeps only the points on the
        surface. 
        
        """
        if struct == None:
            struct = self.struct
        if neighbor_list == None:
            if len(self.vdw_neigh) == 0:
                neighbor_list = self._get_vdw_neighbors(struct)
            else:
                neighbor_list = self.vdw_neigh
        
        geo = struct.get_geo_array()
        ele = struct.geometry["element"]
        
        self.elements_for_visualization = []
        self.surface_triangles = []
        current_grid_idx = 0
        integration_grid = []
        volume = 0
        for idx,pos in enumerate(geo):
            radius = self.vdw[atomic_numbers[ele[idx]]]
            neigh_idx = neighbor_list[idx]
            neigh_coord = geo[neigh_idx]
            neigh_vdw = [self.vdw[atomic_numbers[x]] for x in 
                         ele[neigh_idx]]
            
            azi, colat, weights = spa.grids.lebedev(n=self.lebedev_n)
            coords = spa.utils.sph2cart(azi, colat)
            coords = np.hstack([coords[0][:,None], 
                                coords[1][:,None], 
                                coords[2][:,None]])
            
            ## Modify coords for unit sphere for position and radius
            coords *= radius
            coords += pos
            
            atom_grid = coords
            
            result = cdist(atom_grid, neigh_coord)
            
            bool_result = result < neigh_vdw
            mask_idx = np.where(bool_result == True)[0]
            mask_idx = np.unique(mask_idx)
            
            mask = np.ones(result.shape[0], bool)
            mask[mask_idx] = 0
            keep = atom_grid[mask,:]
            
            integration_grid.append(keep)
            self.elements_for_visualization.append(
                    np.repeat(ele[idx], keep.shape[0]))
            
            ## Need to triangulate in 2D to get correct triangulation
            ## For this reason, need to go back to azi and colat values
#            tri_azi = azi[mask]
#            tri_colat = colat[mask]
#            tri_obj = Triangulation(tri_azi, tri_colat)
#            tri_idx = tri_obj.get_masked_triangles() + current_grid_idx
#            self.surface_triangles.append(tri_idx)
            
            ## Update grid idx for surface triangle idx
            current_grid_idx += keep.shape[0]
            
            ## Add up volume (but this is incorrect due to wrong weights)
            temp_volume = np.sum(weights[mask]) / 3 * radius*radius*radius
            volume += temp_volume
        
        
        ## Need to connect neighboring triagles to finish the triangulization
        ## of the surface grids
        
        
            
        integration_grid = np.vstack(integration_grid)
        self.elements_for_visualization = \
                 np.hstack(self.elements_for_visualization)
#        self.surface_triangles = np.vstack(self.surface_triangles)
                 
        return integration_grid


    def _get_vdw_neighbors(self, struct=None):
        
        if struct == None:
            struct = self.struct
        
        geo = struct.get_geo_array()
        ele = struct.geometry["element"]
        
        vdw_list = [self.vdw[atomic_numbers[x]] for x in ele]
        
        neighbor_list = [[] for x in range(geo.shape[0])]
        for idx,coord in enumerate(geo):

            dist = cdist(coord[None,:], geo)
            dist = dist.ravel()
            temp_vdw = vdw_list[idx]
            temp_neigh = np.where(dist < temp_vdw)[0]
            not_self_idx = np.where(temp_neigh != idx)[0]
            
            temp_neigh = temp_neigh[not_self_idx]
            
            neighbor_list[idx] = temp_neigh
        
        return neighbor_list


    def atom_grid(self, pos, radius):
        """
        Generate Lebedev grid for a position and radius. 
        
        """
        if type(pos) != np.array:
            pos = np.array(pos)
        
        azi, colat, weights = spa.grids.lebedev(n=self.lebedev_n)
        coords = spa.utils.sph2cart(azi, colat)
        coords = np.hstack([coords[0][:,None], 
                            coords[1][:,None], 
                            coords[2][:,None]])
        
        ## Modify coords for unit sphere for position and radius
        coords *= radius
        coords += pos
        
        return coords
    
    
    def visualize_grid(self, grid):
        return Structure.from_geo(grid, self.elements_for_visualization)
    
    
    def visualize_hull(self, hull):
        vert_idx = m.hull.vertices
        vert = m.hull.points[vert_idx]
        ele = np.array(self.elements_for_visualization)
        ele = ele[vert_idx]
        return Structure.from_geo(vert, ele)
    
    
    def grid_to_volume(self, grid, spacing=1):
        
        x_min = np.min(grid[:,0])
        x_max = np.max(grid[:,0])
        y_min = np.min(grid[:,1])
        y_max = np.max(grid[:,1])
        z_min = np.min(grid[:,2])
        z_max = np.max(grid[:,2])
        
        x_vals = np.arange(x_min-spacing, x_max+spacing, spacing)
        y_vals = np.arange(y_min-spacing, y_max+spacing, spacing)
        z_vals = np.arange(z_min-spacing, z_max+spacing, spacing)
        
        x, y, z = np.meshgrid(x_vals, y_vals, z_vals)
        xyz = np.c_[x.flatten(), y.flatten(), z.flatten()]
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        ax.scatter(xyz[:,0],
                xyz[:,1],
                xyz[:,2])
        
        print(xyz.shape)
        
        ### Based on indexing, it should be known exactly what vertex to fill
        ### each point on the grid with. 
        volume = np.zeros((xyz.shape))
        
    
    
#    def struct_to_volume(self, struct=None, spacing=0.5):
#        if struct == None:
#            struct = self.struct
#        
#        ### Making struct volume from the beginning
#        geo = struct.get_geo_array()
#        ele = struct.geometry["element"]
#        struct_radii = np.array([m.vdw[atomic_numbers[x]] for x in ele])
#        
#        ## Get min/max for xyz directions
#        max_geo = np.max(geo + struct_radii[:,None], axis=0)
#        min_geo = np.min(geo - struct_radii[:,None], axis=0)
#        
#        ## Provide spacing at edges
#        max_geo += spacing
#        min_geo -= spacing
#        
#        ## Now generate volume grid
#        x_vals = np.arange(min_geo[0], max_geo[0]+spacing, spacing)
#        y_vals = np.arange(min_geo[1], max_geo[1]+spacing, spacing)
#        z_vals = np.arange(min_geo[2], max_geo[2]+spacing, spacing)
#        
#        volume = np.zeros((x_vals.shape[0], 
#                           y_vals.shape[0], 
#                           z_vals.shape[0]))
#
#        for idx,point in enumerate(geo):
#            
#            ## Index into volume for the center of the atom
#            center_idx = np.round((point-min_geo) / spacing).astype(int)
#            
#            ## Now compute idx to also populate x,y,z directions for given radius
#            rad = struct_radii[idx]
#            num_idx = np.round(rad / spacing)
#            
#            offset = int(num_idx)
#            
#            ## Now, generate off triple offset pairs that add up to the offset to 
#            ## complete spherical grid indexing for atom
#            
#            offset = offset_combination_dict[offset]
#            final_idx = center_idx + offset
#            
#            volume[final_idx[:,0], final_idx[:,1], final_idx[:,2]] = 1
#            
#        ## Volume estimation is now as easy as multiplying the 1 entries in the
#        ## grid by the volume of each block
#        print(np.sum(volume)*spacing*spacing*spacing)
#        print(x_vals.shape, y_vals.shape, z_vals.shape)
#        
#        return volume
        
        
## Find all combinations of small values that lead to less than or equal
## to the largest value. This is equivalent to finding all grid points 
## within a certain radius
#offset_combination_dict = {}
#max_offset_value = np.round(np.max(m.vdw) / 0.25 ) + 1
#total = max_offset_value
#for value in range(int(max_offset_value+1)):
#    print(max_offset_value, value)
#    idx_range = np.arange(-value , value+1)[::-1]
#    
#    ## Sort idx_range array so the final list is sorted by magnitude 
#    ## so that lower index, and positive index, planes are given preference
#    sort_idx = np.argsort(np.abs(idx_range))
#    idx_range = idx_range[sort_idx]
#    all_idx = np.array(
#            np.meshgrid(idx_range,idx_range,idx_range)).T.reshape(-1,3)
#    all_norm = np.linalg.norm(all_idx, axis=-1)
#    take_idx = np.where(all_norm <= value)[0]
#    
#    final_idx = all_idx[take_idx]
#    offset_combination_dict[value] = final_idx
#    total -= 1        
        
    
"""
2D Implementation for marching square

### Interpolate for more accurate volume estimations
x_num = 10
y_num = 12

## Using top left, all other corners can be generated correctly
top_left_idx = np.arange(0,x_num-1)[None,:]
top_right_idx = top_left_idx + 1
bottom_left_idx = top_left_idx + x_num
bottom_right_idx = bottom_left_idx + 1

#top_right_idx = np.arange(1,x_num+1)
#bottom_left_idx = np.arange(x_num, 2*x_num)
#bottom_right_idx = np.arange(x_num+1, 2*x_num+1)

y_columns = np.arange(0,y_num)*x_num
top_left_idx = top_left_idx + y_columns[:,None]
top_right_idx = top_right_idx + y_columns[:,None]
bottom_left_idx = bottom_left_idx + y_columns[:,None]
bottom_right_idx = bottom_right_idx + y_columns[:,None]


idx_pairs = np.c_[top_left_idx.ravel(),
                  top_right_idx.ravel(),
                  bottom_left_idx.ravel(),
                  bottom_right_idx.ravel()]

corresponding_y_position = idx_pairs / x_num
corresponding_y_position = corresponding_y_position.astype(int)
corresponding_x_position = idx_pairs - x_num*corresponding_y_position
corresponding_x_position = corresponding_x_position.astype(int)

points = np.c_[corresponding_x_position.ravel(),
               corresponding_y_position.ravel()]
#top_left_idx

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(points[:,0], points[:,1])

for idx,value in enumerate(points[::4]):
    vert_idx = np.arange(idx*4, idx*4+4)
#    ax.plot(points[vert_idx[:2]][:,0], 
#            points[vert_idx[:2]][:,1])
#    ax.plot(points[vert_idx[2:4]][:,0], 
#            points[vert_idx[2:4]][:,1])
#    ax.plot(points[vert_idx[[0,2]]][:,0], 
#        points[vert_idx[[1,3]]][:,1])
#    points[vert_idx[[1,3]]])
    
#    break

"""


#def equal_axis_aspect(ax):
#    xticks = ax.get_xticks()
#    yticks = ax.get_yticks()
#    zticks = ax.get_zticks()
#    
#    xrange = xticks[-1] - xticks[0]
#    yrange = yticks[-1] - yticks[0]
#    zrange = zticks[-1] - zticks[0]
#    max_range = max([xrange,yrange,zrange]) / 2
#    
#    xmid = np.mean(xticks)
#    ymid = np.mean(yticks)
#    zmid = np.mean(zticks)
#    
#    ax.set_xlim(xmid - max_range, xmid + max_range)
#    ax.set_ylim(ymid - max_range, ymid + max_range)
#    ax.set_zlim(zmid - max_range, zmid + max_range)
#
#### Interpolate for more accurate volume estimations
#spacing=0.25
#m = HullVolume()
#struct = read("/Users/ibier/Research/Volume_Estimation/Datasets/PAHs_MC_Info/TETCEN01.json")
#align(struct)
#volume = m.struct_to_volume(struct, spacing=spacing)
##x_num = 6
##y_num = 3
##z_num = 3
#x_num,y_num,z_num = volume.shape
#
#### Making struct volume from the beginning
#geo = struct.get_geo_array()
#ele = struct.geometry["element"]
#struct_radii = np.array([m.vdw[atomic_numbers[x]] for x in ele])
#
### Get min/max for xyz directions
#max_geo = np.max(geo + struct_radii[:,None], axis=0)
#min_geo = np.min(geo - struct_radii[:,None], axis=0)
#
### Provide spacing at edges
#max_geo += spacing
#min_geo -= spacing
#
### Now generate volume grid
#x_vals = np.arange(min_geo[0], max_geo[0]+spacing, spacing)
#y_vals = np.arange(min_geo[1], max_geo[1]+spacing, spacing)
#z_vals = np.arange(min_geo[2], max_geo[2]+spacing, spacing)
#
#X,Y,Z = np.meshgrid( x_vals, y_vals, z_vals,
#                    indexing="ij")
#grid_point_reference = np.c_[X.ravel(),
#                             Y.ravel(),
#                             Z.ravel()]
#
### This should probably be written as a generated so it doesn't all have to 
### be stored in memory at once, but oh well
#
#
### Start by projecting down Z direction because this is easiest based on the 
### indexing scheme
#z_proj = np.arange(0,z_num-1)
#front_plane_top_left_idx = z_proj
#front_plane_bot_left_idx = front_plane_top_left_idx + 1
#
### Have to move 1 in the Y direction which is the same as z_num
#back_plane_top_left_idx = z_proj + z_num
#back_plane_bot_left_idx = back_plane_top_left_idx + 1
#
### Have to move 1 in the X direction which is the same as z_num*y_num 
#front_plane_top_right_idx = z_proj + y_num*z_num
#front_plane_bot_right_idx = front_plane_top_right_idx + 1
#
### Have to move 1 in the y direction which is the same as z_num
#back_plane_top_right_idx = front_plane_top_right_idx + z_num
#back_plane_bot_right_idx = back_plane_top_right_idx + 1
#
#
#
##### Now project over the Y direction
#y_proj = np.arange(0,y_num-1)[:,None]*(z_num)
#front_plane_top_left_idx = front_plane_top_left_idx + y_proj
#front_plane_bot_left_idx = front_plane_bot_left_idx+ y_proj
#back_plane_top_left_idx = back_plane_top_left_idx+ y_proj
#back_plane_bot_left_idx = back_plane_bot_left_idx+ y_proj
#front_plane_top_right_idx = front_plane_top_right_idx+ y_proj
#front_plane_bot_right_idx = front_plane_bot_right_idx+ y_proj
#back_plane_top_right_idx = back_plane_top_right_idx+ y_proj
#back_plane_bot_right_idx = back_plane_bot_right_idx+ y_proj
#
#
##### Lastly project in X direction
#x_proj = np.arange(0,x_num-1)[:,None,None]*(y_num*z_num)
#front_plane_top_left_idx = front_plane_top_left_idx + x_proj
#front_plane_bot_left_idx = front_plane_bot_left_idx + x_proj
#back_plane_top_left_idx = back_plane_top_left_idx + x_proj
#back_plane_bot_left_idx = back_plane_bot_left_idx + x_proj
#front_plane_top_right_idx = front_plane_top_right_idx + x_proj
#front_plane_bot_right_idx = front_plane_bot_right_idx + x_proj
#back_plane_top_right_idx = back_plane_top_right_idx + x_proj
#back_plane_bot_right_idx = back_plane_bot_right_idx + x_proj
##
#voxel_idx = np.c_[front_plane_top_left_idx.ravel(),
#                  front_plane_bot_left_idx.ravel(),
#                  back_plane_bot_left_idx.ravel(),
#                  back_plane_top_left_idx.ravel(),
#                  front_plane_top_right_idx.ravel(),
#                  front_plane_bot_right_idx.ravel(),
#                  back_plane_bot_right_idx.ravel(),
#                  back_plane_top_right_idx.ravel(),
#                  ]
#
#voxel_mask = np.take(volume, voxel_idx)
#voxel_sum = np.sum(voxel_mask, axis=-1)
#vertex_idx = np.where(np.logical_and(voxel_sum != 0,
#                                     voxel_sum != 8))[0]
#surface_vertex_idx = voxel_idx[vertex_idx][voxel_mask[vertex_idx].astype(bool)]
#surface_vertex = grid_point_reference[surface_vertex_idx]
##voxel_unique,counts = np.unique(voxel_mask, 
##                                return_counts=True, 
##                                axis=0)
##
#
#
#
#
###### Plot grid points
##cart_points = grid_point_reference[voxel_idx.ravel()]
##fig = plt.figure(figsize=(10, 10))
##ax = fig.add_subplot(111, projection='3d')
##ax.scatter(cart_points[:,0],#[0:8],
##           cart_points[:,1],#[0:8],
##           cart_points[:,2],#[0:8],
##           edgecolor="k")
##equal_axis_aspect(ax)
##plt.show()
##plt.close()
#
#
#
#
#
###### Plot Triangulation
#triangul_table = {}
#
#def compute_edge_sites(cube_vertex):
#    pair_idx = np.array([
#            [0,1],
#            [0,3],
#            [2,3],
#            [1,2],
#            
#            [0,4],
#            [3,7],
#            [2,6],
#            [1,5],
#            
#            [4,5],
#            [4,7],
#            [6,7],
#            [5,6],
#            
#            ])
#    
#    pairs = cube_vertex[pair_idx]
#    edge = np.mean(pairs, axis=1)
#    return edge
#
#cart_points = grid_point_reference[voxel_idx.ravel()]
#
###### PLOT CUBE
##fig = plt.figure(figsize=(10, 10))
##ax = fig.add_subplot(111, projection='3d')
##ax.scatter(cart_points[:,0][0:8],
##           cart_points[:,1][0:8],
##           cart_points[:,2][0:8],
###           c=["tab:red", "tab:blue",
###              "tab:green", "tab:orange",
###              "tab:red", "tab:blue",
###              "tab:green", "tab:orange"]
##           )
##
##for idx,point in enumerate(cart_points[0:8]):
##    ax.text(point[0],
##           point[1], 
##           point[2],
##           "{}".format(idx),
##           fontsize=16)
##    
##
##cube_vertex = cart_points[:8]
##edge_vertex = compute_edge_sites(cube_vertex)
##
###### Visualize edge points
##ax.scatter(edge_vertex[:,0],
##           edge_vertex[:,1],
##           edge_vertex[:,2],
##           edgecolor="k",
##           facecolor="tab:red")
##
##for idx,point in enumerate(edge_vertex):
###    idx += 8
##    ax.text(point[0],
##           point[1], 
##           point[2],
##           "{}".format(idx),
##           fontsize=16)
#
#
##ax.plot_trisurf(cart_points[:,0][0:8],
##           cart_points[:,1][0:8],
##           cart_points[:,2][0:8],
##           edgecolor="k")
##ax.plot()
##equal_axis_aspect(ax)
##plt.show()
##plt.close()
#
#
#def apply_symmetry(lookup_idx, vertex_mask):
#    #### Let's perform all rotations on lookup idx first
#    
#    ### Construct symmetry relevant orderings
##    symmetry_idx_list = [
##            [0,1,2,3,4,5,6,7],
##            [1,2,3,0,5,6,7,4],
##            [2,3,0,1,6,7,4,5],
##            [3,0,1,2,7,4,5,6],
##            [4,5,1,0,7,6,2,3],
##            [7,6,5,4,3,2,1,0],
##            [3,2,6,7,0,1,5,4],
##            [4,0,3,7,5,1,2,6],
##            [5,4,7,6,1,0,3,2],
##            [1,5,6,2,0,4,7,3],
##            [6,7,4,5,2,3,0,1]
##            ]
#    symmetry_idx_list = [
#            [0,1,2,3,4,5,6,7],
#            [4,5,1,0,7,6,2,3],
#            [7,6,5,4,3,2,1,0],
#            [3,2,6,7,0,1,5,4],
#            [4,0,3,7,5,1,2,6],
#            [5,4,2,6,1,0,3,2],
#            [1,5,6,3,0,4,6,7],
#            [2,3,0,1,6,5,4,5],
#            [6,7,4,5,2,3,0,1]
#            ]
#    
#    all_lookup_idx = []
#    
#    for idx_list in symmetry_idx_list:
#        all_lookup_idx.append(lookup_idx[idx_list])
#        
#        
#    ### Construct corresponding symmetry relevant ordings for vertex/edge 
#    ### for triangulation
#    edge_symmetry_idx_list = [
#            [0,1,2,3,4,5,6,7,8,9,10,11],
#            [8,4,0,7,9,1,3,11,10,5,2,6],
#            [10,9,8,11,5,4,7,6,2,1,0,3],
#            [2,5,10,6,1,9,11,3,0,4,8,7],
#            [4,9,5,1,8,10,2,0,7,11,6,3],
#            [8,11,10,9,7,6,5,4,0,3,2,1],
#            [7,3,6,11,0,2,10,8,4,1,5,9],
#            [2,3,0,1,6,7,4,5,10,11,8,9],
#            [10,11,8,9,6,7,4,5,2,3,0,1],
#            ]
#    edge_symmetry_idx_list = np.array(edge_symmetry_idx_list)
#    
#    all_tri = vertex_mask[edge_symmetry_idx_list]
#    
#    return all_lookup_idx,all_tri
#
#
#
#
##### Plotting all 
#
#
#
#
#
#
#
#
#vertex_lookup[1,0,0,0,0,0,0,0][first_tri] = 1
#
#temp,tri = apply_symmetry(np.array([1,0,0,0,0,0,0,0]), first_tri)
#
#
#### Visualize triangles
#
#tri_base_idx = np.arange(0,12)
#for it_tri_idx,tri_bool in enumerate(tri):
#    
#    fig = plt.figure(figsize=(10, 10))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(cart_points[:,0][0:8],
#               cart_points[:,1][0:8],
#               cart_points[:,2][0:8],
#    #           c=["tab:red", "tab:blue",
#    #              "tab:green", "tab:orange",
#    #              "tab:red", "tab:blue",
#    #              "tab:green", "tab:orange"]
#               )
#
#    for idx,point in enumerate(cart_points[0:8]):
#        ax.text(point[0],
#               point[1], 
#               point[2],
#               "{}".format(idx),
#               fontsize=16)
#        
#    
#    cube_vertex = cart_points[:8]
#    edge_vertex = compute_edge_sites(cube_vertex)
#    
#    #### Visualize edge points
#    ax.scatter(edge_vertex[:,0],
#               edge_vertex[:,1],
#               edge_vertex[:,2],
#               edgecolor="k",
#               facecolor="tab:red")
#    
#    for idx,point in enumerate(edge_vertex):
#        ax.text(point[0],
#               point[1], 
#               point[2],
#               "{}".format(idx),
#               fontsize=16)
#    
#    ## Tri idx
#    temp_tri_idx = tri_base_idx[tri_bool]
#    temp_tri_idx = temp_tri_idx.tolist()
#    temp_tri_idx.append(temp_tri_idx[0])
#    temp_tri_vertex = edge_vertex[temp_tri_idx,:]
#    ax.plot(temp_tri_vertex[:,0],
#            temp_tri_vertex[:,1],
#            temp_tri_vertex[:,2])
#    
#    center_idx = np.where(temp[it_tri_idx] == 1)[0]
#    center_vert = cart_points[center_idx][0]
#    ax.scatter(center_vert[0],
#               center_vert[1],
#               center_vert[2],
#               color="tab:green",
#               s=100)
#
#
#
##
##fig = plt.figure(figsize=(10, 10))
##ax = fig.add_subplot(111, projection='3d')
##ax.scatter(surface_vertex[:,0],
##           surface_vertex[:,1],
##           surface_vertex[:,2],
##           edgecolor="k")
##equal_axis_aspect(ax)
##plt.show()
##plt.close()
##
##fig = plt.figure(figsize=(10, 10))
##ax = fig.add_subplot(111, projection='3d')
##test_points = np.vstack(grid_point_reference[voxel_idx[0:10000],:])
##ax.voxels(volume)
##equal_axis_aspect(ax)
##plt.show()
##plt.close()
#


if __name__ == "__main__":
    import json 
    
#    from ibslib.plot.utils import colors_from_colormap
#    
#    from ibslib import SDS
#    from ibslib.io import read,write
#    from ibslib.molecules import align
#    from sklearn.decomposition import PCA
#    
#    from scipy.spatial import Delaunay
#    
#    from mpl_toolkits.mplot3d import Axes3D
#    import matplotlib.pyplot as plt
#    from matplotlib.tri.triangulation import Triangulation
#    from matplotlib import cm
#    
#    from sklearn.neighbors import KDTree
#    from scipy.spatial import Voronoi
#    from skimage.segmentation import flood_fill
#    
#    from skimage.measure import marching_cubes_lewiner, find_contours
#    
#    def tetrahedron_volume(a, b, c, d):
#        return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6
#    
##    outstream = SDS("/Users/ibier/Desktop/Temp/SURFACE_GRIDS",
##                    file_format="geo",
##                    overwrite=True)
#    
#    struct_name = "PUDXES02"
####    struct_name = "ICIXOH01"
####    struct_name = "JAGREP"
####    struct_name = "CYACAC"
##    struct_name = "SOZBUE01"
#    
#    struct = read("/Users/ibier/Research/Volume_Estimation/Models/20200217_Model_Test/Dataset/{}.json".format(struct_name))
#    align(struct)
#    m = HullVolume(lebedev_n=65)
#    
##    values = []
##    shapes = []
##    for spacing in np.linspace(0.01,0.1,100)[::-1]:
##        print(spacing)
##        vol,shape = m.struct_to_volume(struct, spacing=spacing)
##        values.append(vol)
##        shapes.append(shape)
##    
##    
##    plt.plot(previous_values)
##    plt.plot(values)
#    
#    
#    spacing = 0.1
#    
#    ### Making struct volume from the beginning
#    geo = struct.get_geo_array()
#    ele = struct.geometry["element"]
#    struct_radii = np.array([m.vdw[atomic_numbers[x]] for x in ele])
#    
#    ## Get min/max for xyz directions
#    max_geo = np.max(geo + struct_radii[:,None], axis=0)
#    min_geo = np.min(geo - struct_radii[:,None], axis=0)
#    
#    ## Provide spacing at edges
#    max_geo += spacing
#    min_geo -= spacing
#    
#    ## Now generate volume grid
#    x_vals = np.arange(min_geo[0], max_geo[0]+spacing, spacing)
#    y_vals = np.arange(min_geo[1], max_geo[1]+spacing, spacing)
#    z_vals = np.arange(min_geo[2], max_geo[2]+spacing, spacing)
#    
#    volume = np.zeros((x_vals.shape[0], 
#                       y_vals.shape[0], 
#                       z_vals.shape[0]))
#
#    for idx,point in enumerate(geo):
#        
#        ## Index into volume for the center of the atom
#        center_idx = np.round((point-min_geo) / spacing).astype(int)
#        
#        ## Now compute idx to also populate x,y,z directions for given radius
#        rad = struct_radii[idx]
#        num_idx = np.round(rad / spacing)
#        
#        offset = int(num_idx)
#        
#        ## Now, generate off triple offset pairs that add up to the offset to 
#        ## complete spherical grid indexing for atom
#        
#        offset = offset_combination_dict[offset]
#        final_idx = center_idx + offset
#        
#        volume[final_idx[:,0], final_idx[:,1], final_idx[:,2]] = 1
#        
#    ## Volume estimation is now as easy as multiplying the 1 entries in the
#    ## grid by the volume of each block
#    print(np.sum(volume)*spacing*spacing*spacing)
#    
#    
#    def wrap_plane(plane, max_contour=1):
#        """
#        Will find a way to wrap obejcts in the plane by following the curvature
#        of the structure but without using large changes. This is identical
#        to wrapping the object in a fabric with a certain amount of stretch 
#        possible in the fabric. 
#        """
#        pass
    
    
    def fill_holes(plane):
        not_plane = np.logical_not(plane).astype(int)
        filled = flood_fill(not_plane, (0,0), 0)
        filled = np.logical_or(plane,filled)

        return filled
    
    
    def fit_parabola(a,b,c):
        x1 = a[0]
        y1 = a[1]
        x2 = b[0]
        y2 = b[1]
        x3 = c[0]
        y3 = c[1]
        denom = (x1-x2) * (x1-x3) * (x2-x3)
        print(denom)
        A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
        B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
        C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * \
                 (x1-x2) * y3) / denom;
        return A,B,C
    
    
#    def marching_square_contour_smoothing(contour):
           
    
        
    
    ## Testing modifications
    X,Y = np.meshgrid(x_vals, y_vals)
    for idx,xy_plane in enumerate(volume.T[10:]):
        if np.sum(xy_plane) == 0:
            continue
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
#        cp = ax.contourf(fill_holes(xy_plane))
#        cp = ax.contourf(X,Y,xy_plane)
        
        
        filled = fill_holes(xy_plane)
        cp = ax.contourf(fill_holes(filled))
        contours = find_contours(filled, 0.8)
        curvature = []
        for contour in contours:   
            ## Last point is repetition of first
            contour = contour[:-1]
            
            ax.plot(contour[:,1], contour[:,0], linewidth=2)
            center = np.mean(contour, axis=0)
            ax.scatter(center[1], center[0])
            
            
            
            
#            print(contour.shape)
        
        center = np.array([int(xy_plane.shape[0] / 2),
                           int(xy_plane.shape[1] / 2)])
                
                
        ax.scatter(center[0], center[1], s=100, c="tab:red")
        
        fig.colorbar(cp)
        plt.show()
        plt.close()
        
        
        break
    
    
    ## Testing parabola fitting
#    a = [0,0]
#    b = [1,1]
#    c = [2,-1]
#    
#    A,B,C = fit_parabola(a,b,c)
#    
#    x = np.arange(0,2+0.1,0.1)
#    y = A*np.square(x) + B*x + C
#    curv = np.repeat(2*A, x.shape)
#    
#    plt.plot(x, y)
#    plt.plot(x, curv)


### Calculating center of gravity for plane
#        x_weights = np.sum(xy_plane, axis=0)
#        x_cumsum = np.cumsum(x_weights)
#        x_mid = x_cumsum[-1] / 2
#        x_cog = np.where(x_cumsum <= x_mid)[0][-1] + 1
#        
#        y_weights = np.sum(xy_plane, axis=1)
#        y_cumsum = np.cumsum(y_weights)
#        y_mid = y_cumsum[-1] / 2
#        y_cog = np.where(y_cumsum <= y_mid)[0][-1] + 1
#        
#        ax.scatter(x_cog,y_cog,s=100,
#                   c="tab:cyan")
#        ax.scatter(int(xy_plane.shape[0] / 2),
#                   int(xy_plane.shape[1] / 2),
#                   s=100,
#                   c="tab:red")
        
    
        
#    print(np.sum(volume)*spacing*spacing*spacing)
        
   
#    Y,Z = np.meshgrid(y_vals,z_vals)
#    for idx,yz_plane in enumerate(np.swapaxes(volume, 1, 2)):
#        if np.sum(yz_plane) == 0:
#            continue
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        
#        cp = ax.contourf(Y,Z,fill_holes(yz_plane))
#        cp = ax.contourf(Y,Z,yz_plane)
#        fig.colorbar(cp)
#        plt.show()
#        plt.close() 
#        
        
#        filled = fill_holes(yz_plane)
#        np.swapaxes(volume, 1, 2)[idx] = filled

        
#    print(np.sum(volume)*spacing*spacing*spacing)
    
#    X,Z = np.meshgrid(x_vals,z_vals)
#    for idx,xz_plane in enumerate(np.swapaxes(volume, 0, 1)):
#        if np.sum(xz_plane) == 0:
#            continue
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        filled = fill_holes(xz_plane)
#        np.swapaxes(volume, 0, 1)[idx] = filled
#        cp = ax.contourf(X,Z,filled)
#        cp = ax.contourf(X,Z,xz_plane.T)
#        fig.colorbar(cp)
#        plt.show()
#        plt.close() 
    
#    print(np.sum(volume)*spacing*spacing*spacing)
     
    
    ### Algorithm for modified volume
#    Y,Z = np.meshgrid(y_vals,z_vals)
#    for idx,yz_plane in enumerate(np.swapaxes(volume, 1, 2)):
#        if np.sum(yz_plane) == 0:
#            continue
##        fig = plt.figure()
##        ax = fig.add_subplot(111)
#        filled = fill_holes(yz_plane)
#        np.swapaxes(volume, 1, 2)[idx] = filled
##        cp = ax.contourf(Y,Z,fill_holes(yz_plane))
##        cp = ax.contourf(Y,Z,yz_plane)
##        fig.colorbar(cp)
##        plt.show()
##        plt.close() 
#        
#    print(np.sum(volume)*spacing*spacing*spacing)
#    
#    X,Z = np.meshgrid(x_vals,z_vals)
#    for idx,xz_plane in enumerate(np.swapaxes(volume, 0, 1)):
#        if np.sum(xz_plane) == 0:
#            continue
##        fig = plt.figure()
##        ax = fig.add_subplot(111)
#        filled = fill_holes(xz_plane)
#        np.swapaxes(volume, 0, 1)[idx] = filled
##        cp = ax.contourf(X,Z,filled)
##        cp = ax.contourf(X,Z,xz_plane.T)
##        fig.colorbar(cp)
##        plt.show()
##        plt.close() 
#    
#    print(np.sum(volume)*spacing*spacing*spacing)
    
    
#    spacing = 0.047
#    
#    ### Making struct volume from the beginning
#    geo = struct.get_geo_array()
#    ele = struct.geometry["element"]
#    struct_radii = np.array([m.vdw[atomic_numbers[x]] for x in ele])
#    
#    ## Get min/max for xyz directions
#    max_geo = np.max(geo + struct_radii[:,None], axis=0)
#    min_geo = np.min(geo - struct_radii[:,None], axis=0)
#    
#    ## Provide spacing at edges
#    max_geo += spacing
#    min_geo -= spacing
#    
#    ## Now generate volume grid
#    x_vals = np.arange(min_geo[0], max_geo[0]+spacing, spacing)
#    y_vals = np.arange(min_geo[1], max_geo[1]+spacing, spacing)
#    z_vals = np.arange(min_geo[2], max_geo[2]+spacing, spacing)
#    
#    volume = np.zeros((x_vals.shape[0], 
#                       y_vals.shape[0], 
#                       z_vals.shape[0]))
#    
#    print(volume.shape)
#    
#    ## Find all combinations of small values that lead to less than or equal
#    ## to the largest value. This is equivalent to finding all grid points 
#    ## within a certain radius
#    offset_combination_dict = {}
#    max_offset_value = np.round(np.max(m.vdw) / spacing) + 1
#    total = max_offset_value
#    for value in range(int(max_offset_value+1)):
##        print(total, value)
#        idx_range = np.arange(-value , value+1)[::-1]
#        
#        ## Sort idx_range array so the final list is sorted by magnitude 
#        ## so that lower index, and positive index, planes are given preference
#        sort_idx = np.argsort(np.abs(idx_range))
#        idx_range = idx_range[sort_idx]
#        all_idx = np.array(
#                np.meshgrid(idx_range,idx_range,idx_range)).T.reshape(-1,3)
#        all_norm = np.linalg.norm(all_idx, axis=-1)
#        take_idx = np.where(all_norm <= value)[0]
#        
#        final_idx = all_idx[take_idx]
#        offset_combination_dict[value] = final_idx
#        total -= 1
#        
##    with open("offset.json", "w") as f:
##        f.write(json.dumps(offset_combination_dict))
#    
#    correction = 1
#    for idx,point in enumerate(geo):
#        
#        ## Index into volume for the center of the atom
#        center_idx = np.round((point-min_geo) / spacing).astype(int)
#        
##        print(idx)
##        print(point)
##        print(center_idx)
##        print(x_vals[center_idx[0]], 
##              y_vals[center_idx[1]], 
##              z_vals[center_idx[2]])
##        
#        ## Now compute idx to also populate x,y,z directions for given radius
#        rad = struct_radii[idx]
#        num_idx = np.round(rad / spacing)
#        
#        ## If value is odd, then count central idx, otherwise, don't count it
#        ## For a small grid spacing, this leads to only a tiny numerical 
#        ## approximation. In addition, because it's a radius, the surface 
#        ## created by this operation will be consistent. 
##        if num_idx % 2 != 0:
##            offset = int((num_idx+1))
##        else:
##            offset = int(num_idx)
#        
#        offset = int(num_idx+correction)
#        correction *= -1
#        
#        ## Now, generate off triple offset pairs that add up to the offset to 
#        ## complete spherical grid indexing for atom
#        
#        offset = offset_combination_dict[offset]
#        final_idx = center_idx + offset
#        
#        volume[final_idx[:,0], final_idx[:,1], final_idx[:,2]] = 1
#        
##        break
#        
#    ## Volume estimation is now as easy as multiplying the 1 entries in the
#    ## grid by the volume of each block
#    print(np.sum(volume)*spacing*spacing*spacing)    
#    
#    
#    def print_corr_value(idx):
#        print(x_vals[idx[0]], y_vals[idx[1]], z_vals[idx[2]])
    
    ## Visualize marching cubes surface
#    verts, faces, normals, values = marching_cubes_lewiner(volume, level=0,
#                                            spacing=(spacing,spacing,spacing))
#    
#    print(verts.shape, faces.shape)
    
    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.voxels(volume, edgecolor='k')
    
#    fig = plt.figure(figsize=(10, 10))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_trisurf(verts[:,0],
#                    verts[:,1],
#                    verts[:,2], 
#                    triangles=faces,
#                    cmap=cm.viridis,
#                    linewidth=0.2, antialiased=True)
    
    ## Can easily compute idx for all points in vdW surface easily
#    all_idx = np.round((grid-min_geo) / spacing)
    
#    all_idx = all_idx.astype(int)
#    volume[all_idx[:,0], all_idx[:,1], all_idx[:,2]] = 1
    
    
    
    
    
    
#    X,Y = np.meshgrid(x_vals, y_vals)
#    for xy_plane in volume.T:
#        if np.sum(xy_plane) == 0:
#            continue
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
##        cp = ax.contourf(X,Y,fill_holes(xy_plane))
#        cp = ax.contourf(X,Y,xy_plane)
#        fig.colorbar(cp)
#        plt.show()
#        plt.close()
#        
#        
#    Y,Z = np.meshgrid(y_vals,z_vals)
#    for yz_plane in np.swapaxes(volume, 1, 2):
#        if np.sum(yz_plane) == 0:
#            continue
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
##        cp = ax.contourf(Y,Z,fill_holes(yz_plane))
#        cp = ax.contourf(Y,Z,yz_plane)
#        fig.colorbar(cp)
#        plt.show()
#        plt.close() 
#    
#    X,Z = np.meshgrid(x_vals,z_vals)
#    for xz_plane in np.swapaxes(volume, 0, 1):
#        if np.sum(xz_plane) == 0:
#            continue
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
##        cp = ax.contourf(X,Z,fill_holes(xz_plane.T))
#        cp = ax.contourf(X,Z,xz_plane.T)
#        fig.colorbar(cp)
#        plt.show()
#        plt.close() 
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    m = HullVolume(lebedev_n=65)
#    m.calc_struct(struct)
##    surface_grid = m._surface_grid()
#    
##    m.grid_to_volume(m.grid, spacing=1)
#    spacing = 0.45x
##    
#    grid = m.grid
##    grid = surface_grid
#    ### Volume method
#    x_min = np.min(grid[:,0])
#    x_max = np.max(grid[:,0])
#    y_min = np.min(grid[:,1])
#    y_max = np.max(grid[:,1])
#    z_min = np.min(grid[:,2])
#    z_max = np.max(grid[:,2])
#    
#    x_vals = np.arange(x_min-spacing, x_max+2*spacing, spacing)
#    y_vals = np.arange(y_min-spacing, y_max+2*spacing, spacing)
#    z_vals = np.arange(z_min-spacing, z_max+2*spacing, spacing)
#    
#    volume = np.zeros((x_vals.shape[0], 
#                       y_vals.shape[0], 
#                       z_vals.shape[0]))
#    
#    offset = np.array([x_min-spacing, 
#                       y_min-spacing, 
#                       z_min-spacing])
#    all_idx = np.round((grid-offset) / spacing)
#    
#    all_idx = all_idx.astype(int)
#    volume[all_idx[:,0], all_idx[:,1], all_idx[:,2]] = 1
#    
#    from skimage.measure import marching_cubes_lewiner, find_contours
#    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#    
#    verts, faces, normals, values = marching_cubes_lewiner(volume, 0,
#                                            spacing=(spacing,spacing,spacing))
#    
#    print(verts.shape, faces.shape)
    
#    fig = plt.figure(figsize=(10, 10))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_trisurf(verts[:,0],
#                    verts[:,1],
#                    verts[:,2], 
#                    triangles=faces,
#                    cmap=cm.viridis,
#                    linewidth=0.2, antialiased=True)
    
#    ax.scatter(verts[:,0], verts[:,1], verts[:,2])
#    mesh = Poly3DCollection(verts[faces])
#    mesh.set_edgecolor('k')
#    ax.add_collection3d(mesh)
    
#    plt.tight_layout()
#    plt.show()
#    
#    raise Exception()
    
#    def fill_holes(plane):
#        not_plane = np.logical_not(plane).astype(int)
#        filled = flood_fill(not_plane, (0,0), 0)
#        filled = np.logical_or(plane,filled)
#
#        return filled
#    
#    X,Y = np.meshgrid(x_vals, y_vals)
#    for xy_plane in volume.T:
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        cp = ax.contourf(X,Y,fill_holes(xy_plane))
#        fig.colorbar(cp)
#        plt.show()
#        plt.close()
#        
#        
#    Y,Z = np.meshgrid(y_vals,z_vals)
#    for yz_plane in np.swapaxes(volume, 1, 2):
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        cp = ax.contourf(Y,Z,fill_holes(yz_plane))
#        fig.colorbar(cp)
#        plt.show()
#        plt.close() 
#    
#    X,Z = np.meshgrid(x_vals,z_vals)
#    for xz_plane in np.swapaxes(volume, 0, 1):
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        cp = ax.contourf(X,Z,fill_holes(xz_plane.T))
#        fig.colorbar(cp)
#        plt.show()
#        plt.close() 
    
    ## Finding contours
#    plane = volume.T[8]
#    not_plane = np.logical_not(plane).astype(int)
#    filled = flood_fill(not_plane, (0,0), 0)
#    filled = np.logical_or(plane,filled)
#    
#    X,Y = np.meshgrid(x_vals, y_vals)
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    cp = ax.contourf(X,Y,plane)
#    fig.colorbar(cp)
#    plt.show()
#    plt.close()
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    cp = ax.contourf(X,Y,filled)
#    fig.colorbar(cp)
#    plt.show()
#    plt.close()
    
    
    
#    contours = find_contours(volume[10], 0.1)
#    
#    fig, ax = plt.subplots(figsize=(10,10))
#    ax.imshow(volume[10])
#    
#    for contour in contours:
#        ax.plot(contour[:,1], contour[:,0], linewidth=2)
#        
#    plt.tight_layout()
    
    
    
#    fig = plt.figure(figsize=(10, 10))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_trisurf(verts[:,0],
#                    verts[:,1],
#                    verts[:,2], 
#                    triangles=faces,
#                    cmap=cm.viridis,
#                    linewidth=0.2, antialiased=True)
    
#    ax.scatter(verts[:,0], verts[:,1], verts[:,2])
#    mesh = Poly3DCollection(verts[faces])
#    mesh.set_edgecolor('k')
#    ax.add_collection3d(mesh)
#    
#    plt.tight_layout()
#    plt.show()
    
    
    
#    x, y, z = np.meshgrid(x_vals, y_vals, z_vals)
#    xyz = np.c_[x.flatten(), y.flatten(), z.flatten()]
#    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    
#    ax.scatter(xyz[:,0],
#            xyz[:,1],
#            xyz[:,2])
#    
#    print(xyz.shape)
    
    ### Based on indexing, it should be known exactly what vertex to fill
    ### each point on the grid with. 
    
    
    ##### KDTree trisurface method
#    kd = KDTree(surface_grid)
#    tri_test = kd.query(surface_grid, k=6, return_distance=False)
#    
#    tri_list = []
#    for entry in tri_test:
#        points = surface_grid[entry,:]
#        tri_obj = Triangulation(points[:,0], points[:,1])
#        tri_idx = tri_obj.get_masked_triangles()
#        tri_list.append(entry[tri_idx])
#    
#    tri_list = np.vstack(tri_list)
    
    
    ## Construct list of verlet positions
    ## Construct verlet lists of interacting verlets
    ## When verlets are removed due to being too close, update verlet lists
    ## Can approximate curvature at each verlet location
    ## Update verlet location where curvature of surface is too large
    ## Move verlet in the opposite direction of curvature 
    ## Check to see if all verlet curvature values satisfy the desired 
    ## maximum curvature

#    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    
##    ax.scatter(surface_grid[:,0],
##                surface_grid[:,1],
##                surface_grid[:,2])
#    ax.plot_trisurf(surface_grid[:,0],
#                    surface_grid[:,1],
#                    surface_grid[:,2], 
##                    triangles=tri_list,
#                    triangles = m.surface_triangles,
##                    cmap=cm.viridis,
#                    linewidth=0.2, antialiased=True)
#    
#    
#    
## Sklearn marhing cubes
#    from skimage.measure import marching_cubes_lewiner
#    
#    verts, faces, normals, values = marching_cubes_lewiner(surface_grid)
#    x_min = np.min(surface_grid[0])
#    x_max = np.max(surface_grid[0])
#    y_min = np.min(surface_grid[1])
#    y_max = np.max(surface_grid[1])
#    
#    x_vals = np.linspace(x_min, x_max, 100)
#    y_vals = np.linspace(y_min, y_max, 100)
#    
#    z_sorted_surface_grid = surface_grid[np.argsort(surface_grid[:,-1])]
    
#    volume = np.dstack([])
 
#"""
#https://stackoverflow.com/questions/29800749/delaunay-triangulation-of-points-from-2d-surface-in-3d-with-python
#
#"""
    
#    from matplotlib import path as mpath
#    
#    def make_star(amplitude=1.0, rotation=0.0):
#        """ Make a star shape
#        """
#        t = np.linspace(0, 2*np.pi, 6) + rotation
#        star = np.zeros((12, 2))
#        star[::2] = np.c_[np.cos(t), np.sin(t)]
#        star[1::2] = 0.5*np.c_[np.cos(t + np.pi / 5), np.sin(t + np.pi / 5)]
#        return amplitude * star
#    
#    def make_stars(n_stars=51, z_diff=0.05):
#        """ Make `2*n_stars-1` stars stacked in 3D
#        """
#        amps = np.linspace(0.25, 1, n_stars)
#        amps = np.r_[amps, amps[:-1][::-1]]
#        rots = np.linspace(0, 2*np.pi, len(amps))
#        zamps = np.linspace
#        stars = []
#        for i, (amp, rot) in enumerate(zip(amps, rots)):
#            star = make_star(amplitude=amp, rotation=rot)
#            height = i*z_diff
#            z = np.full(len(star), height)
#            star3d = np.c_[star, z]
#            stars.append(star3d)
#        return stars
#    
#    def polygon_to_boolean(points, xvals, yvals):
#        """ Convert `points` to a boolean indicator mask
#        over the specified domain
#        """
#        x, y = np.meshgrid(xvals, yvals)
#        xy = np.c_[x.flatten(), y.flatten()]
#        mask = mpath.Path(points).contains_points(xy).reshape(x.shape)
#        return x, y, mask
#    
#        
#    # Make and plot the 2D contours
#    stars3d = make_stars()
#
#    xvals = np.linspace(-1, 1, 101)
#    yvals = np.linspace(-1, 1, 101)
#
#    volume = np.dstack([
#        polygon_to_boolean(star[:,:2], xvals, yvals)[-1]
#        for star in stars3d
#    ]).astype(float)

#    mlab.contour3d(volume, contours=[0.5])
#    mlab.show()

    
#    ax.plot_trisurf(m.hull.points[:,0],
#                    m.hull.points[:,1],
#                    m.hull.points[:,2], 
#                    triangles=m.surface_triangles,
#                    linewidth=0.2, antialiased=True)
    
#    print(m.hull.volume)
    
#    m.calc_struct(struct)
#    surface_grid = m._surface_grid(struct=struct,
#                                   neighbor_list=m.vdw_neigh)
#    grid_struct = m.visualize_grid(surface_grid)
#    grid_struct.struct_id = "{}_grid".format(struct_name)
#    outstream.update(grid_struct)
#        
#    
#    dt = Delaunay(surface_grid)    
#    tets = dt.points[dt.simplices]
#    vol = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1], 
#                                    tets[:, 2], tets[:, 3]))
#    
#    ## Small tetrahedron test assumes that the tetrahedrons formed by 
#    ## only neighboring indices are correct for internal volume
#    points_per_atom,_,_ = spa.grids.lebedev(n=m.lebedev_n)
#    points_per_atom = points_per_atom.shape[0]
#    delta_max = np.max(dt.simplices, axis=-1)
#    delta_min = np.min(dt.simplices, axis=-1)
#    delta = delta_max - delta_min
#    small_idx = np.where(delta < points_per_atom)[0]
#    small_tets = dt.points[dt.simplices[small_idx]]
#    small_vol = np.sum(tetrahedron_volume(small_tets[:, 0], small_tets[:, 1], 
#                                    small_tets[:, 2], small_tets[:, 3])) 
#    
#    print("TET: {}".format(vol))
#    print("HULL: {}".format(m.hull.volume))
#    print("SMALL TET: {}".format(small_vol))
#    print("MC: {}".format(struct.properties["mc_volume"]))
    
#    from mpl_toolkits.mplot3d import Axes3D
#    import matplotlib.pyplot as plt
#    import numpy as np
#    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    
#    dt = Delaunay(surface_grid[:,[0,-1]])
#    
#    ax.plot_trisurf(surface_grid[:,0],
#                    surface_grid[:,1],
#                    surface_grid[:,2], 
#                    triangles=dt.simplices,
#                    linewidth=0.2, antialiased=True)
    
#    tri = np.array([[0,1,2],
#                    [0,1,3],
#                    [0,2,3],
#                    [1,2,3]])
#    for entry in dt.simplices:
#        points = dt.points[entry]
#        ax.plot_trisurf(points[:,0],
#                        points[:,1],
#                        points[:,2],
#                        triangles=tri)
    
#    tets_stack = np.vstack(tets)
#    
#    ax.scatter(tets_stack[:,0],
#               tets_stack[:,1],
#               tets_stack[:,2])
#    ax.plot_trisurf(tets_stack[:,0],
#                   tets_stack[:,1],
#                   tets_stack[:,2], 
#                    triangles=dt.simplices,
#                    linewidth=0.2, antialiased=True)
#    
#    plt.show()
#    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    
#    ax.plot_trisurf(tets_stack[:,0],
#                   tets_stack[:,1],
#                   tets_stack[:,2], 
#                    triangles=dt.simplices,
#                    linewidth=0.2, antialiased=True)
    
    
    ## Neighborlist testing
#    geo = struct.get_geo_array()
#    ele = struct.geometry["element"]
#    
#    vdw_list = [m.vdw[atomic_numbers[x]] for x in ele]
#    
#    neighbor_list = [[] for x in range(geo.shape[0])]
#    for idx,coord in enumerate(geo):
#        
#        dist = cdist(coord[None,:], geo)
#        dist = dist.ravel()
#        temp_vdw = vdw_list[idx]
#        temp_neigh = np.where(dist < temp_vdw)[0]
#        not_self_idx = np.where(temp_neigh != idx)[0]
#        
#        temp_neigh = temp_neigh[not_self_idx]
#        
#        ## Need to correct fo r
#        neighbor_list[idx] = temp_neigh
        
#    radius = 2
#    analytical_volume = (4/3)*np.pi*radius*radius*radius
#    
#    azi, colat, weights = spa.grids.lebedev(n=24)
#    coords = spa.utils.sph2cart(azi, colat)
#    coords = np.hstack([coords[0][:,None], 
#                        coords[1][:,None], 
#                        coords[2][:,None]])    
#    
#    coords *= radius
    
    
    
    
##    # This works but I don't understand why
##    for idx,w in enumerate(weights):
##        volume += (w/3)*radius*radius*radius
##        
##    print(volume, (4/3)*np.pi*radius*radius*radius)
    
    # This is correct because the weights already have a factor of 4*np.pi
    # applied so must divide by three and multiply by the radius.
#    volume = np.sum(weights)
#    volume *= (1/3)*radius*radius*radius
    
#    print(volume,analytical_volume)
    
    
    
    
    
#    ## Cap volume test
#    height = radius - 0.5*radius
#    z_value = radius - height
#    
#    lower_idx = []
#    temp_colat = []
#    for idx,row in enumerate(coords):
#        ## Just need to check height in z direction to determine whether the idx
#        ## should be kept or not. 
#        if row[-1] > z_value:
#            lower_idx.append(idx)
#            temp_colat.append(colat[idx])
#            
#    total_colat = np.max(temp_colat) - np.min(temp_colat)
#    colat_volume = 2*np.pi*radius*radius*radius / 3 * (1 - np.cos(total_colat))
#            
#    cap_volume = 0
#    for w in weights[lower_idx]:
#        cap_volume += (w/3)*radius*radius*radius
#        
#    eq_cap_volume = np.pi * height * height / 3 *(3 * radius - height)
#    
##    print(cap_volume, eq_cap_volume, 
##          colat_volume)
#    
#    print("Analytical: {}".format(eq_cap_volume))
#    print("Lebedev Weights: {}".format(cap_volume))
#    print("Ratio: {}".format(analytical_volume * 
#                             len(lower_idx) / coords.shape[0]))
#    print("Colat Diff: {}".format(colat_volume))
    
    
    
#    import trimesh
    
#    trimesh.util.attach_to_log()
    
#    vertices = m.hull.points.ravel()[m.hull.simplices]
#    faces = np.array([[x for x in range(m.hull.simplices.shape[0])]])
#    mesh = trimesh.Trimesh(vertices=vertices,
#                           faces=faces,
#                           process=False)
#    mesh.show()
    
    
#    from mpl_toolkits.mplot3d import Axes3D
#    import matplotlib.pyplot as plt
#    import numpy as np
#    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    
#    ax.plot_trisurf(m.hull.points[:,0], 
#                    m.hull.points[:,1], 
#                    m.hull.points[:,2], 
#                    triangles=m.hull.simplices,
#                    linewidth=0.2, antialiased=True)
    
    
#    grid_struct = m.visualize_grid(m.grid)
#    grid_struct.struct_id = "{}_grid".format(struct_name)
#    temp_dict = {grid_struct.struct_id: grid_struct}
#    write("/Users/ibier/Desktop/Temp", temp_dict, ["geo", "json"],
#          overwrite=True)
#    
#    hull_struct = m.visualize_hull(m.grid)
#    hull_struct.struct_id = "{}_hull".format(struct_name)
#    temp_dict = {hull_struct.struct_id: hull_struct}
#    write("/Users/ibier/Desktop/Temp", temp_dict, ["geo", "json"],
#          overwrite=True)
#    
#    ## Output original
#    temp_dict = {struct.struct_id: struct}
#    write("/Users/ibier/Desktop/Temp", temp_dict, ["geo", "json"],
#          overwrite=True)
    
      
