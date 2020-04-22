# -*- coding: utf-8 -*-



import numpy as np
from ase.data import vdw_radii,atomic_numbers,covalent_radii

from ibslib import Structure
from ibslib.driver import BaseDriver_


all_radii = []
for idx,value in enumerate(vdw_radii):
    if np.isnan(value):
        value = covalent_radii[idx]
    all_radii.append(value)
all_radii = np.array(all_radii)

def equal_axis_aspect(ax):
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    
    xrange = xticks[-1] - xticks[0]
    yrange = yticks[-1] - yticks[0]
    zrange = zticks[-1] - zticks[0]
    max_range = max([xrange,yrange,zrange]) / 2
    
    xmid = np.mean(xticks)
    ymid = np.mean(yticks)
    zmid = np.mean(zticks)
    
    ax.set_xlim(xmid - max_range, xmid + max_range)
    ax.set_ylim(ymid - max_range, ymid + max_range)
    ax.set_zlim(zmid - max_range, zmid + max_range)


class MarchingCubes(BaseDriver_):
    
    def __init__(self, vdw=all_radii, update=True,
                 cache=0.25, spacing=0.25):
        self.vdw = vdw
        self.update = update
        self.struct = None
        self.spacing = spacing
        self.cache = cache
        self.offset_combination_dict = self.create_offset_dict()
        

    def create_offset_dict(self):
        ### Min value for norm is equal to the radius minus the distace 
        ### between the center of the cube and its vertex
        sqrt_3_over_2_spacing = np.sqrt(3) * self.spacing / 2
        
        ## Find all combinations of small values that lead to less than or equal
        ## to the largest value. This is equivalent to finding all grid points 
        ## within a certain radius
        offset_combination_dict = {}
        max_offset_value = np.round(np.max(self.vdw) / self.cache) + 1
        idx_range = np.arange(-max_offset_value , max_offset_value+1)[::-1]
        sort_idx = np.argsort(np.abs(idx_range))
        idx_range = idx_range[sort_idx]
        all_idx = np.array(
                    np.meshgrid(idx_range,idx_range,idx_range)).T.reshape(-1,3)
        all_idx = all_idx.astype(int)
        all_norm = np.linalg.norm(all_idx, axis=-1)
        
        
        
        for value in range(int(max_offset_value+1)):
#            min_norm = value - sqrt_3_over_2_spacing
            min_norm = value
            take_idx = np.where(all_norm < min_norm)[0]
            
            final_idx = all_idx[take_idx]
            offset_combination_dict[value] = final_idx
            
        return offset_combination_dict
    
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
    
    
    def grid_to_volume(self, grid, spacing=0):
        if spacing == 0:
            spacing = self.spacing
        
        x_min = np.min(grid[:,0])
        x_max = np.max(grid[:,0])
        y_min = np.min(grid[:,1])
        y_max = np.max(grid[:,1])
        z_min = np.min(grid[:,2])
        z_max = np.max(grid[:,2])
        
        x_vals = np.arange(x_min-spacing, x_max+spacing, spacing)
        y_vals = np.arange(y_min-spacing, y_max+spacing, spacing)
        z_vals = np.arange(z_min-spacing, z_max+spacing, spacing)
        
#        x, y, z = np.meshgrid(x_vals, y_vals, z_vals)
#        xyz = np.c_[x.flatten(), y.flatten(), z.flatten()]
        
        volume = np.zeros((x_vals.shape[0], 
                       y_vals.shape[0], 
                       z_vals.shape[0]))
    
        offset = np.array([x_min-spacing, 
                       y_min-spacing, 
                       z_min-spacing])
        all_idx = np.round((grid-offset) / spacing)
        
        all_idx = all_idx.astype(int)
        volume[all_idx[:,0], all_idx[:,1], all_idx[:,2]] = 1
        
        return volume
#        fig = plt.figure()
#        ax = fig.gca(projection='3d')
#        
#        ax.scatter(xyz[:,0],
#                xyz[:,1],
#                xyz[:,2])
#        
#        print(xyz.shape)
#        
#        ### Based on indexing, it should be known exactly what vertex to fill
#        ### each point on the grid with. 
#        volume = np.zeros((xyz.shape))
        
    
    
    def struct_to_volume(self, struct=None, spacing=0,
                         float=True ):
        if spacing == 0:
            spacing = self.spacing
            
        if struct == None:
            struct = self.struct
        
        ### Making struct volume from the beginning
        geo = struct.get_geo_array()
        ele = struct.geometry["element"]
        struct_radii = np.array([self.vdw[atomic_numbers[x]] for x in ele])
        
        ## Get min/max for xyz directions
        max_geo = np.max(geo + struct_radii[:,None], axis=0)
        min_geo = np.min(geo - struct_radii[:,None], axis=0)
        
        ## Provide spacing at edges
        max_geo += spacing
        min_geo -= spacing
        
        ## Now generate volume grid
        x_vals = np.arange(min_geo[0], max_geo[0]+spacing, spacing)
        y_vals = np.arange(min_geo[1], max_geo[1]+spacing, spacing)
        z_vals = np.arange(min_geo[2], max_geo[2]+spacing, spacing)
        
        volume = np.zeros((x_vals.shape[0], 
                           y_vals.shape[0], 
                           z_vals.shape[0]))

        for idx,point in enumerate(geo):
            
            ## Index into volume for the center of the atom
            center_idx = np.round((point-min_geo) / spacing).astype(int)
            
            ## Now compute idx to also populate x,y,z directions for given radius
            rad = struct_radii[idx]
            num_idx = np.round(rad / spacing)
            
            offset = int(num_idx)
            
            ## Now, generate off triple offset pairs that add up to the offset to 
            ## complete spherical grid indexing for atom
            
            offset = self.offset_combination_dict[offset]
            final_idx = center_idx + offset
            
            volume[final_idx[:,0], final_idx[:,1], final_idx[:,2]] = 1
            
        ## Volume estimation is now as easy as multiplying the 1 entries in the
        ## grid by the volume of each block
#        print(np.sum(volume)*spacing*spacing*spacing)
#        print(x_vals.shape, y_vals.shape, z_vals.shape)
        
        return volume
    
    
    
    
    

if __name__ == "__main__":
    """
    This is a test of email notifications for push events. 
    
    """
    from ibslib.io import read,write
    from ibslib.molecules.utils import align 
    
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
#    from .marching_cubes_lookup import tri_connectivity,tostring
#    struct_name = "PUDXES02"
    ####    struct_name = "ICIXOH01"
    ####    struct_name = "JAGREP"
    ####    struct_name = "CYACAC"
    ##    struct_name = "SOZBUE01"
    #    
#    struct = read("/Users/ibier/Research/Volume_Estimation/Models/20200217_Model_Test/Dataset/{}.json".format(struct_name))
#    struct = read("/Users/ibier/Research/Volume_Estimation/Datasets/PAHs_MC_Info/TETCEN01.json")
#    struct = read("/Users/ibier/Desktop/Temp/Atom_Test/hydrogen.json")
#    target = 7.24
#    
#    spacing=0.5
#    m = MarchingCubes(spacing=spacing,
#                      cache=spacing)
#        
#    m.spacing = spacing
#    volume = m.struct_to_volume(struct, spacing=spacing)
#    
#    voxel_volume = spacing*spacing*spacing
#    original_volume = np.sum(volume)*voxel_volume
#    
#    x_num,y_num,z_num = volume.shape
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
#    X,Y,Z = np.meshgrid( x_vals, y_vals, z_vals,
#                        indexing="ij")
#    grid_point_reference = np.c_[X.ravel(),
#                                 Y.ravel(),
#                                 Z.ravel()]
#
#    ## Start by projecting down Z direction because this is easiest based on the 
#    ## indexing scheme
#    z_proj = np.arange(0,z_num-1)
#    front_plane_top_left_idx = z_proj
#    front_plane_bot_left_idx = front_plane_top_left_idx + 1
#    
#    ## Have to move 1 in the Y direction which is the same as z_num
#    back_plane_top_left_idx = z_proj + z_num
#    back_plane_bot_left_idx = back_plane_top_left_idx + 1
#    
#    ## Have to move 1 in the X direction which is the same as z_num*y_num 
#    front_plane_top_right_idx = z_proj + y_num*z_num
#    front_plane_bot_right_idx = front_plane_top_right_idx + 1
#    
#    ## Have to move 1 in the y direction which is the same as z_num
#    back_plane_top_right_idx = front_plane_top_right_idx + z_num
#    back_plane_bot_right_idx = back_plane_top_right_idx + 1
#    
#    
#    
#    #### Now project over the Y direction
#    y_proj = np.arange(0,y_num-1)[:,None]*(z_num)
#    front_plane_top_left_idx = front_plane_top_left_idx + y_proj
#    front_plane_bot_left_idx = front_plane_bot_left_idx+ y_proj
#    back_plane_top_left_idx = back_plane_top_left_idx+ y_proj
#    back_plane_bot_left_idx = back_plane_bot_left_idx+ y_proj
#    front_plane_top_right_idx = front_plane_top_right_idx+ y_proj
#    front_plane_bot_right_idx = front_plane_bot_right_idx+ y_proj
#    back_plane_top_right_idx = back_plane_top_right_idx+ y_proj
#    back_plane_bot_right_idx = back_plane_bot_right_idx+ y_proj
#    
#    
#    #### Lastly project in X direction
#    x_proj = np.arange(0,x_num-1)[:,None,None]*(y_num*z_num)
#    front_plane_top_left_idx = front_plane_top_left_idx + x_proj
#    front_plane_bot_left_idx = front_plane_bot_left_idx + x_proj
#    back_plane_top_left_idx = back_plane_top_left_idx + x_proj
#    back_plane_bot_left_idx = back_plane_bot_left_idx + x_proj
#    front_plane_top_right_idx = front_plane_top_right_idx + x_proj
#    front_plane_bot_right_idx = front_plane_bot_right_idx + x_proj
#    back_plane_top_right_idx = back_plane_top_right_idx + x_proj
#    back_plane_bot_right_idx = back_plane_bot_right_idx + x_proj
#    #
#    voxel_idx = np.c_[front_plane_top_left_idx.ravel(),
#                      front_plane_bot_left_idx.ravel(),
#                      back_plane_bot_left_idx.ravel(),
#                      back_plane_top_left_idx.ravel(),
#                      front_plane_top_right_idx.ravel(),
#                      front_plane_bot_right_idx.ravel(),
#                      back_plane_bot_right_idx.ravel(),
#                      back_plane_top_right_idx.ravel(),
#                      ]
#    
#    voxel_mask = np.take(volume, voxel_idx)
#    voxel_sum = np.sum(voxel_mask, axis=-1)
#    voxel_surface_vertex_idx = np.where(np.logical_and(voxel_sum != 0,
#                                         voxel_sum != 8))[0]
#    
#    ## Get only the non-zero points on the surface for visualization
#    surface_vertex_idx = voxel_idx[voxel_surface_vertex_idx][
#                            voxel_mask[voxel_surface_vertex_idx].astype(bool)]
#    surface_vertex = grid_point_reference[surface_vertex_idx]
#    
#    #### Working on surface triangulation
#    
#    ## Get the voxels that correspond to the surface of the molecule
#    surface_voxel = voxel_mask[voxel_surface_vertex_idx].astype(int)
#    ## Get corresponding grid_point_reference idx for each of the surface voxel
#    ## verticies
#    surface_voxel_vert = voxel_idx[voxel_surface_vertex_idx]
#    
#    edge_coords = []
#    all_triangles = []
#    for entry in grid_point_reference[voxel_idx]:
#        temp_edges = compute_edge_sites(entry)
#        edge_coords.append(temp_edges)
#        all_triangles.append([])
#    
#    voxel_coords = []
#    cube_coords = []
#    coords = []
#    triangles = []
#    for idx,entry in enumerate(surface_voxel):
#        
#        ### Get Cartesian Coordinates index
#        temp_ref_idx = surface_voxel_vert[idx]
#        ### Get populated coordinates
#        voxel_coords.append(grid_point_reference[
#                temp_ref_idx[entry.astype(bool)]])
#        
#        ### Get Cart Cube vertex and edges
#        temp_cube_points = grid_point_reference[temp_ref_idx]
#        temp_edges = compute_edge_sites(temp_cube_points)
#        
#        
#        ### Get Cube edges
#        
#        ### Get the tri_idx for this surface voxel
#        triangles_bool = tri_connectivity[tostring(entry)].astype(bool)
#        array_to_mask = np.repeat(np.arange(0,12)[None,:], 
#                            triangles_bool.shape[0], 
#                            axis=0)
#        tri_idx = array_to_mask[triangles_bool].reshape(-1,3)
#        
#        ### Build triangles for grid point reference
#        tri_idx = tri_idx + len(coords)*12
#        
#        ### Save results for plotting
#        cube_coords.append(temp_cube_points)
#        coords.append(temp_edges)
#        triangles.append(tri_idx)
#        
#        edge_coords[voxel_surface_vertex_idx[idx]] = temp_edges
#        all_triangles[voxel_surface_vertex_idx[idx]] = tri_idx
    
#        
#    
#    #### Plot Surface
##        fig = plt.figure(figsize=(10, 10))
##        ax = fig.add_subplot(111, projection='3d')
##        ax.scatter(surface_vertex[:,0],
##                   surface_vertex[:,1],
##                   surface_vertex[:,2],
##                   edgecolor="k")
##        equal_axis_aspect(ax)
##        plt.show()
##        plt.close()
#    
#    
#    #### Voxel Plot
#    fig = plt.figure(figsize=(10, 10))
#    ax = fig.add_subplot(111, projection='3d')
#    test_points = np.vstack(grid_point_reference[voxel_idx[0:10000],:])
#    ax.voxels(volume)
#    ax.view_init(20, 250)
#    equal_axis_aspect(ax)
#    plt.show()
#    plt.close()
#    
#    ##### Surface Tri Plot
#    fig = plt.figure(figsize=(10, 10))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_trisurf(
#               coords[:,0],
#               coords[:,1],
#               coords[:,2],
#               triangles=triangles,
#               cmap=cm.viridis)
#    equal_axis_aspect(ax)
##    ax.view_init(10, 35)
#    plt.show()
#    plt.close()   
    
    
    ############################################################################
    #### Marching Cubes GIF
    ############################################################################
#    ##### Surface Tri Plot
#    stacked_coords = np.vstack(coords)
#    stacked_triangles = np.vstack(triangles)
#    fig = plt.figure(figsize=(10, 10))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_trisurf(
#               stacked_coords[:,0],
#               stacked_coords[:,1],
#               stacked_coords[:,2],
#               triangles=stacked_triangles,
#               cmap=cm.viridis)
#    equal_axis_aspect(ax)
#    lim = ax.get_xlim()
##    ax.view_init(10, 35)
#    plt.show()
#    plt.close()  
#    
#    
#    all_tri_coords = []
#    all_cube_coords = grid_point_reference[voxel_idx]
#    for idx,vertices in enumerate(all_cube_coords):
#        edges = edge_coords[idx]
#        
#        fig = plt.figure(figsize=(10, 10))
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter(
#                   vertices[:,0],
#                   vertices[:,1],
#                   vertices[:,2],
#                   c="k")
#        
#        ax.scatter(
#                   edges[:,0],
#                   edges[:,1],
#                   edges[:,2],
#                   c="tab:orange")
#        
#        ##### Plotting triangles
#        temp_tri = all_triangles[idx]
#        if len(temp_tri) != 0:
#            for temp_tri_entry in temp_tri:
#                tri_coords = stacked_coords[temp_tri_entry]
#                all_tri_coords.append(tri_coords)
#        
#        if len(all_tri_coords) > 0:
#            for temp_tri_coords in all_tri_coords:
#                if len(temp_tri_coords) > 0:
#                    ax.plot_trisurf(
#                           temp_tri_coords[:,0],
#                           temp_tri_coords[:,1],
#                           temp_tri_coords[:,2],
#                           triangles=[[0,1,2]],
#                           cmap=cm.viridis,
#                           vmin=np.min(stacked_coords),
#                           vmax=np.max(stacked_coords))
#        
#        ax.set_xlim(lim)
#        ax.set_ylim(lim)
#        ax.set_zlim(lim)
#        equal_axis_aspect(ax)
#        
#        fig.savefig("/Users/ibier/Research/Volume_Estimation/Documents/20200324_Meeting/images/marching_cubes_gif/march_{}.png".format(str(idx).zfill(4)),
#                    dpi=100)
#        
#    #    ax.view_init(10, 35)
#        plt.show()
#        plt.close()    
        
#        break
    
    
#    
#    
#    #### Let's make volume adjustment
#    surface_voxel = voxel_mask[voxel_surface_vertex_idx]
#    unique_tri_type,counts = np.unique(surface_voxel, return_counts=True,axis=0)
#    
#    voxel_volume = spacing*spacing*spacing
#    original_volume = np.sum(volume)*voxel_volume
#    adjusted_volume = original_volume.copy()
#    for entry,count in zip(unique_tri_type,counts):
#        temp_volume = tri_volume[tostring(entry.astype(int))]
#        temp_volume = temp_volume*count*voxel_volume
#        adjusted_volume += temp_volume
    
    
    ############################################################################
    ###### Making Images For 20200324
    ############################################################################
    struct_name = "PUDXES02"   
    struct = read("/Users/ibier/Research/Volume_Estimation/Models/20200217_Model_Test/Dataset/{}.json".format(struct_name))
#    struct = read("/Users/ibier/Desktop/Temp/Atom_Test/hydrogen.json")
#    spacing = 1
#    spacing_range = np.arange(0.02,0.5,0.005)
#    target_volume = (4/3)*np.pi*1.2*1.2*1.2
#    volume_list = []
    spacing=0.5
    m = MarchingCubes(spacing=spacing,
                      cache=spacing)
#    for spacing in spacing_range:
#        print(spacing)
#        m.spacing = spacing
#        volume = m.struct_to_volume(struct)
#        volume_list.append(np.sum(volume)*spacing*spacing*spacing)
    
#    m.spacing = 0.1
    volume = m.struct_to_volume(struct)
#    volume_list.append(np.sum(volume)*spacing*spacing*spacing)
        
    
#    fontsize=16
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(spacing_range,
#            volume_list,
#            linewidth=2,
#            c="tab:blue")
#    ax.axhline(target_volume,
#               c="tab:red",
#               linewidth=2)
#    ax.set_ylabel("Calculated Area",
#                  fontsize=fontsize)
    
#    fig.savefig("Voxel_Convergence.pdf")
    
    
    
    
    ############################################################################
    #### skimage marching cubes
    ############################################################################
    
    
#    from skimage.measure import marching_cubes_lewiner
#    verts, faces, normals, values = marching_cubes_lewiner(volume, level=0,
#                                            spacing=(spacing,spacing,spacing))
    
#    print(verts.shape, faces.shape)
#    
#    fig = plt.figure(figsize=(10, 10))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_trisurf(verts[:,0],
#                    verts[:,1],
#                    verts[:,2], 
#                    triangles=faces,
#                    cmap=cm.viridis,
#                    linewidth=0.2, antialiased=True)
#    equal_axis_aspect(ax)
#    fig.savefig("PUDXES02_Marching_Cube_Surface.pdf")
#    plt.show()
#    plt.close()
        
        
    
    ###### Need to debug tri surfaces for marching cubes
    ###### Or something else... IDK what's wrong
    ###### Or maybe tri is right but the coords used is wrong
#    fig = plt.figure(figsize=(10, 10))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(
#                coords[:,0],
#               coords[:,1],
#               coords[:,2],)
#    plt.show()
#    plt.close()
#    
#    
#    for idx in range(8):
#        cube_start = idx*8
#        cube_end = (idx+1)*8
#        fig = plt.figure(figsize=(10, 10))
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter(
#            cube_coords[:,0][cube_start:cube_end],
#            cube_coords[:,1][cube_start:cube_end],
#            cube_coords[:,2][cube_start:cube_end],)
#    
#        edge_start = idx*12
#        edge_end = (idx+1)*12
#        
#        ax.scatter(
#            coords[:,0][edge_start:edge_end],
#            coords[:,1][edge_start:edge_end],
#            coords[:,2][edge_start:edge_end],
#            )
#        
#        ax.plot_trisurf(
#            coords[:,0][triangles[idx]],
#            coords[:,1][triangles[idx]],
#            coords[:,2][triangles[idx]])
#        
#        ax.scatter(voxel_coords[0][0],
#               voxel_coords[0][1],
#               voxel_coords[0][2],
#               s=100)
#        
#        equal_axis_aspect(ax)
#        plt.show()
#        plt.close()
        
        
        
#    fig = plt.figure(figsize=(10, 10))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(
#            cube_coords[:,0][0:8],
#            cube_coords[:,1][0:8],
#            cube_coords[:,2][0:8],)
##    ax.scatter(
##            cube_coords[:,0][16:24],
##            cube_coords[:,1][16:24],
##            cube_coords[:,2][16:24],)
##    ax.scatter(
##            cube_coords[:,0][8:16],
##            cube_coords[:,1][8:16],
##            cube_coords[:,2][8:16],)
#    ax.scatter(
#            coords[:,0][0:12],
#            coords[:,1][0:12],
#            coords[:,2][0:12],
#            )
#    ax.plot_trisurf(
#            coords[:,0][triangles[0]],
#            coords[:,1][triangles[0]],
#            coords[:,2][triangles[0]])
##    ax.plot_trisurf(
##            coords[:,0][triangles[1]],
##            coords[:,1][triangles[1]],
##            coords[:,2][triangles[1]])
##    ax.plot_trisurf(
##            coords[:,0][triangles[2]],
##            coords[:,1][triangles[2]],
##            coords[:,2][triangles[2]])
#    ax.scatter(voxel_coords[0][0],
#               voxel_coords[0][1],
#               voxel_coords[0][2],
#               s=100)
##    ax.plot_trisurf(
##            coords[:,0][triangles[1]],
##            coords[:,1][triangles[1]],
##            coords[:,2][triangles[1]])
#    equal_axis_aspect(ax)
#    plt.show()
#    plt.close()
##    for tri in triangles:
##        fig = plt.figure(figsize=(10, 10))
##        ax = fig.add_subplot(111, projection='3d')
##        ax.plot_trisurf(
##               coords[:,0][tri],
##               coords[:,1][tri],
##               coords[:,2][tri],
##               triangles=[0,1,2],
##               cmap=cm.viridis)
##        plt.show()
##        plt.close()    
#    
#        
#    from skimage.measure import marching_cubes_lewiner, find_contours
#    from skimage.segmentation import flood_fill
#    
#    def fill_holes(plane):
#        not_plane = np.logical_not(plane).astype(int)
#        filled = flood_fill(not_plane, (0,0), 0)
#        filled = np.logical_or(plane,filled)
#
#        return filled
#    
#    
#    ## Testing modifications
#    x_vals = np.arange(0,volume.shape[0]+1)
#    y_vals = np.arange(0,volume.shape[1]+1)
#    X,Y = np.meshgrid(x_vals, y_vals)
#    for idx,xy_plane in enumerate(volume.T[10:]):
#        if np.sum(xy_plane) == 0:
#            continue
#        
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
##        cp = ax.contourf(fill_holes(xy_plane))
##        cp = ax.contourf(X,Y,xy_plane)
#        
#        
#        filled = fill_holes(xy_plane)
#        cp = ax.contourf(filled)
#        contours = find_contours(filled, 0.8)
#        curvature = []
#        
##        for contour in contours:   
##            ## Last point is repetition of first
##            contour = contour[:-1]
##            
##            ax.plot(contour[:,1], contour[:,0], linewidth=2)
##            center = np.mean(contour, axis=0)
##            ax.scatter(center[1], center[0])
#            
#            
#            
#            
##            print(contour.shape)
#        
##        center = np.array([int(xy_plane.shape[0] / 2),
##                           int(xy_plane.shape[1] / 2)])
##                
##                
##        ax.scatter(center[0], center[1], s=100, c="tab:red")
#        
##        fig.colorbar(cp)
#        plt.show()
#        plt.close()
#        
#        fig.savefig("PUDXES02_Filled_Holes.pdf")
#        
#        
#        break
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
##        cp = ax.contourf(fill_holes(xy_plane))
##        cp = ax.contourf(X,Y,xy_plane)
#    
#    
##    filled = fill_holes(xy_plane)
#    cp = ax.contourf(xy_plane)
#    contours = find_contours(filled, 0.8)
#    curvature = []
#    
#    fig.savefig("PUDXES02_Plane.pdf")
    