# -*- coding: utf-8 -*-

"""

Testing algorithms in 2D. Will first start with simple circle then combine 
with multiple circles at different locations. Will implement:
    1. Grid pint interpolation for smoother surfaces
    2. How to perform smoothing with multiple circles. 
    3. Tangental relaxation. 
    
Then, generalize methods for interpolation and relaxation to 3D. 

"""

import numpy as np
import matplotlib.pyplot as plt

"""

Lookup Table

"""
def tostring(array):
    """
    1D array to string
    """
    return ",".join([str(x) for x in array])


def compute_edges(square_vertex):
    edges = []
    edges.append((square_vertex[-1] + square_vertex[0]) / 2)
    edges.append((square_vertex[1] + square_vertex[0]) / 2)
    edges.append((square_vertex[2] + square_vertex[1]) / 2)
    edges.append((square_vertex[2] + square_vertex[3]) / 2)
    
    return np.array(edges)


def test_vertices():
    return np.array([
            [0,0],
            [0,1],
            [1,1],
            [1,0]
            ])
    
def test_edges():
    return np.array([
            [0.5,0],
            [0,0.5],
            [0.5,1],
            [1,0.5]
            ])
    
#### Each entry of tri lookup is a 2D array. This array describes all lines
#### That should be drawn for the surface between the edges of the square
#### http://users.polytech.unice.fr/~lingrand/MarchingCubes/algo.html
tri_lookup = {}

#### Describes how to compute area 
tri_area_lookup = {}

## 1. Empty
tri_lookup[tostring(np.array([0,0,0,0]))] = np.zeros((1,4))
tri_area_lookup[tostring(np.array([0,0,0,0]))] = lambda edge,vertex: 0

## 2. Tri 0
tri_lookup[tostring(np.array([1,0,0,0]))] = np.array([[1,1,0,0]])
tri_area_lookup[tostring(np.array([1,0,0,0]))] = lambda edge,vertex: \
    (edge[1][1] - vertex[0][1])*(edge[0][0] - vertex[0][0])*0.5

## 3. Tri 3
tri_lookup[tostring(np.array([0,0,0,1]))] = np.array([[1,0,0,1]])
tri_area_lookup[tostring(np.array([0,0,0,1]))] = lambda edge,vertex: \
    (edge[3][1] - vertex[3][1])*(vertex[3][0] - edge[0][0])*0.5

## 4. Horizontal Plane
tri_lookup[tostring(np.array([1,0,0,1]))] = np.array([[0,1,0,1]])
tri_area_lookup[tostring(np.array([1,0,0,1]))] = lambda edge,vertex: \
    ((edge[1][1] - vertex[0][1])+(edge[3][1] - vertex[3][1]))*0.5*\
    (vertex[3][0] - vertex[0][0])

## 5. Tri 2
tri_lookup[tostring(np.array([0,0,1,0]))] = np.array([[0,0,1,1]])
tri_area_lookup[tostring(np.array([0,0,1,0]))] = lambda edge,vertex: \
    (vertex[2][1] - edge[3][1])*(vertex[2][0] - edge[2][0])*0.5

## 6. Double Tri 0,2
tri_lookup[tostring(np.array([1,0,1,0]))] = np.array([[1,1,0,0],
                                                      [0,0,1,1]])
tri_area_lookup[tostring(np.array([1,0,1,0]))] = lambda edge,vertex: \
    0
    
## 7. Vertical Plane 2,3
tri_lookup[tostring(np.array([0,0,1,1]))] = np.array([[1,0,1,0]]) 
tri_area_lookup[tostring(np.array([0,0,1,1]))] = lambda edge,vertex: \
    ((vertex[2][0] - edge[2][0])+(vertex[3][0] - edge[0][0]))*0.5*\
    (vertex[2][1] - vertex[3][1])
    
    
## 8. Triple vertex tri 0,2,3
tri_lookup[tostring(np.array([1,0,1,1]))] = np.array([[0,1,1,0]]) 
tri_area_lookup[tostring(np.array([1,0,1,1]))] = lambda edge,vertex: \
    (vertex[1][1] - vertex[0][1])*(vertex[3][0] - vertex[0][0]) - \
    (vertex[1][1] - edge[1][1])*(edge[2][0] - vertex[1][0])*0.5

## 9. Tri 1
tri_lookup[tostring(np.array([0,1,0,0]))] = np.array([[0,1,1,0]]) 
tri_area_lookup[tostring(np.array([0,1,0,0]))] = lambda edge,vertex: \
    (vertex[1][1] - edge[1][1])*(edge[2][0] - vertex[1][0])*0.5


## 10. Vertical Plane 0,1
tri_lookup[tostring(np.array([1,1,0,0]))] = np.array([[1,0,1,0]]) 
tri_area_lookup[tostring(np.array([1,1,0,0]))] = lambda edge,vertex: \
    ((edge[2][0] - vertex[1][0])+(edge[0][0] - vertex[0][0]))*0.5*\
    (vertex[1][1] - vertex[0][1])


## 11. Double Tri 1,3
tri_lookup[tostring(np.array([0,1,0,1]))] = np.array([[1,1,0,0],
                                                      [0,0,1,1]])
tri_area_lookup[tostring(np.array([0,1,0,1]))] = lambda edge,vertex: \
    0
    
## 12. Triple vertex tri 0,1,3
tri_lookup[tostring(np.array([1,1,0,1]))] = np.array([[0,0,1,1]]) 
tri_area_lookup[tostring(np.array([1,1,0,1]))] = lambda edge,vertex: \
    (vertex[1][1] - vertex[0][1])*(vertex[3][0] - vertex[0][0]) - \
    (vertex[2][1] - edge[3][1])*(vertex[2][0] - edge[2][0])*0.5

## 13. Horizontal plane 1,2
tri_lookup[tostring(np.array([0,1,1,0]))] = np.array([[0,1,0,1]]) 
tri_area_lookup[tostring(np.array([0,1,1,0]))] = lambda edge,vertex: \
    ((vertex[1][1] - edge[1][1])+(vertex[2][1] - edge[3][1]))*0.5*\
    (vertex[2][0] - vertex[1][0])

## 14. Triple vertex tri 0,1,2
tri_lookup[tostring(np.array([1,1,1,0]))] = np.array([[1,0,0,1]]) 
tri_area_lookup[tostring(np.array([1,1,1,0]))] = lambda edge,vertex: \
    (vertex[1][1] - vertex[0][1])*(vertex[3][0] - vertex[0][0]) - \
    (vertex[3][0] - edge[0][0])*(edge[3][1] - vertex[3][1])*0.5

## 15. Triple vertex tri 1,2,3
tri_lookup[tostring(np.array([0,1,1,1]))] = np.array([[1,1,0,0]]) 
tri_area_lookup[tostring(np.array([0,1,1,1]))] = lambda edge,vertex: \
    (vertex[1][1] - vertex[0][1])*(vertex[3][0] - vertex[0][0]) - \
    (edge[0][0] - vertex[0][0])*(edge[1][1] - vertex[0][1])*0.5

## 16. All vertex
tri_lookup[tostring(np.array([1,1,1,1]))] = np.array([[0,0,0,0]]) 
tri_area_lookup[tostring(np.array([1,1,1,1]))] = lambda edge,vertex: \
    (vertex[1][1] - vertex[0][1])*(vertex[3][0] - vertex[0][0])


for key,value in tri_lookup.items():
    tri_lookup[key] = tri_lookup[key].astype(bool)



"""

Gadgets 

"""

def equal_axis_aspect(ax):
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    
    xrange = xticks[-1] - xticks[0]
    yrange = yticks[-1] - yticks[0]
    max_range = max([xrange,yrange]) / 2
    
    xmid = np.mean(xticks)
    ymid = np.mean(yticks)
    
    ax.set_xlim(xmid - max_range, xmid + max_range)
    ax.set_ylim(ymid - max_range, ymid + max_range)
    
def plot_grid(coords, c="tab:blue", 
              edgecolor=None, ax=None):
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    ax.scatter(coords[:,0], 
               coords[:,1],
               facecolor=c,
               edgecolor=edgecolor)
    
#    equal_axis_aspect(ax)
    plt.gca().set_aspect('equal', adjustable='box')


def plot_circle(radius, 
                center=[0,0], 
                c="tab:red",
                linewidth=2, 
                ax=None):
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    spacing=0.001
    x_coord = np.arange(-radius,radius+spacing,spacing) + center[0]
    y_coord = np.sqrt(np.abs(np.square(radius) - np.square(x_coord-center[0])))+\
                center[1]
    y_coord_neg = -np.sqrt(np.abs(np.square(radius) - np.square(x_coord-center[0])))\
                 + center[1]
    coords = np.c_[x_coord,
                   y_coord]
    neg_coords = np.c_[x_coord,
                       y_coord_neg]
    
    ax.plot(coords[:,0], 
            coords[:,1],
            c=c,
            linewidth=linewidth)
    ax.plot(neg_coords[:,0], 
            neg_coords[:,1],
            c="tab:purple",
            linewidth=linewidth)
    
#    equal_axis_aspect(ax)
    plt.gca().set_aspect('equal', adjustable='box')
    

def get_grid_area(radius, spacing, center=0):
    radius_point = np.round(radius / spacing)
    #radius_point = np.floor((radius - body_diag) / spacing)
    radius_point = int(radius_point)
    
    ### Now fill in all grid points to the radius 
    idx_range = np.arange(-radius_point , radius_point+1)[::-1]
    sort_idx = np.argsort(np.abs(idx_range))
    idx_range = idx_range[sort_idx]
    
    all_idx = np.array(np.meshgrid(idx_range,
                                   idx_range)).T.reshape(-1,2)
    
    all_idx = all_idx.astype(int)
    all_norm = np.linalg.norm(all_idx*spacing, axis=-1)
    take_idx = np.where(all_norm < radius)[0]
    final_idx = all_idx[take_idx]
    
    final_idx_grid_points = final_idx * spacing
    
    plot_grid(final_idx_grid_points, 
              c="tab:green",
              ax=ax)
    
    ### Each grid point counts for the area around it using a vernoi construction
    grid_area = final_idx_grid_points.shape[0]*spacing*spacing
    
    return grid_area


def plot_march(coords, squares, draw_edges=[], n=0):
    if n <= 0:
        n = squares.shape[0]
    
    for num,entry in enumerate(squares):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_grid(coords, ax=ax)
        square_coords = coords[entry]
        ax.scatter(square_coords[:,0],
                   square_coords[:,1],
                   c=["tab:red", "tab:green", "tab:purple", "tab:cyan"],
                   edgecolor="k",)
        
        plt.show()
        plt.close()
        
        if num >= n:
            break
    

"""

Let's make a class to handle all of this and generalize

"""


class Marching2DCircles():
    
    def __init__(self, radii=[1.2], centers=np.array([[0,0]]), spacing=0.1):
        if type(centers) != np.array:
            centers = np.array(centers)
        if centers.shape[1] != 2:
            raise Exception("Centers need to be X,Y positions")
        
        if len(radii) != centers.shape[0]:
            raise Exception("Radii and centers must have the same "+
                            "number of entries.")
        
        self.radii = radii
        self.centers = centers
        self.spacing = spacing
        
        ### Adjust centers based on spacing so that center lies at grid point
        temp_centered_on_grid = []
        for idx,center in enumerate(self.centers):
            centered_on_grid = np.round((self.centers[idx]) / spacing)*spacing
            temp_centered_on_grid.append(centered_on_grid)
        self.centers = np.vstack(temp_centered_on_grid)
            
        self.limits = self.get_limits()
        ## Coords defines all Cartesian positions of grid
        self.coords = self.generate_coords()
        
    
    def get_limits(self):
        """
        Gets limits that are exactly divisible by spacing
        """
        min_pos = []
        for idx,radius in enumerate(self.radii):
            temp_pos = self.centers[idx] - radius - self.spacing
            temp_pos = (temp_pos / self.spacing - 1).astype(int)*self.spacing
            min_pos.append(temp_pos)
            
        max_pos = []
        for idx,radius in enumerate(self.radii):
            temp_pos = self.centers[idx] + radius + self.spacing
            temp_pos = (temp_pos / self.spacing + 1).astype(int)*self.spacing
            max_pos.append(temp_pos)
            
        return np.min(min_pos,axis=0),np.max(max_pos,axis=0)
    
    
    def generate_coords(self, limits=[], spacing=0):
        if spacing == 0:
            spacing = self.spacing
        if len(limits) == 0:
            limits = self.limits
        
        exact_middle = np.mean(limits, axis=0)
        middle_mult = exact_middle / spacing
        spacing_middle = middle_mult.astype(int)*spacing
        
        ### Calculate number in each direction to use linspace because 
        ### np.arange is susceptable to numerical error using float point spacing
        x_pos_num = np.round(((limits[1][0]) - spacing_middle[0]) / spacing).astype(int)
        x_neg_num = np.round((spacing_middle[0] - limits[0][0]) / spacing).astype(int)
        y_pos_num = np.round((limits[1][1] - spacing_middle[1]) / spacing).astype(int)
        y_neg_num = np.round((spacing_middle[1] - limits[0][1]) / spacing).astype(int)
        
        x_grid_pos = np.linspace(spacing_middle[0],
                               limits[1][0],
                               x_pos_num+1)
        x_grid_neg = np.linspace(limits[0][0], 
                               spacing_middle[0]-spacing, 
                               x_neg_num)
        x_grid = np.hstack([x_grid_neg,x_grid_pos])
        
        y_grid_pos = np.linspace(spacing_middle[1],
                               limits[1][1],
                               y_pos_num+1)
        y_grid_neg = np.linspace(limits[0][1], 
                               spacing_middle[1]-spacing, 
                               y_neg_num)
        y_grid = np.hstack([y_grid_neg,y_grid_pos])
        
        X,Y = np.meshgrid(x_grid,
                          y_grid, 
                          indexing="ij")
        
        coords = np.c_[X.ravel(), 
                       Y.ravel()]
        
        return coords
    
    
    def generate_region(self, 
                        limits=[], 
                        spacing=0):
        """
        Generatess the mapping for every point in the grid being used. Then,
        fills the points based on the radii and centers.
        
        """
        if spacing == 0:
            spacing = self.spacing
        if len(limits) == 0:
            limits = self.limits
        
        exact_middle = np.mean(limits, axis=0)
        middle_mult = exact_middle / spacing
        spacing_middle = middle_mult.astype(int)*spacing
        
        ### Calculate number in each direction to use linspace because 
        ### np.arange is susceptable to numerical error using float point spacing
        x_pos_num = np.round(((limits[1][0]) - spacing_middle[0]) / spacing).astype(int)
        x_neg_num = np.round((spacing_middle[0] - limits[0][0]) / spacing).astype(int)
        y_pos_num = np.round((limits[1][1] - spacing_middle[1]) / spacing).astype(int)
        y_neg_num = np.round((spacing_middle[1] - limits[0][1]) / spacing).astype(int)
        
        x_grid_pos = np.linspace(spacing_middle[0],
                               limits[1][0],
                               x_pos_num+1)
        x_grid_neg = np.linspace(limits[0][0], 
                               spacing_middle[0]-spacing, 
                               x_neg_num)
        x_grid = np.hstack([x_grid_neg,x_grid_pos])
        
        y_grid_pos = np.linspace(spacing_middle[1],
                               limits[1][1],
                               y_pos_num+1)
        y_grid_neg = np.linspace(limits[0][1], 
                               spacing_middle[1]-spacing, 
                               y_neg_num)
        y_grid = np.hstack([y_grid_neg,y_grid_pos])
        
        X,Y = np.meshgrid(x_grid,
                          y_grid, 
                          indexing="ij")
        
        min_loc = np.array([x_grid[0], y_grid[0]])
        
        grid_region = np.zeros(X.shape)
        
        grid_coords = []
        for idx,radius in enumerate(self.radii):
            radius_point = np.round(radius / spacing)
            radius_point = int(radius_point)
            
            ### Fill in all grid points to the radius 
            idx_range = np.arange(-radius_point , radius_point+1)[::-1]
            sort_idx = np.argsort(np.abs(idx_range))
            idx_range = idx_range[sort_idx]
            
            all_idx = np.array(np.meshgrid(idx_range,
                                           idx_range,
                                           )).T.reshape(-1,2)
            
            all_idx = all_idx.astype(int)
            all_norm = np.linalg.norm(all_idx*spacing, axis=-1)
            take_idx = np.where(all_norm <= radius)[0]
            final_idx = all_idx[take_idx]
            
            ### Using idx, adjust to translate into correct cartesian 
            ### coordinates
            temp_grid_coords = final_idx*spacing
            temp_grid_coords = temp_grid_coords+self.centers[idx]-min_loc            
            grid_region_idx = np.round((temp_grid_coords) / spacing)
            grid_region_idx = grid_region_idx.astype(int)
            
            ### Now map into grid region for these indices
            grid_region[grid_region_idx[:,0], grid_region_idx[:,1]] = 1
            
            grid_coords.append(temp_grid_coords)
            
#        grid_coords = np.vstack(grid_coords)
            
        return grid_region,grid_coords
    
    
    def basic_volume(self, grid):
        return np.sum(grid)*self.spacing*self.spacing
    
    
    def basic_marching_squares(self, grid_region, coords):
        x_num = grid_region.shape[1]
        y_num = grid_region.shape[0]
        
        ## Working but IDK why 
        ### Start with X values because they change by value of 1 for indexing
        x_proj = np.arange(0, x_num-1, 1)
        bot_left = x_proj
        
        ## Move in Y direction which is plus the length of x_vals
        top_left = bot_left + x_num
        
        ## Move in X direction which is just plus 1
        bot_right = x_proj + 1
        top_right = top_left + 1
        
        
        ### Now, project along the Y direction
        y_proj = np.arange(0, y_num-1, 1) * x_num
        
        bot_left = bot_left + y_proj[:,None]
        top_left = top_left + y_proj[:,None]
        bot_right = bot_right + y_proj[:,None]
        top_right = top_right + y_proj[:,None]
        
        marching_square_idx = np.c_[
                bot_left.T.ravel(),
                bot_right.T.ravel(),
                top_right.T.ravel(),
                top_left.T.ravel(),]
        
        masked_grid = np.take(grid_region, marching_square_idx)
        draw_edges = []
        draw_vertices = []
        area_arguments = []
        calc_area = 0
        for idx,row in enumerate(masked_grid):
            row_string = tostring(row.astype(int))
            
            row_coords = coords[marching_square_idx[idx]]
            temp_edges = compute_edges(row_coords)
            
            connect_edges_bool = tri_lookup[row_string]
        
            temp_adjusted_area = tri_area_lookup[tostring(row.astype(int))](
                    temp_edges, row_coords)
            
            area_arguments.append((
                    tostring(row.astype(int)),
                    temp_edges, 
                    row_coords))
            
            
            calc_area += temp_adjusted_area
            
            ## For making marching gif
            temp_draw_edges = []
            for temp_mask in connect_edges_bool:  
                temp_masked_edges = temp_edges[temp_mask]
                temp_draw_edges.append(temp_masked_edges)
            draw_edges.append(temp_draw_edges)
            
            draw_vertices.append(row_coords)
            
        return masked_grid,calc_area,draw_vertices,draw_edges,area_arguments
                
    
    def marching_squares(self, grid_region, coords):
        x_num = grid_region.shape[0]
        y_num = grid_region.shape[1]
        
        
        ### Start with Y values because they change by value of 1 for indexing
        y_proj = np.arange(0, y_num-1, 1)
        bot_left = y_proj
        
        # Move in Y direction which is plus 1
        top_left = bot_left + 1
        
        ## Move in X direction which is plus y_num
        bot_right = bot_left + y_num
        top_right = bot_right + 1
        
        
        ### Now, project along the Y direction
        x_proj = np.arange(0, x_num-1, 1) * y_num
        
        bot_left = bot_left + x_proj[:,None]
        top_left = top_left + x_proj[:,None]
        bot_right = bot_right + x_proj[:,None]
        top_right = top_right + x_proj[:,None]
        
        marching_square_idx = np.c_[
                bot_left.ravel(),
                top_left.ravel(),
                top_right.ravel(),
                bot_right.ravel(),
                ]
        
        masked_grid = np.take(grid_region, marching_square_idx)
        
        
        edges_for_region = []
        for row in marching_square_idx:
            row_coords = coords[row]
            edges_for_region.append(compute_edges(row_coords))
        
        masked_grid_sum = np.sum(masked_grid, axis=-1)
        surface_idx = np.where(np.logical_and(masked_grid_sum != 0,
                                              masked_grid_sum != 4))[0]
        
        ### Create hand crafted edge values for surface voxels by projecting 
        ### the edge onto the surface of each circle and then use the circle 
        ### surface that is within the spacing and has a larger distance 
        ### because this will be the point on a potentially overlapping surface
        for surface_iter_idx,temp_surface_idx in enumerate(surface_idx):
            square_idx = marching_square_idx[temp_surface_idx]
            square_coords = coords[square_idx]
            temp_edges = compute_edges(square_coords)
            
            ## Now project edge onto the surface of the sphere
            for edge_idx,point in enumerate(temp_edges):
                ## Store where the point is negative for laters
                mult = np.ones(2)
                mult_idx = np.where(point < 0)[0]
                mult[mult_idx] = -1
                
                ## Project onto surface of each circle present
                temp_projected_point_list = []
                for r_idx,radius in enumerate(self.radii):
                    projected_point = self._proj_edge_2(point, edge_idx, 
                                        radius, r_idx, square_coords)
                    temp_projected_point_list.append(projected_point)
                
                ### Decide which radius to use for this edge projection
                ### The one to use is the one that leads to the largest area
                ### As long as the edge has actually moved
                temp_area_list = []
                active_vertices = masked_grid[temp_surface_idx]
                for entry in temp_projected_point_list:
                    if np.linalg.norm(entry - temp_edges[edge_idx]) < 1e-8:
                        temp_area_list.append(-1)
                        continue
                    
                    test_temp_edges = temp_edges.copy()
                    test_temp_edges[edge_idx] = entry
                    temp_area = tri_area_lookup[tostring(active_vertices.astype(int))](
                                                test_temp_edges, 
                                                square_coords)
                    temp_area_list.append(temp_area)
    
                choice_idx = np.argmax(temp_area_list)
                projected_point = temp_projected_point_list[choice_idx]
                    
                ### Save in edges_for_region
                edges_for_region[temp_surface_idx][edge_idx] = projected_point
                
                
        
        draw_edges = []
        draw_vertices = []
        area_arguments = []
        area_values = []
        calc_area = 0
        for idx,row in enumerate(masked_grid):
            row_string = tostring(row.astype(int))
            
            row_coords = coords[marching_square_idx[idx]]
            ## Use hand crafted edges
            temp_edges = edges_for_region[idx]
            
            connect_edges_bool = tri_lookup[row_string]
        
            temp_adjusted_area = tri_area_lookup[tostring(row.astype(int))](
                    temp_edges, row_coords)
            
            area_arguments.append((
                    tostring(row.astype(int)),
                    temp_edges, 
                    row_coords))
            area_values.append(temp_adjusted_area)
            
            calc_area += temp_adjusted_area
            
            ## For making marching gif
            temp_draw_edges = []
            for temp_mask in connect_edges_bool:  
                temp_masked_edges = temp_edges[temp_mask]
                temp_draw_edges.append(temp_masked_edges)
            draw_edges.append(temp_draw_edges)
            
            draw_vertices.append(row_coords)
            
            
        return masked_grid,marching_square_idx,calc_area,draw_vertices,draw_edges,area_arguments
    
    
    def _proj_edge(self, edge, edge_idx, radius, r_idx, square_coords):
        temp_center = self.centers[r_idx]
        x2_proj = radius*radius - np.square(edge[1] - 
                                   temp_center[1])
        y2_proj = radius*radius - np.square(edge[0] - 
                                   temp_center[0])
        
        if x2_proj < 0:
            proj_x = np.sqrt(-x2_proj)
        else:
            proj_x = np.sqrt(x2_proj)
        
        if y2_proj < 0:
            proj_y = np.sqrt(-y2_proj)
        else:
            proj_y = np.sqrt(y2_proj)
        
        #### Check if should use plus or minus sqrt
        if abs(-edge[0] - proj_x + temp_center[0]) < \
           abs(-edge[0] + proj_x + temp_center[0]):
               proj_x = proj_x * -1
        if abs(-edge[1] - proj_y + temp_center[1]) < \
           abs(-edge[1] + proj_y + temp_center[1]):
               proj_y = proj_y * -1
            
        proj_x = proj_x + temp_center[0]
        proj_y = proj_y + temp_center[1]
        
        ### Now use sign of the point
        proj = np.array([proj_x,proj_y])
        
        ## Now decide how to move the edge. If the projection is 
        ## within the grid spacing, then it's a valid projection
        diff = np.abs(edge - proj)
        temp_take_proj_idx = np.where(diff < mc.spacing)[0]
        projected_point = edge.copy()
        projected_point[temp_take_proj_idx] = proj[temp_take_proj_idx]
        
        ### Edges point movement only has one degree of freedom
        if edge_idx == 0:
            projected_point[1] = edge[1]
        elif edge_idx == 1:
            projected_point[0] = edge[0]
        elif edge_idx == 2:
            projected_point[1] = edge[1]
        elif edge_idx == 3:
            projected_point[0] = edge[0]
            
        ### Have to make sure that point has not left square
        if edge_idx == 0:
            if projected_point[0] < square_coords[0][0]:
                projected_point[0] = edge[0]
            elif projected_point[0] > square_coords[3][0]:
                projected_point[0] = edge[0]
        elif edge_idx == 1:
            if projected_point[1] > square_coords[1][1]:
                projected_point[1] = edge[1]
            elif projected_point[1] < square_coords[0][1]:
                projected_point[1] = edge[1]
        if edge_idx == 2:
            if projected_point[0] < square_coords[1][0]:
                projected_point[0] = edge[0]
            elif projected_point[0] > square_coords[2][0]:
                projected_point[0] = edge[0]
        if edge_idx == 3:
            if projected_point[1] > square_coords[2][1]:
                projected_point[1] = edge[1]
            elif projected_point[1] < square_coords[3][1]:
                projected_point[1] = edge[1]
        
        return projected_point
    
    
    def _proj_edge_2(self, edge, edge_idx, radius, r_idx, square_coords):
        """
        Algorithm is as follows:
        0. Decide if algorithm should be considered for this edge position and 
           circle radius/center pair. If the edge is too far from the center,
           then the projection cannot be valid so the default value is returned. 
        1. Based on edge idx, decide the direction the point can be moved
        2. Perform projection using equation of a circle with radius and center
           by first compute the value under the sqrt, the determining if the pos
           or neg of the sqrt should be used. 
        3. Check if projection has left the square and reject if it has
        4. Otherwise, projection is valid
        """
        center = self.centers[r_idx]
        
        ### Maximum distance away and edge should be from the center to even 
        ### be considered to go through the following algorithm is the radius
        ### plus a correction.
        basic_distance = np.linalg.norm(edge - center)
        if basic_distance > (radius + spacing / 2):
            return edge
        
        ### Project along correct direction
        if edge_idx == 0:
            ## X
            proj2 = radius*radius - np.square(edge[1] - center[1])
        elif edge_idx == 1:
            ## Y
            proj2 = radius*radius - np.square(edge[0] - center[0])
        if edge_idx == 2:
            ## X
            proj2 = radius*radius - np.square(edge[1] - center[1])
        if edge_idx == 3:
            ## Y
            proj2 = radius*radius - np.square(edge[0] - center[0])
        
        if proj2 < 0:
            proj2 = proj2*-1
        
        proj = np.sqrt(proj2)
        
        ### Get compare idx to choose pos or neg sqrt
        if edge_idx == 0:
            ## X
            compare_idx = 0
        elif edge_idx == 1:
            ## Y
            compare_idx = 1
        if edge_idx == 2:
            ## X
            compare_idx = 0
        if edge_idx == 3:
            ## Y
            compare_idx = 1
        
        ### Account for numerical error in assignment of plus or minus sqrt
        if np.abs(edge[compare_idx] - center[compare_idx]) <= 1e-5:
            if compare_idx == 0:
                other_compare_idx = 1
            else:
                other_compare_idx = 0
            ## If on left side, use neg, if on right side use pos
            if (edge[other_compare_idx] - center[other_compare_idx]) < 0:
                proj = -proj
        ### Check if should use plus or minus sqrt
        elif (edge[compare_idx] - center[compare_idx]) <= 0:
            proj = -proj
        
        ### Now add centering
        if edge_idx == 0:
            ## X
            proj += center[0]
        elif edge_idx == 1:
            ## Y
            proj += center[1]
        if edge_idx == 2:
            ## X
            proj += center[0]
        if edge_idx == 3:
            ## Y
            proj += center[1]
        
        
        ## Check if proj moves point by greater than 2/spacing
        limit = self.spacing / 2 + 1e-6
        temp_proj_edge = edge.copy()
        if edge_idx == 0:
            ## X
            temp_proj_edge[0] = proj
        elif edge_idx == 1:
            ## Y
            temp_proj_edge[1] = proj
        if edge_idx == 2:
            ## X
            temp_proj_edge[0] = proj
        if edge_idx == 3:
            ## Y
            temp_proj_edge[1] = proj
    
        distance = np.linalg.norm(edge - temp_proj_edge)
        
        if distance > limit:
            return edge
        else:
            return temp_proj_edge
    
    
    def plot_result(self, draw_edges):
        pass


if __name__ == "__main__":
    """
    
    Using Class
    
    """ 
    pass
#    spacing=0.24
#    radii=[1.2, 1.2]
#    centers=np.array([[0,0], [1,0]])
#    mc = Marching2DCircles(radii=radii, centers=centers,
#                           spacing=spacing)
##    #print(mc.get_limits())
#    test_grid_region,test_grid_coords = mc.generate_region()
#    circle_grid = mc.coords[test_grid_region.ravel().astype(bool)]
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plot_grid(mc.coords,
#              ax=ax)
#    circle_grid = mc.coords[test_grid_region.ravel().astype(bool)]
#    plot_grid(circle_grid,
#            c="tab:green",
#            ax=ax)
#    plt.show()
#    plt.close()
    
    
    
#    
#    masked_grid,calc_area,draw_vertices,draw_edges,area_arguments = \
#             mc.basic_marching_squares(test_grid_region,mc.coords)
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plot_grid(mc.coords, 
#              ax=ax)
#    plot_grid(circle_grid,
#              ax=ax,
#              c="tab:green")
#    for value in draw_edges:
#        for entry in value:
#            if len(entry) > 0:
#                ax.plot(entry[:,0], entry[:,1], 
#                        c="k", linewidth=2)
#    plt.show()
#    plt.close()
#    
    
    
    ##### Random generation test
#    rand_spacing = np.random.uniform(0.075,0.3,size=(100,1))
#    rand_radii = np.random.uniform(0.5, 2, size=(100,5))
#    rand_centers = np.random.uniform(-2,2,size=(100,5,2))
#    
#    for fig_idx,radius in enumerate(rand_radii):
#        spacing = rand_spacing[fig_idx][0]
#        radii = rand_radii[fig_idx]
#        centers = rand_centers[fig_idx]
#        
#        mc = Marching2DCircles(radii=radii, centers=centers,
#                           spacing=spacing)
#        
#        grid_region,grid_coords = mc.generate_region()
#        circle_grid = mc.coords[grid_region.ravel().astype(bool)]
#        
#        masked_grid,marching_square_idx,calc_area,draw_vertices,draw_edges,area_arguments = \
#                 mc.marching_squares(grid_region,mc.coords)
#        fig = plt.figure(figsize=(8,8))
#        ax = fig.add_subplot(111)
#        ax.set_xlim(-5,5)
#        ax.set_ylim(-5,5)
#        plot_grid(mc.coords, 
#                  ax=ax)
#        plot_grid(circle_grid,
#                  ax=ax,
#                  c="tab:green")
#        #plot_circle(1.2, [0,0],ax=ax,
#        #            c="tab:red")
#        for idx,center in enumerate(mc.centers):
#            plot_circle(mc.radii[idx], center=center, ax=ax)
#        for value in draw_edges:
#            for entry in value:
#                if len(entry) > 0:
#                    ax.plot(entry[:,0], entry[:,1], 
#                            c="k", linewidth=2)
#                    
#        
#        fig.savefig("/Users/ibier/Research/Volume_Estimation/Plots/20200326_Random_Marching_Cubes/images/rand_{}.png".format(fig_idx), 
#                    dpi=100)
#        
#        plt.show()
#        plt.close()
        
            
    ########## Fixing 20200326
#    masked_grid,marching_square_idx,calc_area,draw_vertices,draw_edges,area_arguments = \
#             mc.marching_squares(test_grid_region,mc.coords)
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plot_grid(mc.coords, 
#              ax=ax)
#    plot_grid(circle_grid,
#              ax=ax,
#              c="tab:green")
#    #plot_circle(1.2, [0,0],ax=ax,
#    #            c="tab:red")
#    for idx,center in enumerate(mc.centers):
#        plot_circle(mc.radii[idx], center=center, ax=ax)
#    for value in draw_edges:
#        for entry in value:
#            if len(entry) > 0:
#                ax.plot(entry[:,0], entry[:,1], 
#                        c="k", linewidth=2)
#                
#    
#    
#    plt.show()
#    plt.close()
    
#    fig.savefig("Marching_Squares_Consumed_Circled.pdf")
    
    
    
    #### Debugging tool 
#    all_square_verts = mc.coords[marching_square_idx]
#    adjust_idx = 58
#    for square_idx,verts in enumerate(all_square_verts[adjust_idx:]):
#        square_idx += adjust_idx
#        
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        plot_grid(mc.coords, 
#                  ax=ax)
#        plot_grid(circle_grid,
#              ax=ax,
#              c="tab:green")
#        ax.scatter(
#                verts[:,0],
#                verts[:,1],
#                c="k")
#        
#        
#        temp_edges = compute_edges(verts)
#        ## Now project edge onto the surface of the sphere
#        for edge_idx,point in enumerate(temp_edges):
#            ## Project onto surface of each circle present
#            temp_projected_point_list = []
#            for r_idx,radius in enumerate(mc.radii):
#                projected_point = mc._proj_edge_2(point, edge_idx, 
#                                    radius, r_idx, verts)
#                temp_projected_point_list.append(projected_point)
#                
#            ### Decide which radius to use for this edge projection
#            ### The one to use is the one that leads to the largest area
#            ### As long as the edge has actually moved
#            temp_area_list = []
#            active_vertices = masked_grid[square_idx]
#            for entry in temp_projected_point_list:
#                if np.linalg.norm(entry - temp_edges[edge_idx]) < 1e-8:
#                    temp_area_list.append(-1)
#                    continue
#                
#                test_temp_edges = temp_edges.copy()
#                test_temp_edges[edge_idx] = entry
#                temp_area = tri_area_lookup[tostring(active_vertices.astype(int))](
#                                            test_temp_edges, 
#                                            verts)
#                temp_area_list.append(temp_area)
#
#            choice_idx = np.argmax(temp_area_list)
#            projected_point = temp_projected_point_list[choice_idx]
#            
#            temp_edges[edge_idx] = projected_point
#        
#        ax.scatter(
#                temp_edges[:,0],
#                temp_edges[:,1],
#                c="tab:orange")    
#        
#        projected_point = mc._proj_edge_2(temp_edges[1], 1, 
#                                        radius, 1, verts)
#        print(projected_point)
#        ax.scatter(
#                projected_point[0],
#                projected_point[1],
#                c="tab:red",
#                s=100)
#        
#        
#        plt.show()
#        plt.close()
#        
#        break
    
    
    
    """
    
    Let's write Something
    
    """
#    spacing=0.25
#    radii=[1.2, 1.2]
#    centers=np.array([[0,0], [24,-0.5]])
#    mc = Marching2DCircles(radii=radii, centers=centers,
#                           spacing=spacing)
##    #print(mc.get_limits())
#    grid_region,test_grid_coords = mc.generate_region()
#    
#    ### F
#    grid_region[:,:] = 0
#    grid_region[2,1:-1] = 1
#    grid_region[3,1:-1] = 1
#    grid_region[3:10,-2] = 1 
#    grid_region[3:10,-3] = 1 
#    grid_region[3:10,-7] = 1 
#    grid_region[3:10,-8] = 1 
#    
#    grid_region[12,2:-1] = 1
#    grid_region[13,2:-1] = 1
#    grid_region[13:20,3] = 1
#    grid_region[13:20,2] = 1
#    grid_region[20,1:-1] = 1
#    grid_region[21,1:-1] = 1
#    
#    grid_region[24,1:-1] = 1
#    grid_region[25,1:-1] = 1
#    grid_region[25:33,1] = 1
#    grid_region[25:33,2] = 1
#    grid_region[25:33,-2] = 1
#    grid_region[25:33,-3] = 1
#    
#    grid_region[35,1:-1] = 1
#    grid_region[36,1:-1] = 1
#    
#    x = np.arange(36, 43)
#    y = 44 + x*-1 
#    grid_region[x,y] = 1
#    y = 43 + x*-1 
#    grid_region[x,y] = 1
#    y = -30 + x
#    grid_region[x,y] = 1
#    y = -31 + x
#    grid_region[x,y] = 1
#    y = -32 + x
#    grid_region[x,y] = 1
#    
#    grid_region[55,1:-1] = 1
#    grid_region[54,1:-1] = 1
#    grid_region[48:62,-3] = 1
#    grid_region[48:62,-2] = 1
#    
#    grid_region[64,1:-1] = 1
#    grid_region[65,1:-1] = 1
#    grid_region[65:72,-2] = 1
#    grid_region[65:72,-3] = 1
#    grid_region[71,8:-3] = 1
#    grid_region[70,8:-3] = 1
#    grid_region[65:72,-7] = 1
#    grid_region[65:72,-8] = 1
#    x = np.arange(65, 72)
#    y = 73 + x*-1 
#    grid_region[x,y] = 1
#    y = 72 + x*-1 
#    grid_region[x,y] = 1
#    
#    grid_region[75,2:-1] = 1
#    grid_region[76,2:-1] = 1
#    grid_region[76:83,3] = 1
#    grid_region[76:83,2] = 1
#    grid_region[83,1:-1] = 1
#    grid_region[84,1:-1] = 1
#    
#    grid_region[87,1:-1] = 1
#    grid_region[88,1:-1] = 1
#    grid_region[95,1:-1] = 1
#    grid_region[96,1:-1] = 1
#    x = np.arange(87, 93)
#    y = 100 + x*-1 
#    grid_region[x,y] = 1
#    y = 101 + x*-1 
#    grid_region[x,y] = 1
#    x = np.arange(92, 96)
#    y = -84 + x
#    grid_region[x,y] = 1
#    y = -83 + x
#    grid_region[x,y] = 1
#    y = -82 + x
#    grid_region[x,y] = 1
#    
#    grid_region[99,1:-1] = 1
#    grid_region[100,1:-1] = 1
#    grid_region[100:107,-2] = 1
#    grid_region[100:107,-3] = 1
#    grid_region[106,8:-3] = 1
#    grid_region[105,8:-3] = 1
#    grid_region[100:107,-7] = 1
#    grid_region[100:107,-8] = 1
#    
#    
#    
#    
#    
#    masked_grid,calc_area,draw_vertices,draw_edges,area_arguments = \
#             mc.basic_marching_squares(grid_region,mc.coords)
             
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plot_grid(mc.coords, 
#              ax=ax)
#    plot_grid(circle_grid,
#              ax=ax,
#              c="tab:green")

#    plt.show()
#    plt.close()
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plot_grid(mc.coords,
#              ax=ax)
#    region = mc.coords[grid_region.ravel().astype(bool)]
#    plot_grid(region,
#            c="tab:green",
#            ax=ax)
#    
#    for value in draw_edges:
#        for entry in value:
#            if len(entry) > 0:
#                ax.plot(entry[:,0], entry[:,1], 
#                        c="k", linewidth=2)
#    
#    plt.show()
#    plt.close()
#    
#    masked_sum = np.sum(masked_grid,axis=-1)
#    surface_idx = np.where(np.logical_and(masked_sum != 0,
#                                          masked_sum != 4))[0]
#    
#    ##### Gif of writing
#    for idx,surf_idx in enumerate(surface_idx):
#        vertices = draw_vertices[surf_idx]
#        
#        fig = plt.figure(figsize=(16,5))
#        ax = fig.add_subplot(111)
#        
#        plot_grid(mc.coords,
#                  ax=ax,
#                  c="white")
#        ax.set_axis_off()
#        
#        temp_edges = compute_edges(vertices)
#        ax.scatter(vertices[:,0],
#                   vertices[:,1],
#                   c="k")
#        ax.scatter(temp_edges[:,0],
#                   temp_edges[:,1],
#                   c="tab:orange")
#        
#        for value in draw_edges[:surf_idx+1]:
#            for entry in value:
#                if len(entry) > 0:
#                    ax.plot(entry[:,0], entry[:,1], 
#                            c="k", linewidth=2)
#        
#        fig.savefig("/Users/ibier/Research/Volume_Estimation/Plots/20200326_Random_Marching_Cubes/words_surf_only/fig_{}.png"
#                    .format(str(idx).zfill(4)),
#                    dpi=100)
#        
#        plt.show()
#        plt.close()
#        
#        break
        
        
    
   
    
    #print(1.2*1.2*np.pi, calc_area)
    #
    #
    #
    #for i in range(69):
    ##    print(i*10, i*10+10)
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111)
    #    plot_grid(circle_grid,
    #            c="tab:green",
    #            ax=ax)
    #    plot_grid(circle_grid[i,:][None,:],
    #        c="tab:red",
    #        ax=ax)
    #    plt.show()
    #    plt.close()
        
        
    
    #edge_area_test = area_arguments[44]
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plot_grid(edge_area_test[1],
    #          ax=ax,
    #          c="tab:orange")
    #plot_grid(edge_area_test[2],
    #          ax=ax,
    #          c="tab:blue")
    #
    #for idx,entry in enumerate(edge_area_test[2]):
    #    ax.text(entry[0],entry[1],
    #            "{}".format(idx))
    
    #for value in draw_vertices:
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111)
    #    plot_grid(mc.coords, 
    #              ax=ax)
    #    plot_grid(circle_grid,
    #              ax=ax,
    #              c="tab:green")
    #    ax.scatter(value[:,0],
    #            value[:,1],
    #            c="k")
    #    plt.show()
    #    plt.close()
    
    
    """
    
    Let's make a area estimation convergence plot
    
    """
#    spacing_range=np.arange(0.05, 0.2, 0.005)
#    radii=[1.2]
#    centers=np.array([[0,0]])
#    target_area = 1.2*1.2*np.pi
#    area_estimate_voxel = []
#    area_estimate_simple_march = []
#    area_estimate_proj_march = []
#    for spacing in spacing_range:
#        print(spacing)
#        mc = Marching2DCircles(radii=radii, centers=centers,
#                               spacing=spacing)
#        #print(mc.get_limits())
#        grid_region,grid_coords = mc.generate_region()
#        masked_grid,calc_area,draw_vertices,draw_edges_simple,area_arguments = \
#             mc.basic_marching_squares(grid_region,mc.coords)
#             
#        area_estimate_voxel.append(np.sum(grid_region)*
#                                   spacing*spacing)
#        area_estimate_simple_march.append(calc_area)
#        
#        ### Perform marching with projection
#        masked_grid,calc_area,draw_vertices,draw_edges_proj,area_arguments = \
#             mc.marching_squares(grid_region,mc.coords)
#        
#        area_estimate_proj_march.append(calc_area)
#        
#        print(spacing,target_area,
#                area_estimate_voxel[-1],
#                area_estimate_simple_march[-1],
#                area_estimate_proj_march[-1])
    
#    mc = Marching2DCircles(radii=radii, centers=centers,
#                               spacing=0.1)
#    grid_region,grid_coords = mc.generate_region()
#    masked_grid,calc_area,draw_vertices,draw_edges_simple,area_arguments = \
#       mc.basic_marching_squares(grid_region,mc.coords)
#    masked_grid,calc_area,draw_vertices,draw_edges_proj,area_arguments = \
#             mc.marching_squares(grid_region,mc.coords)
        
#    #### Settings
#    fontsize=16
#    fig = plt.figure(figsize=(15,18))
#        
#    ax = fig.add_subplot(321)
#    ax.plot(spacing_range,
#            area_estimate_voxel,
#            linewidth=2,
#            c="tab:blue")
#    ax.axhline(target_area,
#               c="tab:red",
#               linewidth=2)
#    ax.set_ylabel("Calculated Area",
#                  fontsize=fontsize)
#    
#    max_ylim = ax.get_ylim()
#    
#    
#    ax = fig.add_subplot(322)
#    circle_grid = mc.coords[grid_region.ravel().astype(bool)]
#    plot_grid(mc.coords,
#              ax=ax,
#              c="tab:blue")
#    plot_grid(circle_grid,
#              ax=ax,
#              c="tab:green")
#    
#    
#    ###### NEXT
#    
#    ax = fig.add_subplot(323)
#    ax.plot(spacing_range,
#            area_estimate_simple_march,
#            linewidth=2,
#            c="tab:blue")
#    ax.axhline(target_area,
#               c="tab:red",
#               linewidth=2)
#    ax.set_ylim(max_ylim)
#    ax.set_ylabel("Calculated Area",
#                  fontsize=fontsize)
#    
#    ax = fig.add_subplot(324)
#    plot_grid(mc.coords,
#              ax=ax,
#              c="tab:blue")
#    plot_grid(circle_grid,
#              ax=ax,
#              c="tab:green")
#    
#    for value in draw_edges_simple:
#        for entry in value:
#            if len(entry) > 0:
#                ax.plot(entry[:,0], entry[:,1], 
#                        c="k", linewidth=2) 
#                
#    ##### NEXT
#    
#    
#    ax = fig.add_subplot(325)
#    ax.plot(spacing_range,
#            area_estimate_proj_march,
#            linewidth=2,
#            c="tab:blue")
#    ax.axhline(target_area,
#               c="tab:red",
#               linewidth=2)
#    ax.set_ylim(max_ylim)
#    ax.set_xlabel("Grid Spacing",
#                  fontsize=fontsize)
#    ax.set_ylabel("Calculated Area",
#                  fontsize=fontsize)
#    
#    
#    ax = fig.add_subplot(326)
#    plot_grid(mc.coords,
#              ax=ax,
#              c="tab:blue")
#    plot_grid(circle_grid,
#              ax=ax,
#              c="tab:green")
#    
#        
#    for value in draw_edges_proj:
#        for entry in value:
#            if len(entry) > 0:
#                ax.plot(entry[:,0], entry[:,1], 
#                        c="k", linewidth=2)
#    
#    fig.savefig("Marching_Cubes_2D_Convergence.pdf")
             
#    fontsize=16
#    fig = plt.figure(figsize=(15,18))
#    
#    area_estimate_voxel = np.array(area_estimate_voxel)
#    area_estimate_simple_march = np.array(area_estimate_simple_march)
#    area_estimate_proj_march = np.array(area_estimate_proj_march)
#    
#    if np.any(area_estimate_voxel > 4):
#        area_estimate_voxel = area_estimate_voxel / target_area
#    if np.any(area_estimate_simple_march > 4):
#        area_estimate_simple_march = area_estimate_simple_march / target_area
#    if np.any(area_estimate_proj_march > 4):
#        area_estimate_proj_march = area_estimate_proj_march / target_area
#    
#    ax = fig.add_subplot(321)
#    ax.plot(spacing_range,
#            area_estimate_voxel,
#            linewidth=2,
#            c="tab:blue")
#    ax.axhline(1,
#               c="tab:red",
#               linewidth=2)
#    ax.set_ylabel("Calculated Area",
#                  fontsize=fontsize)
#    
#    max_ylim = ax.get_ylim()
#    
#    
#    ax = fig.add_subplot(322)
#    circle_grid = mc.coords[grid_region.ravel().astype(bool)]
#    plot_grid(mc.coords,
#              ax=ax,
#              c="tab:blue")
#    plot_grid(circle_grid,
#              ax=ax,
#              c="tab:green")
#    
#    
#    ###### NEXT
#    
#    ax = fig.add_subplot(323)
#    ax.plot(spacing_range,
#            area_estimate_simple_march,
#            linewidth=2,
#            c="tab:blue")
#    ax.axhline(1,
#               c="tab:red",
#               linewidth=2)
#    ax.set_ylim(max_ylim)
#    ax.set_ylabel("Calculated Area",
#                  fontsize=fontsize)
#    
#    ax = fig.add_subplot(324)
#    plot_grid(mc.coords,
#              ax=ax,
#              c="tab:blue")
#    plot_grid(circle_grid,
#              ax=ax,
#              c="tab:green")
#    
#    for value in draw_edges_simple:
#        for entry in value:
#            if len(entry) > 0:
#                ax.plot(entry[:,0], entry[:,1], 
#                        c="k", linewidth=2) 
                
    ##### NEXT
    
#    ax = fig.add_subplot(325)
#    ax.plot(spacing_range,
#            area_estimate_proj_march,
#            linewidth=2,
#            c="tab:blue")
#    ax.axhline(1,
#               c="tab:red",
#               linewidth=2)
#    ax.set_ylim(max_ylim)
#    ax.set_xlabel("Grid Spacing",
#                  fontsize=fontsize)
#    ax.set_ylabel("Calculated Area",
#                  fontsize=fontsize)
#    
#    
#    ax = fig.add_subplot(326)
#    plot_grid(mc.coords,
#              ax=ax,
#              c="tab:blue")
#    plot_grid(circle_grid,
#              ax=ax,
#              c="tab:green")
#    
#        
#    for value in draw_edges_proj:
#        for entry in value:
#            if len(entry) > 0:
#                ax.plot(entry[:,0], entry[:,1], 
#                        c="k", linewidth=2)
#    
#    fig.savefig("Marching_Cubes_2D_Convergence_Percent.pdf")
        

#"""
#
#Settings
#
#"""
#center = 0 
#spacing = 0.2
#radius = 1.2
#
#
#target_area = radius*radius*np.pi
#
#
#
#"""
#
#Basic area convergence test
#
#"""
#
#spacing_range = np.arange(0.01, 0.1, 0.001)
#
##area_values = []
##for value in spacing_range:
##    area_values.append(get_grid_area(radius, value))
#
##fig = plt.figure(figsize=(6,10))
##
##fontsize=16
#
##fig = plt.figure()
##ax = fig.add_subplot(211)
##ax.plot(spacing_range, area_values)
##ax.axhline(target_area,
##           c="tab:red")
##ylim = ax.get_ylim()
##ax.set_ylabel("Calcualted Area",
##              fontsize=fontsize)
##ax.set_title("Simple Square Approximation",
##             fontsize=fontsize)
#
##ax = fig.add_subplot(222)
##plot_grid(coords,
##          c="tab:blue",
##          ax=ax)
##plot_grid(final_idx_grid_points,
##          c="tab:green",
##          ax=ax)
##plot_circle(radius,
##            ax=ax)
#
#
#
##ax = fig.add_subplot(212)
##ax.plot(spacing_range, adjusted_area_values)
##ax.axhline(target_area,
##           c="tab:red")
##ax.set_ylim(ylim)
##ax.set_ylabel("Calcualted Area",
##              fontsize=fontsize)
##ax.set_title("Marching Squares Smoothed Surface",
##             fontsize=fontsize)
#
##
##ax = fig.add_subplot(224)
##ax.plot_grid(coords,
##             c="tab:blue",
##             ax=ax)
##plot_
#
#
##fig.savefig("/Users/ibier/Research/Volume_Estimation/Plots/20200318_Marching_Squares/convergece_simple.png",
##            dpi=400)
##
##plt.show()
##plt.close()
##
##raise Exception()
#
##adjusted_area_values = []
##for spacing in spacing_range:
#"""
#
#Generate Grid Coords and Volume Shape
#
#"""
#min_val = center - radius - spacing
#max_val = center + radius + spacing
#grid_range = (max_val - min_val) / 2
#
#x_grid_from_0 = np.arange(0,grid_range+spacing,spacing)
#x_vals = np.hstack([-x_grid_from_0[::-1][:-1],x_grid_from_0])
#y_vals = x_vals.copy()
#
#X,Y = np.meshgrid(x_vals,y_vals)
#coords = np.c_[X.ravel(), 
#               Y.ravel()]
#
#grid_region = np.zeros((X.shape[0],
#                        Y.shape[0]))        
#
##    fig = plt.figure(figsize=(6,6))
##    ax = fig.add_subplot(111)
##    plot_grid(coords, ax=ax)
##    plot_circle(radius, ax=ax)
##    
##    plt.gca().set_aspect('equal', adjustable='box')
#
#
#"""
#
#Need to assign weights to each grid point based on the circles present
#
#"""
#
#body_diag = spacing * np.sqrt(2)
#
#radius_point = np.round(radius / spacing)
##radius_point = np.floor((radius - body_diag) / spacing)
#radius_point = int(radius_point)
#
#### Now fill in all grid points to the radius 
#idx_range = np.arange(-radius_point , radius_point+1)[::-1]
#sort_idx = np.argsort(np.abs(idx_range))
#idx_range = idx_range[sort_idx]
#
#all_idx = np.array(np.meshgrid(idx_range,
#                               idx_range)).T.reshape(-1,2)
#
#all_idx = all_idx.astype(int)
#all_norm = np.linalg.norm(all_idx*spacing, axis=-1)
#take_idx = np.where(all_norm < radius)[0]
#
#final_idx = all_idx[take_idx]
#
#final_idx_grid_points = final_idx * spacing
#
##    plot_grid(final_idx_grid_points, 
##              c="tab:green",
##              ax=ax)
#
##### Each grid point counts for the area around it using a vernoi construction
##grid_area = final_idx_grid_points.shape[0]*spacing*spacing
##
##
##### Each point not w/i a body diangonal has a weight of 1
##one_idx = np.where(all_norm < (radius - body_diag))[0]
##one_grid_points = all_idx[one_idx]*spacing
##plot_grid(one_grid_points, 
##          c="tab:orange",
##          ax=ax)
##
##reminader_idx = np.where(all_norm[take_idx] > (radius - body_diag))[0]
##remainder_points = all_idx[take_idx][reminader_idx]*spacing
##
##plot_grid(remainder_points, 
##          c="tab:purple",
##          ax=ax)
##
##
##for point in remainder_points:
##    mult = np.ones(2)
##    mult_idx = np.where(point < 0)[0]
##    mult[mult_idx] = -1
##    
##    proj_along_y = mult[1]*np.sqrt(radius*radius - point[0]*point[0])
##    proj_along_x = mult[0]*np.sqrt(radius*radius - point[1]*point[1])
##    
##    
##    temp_plot_x_x = [point[0], point[0]]
##    temp_plot_x_y = [point[1], proj_along_y]
##    
###    ax.plot(temp_plot_x_x, temp_plot_x_y, linewidth=2,
###            c="k")
##    
##    temp_plot_y_x = [point[0], proj_along_x]
##    temp_plot_y_y = [point[1], point[1]]
###    ax.plot(temp_plot_y_x, temp_plot_y_y, linewidth=2,
###            c="k")
##    
##    ### Can we calculate contribution?
##    diff_x = np.abs(point[0] - proj_along_x)
##    diff_y = np.abs(point[1] - proj_along_y)
###    
###    print(diff_x,diff_y,spacing)
##    
##    if diff_x > spacing:
##        contribution_x = 0
##    else:
##        ax.plot(temp_plot_y_x, temp_plot_y_y, linewidth=2,
##            c="k")
##    if diff_y > spacing:
##        contribution_y = 0
##    else:
##        ax.plot(temp_plot_x_x, temp_plot_x_y, linewidth=2,
##            c="k")
##
##plt.show()
##plt.close()
#
#
#"""
#
#Add each point to the grid_region
#
#"""
#
#grid_region_idx = np.round((final_idx_grid_points - x_vals[0]) / spacing)
#grid_region_idx = grid_region_idx.astype(int)
#grid_region[grid_region_idx[:,0], grid_region_idx[:,1]] = 1
#
#
#"""
#
#Lookup for marching squares
#
#"""
#
#
#
#"""
#
#Get voxel idx mask
#
#"""
#x_num = x_vals.shape[0]
#y_num = y_vals.shape[0]
#
#### Start with X values because they change by value of 1 for indexing
#x_proj = np.arange(0, x_num-1, 1)
#bot_left = x_proj
#
### Move in Y direction which is plus the length of x_vals
#top_left = bot_left + x_num
#
### Move in X direction which is just plus 1
#bot_right = x_proj + 1
#top_right = top_left + 1
#
#
#### Now, project along the Y direction
#y_proj = np.arange(0, y_num-1, 1) * x_num
#
#bot_left = bot_left + y_proj[:,None]
#top_left = top_left + y_proj[:,None]
#bot_right = bot_right + y_proj[:,None]
#top_right = top_right + y_proj[:,None]
#
#
#top_left_coords = np.take(coords, top_left.ravel(), axis=0)
#bot_left_coords = np.take(coords, bot_left.ravel(), axis=0)
#
#top_right_coords = np.take(coords, top_right.ravel(), axis=0)
#bot_right_coords = np.take(coords, bot_right.ravel(), axis=0)
#
#
#marching_square_idx = np.c_[
#        bot_left.ravel(),
#        top_left.ravel(),
#        top_right.ravel(),
#        bot_right.ravel(),]
#
#masked_grid = np.take(grid_region, marching_square_idx)
#
#################################################################################
####### Let's construct modified edge positions for each crossing point
####### Also, need to calculate the area contribution
####### Or, calculate area contribution on the fly
#################################################################################
#
#
######## Original
##edges_for_region = []
##for row in marching_square_idx:
##    row_coords = coords[row]
##    edges_for_region.append(compute_edges(row_coords))
##edges_for_region = np.vstack(edges_for_region)
##
##masked_grid_sum = np.sum(masked_grid, axis=-1)
##surface_idx = np.where(np.logical_and(masked_grid_sum != 0,
##                                      masked_grid_sum != 4))[0]
##
##for temp_surface_idx,idx in enumerate(surface_idx):
##    square_idx = marching_square_idx[idx]
##    square_coords = coords[square_idx]
##    temp_edges = compute_edges(square_coords)
##    
##    ## Now project edge onto the surface of the sphere
##    for edge_idx,point in enumerate(temp_edges):
##        mult = np.ones(2)
##        mult_idx = np.where(point < 0)[0]
##        mult[mult_idx] = -1
##        
##        x2_proj = radius*radius - point[1]*point[1]
##        y2_proj = radius*radius - point[0]*point[0]
##        
##        if x2_proj < 0:
##            proj_x = np.sqrt(-x2_proj)
##        else:
##            proj_x = np.sqrt(x2_proj)
##        
##        if y2_proj < 0:
##            proj_y = np.sqrt(-y2_proj)
##        else:
##            proj_y = np.sqrt(y2_proj)
##            
##        proj = np.array([proj_x,proj_y])*mult
##        ## Now decide how to move the edge
##        diff = np.abs(point - proj)
##        temp_take_proj_idx = np.where(diff < spacing)[0]
##        projected_point = point.copy()
##        projected_point[temp_take_proj_idx] = proj[temp_take_proj_idx]
##        
##        ### Edges point movement only has one degree of freedom
##        if edge_idx == 0:
##            projected_point[1] = point[1]
##        elif edge_idx == 1:
##            projected_point[0] = point[0]
##        elif edge_idx == 2:
##            projected_point[1] = point[1]
##        elif edge_idx == 3:
##            projected_point[0] = point[0]
##            
##        ### Have to make sure that point has not left square
##        if edge_idx == 0:
##            if projected_point[0] < square_coords[0][0]:
##                projected_point[0] = point[0]
##            elif projected_point[0] > square_coords[3][0]:
##                projected_point[0] = point[0]
##        elif edge_idx == 1:
##            if projected_point[1] > square_coords[1][1]:
##                projected_point[1] = point[1]
##            elif projected_point[1] < square_coords[0][1]:
##                projected_point[1] = point[1]
##        if edge_idx == 2:
##            if projected_point[0] < square_coords[1][0]:
##                projected_point[0] = point[0]
##            elif projected_point[0] > square_coords[2][0]:
##                projected_point[0] = point[0]
##        if edge_idx == 3:
##            if projected_point[1] > square_coords[2][1]:
##                projected_point[1] = point[1]
##            elif projected_point[1] < square_coords[3][1]:
##                projected_point[1] = point[1]
##        
##        ### Save in edges_for_region
##        edges_for_region[idx*4+edge_idx] = projected_point
##    
##    active_vertices = masked_grid[idx]
#
####### Plot adjust edges
##square_idx = 19
##values_list = [x for x in zip(bot_left_coords,
##                  top_left_coords,
##                  top_right_coords,
##                  bot_right_coords
##                  )]
##for values in values_list[square_idx:]:
##    values = np.vstack(values)
##    
##    fig = plt.figure(figsize=(6,6))
##    ax = fig.add_subplot(111)
##    plot_grid(coords, ax=ax)
##    plot_circle(radius,ax=ax)
##    plot_grid(values, 
##              c=["tab:red", "tab:green", "tab:purple", "tab:cyan"],
##              edgecolor="k",
##              ax=ax)
##    
##    edges = edges_for_region[square_idx*4:square_idx*4+4,:]
##    
##    plot_grid(edges, 
##              c="tab:orange",
##              edgecolor="k",
##              ax=ax)
##    
##    
##    plt.gca().set_aspect('equal', adjustable='box')
##    plt.show()
##    plt.close()
##    
##    square_idx += 1
##    break
#
#
#
###### Center Testing
##masked_grid,marching_square_idx,calc_area,draw_vertices,draw_edges,area_arguments = \
##         mc.marching_squares(test_grid_region,mc.coords)
##coords = mc.coords
##spacing=mc.spacing
##
##edges_for_region = []
##for row in marching_square_idx:
##    row_coords = coords[row]
##    edges_for_region.append(compute_edges(row_coords))
##edges_for_region = np.vstack(edges_for_region)
##
##masked_grid_sum = np.sum(masked_grid, axis=-1)
##surface_idx = np.where(np.logical_and(masked_grid_sum != 0,
##                                      masked_grid_sum != 4))[0]
##
##square_coords_list = []
##edge_coords_list = []
##proj_list = []
##for temp_surface_idx,idx in enumerate(surface_idx):
##    square_idx = marching_square_idx[idx]
##    square_coords = coords[square_idx]
##    
##    square_coords_list.append(square_coords)
##    
##    temp_edges = compute_edges(square_coords)
##    temp_proj = []
##    ## Now project edge onto the surface of the sphere
##    for edge_idx,point in enumerate(temp_edges):
##        ## Store where the point is negative for laters
##        mult = np.ones(2)
##        mult_idx = np.where(point < 0)[0]
##        mult[mult_idx] = -1
##        
##        ## Project onto surface of each circle present
##        temp_projected_point_list = []
##        
##        #### Radius for loop
##        temp_center = mc.centers[0]
##        
##        x2_proj = radius*radius - np.square(point[1] - 
##                                   temp_center[1])
##        y2_proj = radius*radius - np.square(point[0] - 
##                                   temp_center[0])
##        
##        if x2_proj < 0:
##            proj_x = np.sqrt(-x2_proj)
##        else:
##            proj_x = np.sqrt(x2_proj)
##        
##        if y2_proj < 0:
##            proj_y = np.sqrt(-y2_proj)
##        else:
##            proj_y = np.sqrt(y2_proj)
##        
##        #### Check if should use plus or minus sqrt
##        if abs(-point[0] - proj_x + temp_center[0]) < \
##           abs(-point[0] + proj_x + temp_center[0]):
##               proj_x = proj_x * -1
##        if abs(-point[1] - proj_y + temp_center[1]) < \
##           abs(-point[1] + proj_y + temp_center[1]):
##               proj_y = proj_y * -1
##               
###        calc_sign = np.array([proj_x,proj_y]) + temp_center
###        test_pos = calc_sign + temp_center
###        test_neg = -calc_sign + temp_center
###        if np.linalg.norm(point - test_pos) < np.linalg.norm(point - test_neg):
###            proj_x = proj_x * -1
###            proj_y = proj_y * -1
###        else:
###            pass
##            
##        proj_x = proj_x + temp_center[0]
##        proj_y = proj_y + temp_center[1]
##        
##        temp_proj.append([proj_x,proj_y])
##        
##        ### Now use sign of the point
##        proj = np.array([proj_x,proj_y])
##        
##        ## Now decide how to move the edge. If the projection is 
##        ## within the grid spacing, then it's a valid projection
##        diff = np.abs(point - proj)
##        temp_take_proj_idx = np.where(diff < mc.spacing)[0]
##        projected_point = point.copy()
##        projected_point[temp_take_proj_idx] = proj[temp_take_proj_idx]
##        
##        if len(edge_coords_list) == 15:
##            print(point, 
##                  proj, 
##                  projected_point)
##
##
##        ### Edges point movement only has one degree of freedom
##        if edge_idx == 0:
##            projected_point[1] = point[1]
##        elif edge_idx == 1:
##            projected_point[0] = point[0]
##        elif edge_idx == 2:
##            projected_point[1] = point[1]
##        elif edge_idx == 3:
##            projected_point[0] = point[0]
##            
##        ### Have to make sure that point has not left square
##        if edge_idx == 0:
##            if projected_point[0] < square_coords[0][0]:
##                projected_point[0] = point[0]
##            elif projected_point[0] > square_coords[3][0]:
##                projected_point[0] = point[0]
##        elif edge_idx == 1:
##            if projected_point[1] > square_coords[1][1]:
##                projected_point[1] = point[1]
##            elif projected_point[1] < square_coords[0][1]:
##                projected_point[1] = point[1]
##        if edge_idx == 2:
##            if projected_point[0] < square_coords[1][0]:
##                projected_point[0] = point[0]
##            elif projected_point[0] > square_coords[2][0]:
##                projected_point[0] = point[0]
##        if edge_idx == 3:
##            if projected_point[1] > square_coords[2][1]:
##                projected_point[1] = point[1]
##            elif projected_point[1] < square_coords[3][1]:
##                projected_point[1] = point[1]
##                
##        
###        temp_projected_point_list.append(projected_point)
###        
###        temp_projected_point_list = np.vstack(temp_projected_point_list)
###        x_idx = np.argmax(np.abs(temp_projected_point_list[:,0]))
###        y_idx = np.argmax(np.abs(temp_projected_point_list[:,1]))
###        
###        projected_point = np.zeros(2)
###        projected_point[0] = temp_projected_point_list[:,0][x_idx]
###        projected_point[1] = temp_projected_point_list[:,1][y_idx]
##        
###                print(point, temp_projected_point_list, x_idx,y_idx)
##
##        ### Save in edges_for_region
##        edges_for_region[idx*4+edge_idx] = projected_point
##        
##        temp_edges[edge_idx] = projected_point
##            
##    edge_coords_list.append(temp_edges)
##    proj_list.append(temp_proj)
#        
#
#
####### Plot adjust edges
##square_idx = 14
##for values in square_coords_list[square_idx:]:
##    values = np.vstack(values)
##    
##    fig = plt.figure(figsize=(6,6))
##    ax = fig.add_subplot(111)
##    plot_grid(coords, ax=ax)
##    plot_circle(radius, center=mc.centers[0], ax=ax)
##    plot_grid(values, 
##              c=["tab:red", "tab:green", "tab:purple", "tab:cyan"],
##              edgecolor="k",
##              ax=ax)
##    
##    edges = edges_for_region[square_idx*4:square_idx*4+4,:]
##    edges = edge_coords_list[square_idx]
##    
##    plot_grid(edges, 
##              c="tab:orange",
##              edgecolor="k",
##              ax=ax)
##    
##    for idx,edge in enumerate(edges):
##        ax.text(edge[0],edge[1],
##                "{}".format(idx))
##        
##    temp_proj = np.vstack(proj_list[square_idx])
##    ax.scatter(temp_proj[:,0], temp_proj[:,1],
##               c="tab:pink",
##               s=100)
##    
###    ax.scatter(
###            -np.sqrt(1.2*1.2 - np.square(-0.75 - 0)) -3,
###            -0.75,
###               c="tab:cyan")
##    
##    
##    plt.gca().set_aspect('equal', adjustable='box')
##    plt.show()
##    plt.close()
##    
##    square_idx += 1
##    break
#        
#################################################################################
##### Print masked grid with adjusted edges
#################################################################################
#
##positive_idx = marching_square_idx[masked_grid.astype(bool)]
##positive_coords = coords[positive_idx]
##fig = plt.figure(figsize=(6,6))
##ax = fig.add_subplot(111)
##
##plot_grid(coords,
##          ax=ax)
##plot_circle(radius, ax=ax)
##plot_grid(final_idx_grid_points, 
##          c="tab:green",
##          ax=ax)
##plot_grid(positive_coords,
##          ax=ax)
##
##plot_edges = []
##
##adjusted_area = 0
##for idx,row in enumerate(masked_grid):
##    row_string = tostring(row.astype(int))
##    
##    row_coords = coords[marching_square_idx[idx]]
##    temp_edges = compute_edges(row_coords)
##    temp_edges = np.vstack([edges_for_region[idx*4+x] for x in range(4)])
##    
##    connect_edges_bool = tri_lookup[row_string]
##    
##    for temp_mask in connect_edges_bool:  
##        temp_masked_edges = temp_edges[temp_mask]
##        if len(temp_masked_edges) > 0:
##            ax.plot(temp_masked_edges[:,0],
##                    temp_masked_edges[:,1],
##                    c="black")
##    
##    temp_adjusted_area = tri_area_lookup[tostring(row.astype(int))](temp_edges, 
##                                                                row_coords)
##    
##    adjusted_area += tri_area_lookup[tostring(row.astype(int))](temp_edges, 
##                                                                row_coords)
#    
#    
##    ax.scatter(row_coords[:,0],
##               row_coords[:,1],
##               c="k")
##    
##    activated_vertices = row_coords[row.astype(bool)]
##    if len(activated_vertices) > 0:
##        ax.scatter(activated_vertices[:,0],
##                   activated_vertices[:,1],
##                   c="tab:purple")
#        
##    print(temp_adjusted_area, temp_adjusted_area / (spacing*spacing) * 100)
#    
##    break
#
##    plt.gca().set_aspect('equal', adjustable='box')
##    plt.show()
##    plt.close()
#
##print(target_area,adjusted_area)
#
##adjusted_area_values.append(adjusted_area)
#
#
#################################################################################
##### Print masked grid
#################################################################################
#
##positive_idx = marching_square_idx[masked_grid.astype(bool)]
##positive_coords = coords[positive_idx]
##fig = plt.figure(figsize=(6,6))
##ax = fig.add_subplot(111)
##
##plot_grid(coords,
##          ax=ax)
##plot_circle(radius, ax=ax)
##plot_grid(final_idx_grid_points, 
##          c="tab:green",
##          ax=ax)
##plot_grid(positive_coords,
##          ax=ax)
##
##plot_edges = []
##
##for idx,row in enumerate(masked_grid):
##    row_string = tostring(row.astype(int))
##    
##    row_coords = coords[marching_square_idx[idx]]
##    temp_edges = compute_edges(row_coords)
##    
##    connect_edges_bool = tri_lookup[row_string]
##    
##    for temp_mask in connect_edges_bool:  
##        temp_masked_edges = temp_edges[temp_mask]
##        if len(temp_masked_edges) > 0:
##            ax.plot(temp_masked_edges[:,0],
##                    temp_masked_edges[:,1],
##                    c="black")
##
##plt.gca().set_aspect('equal', adjustable='box')
##plt.show()
##plt.close()
#
#################################################################################
##### Print masked grid gif
#################################################################################
#
##positive_idx = marching_square_idx[masked_grid.astype(bool)]
##positive_coords = coords[positive_idx]
##plot_edges = []
##for idx,row in enumerate(masked_grid):
##    fig = plt.figure(figsize=(6,6))
##    ax = fig.add_subplot(111)
##    
##    plot_grid(coords,
##              ax=ax)
###    plot_circle(radius, ax=ax)
##    plot_grid(final_idx_grid_points, 
##              c="tab:green",
##              ax=ax)
##    plot_grid(positive_coords,
##              ax=ax)
##    
##    row_string = tostring(row.astype(int))
##    
##    row_coords = coords[marching_square_idx[idx]]
##    temp_edges = compute_edges(row_coords)
##    temp_edges = np.vstack([edges_for_region[idx*4+x] for x in range(4)])
##    
##    connect_edges_bool = tri_lookup[row_string]
##    
##    ## Plot surface
##    for temp_mask in connect_edges_bool:  
##        temp_masked_edges = temp_edges[temp_mask]
##        
##        if len(temp_masked_edges) > 0:
##            plot_edges.append(temp_masked_edges)
##        
##        if len(plot_edges) > 0:
##            for entry in plot_edges:
##                ax.plot(entry[:,0],
##                        entry[:,1],
##                        c="black")
##    ## Plot square coords
##    plot_grid(row_coords, 
##              c="k",
##              edgecolor="k",
##              ax=ax)
##    plot_grid(temp_edges, 
##              c="tab:orange",
##              edgecolor="k",
##              ax=ax)
##    
##    plt.gca().set_aspect('equal', adjustable='box')
##    plt.show()
##    
##    fig.savefig("/Users/ibier/Research/Volume_Estimation/Plots/20200318_Marching_Squares/images_adjusted_edges/marching_squares_{}.png".format(str(idx).zfill(5)),
##                dpi=400)
##    
##    plt.close()
#
#
#################################################################################
##### Print Marching Squares
#################################################################################
#
##for values in zip(bot_left_coords,
##                  top_left_coords,
##                  top_right_coords,
##                  bot_right_coords
##                  ):
##    values = np.vstack(values)
##    
##    fig = plt.figure(figsize=(6,6))
##    ax = fig.add_subplot(111)
##    plot_grid(coords, ax=ax)
##    plot_grid(values, 
##              c=["tab:red", "tab:green", "tab:purple", "tab:cyan"],
##              edgecolor="k",
##              ax=ax)
##    
##    
##    
##    edges = compute_edges(values)
##    
##    plot_grid(edges, 
##              c="tab:orange",
##              edgecolor="k",
##              ax=ax)
##    
##    
##    plt.gca().set_aspect('equal', adjustable='box')
##    plt.show()
##    plt.close()
##    
##    break

################################################################################
#### Debugging Marhing Squares
################################################################################
#space_range=np.arange(0.01, 0.1, 0.005)[::-1]
#radii=[1.2]
#centers=np.array([[0,0]])
#area_list = []
#target_area = 1.2*1.2*np.pi
#for spacing in space_range[0:]:
#    mc = Marching2DCircles(radii=radii, centers=centers,
#                           spacing=spacing)
#    
#    test_grid_region,test_grid_coords = mc.generate_region()
#    
#    coords = mc.coords
#    grid_region,grid_coords = mc.generate_region()
#    x_num = grid_region.shape[0]
#    y_num = grid_region.shape[1]
#    
#    
#    ### Start with Y values because they change by value of 1 for indexing
#    y_proj = np.arange(0, y_num-1, 1)
#    bot_left = y_proj
#    
#    # Move in Y direction which is plus 1
#    top_left = bot_left + 1
#    
#    ## Move in X direction which is plus y_num
#    bot_right = bot_left + y_num
#    top_right = bot_right + 1
#    
#    
#    ### Now, project along the Y direction
#    x_proj = np.arange(0, x_num-1, 1) * y_num
#    
#    bot_left = bot_left + x_proj[:,None]
#    top_left = top_left + x_proj[:,None]
#    bot_right = bot_right + x_proj[:,None]
#    top_right = top_right + x_proj[:,None]
#    
#    marching_square_idx = np.c_[
#            bot_left.ravel(),
#            top_left.ravel(),
#            top_right.ravel(),
#            bot_right.ravel(),
#            ]
#    
#    
#    masked_grid = np.take(grid_region, marching_square_idx)
#    
#    
#    edges_for_region = []
#    for row in marching_square_idx:
#        row_coords = coords[row]
#        edges_for_region.append(compute_edges(row_coords))
#    
#    masked_grid_sum = np.sum(masked_grid, axis=-1)
#    surface_idx = np.where(np.logical_and(masked_grid_sum != 0,
#                                          masked_grid_sum != 4))[0]
#    
#    ### Create hand crafted edge values for surface voxels by projecting 
#    ### the edge onto the surface of each circle and then use the circle 
#    ### surface that is within the spacing and has a larger distance 
#    ### because this will be the point on a potentially overlapping surface
#    proj_list = [edges_for_region.copy() for x in range(len(mc.radii))]
#    for surface_iter_idx,temp_surface_idx in enumerate(surface_idx):
#        square_idx = marching_square_idx[temp_surface_idx]
#        square_coords = coords[square_idx]
#        temp_edges = compute_edges(square_coords)
#        
#        ## Now project edge onto the surface of the sphere
#        for edge_idx,point in enumerate(temp_edges):
#            ## Store where the point is negative for laters
#            mult = np.ones(2)
#            mult_idx = np.where(point < 0)[0]
#            mult[mult_idx] = -1
#            
#            ## Project onto surface of each circle present
#            temp_projected_point_list = []
#            for r_idx,radius in enumerate(mc.radii):
#                temp_center = mc.centers[r_idx]
#    
#                x2_proj = radius*radius - np.square(point[1] - 
#                                           temp_center[1])
#                y2_proj = radius*radius - np.square(point[0] - 
#                                           temp_center[0])
#                
#                if x2_proj < 0:
#                    proj_x = np.sqrt(-x2_proj)
#                else:
#                    proj_x = np.sqrt(x2_proj)
#                
#                if y2_proj < 0:
#                    proj_y = np.sqrt(-y2_proj)
#                else:
#                    proj_y = np.sqrt(y2_proj)
#                
#                #### Check if should use plus or minus sqrt
#                if abs(-point[0] - proj_x + temp_center[0]) < \
#                   abs(-point[0] + proj_x + temp_center[0]):
#                       proj_x = proj_x * -1
#                if abs(-point[1] - proj_y + temp_center[1]) < \
#                   abs(-point[1] + proj_y + temp_center[1]):
#                       proj_y = proj_y * -1
#                    
#                proj_x = proj_x + temp_center[0]
#                proj_y = proj_y + temp_center[1]
#                
#                ### Now use sign of the point
#                proj = np.array([proj_x,proj_y])
#                
#                proj_list[r_idx][temp_surface_idx][edge_idx] = proj
#                
#                ## Now decide how to move the edge. If the projection is 
#                ## within the grid spacing, then it's a valid projection
#                diff = np.abs(point - proj)
#                temp_take_proj_idx = np.where(diff < mc.spacing)[0]
#                projected_point = point.copy()
#                projected_point[temp_take_proj_idx] = proj[temp_take_proj_idx]
#                
#                ### Edges point movement only has one degree of freedom
#                if edge_idx == 0:
#                    projected_point[1] = point[1]
#                elif edge_idx == 1:
#                    projected_point[0] = point[0]
#                elif edge_idx == 2:
#                    projected_point[1] = point[1]
#                elif edge_idx == 3:
#                    projected_point[0] = point[0]
#                    
#                ### Have to make sure that point has not left square
#                if edge_idx == 0:
#                    if projected_point[0] < square_coords[0][0]:
#                        projected_point[0] = point[0]
#                    elif projected_point[0] > square_coords[3][0]:
#                        projected_point[0] = point[0]
#                elif edge_idx == 1:
#                    if projected_point[1] > square_coords[1][1]:
#                        projected_point[1] = point[1]
#                    elif projected_point[1] < square_coords[0][1]:
#                        projected_point[1] = point[1]
#                if edge_idx == 2:
#                    if projected_point[0] < square_coords[1][0]:
#                        projected_point[0] = point[0]
#                    elif projected_point[0] > square_coords[2][0]:
#                        projected_point[0] = point[0]
#                if edge_idx == 3:
#                    if projected_point[1] > square_coords[2][1]:
#                        projected_point[1] = point[1]
#                    elif projected_point[1] < square_coords[3][1]:
#                        projected_point[1] = point[1]
#                
#                temp_projected_point_list.append(projected_point)
#            
#            ### Decide which radius to use for this edge projection
#            ### The one to use is the one that leads to the largest area
#            ### As long as the edge has actually moved
#            temp_area_list = []
#            active_vertices = masked_grid[temp_surface_idx]
#            for entry in temp_projected_point_list:
#                if np.linalg.norm(entry - temp_edges[edge_idx]) < 1e-6:
#                    temp_area_list.append(-1)
#                    continue
#                
#                test_temp_edges = temp_edges.copy()
#                test_temp_edges[edge_idx] = entry
#                temp_area = tri_area_lookup[tostring(active_vertices.astype(int))](
#                                            test_temp_edges, 
#                                            square_coords)
#                temp_area_list.append(temp_area)
#
#            choice_idx = np.argmax(temp_area_list)
#            projected_point = temp_projected_point_list[choice_idx]
#                
#            ### Save in edges_for_region
#            edges_for_region[temp_surface_idx][edge_idx] = projected_point
#            
#            
#    
#    draw_edges = []
#    draw_vertices = []
#    area_arguments = []
#    area_values = []
#    calc_area = 0
#    for idx,row in enumerate(masked_grid):
#        row_string = tostring(row.astype(int))
#        
#        row_coords = coords[marching_square_idx[idx]]
#        ## Use hand crafted edges
#        temp_edges = edges_for_region[idx]
#        
#        connect_edges_bool = tri_lookup[row_string]
#    
#        temp_adjusted_area = tri_area_lookup[tostring(row.astype(int))](
#                temp_edges, row_coords)
#        
#        area_arguments.append((
#                tostring(row.astype(int)),
#                temp_edges, 
#                row_coords))
#        area_values.append(temp_adjusted_area)
#        
#        calc_area += temp_adjusted_area
#        
#        ## For making marching gif
#        temp_draw_edges = []
#        for temp_mask in connect_edges_bool:  
#            temp_masked_edges = temp_edges[temp_mask]
#            temp_draw_edges.append(temp_masked_edges)
#        draw_edges.append(temp_draw_edges)
#        
#        draw_vertices.append(row_coords)
#    
#    #### Plot the result
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    circle_grid = mc.coords[grid_region.ravel().astype(bool)]
#    plot_grid(mc.coords, 
#              ax=ax)
#    plot_grid(circle_grid,
#              ax=ax,
#              c="tab:green")
#    for idx,center in enumerate(mc.centers):
#        plot_circle(mc.radii[idx], center=center, ax=ax)
#        
#    for value in draw_edges:
#        for entry in value:
#            if len(entry) > 0:
#                ax.plot(entry[:,0], entry[:,1], 
#                        c="k", linewidth=2)
#                
##    ax.set_xlim([0.5,1])
##    ax.set_ylim([-1.0,-0.5])
##                
#    
#                
#    
#    
#    plt.show()
#    plt.close()
#    
#    area_list.append(calc_area)
#    
#    break
    

#area_list = np.array(area_list)
#area_list = area_list / target_area

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(space_range,area_list)
#ax.axhline(1, c="tab:red")
##ax.axhline(1.2*1.2*np.pi, c="tab:red")
##ax.set_ylim([4.40, 4.70])
#plt.show()
#plt.close()

#square_coords_list = coords[marching_square_idx]
#square_area = spacing*spacing
#square_idx = 218
#r_idx = 0
#for idx,values in enumerate(square_coords_list[square_idx:]):
#    actual_idx = idx + square_idx
#    
#    values = np.vstack(values)
#    
#    fig = plt.figure(figsize=(6,6))
#    ax = fig.add_subplot(111)
#    plot_grid(coords, ax=ax)
#    
#    for idx,center in enumerate(mc.centers):
#        plot_circle(mc.radii[idx], center=center, ax=ax)
#        
#    plot_grid(values, 
#              c=["tab:red", "tab:green", "tab:purple", "tab:cyan"],
#              edgecolor="k",
#              ax=ax)
#    
#    edges = edges_for_region[actual_idx]
#    
#    plot_grid(edges, 
#              c="tab:orange",
#              edgecolor="k",
#              ax=ax)
#    
#    for idx,edge in enumerate(edges):
#        ax.text(edge[0],edge[1],
#                "{}".format(idx))
#        
##    temp_proj = np.vstack(proj_list[r_idx][actual_idx])
##    ax.scatter(temp_proj[:,0], temp_proj[:,1],
##               c="tab:pink",
##               s=50)
#    
#    temp_area = area_values[actual_idx]
#    
#    ax.text(0,0, "{:.2f}".format(temp_area/square_area),
#            fontsize=16)
#    
#    plt.gca().set_aspect('equal', adjustable='box')
#    plt.show()
#    plt.close()
#
#    break







