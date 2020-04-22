# -*- coding: utf-8 -*-


import numpy as np



def basic_cube():
    """
    Cube based on ordering in program
    """
    return np.array([
       [-7.156285  , -3.80337925, -1.95817204],
       [-7.156285  , -3.80337925, -1.70817204],
       [-7.156285  , -3.55337925, -1.70817204],
       [-7.156285  , -3.55337925, -1.95817204],
       [-6.906285  , -3.80337925, -1.95817204],
       [-6.906285  , -3.80337925, -1.70817204],
       [-6.906285  , -3.55337925, -1.70817204],
       [-6.906285  , -3.55337925, -1.95817204]])


def compute_edge_sites(cube_vertex):
    pair_idx = np.array([
            [0,1],
            [0,3],
            [2,3],
            [1,2],
            
            [0,4],
            [3,7],
            [2,6],
            [1,5],
            
            [4,5],
            [4,7],
            [6,7],
            [5,6],
            
            ])
    
    pairs = cube_vertex[pair_idx]
    edge = np.mean(pairs, axis=1)
    return edge


def unit_cube():
    return np.array([
            [0,0,0],
            [0,0,1],
            [0,1,1],
            [0,1,0],
            [1,0,0],
            [1,0,1],
            [1,1,1],
            [1,1,0]
            ])


def all_operations_vertex():
    
    def rot_opposite_faces_x(idx):
        return idx[[4,0,3,7,5,1,2,6]]

    def rot_opposite_faces_y(idx):
        return idx[[3,0,1,2,7,4,5,6]]
    
    def rot_opposite_faces_z(idx):
        return idx[[4,5,1,0,7,6,2,3]]
    
    def rot_cart_frame(idx):
        return idx[[0,4,5,1,3,7,6,2]]
    
    def rot_opposite_edges(idx):
        return idx[[4,7,6,5,0,3,2,1]]
    
    start_idx = np.arange(0,8)
    idx_list = [start_idx]
    for i in range(3):
        idx_list.append(rot_opposite_faces_x(idx_list[-1]))
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(3):
            rot_idx_list.append(rot_opposite_faces_y(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(3):
            rot_idx_list.append(rot_opposite_faces_z(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(2):
            rot_idx_list.append(rot_cart_frame(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(2):
            rot_idx_list.append(rot_opposite_edges(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    all_idx = np.vstack(idx_list)
#    all_idx = np.unique(all_idx,axis=0)
    return all_idx

def all_operations_edge(idx_list):
    
    def rot_opposite_faces_x(idx):
        return idx[[4,9,5,1,8,10,2,0,7,11,6,3]]

    def rot_opposite_faces_y(idx):
        return idx[[1,2,3,0,5,6,7,4,9,10,11,8]]
    
    def rot_opposite_faces_z(idx):
        return idx[[8,4,0,7,9,1,3,11,10,5,2,6]]
    
    def rot_cart_frame(idx):
        return idx[[4,0,7,8,1,3,11,9,5,2,6,10]]
    
    def rot_opposite_edges(idx):
        return idx[[9,8,11,10,4,7,6,5,1,0,3,2]]
    
    start_idx = np.arange(0,12)
    idx_list = [start_idx]
    for i in range(3):
        idx_list.append(rot_opposite_faces_x(idx_list[-1]))
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(3):
            rot_idx_list.append(rot_opposite_faces_y(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(3):
            rot_idx_list.append(rot_opposite_faces_z(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(2):
            rot_idx_list.append(rot_cart_frame(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    temp_idx_list = []
    for entry in idx_list:
        rot_idx_list = [entry]
        for i in range(2):
            rot_idx_list.append(rot_opposite_edges(rot_idx_list[-1]))
        temp_idx_list += rot_idx_list
    idx_list += temp_idx_list
    
    all_idx = np.vstack(idx_list)
#    all_idx = np.unique(all_idx,axis=0)
    
    return all_idx


def apply_vertex_symmetry(vertex_idx):
    #### Let's perform all rotations on lookup idx first
    
    symmetry_idx_list = all_operations_vertex()
    
    all_vertex_idx = []
    
    for idx_list in symmetry_idx_list:
        all_vertex_idx.append(vertex_idx[idx_list])
    
    return all_vertex_idx

def apply_edge_symmetry(edge_idx):
        
        
    ### Construct corresponding symmetry relevant ordings for vertex/edge 
    ### for triangulation
    edge_symmetry_idx_list = all_operations_edge(edge_idx)
    edge_symmetry_idx_list = np.array(edge_symmetry_idx_list)
    
    all_edge_idx = []
    for row in edge_symmetry_idx_list:  
        all_edge_idx.append(edge_idx[:,row])
    
    return all_edge_idx



########### Let's build the vertex lookup table
all_comb = np.meshgrid([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1])
all_comb = np.c_[
        all_comb[0].ravel(),
        all_comb[1].ravel(),
        all_comb[2].ravel(),
        all_comb[3].ravel(),
        all_comb[4].ravel(),
        all_comb[5].ravel(),
        all_comb[6].ravel(),
        all_comb[7].ravel()]
vertex_lookup = np.zeros((2,2,2,2,2,2,2,2,12))
    

def tostring(array):
    """
    1D array to string
    """
    return ",".join([str(x) for x in array])

def fromstring(array_str):
    return np.fromstring(array_str, dtype=int, sep=",")

## Program fourteen primitives
## https://www.researchgate.net/publication/3410984_Brodlie_K_Improving_the_robustness_and_accuracy_of_the_marching_cubes_algorithm_for_isosurfacing_IEEE_Trans_Viz_and_Comput_Graph_91_16-29/figures?lo=1

#### For holding information to operate on using symmetry operations and store 
#### tri connectivity
vertex_mask_idx = np.zeros((15,8)).astype(int)
tri_mask = np.zeros((16,12))

#### Build connectivity dict for these simple cases
#### Each entry is 2D array with one entry per connectivity and number of entries
#### equal to the number of triangles
tri_connectivity = {}

#### Same as tri_connectivity but populated with volume information for volume
#### adjustments to be made for each type. 
#### Valued entered is a ratio out of 1 with respect to the volume of the 
#### voxel that the entry adds. 
tri_volume = {}



#### 1. First entry all zeros
entry = vertex_mask_idx[0]
tri_connectivity[tostring(entry)] = np.zeros((1,12))
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
## Set volume
tri_volume[tostring(entry)] = 0
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 2. Simple Triangle
vertex_mask_idx[1,[0]] = 1
tri_mask[1,[0,1,4]] = 1
entry = vertex_mask_idx[1]
tri_connectivity[tostring(entry)] = np.zeros((1,12))
tri_connectivity[tostring(entry)][0][[0,1,4]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 3. Simple Plane down
vertex_mask_idx[2,[0,4]] = 1
tri_mask[2,[0,1,8,9]] = 1
entry = vertex_mask_idx[2]
tri_connectivity[tostring(entry)] = np.zeros((2,12))
tri_connectivity[tostring(entry)][0][[0,1,9]] = 1
tri_connectivity[tostring(entry)][1][[0,8,9]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.125
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 4. Across face double triangle
vertex_mask_idx[3,[0,5]] = 1
## First Tri
tri_mask[3,[0,1,4]] = 1
## Second Tri
tri_mask[3,[7,8,11]] = 1
entry = vertex_mask_idx[3]
tri_connectivity[tostring(entry)] = np.zeros((2,12))
tri_connectivity[tostring(entry)][0][[0,1,4]] = 1
tri_connectivity[tostring(entry)][1][[7,8,11]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 2*0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 5. Across body double triangle
vertex_mask_idx[4,[0,6]] = 1
## First Tri
tri_mask[4,[0,1,4]] = 1
## Second Tri
tri_mask[4,[6,10,11]] = 1
entry = vertex_mask_idx[4]
tri_connectivity[tostring(entry)] = np.zeros((2,12))
tri_connectivity[tostring(entry)][0][[0,1,4]] = 1
tri_connectivity[tostring(entry)][1][[6,10,11]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 2*0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 6. Three Bottom Corners
vertex_mask_idx[5,[3,4,7]] = 1
tri_mask[5,[1,4,8,10,2]] = 1
entry = vertex_mask_idx[5]
tri_connectivity[tostring(entry)] = np.zeros((3,12))
tri_connectivity[tostring(entry)][0][[1,4,8]] = 1
tri_connectivity[tostring(entry)][1][[1,8,2]] = 1
tri_connectivity[tostring(entry)][2][[2,8,10]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.35416667
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 7. One plane down one tri
vertex_mask_idx[6,[0,4,6]] = 1
## Plane down
tri_mask[6,[0,8,1,9]] = 1
## Upper Tri 6
tri_mask[6,[6,10,11]] = 1
entry = vertex_mask_idx[6]
tri_connectivity[tostring(entry)] = np.zeros((3,12))
tri_connectivity[tostring(entry)][0][[0,1,9]] = 1
tri_connectivity[tostring(entry)][1][[0,8,9]] = 1
tri_connectivity[tostring(entry)][2][[6,10,11]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.125+0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 8. Triple Tri
vertex_mask_idx[7,[1,4,6]] = 1
## Tri 1
tri_mask[7,[0,3,7]] = 1
## Tri 4
tri_mask[7,[4,8,9]] = 1
## Tri 6
tri_mask[7,[6,10,11]] = 1
entry = vertex_mask_idx[7]
tri_connectivity[tostring(entry)] = np.zeros((3,12))
tri_connectivity[tostring(entry)][0][[0,3,7]] = 1
tri_connectivity[tostring(entry)][1][[4,8,9]] = 1
tri_connectivity[tostring(entry)][2][[6,10,11]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 3*0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 9. Middle Plane
vertex_mask_idx[8,[0,3,4,7]] = 1
## Mid Plane
tri_mask[8,[0,2,8,10]] = 1
entry = vertex_mask_idx[8]
tri_connectivity[tostring(entry)] = np.zeros((2,12))
tri_connectivity[tostring(entry)][0][[0,8,10]] = 1
tri_connectivity[tostring(entry)][1][[0,2,10]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.5
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 10. Hexagon
vertex_mask_idx[9,[0,2,3,7]] = 1
## Hexagon
tri_mask[9,[0,3,4,6,9,10]] = 1
entry = vertex_mask_idx[9]
tri_connectivity[tostring(entry)] = np.zeros((4,12))
tri_connectivity[tostring(entry)][0][[0,3,6]] = 1
tri_connectivity[tostring(entry)][1][[0,6,10]] = 1
tri_connectivity[tostring(entry)][2][[0,9,10]] = 1
tri_connectivity[tostring(entry)][3][[0,4,9]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.375
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 11. Double Plane
vertex_mask_idx[10,[0,1,6,7]] = 1
## Plane 1
tri_mask[10,[1,3,4,7]] = 1
## Plane 2
tri_mask[10,[5,6,9,11]] = 1
entry = vertex_mask_idx[10]
tri_connectivity[tostring(entry)] = np.zeros((4,12))
tri_connectivity[tostring(entry)][0][[1,3,7]] = 1
tri_connectivity[tostring(entry)][1][[1,4,7]] = 1
tri_connectivity[tostring(entry)][2][[5,6,11]] = 1
tri_connectivity[tostring(entry)][3][[5,9,11]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.75
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 12. 
vertex_mask_idx[11,[0,3,6,7]] = 1
## Plane
tri_mask[11,[0,2,4,6,9,11]] = 1
entry = vertex_mask_idx[11]
tri_connectivity[tostring(entry)] = np.zeros((4,12))
tri_connectivity[tostring(entry)][0][[4,9,11]] = 1
tri_connectivity[tostring(entry)][1][[2,6,11]] = 1
tri_connectivity[tostring(entry)][2][[0,2,4]] = 1
tri_connectivity[tostring(entry)][3][[2,4,11]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.375
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 13. 6+tri
vertex_mask_idx[12,[1,3,4,7]] = 1
## 6 Plane
tri_mask[12,[1,4,8,10,2]] = 1
## Tri 1
tri_mask[12,[0,3,7]] = 1
entry = vertex_mask_idx[12]
tri_connectivity[tostring(entry)] = np.zeros((4,12))
tri_connectivity[tostring(entry)][0][[1,4,8]] = 1
tri_connectivity[tostring(entry)][1][[1,8,2]] = 1
tri_connectivity[tostring(entry)][2][[2,8,10]] = 1
tri_connectivity[tostring(entry)][3][[0,3,7]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.52083333-0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 14. Quad Tri
vertex_mask_idx[13,[0,2,5,7]] = 1
## Tri 0
tri_mask[13,[0,1,4]] = 1
## Tri 2
tri_mask[13,[2,3,6]] = 1
## Tri 5
tri_mask[13,[7,8,11]] = 1
## Tri 7
tri_mask[13,[5,9,10]] = 1
entry = vertex_mask_idx[13]
tri_connectivity[tostring(entry)] = np.zeros((4,12))
tri_connectivity[tostring(entry)][0][[0,1,4]] = 1
tri_connectivity[tostring(entry)][1][[2,3,6]] = 1
tri_connectivity[tostring(entry)][2][[7,8,11]] = 1
tri_connectivity[tostring(entry)][3][[5,9,10]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 4*0.02083333
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### 15. 
vertex_mask_idx[14,[2,3,4,7]] = 1
entry = vertex_mask_idx[14]
tri_connectivity[tostring(entry)] = np.zeros((4,12))
tri_connectivity[tostring(entry)][0][[1,3,4]] = 1
tri_connectivity[tostring(entry)][1][[4,3,10]] = 1
tri_connectivity[tostring(entry)][2][[3,6,10]] = 1
tri_connectivity[tostring(entry)][3][[4,8,10]] = 1
## Set Opposite
not_entry = np.logical_not(entry).astype(int)
tri_connectivity[tostring(not_entry)] = tri_connectivity[tostring(entry)]
### Set volume
tri_volume[tostring(entry)] = 0.375
tri_volume[tostring(not_entry)] = 1-tri_volume[tostring(entry)]


#### Performing rotations to populate the entire tri_connectivity
iterations = [(keys,values) for keys,values in tri_connectivity.items()]
for key,value in iterations:
    key_array = fromstring(key)
    all_vertex = apply_vertex_symmetry(key_array)
    all_edge = apply_edge_symmetry(value)
    
    for temp_idx,vertex in enumerate(all_vertex):
        tri_connectivity[tostring(vertex)] = all_edge[temp_idx]

iterations = [(keys,values) for keys,values in tri_volume.items()]
for key,value in iterations:
    key_array = fromstring(key)
    all_vertex = apply_vertex_symmetry(key_array)
    
    for temp_idx,vertex in enumerate(all_vertex):
        tri_volume[tostring(vertex)] = value
        

#### Plotting all primitives
def plot_primitives(figname="marching_cubes_primitive.png"):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    cart_points = basic_cube()
    fig = plt.figure(figsize=(24,24))
    
    for entry_idx,vertex_row in enumerate(vertex_mask_idx):
        ax = fig.add_subplot(4,4,entry_idx+1, projection='3d')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.scatter(cart_points[:,0][0:8],
                   cart_points[:,1][0:8],
                   cart_points[:,2][0:8])
        
        ## Add numbering
        for idx,point in enumerate(cart_points[0:8]):
            ax.text(point[0],
                   point[1], 
                   point[2],
                   "{}".format(idx),
                   fontsize=16)
            
        cube_vertex = cart_points[:8]
        edge_vertex = compute_edge_sites(cube_vertex)
        
        #### Visualize edge points
        ax.scatter(edge_vertex[:,0],
                   edge_vertex[:,1],
                   edge_vertex[:,2],
                   edgecolor="k",
                   facecolor="tab:red")
        
        ## Number edge cites
        for idx,point in enumerate(edge_vertex):
            ax.text(point[0],
                   point[1], 
                   point[2],
                   "{}".format(idx),
                   fontsize=16)
        
        ## Plot relevant vertices
        vertex_row_bool = vertex_row.astype(bool)
        temp_vertex = cart_points[vertex_row_bool,:]
        if len(temp_vertex) > 0:
            ax.scatter(
                    temp_vertex[:,0],
                    temp_vertex[:,1],
                    temp_vertex[:,2],
                    c="tab:green",
                    s=100)
        ## Tri idx
        entry = tostring(vertex_row)
        triangles_bool = tri_connectivity[entry].astype(bool)
        array_to_mask = np.repeat(np.arange(0,12)[None,:], 
                                triangles_bool.shape[0], 
                                axis=0)
        tri_idx = array_to_mask[triangles_bool].reshape(-1,3)
        
        if len(tri_idx) != 0:
            ax.plot_trisurf(
                    edge_vertex[:,0],
                    edge_vertex[:,1],
                    edge_vertex[:,2],
                    triangles=tri_idx)
    
    fig.savefig(figname, 
                dpi=400)


##### Plotting all in tri_connectivity
def plot_all_cubes(figname="all_marching_cubes.pdf"):
    cart_points = basic_cube()
    fig = plt.figure(figsize=(48,192))
    
    entry_idx = 0
    for key,value in tri_connectivity.items():
        ax = fig.add_subplot(32,8,entry_idx+1, projection='3d')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.scatter(cart_points[:,0][0:8],
                   cart_points[:,1][0:8],
                   cart_points[:,2][0:8])
        
        ## Add numbering
        for idx,point in enumerate(cart_points[0:8]):
            ax.text(point[0],
                   point[1], 
                   point[2],
                   "{}".format(idx),
                   fontsize=16)
            
        cube_vertex = cart_points[:8]
        edge_vertex = compute_edge_sites(cube_vertex)
        
        #### Visualize edge points
        ax.scatter(edge_vertex[:,0],
                   edge_vertex[:,1],
                   edge_vertex[:,2],
                   edgecolor="k",
                   facecolor="tab:red")
        
        ## Number edge cites
        for idx,point in enumerate(edge_vertex):
            ax.text(point[0],
                   point[1], 
                   point[2],
                   "{}".format(idx),
                   fontsize=16)
            
        ## Plot Triangle
        triangles_bool = value.astype(bool)
        array_to_mask = np.repeat(np.arange(0,12)[None,:], 
                                triangles_bool.shape[0], 
                                axis=0)
        tri_idx = array_to_mask[triangles_bool].reshape(-1,3)
        
        if len(tri_idx) != 0:
            ax.plot_trisurf(
                    edge_vertex[:,0],
                    edge_vertex[:,1],
                    edge_vertex[:,2],
                    triangles=tri_idx)
        
        entry_idx += 1
    
    fig.savefig("all_marching_cubes.pdf")



##### Deriviing volumes for peach primitive
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#
#cart_points = unit_cube()


### 15 and 12 are hard to evaluate volume using this method because they have
### planes in Z direction. However, these contribute 0 volume so they can be 
### ignored. Just have to ignore by hand
#for entry_idx,vertex_row in enumerate(vertex_mask_idx[11:]):
#    fig = plt.figure()
#    ax = fig.add_subplot(111,projection='3d')
#    ax.set_xticks([])
#    ax.set_yticks([])
#    ax.set_zticks([])
#    ax.scatter(cart_points[:,0][0:8],
#               cart_points[:,1][0:8],
#               cart_points[:,2][0:8])
#    
#    ## Add numbering
#    for idx,point in enumerate(cart_points[0:8]):
#        ax.text(point[0],
#               point[1], 
#               point[2],
#               "{}".format(idx),
#               fontsize=16)
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
#    ## Number edge cites
#    for idx,point in enumerate(edge_vertex):
#        ax.text(point[0],
#               point[1], 
#               point[2],
#               "{}".format(idx),
#               fontsize=16)
#    
#    ## Plot relevant vertices
#    vertex_row_bool = vertex_row.astype(bool)
#    temp_vertex = cart_points[vertex_row_bool,:]
#    if len(temp_vertex) > 0:
#        ax.scatter(
#                temp_vertex[:,0],
#                temp_vertex[:,1],
#                temp_vertex[:,2],
#                c="tab:green",
#                s=100)
#    ## Tri idx
#    entry = tostring(vertex_row)
#    triangles_bool = tri_connectivity[entry].astype(bool)
#    array_to_mask = np.repeat(np.arange(0,12)[None,:], 
#                            triangles_bool.shape[0], 
#                            axis=0)
#    tri_idx = array_to_mask[triangles_bool].reshape(-1,3)
#    
#    if len(tri_idx) != 0:
#        ax.plot_trisurf(
#                edge_vertex[:,0],
#                edge_vertex[:,1],
#                edge_vertex[:,2],
#                triangles=tri_idx)
#    
##    tet = [[cart_points[0], edge_vertex[0], edge_vertex[1], edge_vertex[4]],
##           [cart_points[4], edge_vertex[4], edge_vertex[8], edge_vertex[9]],
##           [cart_points[7], edge_vertex[5], edge_vertex[10], edge_vertex[9]],
##           [cart_points[3], edge_vertex[1], edge_vertex[2], edge_vertex[5]],
##           [cart_points[0], edge_vertex[0], edge_vertex[1], edge_vertex[4]]]
##    
##    tetrahedron_volume(tet[0][0],tet[0][1],tet[0][2],cart_points[0])
#    all_tri = edge_vertex[tri_idx]
#    vol = 0
#    for tri_entry in all_tri:
#        xyz = tri_entry
#        d = scipy.spatial.Delaunay(xyz[:,:2])
#        tri = xyz[d.vertices]
#        a = tri[:,0,:2] - tri[:,1,:2]
#        b = tri[:,0,:2] - tri[:,2,:2]
#        proj_area = np.cross(a, b).sum(axis=-1)
#        zavg = tri[:,:,2].sum(axis=1)
#        vol += np.abs(zavg * np.abs(proj_area) / 6.0)
#    
#    print(vol)
#        
##    equal_axis_aspect(ax)
##    tetrahedron_volume(tet[0][0],tet[0][1],tet[0][2],temp_vertex[0]) / 0.015625
#        
#    break