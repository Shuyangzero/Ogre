
import numpy as np

from ibslib.io import read,write
from ibslib.descriptor.compute_descriptor import compute_descriptor_dict
from ibslib.analysis.plotting import prepare_prop_from_dict
from ibslib.descriptor.distance_matrix import calc_euclidean_dist_vectorized

def find_nearest_structs(target_struct, struct_dir, 
                         descriptor='rdf', **kwargs):
    """
    Create a list of the nearest structures using one of the available
    descriptors. 
    
    Arguments
    ---------
    target_struct: Structure
        Structure that the user wants to find neighbors of.
    struct_dir: str
        Path to the structure directory to compare the target structure to.
    descriptor: str
        Name of the descriptor in the properties section of each Structure to 
        use for computing distance. 
        
    Returns
    -------
    struct_dict: dict
        Structure dictionary with the descriptor evaluated and stored for all the structures. 
    Results: dict
        Dictionary containing results (see below where it's created)
        
    """
    # Maybe need to make flexible for geometry or structure types
    if type(target_struct) == str:
        target_struct_obj = read(target_struct)
    else:
        pass
    
    ## Check for valid structure directory
    check_struct_dir(struct_dir)
    struct_dict = read(struct_dir) 
    
    results = find_nearest_structs_dict(target_struct_obj, struct_dict, 
                                        descriptor=descriptor,
                                        **kwargs)
    
    return struct_dict,results


def find_nearest_structs_dict(target_struct_obj, struct_dict, 
                              descriptor='rdf',
                              **kwargs):
    
    struct_dict['target'] = target_struct_obj
    compute_descriptor_dict(struct_dict,descriptor=descriptor,**kwargs)
    
    id_list = [x for x in struct_dict.keys()]
    target_index = id_list.index('target')
    descriptor_key = kwargs['descriptor_key']
    descriptor_list = prepare_prop_from_dict(struct_dict, descriptor_key)
    target_descriptor = descriptor_list[target_index]
    
    distances = calc_euclidean_dist_vectorized(np.array(descriptor_list))
    min_index = np.argsort(distances[target_index,:])
    
    # Doing [1:] because 0 will always be the target compared with the target
    min_index = min_index[1:]
    id_sorted = [id_list[x] for x in min_index]
    descriptor_sorted = [descriptor_list[x] for x in min_index]
    
    results = {
                'all_descriptor_distances': distances,
                'struct_id_list': id_list,
                'target_row': target_index,
                'min_column': min_index,
                'nearest_id': id_sorted,
                'nearest_descriptor': descriptor_sorted,
                'target_descriptor': target_descriptor,
                'descriptor_key': descriptor_key
            }
    return results