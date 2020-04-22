# -*- coding: utf-8 -*-

import numpy as np


def calc_distance_matrix(struct_coll, prop_name='RDF_vector'):
    '''
    Purpose:
        Wrapper for calculating the distance matrix. Will dynamically choose
          to call for a list or a dictionary depending on the struct_coll
          data type.
    '''
    if type(struct_coll) == list:
        dist_matrix = calc_distance_matrix_list(struct_coll, prop_name)
        return dist_matrix
    elif type(struct_coll) == dict:
        print('Dict implementation is not available. Please try using a list')
    return

def calc_distance_matrix_list(struct_list, prop_name='RDF_vector'):
    '''
    Purpose:
        Calculates and return the distance matrix given a list of structure 
          objects.
    1. Prepares matrix with samples x feature vector
    2. Returns matrix of the euclidean distance between all of the samples
    
    '''
    feature_matrix = np.array([struct_list[0].get_property(prop_name)])
    for struct_obj in struct_list[1:]:
        feature_vector = np.array([struct_obj.get_property(prop_name)])
        feature_matrix = np.concatenate([feature_matrix,feature_vector],axis=0)
    dist_matrix = calc_euclidean_dist_vectorized(feature_matrix)
    return dist_matrix

def construct_feature_matrix(struct_list,prop_name='RDF_vector'):
    feature_matrix = np.array([struct_list[0].get_property(prop_name)])
    for struct_obj in struct_list[1:]:
        feature_vector = np.array([struct_obj.get_property(prop_name)])
        feature_matrix = np.concatenate([feature_matrix,feature_vector],axis=0)
    return feature_matrix

def calc_atom_dist(geo_array):
    '''
    Purpose:
        Calculates the distance between atom coordinates in three dimensions. 
    Arguments:
        geo_array: nx3 matrix of atom coordinates
    '''
    num_atoms = np.shape(geo_array)[0]
    geometry_x = geo_array[:,0].reshape(num_atoms,1)
    geometry_y = geo_array[:,1].reshape(num_atoms,1)
    geometry_z = geo_array[:,2].reshape(num_atoms,1)
    dist_x = calc_euclidean_dist_vectorized_squared(geometry_x)
    dist_y = calc_euclidean_dist_vectorized_squared(geometry_y)
    dist_z = calc_euclidean_dist_vectorized_squared(geometry_z)
    dist_matrix = np.sqrt(dist_x+dist_y+dist_z)
    return dist_matrix
    
def calc_euclidean_dist_vectorized(feature_matrix):
    '''
    Purpose:
        Using vectorized operations, calculates the euclidean distance
          between all of the input feature vectors.
    Description:
        Calculates distance as (x-y)^2 = x^2+y^2-2xy 
        x^2 = np.sum(rows of feature matrix)
        y^2 = np.sum(rows of feature matrix) -> transform to column matrix
        xy = np.dot(feature_matrix, feature_matrix.T) to get covariance matrix
              which will an n x n matrix.
    Arguments:
        feature_matrx: np.array().shape = samples x features
    Notes:
        There's an issue here with np.sqrt. If the number is negative due to 
          numerical error, then the function will return NaN instead of 
          rounding to zero. np.clip was added to take care of this case,
          but it could be computationally inefficient. That hasn't been tested.
    '''
    dist_matrix = -2*np.dot(feature_matrix, feature_matrix.T) + \
                  np.sum(np.square(feature_matrix),axis=1) + \
                  np.sum(np.square(feature_matrix),axis=1)[:,np.newaxis]
    dist_matrix = dist_matrix.clip(min=0)
    dist_matrix = np.sqrt(dist_matrix)
    return dist_matrix

def calc_euclidean_dist_vectorized_squared(feature_matrix):
    '''
    Purpose:
        Same as calc_euclidean_dist_vectorized but without square root. 
    '''
    dist_matrix = -2*np.dot(feature_matrix, feature_matrix.T) + \
                  np.sum(np.square(feature_matrix),axis=1) + \
                  np.sum(np.square(feature_matrix),axis=1)[:,np.newaxis]
    dist_matrix = dist_matrix.clip(min=0)
    return dist_matrix