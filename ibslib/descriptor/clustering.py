# -*- coding: utf-8 -*-

from ibslib.descriptor.distance_matrix import calc_euclidean_dist_vectorized_squared
from sklearn.cluster import AffinityPropagation,KMeans

def run_ap_dist_mat(dist_matrix,**kwargs):
    '''
    Purpose:
        Runs affinity propagation from the distance matrix. Includes all 
          standard arguments from sklearn affinity propagation.
    Returns:
        Cluster labels
    '''
    ap_obj = AffinityPropagation(**kwargs)    
    cluster_labels = ap_obj.fit_predict(dist_matrix)
    
    return cluster_labels 

def run_ap(feature_matrix,**kwargs):
    '''
    Purpose: Will call run_ap after computing the distance matrix for you
    '''
    dist_matrix = calc_euclidean_dist_vectorized_squared(feature_matrix)
    return run_ap_dist_mat(dist_matrix,**kwargs)

def run_KMeans(feature_matrix,**kwargs):
    dist_matrix = calc_euclidean_dist_vectorized_squared(feature_matrix)
    return run_KMeans_dist_mat(dist_matrix,**kwargs)

def run_KMeans_dist_mat(dist_matrix,**kwargs):
    k_model = KMeans(**kwargs)
    k_model.fit(dist_matrix)
    return k_model