#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 16:50:35 2018

@author: roy
"""

import numpy as np

def cluster_assignment(data, centroids):
    '''
    Input:
    data - a nPixel x 3 matrix containing the RGB channels of all the data points (pixels), where nPixel = data.shape[0]
    centroids - a K x 3 matrix containing the RGB channels of K centroids, where K = centroids.shape[0]
    
    Output:
    idx - a nPixel-dimensional vector containing the indices of assigned centroids of all the data points (pixels)
    '''
    
    # Commented computations are very inefficient
    # To compute squared Euclidean distances between all data points and all centroids in several steps
    # (Vectorized) calculation of the vector differences (in R^3) of all data points from all centroids
#    difference = data - centroids[:, None]     # shape = (K, nPixel, 3)
#    
#    # Take elementwise square of the above differences, and compute squared Euclidean distances between all data points and all centroids
#    squaredDifference = difference ** 2
#    squaredEuclideanDistance = np.sum(squaredDifference, axis = 2)     # shape = (K, nPixel)

    nPixel = data.shape[0]
    K = centroids.shape[0]
    squaredEuclideanDistance = (np.sum(centroids**2, axis=1).reshape(K, 1) +
                                np.sum(data**2, axis=1).reshape(1, nPixel) - 2 * np.matmul(centroids, data.T))
    
    # Cluster assignment to all the data points
    idx = np.argmin(squaredEuclideanDistance, axis=0)
    
    return idx

def move_centroid(data, idx, K):
    '''
    Input:
    data - a nPixel x 3 matrix containing the RGB channels of all the data points (pixels), where nPixel = data.shape[0]
    idx - a nPixel dimensional vector containing the indices of assigned centroids of all the data points (pixels)
    K - the number of centroids
    
    Output:
    centroids - a K x 3 matrix containing the RGB channels of K updated centroids
    '''    
    
    # Move centroid to the new position
    # For loop over K centroids, but I think there are aggregation & grouping operations for vectorized implementation
    centroids = np.zeros((K, 3))
    for k in range(K):
        centroids[k, :] = np.mean(data[idx == k], axis = 0)
    
    return centroids

def K_means(data, centroids, K, nIteration):
    
    for i in range(nIteration):
        idx = cluster_assignment(data, centroids)
        centroids = move_centroid(data, idx, K)
    
    return idx, centroids

if __name__ == '__main__':
    pass
