import numpy as np
from scipy.spatial import distance
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from part_1.k_means import KMeans

class DaviesBouldinIndex:
    '''
        Use KMeans to cluster unknown data
    '''

    def __init__(self, data, max_iterations=10):
        '''
            Inputs: 
            - data: the data we want to find clusters for
            - max_iterations: the maximum amount of clusters we want to test k=[2, max_iterations+2)
        '''
        self.data = data
        self.max_iterations = max_iterations
        self.n_clusters = None
        self.db_index = None
        self.labels = None
        self.indices = []
    
    def compute_DB_index(self, labels, centroids, n_cluster):
        '''
            Compute the db index of a given partition
            Inputs: 
            - labels: the current label assignments
            - centroids: the current centroids
            - n_clusters: the number of clusters we ran k-means for 
        '''
        clusters = [self.data[labels == k] for k in range(n_cluster)]

        # calculate within cluster distance
        D = [np.mean([distance.euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(clusters)]
        
        # within-to-between cluster ratios
        ratios = []

        for i in range(n_cluster):
            D_ij_list = []
            
            # between cluster distances
            for j in range(n_cluster):
                # don't calculate between a cluster and itself
                if i != j:
                    D_ij = (D[i] + D[j]) / distance.euclidean(centroids[i], centroids[j])
                    D_ij_list.append(D_ij)
            
            # keep track of the worst case within-to-between cluster ratio for cluster i
            ratios.append(max(D_ij_list)) 

        # get mean of all ratio values    
        return np.mean(ratios)
    
    def model_selection(self):
        '''
            Run k_means using a variety of cluster numbers and choose the one with the lowest db_index
        '''
        for i in range(2, self.max_iterations+2):
            # run k-means for i clusters
            kmeans_euclidean = KMeans(n_clusters=i, distance="euclidean")
            kmeans_euclidean.fit(self.data)

            # calcualte the db index
            index = self.compute_DB_index(kmeans_euclidean.labels,  kmeans_euclidean.centroids, i)
            self.indices.append(index)

            # if the index is lower, replace our current index, number of clusters, and labels
            if self.db_index is None or index < self.db_index:
                self.db_index = index
                self.n_clusters = i
                self.labels = kmeans_euclidean.labels