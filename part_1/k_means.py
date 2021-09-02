import numpy as np
from sklearn.metrics import pairwise_distances

class KMeans: 
    '''
        K-Means clustering algorithm using euclidean and mahalanobis distance
    '''

    def __init__(self, n_clusters, max_iterations=200, distance='euclidean', seed=2):
        '''
            Inputs: 
            - n_clusters: number of clusters to find.
            - max_iterations: maximum number of iterations to run.
            - distance: type of distance euclidean/mahalanobis
        '''
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.distance =  distance
        self.labels = None
        self.centroids = None
        self.seed = seed
    
    def initialize_centroids(self, data):
        '''
            Randomly choose n_clusters data points as our initial centroids
            Inputs: 
            - data: the array of points  
        '''
        # Set the seed
        np.random.seed(self.seed)

        # Number of points
        num_points = data.shape[0]

        # Pick n_cluster indices
        indices = np.random.randint(0, num_points, self.n_clusters)

        # Get the centroids
        centroids = data[indices,:]

        return centroids

    def adjust_centroids(self, data, current_labels):
        '''
            One the points are assigned, adjust the centroids
            Inputs:
            - data: the array of points 
            - current_labels: list of current labels for the data points
        '''
        new_centroids = []

        for x in range(self.n_clusters):
            # Get the new centroid
            centroid = data[current_labels == x].mean(axis = 0)

            # Add it to the list of centroids
            new_centroids.append(np.ravel(centroid))

        return np.array(new_centroids)

    def mahalanobis(self, data, centroid, vi):
        '''
            Calculate the mahalanobis distance between the points and the centroid
            Inputs:
            - data: the array of points 
            - centroid: center point for the distance 
            - vi: inverse of the covaraince matrix
        '''
        mean_difference = data - np.mean(centroid)
        temp = np.dot(mean_difference, vi)
        distance = np.dot(temp, mean_difference.transpose())
        return np.sqrt(distance.diagonal().reshape(len(data),1))
    
    def calculate_distance(self, data, centroids, covariance):
        '''
            Calculate the distance between the points and the centroids
            Inputs:
            - data: the array of points 
            - centroids: center points for the clusters 
            - distance: type of distance euclidean/mahalanobis
        '''
        # Euclidean Distance
        if self.distance == "euclidean":
            return pairwise_distances(data, centroids, metric=self.distance)

        # Malahanobis Distance

        # if we don't have previous assignments yet
        if covariance is None:
            return pairwise_distances(data, centroids, metric=self.distance, VI=np.linalg.inv(np.cov(data.transpose())))

        result = []

        # Get the distance between each point and each cluster centroid
        for (cov,centroid) in zip(covariance, centroids):
            distance = self.mahalanobis(data=data, centroid=centroid, vi=np.linalg.inv(cov))
            result = distance if len(result) == 0 else np.concatenate((result, pairwise_distances(data, [centroid], metric=self.distance, VI=np.linalg.inv(cov))), axis=1)
        
        return result

    def assign_clusters(self, data, centroids, covariance):
        '''
            Assign points to a cluster based off their distance
            Inputs:
            - data: the array of points 
            - centroids: center points for the clusters 
            - distance: type of distance euclidean/mahalanobis
        '''
        # Compute distances between each data point and the set of centroids, based on the distance selected:
        distances_from_centroids = self.calculate_distance(data=data, centroids=centroids, covariance=covariance)
        
        # Return cluster assignments
        return np.argmin(distances_from_centroids, axis=1)

    def get_covariance_matrices(self, data, current_assignments):
        '''
            Get the covarince matrix for each cluster
            Inputs:
            - data: the array of points 
            - current_assignments: the current assignments of each point for each cluster
        '''
        cov = []

        for x in range(self.n_clusters):
            temp = np.delete(data, np.where(current_assignments == x), axis = 0)
            cov.append(np.cov(temp.transpose()))

        return cov

    def fit(self, data):
        '''
            Runs k-means on given data
            Inputs:
            - data: the array of points 
        '''

        # Initialize centroids
        centroids = self.initialize_centroids(data=data)
        prev_assignment = None
        covariance = None

        for x in range(self.max_iterations):
            if prev_assignment is not None:
                covariance = self.get_covariance_matrices(data=data, current_assignments=prev_assignment)

            # Make cluster assignments
            current_assignments = self.assign_clusters(data=data, centroids=centroids, covariance=covariance)

            # Adjust the centroids
            centroids = self.adjust_centroids(data=data, current_labels=current_assignments)

            # Check for convergence
            if prev_assignment is not None and (prev_assignment==current_assignments).all():
                break

            prev_assignment = current_assignments[:]

        # Set the labels
        self.labels = current_assignments
        self.centroids = centroids