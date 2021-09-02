# KMeans

## Part 1-A: KMeans Implementation

I designed it so you could specify what type of distance you want to use, Malahanobis or Euclidean and it will run the algorithm using that distance measurement. I did not have any specific heuristic/strategy for picking initial centroids, I chose random indices but I used a specific seed so I could get consistent results. After each round of new assignments I adjust the centroids to reflect the change of the cluster assignments.

Depending on what the distance was set as for the instance of KMeans I would calculate the distance differently. I used `sklearn.metrics` implementation for the Euclidean distance and my own implementation for Mahalanobis. In the Euclidean distance case it just computes the direct difference between the point and the centroid.

For the Malahanobis case I calculated the difference between each point and the centroid, took the dot product with the inverse covariance matrix of each cluster and then the dot product with the difference again and took the square root.

This is just the equation: $\sqrt{(x-\mu)^TS^{-1}(x-\mu)}$

Since Malahanobis adjusts its distance calculation based off the covariance of each cluster it was important to calculate the covariance of each cluster individually (since they could have different shapes). In the first case where we have no assignments I took the covariance of all the data and used that to calculate the distance for the first round. By doing this I could guarantee the covariance matrix would be positive definite.



## Part 1-B: Comparison

[**Experiment #1**](Graphs/experiment_1.png): In this first comparison I use data that has a clear linear trend but does not have a spherical shape. Here you can see that KMeans using Euclidean distance does not do a very good job classifying the points but the Malahanobis implementation is very close to the ground truth.

[**Experiment #2**](Graphs/experiment_2.png): In this second comparison I use data that has a mostly spherical shape. Here you can see that KMeans using Euclidean distance and KMeans using the Malahanobis distance result in about the same accuracy for the clustering of the data in comparison to the ground truth.

[**Experiment #3**](Graphs/experiment_3.png): In this final comparison I use data that has a moon shape. Here you can see that KMeans using Euclidean distance and KMeans using the Malahanobis distance both are not able to properly cluster the data. The fundamental model assumptions of KMeans is that points will be closer to their own cluster centre than to others. This means that the algorithm will be ineffective if the clusters have complicated geometries such as in this case. This is because the boundaries between clusters will always be linear.



## Part 1-C: Conclusions

All of our results follow the theory. We know KMeans using Euclidean distance will only be good at cluster if the clusters have a spherical shape. KMeans using Mahalanobis will be useful when the clusters have more of an ellipsoid shape therefore having some clear linear trend or dispersion. However, it fails when the clusters have more complicated geometry (such as not being Gaussian). Using Mahalanobis distance for KMeans is an improvement and will allow you to more accurately cluster some data but there is still no guarantee that for any shape it can properly classify clusters.



## Part 2-A: Davies Bouldin Index

I started by creating a Davies Bouldin index class which accepts the data and an optional parameter of the number of iterations (clusters) to check for the data. For this example I used my set default 10.

I made the assumption there is at least two clusters and no more than 10 and I ran KMeans on the data using my above Euclidean distance implementation and calculated the Davies Bouldin Index each round. I ran KMeans using the cluster values `[2, max_iterations+2)`. If the Davies Bouldin index was lower for a particular cluster number, then I would save the labels, cluster number and index for that lowest value.

For my implementation of Davies Bouldin Index I first calculated the within cluster distance and then the between cluster distances for each cluster stored and took the maximum. The resulting value was our worst case within-to-between cluster ratio. Lastly, I took the mean of all these ratios and this was out Davies Bouldin index.

This is just the equation: $\frac{1}{k}\sum_{i=1}^{k}\underset{j \leq k, j \neq i}{max} \ D_{ij} $



## Part 2-B: Demonstration

I use the given [cluster validation data](cluster_validation_data.txt) and run my model_selection to get the correct amount of clusters for the data. In the [graph](Graphs/davies_bouldin.png) it shows the results of the KMeans with the lowest Davies Bouldin index value and a graph showing for each cluster value the different index value. For this example there were 3 clusters and in the graph you can see the lowest point is for 3 clusters.



## Part 2-C: Conclusions

All of my results follow the theory. We visually can see that using the DB Index we were able to properly calculate the "goodness" of each clustering run with KMeans to allow us to do model selection and choose the partition that was most accurate. This can be good in cases where you have data and are not actually sure how many clusters or features there are or also just in general validate the results of your clustering algorithm for a given partition.